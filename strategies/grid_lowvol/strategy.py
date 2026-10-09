# -*- coding: utf-8 -*-
"""网格 × 低波动 EGARCH 量化策略 (Grid LowVol Strategy)。

流程 (每日, 严格无未来):
  before_market_open (≤T-1 信息):
    ① RV20 全市场初筛 → 候选池 K=200 (低波动粗过滤, 向量化)
    ② 候选池逐只 EGARCH(1,1) → 预测未来 vol_horizon_days 日累计波动 → 排序 → 目标池 top_n
    ③ 持仓预警: 预测波动 > vol_alert_mult × 目标池中位 → 清仓清单 (单边行情保险丝)
    ④ 补仓清单: 目标池中不在持仓的最低波动股 (数量=清仓数)
  handle_bar (T 日开盘):
    ⑤ 换仓: 开盘清仓预警股 → 开盘建底仓 2 格买入补仓股 (等权)
    ⑥ 日内网格: 每只持仓按当日 5 分钟 K 线时序触发 (GridEngine), 逐笔 broker 成交
    ⑦ T+1 锁定: 当日买入份额不可卖 (策略层今日买入量锁定)

无未来: EGARCH/RV 仅用 ≤T-1 收盘; 网格用当日 K 线时序; 换仓用开盘价;
性能: RV <1s + EGARCH 200只 ≈2s (numba) + 网格 <0.5s ≈ 3s/日。
"""
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from gpsef.core import IStrategy, StrategyContext
from .egarch import EGARCHModel
from .grid_engine import GridEngine
from .data_utils import (preload_daily_returns_fast, top_low_rv, load_minute_5m,
                         minute_of_day, has_dividend_data, load_industry_map,
                         smallcap_pool)

logger = logging.getLogger(__name__)


class GridLowVolStrategy(IStrategy):
    STRATEGY_NAME = 'grid_lowvol'

    def initialize(self, context: StrategyContext):
        cfg = context.config
        self.config = cfg
        self._g = cfg.get('grid_lowvol', {})
        self._dp = context.data_provider

        # ---- 选股与预测 ----
        self.top_n = int(self._g.get('top_n', 10))
        self.candidate_pool = int(self._g.get('candidate_pool', 200))
        self.vol_horizon_days = int(self._g.get('vol_horizon_days', 20))
        self.egarch_window = int(self._g.get('egarch_window', 250))
        self.rv_min_days = int(self._g.get('rv_min_days', 20))
        # [P1-fix] 预警: 阈值 1.5→2.0 + 连续 N 日确认 (滤除低波股日常噪声)
        self.vol_alert_mult = float(self._g.get('vol_alert_mult', 2.0))
        self.vol_alert_days = int(self._g.get('vol_alert_days', 3))

        # ---- 网格 ----
        self.grid_levels = int(self._g.get('grid_levels', 5))
        # [P1-fix] 网格宽度 0.5→1.0σ: 单格利润 ≥ 双边成本 5-10 倍
        self.grid_width_sigma = float(self._g.get('grid_width_sigma', 1.0))
        self.grid_band_sigma = float(self._g.get('grid_band_sigma', 2.0))
        self.base_fraction = float(self._g.get('base_fraction', 0.4))  # 底仓 = 40% 单只预算 (2/5 格)
        # [P0-fix] 趋势闸门: 价格 < MA(趋势闸门周期) → 网格只卖不买 (0=关闭)
        self.trend_gate_ma = int(self._g.get('trend_gate_ma', 20))
        # [2026-08-22 opt-P2] 行业分散: 目标池每行业最多 max_per_industry 只 (0=关闭)
        self.max_per_industry = int(self._g.get('max_per_industry', 3))
        # [2026-08-22 opt] 基础池子集: all=过滤池 | smallcap=过滤池∩市值最低N只 (实验)
        self.pool_subset = str(self._g.get('pool_subset', 'all'))
        self.smallcap_size = int(self._g.get('smallcap_size', 300))

        # [2026-08-22 opt-P0/P1] 组合级风控 (状态机: 0=NORMAL 1=TREND_DOWN 2=HALF 3=FLAT)
        # [2026-08-22 recover] 恢复机制修复: FLAT→HALF 改市场信号(指数≥MA60 连续确认)+峰值重置
        #   (原"回撤收窄<dd_recover_half"死锁: 空仓净值恒定, 条件永不可达 → 永久空仓)
        cg = self._g.get('combo_gate', {})
        self._combo_cfg = {
            'enabled': bool(cg.get('enabled', True)),
            'index_ma': int(cg.get('index_ma', 60)),          # 组合趋势闸门 MA 周期 (0=关闭)
            'dd_half': float(cg.get('dd_half', 0.05)),        # 回撤熔断-半仓
            'dd_flat': float(cg.get('dd_flat', 0.08)),        # 回撤熔断-清仓
            'dd_recover_full': float(cg.get('dd_recover_full', 0.015)),  # HALF→NORMAL 恢复(回撤收窄)
            'trend_confirm_days': int(cg.get('trend_confirm_days', 3)),  # 指数≥MA 连续确认天数 (FLAT/HALF/TREND_DOWN 恢复)
        }
        self._combo_state = 0
        self._idx_above_days = 0       # 指数 ≥ MA 连续天数 (恢复确认)
        self._half_done = False        # 本轮 HALF 半仓动作是否已执行
        self._from_flat_recover = False  # [recover] 是否 FLAT→HALF 恢复模式 (HALF 允许半仓建仓)
        self._prev_nav = float(cfg.get('backtest', {}).get('initial_cash', 1e8))
        self._peak_value = self._prev_nav

        # ---- 交易区间与复权 ----
        bt = cfg.get('backtest', {})
        self.start_date = datetime.strptime(bt['start_date'], '%Y%m%d')
        self.end_date = datetime.strptime(bt['end_date'], '%Y%m%d')
        ref_str = cfg.get('data', {}).get('reference_date', '20260522')
        self._ref_date = datetime.strptime(str(ref_str), '%Y%m%d')
        warmup = int(self._g.get('warmup_days', 280))  # ≥ egarch_window + 30
        from datetime import timedelta
        ext_start = self.start_date - timedelta(days=int(warmup * 1.6) + 30)
        # [2026-08-22 recover] 顺序修复: _load_index_close 需 self.end_date/_ref_date/ext_start
        #   已定义后才可调用 (原位于其定义之前, 会 NameError)
        self._index_close = self._load_index_close(ext_start, self.end_date)

        # ---- 交易日历 ----
        self._calendar = context.calendar or []
        self._full_calendar = getattr(context, 'full_calendar', None) or self._calendar

        # ---- 基础股票池 (DynamicStockPool 过滤池, 与 alpha191 系一致) ----
        # [2026-08-21 pool-fix] 引擎已从 config.data.stock_pool 创建过滤池挂到 context.stock_pool
        #   (ST/ST历史/次新/价格≥3/成交额后10%), 选股必须在过滤池内, 防止全市场池选入退市股/ST股
        self.stock_pool = getattr(context, 'stock_pool', None)
        if self.stock_pool is None:
            logger.warning("[grid_lowvol] context.stock_pool 缺失 (config 无 data.stock_pool 段?) — "
                           "回退全市场池 (可能选入退市/ST股, 强烈建议修复 config)")

        # ---- 全市场日收益率轻量预加载 (直读 raw_daily + 复权因子, 避免 provider 大缓存) ----
        # 预加载覆盖全市场仅为数据完整性; 实际选股在过滤池内 (见 before_market_open)
        logger.info("[grid_lowvol] 预加载全市场日收益率 (轻量直读)...")
        self._ret_map = preload_daily_returns_fast(
            self._all_codes(context.data_provider), ext_start, self.end_date)
        logger.info(f"[grid_lowvol] 日收益率预加载完成: {len(self._ret_map)} 只")

        # ---- 状态 ----
        self._target_pool: List[str] = []
        self._plan = {'to_liquidate': [], 'to_add': [], 'targets': [], 'vol': {}}
        self._today_bought: Dict[str, float] = {}   # T+1: 当日买入份额锁定
        # [P0-fix] 跨日网格状态: code -> GridState (基准=建仓价, 配对完成才重置)
        self._grid_states: Dict[str, object] = {}
        self._alert_hist: Dict[str, int] = {}       # [P1-fix] 预警连续超阈值天数
        self._grid_engine = GridEngine(levels=self.grid_levels,
                                       width_sigma=self.grid_width_sigma,
                                       band_sigma=self.grid_band_sigma)
        self._verbose = bool(self._g.get('verbose', True))

    def _all_codes(self, data_provider) -> List[str]:
        """全市场股票列表 (仅用于预加载收益率数据覆盖, 非选股池)。"""
        try:
            return sorted(data_provider.get_stock_list() or [])
        except Exception:
            return []

    def _load_index_close(self, start: datetime, end: datetime) -> Optional[pd.Series]:
        """预加载沪深300 (sh.000300) 复权收盘价序列 (组合趋势闸门用)。"""
        try:
            df = self._dp.get_daily_data('sh.000300', start, end,
                                         reference_date=self._ref_date)
            if df is None or df.empty or 'close' not in df.columns:
                return None
            c = df[['trade_date', 'close']].copy()
            c['trade_date'] = pd.to_datetime(c['trade_date'])
            return c.set_index('trade_date').sort_index()['close']
        except Exception as e:
            logger.warning(f"[grid_lowvol] 沪深300 加载失败: {e}")
            return None

    def _index_ma60_down(self, date: datetime) -> bool:
        """组合趋势闸门: 沪深300 收盘 < MA(index_ma) (≤T-1, 严格无未来)。"""
        ma = self._combo_cfg['index_ma']
        if ma <= 0 or self._index_close is None:
            return False
        px = self._index_close[self._index_close.index < date]
        if len(px) < ma + 3:
            return False
        return float(px.iloc[-1]) < float(px.iloc[-ma:].mean())

    def _update_combo_state(self, date: datetime):
        """组合级风控状态机 (触发保守优先 + 市场信号恢复 + 峰值重置)。

        状态: 0=NORMAL 1=TREND_DOWN(只卖不买) 2=HALF(半仓) 3=FLAT(空仓)。
        触发: 回撤>dd_flat→FLAT; 回撤>dd_half→HALF; 指数<MA→TREND_DOWN。
        恢复 (2026-08-22 recover 修复死锁: 原 FLAT→HALF 靠"回撤收窄<3%",
              但空仓净值恒定, 条件永不可达 → 永久空仓):
          FLAT→HALF: 指数≥MA60 连续 trend_confirm_days 日确认 → 进入 HALF(半仓试探),
                     同时重置 _peak_value=当前净值 (回撤基线归零重新起算)。
          HALF→NORMAL: 指数确认 trend_confirm_days 日 且 回撤<dd_recover_full (双条件)。
          TREND_DOWN→NORMAL: 指数确认 trend_confirm_days 日。
        触发守卫: 已在 FLAT 不重复触发回撤熔断 (否则 dd 恒定-8% 永远锁死 FLAT, 恢复不可达);
                  HALF 中回撤加深≤-8% 仍升级 FLAT。
        """
        cg = self._combo_cfg
        if not cg['enabled'] or cg['index_ma'] <= 0:
            self._combo_state = 0
            return
        # 净值回撤 (前一交易日净值, 盘前可知, 无未来)
        dd = self._prev_nav / self._peak_value - 1.0 if self._peak_value > 0 else 0.0
        idx_down = self._index_ma60_down(date)
        # 指数 ≥ MA 连续天数维护
        self._idx_above_days = 0 if idx_down else self._idx_above_days + 1

        # ---- 触发 (保守优先; 已在目标状态不重复触发, 保证恢复可达) ----
        if dd <= -cg['dd_flat'] + 1e-9 and self._combo_state != 3:
            self._combo_state = 3          # FLAT (HALF 中加深仍升级; FLAT 中不重复锁死)
            self._from_flat_recover = False
            self._idx_above_days = 0       # [opt2] FLAT 触发清零: 恢复确认从空仓后重新累计
        elif dd <= -cg['dd_half'] + 1e-9 and self._combo_state not in (2, 3):
            self._combo_state = 2          # HALF (熔断来源: 满仓减半, 非恢复模式)
            self._from_flat_recover = False
        elif idx_down and self._combo_state == 0:
            self._combo_state = 1          # TREND_DOWN
        else:
            # ---- 恢复 (市场信号确认 + 滞后带 + 分层) ----
            if self._combo_state == 3 and self._idx_above_days >= cg['trend_confirm_days']:
                self._combo_state = 2          # FLAT → HALF (指数确认)
                self._peak_value = self._prev_nav   # [recover] 峰值重置: 回撤基线归零
                self._half_done = False            # 允许半仓动作 (空仓无持仓, 无操作)
                self._from_flat_recover = True     # 恢复模式: HALF 允许半仓建仓
                self._idx_above_days = 0           # 重新累计 HALF→NORMAL 确认
            elif self._combo_state == 2 and self._idx_above_days >= cg['trend_confirm_days'] \
                    and dd > -cg['dd_recover_full'] + 1e-9:
                self._combo_state = 0          # HALF → NORMAL (指数确认 + 回撤收窄)
                self._from_flat_recover = False
            elif self._combo_state == 1 and self._idx_above_days >= cg['trend_confirm_days']:
                self._combo_state = 0          # TREND_DOWN → NORMAL (指数确认)

    def _base_pool(self, date: datetime) -> List[str]:
        """当日基础股票池 = DynamicStockPool 过滤池 (T-1 视角, 严格无未来)。

        [2026-08-21 pool-fix] 与 alpha191 系一致: ST/ST历史/次新/价格/成交额过滤。
        返回当日过滤后的可交易股票列表 (池子为空 → 回退全市场并告警)。
        """
        if self.stock_pool is not None:
            try:
                pool = self.stock_pool.get_candidates(date, None) or []
                if pool:
                    # [2026-08-22 opt] 子集模式: smallcap = 过滤池 ∩ 市值最低 N 只
                    if self.pool_subset == 'smallcap':
                        pool_small = smallcap_pool(set(pool), date, self.smallcap_size)
                        if len(pool_small) >= self.top_n:
                            pool = pool_small
                        else:
                            logger.warning(f"[grid_lowvol] {date.date()} smallcap 子集不足 "
                                           f"({len(pool_small)}<{self.top_n}), 用全过滤池")
                    return pool
                logger.warning(f"[grid_lowvol] {date.date()} 过滤池为空, 回退全市场")
            except Exception as e:
                logger.warning(f"[grid_lowvol] {date.date()} 过滤池获取失败: {e}, 回退全市场")
        return self._all_codes(self._dp)

    # ------------------------------------------------------------------
    # 盘前: 波动率预测 + 选股 + 预警
    # ------------------------------------------------------------------
    def before_market_open(self, context: StrategyContext):
        date = context.current_dt
        # [2026-08-22 opt-P0/P1] 组合级风控状态更新 (回撤熔断 + 指数趋势 + 分层恢复)
        self._update_combo_state(date)
        self._plan = {'to_liquidate': [], 'to_add': [], 'targets': [], 'vol': {}}

        # ① RV20 初筛 (限定过滤池) → 候选池
        # [2026-08-21 pool-fix] only_codes=过滤池: 全市场选低波动会选入退市股/ST股
        base_pool = set(self._base_pool(date))
        if not base_pool:
            logger.warning(f"[grid_lowvol] {date.date()} 基础池为空, 跳过")
            return
        candidates = top_low_rv(self._ret_map, date, k=self.candidate_pool,
                                min_days=self.rv_min_days, only_codes=base_pool)
        if len(candidates) < self.top_n:
            logger.warning(f"[grid_lowvol] {date.date()} 候选池不足({len(candidates)})")
            return

        # ② 候选池 EGARCH → 未来 h 日累计波动 → 目标池 top_n
        vol = {}
        for code in candidates:
            r = self._ret_map.get(code)
            if r is None:
                continue
            idx = r.index[r.index < date]   # [2026-08-21 exdiv-fix1] 严格≤T-1: 排除当日(盘前不可知)收益率
            if len(idx) < 30:
                continue
            m = EGARCHModel(window=self.egarch_window)
            if m.fit(r.loc[idx].values):
                v = m.predict_cumvol(self.vol_horizon_days)
                if np.isfinite(v) and v > 0:
                    vol[code] = v
        if len(vol) < self.top_n:
            logger.warning(f"[grid_lowvol] {date.date()} EGARCH 有效不足({len(vol)})")
            return
        ranked = sorted(vol.items(), key=lambda x: x[1])
        # [2026-08-22 opt-P2] 行业分散: 按 vol 升序选 top_n, 每行业最多 max_per_industry 只
        #   (控制 BARRA size=-0.86/value=+0.80 风格过度集中; 候选不足时放宽补足)
        if self.max_per_industry > 0:
            ind_map = load_industry_map()
            ind_count = {}
            targets = []
            for c, _ in ranked:
                ind = ind_map.get(c.replace('sh.', '').replace('sz.', ''), '未知')
                if ind_count.get(ind, 0) >= self.max_per_industry:
                    continue
                ind_count[ind] = ind_count.get(ind, 0) + 1
                targets.append(c)
                if len(targets) >= self.top_n:
                    break
            if len(targets) < self.top_n:      # 行业约束过紧 → 按原序补足
                for c, _ in ranked:
                    if c not in targets:
                        targets.append(c)
                    if len(targets) >= self.top_n:
                        break
            self._plan['targets'] = targets[:self.top_n]
        else:
            self._plan['targets'] = [c for c, _ in ranked[:self.top_n]]
        self._plan['vol'] = vol

        # [2026-08-21 exdiv-fix3] 目标池除权数据缺口预警: 无 dividend_data 的股票除权日
        # 走 engine bf fallback (含送转事件不精确) — 每日盘前对缺失目标提示一次
        for code in self._plan['targets']:
            if not has_dividend_data(code):
                logger.warning(f"[grid_lowvol] {date.date()} 目标池 {code} 无除权数据 "
                               f"(dividend_data 缺失), 除权日将走 bf fallback(含送转不精确)")

        # ③ 持仓预警: [P1-fix] 预测波动 > vol_alert_mult × 目标池中位 且 连续 vol_alert_days 日
        #    (低波股波动中位数极低, 单日 1.5× 触发是噪声; 连续 N 日确认滤除, 2.0× 提阈值)
        held = list(context.portfolio.positions.keys())
        median_vol = float(np.median([vol.get(c, 0.0) for c in self._plan['targets']]))
        for code in held:
            if code in vol and vol[code] > self.vol_alert_mult * median_vol:
                self._alert_hist[code] = self._alert_hist.get(code, 0) + 1
            else:
                self._alert_hist[code] = 0
        for code in held:
            if self._alert_hist.get(code, 0) >= self.vol_alert_days:
                self._plan['to_liquidate'].append(code)

        # ④ 补仓: 预警清仓数 + 空位 (未达 top_n) 合计, 从目标池取最低波动且不在持仓的
        liquidated = set(self._plan['to_liquidate'])
        held_valid = [c for c in held if c not in liquidated]
        empty_slots = max(0, self.top_n - len(held_valid))
        n_add = min(len(self._plan['to_liquidate']) + empty_slots, self.top_n)
        add = [c for c in self._plan['targets'] if c not in held][:n_add]
        self._plan['to_add'] = add

        if self._verbose:
            logger.info(f"[grid_lowvol] {date.date()} 目标池top{len(self._plan['targets'])}: "
                        f"{self._plan['targets'][:5]}... 预警清仓{len(self._plan['to_liquidate'])}: "
                        f"{self._plan['to_liquidate']} 补仓{len(add)}: {add}")

    # ------------------------------------------------------------------
    # 开盘: 换仓 + 日内网格
    # ------------------------------------------------------------------
    def handle_bar(self, context: StrategyContext):
        broker = context.broker
        portfolio = context.portfolio
        date = context.current_dt
        plan = self._plan
        self._today_bought = {}
        combo = self._combo_state

        # [2026-08-22 opt-P0/P1] FLAT(3): 清仓全部持仓, 不建仓不网格
        if combo == 3:
            for code in list(portfolio.positions.keys()):
                broker.order_target_percent(code, 0.0, trade_reason='combo_flat')
            self._grid_states.clear()
            self._half_done = False
            self._from_flat_recover = False   # [recover] 清仓退出恢复模式
            self._prev_nav = portfolio.total_value
            self._peak_value = max(self._peak_value, portfolio.total_value)
            return

        # [2026-08-22 opt-P0/P1] HALF(2): 首次进入减半持仓; 之后只卖不买
        # [2026-08-22 opt2] 语义修复: 减半 = 每只持仓目标 = 当前市值×50%
        #   (原 order_target_percent(0.5) 目标=总资产×50%, 持仓市值小时会反向加仓到50%总资产;
        #    2024-06-26 实例: 熔断触发反而加仓6424万 → -15.13% 最大回撤推手)
        if combo == 2 and not self._half_done:
            for code in list(portfolio.positions.keys()):
                pos_ = portfolio.positions.get(code)
                if pos_ is None or pos_.quantity <= 0:
                    continue
                broker.order_target_value(code, float(pos_.market_value) * 0.5,
                                          trade_reason='combo_half')
            self._half_done = True

        # ⑤ 换仓: 开盘清仓预警股 → 开盘建底仓补仓股
        # [2026-08-22 recover] 建仓限制: 仅 NORMAL 满仓建仓; HALF 恢复模式(_from_flat_recover)半仓试探;
        #   TREND_DOWN / 熔断来源 HALF 不新开
        allow_build = (combo == 0) or (combo == 2 and self._from_flat_recover)
        for code in plan['to_liquidate']:
            if code in portfolio.positions:
                broker.order_target_percent(code, 0.0, trade_reason='vol_alert_liquidate')
                self._grid_states.pop(code, None)          # [P0-fix] 清仓 → 移除网格状态
                self._alert_hist[code] = 0                  # 预警计数复位
        if plan['to_add'] and allow_build:
            total_value = portfolio.total_value
            per_stock = total_value / self.top_n if self.top_n else 0.0
            # [2026-08-22 recover] HALF 恢复模式: 半仓预算试探建仓 (分层恢复 空→半→满)
            if combo == 2:
                per_stock *= 0.5
            base_value = per_stock * self.base_fraction
            for code in plan['to_add']:
                pos_before = portfolio.positions.get(code)
                qty_before = float(pos_before.quantity) if pos_before is not None else 0.0
                ok = broker.order_value(code, base_value, trade_reason='rebalance_base')
                if ok:
                    pos_after = portfolio.positions.get(code)
                    qty_after = float(pos_after.quantity) if pos_after is not None else 0.0
                    # [2026-08-21 exdiv-fix2] 锁定"本次买入增量"而非持仓总量:
                    #   已有持仓再补仓时, 原持仓当日仍可网格卖出 (T+1 只约束当日新买入份额)
                    bought_inc = max(0.0, qty_after - qty_before)
                    self._today_bought[code] = self._today_bought.get(code, 0.0) + bought_inc
                    # [P0-fix] 新建仓 → 创建跨日网格状态 (基准=建仓价, 由 _run_grid 初始化)

        # ⑥ 日内网格: 每只持仓按 5 分钟线触发 (跨日 GridState; 买入受组合状态限制)
        for code in list(portfolio.positions.keys()):
            pos = portfolio.positions.get(code)
            if pos is None or pos.quantity <= 0:
                continue
            self._run_grid(broker, portfolio, code, date)

        # ⑦ 日终: 今日买入锁定次日释放 + 净值峰值/前值更新 (风控状态机用)
        self._today_bought = {}
        self._prev_nav = portfolio.total_value
        self._peak_value = max(self._peak_value, portfolio.total_value)
        # HALF→NORMAL 恢复时重置半仓标记 (重新进入 HALF 会再次半仓)
        if combo == 0:
            self._half_done = False

    # ------------------------------------------------------------------
    def _trend_down(self, code: str, date: datetime) -> bool:
        """趋势闸门: 归一化价格 < MA(趋势闸门周期) → 趋势向下 (网格只卖不买)。

        用 _ret_map 收益率序列累乘得归一化价格 (≤T-1, 严格无未来)。
        trend_gate_ma<=0 关闭闸门。
        """
        ma = self.trend_gate_ma
        if ma <= 0:
            return False
        r = self._ret_map.get(code)
        if r is None:
            return False
        idx = r.index[r.index < date]
        px = (1.0 + r.loc[idx]).cumprod()
        if len(px) < ma + 3:
            return False
        return float(px.iloc[-1]) < float(px.iloc[-ma:].mean())

    # ------------------------------------------------------------------
    def _run_grid(self, broker, portfolio, code: str, date: datetime):
        """单只股票当日网格执行 (跨日 GridState + 趋势闸门)。"""
        mdf = minute_of_day(load_minute_5m(code), date)
        if mdf is None or mdf.empty:
            return
        base_price = float(mdf['open'].iloc[0])   # raw 口径 (与成交/持仓成本一致)
        if not np.isfinite(base_price) or base_price <= 0:
            return
        daily_sigma = float(self._plan['vol'].get(code, 0.02))  # 收益率 σ (兜底 2%)
        if not np.isfinite(daily_sigma) or daily_sigma <= 0:
            daily_sigma = 0.02
        # [2026-08-21 bug-fix] 网格步长下限 1%: EGARCH 对极低波/停牌股(窗口内收益率≈0)
        #   的 predict_cumvol 可塌缩至 ~1e-6 → 网格步长→0 → 网格线全挤在基准价附近疯狂触发。
        #   低波股日 σ 通常 1-2%, 1% 下限保证单格利润 ≥ 0.5% (width=1.0) 覆盖双边成本。
        daily_sigma = max(daily_sigma, 0.01)

        # [P0-fix] 跨日网格状态: 首次建仓创建; 配对平衡(completed)且趋势允许 → 重置基准重新锚定
        state = self._grid_states.get(code)
        if state is None:
            state = self._grid_engine.make_state(base_price, daily_sigma)
            self._grid_states[code] = state
        else:
            if state.completed and not self._trend_down(code, date):
                state.reset(base_price)   # 重新锚定当日开盘价 (仅配对完成后)

        # [P0-fix] 趋势闸门: 趋势向下 → 只卖不买; [2026-08-22 combo] 组合状态非 NORMAL 也禁买
        trend_down = self._trend_down(code, date)
        allow_buy = not trend_down and self._combo_state == 0
        # [2026-08-22 opt-P0] 底仓保护: 趋势向下或组合风控时锁底仓 (卖出上限=已买入格数),
        #   只做配对卖出回本, 底仓保留 → 仓位不再归零 (修复平均仓位 32.6% 的现金闲置)
        state.base_locked = trend_down or self._combo_state != 0

        pos = portfolio.positions.get(code)
        held_qty = float(pos.quantity) if pos is not None else 0.0
        today_bought = self._today_bought.get(code, 0.0)
        sellable_qty = max(0.0, held_qty - today_bought)
        # 卖出上限金额 (留整手余量, 防 T+1 越界)
        max_sell_value = sellable_qty * base_price * 0.99 if sellable_qty > 0 else 0.0

        total_value = portfolio.total_value
        per_stock = total_value / self.top_n if self.top_n else 0.0
        grid_value = per_stock / self.grid_levels if self.grid_levels else per_stock

        stats = self._grid_engine.execute_day(
            mdf, broker, code, state, grid_value,
            today_bought_qty=today_bought, max_sell_value=max_sell_value,
            allow_buy=allow_buy)

        if self._verbose and (stats['buys'] > 0 or stats['sells'] > 0):
            logger.info(f"[grid_lowvol] {date.date()} {code} 网格: "
                        f"卖{stats['sells']}格/买{stats['buys']}格 "
                        f"(base={state.base_price:.2f} step={state.step:.3f} "
                        f"sellable={'Y' if allow_buy else 'N'})")
