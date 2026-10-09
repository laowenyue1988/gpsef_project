# -*- coding: utf-8 -*-

"""
GPSEF 事件驱动回测引擎

功能:
    - 事件驱动回测循环
    - 支持多种策略类型
    - 自动生成绩效报告
    - 支持基准对比分析

核心概念:
    - 事件驱动: 每日调用策略的before_market_open和handle_bar
    - 前复权: 所有价格数据统一复权到 canonical reference_date（数据最大日期）
    - 股票池: 动态过滤ST、次新股等
    - 模拟经纪商: 模拟真实交易环境（佣金、滑点、涨跌停等）

复权基准日设计:
    Canonical reference: 始终使用数据中可用最大日期（如2026-12-31）作为基准日。
    adjusted_price = raw × bf(t) / bf(canonical_ref)
    虽然 bf(ref) 在个股收益率中抵消（ret独立于ref），但 quantity = target_value / price
    依赖于绝对价格水平。不同 ref → 不同价格 → 不同股数 → 不同持仓 → 净值累积偏离。
    统一使用 canonical reference 确保不同 end_date 的回测结果完全可比较。

使用方法:
    from gpsef.backtest import BacktestEngine

    engine = BacktestEngine(config)
    report = engine.run(strategy)
"""

import os
import json
import copy
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging

from gpsef.core import IStrategy, StrategyContext, Portfolio
from gpsef.data import LocalDataProvider
from gpsef.pool import DynamicStockPool
from gpsef.execution import SimulatedBroker

# 抑制matplotlib的INFO日志
logging.getLogger('matplotlib').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ============================================================================
# [REF-AUTO-2026-09-10] 复权基准日解析（引擎与策略共用**单一实现**）
# ============================================================================

REF_DATE_AUTO = 'auto'


def resolve_reference_date(data_cfg, end_date_str):
    """解析复权基准日。

    规则（定稿 2026-09-10）:
      · 'auto' / '' / None / **字段缺失** → **回测结束日**（引擎设计的默认意图：
        "复权基准日：从配置文件读取，默认为回测结束日"）
      · 显式 'YYYYMMDD'                   → 使用该固定日期（历史实验复现通道）

    为什么要统一解析（背景）:
      原实现被各配置里硬编码的 '20220908'（数据下载批次的遗留值）覆盖，
      使"默认为回测结束日"的设计意图从未生效 —— 同一策略在不同回测区间下
      使用的复权基准日与区间脱钩，且与"以最新日为基准"的行业惯例不符。
      本函数同时供 `BacktestEngine.trade_reference_date` 与
      `Strategy.reference_date` 使用 → 引擎/经纪商/策略/池 四方基准日严格一致。

    数学性质（为何该口径变更是安全的，实证见 tests/test_ref_date_auto.py）:
      前复权 adj(t) = raw(t) × bf(t)/bf(T)。更换基准日 T→T' 等于对整条序列
      乘以**常数** bf(T')/bf(T) ⇒ 收益率/比率/同日换算(raw↔adj)全部不变；
      而本项目所有"绝对量"过滤（price_min→raw_close、amount_pct→raw_amount、
      市值→raw amount）均使用 **raw 数据** ⇒ 结果不随基准日变化。
    """
    ref = (data_cfg or {}).get('reference_date', None)
    if ref is None:
        ref = end_date_str
    else:
        ref = str(ref).strip()
        if (not ref) or ref.lower() == REF_DATE_AUTO:
            ref = end_date_str
    ref = str(ref).strip()
    if not ref:
        raise ValueError(
            '无法解析复权基准日: data.reference_date 为 auto/缺省 时，'
            'backtest.end_date 必须提供（当前为空）')
    try:
        return datetime.strptime(ref, '%Y%m%d')
    except ValueError as e:
        raise ValueError(
            "复权基准日格式无效: %r（应为 'YYYYMMDD' 或 'auto'）" % (ref,)) from e


def reference_date_source(data_cfg) -> str:
    """基准日来源标注（用于日志/报告可追溯）。"""
    ref = (data_cfg or {}).get('reference_date', None)
    if ref is None or str(ref).strip() == '' or str(ref).strip().lower() == REF_DATE_AUTO:
        return 'auto=回测结束日'
    return 'config(显式 %s)' % str(ref).strip()


def _to_scalar(val):
    """
    将Series或数组转换为标量值。

    输入:
        val: pandas.Series, numpy.ndarray 或其他类型
    输出:
        标量值或np.nan
    作用:
        确保后续运算使用标量而非数组
    调用方:
        BacktestEngine._update_portfolio, _calculate_benchmark_returns
    """
    if hasattr(val, 'iloc'):
        if len(val) > 0:
            val = val.iloc[0]
        else:
            return np.nan
    if hasattr(val, 'item'):
        val = val.item()
    return float(val)


class BacktestEngine:
    """
    事件驱动回测引擎。

    职责:
        - 管理回测生命周期
        - 驱动策略执行
        - 记录每日净值和持仓
        - 生成绩效报告

    组成:
        - 数据提供者: 提供历史价格数据
        - 股票池: 过滤候选股票
        - 经纪商: 模拟订单执行
        - 策略: 实现交易逻辑

    复权基准日:
        使用数据中可用最大日期作为 canonical reference。
        adjusted_price = raw × bf(t) / bf(canonical_ref)
        虽然个股收益率中 bf(ref) 抵消，但 quantity = target_value / price
        依赖于绝对价格水平。统一使用 canonical reference 确保不同 end_date
        的回测结果完全可比较。

    调用方:
        - run_backtest.py: 入口脚本
        - run_optimize.py: 优化器内部
    """

    def __init__(self, config: Dict[str, Any], output_dir: Optional[str] = None):
        """
        初始化回测引擎。

        输入:
            config: 策略配置字典，包含backtest、data、execution等配置
            output_dir: 输出目录路径，若为None则自动生成以策略名_回测区间_时间戳命名的目录
        输出:
            无
        作用:
            初始化数据提供者、股票池、经纪商、策略上下文等
        调用方:
            - run_backtest.py main函数
            - GenericOptimizationProblem.evaluate
        """
        self.config = config
        backtest_cfg = config['backtest']

        # 解析回测区间
        self.start_date = datetime.strptime(backtest_cfg['start_date'], '%Y%m%d')
        self.end_date = datetime.strptime(backtest_cfg['end_date'], '%Y%m%d')
        self.initial_cash = backtest_cfg['initial_cash']

        strategy_name = config['strategy']['name']

        # 自动生成输出目录: 策略名_开始日期_结束日期_时间戳
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            output_dir = f"./results/{strategy_name}_{backtest_cfg['start_date']}_{backtest_cfg['end_date']}_{timestamp}"

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"回测输出目录: {self.output_dir}")

        # 初始化数据提供者
        data_cfg = config['data']

        # 复权基准日：[REF-AUTO-2026-09-10] 'auto'/缺省 → **回测结束日**（引擎设计的默认意图；
        #   原被各配置硬编码 '20220908' 覆盖）；显式 'YYYYMMDD' → 固定基准日（历史复现）。
        #   与策略端共用 resolve_reference_date → 引擎/经纪商/策略/池四方严格一致。
        #   所有买卖和估值都使用此固定基准日，收益率自然包含分红信息
        self.trade_reference_date = resolve_reference_date(data_cfg, backtest_cfg['end_date'])
        logger.info(f"复权基准日: {self.trade_reference_date.date()} "
                    f"[来源: {reference_date_source(data_cfg)}]")

        self.data_provider = LocalDataProvider(data_cfg['base_path'])

        # 获取完整日历（扩展500天，确保策略能获取充足历史数据）
        extended_start = self.start_date - timedelta(days=500)
        full_calendar = self.data_provider.get_trade_calendar(extended_start, self.end_date)
        self.calendar = [d for d in full_calendar if d >= self.start_date]
        self.full_calendar = full_calendar

        logger.info(f"回测区间: {self.start_date.date()} 至 {self.end_date.date()}")
        logger.info(f"交易日数量: {len(self.calendar)}")

        # 初始化股票池
        pool_cfg = data_cfg.get('stock_pool', {})
        self.stock_pool = DynamicStockPool(pool_cfg)

        # 初始化股票池数据
        if hasattr(self.stock_pool, 'initialize'):
            logger.info("正在初始化股票池...")
            min_data_date = self.full_calendar[0] - timedelta(days=100)
            # [FIX-REF-DATE] 传入 end_date，确保池的复权基准日与策略一致
            self.stock_pool.initialize(self.data_provider, min_date=min_data_date,
                                       end_date=self.end_date)

        # 初始化投资组合
        self.portfolio = Portfolio(self.initial_cash)

        # 初始化经纪商（传递固定的trade_reference_date）
        exec_cfg = config.get('execution', {})
        self.broker = SimulatedBroker(
            self.portfolio, self.data_provider, exec_cfg, self.trade_reference_date
        )

        # 加载基准数据
        self.benchmark_data = None
        benchmark_code = backtest_cfg.get('benchmark_code')
        if benchmark_code:
            self.benchmark_data = self._load_benchmark_data(benchmark_code)

        # 构造策略上下文
        self.context = StrategyContext()
        self.context.config = config
        self.context.data_provider = self.data_provider
        self.context.stock_pool = self.stock_pool
        self.context.broker = self.broker
        self.context.portfolio = self.portfolio
        self.context.calendar = self.calendar
        self.context.full_calendar = self.full_calendar
        self.context.output_dir = self.output_dir  # SHAP等分析产物输出路径

        # 回测记录
        self.daily_records = []
        self.daily_positions = []

        # ---------- 回测日志写入本地文件 ----------
        self._log_file_handler = logging.FileHandler(
            os.path.join(self.output_dir, 'backtest.log'),
            encoding='utf-8'
        )
        self._log_file_handler.setLevel(logging.DEBUG)
        self._log_file_handler.setFormatter(
            logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
        )
        for logger_name in ['strategies', 'gpsef']:
            _log = logging.getLogger(logger_name)
            _log.addHandler(self._log_file_handler)
        logging.getLogger(__name__).addHandler(self._log_file_handler)

    def _update_portfolio(self, date: datetime, save_prev: bool = True):
        """
        更新持仓市值。

        输入:
            date: 当前交易日期
            save_prev: 是否在更新市值前保存 prev_market_value
                      盘前估值(T-1)传 False，盘后估值(T)传 True (BUG-4 修复)
        输出:
            无
        作用:
            获取持仓股票的最新收盘价，计算最新市值。
            使用固定的trade_reference_date作为复权基准日。
        调用方:
            BacktestEngine.run

        注意:
            reference_date=trade_reference_date（固定值），
            确保估值价格与买卖价格在同一个前复权体系下。
        """
        prices = {}
        for code in list(self.portfolio.positions.keys()):
            df = self.data_provider.get_daily_data(
                code, date, date, reference_date=self.trade_reference_date
            )
            if not df.empty:
                row = df.iloc[0]
                # BUG-2 修复: 只使用 raw_close，不回退到 adj_close
                # 回退到 adj_close 会导致 market_value 与 cost_basis(raw_open体系) 价格体系混用
                if 'raw_close' in row.index:
                    raw_val = row.get('raw_close', np.nan)
                    if not pd.isna(raw_val):
                        price = _to_scalar(raw_val)
                    else:
                        logger.warning(
                            f"[Valuation] {code} raw_close NaN on {date.date()}, "
                            f"skipping to avoid price system mismatch"
                        )
                        continue
                else:
                    logger.error(
                        f"[Valuation] {code} raw_close column missing! Check M0. Skipping."
                    )
                    continue
                prices[code] = price
        self.portfolio.update_value(prices, save_prev=save_prev)

    def _process_ex_dividend(self, date: datetime, date_idx: int = 0):
        """
        盘前除权除息处理 [v13]。

        核心逻辑:
            对每只持仓股票，查询当日是否有除权事件：
            1. 无事件 → 跳过
            2. 有事件 → 按事件类型分别处理:
               a) 送股/转增 → 调整股数和均价（市值不变）
               b) 现金分红 → 补入现金（税后）
               c) 送股税   → 扣减现金（面值1元×新增股数×税率）

        税率规则（财税[2015]101号）:
            持股 ≤30天: 20%
            持股 31~365天: 10%
            持股 >365天: 0%

        注意:
            - 转增（dividReserveToStockPs）免税（国税发[1997]198号）
            - 送股（dividStocksPs）按面值1元/股计征红利税
            - 配股不在此处理（baostock无数据，v12 bf平滑已覆盖）
            - 现金分红税在除权日预扣（行业惯例，实盘在卖出时扣）
        """
        records = []  # M8: 返回除权记录列表
        for code in list(self.portfolio.positions.keys()):
            pos = self.portfolio.positions[code]

            # Step 1: 查询除权事件
            event = self.data_provider.get_dividend_event(code, date)

            # 无分红数据时回退到旧 bf 方法
            if event is None:
                self._process_ex_dividend_bf_fallback(code, date, pos)
                continue

            cash_ps = event['cash_ps']
            bonus_ps = event['bonus_ps']
            reserve_ps = event['reserve_ps']
            stock_ratio = event['stock_ratio']
            description = event['description']

            # Step 2: 计算持股天数 → 确定税率
            holding_days = (date - pos.buy_date).days if pos.buy_date else 30
            if holding_days <= 30:
                tax_rate = 0.20
            elif holding_days <= 365:
                tax_rate = 0.10
            else:
                tax_rate = 0.0

            # Step 2.5 [v13.2]: 记录分红前个股净值 + 获取 T-1/T 价格
            # 先获取 T-1 价格（用于计算分红前市值 = Q × P_raw(T-1)）
            prev_date = None
            raw_close_before = np.nan
            adj_close_before = np.nan
            raw_close_after = np.nan
            adj_close_after = np.nan
            # [BUG-5修复] 直接从数据源获取T日价格，不依赖pos.current_price
            df_today = self.data_provider.get_daily_data(
                code, date, date, reference_date=self.trade_reference_date
            )
            if not df_today.empty:
                row_today = df_today.iloc[0]
                raw_close_after = _to_scalar(row_today.get('raw_close', np.nan))
                adj_close_after = _to_scalar(row_today.get('close', np.nan))
            # T-1 日价格
            if date_idx > 0:
                prev_date = self.calendar[date_idx - 1]
                df_prev = self.data_provider.get_daily_data(
                    code, prev_date, prev_date, reference_date=self.trade_reference_date
                )
                if not df_prev.empty:
                    prev_row = df_prev.iloc[0]
                    raw_close_before = _to_scalar(prev_row.get('raw_close', np.nan))
                    adj_close_before = _to_scalar(prev_row.get('close', np.nan))
            # Step 2.6 [v13.2]: 分红前个股净值 = Q × P_raw(T-1)
            old_qty = float(pos.quantity)
            if not np.isnan(raw_close_before) and raw_close_before > 0:
                stock_value_before = old_qty * raw_close_before
            else:
                stock_value_before = float(pos.market_value)  # 回退用 T 日估值

            # Step 3: 处理送股/转增（股数变化）
            old_avg_cost = float(pos.avg_cost)

            if stock_ratio > 0:
                new_qty = old_qty * (1 + stock_ratio)
                new_avg_cost = old_avg_cost / (1 + stock_ratio) if old_avg_cost > 0 else 0
                self.portfolio.adjust_position(code, new_qty, new_avg_cost)
            else:
                new_qty = old_qty
                new_avg_cost = old_avg_cost

            # Step 4: 处理现金分红
            net_cash_dividend = 0.0
            gross_cash_dividend = 0.0
            if cash_ps > 0:
                gross_cash_dividend = cash_ps * old_qty
                net_cash_dividend = gross_cash_dividend * (1 - tax_rate)
                self.portfolio.cash += net_cash_dividend

            # Step 5: 处理送股税金（面值1元 × 新增送股数 × 税率）
            bonus_tax = 0.0
            if bonus_ps > 0:
                bonus_tax = 1.0 * bonus_ps * old_qty * tax_rate
                self.portfolio.cash -= bonus_tax

            # Step 6: 重算 total_value
            self.portfolio.total_value = (
                self.portfolio.cash
                + sum(float(p.market_value) for p in self.portfolio.positions.values())
            )

            # Step 7: 策略回调
            cash_tax = gross_cash_dividend * tax_rate  # [v13.2] 现金分红税
            if hasattr(self, '_strategy') and hasattr(self._strategy, 'on_ex_dividend'):
                self._strategy.on_ex_dividend(
                    code, date, stock_ratio, cash_ps, bonus_ps,
                    net_dividend=net_cash_dividend,
                    bonus_tax_amount=bonus_tax,
                    cash_tax=cash_tax,
                )

            # Step 8: 事件类型判定（用于日志）
            type_parts = []
            if cash_ps > 0:
                type_parts.append('现金分红')
            if bonus_ps > 0:
                type_parts.append('送股')
            if reserve_ps > 0:
                type_parts.append('转增')
            event_type = '+'.join(type_parts) if type_parts else '未知'

            # Step 9 [v13.2]: 完整日志 — 个股净值 + 市场涨跌/税收分离
            tax_detail = []
            if cash_ps > 0:
                tax_detail.append(f'现金税={gross_cash_dividend*tax_rate:,.2f}({tax_rate*100:.0f}%)')
            if bonus_ps > 0:
                tax_detail.append(
                    f'送股税={bonus_tax:,.2f}(面值1元×{bonus_ps:.4f}×{old_qty:.0f}×{tax_rate*100:.0f}%)'
                )

            stock_line = f"{old_qty:.0f} → {new_qty:.0f}"
            if stock_ratio > 0:
                stock_line += f" (+{(new_qty-old_qty):.0f}股"
                if bonus_ps > 0:
                    stock_line += f", 送股10送{bonus_ps*10:.1f}"
                if reserve_ps > 0:
                    stock_line += f", 转增10转{reserve_ps*10:.1f}"
                stock_line += ")"
            else:
                stock_line += " (不变)"

            # ──── 个股净值计算 ────
            # 分红后个股净值 = 新股数 × T日raw_close + 税后分红现金 - 送股税
            stock_value_after = new_qty * raw_close_after if not np.isnan(raw_close_after) else 0
            # net_cash_dividend 已加入 portfolio.cash，bonus_tax 已扣除
            # 此处计算"该持仓贡献的"净值，需单独加回
            stock_value_after += net_cash_dividend - bonus_tax
            nv_ratio = stock_value_after / stock_value_before if stock_value_before > 0 else np.nan
            nv_change = stock_value_after - stock_value_before

            # ──── 市场涨跌 / 税收效应分离 ────
            # 除权参考价
            p_ref = (raw_close_before - cash_ps) / (1 + stock_ratio) \
                if not np.isnan(raw_close_before) and raw_close_before > 0 and (1 + stock_ratio) > 0 \
                else np.nan
            market_effect = np.nan  # 市场涨跌导致的净值变化
            total_tax_effect = gross_cash_dividend * tax_rate + bonus_tax  # 税收导致的净值损失
            reinvest_diff = np.nan  # 再投资差异（二阶）
            if not np.isnan(p_ref) and p_ref > 0 and not np.isnan(raw_close_after):
                r_m = raw_close_after / p_ref - 1  # 市场涨跌（相对于除权参考价）
                market_effect = old_qty * raw_close_before * r_m       # 一阶
                reinvest_diff = -old_qty * cash_ps * r_m              # 二阶（通常 < 0.2%）

            # ──── 前复权价变化（= 纯市场涨跌） ────
            adj_change_pct = np.nan
            if not np.isnan(adj_close_before) and adj_close_before > 0 and not np.isnan(adj_close_after):
                adj_change_pct = (adj_close_after / adj_close_before - 1) * 100

            # ──── 原始价变化 ────
            raw_change_pct = np.nan
            if not np.isnan(raw_close_before) and raw_close_before > 0 and not np.isnan(raw_close_after):
                raw_change_pct = (raw_close_after / raw_close_before - 1) * 100

            # ──── 日志组装 ────
            log_lines = [
                f"\n  ========== [{date.date()}] {code} 除权事件 ==========",
                f"  类型: {event_type}",
                f"  公告: {description}",
                f"  ─────── 价格 ───────",
            ]
            if not np.isnan(raw_close_before) and not np.isnan(raw_close_after):
                log_lines.append(
                    f"  原始收盘价: {raw_close_before:.2f} → {raw_close_after:.2f}"
                    f"  (变化={raw_change_pct:+.2f}%)"
                )
            elif not np.isnan(raw_close_after):
                log_lines.append(f"  原始收盘价: N/A → {raw_close_after:.2f}")
            if not np.isnan(adj_close_before) and not np.isnan(adj_close_after):
                log_lines.append(
                    f"  前复权收盘价: {adj_close_before:.2f} → {adj_close_after:.2f}"
                    f"  (变化={adj_change_pct:+.2f}% ≈ 纯市场涨跌)"
                )
            elif not np.isnan(adj_close_after):
                log_lines.append(f"  前复权收盘价: N/A → {adj_close_after:.2f}")
            if not np.isnan(p_ref):
                log_lines.append(f"  除权参考价: {p_ref:.2f}")
            log_lines.extend([
                f"  ─────── 持仓 ───────",
                f"  股数: {stock_line}",
                f"  成本价: {old_avg_cost:.2f} → {new_avg_cost:.2f}",
            ])
            if cash_ps > 0:
                log_lines.append(
                    f"  现金分红: {old_qty:.0f}股 × {cash_ps:.4f}元 = {gross_cash_dividend:,.2f}(税前) / "
                    f"{net_cash_dividend:,.2f}(税后)"
                )
            if bonus_ps > 0:
                log_lines.append(f"  送股税金: {bonus_tax:,.2f}")
            if tax_detail:
                log_lines.append(f"  税率: {tax_rate*100:.0f}% (持股{holding_days}天) | {' | '.join(tax_detail)}")
            log_lines.extend([
                f"  ─────── 个股净值 ───────",
            ])
            if not np.isnan(raw_close_before):
                log_lines.append(
                    f"  分红前: {stock_value_before:,.2f}  ({old_qty:.0f}股 × {raw_close_before:.2f})"
                )
            else:
                log_lines.append(f"  分红前: {stock_value_before:,.2f}")
            if not np.isnan(raw_close_after):
                log_lines.append(
                    f"  分红后: {stock_value_after:,.2f}  ({new_qty:.0f}股 × {raw_close_after:.2f}"
                    f" + 分红净收入{net_cash_dividend-bonus_tax:,.2f})"
                )
            else:
                log_lines.append(f"  分红后: {stock_value_after:,.2f}")
            if not np.isnan(nv_ratio):
                log_lines.append(f"  净值比: {nv_ratio:.6f}  (变化={nv_change:+,.2f})")
            else:
                log_lines.append("  净值比: N/A")
            # ──── 净值变化分解 ────
            if not np.isnan(market_effect):
                log_lines.extend([
                    f"  ─────── 净值变化分解 ───────",
                    f"  市场涨跌效应: {market_effect:+,.2f}  (T-1市值 × 市场涨跌{r_m*100:+.2f}%)",
                    f"  税收损失: -{total_tax_effect:,.2f}  (现金税{gross_cash_dividend*tax_rate:,.2f} + 送股税{bonus_tax:,.2f})",
                    f"  再投资差异: {reinvest_diff:+,.2f}  (二阶，通常<0.2%)",
                    f"  验证: 市场涨跌{market_effect:+,.2f} - 税收{total_tax_effect:,.2f}"
                    f" + 再投资差异{reinvest_diff:+,.2f} = {market_effect-total_tax_effect+reinvest_diff:+,.2f}"
                    f" ≈ 总变化{nv_change:+,.2f}",
                ])
            log_lines.append("  =================================================")
            # M8: 记录除权信息（完整版，供 M10 dividends.csv + positions.csv div_* 列）
            net_div = net_cash_dividend - bonus_tax
            records.append({
                'code': code, 'date': date,
                'event_type': event_type, 'description': description,
                'cash_ps': cash_ps, 'bonus_ps': bonus_ps, 'reserve_ps': reserve_ps,
                'stock_ratio': stock_ratio,
                'old_qty': old_qty, 'new_qty': new_qty,
                'old_avg_cost': old_avg_cost, 'new_avg_cost': new_avg_cost,
                'gross_cash_dividend': gross_cash_dividend,
                'net_cash_dividend': net_cash_dividend,
                'bonus_tax': bonus_tax, 'cash_tax': cash_tax,
                'tax_rate': tax_rate, 'holding_days': holding_days,
                'net_dividend': net_div,
                'net_dividend_for_position': net_div,
                'raw_close_before': raw_close_before, 'raw_close_after': raw_close_after,
                'adj_close_before': adj_close_before, 'adj_close_after': adj_close_after,
                'p_ref': p_ref,
                'stock_value_before': stock_value_before, 'stock_value_after': stock_value_after,
                'nv_ratio': nv_ratio, 'nv_change': nv_change,
                'market_effect': market_effect, 'total_tax_effect': total_tax_effect,
                'reinvest_diff': reinvest_diff,
                'raw_change_pct': raw_change_pct, 'adj_change_pct': adj_change_pct,
            })

            logger.info("\n".join(log_lines))

        return records  # M8

    def _process_ex_dividend_bf_fallback(self, code: str, date: datetime, pos):
        """
        旧版 bf 回退方法：当 dividend_data 不存在时使用。

        此方法对混合事件（含送股/转增）不正确——会把送转部分也当现金补偿。
        仅作为数据缺失时的降级方案，会打印 WARNING。
        """
        df_back = self.data_provider._get_back_factors(code)
        if df_back is None or df_back.empty:
            return

        ts_date = pd.Timestamp(date.date())
        df_back_sorted = df_back.sort_values('dividOperateDate')

        mask_on_or_before = df_back_sorted['dividOperateDate'] <= ts_date
        if not mask_on_or_before.any():
            return
        bf_after = float(
            df_back_sorted.loc[mask_on_or_before[mask_on_or_before].index[-1], 'backAdjustFactor']
        )

        mask_before = df_back_sorted['dividOperateDate'] < ts_date
        if not mask_before.any():
            return
        bf_before = float(
            df_back_sorted.loc[mask_before[mask_before].index[-1], 'backAdjustFactor']
        )

        if bf_after <= bf_before * 1.001:
            return

        logger.warning(
            f"[v13-fallback] {code} 在 {date.date()} 无分红数据，"
            f"使用旧 bf 方法（对含送转事件不正确！请运行 download_data.py --mode dividend）"
        )

        lookback_start = date - timedelta(days=15)
        df = self.data_provider.get_daily_data(
            code, lookback_start, date, reference_date=self.trade_reference_date
        )
        if df is None or df.empty:
            return

        dates_norm = pd.to_datetime(df['trade_date']).dt.normalize()
        prev_mask = dates_norm < ts_date
        if not prev_mask.any():
            return
        prev_row = df.loc[prev_mask[prev_mask].index[-1]]

        if 'raw_close' in prev_row.index:
            prev_close = _to_scalar(prev_row['raw_close'])
            if pd.isna(prev_close):
                prev_close = _to_scalar(prev_row['close'])
        else:
            prev_close = _to_scalar(prev_row['close'])

        if np.isnan(prev_close) or prev_close <= 0:
            return

        dividend_per_share = prev_close * (1.0 - bf_before / bf_after)
        total_dividend = dividend_per_share * float(pos.quantity)

        holding_days = (date - pos.buy_date).days if pos.buy_date else 30
        if holding_days <= 30:
            tax_rate = 0.20
        elif holding_days <= 365:
            tax_rate = 0.10
        else:
            tax_rate = 0.0
        net_dividend = total_dividend * (1.0 - tax_rate)

        self.portfolio.cash += net_dividend
        self.portfolio.total_value = (
            self.portfolio.cash
            + sum(float(p.market_value) for p in self.portfolio.positions.values())
        )

    def _load_benchmark_data(self, benchmark_code: str) -> Optional[pd.DataFrame]:
        """
        加载基准指数数据（本地优先，缺失时下载并增量存储）。

        输入:
            benchmark_code: 基准指数代码，如'sh.000300'
        输出:
            基准指数日线DataFrame或None
        策略:
            1. 优先从 data/raw_daily/<code>.parquet 加载本地数据
            2. 本地缺失时从 baostock 下载
            3. 下载成功后存储到本地 parquet 供后续复用
        """

        # ── 1. 确定本地文件路径 ──
        # [2026-08-31 修正] 优先 data/index_daily/{code}.parquet (无下划线, 与
        #   strategy._load_breaker_index_close / risk.market_state 统一); 
        #   回退旧格式 data/raw_daily/{code}.parquet (下划线) 保证兼容。
        code_file = benchmark_code.replace('.', '')   # sh.000300 → sh000300
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        local_path = os.path.join(project_root, 'data', 'index_daily', f'{code_file}.parquet')
        legacy_path = os.path.join(project_root, 'data', 'raw_daily',
                                   f"{benchmark_code.replace('.', '_')}.parquet")
        if not os.path.exists(local_path) and os.path.exists(legacy_path):
            local_path = legacy_path

        # ── 2. 优先从本地加载 ──
        if os.path.exists(local_path):
            try:
                df = pd.read_parquet(local_path)
                if 'date' in df.columns and 'trade_date' not in df.columns:
                    df = df.rename(columns={'date': 'trade_date'})
                if 'trade_date' in df.columns:
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                    df = df[(df['trade_date'] >= pd.Timestamp(self.start_date))
                            & (df['trade_date'] <= pd.Timestamp(self.end_date))]
                logger.info("基准数据从本地加载: %s (%d条)", local_path, len(df))
                return df if not df.empty else None
            except Exception as e:
                logger.warning("本地基准数据读取失败: %s，回退联网下载", e)

        # ── 3. 联网下载（本地缺失或读取失败时） ──
        try:
            import baostock as bs
            from strategies.alpha191.data_utils import get_baostock_session

            session = get_baostock_session()
            session.ensure_logged_in()

            start_fmt = self.start_date.strftime('%Y-%m-%d')
            end_fmt = self.end_date.strftime('%Y-%m-%d')

            rs = bs.query_history_k_data_plus(
                benchmark_code,
                "date,code,open,high,low,close,preclose,volume,amount,pctChg",
                start_date=start_fmt,
                end_date=end_fmt,
                frequency="d"
            )

            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())

            if data_list:
                df = pd.DataFrame(data_list, columns=rs.fields)
                df['date'] = pd.to_datetime(df['date'])

                numeric_cols = ['open', 'high', 'low', 'close', 'preclose',
                                'volume', 'amount', 'pctChg']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                df = df.rename(columns={'date': 'trade_date'})
                logger.info("基准数据下载成功: %d条", len(df))

                # ── 4. 存储到本地 parquet ──
                try:
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    df.to_parquet(local_path, index=False)
                    logger.info("基准数据已存储到: %s", local_path)
                except Exception as e:
                    logger.warning("基准数据本地存储失败: %s", e)

                return df
            else:
                logger.warning("基准数据下载返回空列表")
                return None

        except Exception as e:
            logger.warning("获取基准数据失败: %s", e)
            return None

    def _calculate_benchmark_returns(self) -> Optional[pd.DataFrame]:
        """
        计算基准累计收益率。

        输入:
            无
        输出:
            包含trade_date和benchmark_cumulative列的DataFrame
        作用:
            计算基准指数从起始到当前的累计收益率
        调用方:
            BacktestEngine._generate_report
        """
        if self.benchmark_data is None:
            return None

        benchmark = self.benchmark_data[
            (self.benchmark_data['trade_date'] >= self.start_date) &
            (self.benchmark_data['trade_date'] <= self.end_date)
            ].copy()

        if benchmark.empty:
            return None

        benchmark = benchmark.sort_values('trade_date').reset_index(drop=True)
        benchmark['benchmark_return'] = benchmark['close'].pct_change()
        benchmark['benchmark_cumulative'] = (benchmark['close'] / benchmark['close'].iloc[0] - 1) * 100

        return benchmark[['trade_date', 'benchmark_cumulative']].rename(columns={'trade_date': 'date'})

    def run(self, strategy: IStrategy) -> Dict[str, Any]:
        """
        执行回测。

        输入:
            strategy: 策略实例（必须实现IStrategy接口）
        输出:
            绩效指标字典
        作用:
            - 初始化策略
            - 逐日执行事件驱动循环
            - 记录每日数据
            - 生成绩效报告
        调用方:
            - run_backtest.py
            - GenericOptimizationProblem.evaluate
        """
        logger.info(f"回测开始: {self.start_date.date()} -> {self.end_date.date()}")
        logger.info(f"初始资金: {self.initial_cash:,.2f}")
        logger.info(f"复权基准日: {self.trade_reference_date.date()}")

        # 初始化策略
        strategy.initialize(self.context)
        self._strategy = strategy

        # 事件驱动循环
        for i, date in enumerate(self.calendar):
            self.context.current_dt = date
            self.broker.set_current_date(date)

            # [BUG-1修复] 盘前用T-1日收盘价估值（消除前瞻偏差）
            # 首日(i=0)无需估值，total_value保持initial_cash
            if i > 0:
                self._update_portfolio(self.calendar[i - 1], save_prev=False)

            # [M4.2-v12] 除权除息现金补偿（估值后、交易前）
            dividend_records = self._process_ex_dividend(date, date_idx=i)

            # M9: 更新各持仓的 accumulated_dividend_net
            for rec in dividend_records:
                if rec['code'] in self.portfolio.positions:
                    pos = self.portfolio.positions[rec['code']]
                    net_div = rec.get('net_dividend', 0.0)
                    if hasattr(pos, 'accumulated_dividend_net'):
                        pos.accumulated_dividend_net += net_div

            # 调用策略钩子
            strategy.before_market_open(self.context)
            strategy.handle_bar(self.context)

            # 盘后更新持仓估值（T日raw_close）
            self._update_portfolio(date)

            # M10: 计算当日分红净收入（汇总所有持仓）
            dividend_income = sum(rec.get('net_dividend', 0.0) for rec in dividend_records)
            # M10: 汇总所有持仓的累计分红净收入
            accumulated_dividend_total = sum(
                getattr(pos, 'accumulated_dividend_net', 0.0)
                for pos in self.portfolio.positions.values()
            )

            # [BUG-1修复] 统一在盘后记录daily，所有字段为盘后值
            daily = {
                'date': date,
                'cash': self.portfolio.cash,
                'position_value': self.portfolio.total_value - self.portfolio.cash,
                'total_value': self.portfolio.total_value,
                'total_value_eod': self.portfolio.total_value,
                'daily_return': 0.0,
                # ──── M10 新增字段 ────
                'dividend_income': dividend_income,
                'accumulated_dividend_net': accumulated_dividend_total,
            }
            if i > 0:
                prev = self.daily_records[-1].get('total_value_eod',
                                                   self.daily_records[-1]['total_value'])
                daily['daily_return'] = (self.portfolio.total_value / prev) - 1
            self.daily_records.append(daily)

            # 记录持仓快照（盘后）
            # M9: 构建 dividend_records 字典，供 M10 positions.csv / dividends.csv 使用
            div_dict = {}
            for rec in dividend_records:
                div_dict[rec['code']] = rec

            positions_snapshot = {
                'date': date,
                # BUG-1 修复: 使用 deepcopy 避免所有历史快照被最终值覆盖
                'positions': {code: copy.deepcopy(pos)
                              for code, pos in self.portfolio.positions.items()},
                'dividend_records': div_dict,  # M9
            }
            self.daily_positions.append(positions_snapshot)

            # 进度日志
            if (i + 1) % 20 == 0 or i == 0:
                logger.info(f"回测进度: {i + 1}/{len(self.calendar)} ({100 * (i + 1) / len(self.calendar):.1f}%)")

        logger.info("回测完成，正在生成报告...")

        # 生成绩效报告
        return self._generate_report()

    def _generate_report(self) -> Dict[str, Any]:
        """
        生成绩效报告。

        输入:
            无
        输出:
            绩效指标字典
        作用:
            - 计算各类绩效指标
            - 保存CSV和JSON报告
            - 生成可视化图表
        调用方:
            BacktestEngine.run
        """
        df = pd.DataFrame(self.daily_records)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # 合并基准收益
        if self.benchmark_data is not None:
            benchmark_df = self._calculate_benchmark_returns()
            if benchmark_df is not None:
                benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
                benchmark_df.set_index('date', inplace=True)
                df = df.merge(benchmark_df[['benchmark_cumulative']], left_index=True, right_index=True, how='left')

        if 'benchmark_cumulative' not in df.columns:
            df['benchmark_cumulative'] = np.nan

        # ===== 计算绩效指标 =====

        # 基础收益指标
        total_return = (df['total_value_eod'].iloc[-1] / self.initial_cash) - 1
        daily_returns = df['daily_return'].dropna()
        n_days = len(df)
        years = n_days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        annual_vol = daily_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # 最大回撤
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        max_dd_date = drawdown.idxmin() if not drawdown.empty else None

        # 胜率与盈亏比
        win_rate = (daily_returns > 0).mean()
        avg_gain = daily_returns[daily_returns > 0].mean()
        avg_loss = abs(daily_returns[daily_returns < 0].mean())
        profit_loss_ratio = avg_gain / avg_loss if avg_loss > 0 else 0

        # 组装指标
        metrics = {
            'strategy': self.config['strategy']['name'],
            'backtest_period': f"{self.start_date.date()} to {self.end_date.date()}",
            'initial_cash': self.initial_cash,
            'final_value': df['total_value_eod'].iloc[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'max_drawdown_date': str(max_dd_date.date()) if max_dd_date else None,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'total_commission': self.broker.total_commission,
            'total_tax': self.broker.total_tax,
            'total_orders': len(self.broker.orders),
        }

        # ===== 保存报告文件 =====

        # 每日净值
        df.to_csv(os.path.join(self.output_dir, 'daily_values.csv'))

        # 订单记录
        if self.broker.orders:
            pd.DataFrame(self.broker.orders).to_csv(
                os.path.join(self.output_dir, 'orders.csv'), index=False
            )

        # 持仓记录
        if self.daily_positions:
            pos_records = []
            for rec in self.daily_positions:
                date = rec['date']
                div_records = rec.get('dividend_records', {})
                for code, pos in rec['positions'].items():
                    # ──── 持仓天数 ────
                    holding_days = 0
                    if hasattr(pos, 'buy_date') and pos.buy_date is not None:
                        holding_days = (date - pos.buy_date).days

                    # ──── 日盈亏（含分红） ────
                    # M10 + R4: 首日(prev_market_value=0)的daily_pnl标记为NaN
                    prev_mv = float(getattr(pos, 'prev_market_value', 0.0))
                    if prev_mv > 0:
                        # 价格变动
                        price_pnl = pos.market_value - prev_mv
                        # 当日分红净收入（仅除权日有值）
                        div_data = div_records.get(code, {})
                        day_div_net = div_data.get('net_dividend_for_position', 0.0) or 0.0
                        # 总日盈亏 = 价格变动 + 分红收入
                        daily_pnl_amount = price_pnl + day_div_net
                        daily_pnl_ratio = daily_pnl_amount / prev_mv
                    else:
                        # 首日或prev_market_value为0（新建仓当天）
                        daily_pnl_amount = np.nan
                        daily_pnl_ratio = np.nan

                    # ──── 累计盈亏（含分红） ────
                    cost_basis = pos.avg_cost * pos.quantity
                    price_pnl_total = pos.market_value - cost_basis
                    accumulated_div = float(getattr(pos, 'accumulated_dividend_net', 0.0))
                    total_pnl_amount = price_pnl_total + accumulated_div
                    total_pnl_ratio = (
                        total_pnl_amount / cost_basis
                        if cost_basis > 0 else 0.0
                    )

                    record = {
                        # ──── 原有列 ────
                        'date': date,
                        'code': code,
                        'quantity': pos.quantity,
                        'avg_cost': pos.avg_cost,
                        'market_value': pos.market_value,
                        # ──── M10 新增列 ────
                        'prev_market_value': prev_mv,
                        'accumulated_dividend_net': accumulated_div,
                        'holding_days': holding_days,
                        'daily_pnl_amount': daily_pnl_amount,
                        'daily_pnl_ratio': daily_pnl_ratio,
                        'total_pnl_amount': total_pnl_amount,
                        'total_pnl_ratio': total_pnl_ratio,
                    }

                    # M10: 分红列（仅除权日有值，其余为空）
                    if div_data:
                        record.update({
                            'div_event_type': div_data.get('event_type', ''),
                            'div_description': div_data.get('description', ''),
                            'div_raw_close_before': div_data.get('raw_close_before'),
                            'div_raw_close_after': div_data.get('raw_close_after'),
                            'div_raw_change_pct': div_data.get('raw_change_pct'),
                            'div_adj_close_before': div_data.get('adj_close_before'),
                            'div_adj_close_after': div_data.get('adj_close_after'),
                            'div_adj_change_pct': div_data.get('adj_change_pct'),
                            'div_p_ref': div_data.get('p_ref'),
                            'div_old_qty': div_data.get('old_qty'),
                            'div_new_qty': div_data.get('new_qty'),
                            'div_old_avg_cost': div_data.get('old_avg_cost'),
                            'div_new_avg_cost': div_data.get('new_avg_cost'),
                            'div_cash_ps': div_data.get('cash_ps'),
                            'div_bonus_ps': div_data.get('bonus_ps'),
                            'div_reserve_ps': div_data.get('reserve_ps'),
                            'div_stock_ratio': div_data.get('stock_ratio'),
                            'div_gross_cash': div_data.get('gross_cash_dividend'),
                            'div_net_cash': div_data.get('net_cash_dividend'),
                            'div_bonus_tax': div_data.get('bonus_tax'),
                            'div_cash_tax': div_data.get('cash_tax'),
                            'div_tax_rate': div_data.get('tax_rate'),
                            'div_holding_days': div_data.get('holding_days'),
                            'div_nv_ratio': div_data.get('nv_ratio'),
                            'div_nv_change': div_data.get('nv_change'),
                            'div_market_effect': div_data.get('market_effect'),
                            'div_tax_effect': div_data.get('total_tax_effect'),
                            'div_reinvest_diff': div_data.get('reinvest_diff'),
                            'div_stock_value_before': div_data.get('stock_value_before'),
                            'div_stock_value_after': div_data.get('stock_value_after'),
                        })

                    pos_records.append(record)

            if pos_records:
                pd.DataFrame(pos_records).to_csv(
                    os.path.join(self.output_dir, 'positions.csv'), index=False
                )

            # M10: 生成独立分红事件文件
            div_records_list = []
            for rec in self.daily_positions:
                date = rec['date']
                div_records = rec.get('dividend_records', {})
                for code, div_data in div_records.items():
                    row = {'date': date, 'code': code}
                    row.update(div_data)
                    div_records_list.append(row)

            if div_records_list:
                div_df = pd.DataFrame(div_records_list)
                # 调整列顺序
                div_col_order = [
                    'date', 'code', 'event_type', 'description',
                    'raw_close_before', 'raw_close_after', 'raw_change_pct',
                    'adj_close_before', 'adj_close_after', 'adj_change_pct',
                    'p_ref', 'old_qty', 'new_qty', 'old_avg_cost', 'new_avg_cost',
                    'cash_ps', 'bonus_ps', 'reserve_ps', 'stock_ratio',
                    'gross_cash_dividend', 'net_cash_dividend', 'bonus_tax', 'cash_tax',
                    'tax_rate', 'holding_days',
                    'stock_value_before', 'stock_value_after', 'nv_ratio', 'nv_change',
                    'market_effect', 'total_tax_effect', 'reinvest_diff',
                ]
                div_df = div_df[[c for c in div_col_order if c in div_df.columns]]
                div_df.to_csv(
                    os.path.join(self.output_dir, 'dividends.csv'), index=False
                )

        # JSON报告
        with open(os.path.join(self.output_dir, 'report.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        # 文本报告
        with open(os.path.join(self.output_dir, 'report.txt'), 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"策略回测报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"策略名称: {metrics['strategy']}\n")
            f.write(f"回测区间: {metrics['backtest_period']}\n")
            f.write(f"复权基准日: {self.trade_reference_date.date()}\n")
            f.write(f"初始资金: {metrics['initial_cash']:,.0f}\n")
            f.write(f"最终资金: {metrics['final_value']:,.0f}\n")
            f.write(f"总收益率: {metrics['total_return']:.2%}\n")
            f.write(f"年化收益率: {metrics['annual_return']:.2%}\n")
            f.write(f"年化波动率: {metrics['annual_volatility']:.2%}\n")
            f.write(f"夏普比率: {metrics['sharpe_ratio']:.2f}\n")
            f.write(f"最大回撤: {metrics['max_drawdown']:.2%} (日期: {metrics['max_drawdown_date']})\n")
            f.write(f"日胜率: {metrics['win_rate']:.2%}\n")
            f.write(f"盈亏比: {metrics['profit_loss_ratio']:.2f}\n")
            f.write(f"总佣金: {metrics['total_commission']:.2f}\n")
            f.write(f"总印花税: {metrics['total_tax']:.2f}\n")
            f.write(f"总订单数: {metrics['total_orders']}\n")
            f.write("=" * 60 + "\n")

        # 绘制图表
        self._plot_performance(df, metrics, drawdown)

        logger.info(f"绩效报告已保存: {self.output_dir}")

        return metrics

    def _plot_performance(self, df: pd.DataFrame, metrics: Dict, drawdown: pd.Series):
        """
        绘制绩效图表。

        输入:
            df: 每日净值DataFrame
            metrics: 绩效指标字典
            drawdown: 回撤序列
        输出:
            无
        作用:
            生成净值曲线和回撤曲线图表
        调用方:
            BacktestEngine._generate_report
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            # 净值曲线
            ax1 = axes[0]
            nav = df['total_value_eod'] / self.initial_cash
            ax1.plot(nav.index, nav, label='Strategy NAV', linewidth=2, color='blue')

            if 'benchmark_cumulative' in df.columns:
                benchmark_nav = 1 + df['benchmark_cumulative'] / 100
                ax1.plot(benchmark_nav.index, benchmark_nav, label='Benchmark',
                         linewidth=2, color='red', linestyle='--')

            ax1.set_title('Strategy vs Benchmark')
            ax1.set_ylabel('Net Asset Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

            # 回撤曲线
            ax2 = axes[1]
            valid_idx = drawdown.index
            ax2.fill_between(valid_idx, drawdown * 100, 0, color='red', alpha=0.3)
            ax2.plot(valid_idx, drawdown * 100, color='red', linewidth=1)
            ax2.set_title(f'Drawdown (Max: {metrics["max_drawdown"] * 100:.2f}%)')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'performance.png'), dpi=150)
            plt.close()

        except Exception as e:
            logger.warning(f"绘图失败: {e}")

        # ===== 保存完整配置文件（便于回测结果与参数对应） =====
        try:
            import yaml as _yaml
            # 1. 保存 YAML 原文（保留注释和可读性）
            source_file = self.config.get('_source_file')
            if source_file and os.path.exists(source_file):
                shutil.copy2(source_file, os.path.join(self.output_dir, 'config.yaml'))
                logger.info(f"配置文件(YAML原文)已保存: {self.output_dir}/config.yaml")
            else:
                with open(os.path.join(self.output_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
                    _yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                logger.info(f"配置文件(YAML重建)已保存: {self.output_dir}/config.yaml")
            # 2. 保存 JSON 全量参数（便于程序化读取）
            config_copy = {k: v for k, v in self.config.items() if k != '_source_file'}
            with open(os.path.join(self.output_dir, 'config_full.json'), 'w', encoding='utf-8') as f:
                json.dump(config_copy, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"配置文件(JSON全量)已保存: {self.output_dir}/config_full.json")
        except Exception as e:
            logger.warning(f"保存配置文件失败: {e}")
