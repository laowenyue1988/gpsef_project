# -*- coding: utf-8 -*-
"""日内网格引擎: 按 5 分钟 K 线时序触发网格买卖 (通过 broker 逐笔成交)。

[2026-08-21 P0+P1 重构] 经典跨日网格 (修复"每日重置基准→阴跌市只买不卖"缺陷):

- GridState: 跨日网格状态。基准价 = 首次建仓价 (固定); 格距 step = width_sigma×σ×base;
  价格"穿越"网格线触发买卖: 上穿卖线(基准+k·step) → 卖 1 格底仓; 下穿买线(基准-k·step) → 买 1 格。
  累计卖出 ≤ n_up (底仓上限), 累计买入 ≤ n_dn (满格停机)。bought==sold (配对平衡) 时 completed,
  策略可重置基准重新锚定 — 阴跌市买入>卖出 → 永不重置 → 反弹回基准才卖 (经典网格盈利机制)。
- execute_day: 逐 K 线 scan → 先卖后买 → broker 逐格成交; allow_buy=False (趋势闸门) 时只卖不买;
  每格金额 grid_value = (组合总资产/top_n)/levels。
- 成交: broker.order_value(exec_price_override=网格线, apply_slippage=False, trade_time=K线时间)。
- T+1: 卖出上限 = 持仓量 − 当日买入量 (策略层锁定, 见 max_sell_value)。
"""
import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class GridState:
    """跨日网格状态 (经典配对网格, 每穿越一条线交易一次)。

    - base_price: 基准价 (首次建仓价, 配对完成可重置)
    - step: 格距 (绝对价格)
    - n_up: 上方卖线数 (底仓可卖格数)
    - n_dn: 下方买线数 (满格停机上限)
    - prev_price: 上一 K 线收盘价 (隔夜跳空穿越检测)
    - sold/bought: 累计卖出/买入格数 (用于上限与配对判断)
    """

    def __init__(self, base_price: float, step: float, n_up: int, n_dn: int):
        self.base_price = float(base_price)
        self.step = max(float(step), 1e-9)
        self.n_up = int(n_up)
        self.n_dn = int(n_dn)
        self.prev_price = float(base_price)
        self.sold = 0
        self.bought = 0
        # [2026-08-22 opt-P0] 底仓保护: 趋势闸门(只卖不买)期间锁定底仓,
        #   卖出上限 = 已买入格数 (配对卖出回本即可), 底仓保留 → 仓位不再归零
        self.base_locked = False

    @property
    def completed(self) -> bool:
        """配对平衡 (卖=买) → 网格可重置 (策略在趋势允许时重锚定)。"""
        return self.bought == self.sold

    def reset(self, new_base: Optional[float] = None):
        """重置网格: 新基准 = 指定价或当前价, 格数清零。"""
        self.base_price = float(new_base) if new_base is not None else self.prev_price
        self.prev_price = self.base_price
        self.sold = 0
        self.bought = 0

    def scan(self, high: float, low: float, close: float) -> Tuple[int, int]:
        """扫描一根 K 线: 返回 (n_sell, n_buy) 穿越格数 (受累计格数上限约束)。

        穿越检测: 上穿卖线 (prev < line ≤ high) → 卖; 下穿买线 (prev > line ≥ low) → 买。
        prev 更新为 close 后, 需价格离开线另一侧才再次触发 (经典网格"穿越一次交易一次")。
        [2026-08-21 bug-fix] 上限用"实时占用"判断 (sold+n_sell/bought+n_buy), 防止
        单根 K 线同时穿越多条线时超买/超卖 (原快照判断: bought=1 时穿 2 线 → 买 3 格 > n_dn)。
        [2026-08-22 opt-P0] base_locked=True 时卖出上限 = bought (只卖配对格, 底仓锁死)。
        """
        n_sell = n_buy = 0
        sell_limit = self.bought if self.base_locked else self.n_up
        for k in range(1, self.n_up + 1):
            if self.sold + n_sell >= sell_limit:
                break
            line = self.base_price + k * self.step
            if self.prev_price < line <= high:
                n_sell += 1
        for k in range(1, self.n_dn + 1):
            if self.bought + n_buy >= self.n_dn:
                break
            line = self.base_price - k * self.step
            if self.prev_price > line >= low:
                n_buy += 1
        self.prev_price = float(close)
        return n_sell, n_buy

    def commit(self, n_sell: int, n_buy: int):
        self.sold += int(n_sell)
        self.bought += int(n_buy)


class GridEngine:
    def __init__(self, levels: int = 5, width_sigma: float = 1.0, band_sigma: float = 2.0):
        self.levels = int(levels)
        self.width_sigma = float(width_sigma)
        self.band_sigma = float(band_sigma)
        n_up = max(1, self.levels // 2)        # 上方卖格数 (2)
        n_dn = max(1, self.levels // 2)        # 下方买格数 (2)
        self.n_up = n_up
        self.n_dn = n_dn

    def build_grid(self, base_price: float, daily_sigma: float) -> Tuple[List[float], List[float]]:
        """返回 (sell_lines, buy_lines): 绝对价格, 升序卖出线 / 降序买入线 (诊断用)。

        σ 为收益率量纲 → 价格步长 = width_sigma × σ × base_price。
        """
        step = self.width_sigma * max(daily_sigma, 1e-6) * base_price
        sell = [base_price + step * k for k in range(1, self.n_up + 1)]
        buy = [base_price - step * k for k in range(1, self.n_dn + 1)]
        return sell, buy

    def make_state(self, base_price: float, daily_sigma: float) -> GridState:
        """创建跨日网格状态 (基准 = 建仓价, 格距 = width_sigma×σ×base)。"""
        step = self.width_sigma * max(daily_sigma, 1e-6) * base_price
        return GridState(base_price, step, self.n_up, self.n_dn)

    def execute_day(self, minute_df, broker, code: str, state: GridState,
                    grid_value: float, today_bought_qty: float = 0.0,
                    max_sell_value: Optional[float] = None,
                    allow_buy: bool = True) -> dict:
        """执行某股票当日网格 (按 5 分钟 K 线时序扫描跨日状态)。

        Args:
            minute_df: 当日 5 分钟 K 线 (按时间排序, 含 high/low/close);
            broker: 框架 broker;
            code: 股票代码;
            state: 跨日 GridState (含基准/格距/累计格数);
            grid_value: 每格目标金额;
            today_bought_qty: 当日已买入数量 (T+1 锁定, 卖出时从可卖中扣除);
            max_sell_value: 卖出金额上限 (None=不限制; 通常 = 可卖数量×价格);
            allow_buy: False = 趋势闸门关闭买入 (只卖不买)。

        Returns: {'sells': n, 'buys': n, 'sold_value': float, 'bought_value': float}
        """
        if minute_df is None or minute_df.empty:
            return {'sells': 0, 'buys': 0, 'sold_value': 0.0, 'bought_value': 0.0}
        n_sell_total = n_buy_total = 0
        sold_value = bought_value = 0.0

        for _, row in minute_df.iterrows():
            hi = float(row['high'])
            lo = float(row['low'])
            cl = float(row['close'])
            ts = str(row.get('trade_time', ''))
            n_sell, n_buy = state.scan(hi, lo, cl)
            # ---- 先卖 (上涨穿越卖线): 底仓逐格卖出 ----
            for _ in range(n_sell):
                k = state.sold + 1                     # 下一条未卖格线
                line = state.base_price + k * state.step
                sell_value = min(grid_value, max_sell_value) if max_sell_value else grid_value
                if sell_value <= 0:
                    break
                ok = broker.order_value(code, -sell_value,
                                        trade_time=ts,
                                        exec_price_override=line,
                                        apply_slippage=False,
                                        trade_reason='grid_sell')
                if ok:
                    state.commit(1, 0)
                    n_sell_total += 1
                    sold_value += sell_value
                else:
                    break
            # ---- 后买 (下跌穿越买线): 现金逐格买入 ----
            if allow_buy:
                for _ in range(n_buy):
                    if state.bought >= state.n_dn:      # [bug-fix] 满格停机防御
                        break
                    k = state.bought + 1               # 下一条未买格线
                    line = state.base_price - k * state.step
                    ok = broker.order_value(code, grid_value,
                                            trade_time=ts,
                                            exec_price_override=line,
                                            apply_slippage=False,
                                            trade_reason='grid_buy')
                    if ok:
                        state.commit(0, 1)
                        n_buy_total += 1
                        bought_value += grid_value
                    else:
                        break
        return {'sells': n_sell_total, 'buys': n_buy_total,
                'sold_value': sold_value, 'bought_value': bought_value}
