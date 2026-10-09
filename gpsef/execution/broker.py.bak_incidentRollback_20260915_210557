# -*- coding: utf-8 -*-

import logging

"""
GPSEF 模拟经纪商模块

模拟真实证券交易的执行环境，包括：
- 交易费用（佣金、印花税）
- 滑点
- 涨跌停限制
- 最小交易单位

复权基准日设计：
    所有价格查询统一使用 trade_reference_date（从配置文件读取，默认为回测结束日）。
    这样买卖价格和持仓估值都在同一个前复权体系下，bf[ref]在收益率计算中完全抵消，
    得到的收益率等于后复权收益率（含分红），无需额外处理分红现金入账。

    数学证明：
      买入价: adj_buy = raw[t1] x bf[t1] / bf[ref]
      卖出价: adj_sell = raw[t2] x bf[t2] / bf[ref]
      收益率: adj_sell / adj_buy - 1
            = (raw[t2] x bf[t2] / bf[ref]) / (raw[t1] x bf[t1] / bf[ref]) - 1
            = (raw[t2] x bf[t2]) / (raw[t1] x bf[t1]) - 1
            = 后复权收益率（含分红）

使用方法:
    from gpsef.execution import SimulatedBroker

    broker = SimulatedBroker(portfolio, data_provider, config, trade_reference_date)
    broker.order_target_percent('sh.600000', 0.2)  # 买入至20%仓位
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime, timedelta

from gpsef.core.portfolio import Portfolio
from gpsef.data.base import BaseDataProvider


def _to_scalar(val):
    """
    将Series或数组转换为标量值。

    输入:
        val: pandas.Series, numpy.ndarray 或其他类型
    输出:
        float标量值或np.nan
    作用:
        确保价格和数量是标量而非数组
    调用方:
        SimulatedBroker各方法
    """
    if hasattr(val, 'iloc'):
        if len(val) > 0:
            val = val.iloc[0]
        else:
            return np.nan
    if hasattr(val, 'item'):
        val = val.item()
    return float(val)


# ═══════ [M3.1-v12] A股涨跌停规则表 — 按板块精确区分 ═══════
_LIMIT_RULES = {
    'sh.600': (0.10, 0.05),   # 主板 ±10%, ST ±5%
    'sh.601': (0.10, 0.05),
    'sh.603': (0.10, 0.05),
    'sh.605': (0.10, 0.05),
    'sz.000': (0.10, 0.05),
    'sz.001': (0.10, 0.05),   # 深主板注册制
    'sz.002': (0.10, 0.05),
    'sz.300': (0.20, 0.20),   # 创业板 20%（含ST）
    'sz.301': (0.20, 0.20),   # 创业板注册制
    'sh.688': (0.20, 0.20),   # 科创板 20%（含ST）
}
_LIMIT_EPS = 1e-9  # [v10修正] 仅防浮点ULP误差


def _find_limit_rule(code: str):
    """按前缀匹配涨跌停规则，未匹配默认主板10%"""
    for prefix, rule in _LIMIT_RULES.items():
        if code.startswith(prefix):
            return rule
    return (0.10, 0.05)


class SimulatedBroker:
    """
    模拟经纪商。

    职责:
        - 执行买入/卖出订单
        - 计算交易费用
        - 处理涨跌停限制
        - 记录订单历史

    交易规则:
        - 佣金：默认万分之2.5（双向）
        - 印花税：千分之1（仅卖出）
        - 最低佣金：5元
        - 滑点：默认千分之1
        - 涨跌停：限制买入涨停股、卖出跌停股
        - 分数股：回测使用分数股（非100整数倍），消除离散取整导致的基准日敏感
          这是业界最佳实践——100股整数倍是市场微观结构，不应影响alpha评估

    复权基准日:
        所有价格查询使用 trade_reference_date（固定值，来自配置文件）。
        这确保买卖和估值在同一个前复权体系下，收益率自然包含分红。

    属性:
        portfolio: 投资组合实例
        data_provider: 数据提供者
        trade_reference_date: 前复权基准日（固定值）
        current_date: 当前交易日期
        orders: 订单历史列表

    使用注意:
        - 订单以开盘价执行
        - 买入数量为100的整数倍
        - 资金不足时自动调整买入数量
    """

    def __init__(
            self,
            portfolio: Portfolio,
            data_provider: BaseDataProvider,
            config: Dict[str, Any],
            trade_reference_date: datetime
    ):
        """
        初始化模拟经纪商。

        输入:
            portfolio: 投资组合实例
            data_provider: 数据提供者实例
            config: 执行配置字典
            trade_reference_date: 前复权基准日（固定值，来自config.yaml的data.reference_date）
        输出:
            无
        作用:
            创建经纪商实例，设置交易参数和复权基准日
        调用方:
            BacktestEngine.__init__
        """
        self.portfolio = portfolio
        self.data_provider = data_provider
        self.trade_reference_date = trade_reference_date

        # 交易费用参数（A股实盘费率）
        self.commission_rate = config.get('commission_rate', 0.0003)       # 佣金 0.3‰ 双向
        self.tax_rate = config.get('tax_rate', 0.0005)                     # 印花税 0.5‰ 仅卖出
        self.min_commission = config.get('min_commission', 5.0)            # 最低佣金 5元
        self.slippage = config.get('slippage', 0.001)                      # 滑点 0.1%
        self.transfer_fee_rate = config.get('transfer_fee_rate', 0.00001)  # 过户费 0.01‰ 双向
        # limit_threshold 已删除 → M3.1 _LIMIT_RULES 板块自动查表

        # 运行时状态
        self.current_date = None
        self.orders: List[Dict[str, Any]] = []
        self.total_commission = 0.0
        self.total_tax = 0.0
        self._raw_price_cache: Dict[tuple, dict] = {}  # (code, yyyymmdd) → {open/close}

    def set_current_date(self, date: datetime):
        """
        设置当前交易日期。

        输入:
            date: 当前交易日
        输出:
            无
        作用:
            更新当前交易日期，用于获取当日数据
        调用方:
            BacktestEngine.run
        """
        self.current_date = date

    def _get_raw_price(self, code: str, price_type: str = 'open') -> float:
        """
        从缓存获取原始（未复权）价格。

        [v12] raw_open/raw_close 仅在 M0 生效后存在。不可用时返回 NaN，
        绝不回退到 adj_open / adj_close（混合价格体系比不交易更危险）。

        输入:
            code: 股票代码
            price_type: 'open' 或 'close'
        输出:
            原始价格 float，不可用时返回 np.nan
        """
        date_key = (code, self.current_date.strftime('%Y%m%d'))
        if date_key in self._raw_price_cache:
            return self._raw_price_cache[date_key].get(price_type, np.nan)

        df = self.data_provider.get_daily_data(
            code, self.current_date, self.current_date,
            reference_date=self.trade_reference_date
        )
        if df is None or df.empty:
            return np.nan

        row = df.iloc[0]
        cache_entry = {}

        col_name = 'raw_open' if price_type == 'open' else 'raw_close'
        if col_name in row.index:
            val = row.get(col_name, np.nan)
            if not pd.isna(val):
                cache_entry[price_type] = _to_scalar(val)

        # 同时缓存另一个价格类型
        other_type = 'close' if price_type == 'open' else 'open'
        other_col = 'raw_close' if price_type == 'open' else 'raw_open'
        if other_col in row.index:
            val = row.get(other_col, np.nan)
            if not pd.isna(val):
                cache_entry[other_type] = _to_scalar(val)

        if not cache_entry:
            # [v12] 不回退到adj值 — 混合价格体系比不交易更危险
            logger.warning(
                f"[RawPrice] {code} raw_open/raw_close not available at {self.current_date.date()}. "
                f"Ensure M0 (local_provider.py raw_open save) is applied. Trade skipped."
            )

        self._raw_price_cache[date_key] = cache_entry
        return cache_entry.get(price_type, np.nan)

    def _get_exec_price(self, code: str, is_buy: bool) -> float:
        """
        获取原始开盘价执行价格（含滑点）。

        使用 raw_open（原始未复权价）而非 adj_open。
        """
        raw_open = self._get_raw_price(code, 'open')
        if np.isnan(raw_open) or raw_open <= 0:
            return np.nan
        if is_buy:
            return raw_open * (1 + self.slippage)
        else:
            return raw_open * (1 - self.slippage)

    def _check_limit(self, code: str, is_buy: bool) -> bool:
        """
        方案C涨跌停检查：统一前复权体系。

        算法: change = (adj_open_T - adj_close_{T-1}) / adj_close_{T-1}
        数学等价于 (raw_open_T - preclose) / preclose
        但两个操作数同在复权列表中 → 代码自明且安全。

        [v12] bf平滑后配股日adj价格连续，方案C不受影响。

        容差: _LIMIT_EPS=1e-9 仅防浮点ULP误差
        """
        lookback_start = self.current_date - timedelta(days=15)
        df = self.data_provider.get_daily_data(
            code, lookback_start, self.current_date,
            reference_date=self.trade_reference_date
        )
        if df is None or df.empty:
            return True

        ts_date = pd.Timestamp(self.current_date.date())
        dates_norm = pd.to_datetime(df['trade_date']).dt.normalize()

        # 当日行（T）
        today_mask = dates_norm == ts_date
        if not today_mask.any():
            return True
        today_row = df.loc[today_mask[today_mask].index[-1]]
        adj_open_today = _to_scalar(today_row['open'])
        if np.isnan(adj_open_today):
            return True

        # [FIX-A2-20260915] 停牌守卫: tradestatus != 1 表示当日停牌, 订单不可成交。
        #   原实现只查涨跌停 ⇒ 配合分钟线零值 bar 会在停牌日产生"成交"
        #   (实测 Q2c 胜出包 5 笔 / P0_Bare 2 笔)。缺失或 NaN 时维持放行(兼容)。
        try:
            _ts = _to_scalar(today_row.get('tradestatus', None))
            if _ts is not None and not (isinstance(_ts, float) and np.isnan(_ts)):
                if abs(float(_ts) - 1.0) > 1e-9:
                    return False
        except Exception:
            pass

        # 前日行（T-1）
        prev_mask = dates_norm < ts_date
        if not prev_mask.any():
            return True
        prev_row = df.loc[prev_mask[prev_mask].index[-1]]
        adj_close_prev = _to_scalar(prev_row['close'])
        if np.isnan(adj_close_prev) or adj_close_prev <= 0:
            return True

        change = (adj_open_today - adj_close_prev) / adj_close_prev

        # 板块限幅
        is_st = int(_to_scalar(today_row.get('isST', 0))) == 1
        limit = _find_limit_rule(code)[1] if is_st else _find_limit_rule(code)[0]

        if is_buy:
            return change < limit - _LIMIT_EPS    # 涨停不可买
        else:
            return change > -limit + _LIMIT_EPS   # 跌停不可卖

    def order_value(self, code: str, value: float,
                    trade_time: str = None,
                    exec_price_override: float = None,
                    apply_slippage: bool = True,
                    trade_reason: str = None,
                    ) -> bool:
        """
        按目标金额下单 — v12 模拟实盘交易。

        交易规则:
        - 买入/卖出均取整到整手（lot_size）
        - 科创板: lot=200, inc=1（200股起，1股递增）
        - 主板/创业板: lot=100, inc=100
        - T+1 卖出限制（基于 Position.buy_date）
        - 三费: 佣金(0.3‰双向) + 印花税(0.05‰卖出) + 过户费(0.01‰双向)
        - 资金不足时自动调整为最大可买
        - 零股规则: 取整后剩余<1手则清仓

        新增参数 (M4):
            trade_time: 交易时间 'YYYY-MM-DD HH:MM:SS'，None 则默认为当日开盘时间
            exec_price_override: 指定成交价（覆盖默认 raw_open+slippage），
                                 用于分钟线止盈止损等场景
                                 【预留，当前仅卖出使用】
            apply_slippage: 是否对 exec_price_override 施加滑点，默认 True
                           分钟线触发价本身已是保守估计，建议传 False
            trade_reason: 订单原因标签（如 'profit_take'/'stop_loss'/'rebalance'/'expire'）
        """
        if hasattr(value, 'item'):
            value = float(value)
        if value == 0:
            return True

        is_buy = value > 0
        abs_value = abs(value)

        # M4: 交易时间
        if trade_time is None:
            _trade_time = f"{self.current_date.strftime('%Y-%m-%d')} 09:30:00"
        else:
            _trade_time = trade_time

        # 涨跌停检查
        if not self._check_limit(code, is_buy):
            return False

        # 执行价格
        # M4: 支持 exec_price_override 覆盖默认价格
        if exec_price_override is not None:
            if apply_slippage:
                if is_buy:
                    exec_price = exec_price_override * (1 + self.slippage)
                else:
                    exec_price = exec_price_override * (1 - self.slippage)
            else:
                exec_price = exec_price_override
        else:
            exec_price = self._get_exec_price(code, is_buy)
        if np.isnan(exec_price) or exec_price <= 0:
            return False

        # 交易单位
        lot_size = 200 if code.startswith('sh.688') else 100
        increment = 1 if code.startswith('sh.688') else 100

        if is_buy:
            # ============ 买入 ============
            target_qty = abs_value / exec_price
            quantity = int(target_qty // lot_size) * lot_size
            if quantity >= lot_size:
                remainder = target_qty - quantity
                additional = int(remainder // increment) * increment
                quantity += additional
            if quantity < lot_size:
                return False

            actual_value = quantity * exec_price
            commission = max(actual_value * self.commission_rate, self.min_commission)
            transfer_fee = actual_value * self.transfer_fee_rate
            total_cost = actual_value + commission + transfer_fee

            # M4: 交易前资金快照
            cash_before = self.portfolio.cash

            # 资金不足：调整为最大可买
            if total_cost > self.portfolio.cash:
                max_qty = int(
                    (self.portfolio.cash - self.min_commission)
                    / (exec_price * (1 + self.commission_rate + self.transfer_fee_rate))
                )
                quantity = int(max_qty // lot_size) * lot_size
                if quantity >= lot_size:
                    remainder = max_qty - quantity
                    additional = int(remainder // increment) * increment
                    quantity += additional
                if quantity < lot_size:
                    return False
                actual_value = quantity * exec_price
                commission = max(actual_value * self.commission_rate, self.min_commission)
                transfer_fee = actual_value * self.transfer_fee_rate
                total_cost = actual_value + commission + transfer_fee

            # [AUDIT] Log actual buy quantity
            if is_buy and quantity > 0:
                logger = logging.getLogger(__name__)
                logger.info(f"[BROKER AUDIT] BUY {code}: value={abs_value:.0f} total={self.portfolio.total_value:,.0f} "
                           f"qty={quantity} price={exec_price:.4f} cost={total_cost:.0f} cash_left={self.portfolio.cash - total_cost:,.0f}")
            self.portfolio.cash -= total_cost
            self.portfolio.add_position(code, quantity, exec_price,
                                         buy_date=self.current_date)
            self.total_commission += commission
            # [BUG-2修复] 重算total_value
            self.portfolio.total_value = (
                self.portfolio.cash
                + sum(float(p.market_value) for p in self.portfolio.positions.values()))

            self.orders.append({
                # ──── 原有字段 ────
                'date': self.current_date, 'code': code, 'direction': 'buy',
                'quantity': quantity, 'price': exec_price,
                'value': actual_value, 'commission': commission, 'tax': 0.0,
                'transfer_fee': transfer_fee,
                # ──── M4 新增字段 ────
                'trade_time': _trade_time,
                'trade_reason': trade_reason or 'default',
                'cash_before': cash_before,
                'cash_after': self.portfolio.cash,
                'avg_cost': exec_price,           # 买入时=成交价=建仓成本
                'profit_amount': np.nan,           # 买入时无盈亏
                'profit_ratio': np.nan,
                'real_profit_amount': np.nan,      # 买入时无盈亏
                'real_profit_ratio': np.nan,
            })

        else:
            # ============ 卖出 ============
            pos = self.portfolio.positions.get(code)
            if pos is None:
                return False

            # T+1: 检查买入日期
            if hasattr(pos, 'buy_date') and pos.buy_date is not None:
                if pos.buy_date.date() >= self.current_date.date():
                    return False

            # M4: 交易前读取持仓信息（必须在 remove_position 之前）
            cash_before = self.portfolio.cash
            avg_cost_before = float(pos.avg_cost)
            accumulated_div = float(getattr(pos, 'accumulated_dividend_net', 0.0))

            sell_qty = abs_value / exec_price

            # 清仓判断（目标金额接近市值 或 卖出数量接近持仓）
            is_liquidation = (
                abs(abs_value - pos.market_value) < 0.01
                or sell_qty >= pos.quantity * 0.99
            )

            if is_liquidation:
                sell_qty = pos.quantity  # 清仓不做取整
            else:
                # 非清仓卖出：向下取整到 lot_size 整数倍
                sell_qty = int(sell_qty // lot_size) * lot_size
                if sell_qty <= 0:
                    return False
                sell_qty = min(sell_qty, pos.quantity)

                # 零股规则：取整后剩余不足一手则清仓
                remaining = pos.quantity - sell_qty
                if 0 < remaining < lot_size:
                    sell_qty = pos.quantity

            if sell_qty <= 0:
                return False

            actual_value = sell_qty * exec_price
            commission = max(actual_value * self.commission_rate, self.min_commission)
            tax = actual_value * self.tax_rate
            transfer_fee = actual_value * self.transfer_fee_rate

            # M4: 计算比例分红收入（部分卖出时按比例分配）
            if sell_qty < pos.quantity and pos.quantity > 0:
                proportional_div = accumulated_div * (sell_qty / pos.quantity)
            else:
                proportional_div = accumulated_div  # 清仓时取全部

            # M4: 计算盈亏
            cost_basis = avg_cost_before * sell_qty
            profit_amount = exec_price * sell_qty - cost_basis - commission - tax - transfer_fee
            profit_ratio = profit_amount / cost_basis if cost_basis > 0 else 0.0
            real_profit_amount = profit_amount + proportional_div
            real_profit_ratio = real_profit_amount / cost_basis if cost_basis > 0 else 0.0

            self.portfolio.cash += actual_value - commission - tax - transfer_fee
            self.portfolio.remove_position(code, sell_qty)
            # [BUG-3修复] 更新剩余持仓的market_value
            if code in self.portfolio.positions:
                pos = self.portfolio.positions[code]
                pos.market_value = pos.quantity * exec_price
            # [BUG-2修复] 重算total_value
            self.portfolio.total_value = (
                self.portfolio.cash
                + sum(float(p.market_value) for p in self.portfolio.positions.values()))
            self.total_commission += commission
            self.total_tax += tax

            self.orders.append({
                # ──── 原有字段 ────
                'date': self.current_date, 'code': code, 'direction': 'sell',
                'quantity': sell_qty, 'price': exec_price,
                'value': actual_value, 'commission': commission, 'tax': tax,
                'transfer_fee': transfer_fee,
                # ──── M4 新增字段 ────
                'trade_time': _trade_time,
                'trade_reason': trade_reason or 'default',
                'cash_before': cash_before,
                'cash_after': self.portfolio.cash,
                'avg_cost': avg_cost_before,
                'profit_amount': profit_amount,        # 价格盈亏 = (卖价-成本)×数量-费用
                'profit_ratio': profit_ratio,
                'real_profit_amount': real_profit_amount,  # 含分红盈亏 = 价格盈亏+分红净收入
                'real_profit_ratio': real_profit_ratio,
            })

        return True

    def order_target_percent(self, code: str, percent: float,
                             trade_time: str = None,
                             exec_price_override: float = None,
                             apply_slippage: bool = True,
                             trade_reason: str = None,
                             ) -> bool:
        """
        按目标仓位比例下单。

        输入:
            code: 股票代码
            percent: 目标仓位比例（0~1之间）
        输出:
            是否成功执行
        作用:
            将指定股票调整至目标仓位比例
            比例大于当前：买入
            比例小于当前：卖出
        调用方:
            策略的handle_bar
        示例:
            broker.order_target_percent('sh.600000', 0.2)  # 买入至20%仓位
            broker.order_target_percent('sh.600000', 0.0)  # 清仓
        """
        total = float(self.portfolio.total_value)
        target = total * percent

        pos = self.portfolio.positions.get(code)
        current = float(pos.market_value) if pos else 0.0

        return self.order_value(code, target - current,
                                trade_time=trade_time,
                                exec_price_override=exec_price_override,
                                apply_slippage=apply_slippage,
                                trade_reason=trade_reason)

    def order_target_value(self, code: str, value: float,
                           trade_time: str = None,
                           exec_price_override: float = None,
                           apply_slippage: bool = True,
                           trade_reason: str = None,
                           ) -> bool:
        """
        按目标市值下单。

        输入:
            code: 股票代码
            value: 目标市值
        输出:
            是否成功执行
        作用:
            将指定股票调整至目标市值
        调用方:
            策略的handle_bar
        示例:
            broker.order_target_value('sh.600000', 10000)  # 持有约10000元市值
        """
        pos = self.portfolio.positions.get(code)
        current = float(pos.market_value) if pos else 0.0

        return self.order_value(code, value - current,
                                trade_time=trade_time,
                                exec_price_override=exec_price_override,
                                apply_slippage=apply_slippage,
                                trade_reason=trade_reason)
