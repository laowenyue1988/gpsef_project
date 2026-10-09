# -*- coding: utf-8 -*-

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
from typing import Dict, Any, List
from datetime import datetime

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

        # 交易费用参数
        self.commission_rate = config.get('commission_rate', 0.00025)  # 佣金率
        self.tax_rate = config.get('tax_rate', 0.001)  # 印花税率
        self.min_commission = config.get('min_commission', 5.0)  # 最低佣金
        self.slippage = config.get('slippage', 0.001)  # 滑点
        self.limit_threshold = config.get('limit_threshold', 0.098)  # 涨跌停阈值

        # 运行时状态
        self.current_date = None
        self.orders: List[Dict[str, Any]] = []
        self.total_commission = 0.0
        self.total_tax = 0.0

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

    def _get_exec_price(self, code: str, is_buy: bool) -> float:
        """
        获取执行价格（含滑点）。

        输入:
            code: 股票代码
            is_buy: 是否为买入
        输出:
            执行价格
        作用:
            获取当日开盘价（基于固定reference_date前复权），并根据买卖方向添加滑点
        调用方:
            SimulatedBroker.order_value

        注意:
            reference_date=trade_reference_date（固定值），不是current_date。
            这确保买卖价格在同一个前复权体系下，收益率自然包含分红信息。
        """
        df = self.data_provider.get_daily_data(
            code, self.current_date, self.current_date, reference_date=self.trade_reference_date
        )

        if df.empty:
            return np.nan

        price = _to_scalar(df.iloc[0]['open'])

        if is_buy:
            # 买入：价格上涨滑点
            return price * (1 + self.slippage)
        else:
            # 卖出：价格下降滑点
            return price * (1 - self.slippage)

    def _check_limit(self, code: str, is_buy: bool) -> bool:
        """
        检查涨跌停限制。

        输入:
            code: 股票代码
            is_buy: 是否为买入
        输出:
            True表示可以交易，False表示涨跌停限制
        作用:
            - 买入时：限制买入涨停股
            - 卖出时：限制卖出跌停股
        调用方:
            SimulatedBroker.order_value

        注意:
            涨跌停判断使用 (open - preclose) / preclose。
            虽然公式数学上 ref-independent（bf/bf[ref] 抵消），
            但 FP64 下不同 trade_reference_date 会产生 1-2 ULP 的差异，
            在精确 ±10% 边界上导致两个 ref 的 True/False 判定不同。
            因此强制使用 reference_date=current_date 确保比特级一致性。
        """
        df = self.data_provider.get_daily_data(
            code, self.current_date, self.current_date, reference_date=self.current_date
        )

        if df.empty:
            return False

        row = df.iloc[0]

        if 'preclose' not in row:
            return True

        preclose = _to_scalar(row['preclose'])
        open_price = _to_scalar(row['open'])

        if preclose == 0:
            return True

        change = (open_price - preclose) / preclose

        if is_buy:
            # 涨停股不能买入
            return change <= self.limit_threshold
        else:
            # 跌停股不能卖出
            return change >= -self.limit_threshold

    def order_value(self, code: str, value: float) -> bool:
        """
        按目标金额下单。

        输入:
            code: 股票代码
            value: 目标金额（正数=买入，负数=卖出）
        输出:
            是否成功执行
        作用:
            - 正数：买入指定金额的股票
            - 负数：卖出指定金额的股票
            - 自动计算实际可交易数量
        调用方:
            SimulatedBroker.order_target_percent
            策略的handle_bar
        示例:
            broker.order_value('sh.600000', 10000)  # 买入约10000元的股票
            broker.order_value('sh.600000', -10000) # 卖出约10000元的股票
        """
        if hasattr(value, 'item'):
            value = float(value)



        if value == 0:
            return True

        is_buy = value > 0
        abs_value = abs(value)
        # 检查数据可用性
        df_check = self.data_provider.get_daily_data(
            code, self.current_date, self.current_date, reference_date=self.trade_reference_date
        )
        if df_check is None or df_check.empty:
            return False

        # 检查涨跌停
        if not self._check_limit(code, is_buy):
            return False

        # 获取执行价格
        exec_price = self._get_exec_price(code, is_buy)
        if np.isnan(exec_price):
            return False

        # 计算可交易数量（分数股，消除100股整数倍的回测偏差）
        # 业界最佳实践：回测使用分数股，因为100股整数倍是市场微观结构
        # 不应影响alpha策略评估。使用整数倍会导致不同复权基准日下产生不可比较的结果
        quantity = abs_value / exec_price
        if quantity <= 0:
            return False

        # 计算实际金额（分数股下等于目标金额）
        actual_value = abs_value

        if is_buy:
            # ===== 买入 =====
            commission = max(actual_value * self.commission_rate, self.min_commission)
            tax = 0.0  # 买入不收印花税
            # 计算总成本
            total_cost = actual_value + commission

            # 资金不足时调整数量
            if total_cost > self.portfolio.cash:
                # 用剩余现金尽量买入
                max_affordable = (self.portfolio.cash - self.min_commission) / (1 + self.commission_rate)
                if max_affordable <= 0:
                    return False
                quantity = max_affordable / exec_price
                actual_value = max_affordable
                commission = max(actual_value * self.commission_rate, self.min_commission)
                total_cost = actual_value + commission

            # 更新现金和持仓
            self.portfolio.cash -= total_cost
            self.portfolio.add_position(code, quantity, exec_price)
            self.total_commission += commission

        else:
            # ===== 卖出 =====
            pos = self.portfolio.positions.get(code)
            if pos is None:
                return False

            # 判断是否为清仓（目标金额接近持仓市值）
            if abs(abs_value - pos.market_value) < 0.01 or abs_value / exec_price < pos.quantity * 0.01:
                # 清仓模式：卖出全部持仓
                sell_qty = pos.quantity
                actual_value = pos.market_value  # 按市值清仓
            else:
                sell_qty = abs_value / exec_price  # 分数股卖出
                sell_qty = min(sell_qty, pos.quantity)
                if sell_qty <= 0:
                    return False
                actual_value = sell_qty * exec_price
            commission = max(actual_value * self.commission_rate, self.min_commission)
            tax = actual_value * self.tax_rate  # 卖出收取印花税

            # 更新现金和持仓
            self.portfolio.cash += actual_value - commission - tax
            self.portfolio.remove_position(code, sell_qty)
            self.total_commission += commission
            self.total_tax += tax

        # 记录订单
        self.orders.append({
            'date': self.current_date,
            'code': code,
            'direction': 'buy' if is_buy else 'sell',
            'quantity': quantity if is_buy else sell_qty,
            'price': exec_price,
            'value': actual_value,
            'commission': commission,
            'tax': tax
        })

        return True

    def order_target_percent(self, code: str, percent: float) -> bool:
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

        return self.order_value(code, target - current)

    def order_target_value(self, code: str, value: float) -> bool:
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

        return self.order_value(code, value - current)
