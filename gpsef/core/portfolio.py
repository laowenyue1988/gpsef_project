from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class Position:
    code: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    buy_date: Optional[datetime] = None
    # ──── M1 新增 ────
    prev_market_value: float = 0.0               # 前一交易日市值（用于 daily_pnl）
    accumulated_dividend_net: float = 0.0        # 持仓期间累计分红净收入（税后-送股税）

class Portfolio:
    def __init__(self, initial_cash: float = 1000000.0):
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.total_value = initial_cash

    def update_value(self, prices: Dict[str, float], save_prev: bool = True) -> list:
        """
        更新持仓市值。

        BUG-3 修复: 返回估值失败的股票代码列表
        BUG-4 修复: save_prev=False 时跳过 prev_market_value 的保存
        """
        pos_val = 0.0
        failed = []
        for code, pos in self.positions.items():
            if code in prices:
                price = prices[code]
                # 强制转换为 float 标量
                if hasattr(price, 'item'):
                    price = float(price)
                # BUG-4: 仅在 save_prev=True 时保存前日市值
                if save_prev:
                    pos.prev_market_value = pos.market_value
                pos.current_price = price
                pos.market_value = float(pos.quantity) * price
            else:
                # BUG-3: 记录估值失败的股票
                failed.append(code)
                logger.warning(f"[Portfolio] {code} not in prices, mv stays {pos.market_value:.2f}")
            pos_val += pos.market_value
        self.total_value = self.cash + pos_val
        return failed

    def add_position(self, code: str, quantity: float, price: float,
                     buy_date: Optional[datetime] = None):
        if code in self.positions:
            pos = self.positions[code]
            total_cost = pos.quantity * pos.avg_cost + quantity * price
            pos.quantity += quantity
            pos.avg_cost = total_cost / pos.quantity if pos.quantity > 0 else 0
            # [BUG-4修复] 更新market_value
            pos.market_value = pos.quantity * price
        else:
            pos = Position(code, quantity, price, buy_date=buy_date)
            # [BUG-4修复] 新建仓设置market_value
            pos.market_value = quantity * price
            self.positions[code] = pos

    def remove_position(self, code: str, quantity: float):
        if code in self.positions:
            pos = self.positions[code]
            old_qty = pos.quantity
            pos.quantity -= quantity
            if pos.quantity <= 0:
                del self.positions[code]
            else:
                # M2: 比例缩减累计分红净收入
                if old_qty > 0 and pos.accumulated_dividend_net > 0:
                    pos.accumulated_dividend_net *= (pos.quantity / old_qty)

    def adjust_position(self, code: str, new_quantity: float, new_avg_cost: float):
        """
        因送股/转增调整持仓数量和均价。

        关键性质：此操作不改变持仓市值（因为除权后单价等比例下降），
        也不改变现金和总净值。调用方需在完成后重算 total_value。

        BUG-5 修复: 更新 market_value = new_quantity * new_avg_cost，
        确保除权后 quantity/avgcost/marketvalue 三者联动一致。

        Args:
            code: 股票代码
            new_quantity: 除权后新数量 = old_quantity * (1 + stock_ratio)
            new_avg_cost: 除权后新均价 = old_avg_cost / (1 + stock_ratio)
        """
        if code in self.positions:
            pos = self.positions[code]
            pos.quantity = new_quantity
            pos.avg_cost = new_avg_cost
            # BUG-5: 除权不改变市值（qty↑ 等比例抵消 price↓）
            pos.market_value = new_quantity * new_avg_cost