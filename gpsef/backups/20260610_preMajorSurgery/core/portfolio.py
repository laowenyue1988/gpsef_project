from dataclasses import dataclass, field
from typing import Dict

@dataclass
class Position:
    code: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0

class Portfolio:
    def __init__(self, initial_cash: float = 1000000.0):
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.total_value = initial_cash

    def update_value(self, prices: Dict[str, float]):
        pos_val = 0.0
        for code, pos in self.positions.items():
            if code in prices:
                price = prices[code]
                # 强制转换为 float 标量
                if hasattr(price, 'item'):
                    price = float(price)
                pos.current_price = price
                pos.market_value = float(pos.quantity) * price
                pos_val += pos.market_value
        self.total_value = self.cash + pos_val

    def add_position(self, code: str, quantity: float, price: float):
        if code in self.positions:
            pos = self.positions[code]
            total_cost = pos.quantity * pos.avg_cost + quantity * price
            pos.quantity += quantity
            pos.avg_cost = total_cost / pos.quantity if pos.quantity > 0 else 0
        else:
            self.positions[code] = Position(code, quantity, price)

    def remove_position(self, code: str, quantity: float):
        if code in self.positions:
            pos = self.positions[code]
            pos.quantity -= quantity
            if pos.quantity <= 0:
                del self.positions[code]