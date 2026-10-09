from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
from .portfolio import Portfolio
from .interfaces import BaseDataProvider, BaseStockPool

@dataclass
class StrategyContext:
    config: Dict[str, Any] = field(default_factory=dict)
    data_provider: Optional[BaseDataProvider] = None
    stock_pool: Optional[BaseStockPool] = None
    broker: Optional[Any] = None
    current_dt: Optional[datetime] = None
    portfolio: Optional[Portfolio] = None
    calendar: List[datetime] = field(default_factory=list)
    user_cache: Dict[str, Any] = field(default_factory=dict)