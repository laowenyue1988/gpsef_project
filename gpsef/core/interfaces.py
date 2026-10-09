# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

class BaseDataProvider(ABC):
    @abstractmethod
    def get_daily_data(self, code: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """返回前复权日线数据，必须包含 trade_date, open, high, low, close, volume, amount, tradestatus, isST"""
        pass

    @abstractmethod
    def get_trade_calendar(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        pass

    @abstractmethod
    def get_stock_list(self) -> List[str]:
        pass

    def get_benchmark_data(self, code: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        return None

    def get_factor_snapshot(self, date: datetime) -> Optional[pd.DataFrame]:
        return None


class BaseStockPool(ABC):
    @abstractmethod
    def get_candidates(self, date: datetime, context: 'StrategyContext') -> List[str]:
        pass


class BaseFactorCalculator(ABC):
    @abstractmethod
    def calculate(self, df_daily: pd.DataFrame, reference_date: Optional[datetime] = None) -> pd.DataFrame:
        pass

    @property
    def factor_names(self) -> List[str]:
        return []


class BaseModelManager(ABC):
    @abstractmethod
    def train(self, train_data: pd.DataFrame, feature_cols: List[str], **kwargs) -> bool:
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.Series:
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass


class IStrategy(ABC):
    @abstractmethod
    def initialize(self, context: 'StrategyContext'):
        pass

    @abstractmethod
    def before_market_open(self, context: 'StrategyContext'):
        pass

    @abstractmethod
    def handle_bar(self, context: 'StrategyContext'):
        pass