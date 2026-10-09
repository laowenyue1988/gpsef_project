# -*- coding: utf-8 -*-
"""Alpha191 专用股票池管理器（保留原逻辑，但改为使用框架 DataProvider）"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
import logging
from .data_utils import get_trade_calendar

logger = logging.getLogger(__name__)

class StockPoolManager:
    def __init__(self):
        self.stock_basic_info = None
        self._st_history_df = None
        self._candidate_pool_cache = {}

    def load_stock_basic_info(self, df_daily: pd.DataFrame):
        self.stock_basic_info = df_daily.groupby('ts_code').agg({'trade_date': 'min'}).reset_index()
        self.stock_basic_info.columns = ['ts_code', 'list_date']
        self.stock_basic_info['list_date'] = pd.to_datetime(self.stock_basic_info['list_date'])
        if 'isST' in df_daily.columns:
            st_mask = pd.to_numeric(df_daily['isST'], errors='coerce').fillna(0) == 1
            if st_mask.any():
                self._st_history_df = df_daily.loc[st_mask, ['ts_code', 'trade_date']].copy()
                self._st_history_df['trade_date'] = pd.to_datetime(self._st_history_df['trade_date'])

    def get_candidate_pool(self, date: datetime, df_daily: pd.DataFrame, stock_list: Optional[List[str]] = None) -> List[str]:
        if date in self._candidate_pool_cache:
            full = self._candidate_pool_cache[date]
        else:
            full = self._compute_candidate_pool(date, df_daily)
            self._candidate_pool_cache[date] = full
        if stock_list:
            return [s for s in full if s in stock_list]
        return full

    def _compute_candidate_pool(self, date: datetime, df_daily: pd.DataFrame) -> List[str]:
        trade_days = get_trade_calendar((date - timedelta(days=30)).strftime('%Y%m%d'), date.strftime('%Y%m%d'))
        trade_days = trade_days[trade_days['calendar_date'] < date]
        if len(trade_days) < 2:
            return []
        t_minus_1 = trade_days['calendar_date'].iloc[-1]
        t_minus_2 = trade_days['calendar_date'].iloc[-2]

        def tradable(d):
            mask = df_daily['trade_date'] == d
            data = df_daily[mask]
            if 'tradestatus' in data.columns:
                data = data[pd.to_numeric(data['tradestatus'], errors='coerce').fillna(0) == 1]
            return set(data['ts_code'].unique())

        valid = tradable(t_minus_1) & tradable(t_minus_2)
        valid_prefixes = ('sh.600', 'sh.601', 'sh.603', 'sh.605', 'sh.688', 'sz.000', 'sz.002', 'sz.300')
        candidates = [c for c in valid if c.startswith(valid_prefixes)]

        # ST过滤（当日）
        t1_data = df_daily[df_daily['trade_date'] == t_minus_1]
        t1_data = t1_data[t1_data['ts_code'].isin(candidates)]
        if 'isST' in t1_data.columns:
            st_vals = pd.to_numeric(t1_data['isST'], errors='coerce').fillna(0)
            candidates = t1_data.loc[st_vals == 0, 'ts_code'].tolist()

        # 次新股过滤
        if self.stock_basic_info is not None:
            list_map = self.stock_basic_info.set_index('ts_code')['list_date'].to_dict()
            final = []
            for c in candidates:
                list_d = list_map.get(c)
                if list_d is None:
                    continue
                days_since = (t_minus_1 - list_d).days
                hist = df_daily[(df_daily['ts_code'] == c) & (df_daily['trade_date'] <= t_minus_1)]
                if days_since >= 60 and len(hist) >= 60:
                    final.append(c)
            candidates = final

        # ST历史过滤
        if self._st_history_df is not None and not self._st_history_df.empty:
            recent = self._st_history_df[(self._st_history_df['trade_date'] >= t_minus_1 - timedelta(days=180)) &
                                         (self._st_history_df['trade_date'] <= t_minus_1)]
            st_codes = set(recent['ts_code'].unique())
            candidates = [c for c in candidates if c not in st_codes]

        # 价格过滤（最近3日>=1元）
        last_three = trade_days['calendar_date'].iloc[-3:].tolist()
        price_data = df_daily[(df_daily['trade_date'].isin(last_three)) & (df_daily['ts_code'].isin(candidates))]
        if not price_data.empty:
            pivot = price_data.pivot_table(index='ts_code', columns='trade_date', values='close', aggfunc='first').dropna()
            candidates = pivot[(pivot >= 1).all(axis=1)].index.tolist()

        # 成交额过滤（后10%）
        if candidates:
            amt = df_daily[(df_daily['trade_date'] == t_minus_1) & (df_daily['ts_code'].isin(candidates))]
            amt = amt.set_index('ts_code')['amount'].dropna()
            if len(amt) > 0:
                threshold = amt.quantile(0.1)
                candidates = amt[amt > threshold].index.tolist()

        return candidates