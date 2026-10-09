# -*- coding: utf-8 -*-

"""
动态股票池模块，提供高性能的候选股票过滤功能。
"""

import logging
from typing import List, Set, Dict, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BaseStockPool:
    """股票池基类"""

    def get_candidates(self, date, context) -> List[str]:
        """获取指定日期的候选股票池"""
        raise NotImplementedError


class DynamicStockPool(BaseStockPool):
    """动态候选股票池，向量化截面过滤版。"""

    # [O2-fix-20260910] 前缀表补全: 原缺 sh.689 (科创板CDR) / sz.003 (深主板新段) /
    #   sz.302 (创业板注册制新段) → 三个真实股票段被整段排除在宇宙外 (下载器与策略侧
    #   data_utils 的全量名单早已包含, 此前靠策略实例级遮蔽绕过 —— 家族其余策略仍在错误
    #   名单上)。本表为**唯一真值源**; 一致性由 tests/test_universe_consistency.py 守卫。
    VALID_PREFIXES = (
        'sh.600', 'sh.601', 'sh.603', 'sh.605', 'sh.688', 'sh.689',
        'sz.000', 'sz.001', 'sz.002', 'sz.003', 'sz.300', 'sz.301', 'sz.302'
    )

    def __init__(self, config: dict):
        self.filters = config.get('filters', ['st', 'new_stock'])
        self.price_min = None
        self.amount_pct = None
        for f in self.filters:
            if f.startswith('price_min:'):
                self.price_min = float(f.split(':')[1])
            elif f.startswith('amount_pct:'):
                self.amount_pct = float(f.split(':')[1])
        # ===== 核心数据结构 =====
        # 上市日期映射 {code: list_date}
        self._list_date_map: Dict[str, pd.Timestamp] = {}
        # 日线截面矩阵：dict {date -> DataFrame(index=code, columns=[close, amount, isST, tradestatus])}
        self._cross_section: Dict[pd.Timestamp, pd.DataFrame] = {}
        # 交易日列表（排序后）
        self._sorted_dates: List[pd.Timestamp] = []
        # 日期索引映射（加速查找 T-1, T-2）
        self._date_idx_map: Dict[pd.Timestamp, int] = {}
        # ST 历史记录 {code: set(dates)}
        self._st_history: Dict[str, Set[pd.Timestamp]] = {}
        # 缓存
        self._cache: Dict[datetime, List[str]] = {}

    def initialize(self, data_provider, min_date: Optional[datetime] = None, end_date: Optional[datetime] = None):
        """
        初始化：一次性加载全量日线，构建截面矩阵。

        [FIX-REF-DATE] 修复：新增 end_date 参数，避免使用 datetime.now() 导致未来数据泄露。
        调用方应在回测场景下传入回测结束日期，在实盘场景下才使用当前时间。

        输入：
            data_provider: 数据提供者实例
            min_date: 数据加载起始日期（可选，默认向前400天）
            end_date: 数据加载结束日期（可选，默认使用 datetime.now()）
        """
        logger.info("正在初始化 DynamicStockPool（高性能版）...")

        # 1. 获取全市场股票列表
        all_stocks = data_provider.get_stock_list()
        logger.info(f"全市场股票数量: {len(all_stocks)}")

        # 2. 确定数据范围
        # [FIX-REF-DATE] 修复：优先使用传入的 end_date，避免未来数据泄露
        if end_date is None:
            end_date = datetime.now()
        if min_date is None:
            min_date = end_date - timedelta(days=400)

        logger.info(f"数据加载范围: {min_date.date()} 至 {end_date.date()}")

        # 3. 批量加载日线，一次性收集
        records = []

        for code in all_stocks:
            try:
                # [FIX-REF-DATE] 注意：此处仍使用 end_date 作为 reference_date
                # 这是因为我们需要获取"当前视角"下的复权价格，用于截面选股
                df = data_provider.get_daily_data(
                    code, min_date, end_date, reference_date=end_date
                )
                if df.empty:
                    continue
                sub = df[['trade_date', 'tradestatus', 'isST', 'close', 'raw_close', 'amount', 'raw_amount']].copy()
                sub['tradestatus'] = pd.to_numeric(sub['tradestatus'], errors='coerce').fillna(0).astype(int)
                sub['isST'] = pd.to_numeric(sub['isST'], errors='coerce').fillna(0).astype(int)
                sub['code'] = code
                sub['close'] = pd.to_numeric(sub['close'], errors='coerce')
                sub['raw_close'] = pd.to_numeric(sub['raw_close'], errors='coerce')
                sub['amount'] = pd.to_numeric(sub['amount'], errors='coerce')
                sub['raw_amount'] = pd.to_numeric(sub['raw_amount'], errors='coerce')
                # 上市日期
                self._list_date_map[code] = df['trade_date'].min()
                # ST 历史
                st_dates = sub.loc[sub['isST'] == 1, 'trade_date'].tolist()
                if st_dates:
                    self._st_history[code] = set(pd.to_datetime(st_dates))
                records.append(sub)
            except Exception as e:
                logger.debug(f"加载 {code} 失败: {e}")

        if not records:
            logger.error("未能加载任何日线数据")
            return

        # 4. 构建全量 DataFrame
        all_df = pd.concat(records, ignore_index=True)
        all_df['trade_date'] = pd.to_datetime(all_df['trade_date'])
        logger.info(f"日线记录总量: {len(all_df)}")

        # 5. 构建截面矩阵：按日期分组，每个日期一个 code×columns 的 DataFrame
        for date, group in all_df.groupby('trade_date'):
            section = group.set_index('code')[['close', 'raw_close', 'amount', 'raw_amount', 'isST', 'tradestatus']]
            self._cross_section[date] = section

        # 6. 排序日期并建立索引
        self._sorted_dates = sorted(self._cross_section.keys())
        self._date_idx_map = {d: i for i, d in enumerate(self._sorted_dates)}

        logger.info(f"截面矩阵构建完成: {len(self._cross_section)} 个交易日, "
                    f"{len(self._list_date_map)} 只股票")

    def _get_prev_trade_dates(self, date: datetime, n: int = 3) -> List[pd.Timestamp]:
        """
        获取 date 之前的 n 个交易日（不包含 date 本身）。
        返回 [T-1, T-2, ..., T-n]，若不足则返回实际数量。
        """
        target = pd.Timestamp(date.date())
        idx = np.searchsorted(self._sorted_dates, target, side='left')
        if idx < n:
            return []
        return self._sorted_dates[idx - n: idx]

    def _get_tradable_codes(self, date: pd.Timestamp) -> Set[str]:
        """获取某日可交易的股票代码集合。"""
        section = self._cross_section.get(date)
        if section is None:
            return set()
        mask = section['tradestatus'] == 1
        return set(section[mask].index)

    def _get_section(self, date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """获取某日的截面数据。"""
        return self._cross_section.get(date)

    def get_candidates(self, date: datetime, context: 'StrategyContext') -> List[str]:
        """
        获取指定交易日的候选股票池（向量化过滤）。
        """
        if date in self._cache:
            return self._cache[date]

        # ===== 1. 获取历史交易日 =====
        prev_dates = self._get_prev_trade_dates(date, n=3)
        if len(prev_dates) < 2:
            logger.warning(f"[{date.date()}] 历史交易日不足，返回空")
            return []

        t_minus_1 = prev_dates[-1]
        t_minus_2 = prev_dates[-2]

        # ===== 2. 可交易过滤（T-1 和 T-2 均可交易） =====
        tradable_t1 = self._get_tradable_codes(t_minus_1)
        tradable_t2 = self._get_tradable_codes(t_minus_2)
        valid = tradable_t1 & tradable_t2

        # 前缀过滤
        candidates = [c for c in valid if c.startswith(self.VALID_PREFIXES)]
        if not candidates:
            self._cache[date] = []
            return []

        # ===== 3. 向量化截面过滤 =====
        section_t1 = self._get_section(t_minus_1)
        if section_t1 is None:
            self._cache[date] = []
            return []

        cand_set = set(candidates)
        sec = section_t1[section_t1.index.isin(cand_set)].copy()
        if sec.empty:
            self._cache[date] = []
            return []

        # ===== 4. ST 过滤（T-1 不为 ST） =====
        if 'st' in self.filters:
            sec = sec[sec['isST'] == 0]
            logger.debug(f"[{date.date()}] ST过滤后: {len(sec)}")

        # ===== 5. ST 历史过滤（180天内未被 ST） =====
        if self._st_history and 'st' in self.filters:
            check_start = t_minus_1 - timedelta(days=180)
            st_recent_codes = set()
            for code, st_dates in self._st_history.items():
                for d in st_dates:
                    if check_start <= d <= t_minus_1:
                        st_recent_codes.add(code)
                        break
            if st_recent_codes:
                sec = sec[~sec.index.isin(st_recent_codes)]
                logger.debug(f"[{date.date()}] ST历史过滤后: {len(sec)}")

        # ===== 6. 次新股过滤（上市满60个交易日） =====
        if 'new_stock' in self.filters and not sec.empty:
            t1_idx = self._date_idx_map.get(t_minus_1, -1)
            if t1_idx >= 60:
                surviving = []
                # 预取最近120个交易日的截面，避免重复取
                start_idx = max(0, t1_idx - 119)
                recent_dates = self._sorted_dates[start_idx: t1_idx + 1]
                recent_sections = [self._cross_section.get(d) for d in recent_dates]
                for code in sec.index:
                    list_date = self._list_date_map.get(code)
                    if list_date is None:
                        continue
                    if (t_minus_1 - list_date).days < 60:
                        continue
                    # 统计该股票在最近截面中出现的次数
                    count = 0
                    for s in recent_sections:
                        if s is not None and code in s.index:
                            count += 1
                    if count >= 60:
                        surviving.append(code)
                sec = sec[sec.index.isin(surviving)]
            logger.debug(f"[{date.date()}] 次新股过滤后: {len(sec)}")

        # ===== 7. 价格过滤（最近3日收盘价均 >= min_price） =====
        min_price = self.price_min if self.price_min is not None else 1.0
        if not sec.empty:
            price_ok_codes = set(sec.index)
            for d in prev_dates:
                s = self._get_section(d)
                if s is None:
                    continue
                low_price = s[(s.index.isin(price_ok_codes)) & (s['raw_close'] < min_price)].index
                price_ok_codes -= set(low_price)
            sec = sec[sec.index.isin(price_ok_codes)]
            logger.debug(f"[{date.date()}] 价格过滤(>={min_price})后: {len(sec)}")

        # ===== 8. 成交额过滤（剔除后 x%） =====
        if self.amount_pct is not None and not sec.empty:
            amounts = sec['raw_amount'].dropna()
            amounts = amounts[amounts > 0]
            if not amounts.empty:
                threshold = amounts.quantile(self.amount_pct)
                sec = sec[sec['raw_amount'] >= threshold]
                logger.debug(f"[{date.date()}] 成交额过滤后: {len(sec)}")

        # ===== 9. 返回 =====
        result = sec.index.tolist()
        self._cache[date] = result
        logger.info(f"[{date.date()}] 最终候选池: {len(result)} 只")
        return result
