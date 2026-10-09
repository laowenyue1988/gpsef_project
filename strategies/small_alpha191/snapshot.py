# -*- coding: utf-8 -*-
"""
PrimarySelectionSnapshotManager - 基本面主模型选股快照管理器。

【改造说明】
将缓存结构从 Dict[datetime, List[str]] 改为 Dict[datetime, Dict]，
每个缓存项现在包含：
- 'stocks': List[str] - 推荐的股票代码列表
- 'rank_pcts': Dict[str, float] - 每只股票的排名百分位（1=最高排名）

【设计目的】
在元标注模型训练过程中，需要对训练窗口内的每个历史交易日重新运行主模型选股
（历史回放模式）。如果每次训练都实时调用主模型选股，会导致同一个交易日被
重复选股数百次，效率极低。

本模块通过预生成并缓存每个交易日的主模型选股结果，将训练时的实时选股
转换为 O(1) 的缓存读取，大幅提升训练效率。

【快照范围】
覆盖 [回测开始日期 - pre_days, 回测结束日期] 的所有交易日，
其中 pre_days 默认为 120 天（确保训练窗口有足够的历史数据）。

【使用流程】
1. 在策略 initialize() 中，主模型加载成功后，调用 generate_snapshots()
   预生成整个回测区间的选股快照。
2. 在 _get_primary_recommendations() 中，优先从快照读取当前日期的选股结果。
3. 在 _get_primary_recommendations_for_date() 中，从快照读取历史日期的选股结果。
4. 如果某天快照不存在，回退到实时调用主模型选股（降级保护）。

【向后兼容】
支持旧格式快照（只包含stocks字段）的加载，旧格式快照的rank_pcts字段为空dict。
"""
import os
import time
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Union
import pandas as pd

logger = logging.getLogger(__name__)


class PrimarySelectionSnapshotManager:
    """
    基本面主模型选股快照管理器。

    按日期保存为独立 JSON 文件，支持增量加载和跨回测区间复用。

    【改造】缓存结构更新：
    - 新格式: Dict[datetime, Dict]，每个项包含 'stocks' 和 'rank_pcts'
    - 向后兼容: 旧格式快照（只有stocks）加载后rank_pcts为空dict
    """

    def __init__(self, snapshot_dir: str = './snapshots'):
        """
        初始化快照管理器。

        Args:
            snapshot_dir: 快照持久化目录。每个交易日的快照保存为单独的
                         JSON 文件 ({date}.json)，方便查看和增量更新。
        """
        self.snapshot_dir = snapshot_dir
        # 【改造】缓存结构从 List[str] 改为 Dict
        # 每个项格式: {'stocks': List[str], 'rank_pcts': Dict[str, float]}
        self._cache: Dict[datetime, Dict] = {}
        self._generated = False
        os.makedirs(snapshot_dir, exist_ok=True)

    # ==================== 核心接口 ====================

    def generate_snapshots(self, calendar: List[datetime],
                           get_recommendations_func: Callable[[datetime], tuple],
                           pre_days: int = 0) -> Dict[datetime, Dict]:
        """
        预生成主模型选股快照，支持增量加载已有快照。

        【改造】get_recommendations_func 签名变更：
        - 旧签名: f(date) -> List[str]
        - 新签名: f(date) -> tuple(List[str], Dict[str, float])
                 返回 (推荐股票列表, 排名百分位字典)

        Args:
            calendar: 需要生成快照的交易日列表（已排序）
            get_recommendations_func: 主模型选股函数，
                                     签名为 f(date) -> (List[str], Dict[str, float])
                                     返回 (推荐股票列表, 排名百分位字典)
            pre_days: 已在 calendar 中包含，此处忽略

        Returns:
            Dict[datetime, Dict]: 所有选股快照（含已有+新生成的）
            每个Dict格式: {'stocks': List[str], 'rank_pcts': Dict[str, float]}
        """
        if not calendar:
            logger.warning("交易日历为空，无法生成快照")
            return self._cache

        logger.info("=" * 60)
        logger.info(f"【快照生成】目标: {len(calendar)} 个交易日")
        logger.info(f"  范围: {calendar[0].date()} ~ {calendar[-1].date()}")
        logger.info(f"  目录: {self.snapshot_dir}")
        logger.info("=" * 60)

        # 第1步：扫描已有快照
        existing = self._scan_existing_snapshots(calendar)
        logger.info(f"【快照】已有 {len(existing)} 个快照可从本地加载")

        # 第2步：需要新生成的日期
        missing_dates = [d for d in calendar if d not in existing]
        logger.info(f"【快照】需生成 {len(missing_dates)} 个新快照")

        # 第3步：生成缺失的快照
        start_time = time.time()
        success_count = 0
        fail_count = 0
        skipped_count = len(existing)

        for i, date in enumerate(missing_dates):
            try:
                # 【改造】新签名返回 (stocks, rank_pcts)
                result = get_recommendations_func(date)
                if isinstance(result, tuple) and len(result) == 2:
                    stocks, rank_pcts = result
                elif isinstance(result, list):
                    # 向后兼容：旧函数只返回列表
                    stocks = result
                    rank_pcts = {}
                else:
                    stocks = []
                    rank_pcts = {}

                self._cache[date] = {
                    'stocks': list(stocks) if stocks else [],
                    'rank_pcts': dict(rank_pcts) if rank_pcts else {}
                }
                self._save_snapshot_to_disk(date, self._cache[date])
                if stocks:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logger.warning(f"快照生成失败 {date.date()}: {e}")
                self._cache[date] = {'stocks': [], 'rank_pcts': {}}
                self._save_snapshot_to_disk(date, self._cache[date])
                fail_count += 1

            # 每50天打印一次进度
            if (i + 1) % 50 == 0 or (i + 1) == len(missing_dates):
                elapsed = time.time() - start_time
                logger.info(
                    f"  进度: {i + 1}/{len(missing_dates)}, "
                    f"成功: {success_count}, 失败: {fail_count}, "
                    f"耗时: {elapsed:.1f}s"
                )

        total_elapsed = time.time() - start_time
        self._generated = True

        logger.info("=" * 60)
        logger.info(
            f"【快照生成完成】总计: {len(self._cache)} 天, "
            f"加载: {skipped_count}, 新生成: {success_count + fail_count}, "
            f"成功: {success_count}, 失败: {fail_count}, "
            f"耗时: {total_elapsed:.2f}s"
        )
        logger.info("=" * 60)

        return self._cache

    def get(self, date: datetime) -> Optional[List[str]]:
        """
        获取指定日期的选股结果。

        Args:
            date: 交易日

        Returns:
            List[str] 或 None: 该日的主模型选股结果。None 表示快照不存在。
        """
        # 直接查找
        if date in self._cache:
            val = self._cache[date]
            return val.get('stocks', [])

        # 尝试从磁盘加载（可能之前扫描时未覆盖）
        disk_val = self._load_snapshot_from_disk(date)
        if disk_val is not None:
            self._cache[date] = disk_val
            return disk_val.get('stocks', [])

        # pd.Timestamp 匹配
        for cached_date in list(self._cache.keys()):
            if pd.Timestamp(cached_date) == pd.Timestamp(date):
                val = self._cache[cached_date]
                return val.get('stocks', [])

        return None

    def get_rank_pcts(self, date: datetime) -> Dict[str, float]:
        """
        【新增】获取指定日期的股票排名百分位。

        Args:
            date: 交易日

        Returns:
            Dict[str, float]: {code: rank_pct}，rank_pct范围[0, 1]
                              如果快照不存在或旧格式快照，返回空dict。
        """
        # 直接查找
        if date in self._cache:
            return self._cache[date].get('rank_pcts', {})

        # 尝试从磁盘加载
        disk_val = self._load_snapshot_from_disk(date)
        if disk_val is not None:
            self._cache[date] = disk_val
            return disk_val.get('rank_pcts', {})

        # pd.Timestamp 匹配
        for cached_date in list(self._cache.keys()):
            if pd.Timestamp(cached_date) == pd.Timestamp(date):
                return self._cache[cached_date].get('rank_pcts', {})

        return {}

    def get_snapshot(self, date: datetime) -> Optional[Dict]:
        """
        【新增】获取指定日期的完整快照（包含stocks和rank_pcts）。

        Args:
            date: 交易日

        Returns:
            Dict 或 None: {'stocks': List[str], 'rank_pcts': Dict[str, float]}
        """
        # 直接查找
        if date in self._cache:
            return self._cache[date]

        # 尝试从磁盘加载
        disk_val = self._load_snapshot_from_disk(date)
        if disk_val is not None:
            self._cache[date] = disk_val
            return disk_val

        # pd.Timestamp 匹配
        for cached_date in list(self._cache.keys()):
            if pd.Timestamp(cached_date) == pd.Timestamp(date):
                return self._cache[cached_date]

        return None

    def get_or_fallback(self, date: datetime,
                        fallback_func: Callable[[datetime], List[str]]) -> List[str]:
        """
        获取指定日期的选股结果，如果不存在则调用 fallback 函数。

        Args:
            date: 交易日
            fallback_func: 回退函数，签名为 f(date) -> List[str]

        Returns:
            List[str]: 选股结果
        """
        cached = self.get(date)
        if cached is not None:
            return cached

        logger.debug(f"快照不存在 {date.date()}，回退到实时选股")
        try:
            return fallback_func(date)
        except Exception as e:
            logger.warning(f"实时选股也失败 {date.date()}: {e}")
            return []

    def has_snapshot(self, date: datetime) -> bool:
        """检查指定日期是否有快照。"""
        return self.get(date) is not None

    def is_generated(self) -> bool:
        """检查快照是否已生成。"""
        return self._generated

    def snapshot_count(self) -> int:
        """返回已缓存的快照数量。"""
        return len(self._cache)

    def all_dates(self) -> List[datetime]:
        """返回所有有快照的日期。"""
        return sorted(self._cache.keys())

    def clear(self):
        """清空缓存（不删除磁盘文件）。"""
        self._cache = {}
        self._generated = False

    # ==================== 内部方法 ====================

    def _snapshot_filename(self, date: datetime) -> str:
        """生成指定日期的快照文件名。"""
        date_str = date.strftime('%Y-%m-%d') if isinstance(date, datetime) else str(date)
        return os.path.join(self.snapshot_dir, f"{date_str}.json")

    def _save_snapshot_to_disk(self, date: datetime, snapshot_data: Dict):
        """
        将单日期快照保存为 JSON 文件。

        Args:
            date: 交易日
            snapshot_data: Dict，包含 'stocks' 和 'rank_pcts'
        """
        filepath = self._snapshot_filename(date)
        try:
            data = {
                'date': date.strftime('%Y-%m-%d') if isinstance(date, datetime) else str(date),
                'count': len(snapshot_data.get('stocks', [])),
                'stocks': snapshot_data.get('stocks', []),
                'rank_pcts': snapshot_data.get('rank_pcts', {}),
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"快照保存失败 {filepath}: {e}")

    def _load_snapshot_from_disk(self, date: datetime) -> Optional[Dict]:
        """
        从磁盘加载单日期快照。

        【向后兼容】支持旧格式快照（只有stocks字段）的加载。

        Args:
            date: 交易日

        Returns:
            Dict 或 None: {'stocks': List[str], 'rank_pcts': Dict[str, float]}
        """
        filepath = self._snapshot_filename(date)
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 【改造】新格式包含 stocks 和 rank_pcts
            # 向后兼容：旧格式只有 stocks 字段
            stocks = data.get('stocks', [])
            rank_pcts = data.get('rank_pcts', {})

            # 如果 rank_pcts 不存在（旧格式快照），返回空dict
            if rank_pcts is None:
                rank_pcts = {}

            return {
                'stocks': stocks if stocks else [],
                'rank_pcts': rank_pcts
            }
        except Exception:
            return None

    def _scan_existing_snapshots(self, calendar: List[datetime]) -> Dict[datetime, Dict]:
        """
        扫描目录中已存在的快照文件，加载与 calendar 中日期匹配的快照。

        【修正9】如果已有快照的 rank_pcts 为空（旧格式快照），
        则不纳入 existing，强制重新生成，确保 rank_pcts 正确填充。

        Returns:
            Dict[datetime, Dict]: 已成功加载的已有快照
            每个Dict格式: {'stocks': List[str], 'rank_pcts': Dict[str, float]}
        """
        existing = {}
        skipped_empty_rank = 0
        for date in calendar:
            snapshot = self._load_snapshot_from_disk(date)
            if snapshot is not None:
                # 【修正9】如果 rank_pcts 为空，跳过该快照，强制重新生成
                if not snapshot.get('rank_pcts'):
                    skipped_empty_rank += 1
                    continue
                self._cache[date] = snapshot
                existing[date] = snapshot
        if skipped_empty_rank > 0:
            logger.info(f"【快照】跳过 {skipped_empty_rank} 个旧快照（rank_pcts为空，将重新生成）")
        return existing
