# -*- coding: utf-8 -*-
"""
成交量分布分析模块（Volume Profile Analysis）
基于《威科夫2.0：市场结构、成交量分布与订单流》

核心概念：
- POC (Point of Control): 控制点，成交量最大的价格水平
- HVN (High Volume Node): 高量节点，高成交量价格区域
- LVN (Low Volume Node): 低量节点，低成交量价格区域
- VA (Value Area): 价值区域，通常包含70%成交量

Author: BULLION GPSEF Framework
Date: 2026-05-22
Version: 2.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class VolumeProfileResult:
    """成交量分布结果"""
    poc: float  # 控制点价格
    hvn_zones: List[Tuple[float, float, float]]  # 高量区: (最低价, 最高价, 成交量)
    lvn_zones: List[Tuple[float, float, float]]  # 低量区: (最低价, 最高价, 成交量)
    value_area_high: float  # 价值区域上限
    value_area_low: float  # 价值区域下限
    total_volume: float  # 总成交量
    volume_in_value_area: float  # 价值区域内成交量
    price_bins: np.ndarray  # 价格区间
    volume_by_bin: np.ndarray  # 各价格区间成交量
    time_frame: str  # 时间周期


class VolumeProfileAnalyzer:
    """成交量分布分析器"""

    def __init__(self,
                 value_area_pct: float = 0.70,
                 bin_size: Optional[float] = None,
                 min_bins: int = 20,
                 max_bins: int = 100):
        """
        初始化成交量分布分析器

        Args:
            value_area_pct: 价值区域包含的成交量比例，默认70%
            bin_size: 价格区间大小，None则自动计算
            min_bins: 最小价格区间数量
            max_bins: 最大价格区间数量
        """
        self.value_area_pct = value_area_pct
        self.bin_size = bin_size
        self.min_bins = min_bins
        self.max_bins = max_bins

    def analyze(self,
                high: np.ndarray,
                low: np.ndarray,
                volume: np.ndarray,
                time_frame: str = "daily") -> VolumeProfileResult:
        """
        分析成交量分布

        Args:
            high: 最高价数组
            low: 最低价数组
            volume: 成交量数组
            time_frame: 时间周期标识

        Returns:
            VolumeProfileResult: 成交量分布分析结果
        """
        if len(high) == 0 or len(high) != len(low) or len(high) != len(volume):
            raise ValueError("输入数组长度不一致或为空")

        # 计算价格范围
        price_min = np.min(low)
        price_max = np.max(high)
        price_range = price_max - price_min

        # 确定价格区间大小
        if self.bin_size is None:
            # 自动计算合理的bin大小
            n_bins = min(self.max_bins, max(self.min_bins, int(np.sqrt(len(high)) * 2)))
            bin_size = price_range / n_bins
        else:
            bin_size = self.bin_size
            n_bins = int(price_range / bin_size) + 1

        # 创建价格区间
        price_bins = np.linspace(price_min, price_max, n_bins)

        # 计算每个价格区间的成交量
        volume_by_bin = self._calculate_volume_by_bin(high, low, volume, price_bins)

        # 找到POC
        poc_idx = np.argmax(volume_by_bin)
        poc = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2 if poc_idx < len(price_bins) - 1 else price_bins[
            poc_idx]

        # 计算价值区域
        vah, val = self._calculate_value_area(price_bins, volume_by_bin, poc_idx)

        # 识别HVN和LVN区域
        hvn_zones = self._identify_hvn_zones(price_bins, volume_by_bin)
        lvn_zones = self._identify_lvn_zones(price_bins, volume_by_bin)

        total_volume = np.sum(volume_by_bin)
        va_volume = np.sum(volume_by_bin[(price_bins[:-1] >= val) & (price_bins[1:] <= vah)])

        return VolumeProfileResult(
            poc=poc,
            hvn_zones=hvn_zones,
            lvn_zones=lvn_zones,
            value_area_high=vah,
            value_area_low=val,
            total_volume=total_volume,
            volume_in_value_area=va_volume,
            price_bins=price_bins,
            volume_by_bin=volume_by_bin,
            time_frame=time_frame
        )

    def _calculate_volume_by_bin(self,
                                 high: np.ndarray,
                                 low: np.ndarray,
                                 volume: np.ndarray,
                                 price_bins: np.ndarray) -> np.ndarray:
        """
        计算每个价格区间的成交量

        使用OHLC蜡烛覆盖的价格区间分布成交量
        """
        n_bins = len(price_bins) - 1
        volume_by_bin = np.zeros(n_bins)

        for i in range(len(high)):
            candle_high = high[i]
            candle_low = low[i]
            candle_volume = volume[i]

            # 找到蜡烛覆盖的bin
            start_bin = np.searchsorted(price_bins, candle_low, side='right') - 1
            end_bin = np.searchsorted(price_bins, candle_high, side='right') - 1

            start_bin = max(0, start_bin)
            end_bin = min(n_bins - 1, end_bin)

            if start_bin > end_bin:
                continue

            # 在覆盖的bin中均匀分配成交量
            n_covered = end_bin - start_bin + 1
            if n_covered > 0:
                volume_per_bin = candle_volume / n_covered
                volume_by_bin[start_bin:end_bin + 1] += volume_per_bin

        return volume_by_bin

    def _calculate_value_area(self,
                              price_bins: np.ndarray,
                              volume_by_bin: np.ndarray,
                              poc_idx: int) -> Tuple[float, float]:
        """
        计算价值区域(Value Area)

        从POC开始向上下扩展，直到包含value_area_pct的成交量
        """
        total_volume = np.sum(volume_by_bin)
        target_volume = total_volume * self.value_area_pct

        n_bins = len(volume_by_bin)
        va_high_idx = poc_idx
        va_low_idx = poc_idx
        current_volume = volume_by_bin[poc_idx]

        # 向上下扩展，直到达到目标成交量
        while current_volume < target_volume and (va_low_idx > 0 or va_high_idx < n_bins - 1):
            # 计算上下扩展一个bin的新增成交量
            up_volume = volume_by_bin[va_high_idx + 1] if va_high_idx < n_bins - 1 else 0
            down_volume = volume_by_bin[va_low_idx - 1] if va_low_idx > 0 else 0

            # 选择成交量较大的一侧扩展
            if up_volume >= down_volume and va_high_idx < n_bins - 1:
                va_high_idx += 1
                current_volume += up_volume
            elif va_low_idx > 0:
                va_low_idx -= 1
                current_volume += down_volume
            else:
                break

        vah = price_bins[va_high_idx + 1] if va_high_idx + 1 < len(price_bins) else price_bins[-1]
        val = price_bins[va_low_idx]

        return vah, val

    def _identify_hvn_zones(self,
                            price_bins: np.ndarray,
                            volume_by_bin: np.ndarray,
                            threshold_multiplier: float = 1.3) -> List[Tuple[float, float, float]]:
        """
        识别高量节点区域(HVN - High Volume Nodes)

        HVN表示成交密集区，通常是支撑或阻力位
        """
        avg_volume = np.mean(volume_by_bin)
        threshold = avg_volume * threshold_multiplier

        hvn_zones = []
        current_hvn_start = None
        current_hvn_volume = 0

        for i in range(len(volume_by_bin)):
            if volume_by_bin[i] >= threshold:
                if current_hvn_start is None:
                    current_hvn_start = i
                current_hvn_volume += volume_by_bin[i]
            else:
                if current_hvn_start is not None:
                    # 结束一个HVN区域
                    zone_low = price_bins[current_hvn_start]
                    zone_high = price_bins[i]
                    hvn_zones.append((zone_low, zone_high, current_hvn_volume))
                    current_hvn_start = None
                    current_hvn_volume = 0

        # 处理最后一个可能的HVN区域
        if current_hvn_start is not None:
            zone_low = price_bins[current_hvn_start]
            zone_high = price_bins[-1]
            hvn_zones.append((zone_low, zone_high, current_hvn_volume))

        # 按成交量排序，返回前3个最大的HVN
        hvn_zones.sort(key=lambda x: x[2], reverse=True)
        return hvn_zones[:3]

    def _identify_lvn_zones(self,
                            price_bins: np.ndarray,
                            volume_by_bin: np.ndarray,
                            threshold_multiplier: float = 0.7) -> List[Tuple[float, float, float]]:
        """
        识别低量节点区域(LVN - Low Volume Nodes)

        LVN表示成交稀疏区，通常是突破的关键位置
        """
        avg_volume = np.mean(volume_by_bin)
        threshold = avg_volume * threshold_multiplier

        lvn_zones = []
        current_lvn_start = None
        current_lvn_volume = 0

        for i in range(len(volume_by_bin)):
            if volume_by_bin[i] <= threshold:
                if current_lvn_start is None:
                    current_lvn_start = i
                current_lvn_volume += volume_by_bin[i]
            else:
                if current_lvn_start is not None:
                    zone_low = price_bins[current_lvn_start]
                    zone_high = price_bins[i]
                    lvn_zones.append((zone_low, zone_high, current_lvn_volume))
                    current_lvn_start = None
                    current_lvn_volume = 0

        if current_lvn_start is not None:
            zone_low = price_bins[current_lvn_start]
            zone_high = price_bins[-1]
            lvn_zones.append((zone_low, zone_high, current_lvn_volume))

        # 按成交量从小到大排序，返回前3个最稀疏的LVN
        lvn_zones.sort(key=lambda x: x[2])
        return lvn_zones[:3]

    def get_price_position_score(self,
                                 current_price: float,
                                 vp_result: VolumeProfileResult) -> float:
        """
        计算当前价格在成交量分布中的位置得分

        Returns:
            0.0~1.0，越高表示价格位置越有利（在价值区域下方或LVN附近突破）
        """
        poc = vp_result.poc
        vah = vp_result.value_area_high
        val = vp_result.value_area_low

        # 价格相对于POC的位置
        price_vs_poc = (current_price - poc) / (poc + 1e-8)

        # 检查是否在价值区域内
        in_value_area = val <= current_price <= vah

        # 检查是否接近LVN区域（潜在突破位）
        near_lvn = False
        for lvn_low, lvn_high, _ in vp_result.lvn_zones:
            if abs(current_price - (lvn_low + lvn_high) / 2) < (lvn_high - lvn_low) * 0.5:
                near_lvn = True
                break

        # 综合评分
        score = 0.5

        if current_price > poc:
            score += 0.2  # 价格在POC之上
        if in_value_area:
            score += 0.1  # 在价值区域内
        if near_lvn and current_price > vah:
            score += 0.2  # 从LVN向上突破

        return min(1.0, score)

    def detect_breakout_through_lvn(self,
                                    current_price: float,
                                    prev_price: float,
                                    vp_result: VolumeProfileResult) -> float:
        """
        检测通过LVN的突破

        Returns:
            突破强度得分 0.0~1.0
        """
        breakout_score = 0.0

        for lvn_low, lvn_high, _ in vp_result.lvn_zones:
            lvn_mid = (lvn_low + lvn_high) / 2

            # 从下向上突破LVN
            if prev_price <= lvn_mid <= current_price:
                # 成交量确认
                breakout_score += 0.3

            # 价格在LVN之上且远离LVN
            if current_price > lvn_high:
                breakout_score += 0.2

        return min(1.0, breakout_score)
