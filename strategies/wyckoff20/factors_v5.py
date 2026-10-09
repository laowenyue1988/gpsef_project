# -*- coding: utf-8 -*-
"""
Wyckoff 2.0 V5 因子引擎 - 40因子体系

【v5.1 升级说明】
基于V5.0的36因子体系，新增4个因子并升级2个V4.0因子，共40个因子：
- 新增4个因子：accumulation_score(S-01), vwap_deviation(V-06), vol_pulse(V-07), hvn_support(V-08)
- 升级effort_result为O-04增强版（方向感知+持续性追踪）
- 升级micro_order_flow为O-06增强版（A股日内U型加权）
- 保留V4.0全部20个因子方法
- 保留V5.0全部16个DeepSeek因子方法

每个因子输出3类特征：raw_value, score(0~1), confidence(0~1)

Author: BULLION GPSEF Framework
Date: 2026-05-26
Version: 5.1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FactorResult:
    """单个因子计算结果"""
    factor_name: str
    score: float  # 0.0~1.0 标准化得分
    raw_value: float  # 原始值
    confidence: float  # 置信度 0.0~1.0
    sub_values: Dict  # 子维度值


class WyckoffFactorEngineV5:
    """威科夫2.0 V5因子引擎 - 40因子体系"""

    # === V4.0 原有20个因子 ===
    V4_FACTORS = [
        'supply_demand_momentum', 'price_volume_divergence', 'no_supply',
        'volume_profile', 'micro_order_flow', 'market_structure',
        'accumulation_power', 'breakout_validity', 'vwap_anchor', 'delta_momentum',
        'effort_result', 'closing_position', 'volume_concentration',
        'spread_ratio', 'up_volume_ratio', 'demand_supply_ratio',
        'turnover_momentum', 'price_position', 'volume_trend', 'smart_money_flow',
    ]

    # === V5.0新增16个DeepSeek因子 ===
    NEW_FACTORS_V5 = [
        # Market Structure (S-02~S-06)
        'spring_quality',  # S-02
        'jump_creek',  # S-03
        'stage_continuous',  # S-04
        'ut_utad',  # S-05
        'sos_strength',  # S-06
        # Volume Profile (V-01~V-05)
        'vpoc_momentum',  # V-01
        'va_convergence',  # V-02
        'vol_skewness',  # V-03
        'lvn_breakout',  # V-04
        'vol_entropy',  # V-05
        # Order Flow (O-01~O-03, O-05, O-07~O-08)
        'climax',  # O-01
        'no_demand',  # O-02
        'stopping_vol',  # O-03
        'weis_wave',  # O-05
        'big_flow',  # O-07
        'imbalance',  # O-08
    ]

    # === V5.1新增4个因子 ===
    NEW_FACTORS_V51 = [
        'accumulation_score',  # S-01 吸筹阶段评分
        'vwap_deviation',  # V-06 VWAP乖离率
        'vol_pulse',  # V-07 成交量脉冲
        'hvn_support',  # V-08 HVN支撑强度
    ]

    # === 全部40个因子 ===
    ALL_FACTORS = V4_FACTORS + NEW_FACTORS_V5 + NEW_FACTORS_V51

    DEFAULT_WEIGHTS = {
        # V4.0 factors
        'supply_demand_momentum': 0.07,
        'price_volume_divergence': 0.04,
        'no_supply': 0.05,
        'volume_profile': 0.05,
        'micro_order_flow': 0.04,
        'market_structure': 0.07,
        'accumulation_power': 0.04,
        'breakout_validity': 0.03,
        'vwap_anchor': 0.03,
        'delta_momentum': 0.04,
        'effort_result': 0.04,
        'closing_position': 0.03,
        'volume_concentration': 0.03,
        'spread_ratio': 0.03,
        'up_volume_ratio': 0.04,
        'demand_supply_ratio': 0.04,
        'turnover_momentum': 0.03,
        'price_position': 0.04,
        'volume_trend': 0.04,
        'smart_money_flow': 0.04,
        # V5.0 New DeepSeek factors
        'spring_quality': 0.04,
        'jump_creek': 0.04,
        'stage_continuous': 0.04,
        'ut_utad': 0.03,
        'sos_strength': 0.04,
        'vpoc_momentum': 0.04,
        'va_convergence': 0.03,
        'vol_skewness': 0.03,
        'lvn_breakout': 0.03,
        'vol_entropy': 0.03,
        'climax': 0.03,
        'no_demand': 0.03,
        'stopping_vol': 0.03,
        'weis_wave': 0.04,
        'big_flow': 0.04,
        'imbalance': 0.04,
        # V5.1 New 4 factors
        'accumulation_score': 0.04,
        'vwap_deviation': 0.03,
        'vol_pulse': 0.03,
        'hvn_support': 0.03,
    }

    def __init__(self, factor_weights: Optional[Dict[str, float]] = None):
        self.weights = factor_weights or self.DEFAULT_WEIGHTS.copy()
        self._vp_analyzer = None

    @property
    def vp_analyzer(self):
        if self._vp_analyzer is None:
            from .volume_profile import VolumeProfileAnalyzer
            self._vp_analyzer = VolumeProfileAnalyzer()
        return self._vp_analyzer

    # ================================================================
    #  主入口 - 计算全部40个因子
    # ================================================================

    def calculate_all_factors(self,
                              df_daily: pd.DataFrame,
                              df_5min: Optional[pd.DataFrame] = None,
                              current_date: Optional[str] = None) -> Dict[str, FactorResult]:
        """计算全部40个威科夫因子"""
        if df_daily is None or len(df_daily) < 20:
            raise ValueError("日线数据不能为空或长度不足20天")

        results = {}

        # V4.0 原有20个因子
        results['supply_demand_momentum'] = self._factor_supply_demand(df_daily)
        results['price_volume_divergence'] = self._factor_pv_divergence(df_daily)
        results['no_supply'] = self._factor_no_supply(df_daily)
        results['volume_profile'] = self._factor_volume_profile(df_daily)
        results['micro_order_flow'] = self._factor_micro_order_flow(df_daily, df_5min)
        results['market_structure'] = self._factor_market_structure(df_daily)
        results['accumulation_power'] = self._factor_accumulation_power(df_daily)
        results['breakout_validity'] = self._factor_breakout(df_daily)
        results['vwap_anchor'] = self._factor_vwap(df_daily, df_5min)
        results['delta_momentum'] = self._factor_delta(df_daily)
        results['effort_result'] = self._factor_effort_result(df_daily)
        results['closing_position'] = self._factor_closing_position(df_daily)
        results['volume_concentration'] = self._factor_volume_concentration(df_daily)
        results['spread_ratio'] = self._factor_spread_ratio(df_daily)
        results['up_volume_ratio'] = self._factor_up_volume_ratio(df_daily)
        results['demand_supply_ratio'] = self._factor_demand_supply_ratio(df_daily)
        results['turnover_momentum'] = self._factor_turnover_momentum(df_daily)
        results['price_position'] = self._factor_price_position(df_daily)
        results['volume_trend'] = self._factor_volume_trend(df_daily)
        results['smart_money_flow'] = self._factor_smart_money_flow(df_daily)

        # V5.0新增16个DeepSeek因子
        results['spring_quality'] = self._factor_spring_quality(df_daily)
        results['jump_creek'] = self._factor_jump_creek(df_daily)
        results['stage_continuous'] = self._factor_stage_continuous(df_daily)
        results['ut_utad'] = self._factor_ut_utad(df_daily)
        results['sos_strength'] = self._factor_sos_strength(df_daily)
        results['vpoc_momentum'] = self._factor_vpoc_momentum(df_daily)
        results['va_convergence'] = self._factor_va_convergence(df_daily)
        results['vol_skewness'] = self._factor_vol_skewness(df_daily)
        results['lvn_breakout'] = self._factor_lvn_breakout(df_daily)
        results['vol_entropy'] = self._factor_vol_entropy(df_daily)
        results['climax'] = self._factor_climax(df_daily)
        results['no_demand'] = self._factor_no_demand(df_daily)
        results['stopping_vol'] = self._factor_stopping_vol(df_daily)
        results['weis_wave'] = self._factor_weis_wave(df_daily)
        results['big_flow'] = self._factor_big_flow(df_daily)
        results['imbalance'] = self._factor_imbalance(df_daily)

        # V5.1新增4个因子
        results['accumulation_score'] = self._factor_accumulation_score(df_daily)
        results['vwap_deviation'] = self._factor_vwap_deviation(df_daily)
        results['vol_pulse'] = self._factor_vol_pulse(df_daily)
        results['hvn_support'] = self._factor_hvn_support(df_daily)

        return results

    def compute_composite_score(self, factor_results: Dict[str, FactorResult]) -> dict:
        """加权线性综合得分"""
        weighted_sum = 0.0
        weight_sum = 0.0
        for name, result in factor_results.items():
            w = self.weights.get(name, 0.0)
            weighted_sum += result.score * w * result.confidence
            weight_sum += w * result.confidence
        total = weighted_sum / weight_sum if weight_sum > 0 else 0.5

        if total >= 0.75:
            rec = "强烈看好"
        elif total >= 0.60:
            rec = "看好"
        elif total >= 0.45:
            rec = "中性"
        elif total >= 0.30:
            rec = "谨慎"
        else:
            rec = "看空"

        return {
            'total_score': total,
            'recommendation': rec,
            'factor_scores': {k: v.score for k, v in factor_results.items()},
            'factor_raw_values': {k: v.raw_value for k, v in factor_results.items()},
            'factor_confidences': {k: v.confidence for k, v in factor_results.items()},
        }

    # ================================================================
    #  V4.0 原有10个因子（保留原逻辑，完全不变）
    # ================================================================

    def _factor_supply_demand(self, df: pd.DataFrame) -> FactorResult:
        if len(df) < 60:
            return FactorResult('supply_demand_momentum', 0.5, 0.0, 0.2, {})
        df = df.tail(120).copy()
        price_chg = df['close'].values - df['open'].values
        prev_close = np.roll(df['close'].values, 1)
        prev_close[0] = df['open'].values[0]
        tr = np.maximum(
            df['high'].values - df['low'].values,
            np.maximum(
                np.abs(df['high'].values - prev_close),
                np.abs(df['low'].values - prev_close)
            )
        )
        vol = df['volume'].values
        ds = np.where(tr > 1e-8, price_chg * vol / tr, 0.0)

        def ew_mean(arr, window, decay):
            w = np.array([decay ** i for i in range(window)])[::-1]
            w = w / w.sum()
            return np.dot(arr[-window:], w)

        short_mom = ew_mean(ds, 20, 0.90)
        mid_mom = ew_mean(ds, 60, 0.95)
        acceleration = short_mom - mid_mom
        score = self._normalize_triple(short_mom, mid_mom, acceleration,
                                       pos_weight=0.45, trend_weight=0.35, accel_weight=0.20)
        confidence = min(1.0, len(df) / 60.0)
        return FactorResult('supply_demand_momentum', score, short_mom, confidence,
                            {'short': short_mom, 'mid': mid_mom, 'accel': acceleration})

    def _factor_pv_divergence(self, df: pd.DataFrame) -> FactorResult:
        if len(df) < 30:
            return FactorResult('price_volume_divergence', 0.5, 0.0, 0.2, {})
        df = df.tail(60).copy()
        window = 20
        spreads = df['high'].values - df['low'].values
        vols = df['volume'].values
        avg_vol = pd.Series(vols).rolling(window, min_periods=1).mean().values
        avg_spread = pd.Series(spreads).rolling(window, min_periods=1).mean().values
        rel_vol = np.where(avg_vol > 0, vols / avg_vol, 1.0)
        rel_spread = np.where(avg_spread > 0, spreads / avg_spread, 1.0)
        direction = np.where(df['close'].values >= df['open'].values, 1.0, -1.0)
        divergence = (rel_vol - rel_spread) * direction
        div_short = np.mean(divergence[-10:]) if len(divergence) >= 10 else 0.0
        div_mid = np.mean(divergence[-window:])
        div_accel = div_short - div_mid
        score = 0.5
        if div_mid < 0 and div_accel > 0:
            score += 0.3
        elif div_mid > 0 and direction[-1] > 0:
            score += 0.15
        if div_short < 0 and div_accel > 0:
            score += 0.2
        score = np.clip(score, 0, 1)
        confidence = min(1.0, len(df) / 60.0)
        return FactorResult('price_volume_divergence', score, div_mid, confidence,
                            {'div_short': div_short, 'div_mid': div_mid, 'accel': div_accel})

    def _factor_no_supply(self, df: pd.DataFrame) -> FactorResult:
        if len(df) < 30:
            return FactorResult('no_supply', 0.5, 0.0, 0.2, {})
        df = df.tail(40).copy()
        window = 20
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        vols = df['volume'].values
        spreads = highs - lows
        no_supply_count = 0
        last_ns_idx = -1
        for i in range(window, len(df)):
            cond_a = closes[i] < closes[i - 1]
            avg_spread = np.mean(spreads[i - window:i])
            cond_b = spreads[i] < 0.5 * avg_spread if avg_spread > 0 else False
            avg_vol = np.mean(vols[i - window:i])
            cond_c = vols[i] < 0.5 * avg_vol if avg_vol > 0 else False
            if cond_a and cond_b and cond_c:
                no_supply_count += 1
                last_ns_idx = i
        ns_ratio = no_supply_count / window
        response_ret = 0.0
        if last_ns_idx >= 0 and last_ns_idx + 5 < len(df):
            response_ret = closes[last_ns_idx + 5] / closes[last_ns_idx] - 1.0
        score = 0.5
        if ns_ratio > 0.10:
            score += 0.2
        if ns_ratio > 0.20:
            score += 0.15
        if response_ret > 0:
            score += 0.15
        score = np.clip(score, 0, 1)
        confidence = min(1.0, len(df) / 40.0)
        return FactorResult('no_supply', score, ns_ratio, confidence,
                            {'ns_ratio': ns_ratio, 'response_ret': response_ret})

    def _factor_volume_profile(self, df: pd.DataFrame) -> FactorResult:
        if len(df) < 30:
            return FactorResult('volume_profile', 0.5, 0.0, 0.2, {})
        df_vp = df.tail(60).copy()
        try:
            vp = self.vp_analyzer.analyze(
                high=df_vp['high'].values,
                low=df_vp['low'].values,
                volume=df_vp['volume'].values
            )
            current = df_vp['close'].iloc[-1]
            price_range = df_vp['high'].max() - df_vp['low'].min()
            poc_dist = (current - vp.poc) / price_range if price_range > 0 else 0.0
            poc_idx = np.argmax(vp.volume_by_bin)
            upper_vol = np.sum(vp.volume_by_bin[poc_idx:])
            lower_vol = np.sum(vp.volume_by_bin[:poc_idx + 1])
            vol_skew = (upper_vol - lower_vol) / (upper_vol + lower_vol + 1e-8)
            in_va = vp.value_area_low <= current <= vp.value_area_high
            above_va = current > vp.value_area_high
            score = 0.5
            if poc_dist > 0:
                score += 0.15
            if vol_skew > 0.1:
                score += 0.15
            if above_va:
                score += 0.10
            elif in_va:
                score += 0.10
            score = np.clip(score, 0, 1)
            confidence = min(1.0, len(df_vp) / 60.0)
            return FactorResult('volume_profile', score, poc_dist, confidence,
                                {'poc_dist': poc_dist, 'vol_skew': vol_skew, 'in_va': in_va, 'above_va': above_va})
        except Exception as e:
            return FactorResult('volume_profile', 0.5, 0.0, 0.2, {'error': str(e)})

    def _factor_micro_order_flow(self, df_daily: pd.DataFrame,
                                 df_5min: Optional[pd.DataFrame]) -> FactorResult:
        """
        O-06 增强版：A股日内U型加权订单流因子
        9:30-10:00(weight=1.5) 开盘30分钟主力行动密集
        10:00-11:25(weight=1.0) 正常交易
        11:25-11:30(weight=0.8) 午盘清淡
        13:00-14:00(weight=1.0) 正常交易
        14:00-14:30(weight=1.0) 正常交易
        14:30-15:00(weight=1.2) 收盘博弈结算期
        """
        if df_5min is not None and len(df_5min) > 20:
            return self._compute_5min_order_flow_weighted(df_5min)
        else:
            return self._estimate_order_flow_from_daily(df_daily)

    def _get_time_weight(self, bar_time: str) -> float:
        """
        根据5分钟K线时间戳返回A股日内时段权重
        bar_time格式: "HH:MM" 或 "HH:MM:SS"
        """
        try:
            if isinstance(bar_time, str):
                time_str = bar_time[:5]  # 取HH:MM
            else:
                time_str = str(bar_time)[:5]

            hour = int(time_str.split(':')[0])
            minute = int(time_str.split(':')[1])

            # 9:30-10:00 开盘30分钟
            if hour == 9 and minute >= 30:
                return 1.5
            elif hour == 10:
                return 1.0
            # 11:25-11:30 午盘清淡
            elif hour == 11 and minute >= 25:
                return 0.8
            # 14:30-15:00 收盘博弈
            elif hour == 14 and minute >= 30:
                return 1.2
            elif hour == 14 or hour == 13:
                return 1.0
            else:
                return 1.0
        except Exception:
            return 1.0

    def _compute_5min_order_flow_weighted(self, df_5min: pd.DataFrame) -> FactorResult:
        """增强版5分钟订单流计算，支持A股日内时段加权"""
        df_5min = df_5min.tail(2400).copy()
        if not isinstance(df_5min.index, pd.DatetimeIndex):
            df_5min.index = pd.to_datetime(df_5min.index)
        daily_groups = df_5min.groupby(df_5min.index.date)
        daily_of_values = []
        daily_returns = []

        for date, group in daily_groups:
            if len(group) < 5:
                continue
            bars = group.sort_index()
            day_of = 0.0
            vol_window = min(20, len(bars) - 1)

            for k in range(1, len(bars)):
                bar = bars.iloc[k]
                spread = bar['high'] - bar['low']
                if spread > 1e-8:
                    direction = (bar['close'] - bar['open']) / spread
                else:
                    direction = 0.0
                recent_vol = bars['volume'].iloc[max(0, k - vol_window):k].mean()
                intensity = bar['volume'] / recent_vol if recent_vol > 0 else 1.0

                # 获取时间权重
                if hasattr(bars.index[k], 'strftime'):
                    time_str = bars.index[k].strftime('%H:%M')
                else:
                    time_str = str(bars.index[k])[11:16] if len(str(bars.index[k])) > 16 else "10:00"
                time_weight = self._get_time_weight(time_str)

                day_of += direction * intensity * time_weight

            rets = bars['close'].pct_change().dropna()
            day_vol = rets.std() if len(rets) > 1 else 0.01
            daily_of = day_of / day_vol if day_vol > 0 else day_of
            daily_of_values.append(daily_of)
            daily_returns.append(bars['close'].iloc[-1] / bars['close'].iloc[0] - 1.0)

        if len(daily_of_values) < 5:
            return self._estimate_order_flow_from_daily(None)
        recent = daily_of_values[-20:] if len(daily_of_values) >= 20 else daily_of_values
        micro_of = np.mean(recent)
        of_std = np.std(recent) if len(recent) > 1 else 1.0
        of_trend = micro_of / of_std if of_std > 0 else micro_of
        reversal_count = sum(1 for of, ret in zip(recent, daily_returns[-len(recent):]) if of > 0 and ret < 0)
        reversal_ratio = reversal_count / len(recent)
        score = 0.5
        if micro_of > 0:
            score += 0.25
        if of_trend > 1.0:
            score += 0.15
        if reversal_ratio < 0.3:
            score += 0.10
        score = np.clip(score, 0, 1)
        confidence = min(1.0, len(recent) / 20.0)
        return FactorResult('micro_order_flow', score, micro_of, confidence,
                            {'micro_of': micro_of, 'of_trend': of_trend, 'reversal_ratio': reversal_ratio})

    def _estimate_order_flow_from_daily(self, df_daily: Optional[pd.DataFrame]) -> FactorResult:
        if df_daily is None or len(df_daily) < 20:
            return FactorResult('micro_order_flow', 0.5, 0.0, 0.2, {'estimated': True})
        df = df_daily.tail(30).copy()
        daily_of = []
        for i in range(len(df)):
            spread = df['high'].iloc[i] - df['low'].iloc[i]
            if spread > 1e-8:
                direction = (df['close'].iloc[i] - df['open'].iloc[i]) / spread
            else:
                direction = 0.0
            avg_vol = df['volume'].rolling(20, min_periods=1).mean().iloc[i]
            intensity = df['volume'].iloc[i] / avg_vol if avg_vol > 0 else 1.0
            daily_of.append(direction * intensity)
        micro_of = np.mean(daily_of[-20:])
        score = 0.5 + 0.3 * np.tanh(micro_of / 2.0)
        score = np.clip(score, 0, 1)
        return FactorResult('micro_order_flow', score, micro_of, 0.4,
                            {'micro_of': micro_of, 'estimated': True})

    def _factor_market_structure(self, df: pd.DataFrame) -> FactorResult:
        if len(df) < 60:
            return FactorResult('market_structure', 0.3, 0.0, 0.2, {})
        df = df.tail(120).copy()
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values
        sc = self._detect_selling_climax(highs, lows, closes, volumes)
        ar = self._detect_auto_rally(highs, lows, closes, sc)
        st = self._detect_secondary_test(highs, lows, closes, volumes, sc)
        spring = self._detect_spring(highs, lows, closes, volumes, sc)
        sos = self._detect_sign_of_strength(highs, lows, closes, volumes, ar)
        lps = self._detect_last_point_support(highs, lows, closes, volumes, sos)
        stage, completeness = self._determine_stage(sc, ar, st, spring, sos, lps)
        stage_score_map = {'A': 0.2, 'B': 0.35, 'C': 0.55, 'D': 0.75, 'E': 0.95}
        base_score = stage_score_map.get(stage, 0.3)
        score = base_score * 0.6 + completeness * 0.4
        score = np.clip(score, 0, 1)
        events = {'SC': sc['detected'], 'AR': ar['detected'], 'ST': st['detected'],
                  'Spring': spring['detected'], 'SOS': sos['detected'], 'LPS': lps['detected']}
        confidence = min(1.0, len(df) / 120.0)
        return FactorResult('market_structure', score, stage_score_map.get(stage, 0.3), confidence,
                            {'stage': stage, 'completeness': completeness, 'events': events})

    def _find_swing_points(self, prices, lookback=5, find_high=True):
        points = []
        for i in range(lookback, len(prices) - lookback):
            window = prices[i - lookback:i + lookback + 1]
            if find_high:
                if prices[i] == np.max(window):
                    points.append(i)
            else:
                if prices[i] == np.min(window):
                    points.append(i)
        return points

    def _detect_selling_climax(self, highs, lows, closes, volumes):
        if len(closes) < 25:
            return {'detected': False, 'index': -1}
        for i in range(20, len(closes)):
            ret_5d = closes[i] / closes[max(0, i - 5)] - 1.0
            avg_vol = np.mean(volumes[max(0, i - 20):i])
            if ret_5d < -0.10 and volumes[i] > avg_vol * 1.5:
                return {'detected': True, 'index': i}
        return {'detected': False, 'index': -1}

    def _detect_auto_rally(self, highs, lows, closes, sc):
        if not sc['detected']:
            return {'detected': False}
        for i in range(sc['index'] + 1, min(sc['index'] + 30, len(closes))):
            ret = closes[i] / closes[sc['index']] - 1.0
            if 0.10 <= ret <= 0.30:
                return {'detected': True, 'index': i}
        return {'detected': False}

    def _detect_secondary_test(self, highs, lows, closes, volumes, sc):
        if not sc['detected']:
            return {'detected': False}
        for i in range(sc['index'] + 5, min(sc['index'] + 60, len(closes))):
            if closes[i] <= closes[sc['index']] * 1.05:
                avg_vol = np.mean(volumes[max(0, i - 20):i])
                if volumes[i] < avg_vol * 0.8:
                    return {'detected': True, 'index': i}
        return {'detected': False}

    def _detect_spring(self, highs, lows, closes, volumes, sc):
        if not sc['detected']:
            return {'detected': False}
        for i in range(sc['index'] + 3, min(sc['index'] + 60, len(closes))):
            if lows[i] < lows[max(0, i - 5):i].min() * 1.01:
                if closes[i] > lows[i] * 1.02:
                    return {'detected': True, 'index': i}
        return {'detected': False}

    def _detect_sign_of_strength(self, highs, lows, closes, volumes, ar):
        if not ar['detected']:
            return {'detected': False}
        for i in range(ar['index'] + 1, min(ar['index'] + 20, len(closes))):
            if closes[i] > closes[max(0, i - 10):i].max() * 1.02:
                avg_vol = np.mean(volumes[max(0, i - 20):i])
                if volumes[i] > avg_vol * 1.3:
                    return {'detected': True, 'index': i}
        return {'detected': False}

    def _detect_last_point_support(self, highs, lows, closes, volumes, sos):
        if not sos['detected']:
            return {'detected': False}
        for i in range(sos['index'] + 2, min(sos['index'] + 30, len(closes))):
            if lows[i] > lows[max(0, i - 5):i].min():
                return {'detected': True, 'index': i}
        return {'detected': False}

    def _determine_stage(self, sc, ar, st, spring, sos, lps):
        events = [sc, ar, st, spring, sos, lps]
        count = sum(1 for e in events if e['detected'])
        completeness = count / 6.0
        if sos['detected'] and lps['detected']:
            stage = 'D'
        elif sos['detected']:
            stage = 'C'
        elif spring['detected']:
            stage = 'B'
        elif ar['detected'] or st['detected']:
            stage = 'A'
        else:
            stage = 'A'
        return stage, completeness

    def _factor_accumulation_power(self, df: pd.DataFrame) -> FactorResult:
        if len(df) < 60:
            return FactorResult('accumulation_power', 0.3, 0.0, 0.2, {})
        df = df.tail(120).copy()
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values
        swing_highs = self._find_swing_points(highs, lookback=5, find_high=True)
        swing_lows = self._find_swing_points(lows, lookback=5, find_high=False)
        osc_range = df['high'].max() - df['low'].min()
        pre_vol = np.mean(volumes[max(0, len(volumes) - 40):-20]) if len(volumes) >= 40 else np.mean(volumes)
        cum_vol = np.sum(volumes[-20:])
        if len(swing_lows) >= 2:
            oscillation_count = len(swing_lows)
            range_ratio = osc_range / (df['close'].iloc[-1] + 1e-8)
            vol_ratio = cum_vol / pre_vol if pre_vol > 0 else 1.0
            acc_power = oscillation_count * range_ratio * vol_ratio
        else:
            oscillation_count = 0
            acc_power = 0.0
        score = 0.3
        if acc_power > 0:
            score = 1.0 - np.exp(-acc_power / 5.0)
            score = np.clip(score, 0.2, 0.95)
        confidence = min(1.0, len(df) / 120.0)
        return FactorResult('accumulation_power', score, acc_power, confidence,
                            {'oscillation_count': oscillation_count,
                             'range_ratio': osc_range / (df['close'].iloc[-1] + 1e-8), 'acc_power': acc_power})

    def _factor_breakout(self, df: pd.DataFrame) -> FactorResult:
        if len(df) < 25:
            return FactorResult('breakout_validity', 0.3, 0.0, 0.2, {})
        df = df.tail(25).copy()
        current_close = df['close'].iloc[-1]
        current_open = df['open'].iloc[-1]
        current_vol = df['volume'].iloc[-1]
        avg_vol = df['volume'].iloc[:-1].mean()
        prior_high = df['high'].iloc[:-5].max()
        breakout = current_close > prior_high * 1.01
        vol_confirm = current_vol > avg_vol * 1.5
        bar_size = (current_close - current_open) / (current_open + 1e-8)
        is_bullish = bar_size > 0.02
        retest_ok = True
        if len(df) >= 5:
            for i in range(-5, -1):
                if df['low'].iloc[i] < prior_high * 0.97:
                    retest_ok = False
                    break
        score = 0.0
        if breakout:
            score += 0.35
            if vol_confirm:
                score += 0.25
            if is_bullish:
                score += 0.20
            if retest_ok:
                score += 0.20
        else:
            dist = (prior_high - current_close) / prior_high
            if dist < 0.05:
                score += 0.3
            elif dist < 0.10:
                score += 0.15
        confidence = min(1.0, len(df) / 25.0)
        return FactorResult('breakout_validity', min(1.0, score), bar_size, confidence,
                            {'breakout': breakout, 'vol_confirm': vol_confirm, 'retest_ok': retest_ok})

    def _factor_vwap(self, df_daily: pd.DataFrame, df_5min: Optional[pd.DataFrame]) -> FactorResult:
        if len(df_daily) < 20:
            return FactorResult('vwap_anchor', 0.3, 0.0, 0.2, {})
        df = df_daily.tail(20).copy()
        vwap_5d = self._calc_vwap(df.tail(5))
        vwap_10d = self._calc_vwap(df.tail(10))
        vwap_20d = self._calc_vwap(df.tail(20))
        price = df['close'].iloc[-1]
        trend = (vwap_5d - vwap_20d) / vwap_20d if vwap_20d > 0 else 0
        score = 0.0
        if price > vwap_5d:
            score += 0.25
        if price > vwap_10d:
            score += 0.20
        if price > vwap_20d:
            score += 0.20
        if trend > 0:
            score += 0.20
        if trend > 0.02:
            score += 0.15
        score = np.clip(score, 0, 1)
        confidence = min(1.0, len(df) / 20.0)
        return FactorResult('vwap_anchor', score, vwap_5d, confidence,
                            {'vwap_5d': vwap_5d, 'vwap_10d': vwap_10d, 'vwap_20d': vwap_20d, 'trend': trend})

    def _calc_vwap(self, df):
        tp = (df['high'] + df['low'] + df['close']) / 3
        return np.sum(tp * df['volume']) / np.sum(df['volume'])

    def _factor_delta(self, df: pd.DataFrame) -> FactorResult:
        if len(df) < 20:
            return FactorResult('delta_momentum', 0.5, 0.0, 0.2, {})
        df = df.tail(30).copy()
        deltas = []
        for i in range(len(df)):
            body = df['close'].iloc[i] - df['open'].iloc[i]
            total_range = df['high'].iloc[i] - df['low'].iloc[i]
            ratio = abs(body) / (total_range + 1e-8)
            if body > 0:
                deltas.append(df['volume'].iloc[i] * ratio)
            else:
                deltas.append(-df['volume'].iloc[i] * ratio)
        cum_delta = np.sum(deltas[-20:])
        trend = np.mean(deltas[-5:]) / (np.mean(deltas[-20:]) + 1e-8)
        accel = np.mean(deltas[-3:]) - np.mean(deltas[-6:-3]) if len(deltas) >= 6 else 0
        score = 0.5
        if cum_delta > 0:
            score += 0.20
        if trend > 1.2:
            score += 0.15
        elif trend > 1.0:
            score += 0.08
        if accel > 0:
            score += 0.15
        score = np.clip(score, 0, 1)
        confidence = min(1.0, len(df) / 30.0)
        return FactorResult('delta_momentum', score, cum_delta, confidence,
                            {'cum_delta': cum_delta, 'trend': trend, 'accel': accel})

    # ================================================================
    #  V4.0 新增10个连续型因子（保留原逻辑，完全不变）
    # ================================================================

    def _factor_effort_result(self, df: pd.DataFrame) -> FactorResult:
        """
        O-04 增强版：努力vs结果因子（方向感知+持续性追踪）

        升级内容：
        1. 方向感知：上涨趋势中放量滞涨→负信号，下跌趋势中放量滞跌→正信号
        2. 持续性追踪：最近5日中背离方向一致的天数占比

        raw_value = |vol_deviation| × (1 - spread_efficiency) × sign(trend)
        - vol_deviation = (V - MA(V,20)) / MA(V,20)
        - spread_efficiency = |C - O| / (H - L + 1e-8)
        - sign(trend) = +1(上涨趋势) 或 -1(下跌趋势)
        """
        if len(df) < 20:
            return FactorResult('effort_result', 0.5, 0.0, 0.2, {})
        df = df.tail(60).copy()

        # 计算趋势方向（MA20 > MA60 为上涨趋势）
        ma20 = df['close'].rolling(20, min_periods=10).mean()
        ma60 = df['close'].rolling(60, min_periods=30).mean()
        trend_sign = 1.0 if ma20.iloc[-1] > ma60.iloc[-1] else -1.0

        # 计算成交量偏差
        ma_vol = df['volume'].rolling(20, min_periods=10).mean()
        vol_deviation = (df['volume'] - ma_vol) / (ma_vol + 1e-8)

        # 计算K线效率（实体占比）
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        open_p = df['open'].values
        spread_efficiency = np.abs(close - open_p) / (high - low + 1e-8)

        # 计算每日背离值
        daily_divergence = vol_deviation.values * (1 - spread_efficiency) * trend_sign

        # 计算raw_value（加权平均）
        weights = np.exp(np.linspace(-1, 0, len(daily_divergence)))
        weights = weights / weights.sum()
        raw_value = np.sum(daily_divergence[-20:] * weights[-20:])

        # 计算持续性（最近5日背离方向一致占比）
        recent_div = daily_divergence[-5:]
        pos_count = np.sum(recent_div > 0)
        neg_count = np.sum(recent_div < 0)
        persistence = max(pos_count, neg_count) / 5.0 if len(recent_div) == 5 else 0.5

        # score计算
        score = 0.5
        if abs(raw_value) > 0.3:  # 有明显背离
            if raw_value > 0:  # 正值背离
                if trend_sign > 0:  # 上涨趋势中放量滞涨（负面）
                    score = 0.3 + 0.2 * (1 - persistence)
                else:  # 下跌趋势中放量滞跌（正面）
                    score = 0.7 + 0.3 * persistence
            else:  # 负值背离
                if trend_sign > 0:  # 上涨趋势中缩量上涨（正面）
                    score = 0.6 + 0.3 * persistence
                else:  # 下跌趋势中缩量下跌（中性）
                    score = 0.4 + 0.1 * (1 - persistence)

        score = np.clip(score, 0, 1)
        confidence = min(1.0, len(df) / 30.0)

        return FactorResult('effort_result', score, raw_value, confidence,
                            {'vol_deviation_mean': float(np.mean(vol_deviation.values[-20:])),
                             'spread_efficiency_mean': float(np.mean(spread_efficiency[-20:])),
                             'trend_sign': trend_sign,
                             'persistence': persistence})

    def _factor_closing_position(self, df: pd.DataFrame) -> FactorResult:
        if len(df) < 5:
            return FactorResult('closing_position', 0.5, 0.0, 0.2, {})
        df = df.tail(20).copy()
        today_hl = df['high'].iloc[-1] - df['low'].iloc[-1]
        today_cp = (df['close'].iloc[-1] - df['low'].iloc[-1]) / today_hl if today_hl > 1e-8 else 0.5
        hls = df['high'] - df['low']
        cps = (df['close'] - df['low']) / hls.replace(0, 1e-8)
        raw_value = np.mean(cps.tail(20))
        score = raw_value
        confidence = min(1.0, len(df) / 20.0)
        return FactorResult('closing_position', score, raw_value, confidence,
                            {'today_cp': today_cp, 'mean_cp': np.mean(cps.tail(5))})

    def _factor_volume_concentration(self, df: pd.DataFrame) -> FactorResult:
        if len(df) < 30:
            return FactorResult('volume_concentration', 0.5, 0.0, 0.2, {})
        df = df.tail(60).copy()
        try:
            vp = self.vp_analyzer.analyze(
                high=df['high'].values,
                low=df['low'].values,
                volume=df['volume'].values
            )
            total_vol = np.sum(df['volume'].values)
            price_bins = np.linspace(df['low'].min(), df['high'].max(), 51)
            va_bins = (price_bins >= vp.value_area_low) & (price_bins <= vp.value_area_high)
            va_vol_pct = np.sum(vp.volume_by_bin[va_bins[:-1]]) / total_vol if total_vol > 0 else 0.5
            raw_value = va_vol_pct
            if 0.5 <= raw_value <= 0.7:
                score = 0.7
            elif raw_value > 0.7:
                score = 0.8
            else:
                score = 0.5
            confidence = min(1.0, len(df) / 60.0)
            return FactorResult('volume_concentration', score, raw_value, confidence,
                                {'va_low': vp.value_area_low, 'va_high': vp.value_area_high})
        except Exception:
            return FactorResult('volume_concentration', 0.5, 0.5, 0.2, {'error': True})

    def _factor_spread_ratio(self, df: pd.DataFrame) -> FactorResult:
        if len(df) < 20:
            return FactorResult('spread_ratio', 0.5, 0.0, 0.2, {})
        df = df.tail(30).copy()
        today_spread = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1] if df['close'].iloc[
                                                                                                -1] > 0 else 0
        spreads = (df['high'] - df['low']) / df['close'].replace(0, 1e-8)
        avg_spread = np.mean(spreads.tail(20))
        raw_value = today_spread / avg_spread if avg_spread > 0 else 1.0
        score = 0.5
        if raw_value < 0.5:
            score = 0.3
        elif raw_value < 0.8:
            score = 0.4
        elif raw_value <= 1.2:
            score = 0.6
        elif raw_value <= 2.0:
            score = 0.5
        else:
            score = 0.4
        confidence = min(1.0, len(df) / 25.0)
        return FactorResult('spread_ratio', score, raw_value, confidence,
                            {'today_spread': today_spread, 'avg_spread': avg_spread})

    def _factor_up_volume_ratio(self, df: pd.DataFrame) -> FactorResult:
        if len(df) < 20:
            return FactorResult('up_volume_ratio', 0.5, 0.0, 0.2, {})
        df = df.tail(30).copy()
        price_up = df['close'] > df['close'].shift(1)
        up_vol = np.sum(df.loc[price_up, 'volume'].tail(20))
        down_vol = np.sum(df.loc[~price_up, 'volume'].tail(20))
        raw_value = up_vol / down_vol if down_vol > 0 else 1.0
        score = 0.5
        if raw_value > 1.5:
            score = 0.8
        elif raw_value > 1.2:
            score = 0.7
        elif raw_value < 0.5:
            score = 0.2
        elif raw_value < 0.8:
            score = 0.3
        else:
            score = 0.5
        confidence = min(1.0, len(df) / 25.0)
        return FactorResult('up_volume_ratio', score, raw_value, confidence,
                            {'up_vol': up_vol, 'down_vol': down_vol})

    def _factor_demand_supply_ratio(self, df: pd.DataFrame) -> FactorResult:
        if len(df) < 20:
            return FactorResult('demand_supply_ratio', 0.5, 0.0, 0.2, {})
        df = df.tail(30).copy()
        body = df['close'] - df['open']
        demand = np.where(body > 0, df['volume'], 0)
        supply = np.where(body < 0, df['volume'], 0)
        net_demand = np.sum(demand[-20:]) - np.sum(supply[-20:])
        total_vol = np.sum(df['volume'].tail(20))
        raw_value = net_demand / total_vol if total_vol > 0 else 0.0
        score = (raw_value + 1) / 2
        score = np.clip(score, 0, 1)
        confidence = min(1.0, len(df) / 25.0)
        return FactorResult('demand_supply_ratio', score, raw_value, confidence,
                            {'net_demand': net_demand, 'total_vol': total_vol})

    def _factor_turnover_momentum(self, df: pd.DataFrame) -> FactorResult:
        if len(df) < 20 or 'turn' not in df.columns:
            return FactorResult('turnover_momentum', 0.5, 0.0, 0.2, {})
        df = df.tail(30).copy()
        turn_col = df['turn'].ffill().fillna(0)
        current_turn = turn_col.iloc[-1]
        avg_turn = np.mean(turn_col.tail(20))
        raw_value = current_turn / avg_turn if avg_turn > 0 else 1.0
        score = 0.5
        if raw_value > 2.0:
            score = 0.6
        elif raw_value > 1.5:
            score = 0.7
        elif raw_value > 1.0:
            score = 0.6
        elif raw_value > 0.5:
            score = 0.4
        else:
            score = 0.3
        confidence = min(1.0, len(df) / 25.0)
        return FactorResult('turnover_momentum', score, raw_value, confidence,
                            {'current_turn': current_turn, 'avg_turn': avg_turn})

    def _factor_price_position(self, df: pd.DataFrame) -> FactorResult:
        if len(df) < 20:
            return FactorResult('price_position', 0.5, 0.0, 0.2, {})
        df = df.tail(120).copy()
        current = df['close'].iloc[-1]
        price_min = df['low'].min()
        price_max = df['high'].max()
        raw_value = (current - price_min) / (price_max - price_min) if price_max > price_min else 0.5
        score = raw_value
        confidence = min(1.0, len(df) / 60.0)
        return FactorResult('price_position', score, raw_value, confidence,
                            {'current': current, 'min': price_min, 'max': price_max})

    def _factor_volume_trend(self, df: pd.DataFrame) -> FactorResult:
        if len(df) < 20:
            return FactorResult('volume_trend', 0.5, 0.0, 0.2, {})
        df = df.tail(30).copy()
        vol = df['volume'].tail(20).values
        x = np.arange(len(vol))
        x_mean = np.mean(x)
        vol_mean = np.mean(vol)
        numerator = np.sum((x - x_mean) * (vol - vol_mean))
        denominator = np.sum((x - x_mean) ** 2)
        slope = numerator / denominator if denominator > 0 else 0.0
        raw_value = slope / vol_mean if vol_mean > 0 else 0.0
        score = 0.5 + 0.3 * np.tanh(raw_value * 10)
        score = np.clip(score, 0, 1)
        confidence = min(1.0, len(df) / 25.0)
        return FactorResult('volume_trend', score, raw_value, confidence,
                            {'slope': slope, 'mean_vol': vol_mean})

    def _factor_smart_money_flow(self, df: pd.DataFrame) -> FactorResult:
        if len(df) < 20:
            return FactorResult('smart_money_flow', 0.5, 0.0, 0.2, {})
        df = df.tail(30).copy()
        smart_scores = []
        for i in range(len(df)):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            open_p = df['open'].iloc[i]
            close = df['close'].iloc[i]
            body_top = max(open_p, close)
            body_bottom = min(open_p, close)
            range_hl = high - low
            if range_hl < 1e-8:
                smart_scores.append(0)
                continue
            cp = (close - low) / range_hl
            is_up = close > open_p
            if is_up:
                smart = 0.5 + (cp - 0.5)
            else:
                smart = cp
            smart_scores.append(np.clip(smart, 0, 1))
        raw_value = np.mean(smart_scores[-20:])
        score = raw_value
        confidence = min(1.0, len(df) / 25.0)
        return FactorResult('smart_money_flow', score, raw_value, confidence,
                            {'today_smart': smart_scores[-1] if smart_scores else 0.5,
                             'trend': np.mean(smart_scores[-5:]) - np.mean(smart_scores[-10:-5])})

    # ================================================================
    #  V5.0新增16因子 - 市场结构类
    # ================================================================

    def _factor_spring_quality(self, df: pd.DataFrame) -> FactorResult:
        """
        S-02: Spring质量因子
        公式: Q_spring = Recovery_ratio × Volume_confirmation
        Recovery_ratio = (close - low) / (high - low), 越高越好（完全收回=1）
        Volume_confirmation = vol / MA(vol, 20), 放量确认
        高分意味着"市场已准备好向上突破"
        """
        if len(df) < 30:
            return FactorResult('spring_quality', 0.5, 0.0, 0.2, {})

        df = df.tail(60).copy()
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        vols = df['volume'].values

        # 寻找最近Spring信号（在SC后发生的价格短暂破低后收回）
        sc = self._detect_selling_climax(highs, lows, closes, vols)
        if not sc['detected']:
            # 没有SC，回退：寻找近期低点附近的缩量反弹
            window = 20
            best_quality = 0.0
            for i in range(window, len(df)):
                local_low = np.min(lows[max(0, i - 10):i + 1])
                # 价格接近局部低点
                if lows[i] <= local_low * 1.01:
                    recovery = (closes[i] - lows[i]) / (highs[i] - lows[i] + 1e-8)
                    avg_vol = np.mean(vols[max(0, i - 20):i])
                    vol_ratio = vols[i] / avg_vol if avg_vol > 0 else 1.0
                    quality = recovery * np.clip(vol_ratio, 0.5, 2.0)
                    if quality > best_quality:
                        best_quality = quality
            raw_value = best_quality
        else:
            # 在SC后找Spring
            spring_idx = -1
            for i in range(sc['index'] + 2, min(sc['index'] + 40, len(closes))):
                if lows[i] < lows[max(0, i - 5):i].min() * 1.005:
                    spring_idx = i
                    break
            if spring_idx > 0:
                recovery = (closes[spring_idx] - lows[spring_idx]) / (highs[spring_idx] - lows[spring_idx] + 1e-8)
                avg_vol = np.mean(vols[max(0, spring_idx - 20):spring_idx])
                vol_ratio = vols[spring_idx] / avg_vol if avg_vol > 0 else 1.0
                raw_value = recovery * np.clip(vol_ratio, 0.5, 2.0)
            else:
                raw_value = 0.0

        # 标准化
        score = np.clip(raw_value, 0, 1)
        confidence = min(1.0, len(df) / 60.0)
        return FactorResult('spring_quality', score, raw_value, confidence,
                            {'spring_quality': raw_value})

    def _factor_jump_creek(self, df: pd.DataFrame) -> FactorResult:
        """
        S-03: 跳跃小溪因子
        公式: J_creek = (P_current - P_breakout) / (P_high_after_breakout - P_breakout)
        价格突破盘整区间上沿后回踩的深度
        理想情况：价格沿阻力位上沿窄幅整理，不深跌回区间内
        """
        if len(df) < 40:
            return FactorResult('jump_creek', 0.5, 0.0, 0.2, {})

        df = df.tail(80).copy()
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        # 识别盘整区间（上沿=60日高点，下沿=60日低点）
        range_high = np.max(highs[:-10])
        range_low = np.min(lows[:-10])
        range_width = range_high - range_low

        if range_width < 1e-8:
            return FactorResult('jump_creek', 0.5, 0.0, 0.2, {})

        # 检测近期突破（收盘价突破60日高点）
        breakout_idx = -1
        for i in range(len(closes) - 5, max(20, len(closes) - 20), -1):
            if closes[i] > range_high * 1.01:
                breakout_idx = i
                break

        if breakout_idx < 0:
            # 无突破，检查是否接近上沿
            dist_to_high = (range_high - closes[-1]) / range_width
            raw_value = 1.0 - dist_to_high  # 越接近上沿越好
        else:
            # 计算突破后的回踩深度
            breakout_px = range_high
            high_after = np.max(closes[breakout_idx:breakout_idx + 10]) if breakout_idx + 10 < len(closes) else closes[
                -1]
            current_px = closes[-1]
            # 回踩但未完全回落到区间内
            if current_px > range_low:
                raw_value = (current_px - breakout_px) / (high_after - breakout_px + 1e-8)
                raw_value = np.clip(raw_value, 0, 1)
            else:
                raw_value = 0.0

        score = np.clip(raw_value, 0, 1)
        confidence = min(1.0, len(df) / 60.0)
        return FactorResult('jump_creek', score, raw_value, confidence,
                            {'jump_creek': raw_value, 'range_high': range_high, 'range_low': range_low})

    def _factor_stage_continuous(self, df: pd.DataFrame) -> FactorResult:
        """
        S-04: 结构阶段连续评分因子
        公式: S_continuous = Σ(w_i × event_i) × Time_decay
        权重: SC=0.10, Spring=0.25, SOS=0.20, LPS=0.15, 突破确认=0.30
        输出0~1连续值，越高越接近拉升启动点
        """
        if len(df) < 60:
            return FactorResult('stage_continuous', 0.3, 0.0, 0.2, {})

        df = df.tail(120).copy()
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        vols = df['volume'].values

        sc = self._detect_selling_climax(highs, lows, closes, vols)
        ar = self._detect_auto_rally(highs, lows, closes, sc)
        st = self._detect_secondary_test(highs, lows, closes, vols, sc)
        spring = self._detect_spring(highs, lows, closes, vols, sc)
        sos = self._detect_sign_of_strength(highs, lows, closes, vols, ar)
        lps = self._detect_last_point_support(highs, lows, closes, vols, sos)

        # 突破确认
        recent_high = np.max(closes[:-5])
        breakout_confirm = closes[-1] > recent_high * 1.01

        # 加权评分
        score = 0.0
        score += 0.10 if sc['detected'] else 0.0
        score += 0.25 if spring['detected'] else 0.0
        score += 0.20 if sos['detected'] else 0.0
        score += 0.15 if lps['detected'] else 0.0
        score += 0.30 if breakout_confirm else 0.0

        # 时间衰减：事件越近越好
        events = [sc, ar, st, spring, sos, lps]
        most_recent_event = -1
        for ev in events:
            if ev['detected'] and ev.get('index', -1) > most_recent_event:
                most_recent_event = ev['index']

        if most_recent_event > 0:
            days_ago = len(closes) - 1 - most_recent_event
            time_decay = np.exp(-days_ago / 30.0)  # 30天半衰期
            score *= (0.5 + 0.5 * time_decay)

        score = np.clip(score, 0, 1)
        confidence = min(1.0, len(df) / 120.0)
        return FactorResult('stage_continuous', score, score, confidence,
                            {'score': score, 'sc': sc['detected'], 'spring': spring['detected'],
                             'sos': sos['detected'], 'lps': lps['detected'], 'breakout': breakout_confirm})

    def _factor_ut_utad(self, df: pd.DataFrame) -> FactorResult:
        """
        S-05: 假突破（UT/UTAD）识别因子
        公式: UT_score = (upper_wick / bar_range) × Volume_confirmation
        价格在阻力位上方冲高后大幅回落（收盘接近当日低点），且伴随异常放量
        高分=强UT信号=应排除（负面因子）
        输出取反：高分→看多（不是UT）, 低分→看空（是UT）
        """
        if len(df) < 20:
            return FactorResult('ut_utad', 0.5, 0.0, 0.2, {})

        df = df.tail(40).copy()
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        vols = df['volume'].values
        opens = df['open'].values

        # 检测近期UT信号
        ut_score = 0.0
        for i in range(10, len(df)):
            bar_range = highs[i] - lows[i]
            if bar_range < 1e-8:
                continue
            upper_wick = highs[i] - max(opens[i], closes[i])
            upper_wick_ratio = upper_wick / bar_range

            avg_vol = np.mean(vols[max(0, i - 20):i])
            vol_ratio = vols[i] / avg_vol if avg_vol > 0 else 1.0

            # 判断是否在阻力位附近（接近60日高点）
            resistance = np.max(highs[max(0, i - 60):i])
            near_resistance = highs[i] >= resistance * 0.98

            # 收盘远离高点（收盘接近低点=主力出货）
            close_ratio = (highs[i] - closes[i]) / bar_range

            if upper_wick_ratio > 0.4 and close_ratio > 0.6 and near_resistance:
                ut_signal = upper_wick_ratio * np.clip(vol_ratio, 1.0, 3.0)
                if ut_signal > ut_score:
                    ut_score = ut_signal

        # 标准化: ut_score 0~1.5 → score (UT越强→ut_score越大→反转后score越高=看多)
        # ut_score=0(无UT) → score=1.0(正常), ut_score>1.0 → score=0.0(UT警告)
        raw_value = np.clip(1.0 - ut_score, 0, 1)
        score = raw_value  # 高分=无UT=正常
        confidence = min(1.0, len(df) / 40.0)
        return FactorResult('ut_utad', score, raw_value, confidence,
                            {'ut_score': ut_score, 'is_ut': ut_score > 0.5})

    def _factor_sos_strength(self, df: pd.DataFrame) -> FactorResult:
        """
        S-06: SOS（强势信号）强度因子
        公式: SOS_strength = (Pct_breakout / ATR_20) × (Vol / MA_vol_20) × Body_ratio
        第一项: 突破幅度（经ATR标准化）
        第二项: 成交量确认
        第三项: K线实体占比（实体越长，上影线越短，买方控制力越强）
        """
        if len(df) < 30:
            return FactorResult('sos_strength', 0.5, 0.0, 0.2, {})

        df = df.tail(60).copy()
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        vols = df['volume'].values
        opens = df['open'].values

        # 计算ATR
        tr_arr = np.maximum(highs[1:] - lows[1:],
                            np.maximum(np.abs(highs[1:] - closes[:-1]),
                                       np.abs(lows[1:] - closes[:-1])))
        atr = np.mean(tr_arr[-20:]) if len(tr_arr) >= 20 else 1.0

        # 检测SOS：放量长阳突破盘整区间
        sos_score = 0.0
        for i in range(20, len(df)):
            # 区间高点（过去20日的局部高点）
            range_high = np.max(opens[max(0, i - 20):i])

            if closes[i] > range_high * 1.01:  # 突破
                avg_vol = np.mean(vols[max(0, i - 20):i])
                vol_ratio = vols[i] / avg_vol if avg_vol > 0 else 1.0

                pct_breakout = (closes[i] - range_high) / (atr + 1e-8)

                bar_range = highs[i] - lows[i]
                body = abs(closes[i] - opens[i])
                upper_wick = highs[i] - max(opens[i], closes[i])
                body_ratio = body / (bar_range + 1e-8)

                sos = pct_breakout * np.clip(vol_ratio, 0.5, 3.0) * body_ratio
                if sos > sos_score:
                    sos_score = sos

        raw_value = np.clip(sos_score, 0, 2.0)
        # 标准化到0~1
        score = np.clip(raw_value / 1.5, 0, 1)
        confidence = min(1.0, len(df) / 60.0)
        return FactorResult('sos_strength', score, raw_value, confidence,
                            {'sos_strength': sos_score})

    # ================================================================
    #  V5.0新增16因子 - 成交量分布类
    # ================================================================

    def _factor_vpoc_momentum(self, df: pd.DataFrame) -> FactorResult:
        """
        V-01: VPOC动量因子
        公式: VPOC_momentum = (VPOC_20d - VPOC_40d_ago) / ATR_60
        计算过去20个交易日VPOC相对于再往前20个交易日VPOC的变化，
        经60日价格波动率标准化
        """
        if len(df) < 60:
            return FactorResult('vpoc_momentum', 0.5, 0.0, 0.2, {})

        df = df.tail(120).copy()
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        vols = df['volume'].values

        def get_vpoc(df_win):
            """计算窗口内的VPOC（成交量加权价格）"""
            h = df_win['high'].values
            l = df_win['low'].values
            v = df_win['volume'].values
            price_range = h.max() - l.min()
            if price_range < 1e-8:
                return (h.max() + l.min()) / 2
            bins = np.linspace(l.min(), h.max(), 51)
            bin_width = bins[1] - bins[0]
            vol_hist = np.zeros(len(bins) - 1)
            for i in range(len(df_win)):
                low_idx = int((df_win['low'].iloc[i] - l.min()) / bin_width)
                high_idx = int((df_win['high'].iloc[i] - l.min()) / bin_width)
                low_idx = max(0, min(low_idx, len(vol_hist) - 1))
                high_idx = max(0, min(high_idx, len(vol_hist) - 1))
                vol_hist[low_idx:high_idx + 1] += df_win['volume'].iloc[i]
            return bins[np.argmax(vol_hist)]

        # 计算ATR60
        tr_arr = np.maximum(highs[1:] - lows[1:],
                            np.maximum(np.abs(highs[1:] - closes[:-1]),
                                       np.abs(lows[1:] - closes[:-1])))
        atr = np.mean(tr_arr[-60:]) if len(tr_arr) >= 60 else 1.0

        vpoc_current = get_vpoc(df.tail(20))
        vpoc_past = get_vpoc(df.iloc[-40:-20]) if len(df) >= 40 else get_vpoc(df.tail(20))

        raw_value = (vpoc_current - vpoc_past) / (atr + 1e-8)
        # 标准化: 除以2作为大致范围
        score = 0.5 + 0.3 * np.tanh(raw_value / 2.0)
        score = np.clip(score, 0, 1)
        confidence = min(1.0, len(df) / 60.0)
        return FactorResult('vpoc_momentum', score, raw_value, confidence,
                            {'vpoc_current': vpoc_current, 'vpoc_past': vpoc_past, 'momentum': raw_value})

    def _factor_va_convergence(self, df: pd.DataFrame) -> FactorResult:
        """
        V-02: 价值区域收敛/发散因子
        公式: VA_ratio = VAwidth_20d / VAwidth_60d
        当比率小于0.7时，表明近期价值区域显著收窄，市场即将选择方向
        """
        if len(df) < 60:
            return FactorResult('va_convergence', 0.5, 0.0, 0.2, {})

        df_full = df.tail(120).copy()

        def calc_va_width(df_win):
            h = df_win['high'].values
            l = df_win['low'].values
            v = df_win['volume'].values
            price_range = h.max() - h.min()
            if price_range < 1e-8:
                return 0.0
            bins = np.linspace(l.min(), h.max(), 51)
            bin_width = bins[1] - bins[0]
            vol_hist = np.zeros(len(bins) - 1)
            for i in range(len(df_win)):
                low_idx = max(0, int((df_win['low'].iloc[i] - l.min()) / bin_width))
                high_idx = min(int((df_win['high'].iloc[i] - l.min()) / bin_width), len(vol_hist) - 1)
                vol_hist[low_idx:high_idx + 1] += df_win['volume'].iloc[i]
            total_vol = np.sum(vol_hist)
            cumsum = np.cumsum(vol_hist)
            threshold_70 = total_vol * 0.70
            # 找到70%成交量的价格范围
            idx_low = np.searchsorted(cumsum, threshold_70 * 0.15)
            idx_high = np.searchsorted(cumsum, threshold_70 * 0.85)
            if idx_high >= len(bins) - 1:
                idx_high = len(bins) - 2
            return bins[idx_high] - bins[idx_low]

        va_20d = calc_va_width(df_full.tail(20))
        va_60d = calc_va_width(df_full.tail(60))

        raw_value = va_20d / (va_60d + 1e-8) if va_60d > 0 else 1.0
        # 收窄(小值)→有方向选择的可能→中性或正向
        # 正常范围0.7~1.3，过小=收窄（可能突破前兆→正向），过大=发散
        score = 0.5
        if raw_value < 0.7:
            score = 0.7  # 收窄，市场蓄势
        elif raw_value > 1.3:
            score = 0.4  # 发散，方向不明
        score = np.clip(score, 0, 1)
        confidence = min(1.0, len(df) / 60.0)
        return FactorResult('va_convergence', score, raw_value, confidence,
                            {'va_20d': va_20d, 'va_60d': va_60d, 'ratio': raw_value})

    def _factor_vol_skewness(self, df: pd.DataFrame) -> FactorResult:
        """
        V-03: 成交量分布偏度因子
        公式: S_vol = (Σ(V_above_POC) - Σ(V_below_POC)) / Total_V
        阳性=成交量偏向上方（买方在更高价位积极交易），阴性=偏向下方的防御性特征
        """
        if len(df) < 30:
            return FactorResult('vol_skewness', 0.5, 0.0, 0.2, {})

        df_vp = df.tail(60).copy()
        try:
            vp = self.vp_analyzer.analyze(
                high=df_vp['high'].values,
                low=df_vp['low'].values,
                volume=df_vp['volume'].values
            )
            poc = vp.poc
            vol_bins = vp.volume_by_bin
            price_bins = vp.price_bins

            # POC位置索引
            poc_idx = np.searchsorted(price_bins, poc)
            poc_idx = min(poc_idx, len(vol_bins) - 1)

            vol_above = np.sum(vol_bins[poc_idx + 1:]) if poc_idx + 1 < len(vol_bins) else 0
            vol_below = np.sum(vol_bins[:poc_idx + 1]) if poc_idx >= 0 else 0
            total_vol = vol_above + vol_below + 1e-8

            raw_value = (vol_above - vol_below) / total_vol
            # 标准化到0~1
            score = 0.5 + 0.25 * np.tanh(raw_value)
            score = np.clip(score, 0, 1)
            confidence = min(1.0, len(df) / 60.0)
            return FactorResult('vol_skewness', score, raw_value, confidence,
                                {'vol_above': vol_above, 'vol_below': vol_below, 'poc': poc})
        except Exception:
            return FactorResult('vol_skewness', 0.5, 0.0, 0.2, {})

    def _factor_lvn_breakout(self, df: pd.DataFrame) -> FactorResult:
        """
        V-04: LVN突破概率因子
        公式: P_lvn_breakout = (P_current - P_lvn_above) / ATR_20 / OP_20d
        当前价格上方最近的LVN价格位置与当前价格的距离，
        经ATR和20日开价（OP）标准化
        """
        if len(df) < 30:
            return FactorResult('lvn_breakout', 0.5, 0.0, 0.2, {})

        df_vp = df.tail(60).copy()
        try:
            vp = self.vp_analyzer.analyze(
                high=df_vp['high'].values,
                low=df_vp['low'].values,
                volume=df_vp['volume'].values
            )
            current_px = df_vp['close'].iloc[-1]

            # 找当前价格上方的LVN
            vol_bins = vp.volume_by_bin
            price_bins = vp.price_bins
            total_vol = np.sum(vol_bins) + 1e-8
            vol_pct = vol_bins / total_vol

            # 找低量区（成交量占比<平均值的50%）
            avg_vol_pct = 1.0 / len(vol_bins)
            lvn_mask = vol_pct < avg_vol_pct * 0.5

            # 当前价格上方的LVN
            above_mask = price_bins[:-1] > current_px
            candidate_bins = lvn_mask[:-1] & above_mask
            if not np.any(candidate_bins):
                # 没有找到，用下一个低量区
                lvn_px = price_bins[np.argmax(vol_pct > avg_vol_pct * 0.5) + 1] if np.any(
                    vol_pct > avg_vol_pct * 0.5) else current_px
            else:
                lvn_px = np.min(price_bins[:-1][candidate_bins])

            # ATR20
            closes = df_vp['close'].values
            highs = df_vp['high'].values
            lows = df_vp['low'].values
            tr_arr = np.maximum(highs[1:] - lows[1:],
                                np.maximum(np.abs(highs[1:] - closes[:-1]),
                                           np.abs(lows[1:] - closes[:-1])))
            atr20 = np.mean(tr_arr[-20:]) if len(tr_arr) >= 20 else 1.0

            # OP20（开盘价波动）
            opens = df_vp['open'].values
            op_std = np.std(opens[-20:]) if len(opens) >= 20 else 1.0

            distance = lvn_px - current_px
            raw_value = distance / (atr20 * op_std / current_px + 1e-8)
            score = 0.5 + 0.25 * np.tanh(raw_value / 2.0)
            score = np.clip(score, 0, 1)
            confidence = min(1.0, len(df) / 60.0)
            return FactorResult('lvn_breakout', score, raw_value, confidence,
                                {'lvn_px': lvn_px, 'current_px': current_px, 'distance': distance})
        except Exception:
            return FactorResult('lvn_breakout', 0.5, 0.0, 0.2, {})

    def _factor_vol_entropy(self, df: pd.DataFrame) -> FactorResult:
        """
        V-05: 成交量熵因子
        公式: H_vol = -Σ(p_i × log(p_i)), p_i = V_i / ΣV
        低熵（成交量高度集中）=主力控盘程度高=正向
        高熵=市场方向不明确
        """
        if len(df) < 30:
            return FactorResult('vol_entropy', 0.5, 0.0, 0.2, {})

        df_vp = df.tail(60).copy()
        try:
            vp = self.vp_analyzer.analyze(
                high=df_vp['high'].values,
                low=df_vp['low'].values,
                volume=df_vp['volume'].values
            )
            vol_bins = vp.volume_by_bin
            total_vol = np.sum(vol_bins) + 1e-8
            p = vol_bins / total_vol
            p = p[p > 0]  # 去掉0值避免log(0)
            entropy = -np.sum(p * np.log(p))
            # 最大熵 = log(N)，标准化
            n_bins = len(vol_bins)
            max_entropy = np.log(n_bins)
            raw_value = entropy / (max_entropy + 1e-8) if max_entropy > 0 else 0.5
            # 低熵→高分（主力控盘）
            score = 1.0 - raw_value
            score = np.clip(score, 0, 1)
            confidence = min(1.0, len(df) / 60.0)
            return FactorResult('vol_entropy', score, raw_value, confidence,
                                {'entropy': entropy, 'max_entropy': max_entropy})
        except Exception:
            return FactorResult('vol_entropy', 0.5, 0.5, 0.2, {})

    # ================================================================
    #  V5.0新增16因子 - 订单流/VSA类
    # ================================================================

    def _factor_climax(self, df: pd.DataFrame) -> FactorResult:
        """
        O-01: VSA买入高潮/卖出高潮识别因子
        公式: S_climax = sign(rev) × |close - low| / (high - low) × vol / MA(vol, 20)
        sign(rev)在卖出高潮取正值（低位放量长阴后企稳），在买入高潮取负值
        当前价格在60日价格区间中的相对位置决定了高潮的类型归属
        """
        if len(df) < 30:
            return FactorResult('climax', 0.5, 0.0, 0.2, {})

        df = df.tail(60).copy()
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        vols = df['volume'].values

        # 60日价格区间位置
        price_60d_high = np.max(highs[:-5])
        price_60d_low = np.min(lows[:-5])
        price_range_60d = price_60d_high - price_60d_low
        price_pos = (closes[-1] - price_60d_low) / (price_range_60d + 1e-8)

        # 寻找高潮信号
        climax_score = 0.0
        for i in range(5, len(df)):
            bar_range = highs[i] - lows[i]
            if bar_range < 1e-8:
                continue
            avg_vol = np.mean(vols[max(0, i - 20):i])
            vol_ratio = vols[i] / (avg_vol + 1e-8)

            # 大成交量+大振幅
            if vol_ratio > 1.5 and bar_range / closes[i] > 0.03:
                # 卖出高潮：低位+收盘从低点反弹
                if lows[i] < np.percentile(lows[max(0, i - 20):i + 1], 20):
                    recovery = (closes[i] - lows[i]) / (bar_range + 1e-8)
                    climax = recovery * np.clip(vol_ratio, 1.0, 3.0)
                    if climax > climax_score:
                        climax_score = climax
                # 买入高潮：高位+收盘从高点回落（负面）
                elif highs[i] > np.percentile(highs[max(0, i - 20):i + 1], 80):
                    decline = (highs[i] - closes[i]) / (bar_range + 1e-8)
                    climax = -decline * np.clip(vol_ratio, 1.0, 3.0) * 0.5
                    if climax < climax_score:
                        climax_score = climax

        raw_value = climax_score
        # 标准化: 0~1, 0=无信号, 1=强卖出高潮(正向)
        score = np.clip(0.5 + climax_score / 2.0, 0, 1)
        confidence = min(1.0, len(df) / 60.0)
        return FactorResult('climax', score, raw_value, confidence,
                            {'climax_score': climax_score, 'price_pos_60d': price_pos})

    def _factor_no_demand(self, df: pd.DataFrame) -> FactorResult:
        """
        O-02: VSA无需求因子
        公式: ND_ratio = Count(NoDemand_days) / 20
        NoDemand日条件: C>C-1（价格收涨）, Spread<0.5×MA(Spread,20)（波幅狭窄）,
                         Vol<0.7×MA(Vol,20)（成交量萎缩）
        高ND_ratio=上涨无力=负面（选股中应降权）
        """
        if len(df) < 30:
            return FactorResult('no_demand', 0.5, 0.0, 0.2, {})

        df = df.tail(60).copy()
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        vols = df['volume'].values

        spreads = highs - lows
        ma_spread = pd.Series(spreads).rolling(20, min_periods=1).mean().values
        ma_vol = pd.Series(vols).rolling(20, min_periods=1).mean().values

        nd_count = 0
        for i in range(20, len(df)):
            price_up = closes[i] > closes[i - 1]
            narrow_spread = spreads[i] < 0.5 * ma_spread[i]
            low_vol = vols[i] < 0.7 * ma_vol[i]
            if price_up and narrow_spread and low_vol:
                nd_count += 1

        raw_value = nd_count / 20.0
        # 无需求=负面→分数要反转：无需求越少→分数越高
        score = 1.0 - np.clip(raw_value, 0, 1)
        score = np.clip(score, 0, 1)
        confidence = min(1.0, len(df) / 60.0)
        return FactorResult('no_demand', score, raw_value, confidence,
                            {'nd_count': nd_count, 'nd_ratio': raw_value})

    def _factor_stopping_vol(self, df: pd.DataFrame) -> FactorResult:
        """
        O-03: VSA停止成交量因子
        公式: S_stop = Vol_ratio × CloseRecovery × Downtrend_confirmation
        在下跌趋势背景下，放量且收盘价远离低点（买方在盘中低位大量承接后推高价格）
        """
        if len(df) < 30:
            return FactorResult('stopping_vol', 0.5, 0.0, 0.2, {})

        df = df.tail(60).copy()
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        vols = df['volume'].values
        opens = df['open'].values

        # 下跌趋势确认：20日均线向下
        ma20 = pd.Series(closes).rolling(20, min_periods=10).mean().values
        downtrend = ma20[-1] < ma20[-10] if len(ma20) >= 20 else False

        stopping_score = 0.0
        for i in range(10, len(df)):
            bar_range = highs[i] - lows[i]
            if bar_range < 1e-8:
                continue
            avg_vol = np.mean(vols[max(0, i - 20):i])
            vol_ratio = vols[i] / (avg_vol + 1e-8)

            # 放量阳线
            bullish = closes[i] > opens[i]
            close_recovery = (closes[i] - lows[i]) / (bar_range + 1e-8)
            near_low_start = lows[i] < np.percentile(lows[max(0, i - 10):i + 1], 30)

            if bullish and close_recovery > 0.5 and near_low_start and vol_ratio > 1.3:
                stopping = np.clip(vol_ratio, 1.0, 3.0) * close_recovery
                if stopping > stopping_score:
                    stopping_score = stopping

        raw_value = stopping_score
        score = 0.5 + 0.3 * np.tanh(raw_value / 2.0) if downtrend else 0.5 + 0.15 * np.tanh(raw_value / 2.0)
        score = np.clip(score, 0, 1)
        confidence = min(1.0, len(df) / 60.0)
        return FactorResult('stopping_vol', score, raw_value, confidence,
                            {'stopping_score': stopping_score, 'downtrend': downtrend})

    def _factor_weis_wave(self, df: pd.DataFrame) -> FactorResult:
        """
        O-05: 韦斯波累积/派发因子
        公式: W_score = (ΣV_upwaves - ΣV_downwaves) / ΣV_all_waves
        正值=上涨波段成交量占优（累积），负值=下跌波段成交量占优（派发）
        波识别：局部极值点，最小波幅阈值=1.5×ATR
        """
        if len(df) < 40:
            return FactorResult('weis_wave', 0.5, 0.0, 0.2, {})

        df = df.tail(80).copy()
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        vols = df['volume'].values

        # ATR
        tr_arr = np.maximum(highs[1:] - lows[1:],
                            np.maximum(np.abs(highs[1:] - closes[:-1]),
                                       np.abs(lows[1:] - closes[:-1])))
        atr = np.mean(tr_arr[-20:]) if len(tr_arr) >= 20 else 1.0
        min_wave_size = 1.5 * atr

        # 识别摆动点
        swing_highs = self._find_swing_points(highs, lookback=5, find_high=True)
        swing_lows = self._find_swing_points(lows, lookback=5, find_high=False)
        swing_points = sorted([(idx, 'H') for idx in swing_highs] + [(idx, 'L') for idx in swing_lows])

        if len(swing_points) < 3:
            return FactorResult('weis_wave', 0.5, 0.0, 0.2, {})

        vol_up = 0.0
        vol_down = 0.0
        for k in range(1, len(swing_points)):
            idx_prev, _ = swing_points[k - 1]
            idx_curr, _ = swing_points[k]
            wave_size = abs(closes[idx_curr] - closes[idx_prev])
            wave_vol = np.sum(vols[idx_prev:idx_curr + 1])
            if wave_size >= min_wave_size:
                if closes[idx_curr] > closes[idx_prev]:
                    vol_up += wave_vol
                else:
                    vol_down += wave_vol

        total_vol = vol_up + vol_down + 1e-8
        raw_value = (vol_up - vol_down) / total_vol
        score = 0.5 + 0.3 * np.tanh(raw_value * 3.0)
        score = np.clip(score, 0, 1)
        confidence = min(1.0, len(df) / 60.0)
        return FactorResult('weis_wave', score, raw_value, confidence,
                            {'vol_up': vol_up, 'vol_down': vol_down, 'net': raw_value})

    def _factor_big_flow(self, df: pd.DataFrame) -> FactorResult:
        """
        O-07: 大单资金流方向因子（基于日线近似）
        公式: BigFlow = MA20( direction × vol_intensity )
        direction = sign(VWAP - VWAP_prev) — 成交均价趋势
        vol_intensity = vol / MA(vol, 20)
        日线近似：用收盘价相对开盘价的强度×成交量强度
        """
        if len(df) < 30:
            return FactorResult('big_flow', 0.5, 0.0, 0.2, {})

        df = df.tail(60).copy()
        closes = df['close'].values
        opens = df['open'].values
        vols = df['volume'].values
        highs = df['high'].values
        lows = df['low'].values

        # 日线近似：计算每日"大资金"强度
        daily_scores = []
        for i in range(len(df)):
            # 方向：收盘价相对开盘价的强度
            body = closes[i] - opens[i]
            # 用当日VWAP近似（high+low+close）/3
            vwap = (highs[i] + lows[i] + closes[i]) / 3
            vwap_prev = (highs[i - 1] + lows[i - 1] + closes[i - 1]) / 3 if i > 0 else vwap
            direction = 1 if vwap > vwap_prev else -1

            # 成交量强度
            avg_vol = np.mean(vols[max(0, i - 20):i + 1])
            vol_intensity = vols[i] / (avg_vol + 1e-8)

            daily_scores.append(direction * np.clip(vol_intensity, 0.5, 3.0))

        raw_value = np.mean(daily_scores[-20:]) if len(daily_scores) >= 20 else np.mean(daily_scores)
        score = 0.5 + 0.3 * np.tanh(raw_value / 1.5)
        score = np.clip(score, 0, 1)
        confidence = min(1.0, len(df) / 60.0)
        return FactorResult('big_flow', score, raw_value, confidence,
                            {'big_flow': raw_value})

    def _factor_imbalance(self, df: pd.DataFrame) -> FactorResult:
        """
        O-08: 订单流失衡累积因子（基于日线近似）
        公式: I_balance = (ΣV_buy - ΣV_sell) / ΣV_total
        buy: 收盘>开盘的交易日成交量
        日线近似：用价格方向×成交量估算
        """
        if len(df) < 20:
            return FactorResult('imbalance', 0.5, 0.0, 0.2, {})

        df = df.tail(60).copy()
        closes = df['close'].values
        opens = df['open'].values
        vols = df['volume'].values

        buy_vol = 0.0
        sell_vol = 0.0
        for i in range(len(df)):
            if closes[i] > opens[i]:
                buy_vol += vols[i]
            else:
                sell_vol += vols[i]

        total_vol = buy_vol + sell_vol + 1e-8
        raw_value = (buy_vol - sell_vol) / total_vol
        # raw_value: -1~1 → score: 0~1
        score = (raw_value + 1) / 2
        score = np.clip(score, 0, 1)
        confidence = min(1.0, len(df) / 60.0)
        return FactorResult('imbalance', score, raw_value, confidence,
                            {'buy_vol': buy_vol, 'sell_vol': sell_vol, 'net_ratio': raw_value})

    # ================================================================
    #  V5.1新增4个因子
    # ================================================================

    def _factor_accumulation_score(self, df: pd.DataFrame) -> FactorResult:
        """
        S-01: 吸筹阶段评分因子
        理论依据：威科夫因果定律，吸筹区间结构完整度决定后续拉升可靠性

        公式: accumulation_score = Σ(w_i × event_i)
        事件权重：SC=0.10, Spring=0.25, SOS=0.20, LPS=0.15, Breakout=0.30

        实现逻辑：
        - SC（卖出高潮）识别：20日内最低价附近的放量长阴线，相对成交量>2.0
        - Spring识别：价格跌破20日最低价后收回（close > 20日最低价 × 1.02），且当日成交量>MA(V,20)×1.5
        - SOS识别：放量长阳突破20日最高价，close > open且(close-open)/(high-low)>0.6且volume>MA(V,20)×1.5
        - LPS识别：回踩至前期支撑位附近获得支撑（close>open且low接近20日最低价×1.05以内）
        - Breakout识别：收盘价创60日新高
        - 每个事件评分0~1（满足条件程度），加权求和
        """
        if len(df) < 30:
            return FactorResult('accumulation_score', 0.3, 0.0, 0.2, {})

        df = df.tail(120).copy()
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        vols = df['volume'].values
        opens = df['open'].values

        # 事件权重
        EVENT_WEIGHTS = {'SC': 0.10, 'Spring': 0.25, 'SOS': 0.20, 'LPS': 0.15, 'Breakout': 0.30}

        # 计算各事件评分
        scores = {'SC': 0.0, 'Spring': 0.0, 'SOS': 0.0, 'LPS': 0.0, 'Breakout': 0.0}

        # 20日最低价/最高价
        low_20 = pd.Series(lows).rolling(20, min_periods=1).min().values
        high_20 = pd.Series(highs).rolling(20, min_periods=1).max().values
        ma_vol_20 = pd.Series(vols).rolling(20, min_periods=10).mean().values

        # 60日最高价（用于Breakout）
        high_60 = np.max(highs[:-5]) if len(highs) >= 65 else np.max(highs)

        # 事件1: SC（卖出高潮）- 20日内最低价附近的放量长阴
        for i in range(20, len(df)):
            is_near_low = lows[i] <= low_20[i] * 1.02
            vol_ratio = vols[i] / (ma_vol_20[i] + 1e-8)
            is_big_vol = vol_ratio > 2.0
            is_down_bar = closes[i] < opens[i]
            if is_near_low and is_big_vol and is_down_bar:
                # 根据放量程度评分
                scores['SC'] = min(1.0, vol_ratio / 3.0)
                break

        # 事件2: Spring - 跌破20日最低后收回
        for i in range(25, len(df)):
            broke_low = lows[i] < low_20[i - 1] * 0.99  # 跌破
            recovered = closes[i] > low_20[i - 1] * 1.02  # 收回
            vol_ratio = vols[i] / (ma_vol_20[i] + 1e-8)
            if broke_low and recovered and vol_ratio > 1.5:
                scores['Spring'] = min(1.0, vol_ratio / 2.5)
                break

        # 事件3: SOS - 放量长阳突破20日最高
        for i in range(20, len(df)):
            broke_high = closes[i] > high_20[i - 1] * 1.01
            is_up_bar = closes[i] > opens[i]
            bar_range = highs[i] - lows[i]
            body_ratio = (closes[i] - opens[i]) / (bar_range + 1e-8) if bar_range > 1e-8 else 0
            vol_ratio = vols[i] / (ma_vol_20[i] + 1e-8)
            if broke_high and is_up_bar and body_ratio > 0.6 and vol_ratio > 1.5:
                scores['SOS'] = min(1.0, body_ratio * vol_ratio / 2.0)
                break

        # 事件4: LPS - 回踩支撑获得支撑
        for i in range(25, len(df)):
            is_up_bar = closes[i] > opens[i]
            near_support = lows[i] <= low_20[i - 1] * 1.05
            vol_ratio = vols[i] / (ma_vol_20[i] + 1e-8)
            if is_up_bar and near_support:
                scores['LPS'] = min(1.0, vol_ratio / 2.0)
                break

        # 事件5: Breakout - 收盘创60日新高
        if closes[-1] > high_60 * 1.01:
            vol_ratio = vols[-1] / (ma_vol_20[-1] + 1e-8)
            scores['Breakout'] = min(1.0, vol_ratio / 2.0)

        # 加权求和
        raw_value = sum(scores[k] * EVENT_WEIGHTS[k] for k in EVENT_WEIGHTS)
        raw_value = np.clip(raw_value, 0, 1)

        # score与raw_value相同（已在0~1区间）
        score = raw_value
        confidence = min(1.0, len(df) / 120.0)

        return FactorResult('accumulation_score', score, raw_value, confidence,
                            {'scores': scores, 'weighted_sum': raw_value})

    def _factor_vwap_deviation(self, df: pd.DataFrame) -> FactorResult:
        """
        V-06: VWAP乖离率因子
        理论依据：VWAP是主力资金平均成本锚点，价格相对VWAP位置+VWAP趋势方向判断市场偏好

        公式: D_vwap = (P_close - VWAP_20d) / VWAP_20d × 100%
              Slope_vwap = (VWAP_20d - VWAP_60d) / σ(P,60d)
        VWAP_20d = Σ(close × volume) / Σ(volume) over 20日
        """
        if len(df) < 60:
            return FactorResult('vwap_deviation', 0.5, 0.0, 0.2, {})

        df = df.tail(120).copy()
        closes = df['close'].values
        vols = df['volume'].values

        # 计算VWAP20d
        def calc_vwap_series(df_win):
            tp = (df_win['high'] + df_win['low'] + df_win['close']) / 3
            return (tp * df_win['volume']).sum() / (df_win['volume'].sum() + 1e-8)

        vwap_20d = calc_vwap_series(df.tail(20))

        # 计算VWAP60d（使用60日窗口的VWAP）
        vwap_60d = calc_vwap_series(df.tail(60)) if len(df) >= 60 else vwap_20d

        # 计算价格波动率
        price_std = np.std(closes[-60:]) if len(closes) >= 60 else np.std(closes)

        # 计算乖离率
        current_price = closes[-1]
        D_vwap = (current_price - vwap_20d) / (vwap_20d + 1e-8) * 100.0

        # 计算VWAP斜率
        Slope_vwap = (vwap_20d - vwap_60d) / (price_std + 1e-8)

        # 综合评分
        if D_vwap > 0 and Slope_vwap > 0:
            # 正乖离 + 正斜率 = 最强信号
            score = 0.7 + 0.3 * min(abs(D_vwap) / 5.0, 1.0)
        elif D_vwap > 0 and Slope_vwap < 0:
            # 正乖离 + 负斜率 = 次强信号
            score = 0.5 + 0.2 * min(abs(D_vwap) / 5.0, 1.0)
        elif D_vwap < 0 and Slope_vwap > 0:
            # 负乖离 + 正斜率 = 较弱信号
            score = 0.4 + 0.1 * min(abs(D_vwap) / 5.0, 1.0)
        else:
            # 负乖离 + 负斜率 = 最弱信号
            score = 0.3 * (1 - min(abs(D_vwap) / 5.0, 1.0))

        score = np.clip(score, 0, 1)
        confidence = min(1.0, len(df) / 60.0)

        return FactorResult('vwap_deviation', score, D_vwap, confidence,
                            {'D_vwap': D_vwap, 'Slope_vwap': Slope_vwap, 'vwap_20d': vwap_20d, 'vwap_60d': vwap_60d})

    def _factor_vol_pulse(self, df: pd.DataFrame) -> FactorResult:
        """
        V-07: 成交量脉冲因子
        理论依据：成交量突然放大代表主力集中行动，脉冲方向+影线质量决定信号性质

        公式: Pulse = relative_volume × direction × (1 - shadow_ratio)
        - relative_volume = V / MA(V, 20)
        - direction = sign(close - open)，阳线=+1，阴线=-1
        - shadow_ratio: 阳线时关注上影线，阴线时关注下影线（反方向影线）
        """
        if len(df) < 20:
            return FactorResult('vol_pulse', 0.5, 0.0, 0.2, {})

        df = df.tail(60).copy()
        closes = df['close'].values
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        vols = df['volume'].values

        # 计算相对成交量
        ma_vol_20 = pd.Series(vols).rolling(20, min_periods=10).mean().values

        # 计算每日脉冲
        pulses = []
        for i in range(len(df)):
            relative_volume = vols[i] / (ma_vol_20[i] + 1e-8)

            # 方向
            direction = 1.0 if closes[i] > opens[i] else -1.0

            # 影线比例
            bar_range = highs[i] - lows[i]
            if bar_range < 1e-8:
                shadow_ratio = 0.0
            else:
                if direction > 0:  # 阳线，关注上影线
                    shadow_ratio = (highs[i] - closes[i]) / bar_range
                else:  # 阴线，关注下影线
                    shadow_ratio = (opens[i] - lows[i]) / bar_range

            # 脉冲值
            pulse = relative_volume * direction * (1 - shadow_ratio)
            pulses.append(pulse)

        pulses = np.array(pulses)

        # 取最近10日内的最大脉冲值
        recent_pulses = pulses[-10:] if len(pulses) >= 10 else pulses
        raw_value = np.max(np.abs(recent_pulses)) * np.sign(recent_pulses[np.argmax(np.abs(recent_pulses))])

        # 计算最近5日脉冲方向一致性
        pulse_direction_consistency = 0.5
        if len(pulses) >= 5:
            recent_5 = pulses[-5:]
            pos_count = np.sum(recent_5 > 0)
            neg_count = np.sum(recent_5 < 0)
            pulse_direction_consistency = max(pos_count, neg_count) / 5.0

        # sigmoid标准化
        def sigmoid(x, center=0, scale=3):
            return 1.0 / (1.0 + np.exp(-(x - center) / scale))

        # score标准化到0~1
        score = sigmoid(raw_value, center=0, scale=3)

        # 置信度
        confidence = 0.5 + 0.5 * min(1.0, len(df) / 40.0) * (1 - abs(pulse_direction_consistency - 0.5))
        confidence = np.clip(confidence, 0, 1)

        return FactorResult('vol_pulse', score, raw_value, confidence,
                            {'max_pulse': raw_value, 'direction_consistency': pulse_direction_consistency})

    def _factor_hvn_support(self, df: pd.DataFrame) -> FactorResult:
        """
        V-08: HVN支撑强度因子
        理论依据：HVN代表市场深度博弈区域，价格回落到HVN上方获得支撑时构成强支撑

        公式: S_hvn = vol_share × time_share × proximity
        - HVN：成交量>平均成交量×1.5的价格区间
        - vol_share = HVN区间成交量 / 总成交量
        - time_share = 价格在HVN区间±2%范围内的交易日占比（20日窗口）
        - proximity = 1 - (close - HVN价格) / (close × 0.05)，价格越接近HVN得分越高
        """
        if len(df) < 30:
            return FactorResult('hvn_support', 0.3, 0.0, 0.2, {})

        df_vp = df.tail(60).copy()
        try:
            # 使用VolumeProfile分析
            vp = self.vp_analyzer.analyze(
                high=df_vp['high'].values,
                low=df_vp['low'].values,
                volume=df_vp['volume'].values
            )

            current_price = df_vp['close'].iloc[-1]
            closes = df_vp['close'].values
            highs = df_vp['high'].values
            lows = df_vp['low'].values
            vols = df_vp['volume'].values

            # 构建价格-成交量分布
            price_bins = vp.price_bins
            vol_bins = vp.volume_by_bin
            total_vol = np.sum(vol_bins) + 1e-8

            # 找到平均成交量
            avg_vol_per_bin = total_vol / len(vol_bins)

            # 识别HVN区间（成交量>平均×1.5）
            hvn_threshold = avg_vol_per_bin * 1.5
            hvn_mask = vol_bins > hvn_threshold

            if not np.any(hvn_mask):
                # 没有找到明显HVN，返回中性值
                return FactorResult('hvn_support', 0.3, 0.0, 0.3, {'hvn_exists': False})

            # 找到当前价格下方最近的HVN
            hvn_prices = price_bins[:-1][hvn_mask]
            below_mask = hvn_prices < current_price
            if not np.any(below_mask):
                # 当前价格在所有HVN下方
                hvn_price = np.min(hvn_prices)
            else:
                hvn_price = np.max(hvn_prices[below_mask])

            # vol_share
            hvn_vol = np.sum(vol_bins[hvn_mask])
            vol_share = hvn_vol / total_vol

            # time_share（价格停留在HVN区间±2%的时间占比）
            hvn_lower = hvn_price * 0.98
            hvn_upper = hvn_price * 1.02
            in_hvn_zone = ((lows >= hvn_lower) & (highs <= hvn_upper)) | \
                          ((lows <= hvn_price) & (highs >= hvn_price))
            time_share = np.mean(in_hvn_zone[-20:]) if len(in_hvn_zone) >= 20 else 0.5

            # proximity
            proximity = 1 - abs(current_price - hvn_price) / (current_price * 0.05 + 1e-8)
            proximity = np.clip(proximity, 0, 1)

            # 综合评分
            raw_value = vol_share * time_share * proximity
            raw_value = np.clip(raw_value, 0, 1)
            score = raw_value  # 已在0~1区间

            confidence = min(1.0, len(df) / 40.0)

            return FactorResult('hvn_support', score, raw_value, confidence,
                                {'hvn_exists': True, 'hvn_price': hvn_price,
                                 'vol_share': vol_share, 'time_share': time_share,
                                 'proximity': proximity})
        except Exception:
            return FactorResult('hvn_support', 0.3, 0.0, 0.2, {'error': True})

    # ================================================================
    #  工具方法
    # ================================================================

    def _normalize_triple(self, short, mid, accel,
                          pos_weight=0.4, trend_weight=0.35, accel_weight=0.25) -> float:
        """将三元组标准化为0~1"""
        score = 0.5
        if short > 0:
            score += pos_weight * 0.5
        if mid > 0:
            score += pos_weight * 0.5
        if short > mid:
            score += trend_weight
        if accel > 0:
            score += accel_weight
        return np.clip(score, 0, 1)
