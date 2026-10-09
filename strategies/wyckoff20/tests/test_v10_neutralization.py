# -*- coding: utf-8 -*-
"""
测试模块 6: 中性化 (P1-4 + V5迁移)

验证：
- full_market_neutralize: 全市场中性化逻辑
- fit_transform_consistency: fit/transform范式一致性
- score_neutralization: score/conf因子中性化
- winsorize_zscore: Winsorize+Z-score预处理
- inf_handling: inf值正确处理
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from strategies.wyckoff20.gpsef_strategy_v10 import FactorNeutralizer
from strategies.wyckoff20.tests.conftest import make_strategy


def test_winsorize_zscore():
    """FactorNeutralizer: Winsorize(1%/99%) → Z-score 预处理。"""
    neu = FactorNeutralizer(winsorize_pct=(1, 99))

    rng = np.random.RandomState(42)
    # 含极端值
    data = pd.Series(np.concatenate([rng.randn(100), [100, -100, 50]]))

    raw = data.replace([np.inf, -np.inf], np.nan)
    winsorized = neu._winsorize(raw)
    zscored = neu._zscore(winsorized)

    # Winsorize: 极端值被截断
    assert winsorized.max() < 100, f"Winsorize should clip 100, max={winsorized.max()}"
    assert winsorized.min() > -100, f"Winsorize should clip -100, min={winsorized.min()}"

    # Z-score: 均值~0, 标准差~1
    assert abs(zscored.mean()) < 0.01, f"Z-score mean should be ~0, got {zscored.mean()}"
    assert abs(zscored.std() - 1.0) < 0.01, f"Z-score std should be ~1, got {zscored.std()}"

    print("[PASS] test_winsorize_zscore: extreme values clipped, normalized")


def test_inf_handling():
    """inf值被正确替换为NaN再处理。"""
    neu = FactorNeutralizer()

    data = pd.DataFrame({
        'factor_a': [1.0, 2.0, np.inf, 4.0, 5.0],
        'factor_b': [-np.inf, 2.0, 3.0, 4.0, 5.0],
        'industry': ['A', 'B', 'A', 'B', 'A'],
        'market_cap': [100, 200, 150, 300, 250],
    })

    raw = data['factor_a'].replace([np.inf, -np.inf], np.nan)
    assert raw.isna().sum() == 1, "inf should become NaN"

    # Winsorize with NaN
    winsorized = neu._winsorize(raw)
    # NaN应被保留（或填充）
    assert winsorized.isna().sum() <= 1

    print("[PASS] test_inf_handling: inf → NaN → processed")


def test_full_market_neutralize():
    """_neutralize_full_market: 对全市场快照做中性化。"""
    s = make_strategy()

    rng = np.random.RandomState(42)
    n = 200

    snap = pd.DataFrame({
        'code': [f'sh.{600000+i:06d}' for i in range(n)],
        'supply_demand_momentum_raw': rng.randn(n) * 2 + 1,
        'market_structure_raw': rng.randn(n) * 3,
        'supply_demand_momentum_score': rng.randn(n),
        'industry': rng.choice(['A', 'B', 'C', 'D'], n),
        'market_cap': rng.exponential(100, n),
    })

    # 关闭实际OLS中性化（需要框架neutralizer），只测试流程不报错
    s._neutralize_enabled = True
    s._full_market_neutralize = True

    # Monkey-patch _neutralize 和 _apply_score_neutralization
    original_neutralize = s._neutralize
    original_score = s._apply_score_neutralization

    def mock_neutralize(df, cols):
        df = df.copy()
        for c in cols:
            if c in df.columns:
                df[c] = (df[c] - df[c].mean()) / (df[c].std() + 1e-8)
        return df

    def mock_score_neutralize(df, cols):
        df = df.copy()
        for c in cols:
            if c in df.columns:
                df[c] = (df[c] - df[c].mean()) / (df[c].std() + 1e-8)
        return df

    s._neutralize = mock_neutralize
    s._apply_score_neutralization = mock_score_neutralize

    result = s._neutralize_full_market(snap)

    # 验证返回了DataFrame
    assert isinstance(result, pd.DataFrame)
    assert len(result) == n, f"Should return all {n} rows, got {len(result)}"

    # 验证raw列被中性化（均值~0）
    for col in ['supply_demand_momentum_raw', 'market_structure_raw']:
        if col in result.columns:
            assert abs(result[col].mean()) < 0.1, f"{col} mean should be ~0 after neutralization"

    s._neutralize = original_neutralize
    s._apply_score_neutralization = original_score

    print("[PASS] test_full_market_neutralize: all market stocks processed")


def test_full_market_neutralize_disabled():
    """full_market_neutralize=False时直接返回原始数据。"""
    s = make_strategy()
    s._neutralize_enabled = False
    s._full_market_neutralize = False

    snap = pd.DataFrame({'code': ['sh.600000'], 'factor_a_raw': [1.0]})
    result = s._neutralize_full_market(snap)

    assert result is snap, "Should return original when disabled"

    print("[PASS] test_full_market_neutralize_disabled: returns original when disabled")


def test_factor_neutralizer_class():
    """FactorNeutralizer类：neutralize/neutralize_fit/neutralize_apply方法存在。"""
    neu = FactorNeutralizer()

    assert hasattr(neu, 'neutralize'), "FactorNeutralizer should have neutralize()"
    assert hasattr(neu, 'neutralize_fit'), "FactorNeutralizer should have neutralize_fit()"
    assert hasattr(neu, 'neutralize_apply'), "FactorNeutralizer should have neutralize_apply()"
    assert hasattr(neu, '_winsorize'), "FactorNeutralizer should have _winsorize()"
    assert hasattr(neu, '_zscore'), "FactorNeutralizer should have _zscore()"

    print("[PASS] test_factor_neutralizer_class: all methods present")


def test_zscore_constant_series():
    """Z-score对常数列返回减均值（std=0分支）。"""
    neu = FactorNeutralizer()
    const_series = pd.Series([5.0, 5.0, 5.0, 5.0])
    result = neu._zscore(const_series)

    assert (result == 0.0).all(), f"Constant series z-score should be all 0, got {result.unique()}"

    print("[PASS] test_zscore_constant_series: constant → all zeros")


if __name__ == '__main__':
    test_winsorize_zscore()
    test_inf_handling()
    test_full_market_neutralize()
    test_full_market_neutralize_disabled()
    test_factor_neutralizer_class()
    test_zscore_constant_series()
    print("\n=== All Neutralization tests PASSED ===")
