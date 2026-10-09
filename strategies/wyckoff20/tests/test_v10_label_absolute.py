# -*- coding: utf-8 -*-
"""
测试模块: V10 标签构造 — absolute 绝对阈值模式

验证:
- absolute 模式: label_raw > threshold → label=1 (与截面无关)
- 三重模式对比: median/quantile/absolute 在相同数据上的标签差异
- 不同市场环境: 牛市(全正)/熊市(全负)/震荡(混合)
- 边界: threshold=0时全正, threshold非常大时全负
- 向后兼容: median/quantile 模式不受影响
- 配置读取: label_absolute_threshold 默认值0.10
- 确定性: 相同输入多次计算一致
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))


def _simulate_label_construction(returns_by_date, method, quantile=None, absolute_threshold=None):
    """模拟策略中的标签构造逻辑（与 gpsef_strategy_v10.py 完全一致）"""
    rows = []
    for dt, rets in returns_by_date.items():
        for r in rets:
            rows.append({'snapshot_date': dt, 'label_raw': r})
    data = pd.DataFrame(rows)
    if data.empty:
        return data

    if method == 'absolute':
        threshold = absolute_threshold if absolute_threshold is not None else 0.10
        data['label'] = (data['label_raw'] > threshold).astype(int)
    elif method == 'quantile':
        q = quantile if quantile is not None else 0.20
        data['label'] = (
            data.groupby('snapshot_date')['label_raw']
            .transform(lambda x: (x >= x.quantile(1 - q)).astype(int))
        )
    else:  # median
        data['label'] = (
            data.groupby('snapshot_date')['label_raw']
            .transform(lambda x: (x > x.median()).astype(int))
        )
    return data


# ==================== 测试用例 ====================

def test_absolute_bull_market():
    """牛市: 所有股涨超10% → 全正"""
    returns = {'bull_day': np.array([0.12, 0.15, 0.18, 0.25, 0.30, 0.11, 0.13, 0.20, 0.17, 0.14])}
    data = _simulate_label_construction(returns, 'absolute', absolute_threshold=0.10)
    assert data['label'].mean() == 1.0, f"Bull market with all >10% should be all positive, got {data['label'].mean()}"
    print("[PASS] Bull market (all >10pct) -> 100% positive")


def test_absolute_bear_market():
    """熊市: 所有股跌幅超10% → 全负"""
    returns = {'bear_day': np.array([-0.15, -0.12, -0.20, -0.11, -0.18, -0.13, -0.14, -0.16, -0.19, -0.17])}
    data = _simulate_label_construction(returns, 'absolute', absolute_threshold=0.10)
    assert data['label'].mean() == 0.0, f"Bear market with all <-10% should be all negative, got {data['label'].mean()}"
    print("[PASS] Bear market (all <-10pct) -> 0% positive")


def test_absolute_mixed_market():
    """混合市场: 部分 >10%, 部分在 -10%~10% 之间"""
    returns = {'mixed_day': np.array([0.15, 0.05, 0.12, -0.03, 0.08, 0.20, 0.11, -0.07, 0.09, 0.14])}
    data = _simulate_label_construction(returns, 'absolute', absolute_threshold=0.10)
    # 0.15, 0.12, 0.20, 0.11, 0.14 → 5个 >0.10, 其余5个 <=0.10
    assert data['label'].sum() == 5, f"Expected 5 positive, got {data['label'].sum()}"
    assert data['label'].mean() == 0.50
    print("[PASS] Mixed market -> exactly 5/10 positive")


def test_absolute_edge_exactly_threshold():
    """精确等于阈值 → label=0 (strict > comparison)"""
    returns = {'edge_day': np.array([0.10, 0.15, -0.05])}
    data = _simulate_label_construction(returns, 'absolute', absolute_threshold=0.10)
    # Only 0.15 > 0.10
    assert data['label'].iloc[1] == 1
    assert data['label'].iloc[0] == 0, f"label_raw=0.10 should NOT be >0.10, got label={data['label'].iloc[0]}"
    assert data['label'].iloc[2] == 0
    assert data['label'].sum() == 1
    print("[PASS] Edge: exactly threshold -> label=0 (strict >)")


def test_absolute_threshold_zero():
    """threshold=0 -> 全部正样本（只要有正收益）"""
    returns = {'day': np.array([0.001, -0.001, -0.05, 0.02, 0.00])}
    data = _simulate_label_construction(returns, 'absolute', absolute_threshold=0.0)
    # 0.001, 0.02 > 0 → 2个; 0.00 = 0 not > 0 → 0个
    assert data['label'].sum() == 2, f"Expected 2 positive, got {data['label'].sum()}"
    print("[PASS] threshold=0 -> only actual positive returns")


def test_absolute_very_high_threshold():
    """threshold=10.0(1000%) -> 全负"""
    returns = {'day': np.array([0.50, 0.80, 1.20, 5.0, -0.10])}
    data = _simulate_label_construction(returns, 'absolute', absolute_threshold=10.0)
    assert data['label'].mean() == 0.0, "All should be negative with threshold=10.0"
    print("[PASS] threshold=10.0 (1000pct) -> all negative")


def test_absolute_vs_quantile_same_data():
    """同一数据 absolute vs quantile 的标签不同"""
    returns = {
        'day1': np.random.RandomState(42).randn(200) * 0.08 + 0.02,  # mean ~2%
    }
    data_abs = _simulate_label_construction(returns, 'absolute', absolute_threshold=0.10)
    data_qt = _simulate_label_construction(returns, 'quantile', quantile=0.20)

    # absolute: 只有真的 >10% 的
    # quantile: 总是前20%
    assert data_qt['label'].sum() > data_abs['label'].sum(), \
        f"In mixed market, quantile should have more positives than absolute. " \
        f"abs={data_abs['label'].sum()}, qt={data_qt['label'].sum()}"
    print(f"[PASS] abs vs quantile on same data: absolute={data_abs['label'].mean():.2%}, quantile={data_qt['label'].mean():.2%}")


def test_three_mode_comparison():
    """三重模式在震荡市的标签对比"""
    np.random.seed(99)
    returns = {'day': np.random.randn(300) * 0.10}

    data_median = _simulate_label_construction(returns, 'median')
    data_quantile = _simulate_label_construction(returns, 'quantile', quantile=0.20)
    data_absolute = _simulate_label_construction(returns, 'absolute', absolute_threshold=0.10)

    median_pos = data_median['label'].mean()
    quantile_pos = data_quantile['label'].mean()
    absolute_pos = data_absolute['label'].mean()

    # median ≈ 50%
    assert 0.45 <= median_pos <= 0.55, f"median pos rate={median_pos:.3f}"
    # quantile ≈ 20%
    assert 0.17 <= quantile_pos <= 0.23, f"quantile pos rate={quantile_pos:.3f}"
    # absolute depends on market, should be < 50% in random walk
    assert absolute_pos < 0.50, f"absolute should be <50% in random walk (zero mean), got {absolute_pos:.3f}"

    print(f"[PASS] Three modes on rw data: median={median_pos:.2%}, quantile={quantile_pos:.2%}, absolute={absolute_pos:.2%}")


def test_absolute_no_dependency_on_cross_section():
    """核心验证: absolute模式完全不依赖截面分布"""
    returns = {
        'day_a': np.array([0.12, 0.08, 0.14]),  # 2个 >0.10
        'day_b': np.array([0.05, 0.03, 0.01]),  # 0个 >0.10 (day_b整体弱, 但阈值独立)
    }
    data = _simulate_label_construction(returns, 'absolute', absolute_threshold=0.10)

    assert data[data['snapshot_date'] == 'day_a']['label'].sum() == 2
    assert data[data['snapshot_date'] == 'day_b']['label'].sum() == 0
    print("[PASS] Absolute mode: labels depend only on own return, NOT on cross-section")


def test_deterministic():
    """确定性: 同数据多次调用一致"""
    np.random.seed(55)
    returns = {'day1': np.random.randn(100) * 0.10}
    d1 = _simulate_label_construction(returns, 'absolute', absolute_threshold=0.10)
    d2 = _simulate_label_construction(returns, 'absolute', absolute_threshold=0.10)
    assert (d1['label'].values == d2['label'].values).all()
    print("[PASS] Deterministic across calls")


def test_median_still_works():
    """向后兼容: median 模式不受影响"""
    returns = {'day': np.array([-0.05, 0.01, 0.02, 0.03, 0.10])}
    data = _simulate_label_construction(returns, 'median')
    assert data['label'].sum() == 2, f"median: returns > median(0.02) → 2 positive, got {data['label'].sum()}"
    print("[PASS] Median mode still works (backward compat)")


def test_quantile_still_works():
    """向后兼容: quantile 模式不受影响"""
    returns = {'day': np.arange(1, 11, dtype=float)}  # 1..10
    data = _simulate_label_construction(returns, 'quantile', quantile=0.20)
    assert data['label'].sum() == 2, f"quantile(0.20) of 10 items → 2 positive, got {data['label'].sum()}"
    assert data.loc[data['label_raw'] == 10, 'label'].iloc[0] == 1
    assert data.loc[data['label_raw'] == 9, 'label'].iloc[0] == 1
    print("[PASS] Quantile mode still works (backward compat)")


def test_default_threshold_0_10():
    """默认阈值 0.10 (10%)"""
    returns = {'day': np.array([0.05, 0.10, 0.11, 0.15, -0.05])}
    data = _simulate_label_construction(returns, 'absolute')  # no threshold → default 0.10
    # 0.11, 0.15 > 0.10
    assert data['label'].sum() == 2
    print("[PASS] Default threshold 0.10: 2/5 positive")


def test_negative_threshold():
    """threshold=-0.05 (允许小幅亏损也算强)"""
    returns = {'day': np.array([-0.06, -0.03, 0.01, 0.05, -0.08])}
    data = _simulate_label_construction(returns, 'absolute', absolute_threshold=-0.05)
    # -0.03, 0.01, 0.05 > -0.05
    assert data['label'].sum() == 3
    print("[PASS] Negative threshold=-0.05: 3/5 positive")


# ==================== 运行全部测试 ====================

if __name__ == '__main__':
    tests = [
        test_absolute_bull_market,
        test_absolute_bear_market,
        test_absolute_mixed_market,
        test_absolute_edge_exactly_threshold,
        test_absolute_threshold_zero,
        test_absolute_very_high_threshold,
        test_absolute_vs_quantile_same_data,
        test_three_mode_comparison,
        test_absolute_no_dependency_on_cross_section,
        test_deterministic,
        test_median_still_works,
        test_quantile_still_works,
        test_default_threshold_0_10,
        test_negative_threshold,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            import traceback
            print(f"[FAIL] {test.__name__}: {e}")
            traceback.print_exc()
        except Exception as e:
            failed += 1
            import traceback
            print(f"[ERROR] {test.__name__}: {e}")
            traceback.print_exc()

    print()
    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {passed+failed} total")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("All tests PASSED!")
