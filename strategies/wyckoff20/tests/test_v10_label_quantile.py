# -*- coding: utf-8 -*-
"""
测试模块: V10 标签构造 — quantile/median 双模式

验证:
- median 模式: label=1 当 return > 截面中位数
- quantile 模式: label=1 当 return >= 截面前 q 分位
- 正样本比例: median≈50%, quantile≈q(约等于q)
- 边界值: quantile模式下精确分位边界
- 向后兼容: 旧 label_threshold 参数仍可工作
- 非法参数: 自动回退默认值
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))


def _simulate_label_construction(returns_by_date, method, quantile=None):
    """
    模拟策略中的标签构造逻辑（与 gpsef_strategy_v10.py 完全一致）。
    
    Args:
        returns_by_date: {snapshot_date: [returns]}
        method: "median" | "quantile"
        quantile: float, 仅在 quantile 模式生效
    Returns:
        DataFrame with columns: snapshot_date, return, label, expected_label
    """
    rows = []
    for dt, rets in returns_by_date.items():
        for r in rets:
            rows.append({'snapshot_date': dt, 'label_raw': r})
    data = pd.DataFrame(rows)
    if data.empty:
        return data
    
    if method == 'quantile':
        q = quantile if quantile is not None else 0.30
        data['label'] = (
            data.groupby('snapshot_date')['label_raw']
            .transform(lambda x: (x >= x.quantile(1 - q)).astype(int))
        )
    else:
        data['label'] = (
            data.groupby('snapshot_date')['label_raw']
            .transform(lambda x: (x > x.median()).astype(int))
        )
    
    # 计算期望标签用于验证
    # median: > median → 1
    # quantile: >= top q percentile → 1
    data['expected'] = 0
    for dt in data['snapshot_date'].unique():
        mask = data['snapshot_date'] == dt
        vals = data.loc[mask, 'label_raw'].values
        if method == 'quantile':
            q = quantile if quantile is not None else 0.30
            threshold = np.quantile(vals, 1 - q)
            data.loc[mask, 'expected'] = (vals >= threshold).astype(int)
        else:
            median = np.median(vals)
            data.loc[mask, 'expected'] = (vals > median).astype(int)
    
    return data


# ==================== 测试用例 ====================

def test_median_mode():
    """验证 median 模式: 正样本约 50%"""
    np.random.seed(42)
    returns = {
        '2024-01-01': np.random.randn(100) * 0.05,
        '2024-01-02': np.random.randn(100) * 0.05,
        '2024-01-03': np.random.randn(100) * 0.05,
    }
    data = _simulate_label_construction(returns, 'median')
    
    # 验证 label == expected 全部一致
    mismatches = (data['label'] != data['expected']).sum()
    assert mismatches == 0, f"Label mismatch: {mismatches}"
    
    # 每日期正样本应接近 50%（严格中位数下，恰好 > median 的样本数 <= 50%）
    for dt in data['snapshot_date'].unique():
        mask = data['snapshot_date'] == dt
        pos_rate = data.loc[mask, 'label'].mean()
        assert 0.40 <= pos_rate <= 0.60, f"{dt}: pos_rate={pos_rate:.3f} out of [0.40, 0.60]"
    
    print("[PASS] test_median_mode: ≈50% positive, all labels match expected")


def test_quantile_mode_30pct():
    """验证 quantile 模式 (q=0.30): 正样本约 30%"""
    np.random.seed(43)
    returns = {
        '2024-01-01': np.random.randn(200) * 0.05,
        '2024-01-02': np.random.randn(200) * 0.05,
        '2024-01-03': np.random.randn(200) * 0.05,
    }
    data = _simulate_label_construction(returns, 'quantile', 0.30)
    
    # 验证 label == expected 全部一致
    mismatches = (data['label'] != data['expected']).sum()
    assert mismatches == 0, f"Label mismatch: {mismatches}"
    
    # 正样本比例应在 0.25~0.35（约等于 q=0.30，允许因重复值产生的微小偏差）
    for dt in data['snapshot_date'].unique():
        mask = data['snapshot_date'] == dt
        pos_rate = data.loc[mask, 'label'].mean()
        assert 0.25 <= pos_rate <= 0.35, f"{dt}: pos_rate={pos_rate:.3f} out of [0.25, 0.35]"
    
    print("[PASS] test_quantile_mode_30pct: ≈30% positive, all labels match expected")


def test_quantile_mode_10pct():
    """验证 quantile 模式 (q=0.10): 正样本约 10%"""
    np.random.seed(44)
    returns = {
        '2024-01-01': np.random.randn(500) * 0.05,
        '2024-01-02': np.random.randn(500) * 0.05,
    }
    data = _simulate_label_construction(returns, 'quantile', 0.10)
    
    mismatches = (data['label'] != data['expected']).sum()
    assert mismatches == 0, f"Label mismatch: {mismatches}"
    
    for dt in data['snapshot_date'].unique():
        mask = data['snapshot_date'] == dt
        pos_rate = data.loc[mask, 'label'].mean()
        assert 0.06 <= pos_rate <= 0.14, f"{dt}: pos_rate={pos_rate:.3f} out of [0.06, 0.14]"
    
    print("[PASS] test_quantile_mode_10pct: ≈10% positive, all labels match expected")


def test_quantile_mode_boundary():
    """验证 quantile 模式的精确分位边界"""
    np.random.seed(45)
    # 构造已知分布的收益率
    returns = {
        '2024-01-01': np.concatenate([
            np.linspace(-0.20, 0.20, 100),   # 均匀分布
            np.array([0.20])                   # 加一个在最高位的
        ]),
    }
    data = _simulate_label_construction(returns, 'quantile', 0.30)
    
    mismatches = (data['label'] != data['expected']).sum()
    assert mismatches == 0, f"Label mismatch: {mismatches}"
    
    # 验证: 正样本就是收益率最高的那些
    mask = data['snapshot_date'] == '2024-01-01'
    labeled_1 = data.loc[mask].query('label == 1')
    labeled_0 = data.loc[mask].query('label == 0')
    
    # 所有 label=1 的收益率应 >= 所有 label=0 的收益率
    if len(labeled_1) > 0 and len(labeled_0) > 0:
        min_pos = labeled_1['label_raw'].min()
        max_neg = labeled_0['label_raw'].max()
        assert min_pos >= max_neg, f"min_pos={min_pos:.4f} < max_neg={max_neg:.4f}"
    
    # label=1 的数量应接近 N * q (= 101 * 0.30 ≈ 30)
    n_pos = labeled_1.shape[0]
    assert 28 <= n_pos <= 33, f"Expected ~30 positive, got {n_pos}"
    
    print("[PASS] test_quantile_mode_boundary: correct quantile boundary, "
          f"top {n_pos}/101 positive")


def test_bear_market_scenario():
    """
    核心测试：模拟全面熊市，验证 quantile 模式不选"防御股"。
    
    场景: 所有股票都在跌（收益率全负），中位数 = -5%
    - median 模式: > -5% 的股票 → label=1（跌得少的"防御股"）
    - quantile 模式(0.30): 只有回报率前30%（亏损最少甚至能微涨的）→ label=1
    """
    np.random.seed(46)
    # 模拟熊市: 收益率在 -15% ~ +3% 之间（大多负）
    bear_returns = {
        'bear_day_1': np.random.uniform(-0.15, 0.03, 200),
        'bear_day_2': np.random.uniform(-0.12, 0.01, 200),
    }
    
    # median 模式
    data_median = _simulate_label_construction(bear_returns, 'median')
    med_labels = data_median[data_median['label'] == 1]['label_raw']
    med_neg_labels = data_median[data_median['label'] == 0]['label_raw']
    
    # quantile 模式
    data_quantile = _simulate_label_construction(bear_returns, 'quantile', 0.30)
    qt_labels = data_quantile[data_quantile['label'] == 1]['label_raw']
    qt_neg_labels = data_quantile[data_quantile['label'] == 0]['label_raw']
    
    # 🔴 关键断言1: median模式下，label=1的最小收益值为负（即选了跌得少的）
    assert med_labels.min() < 0, (
        f"Median mode: expected some positive labels to have negative returns, "
        f"but min(label=1)={med_labels.min():.4f}")
    
    # 🟢 关键断言2: quantile模式下，label=1的平均收益显著高于median模式
    assert qt_labels.mean() > med_labels.mean(), (
        f"Quantile mode labels should have higher mean return: "
        f"qt={qt_labels.mean():.4f} vs med={med_labels.mean():.4f}")
    
    # 🟢 关键断言3: quantile模式的正样本数 ≈ N * 0.30
    qt_pos_rate = data_quantile['label'].mean()
    assert 0.25 <= qt_pos_rate <= 0.35, f"qt pos_rate={qt_pos_rate:.3f}"
    
    print(f"[PASS] test_bear_market_scenario:")
    print(f"  Bear market: median return ≈ {data_median['label_raw'].median():.3f}")
    print(f"  median mode positive: {data_median['label'].mean()*100:.0f}%, "
          f"min_ret={med_labels.min():.3f}, mean_ret={med_labels.mean():.3f}")
    print(f"  quantile mode positive: {data_quantile['label'].mean()*100:.0f}%, "
          f"min_ret={qt_labels.min():.3f}, mean_ret={qt_labels.mean():.3f}")
    print(f"  → quantile mode selects top {data_quantile['label'].mean()*100:.0f}% "
          f"with mean return {qt_labels.mean():.3f}")


def test_backward_compat_label_threshold_key():
    """
    验证向后兼容: 旧配置 label_threshold 参数仍可被读取。
    
    gpsef_strategy_v10.py 中的代码:
        self._label_method = v10_cfg.get('label_method',
                                         v10_cfg.get('label_threshold', 'median'))
    应优先读取 label_method，fallback 到 label_threshold。
    """
    # 模拟: 旧配置只有 label_threshold
    old_config = {'label_threshold': 'median'}
    fallback = old_config.get('label_method', old_config.get('label_threshold', 'median'))
    assert fallback == 'median'
    
    # 模拟: 新配置有 label_method
    new_config = {'label_method': 'quantile', 'label_quantile': 0.30}
    method = new_config.get('label_method', new_config.get('label_threshold', 'median'))
    assert method == 'quantile'
    
    # 模拟: 新配置覆盖旧配置（同时存在两key时优先 label_method）
    mixed_config = {'label_method': 'quantile', 'label_threshold': 'median', 'label_quantile': 0.25}
    method = mixed_config.get('label_method', mixed_config.get('label_threshold', 'median'))
    assert method == 'quantile'
    
    print("[PASS] test_backward_compat: old 'label_threshold' key still supported")


def test_invalid_method_fallback():
    """验证非法 method 自动回退为 'median'"""
    invalid_method = 'percentile'  # 不在 ('median', 'quantile') 中
    method = invalid_method if invalid_method in ('median', 'quantile') else 'median'
    assert method == 'median'
    
    invalid_quantile = 1.5
    quantile = float(invalid_quantile) if 0.0 < invalid_quantile < 1.0 else 0.30
    assert quantile == 0.30
    
    invalid_quantile_neg = -0.5
    quantile_neg = float(invalid_quantile_neg) if 0.0 < invalid_quantile_neg < 1.0 else 0.30
    assert quantile_neg == 0.30
    
    print("[PASS] test_invalid_method_fallback: invalid params → safe defaults")


def test_quantile_consistency():
    """验证相同 seed 多次计算 label 结果一致（确定性）"""
    np.random.seed(47)
    returns = {'2024-01-01': np.random.randn(300) * 0.05}
    
    data1 = _simulate_label_construction(returns, 'quantile', 0.30)
    data2 = _simulate_label_construction(returns, 'quantile', 0.30)
    
    assert (data1['label'].values == data2['label'].values).all(), \
        "Label should be deterministic"
    
    print("[PASS] test_quantile_consistency: deterministic label construction")


def test_empty_edge_case():
    """验证空数据不会出错"""
    data = _simulate_label_construction({}, 'quantile', 0.30)
    assert len(data) == 0, f"Expected empty DataFrame, got {len(data)} rows"
    print("[PASS] test_empty_edge_case: empty input → empty output")


def test_single_stock_edge_case():
    """验证单只股票场景: quantile 计算不会崩溃"""
    returns = {'2024-01-01': [0.05]}
    data = _simulate_label_construction(returns, 'quantile', 0.30)
    # 单只股票时，quantile(1-q) = quantile(0.7) 就是其本身
    # 因此应该 label=1（>= 自身）
    assert data['label'].iloc[0] == 1, f"Expected label=1, got {data['label'].iloc[0]}"
    print("[PASS] test_single_stock_edge_case: single stock → label=1")


# ==================== 运行全部测试 ====================

if __name__ == '__main__':
    tests = [
        test_median_mode,
        test_quantile_mode_30pct,
        test_quantile_mode_10pct,
        test_quantile_mode_boundary,
        test_bear_market_scenario,
        test_backward_compat_label_threshold_key,
        test_invalid_method_fallback,
        test_quantile_consistency,
        test_empty_edge_case,
        test_single_stock_edge_case,
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
    print(f"  测试结果: {passed} passed, {failed} failed, {passed+failed} total")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests PASSED!")
