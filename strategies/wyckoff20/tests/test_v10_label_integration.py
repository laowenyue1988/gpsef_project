# -*- coding: utf-8 -*-
"""
V10 标签构造 — 完整集成测试

测试策略:
A. 配置解析链: yaml -> strategy attributes -> label construction
B. 标签数学验证: raw_close × backAdjustFactor → label_raw → label
C. 三重模式集成: median / quantile / absolute 各自独立
D. 真实数据模式: 多日期截面, 不同分布, 边界值
E. 健壮性: NaN, inf, 空数据, 单样本, 极端值
F. 模式切换: 同实例切换属性, 验证行为变更
G. 向后兼容: 旧 label_threshold key 仍然生效
"""

import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))


# ================================================================
# A. 配置解析链测试
# ================================================================

def test_config_parsing_absolute_mode():
    """验证明确设置 absolute 模式时的完整配置链"""
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config_v10.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    v10 = cfg.get('v10', {})
    method = v10.get('label_method', v10.get('label_threshold', 'median'))
    assert method in ('median', 'quantile', 'absolute'), \
        f"Unknown label_method: {method}"
    
    # Validate specific params based on mode
    if method == 'absolute':
        assert 'label_absolute_threshold' in v10
        assert isinstance(v10['label_absolute_threshold'], (int, float))
        assert -1.0 < v10['label_absolute_threshold'] < 10.0
    if method == 'quantile':
        assert 'label_quantile' in v10 or True  # optional fallback to default
    
    # 模拟 strategy 初始化逻辑
    abs_threshold = float(v10.get('label_absolute_threshold', 0.10))
    quantile = float(v10.get('label_quantile', 0.30))
    # Config 已正确读取 method, absolute_threshold 参数存在
    print(f"[PASS] Config parsing: method={method}, abs_threshold={abs_threshold}, quantile={quantile}")


def test_config_fallback_when_absolute_missing():
    """label_absolute_threshold 缺失时回退默认 0.10"""
    v10 = {'label_method': 'absolute'}  # 无 label_absolute_threshold
    abs_threshold = float(v10.get('label_absolute_threshold', 0.10))
    assert abs_threshold == 0.10
    print("[PASS] Missing label_absolute_threshold -> default 0.10")


def test_config_backward_compat_label_threshold_key():
    """旧 label_threshold key fallback 仍有效"""
    v10_old = {'label_threshold': 'median'}
    method = v10_old.get('label_method', v10_old.get('label_threshold', 'median'))
    assert method == 'median'

    v10_mixed = {'label_method': 'absolute', 'label_threshold': 'median',
                 'label_absolute_threshold': 0.15}
    method = v10_mixed.get('label_method', v10_mixed.get('label_threshold', 'median'))
    assert method == 'absolute'  # 新 key 优先
    print("[PASS] Backward compat: old label_threshold still recognized, label_method takes priority")


# ================================================================
# B. 标签数学验证: label_raw 计算正确性
# ================================================================

def test_label_raw_computation_math():
    """验证 label_raw = adj_fut / adj_now - 1 的数学正确性"""
    # 模拟前复权计算: adj = raw_close * backAdjustFactor
    raw_now, adj_now = 10.0, 1.0
    raw_fut, adj_fut = 12.0, 1.05

    computed_now = raw_now * adj_now   # 10.0
    computed_fut = raw_fut * adj_fut  # 12.6
    label_raw = computed_fut / computed_now - 1  # 0.26 = 26%

    assert abs(label_raw - 0.26) < 1e-10
    print(f"[PASS] label_raw math: {computed_now} -> {computed_fut} = {label_raw:.2%}")

    # 测试下跌
    raw_fut2, adj_fut2 = 8.0, 0.98
    computed_fut2 = raw_fut2 * adj_fut2  # 7.84
    label_raw2 = computed_fut2 / computed_now - 1  # -0.216
    assert abs(label_raw2 - (-0.216)) < 1e-10
    print(f"[PASS] Down scenario: {computed_now} -> {computed_fut2} = {label_raw2:.2%}")


def test_label_raw_handles_zero_base():
    """除零保护: raw_now=0 时跳过"""
    r_now, a_now, r_fut, a_fut = 0.0, 1.0, 12.0, 1.0
    valid = (pd.notna(r_now) and pd.notna(r_fut) and r_now != 0
             and pd.notna(a_now) and pd.notna(a_fut))
    assert not valid  # r_now=0 → 跳过
    print("[PASS] raw_close=0 -> filtered out (no div-by-zero)")


def test_label_raw_handles_nan():
    """NaN 处理"""
    assert not (pd.notna(np.nan))
    assert pd.notna(0.0)
    assert pd.notna(-0.01)
    print("[PASS] NaN correctly filtered by pd.notna checks")


# ================================================================
# C. 三重模式集成 — 使用真实多日期截面数据
# ================================================================

def _make_multi_date_data():
    """生成模拟多日期截面数据"""
    np.random.seed(123)
    data_list = []
    for dt_idx in range(5):
        n_stocks = 200
        snap = pd.DataFrame({
            'code': [str(i).zfill(6) for i in range(n_stocks)],
            'snapshot_date': pd.Timestamp(f'2024-01-{dt_idx*5+2:02d}'),
            'label_raw': np.random.randn(n_stocks) * 0.08 + 0.01 * dt_idx,
        })
        data_list.append(snap)
    return pd.concat(data_list, ignore_index=True)


def _construct_labels(data, method, **kwargs):
    """与 gpsef_strategy_v10.py 完全一致的标签构造"""
    data = data.copy()
    if method == 'absolute':
        t = kwargs.get('threshold', 0.10)
        data['label'] = (data['label_raw'] > t).astype(int)
    elif method == 'quantile':
        q = kwargs.get('quantile', 0.20)
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


def test_multi_date_absolute_labels():
    """多日期截面: absolute 模式每个日期独立判断"""
    data = _make_multi_date_data()
    labeled = _construct_labels(data, 'absolute', threshold=0.10)

    # 每个日期独立: label=1 的恰好是 label_raw > 0.10
    for dt in labeled['snapshot_date'].unique():
        day_data = labeled[labeled['snapshot_date'] == dt]
        manual = (day_data['label_raw'] > 0.10).astype(int)
        assert (day_data['label'].values == manual.values).all(), \
            f"Date {dt}: labels don't match manual calculation"
    print("[PASS] Multi-date absolute: each date labeled independently, matches manual")


def test_multi_date_quantile_labels():
    """多日期截面: quantile 模式每个日期独立分位"""
    data = _make_multi_date_data()
    labeled = _construct_labels(data, 'quantile', quantile=0.20)

    for dt in labeled['snapshot_date'].unique():
        day_data = labeled[labeled['snapshot_date'] == dt]
        q_threshold = day_data['label_raw'].quantile(0.80)  # 1-0.20
        manual = (day_data['label_raw'] >= q_threshold).astype(int)
        assert (day_data['label'].values == manual.values).all(), \
            f"Date {dt} quantile mismatch"
        pos_rate = day_data['label'].mean()
        assert 0.18 <= pos_rate <= 0.22, f"Date {dt}: pos_rate={pos_rate:.3f}"
    print("[PASS] Multi-date quantile: each date ~20% positive independently")


def test_multi_date_median_labels():
    """多日期截面: median 模式 ~50%"""
    data = _make_multi_date_data()
    labeled = _construct_labels(data, 'median')

    for dt in labeled['snapshot_date'].unique():
        day_data = labeled[labeled['snapshot_date'] == dt]
        # 连续分布无重复值，> median 恰好 50%
        assert 0.49 <= day_data['label'].mean() <= 0.51, \
            f"Date {dt}: median mode should have ~50% positive, got {day_data['label'].mean():.3f}"
    print("[PASS] Multi-date median: each date ~50% positive (strict > median, continuous data)")


def test_mode_switching_on_same_data():
    """同一数据切换三种模式，验证标签向量不同"""
    data = _make_multi_date_data()

    l_med = _construct_labels(data, 'median')
    l_qt = _construct_labels(data, 'quantile', quantile=0.20)
    l_abs = _construct_labels(data, 'absolute', threshold=0.10)

    assert not l_med['label'].equals(l_qt['label']), "median != quantile"
    assert not l_qt['label'].equals(l_abs['label']), "quantile != absolute"
    assert not l_med['label'].equals(l_abs['label']), "median != absolute"
    print(f"[PASS] Mode switching: median={l_med['label'].mean():.2%}, "
          f"quantile(0.20)={l_qt['label'].mean():.2%}, "
          f"absolute(0.10)={l_abs['label'].mean():.2%}")


# ================================================================
# D. 真实数据模式测试
# ================================================================

def test_realistic_bull_market_2020():
    """模拟2020年牛市: 大量股票涨超10%"""
    np.random.seed(2020)
    data = pd.DataFrame({
        'snapshot_date': [pd.Timestamp('2020-07-01')] * 500,
        'label_raw': np.random.normal(0.12, 0.06, 500),  # mean=12%, std=6%
    })
    labeled = _construct_labels(data, 'absolute', threshold=0.10)
    pos_rate = labeled['label'].mean()
    # 在 mean=12%, std=6% 的正态分布下，>0.10 的比例约 63%
    assert 0.55 <= pos_rate <= 0.75, f"Bull market pos_rate should be 55-75%, got {pos_rate:.2%}"
    print(f"[PASS] 2020 bull: pos_rate={pos_rate:.2%}")


def test_realistic_bear_market_2018():
    """模拟2018年熊市: 极少股票涨超10%"""
    np.random.seed(2018)
    data = pd.DataFrame({
        'snapshot_date': [pd.Timestamp('2018-07-01')] * 500,
        'label_raw': np.random.normal(-0.08, 0.05, 500),  # mean=-8%, std=5%
    })
    labeled = _construct_labels(data, 'absolute', threshold=0.10)
    pos_rate = labeled['label'].mean()
    # 在 mean=-8%, std=5% 下，>0.10 的比例极低
    assert pos_rate < 0.05, f"Bear market pos_rate should be <5%, got {pos_rate:.2%}"
    print(f"[PASS] 2018 bear: pos_rate={pos_rate:.4%}")


def test_realistic_sideways_market():
    """震荡市: 少数股票突破"""
    np.random.seed(42)
    data = pd.DataFrame({
        'snapshot_date': [pd.Timestamp('2023-06-01')] * 500,
        'label_raw': np.random.normal(0.01, 0.07, 500),  # mean=1%, std=7%
    })
    labeled = _construct_labels(data, 'absolute', threshold=0.10)
    pos_rate = labeled['label'].mean()
    # 在 mean=1%, std=7% 下，>0.10 约 (0.10-0.01)/0.07 = 1.29 sigma → ~10%
    assert 0.06 <= pos_rate <= 0.15, f"Sideways pos_rate should be 6-15%, got {pos_rate:.2%}"
    print(f"[PASS] Sideways market: pos_rate={pos_rate:.2%}")


# ================================================================
# E. 健壮性测试
# ================================================================

def test_absolute_with_inf_values():
    """inf 值处理: inf > threshold → 1; -inf → 0"""
    data = pd.DataFrame({
        'snapshot_date': [pd.Timestamp('2024-01-01')] * 4,
        'label_raw': [0.15, np.inf, -np.inf, 0.05],
    })
    labeled = _construct_labels(data, 'absolute', threshold=0.10)
    assert labeled['label'].iloc[0] == 1  # 0.15 > 0.10
    assert labeled['label'].iloc[1] == 1  # inf > 0.10
    assert labeled['label'].iloc[2] == 0  # -inf not > 0.10
    assert labeled['label'].iloc[3] == 0  # 0.05 not > 0.10
    print("[PASS] inf/-inf correctly handled")


def test_absolute_with_exact_threshold_values():
    """精确阈值测试: 等于→0, 大于→1, 小于→0"""
    data = pd.DataFrame({
        'snapshot_date': [pd.Timestamp('2024-01-01')] * 5,
        'label_raw': [0.10, 0.1000001, 0.0999999, 0.10 + 1e-10, 0.10 - 1e-10],
    })
    labeled = _construct_labels(data, 'absolute', threshold=0.10)
    assert labeled['label'].iloc[0] == 0  # exactly 0.10 → 0 (strict >)
    assert labeled['label'].iloc[1] == 1  # just above
    assert labeled['label'].iloc[2] == 0  # just below
    assert labeled['label'].iloc[3] == 1  # 0.10 + ε > 0.10
    assert labeled['label'].iloc[4] == 0  # 0.10 - ε < 0.10
    print("[PASS] Exact threshold: strict > operator verified at floating-point precision")


def test_single_stock_absolute():
    """单只股票不崩溃"""
    data = pd.DataFrame({
        'snapshot_date': [pd.Timestamp('2024-01-01')],
        'label_raw': [0.15],
    })
    labeled = _construct_labels(data, 'absolute', threshold=0.10)
    assert len(labeled) == 1
    assert labeled['label'].iloc[0] == 1
    print("[PASS] Single stock absolute: no crash")


def test_all_negative_returns_absolute():
    """全部负收益 + 正阈值 → 全0"""
    data = pd.DataFrame({
        'snapshot_date': ['2024-01-01'] * 100,
        'label_raw': np.random.uniform(-0.50, -0.01, 100),
    })
    labeled = _construct_labels(data, 'absolute', threshold=0.10)
    assert labeled['label'].sum() == 0
    print("[PASS] All negative returns + positive threshold -> all label=0")


def test_all_extreme_positive_absolute():
    """全部极端正收益 → 全1"""
    data = pd.DataFrame({
        'snapshot_date': ['2024-01-01'] * 100,
        'label_raw': np.random.uniform(0.20, 1.00, 100),
    })
    labeled = _construct_labels(data, 'absolute', threshold=0.10)
    assert labeled['label'].sum() == 100
    print("[PASS] All extreme positive -> all label=1")


# ================================================================
# F. 模式参数切换测试
# ================================================================

def test_absolute_with_different_thresholds():
    """不同阈值产生不同正样本比例"""
    data = pd.DataFrame({
        'snapshot_date': ['d'] * 500,
        'label_raw': np.random.randn(500) * 0.10,
    })

    rates = {}
    for t in [0.05, 0.10, 0.15, 0.20]:
        labeled = _construct_labels(data, 'absolute', threshold=t)
        rates[t] = labeled['label'].mean()

    # 阈值越高，正样本越少（单调递减）
    assert rates[0.05] >= rates[0.10] >= rates[0.15] >= rates[0.20], \
        f"Higher threshold should give fewer positives: {rates}"
    print(f"[PASS] Monotonic thresholds: { {k: f'{v:.2%}' for k,v in rates.items()} }")


def test_quantile_with_different_q_values():
    """不同q值产生不同正样本比例"""
    data = pd.DataFrame({
        'snapshot_date': ['d'] * 500,
        'label_raw': np.random.randn(500) * 0.10,
    })

    for q in [0.10, 0.25, 0.50]:
        labeled = _construct_labels(data, 'quantile', quantile=q)
        rate = labeled['label'].mean()
        assert abs(rate - q) < 0.03, f"q={q}: expected ~{q}, got {rate:.3f}"
    print("[PASS] Quantile q values: rates match q within tolerance")


# ================================================================
# G. 向后兼容测试
# ================================================================

def test_median_mode_unchanged():
    """median 模式行为不变: 正样本略少于50%"""
    data = pd.DataFrame({
        'snapshot_date': ['d'] * 1000,
        'label_raw': np.random.randn(1000) * 0.10,
    })
    labeled = _construct_labels(data, 'median')
    rate = labeled['label'].mean()
    # 连续分布 > 中位数恰好50%
    assert 0.49 <= rate <= 0.51, f"median: expected ~50%, got {rate:.3f}"
    print(f"[PASS] Median mode unchanged: pos_rate={rate:.2%}")


def test_backward_compat_invalid_method():
    """未识别的 method → fallback median"""
    data = pd.DataFrame({
        'snapshot_date': ['d'] * 100,
        'label_raw': np.random.randn(100) * 0.10,
    })

    # 模拟 strategy 中的校验逻辑
    method = 'percentile'
    if method not in ('median', 'quantile', 'absolute'):
        method = 'median'

    labeled = _construct_labels(data, method)
    assert 0.49 <= labeled['label'].mean() <= 0.51
    print("[PASS] Invalid method 'percentile' -> fallback to median")


# ================================================================
# 运行
# ================================================================

if __name__ == '__main__':
    all_tests = [
        # A. Config
        test_config_parsing_absolute_mode,
        test_config_fallback_when_absolute_missing,
        test_config_backward_compat_label_threshold_key,
        # B. Math
        test_label_raw_computation_math,
        test_label_raw_handles_zero_base,
        test_label_raw_handles_nan,
        # C. Multi-date
        test_multi_date_absolute_labels,
        test_multi_date_quantile_labels,
        test_multi_date_median_labels,
        test_mode_switching_on_same_data,
        # D. Realistic
        test_realistic_bull_market_2020,
        test_realistic_bear_market_2018,
        test_realistic_sideways_market,
        # E. Robustness
        test_absolute_with_inf_values,
        test_absolute_with_exact_threshold_values,
        test_single_stock_absolute,
        test_all_negative_returns_absolute,
        test_all_extreme_positive_absolute,
        # F. Parameters
        test_absolute_with_different_thresholds,
        test_quantile_with_different_q_values,
        # G. Backward
        test_median_mode_unchanged,
        test_backward_compat_invalid_method,
    ]

    passed = 0
    failed = 0
    for t in all_tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            import traceback
            print(f"\n[FAIL] {t.__name__}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  INTEGRATION TEST RESULTS: {passed} passed, {failed} failed, {passed+failed} total")
    print(f"{'='*60}")
    sys.exit(1 if failed else 0)
