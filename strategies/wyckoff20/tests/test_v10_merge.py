# -*- coding: utf-8 -*-
"""
测试模块 7: V5合并完整性验证

验证：
- no_v5_import: V10文件不导入V5策略
- no_v5_inheritance: V10不继承V5类
- all_factors_present: 40个因子全部存在
- all_methods_present: V5核心方法已迁移到V10
- compute_sample_weights: 序贯唯一性权重函数存在且可用
- xgb_wrappers: XGBRegressorWrapper/XGBClassifierWrapper存在
"""
import sys
import os
import ast
import inspect
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from strategies.wyckoff20.gpsef_strategy_v10 import (
    WyckoffV10Strategy,
    FactorNeutralizer,
    XGBRegressorWrapper,
    XGBClassifierWrapper,
    compute_sample_weights,
)
from strategies.wyckoff20.tests.conftest import make_strategy


def test_no_v5_import():
    """V10文件源码不导入V5策略。"""
    v10_path = os.path.join(os.path.dirname(__file__), '..', 'gpsef_strategy_v10.py')
    with open(v10_path, 'r', encoding='utf-8') as f:
        source = f.read()

    # 不应有 import gpsef_strategy 或 from gpsef_strategy import
    v5_import_patterns = [
        'from strategies.wyckoff20.gpsef_strategy import',
        'from strategies.wyckoff20.gpsef_strategy_v5',
        'import gpsef_strategy_v5',
        'gpsef_strategy_v5_backup',
        'Wyckoff20GPSEFStrategyV5',
    ]

    for pattern in v5_import_patterns:
        assert pattern not in source, f"V10 should not reference V5: found '{pattern}'"

    print("[PASS] test_no_v5_import: no V5 imports in V10 source")


def test_no_v5_inheritance():
    """V10策略类不继承V5。"""
    bases = [base.__name__ for base in WyckoffV10Strategy.__bases__]
    assert 'Wyckoff20GPSEFStrategyV5' not in bases, f"V10 should not inherit V5, bases: {bases}"
    assert 'IStrategy' in bases, f"V10 should inherit IStrategy, bases: {bases}"

    print(f"[PASS] test_no_v5_inheritance: bases = {bases}")


def test_all_factors_present():
    """40个因子全部在ALL_FACTORS列表中。"""
    expected_factors = [
        'supply_demand_momentum', 'price_volume_divergence', 'no_supply',
        'volume_profile', 'micro_order_flow', 'market_structure',
        'accumulation_power', 'breakout_validity', 'vwap_anchor', 'delta_momentum',
        'effort_result', 'closing_position', 'volume_concentration',
        'spread_ratio', 'up_volume_ratio', 'demand_supply_ratio',
        'turnover_momentum', 'price_position', 'volume_trend', 'smart_money_flow',
        'spring_quality', 'jump_creek', 'stage_continuous', 'ut_utad', 'sos_strength',
        'vpoc_momentum', 'va_convergence', 'vol_skewness', 'lvn_breakout', 'vol_entropy',
        'climax', 'no_demand', 'stopping_vol', 'weis_wave', 'big_flow', 'imbalance',
        'accumulation_score', 'vwap_deviation', 'vol_pulse', 'hvn_support',
    ]

    actual_factors = WyckoffV10Strategy.ALL_FACTORS

    assert len(actual_factors) == 40, f"Expected 40 factors, got {len(actual_factors)}"
    for f in expected_factors:
        assert f in actual_factors, f"Factor '{f}' missing from ALL_FACTORS"

    print(f"[PASS] test_all_factors_present: 40 factors verified")


def test_all_methods_present():
    """V5核心方法已迁移到V10。"""
    required_methods = [
        # 数据加载
        '_preload_daily_data',
        '_get_trading_day_offset',
        '_get_previous_snapshot_date',
        '_load_snapshot',
        '_get_close_prices',
        '_get_label_components',
        '_get_snapshot_dates',
        # 中性化
        '_build_neutral_X',
        '_neutralize',
        '_apply_score_neutralization',
        # 特征工程
        '_prepare_features',
        '_build_feature_matrix',
        # 特征选择
        '_calc_psi',
        '_sanitize_features',
        '_compute_ic_vectorized',
        '_compute_mda_oos',
        '_select_features',  # 保留作为工具方法, 主流程不再调用
        '_prefilter_features',  # 替代原_select_features_cv
        # V10新增
        '_purged_kfold_cv',
        '_train_ensemble',
        '_predict_ensemble',
        '_calibrate_proba',
        '_apply_calibration',
        '_auc_gate',
        '_neutralize_full_market',
        # 训练
        '_build_training_dataset',
        '_calc_model_performance',
        # 策略接口
        'initialize',
        'before_market_open',
        'handle_bar',
        'recommend',
    ]

    missing = []
    for method in required_methods:
        if not hasattr(WyckoffV10Strategy, method):
            missing.append(method)

    assert len(missing) == 0, f"Missing methods: {missing}"

    print(f"[PASS] test_all_methods_present: {len(required_methods)} methods verified")


def test_compute_sample_weights():
    """compute_sample_weights函数存在且可用。"""
    from datetime import datetime, timedelta

    # 创建合成训练数据
    base_date = datetime(2025, 1, 1)
    calendar = [base_date + timedelta(days=i) for i in range(100)]

    rng = np.random.RandomState(42)
    records = []
    for d in range(20):
        dt = base_date + timedelta(days=d * 5)
        for s in range(50):
            records.append({
                'snapshot_date': dt,
                'code': f'sh.{600000+s:06d}',
            })

    train_data = pd.DataFrame(records)

    weights = compute_sample_weights(train_data, calendar, max_hold_days=20)

    assert len(weights) == len(train_data), f"Weights length mismatch: {len(weights)} vs {len(train_data)}"
    assert np.all(weights > 0), "All weights should be positive"
    assert abs(weights.mean() - 1.0) < 0.01, f"Mean weight should be ~1.0, got {weights.mean()}"

    print(f"[PASS] test_compute_sample_weights: {len(weights)} weights, mean={weights.mean():.4f}")


def test_xgb_wrappers():
    """XGBRegressorWrapper/XGBClassifierWrapper类存在。"""
    assert hasattr(XGBRegressorWrapper, 'predict'), "XGBRegressorWrapper should have predict()"
    assert hasattr(XGBRegressorWrapper, 'fit'), "XGBRegressorWrapper should have fit()"

    assert hasattr(XGBClassifierWrapper, 'predict'), "XGBClassifierWrapper should have predict()"
    assert hasattr(XGBClassifierWrapper, 'predict_proba'), "XGBClassifierWrapper should have predict_proba()"
    assert hasattr(XGBClassifierWrapper, 'fit'), "XGBClassifierWrapper should have fit()"

    print("[PASS] test_xgb_wrappers: both wrapper classes present with correct methods")


def test_ic_gate_removed_from_source():
    """V10源码中不包含IC门控相关代码。"""
    v10_path = os.path.join(os.path.dirname(__file__), '..', 'gpsef_strategy_v10.py')
    with open(v10_path, 'r', encoding='utf-8') as f:
        source = f.read()

    # 不应有IC门控相关代码
    ic_gate_patterns = [
        'val_ic_buy_threshold',
        'val_ic_sell_threshold',
        'val_ic_strong_threshold',
        '_ic_gate',
        'ic_gate_mode',
    ]

    for pattern in ic_gate_patterns:
        assert pattern not in source, f"IC gate pattern '{pattern}' found in V10 source"

    print("[PASS] test_ic_gate_removed_from_source: no IC gate code in V10")


def test_config_ic_gate_removed():
    """config_v10.yaml中不包含IC门控参数。"""
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config_v10.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    flat_keys = set()

    def flatten(d, prefix=""):
        if isinstance(d, dict):
            for k, v in d.items():
                full_key = f"{prefix}.{k}" if prefix else k
                flat_keys.add(full_key)
                if isinstance(v, dict):
                    flatten(v, full_key)

    flatten(config)

    # 搜索 IC gate 残留键: 含 'ic' + ('threshold'/'gate'), 且排除 SHAP 段的 monotonicity_threshold
    # 原因: flatten 保留层级前缀 (如 shap.monotonicity_threshold),
    #       'ic' 出现在 'monoton**ic**ity' 中 → false positive, 需要排除
    ic_gate_keys = [k for k in flat_keys
                    if ('ic' in k.lower() and ('threshold' in k.lower() or 'gate' in k.lower()))
                    and 'shap.' not in k.lower()]
    assert len(ic_gate_keys) == 0, f"IC gate keys in config: {ic_gate_keys}"

    print("[PASS] test_config_ic_gate_removed: no IC gate params in config")


if __name__ == '__main__':
    test_no_v5_import()
    test_no_v5_inheritance()
    test_all_factors_present()
    test_all_methods_present()
    test_compute_sample_weights()
    test_xgb_wrappers()
    test_ic_gate_removed_from_source()
    test_config_ic_gate_removed()
    print("\n=== All V5 Merge tests PASSED ===")
