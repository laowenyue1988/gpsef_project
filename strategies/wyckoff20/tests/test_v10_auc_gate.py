# -*- coding: utf-8 -*-
"""
测试模块 5: AUC门控 (P0-2: 替换IC门控, 对齐Alpha191)

验证：
- full_mode: AUC >= 0.55 → full, 1.0x仓位
- half_mode: 0.50 <= AUC < 0.55 → half, 0.5x仓位
- empty_mode: AUC < 0.50 → empty, 0.0x仓位（空仓）
- gate_boundaries: 精确边界值
- no_ic_gate: 确认IC门控参数已移除
- config_check: 配置文件参数对齐Alpha191
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from strategies.wyckoff20.tests.conftest import make_strategy


def test_full_mode():
    """AUC >= 0.55 → full, 1.0x仓位。"""
    s = make_strategy()
    mode, scale = s._auc_gate(0.60)
    assert mode == "full", f"AUC=0.60 should be full, got {mode}"
    assert scale == 1.0, f"Full scale should be 1.0, got {scale}"

    # 边界值
    mode, scale = s._auc_gate(0.55)
    assert mode == "full", f"AUC=0.55 (boundary) should be full, got {mode}"
    assert scale == 1.0

    # 极高AUC
    mode, scale = s._auc_gate(0.99)
    assert mode == "full"
    assert scale == 1.0

    print("[PASS] test_full_mode: AUC>=0.55 → full, 1.0x")


def test_half_mode():
    """0.50 <= AUC < 0.55 → half, 0.5x仓位。"""
    s = make_strategy()

    mode, scale = s._auc_gate(0.52)
    assert mode == "half", f"AUC=0.52 should be half, got {mode}"
    assert scale == 0.5, f"Half scale should be 0.5, got {scale}"

    # 边界值
    mode, scale = s._auc_gate(0.50)
    assert mode == "half", f"AUC=0.50 (boundary) should be half, got {mode}"
    assert scale == 0.5

    # 上边界（0.549999...）
    mode, scale = s._auc_gate(0.549)
    assert mode == "half", f"AUC=0.549 should be half, got {mode}"
    assert scale == 0.5

    print("[PASS] test_half_mode: 0.50<=AUC<0.55 → half, 0.5x")


def test_empty_mode():
    """AUC < 0.50 → empty, 0.0x仓位（空仓, 对齐Alpha191）。"""
    s = make_strategy()

    mode, scale = s._auc_gate(0.45)
    assert mode == "empty", f"AUC=0.45 should be empty, got {mode}"
    assert scale == 0.0, f"Empty scale should be 0.0, got {scale}"

    # 极低AUC
    mode, scale = s._auc_gate(0.30)
    assert mode == "empty"
    assert scale == 0.0

    # 接近0
    mode, scale = s._auc_gate(0.01)
    assert mode == "empty"
    assert scale == 0.0

    print("[PASS] test_empty_mode: AUC<0.50 → empty, 0.0x")


def test_gate_boundaries():
    """精确边界值测试。"""
    s = make_strategy()

    # 0.55 → full (>=)
    assert s._auc_gate(0.55)[0] == "full"
    # 0.549999 → half
    assert s._auc_gate(0.5499)[0] == "half"
    # 0.50 → half (>=)
    assert s._auc_gate(0.50)[0] == "half"
    # 0.499999 → empty
    assert s._auc_gate(0.4999)[0] == "empty"

    print("[PASS] test_gate_boundaries: 0.55/0.50 boundaries correct")


def test_no_ic_gate_params():
    """确认策略实例没有IC门控相关属性。"""
    s = make_strategy()

    # IC门控相关属性不应存在
    ic_attrs = ['_val_ic_buy_threshold', '_val_ic_sell_threshold',
                '_val_ic_strong_threshold', '_ic_gate_threshold']

    for attr in ic_attrs:
        assert not hasattr(s, attr), f"IC gate attribute {attr} should not exist"

    print("[PASS] test_no_ic_gate_params: IC gate attributes removed")


def test_config_aligned():
    """确认config_v10.yaml中AUC门控参数对齐Alpha191。"""
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config_v10.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 确认旧参数名已移除
    assert 'auc_gate_normal' not in config, "auc_gate_normal should be renamed to auc_gate_full"
    assert 'auc_gate_reduced' not in config, "auc_gate_reduced should be renamed to auc_gate_min"

    # 确认新参数名存在（对齐Alpha191: 0.48/0.52 离散半仓）
    assert 'auc_gate_enabled' in config, "auc_gate_enabled missing from config"
    assert 'auc_gate_min' in config, "auc_gate_min missing from config"
    assert 'auc_gate_full' in config, "auc_gate_full missing from config"
    assert config['auc_gate_enabled'] == True
    assert config['auc_gate_min'] == 0.48, \
        f"auc_gate_min should be 0.48 (离散半仓下界), got {config['auc_gate_min']}"
    assert config['auc_gate_full'] == 0.52, \
        f"auc_gate_full should be 0.52 (离散半仓上界), got {config['auc_gate_full']}"

    print("[PASS] test_config_aligned: AUC gate config aligned with Alpha191 (0.48/0.52)")


def test_custom_thresholds():
    """自定义AUC门控阈值。"""
    s = make_strategy()
    s._auc_gate_full = 0.60
    s._auc_gate_min = 0.55

    # AUC=0.58 → half (0.55<=0.58<0.60)
    mode, scale = s._auc_gate(0.58)
    assert mode == "half", f"Custom threshold: AUC=0.58 should be half"
    assert scale == 0.5

    # AUC=0.62 → full (>=0.60)
    mode, scale = s._auc_gate(0.62)
    assert mode == "full"
    assert scale == 1.0

    # AUC=0.54 → empty (<0.55)
    mode, scale = s._auc_gate(0.54)
    assert mode == "empty"
    assert scale == 0.0

    print("[PASS] test_custom_thresholds: custom 0.60/0.55 works")


if __name__ == '__main__':
    test_full_mode()
    test_half_mode()
    test_empty_mode()
    test_gate_boundaries()
    test_no_ic_gate_params()
    test_config_aligned()
    test_custom_thresholds()
    print("\n=== All AUC Gate tests PASSED ===")
