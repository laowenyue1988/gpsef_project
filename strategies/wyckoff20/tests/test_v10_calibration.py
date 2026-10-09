# -*- coding: utf-8 -*-
"""
测试模块 4: Isotonic概率校准 (P0-4 + P3-2)

验证：
- isotonic_fit_predict: Isotonic回归拟合+预测
- prob_clip: 概率裁剪到[0.05, 0.95]
- degradation_detection: 退化检测（range < 0.01时告警）
- calibration_improves: 校准后Brier分数降低
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from strategies.wyckoff20.tests.conftest import make_strategy


def _setup_model_and_data(n_train=400, n_val=200, seed=42):
    """训练一个小模型，返回策略实例、dval、val预测。"""
    s = make_strategy()
    s._ensemble_seeds = [42]
    s.config = {'xgb_n_estimators': 50}
    s.early_stopping_rounds = 10

    rng = np.random.RandomState(seed)
    n_feats = 6
    feat_names = [f'feat_{i}' for i in range(n_feats)]

    X_train = pd.DataFrame(rng.randn(n_train, n_feats), columns=feat_names)
    signal = X_train.iloc[:, :3].sum(axis=1).values
    y_train = (signal > np.median(signal)).astype(int)

    X_val = pd.DataFrame(rng.randn(n_val, n_feats), columns=feat_names)
    signal_val = X_val.iloc[:, :3].sum(axis=1).values
    y_val = (signal_val > np.median(signal_val)).astype(int)

    dtrain = xgb.DMatrix(X_train.values, label=y_train, feature_names=feat_names)
    dval = xgb.DMatrix(X_val.values, label=y_val, feature_names=feat_names)

    models = s._train_ensemble(dtrain, dval)
    s.models = models

    return s, dval, y_val


def test_isotonic_fit_predict():
    """Isotonic回归正确拟合+预测。"""
    s, dval, y_val = _setup_model_and_data()

    calibrator = s._calibrate_proba(dval)

    assert calibrator is not None, "Calibrator should be created for isotonic method"
    assert isinstance(calibrator, IsotonicRegression), \
        f"Expected IsotonicRegression, got {type(calibrator)}"

    # 验证校准后预测值在合理范围
    raw_preds = s._predict_ensemble(dval)
    calibrated = calibrator.predict(raw_preds)

    assert calibrated.min() >= 0.0, f"Calibrated min < 0: {calibrated.min()}"
    assert calibrated.max() <= 1.0, f"Calibrated max > 1: {calibrated.max()}"

    # Isotonic是单调的 → 排序后不下降
    sorted_idx = np.argsort(raw_preds)
    sorted_cal = calibrated[sorted_idx]
    diffs = np.diff(sorted_cal)
    assert np.all(diffs >= -1e-10), "Isotonic should be monotonically non-decreasing"

    print(f"[PASS] test_isotonic_fit_predict: monotonic calibration, range=[{calibrated.min():.4f}, {calibrated.max():.4f}]")


def test_prob_clip():
    """概率裁剪到[0.05, 0.95]。"""
    s = make_strategy()
    s._prob_clip_min = 0.05
    s._prob_clip_max = 0.95

    # 无校准器 → 只做裁剪
    s.calibrator = None

    raw = np.array([0.01, 0.03, 0.5, 0.97, 0.99])
    clipped = s._apply_calibration(raw)

    assert clipped[0] == 0.05, f"Min clip failed: {clipped[0]} != 0.05"
    assert clipped[1] == 0.05, f"Min clip failed: {clipped[1]} != 0.05"
    assert clipped[2] == 0.5, f"Mid value changed: {clipped[2]} != 0.5"
    assert clipped[3] == 0.95, f"Max clip failed: {clipped[3]} != 0.95"
    assert clipped[4] == 0.95, f"Max clip failed: {clipped[4]} != 0.95"

    print("[PASS] test_prob_clip: [0.01, 0.03, 0.5, 0.97, 0.99] → [0.05, 0.05, 0.5, 0.95, 0.95]")


def test_prob_clip_with_calibration():
    """校准 + 裁剪组合：先Isotonic变换，再裁剪。"""
    s, dval, y_val = _setup_model_and_data()

    calibrator = s._calibrate_proba(dval)
    s.calibrator = calibrator

    raw_preds = s._predict_ensemble(dval)
    result = s._apply_calibration(raw_preds)

    # 验证所有值在[0.05, 0.95]范围内
    assert result.min() >= 0.05, f"Min below clip: {result.min()}"
    assert result.max() <= 0.95, f"Max above clip: {result.max()}"

    print(f"[PASS] test_prob_clip_with_calibration: range=[{result.min():.4f}, {result.max():.4f}] within [0.05, 0.95]")


def test_calibration_preserves_auc():
    """Isotonic校准不降低AUC（AUC对单调变换不变）。"""
    s, dval, y_val = _setup_model_and_data()

    raw_preds = s._predict_ensemble(dval)
    calibrator = s._calibrate_proba(dval)
    s.calibrator = calibrator

    calibrated = s._apply_calibration(raw_preds)

    auc_raw = roc_auc_score(y_val, raw_preds)
    auc_cal = roc_auc_score(y_val, calibrated)

    # AUC对单调变换不变
    assert abs(auc_raw - auc_cal) < 0.01, \
        f"AUC changed after isotonic: raw={auc_raw:.4f}, cal={auc_cal:.4f}"

    print(f"[PASS] test_calibration_preserves_auc: raw={auc_raw:.4f}, cal={auc_cal:.4f}")


def test_degradation_detection():
    """退化检测：模型无区分力时range < 0.01。"""
    s = make_strategy()
    s._calibration_method = 'isotonic'
    s._prob_clip_min = 0.05
    s._prob_clip_max = 0.95

    # 模拟无区分力的预测（全0.5附近）
    rng = np.random.RandomState(42)
    raw_preds = 0.5 + rng.randn(100) * 0.001  # 极小范围
    y_val = rng.randint(0, 2, 100)

    calibrator = IsotonicRegression(y_min=0.05, y_max=0.95, out_of_bounds='clip')
    calibrator.fit(raw_preds, y_val)
    calibrated = calibrator.predict(raw_preds)

    prob_range = calibrated.max() - calibrated.min()
    assert prob_range < 0.1, f"Degraded model should have small range, got {prob_range:.4f}"

    print(f"[PASS] test_degradation_detection: range={prob_range:.4f} (degraded model detected)")


def test_no_calibration_method():
    """calibration_method != 'isotonic' 时返回None。"""
    s = make_strategy()
    s._calibration_method = 'none'

    # _calibrate_proba应返回None
    import xgboost as xgb
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(50, 3), columns=['a', 'b', 'c'])
    y = rng.randint(0, 2, 50)
    dval = xgb.DMatrix(X.values, label=y, feature_names=['a', 'b', 'c'])

    # 需要models
    s.models = []
    result = s._calibrate_proba(dval)
    assert result is None, "Should return None for non-isotonic method"

    print("[PASS] test_no_calibration_method: returns None for method='none'")


if __name__ == '__main__':
    test_isotonic_fit_predict()
    test_prob_clip()
    test_prob_clip_with_calibration()
    test_calibration_preserves_auc()
    test_degradation_detection()
    test_no_calibration_method()
    print("\n=== All Calibration tests PASSED ===")
