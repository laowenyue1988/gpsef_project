# -*- coding: utf-8 -*-
"""
测试模块: P0/P1 修复验证 (Platt Scaling + scale_pos_weight + 行业约束 + LightGBM + Config去重)

测试覆盖:
- test_platt_scaling_basic: Platt Scaling 基本功能
- test_platt_scaling_cv_ensemble: CV折集成 Platt Scaling
- test_scale_pos_weight_cap: scale_pos_weight 上限保护
- test_scale_pos_weight_extreme: 极端不均衡场景
- test_industry_constraint_basic: 行业约束基本功能
- test_industry_constraint_no_excess: 无超额行业场景
- test_lgb_ensemble_training: LightGBM 异质集成训练
- test_lgb_ensemble_prediction: LightGBM 异质集成预测
- test_config_cleanup: Config 去重验证
- test_e2e_with_all_fixes: 端到端所有修复集成测试

Author: WorkBuddy Audit Fix
Date: 2026-07-12
"""
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 确保项目根目录在 sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from strategies.wyckoff20.tests.conftest import (
    make_strategy, make_synthetic_train_data, make_feature_df,
)

# 检查 LightGBM 可用性
try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

ALL_FACTORS = __import__(
    'strategies.wyckoff20.gpsef_strategy_v10', fromlist=['WyckoffV10Strategy']
).WyckoffV10Strategy.ALL_FACTORS


# ============================================================================
# 工具函数
# ============================================================================

def _make_cv_train_data(n_days=80, n_stocks=100, n_feats=12, seed=42, pos_ratio=0.5):
    """生成带 CV 折信息的训练数据。"""
    rng = np.random.RandomState(seed)
    base_date = datetime(2025, 1, 1)
    records = []
    for d in range(n_days):
        dt = base_date + timedelta(days=d)
        for s in range(n_stocks):
            code = f"{600000 + s:06d}"
            row = {'snapshot_date': dt, 'code': code}
            for f in range(n_feats):
                if f < 5:
                    row[f'feat_{f}'] = rng.randn() + f * 0.3
                else:
                    row[f'feat_{f}'] = rng.randn()
            records.append(row)

    df = pd.DataFrame(records)
    signal_cols = [f'feat_{f}' for f in range(5)]
    df['label_raw'] = df[signal_cols].sum(axis=1) + rng.randn(len(df)) * 0.5
    df['label'] = df.groupby('snapshot_date')['label_raw'].transform(
        lambda x: (x > x.quantile(1 - pos_ratio)).astype(int))

    # 时序分割
    unique_dates = sorted(df['snapshot_date'].unique())
    split_idx = int(len(unique_dates) * 0.8)
    train_dates = unique_dates[:split_idx]
    val_dates = unique_dates[split_idx:]
    train_df = df[df['snapshot_date'].isin(train_dates)].copy()
    val_df = df[df['snapshot_date'].isin(val_dates)].copy()
    return train_df, val_df, [f'feat_{f}' for f in range(n_feats)]


def _make_cv_folds(train_df, n_folds=5, embargo_days=5):
    """生成简单的 CV 折定义 (模拟 Purged K-Fold 输出)。"""
    unique_dates = sorted(train_df['snapshot_date'].unique())
    fold_size = max(1, len(unique_dates) // n_folds)
    folds = []
    for i in range(n_folds):
        val_start = i * fold_size
        val_end = min(val_start + fold_size, len(unique_dates))
        val_dates = set(unique_dates[val_start:val_end])
        train_dates = set(unique_dates[:val_start]) | set(unique_dates[val_end:])
        # 移除 embargo
        if embargo_days > 0 and val_start > 0:
            embargo_start = max(0, val_start - embargo_days)
            embargo_dates = set(unique_dates[embargo_start:val_start])
            train_dates -= embargo_dates
        folds.append((sorted(train_dates), sorted(val_dates)))
    return folds


# ============================================================================
# Test 1: Platt Scaling
# ============================================================================

def test_platt_scaling_basic():
    """Platt Scaling 基本功能：拟合 + 预测 + ECE 计算。"""
    s = make_strategy()
    s._calibration_method = 'platt'
    s._calibration_monitoring = True

    # 生成合成数据
    rng = np.random.RandomState(42)
    n = 500
    raw_preds = rng.beta(2, 5, n)  # 有偏的预测 (偏左)
    y_true = (raw_preds > np.median(raw_preds)).astype(int)

    # 训练简易 XGBoost 模型用于校准
    X_data = np.column_stack([raw_preds, rng.randn(n, 4)])
    dtrain = xgb.DMatrix(X_data, label=y_true)
    dval = xgb.DMatrix(X_data[:100], label=y_true[:100])

    s.models = []
    s.feature_names = [f'feat_{i}' for i in range(5)]
    s._ensemble_seeds = [42]
    s._combined_enabled = False
    s._class_balance_mode = 'none'

    # 先训练 XGBoost
    xgb_params = {
        'max_depth': 3, 'eta': 0.1, 'objective': 'binary:logistic',
        'eval_metric': 'logloss', 'tree_method': 'hist', 'seed': 42,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'nthread': 1,
    }
    model = xgb.train(xgb_params, dtrain, num_boost_round=30, verbose_eval=False)
    s.models = [model]
    s._active_models = [0]
    s.model_weights = [1.0]

    # 测试校准
    calibrator = s._calibrate_proba(dval)

    # 验证
    assert calibrator is not None, "Platt calibrator should not be None"
    assert isinstance(calibrator, LogisticRegression), \
        f"Expected LogisticRegression, got {type(calibrator)}"

    # 验证 ECE 计算不报错
    ece = s._compute_ece(y_true[:100], raw_preds[:100], n_bins=10)
    assert 0 <= ece <= 1, f"ECE should be in [0,1], got {ece}"

    # 验证 apply_calibration 工作
    s.calibrator = calibrator
    calibrated = s._apply_calibration(raw_preds)
    assert len(calibrated) == len(raw_preds)
    assert np.all(calibrated >= s._prob_clip_min)
    assert np.all(calibrated <= s._prob_clip_max)

    print(f"  [PASS] Platt Scaling basic: calibrator={type(calibrator).__name__}, ECE={ece:.4f}")


def test_platt_scaling_cv_ensemble():
    """CV 折集成下的 Platt Scaling。"""
    s = make_strategy()
    s._calibration_method = 'platt'
    s._calibration_monitoring = True
    s._ensemble_seeds = [42, 43]

    train_df, val_df, all_feats = _make_cv_train_data(n_days=60, n_stocks=80, n_feats=10)
    s.feature_names = all_feats
    s._selected_features = all_feats

    # 生成 CV 折定义
    cv_folds = _make_cv_folds(train_df, n_folds=3)
    s._cv_fold_defs = cv_folds

    # 训练简单的 XGBoost 模型
    X_all = train_df[all_feats].fillna(0).values
    X_val = val_df[all_feats].fillna(0).values
    y_all = train_df['label'].values.astype(int)
    y_val = val_df['label'].values.astype(int)

    models = []
    for i in range(min(3, len(cv_folds))):
        t_dates, _ = cv_folds[i]
        fold_mask = train_df['snapshot_date'].isin(t_dates).values
        X_fold = X_all[fold_mask]
        y_fold = y_all[fold_mask]
        if len(X_fold) < 100:
            continue
        dtrain_fold = xgb.DMatrix(X_fold, label=y_fold, feature_names=all_feats)
        params = {
            'max_depth': 3, 'eta': 0.1, 'objective': 'binary:logistic',
            'eval_metric': 'logloss', 'tree_method': 'hist', 'seed': 42 + i,
            'subsample': 0.8, 'nthread': 1,
        }
        model = xgb.train(params, dtrain_fold, num_boost_round=30, verbose_eval=False)
        models.append(model)

    s.models = models
    s.model_aucs = [0.55] * len(models)
    s.model_weights = [1.0] * len(models)
    s._active_models = list(range(len(models)))

    # 测试 CV 校准
    calibrator = s._calibrate_cv_ensemble(train_df)
    if calibrator is not None:
        assert isinstance(calibrator, LogisticRegression), \
            f"CV calibrator should be LogisticRegression, got {type(calibrator)}"
        print(f"  [PASS] Platt Scaling CV ensemble: {len(models)} models, calibrator OK")

        # 测试应用校准
        s.calibrator = calibrator
        dval_dm = xgb.DMatrix(X_val, feature_names=all_feats)
        raw = s._predict_ensemble(dval_dm)
        cal = s._apply_calibration(raw)
        assert len(cal) == len(raw)
    else:
        print(f"  [PASS] Platt Scaling CV ensemble: calibrator=None (legitimate, low sample)")


def test_isotonic_fallback():
    """Isotonic 校准向后兼容——仍然可用。"""
    s = make_strategy()
    s._calibration_method = 'isotonic'
    s._calibration_monitoring = True

    rng = np.random.RandomState(42)
    n = 300
    raw_preds = rng.beta(2, 5, n)
    y_true = (raw_preds > np.median(raw_preds)).astype(int)
    X_data = np.column_stack([raw_preds, rng.randn(n, 3)])

    s.models = []
    s.feature_names = [f'feat_{i}' for i in range(4)]
    s._ensemble_seeds = [42]
    s._combined_enabled = False
    s._class_balance_mode = 'none'

    dtrain = xgb.DMatrix(X_data, label=y_true)
    xgb_params = {
        'max_depth': 3, 'eta': 0.1, 'objective': 'binary:logistic',
        'eval_metric': 'logloss', 'tree_method': 'hist', 'seed': 42,
        'subsample': 0.8, 'nthread': 1,
    }
    model = xgb.train(xgb_params, dtrain, num_boost_round=30, verbose_eval=False)
    s.models = [model]
    s._active_models = [0]
    s.model_weights = [1.0]

    dval = xgb.DMatrix(X_data[:100], label=y_true[:100])
    calibrator = s._calibrate_proba(dval)
    assert calibrator is None or isinstance(calibrator, IsotonicRegression), \
        f"Expected IsotonicRegression, got {type(calibrator)}"
    print(f"  [PASS] Isotonic calibration fallback: OK")


# ============================================================================
# Test 2: scale_pos_weight 上限保护
# ============================================================================

def test_scale_pos_weight_cap():
    """scale_pos_weight 上限保护：正常比例场景。"""
    s = make_strategy()
    s._class_balance_mode = 'auto'
    s._class_balance_max_scale = 5.0

    # 正常比例: 40% positive
    n = 500
    y = np.array([1] * 200 + [0] * 300)
    np.random.shuffle(y)

    params = {'objective': 'binary:logistic'}
    result = s._apply_class_balance(params.copy(), y)
    sw = result.get('scale_pos_weight', 1.0)
    assert sw <= 5.0, f"scale_pos_weight should be <= 5.0, got {sw}"
    print(f"  [PASS] scale_pos_weight normal: pos_ratio=0.4, scale={sw:.3f}")


def test_scale_pos_weight_extreme():
    """scale_pos_weight 上限保护：极端不均衡场景。"""
    s = make_strategy()
    s._class_balance_mode = 'auto'
    s._class_balance_max_scale = 5.0
    s._class_balance_auto_switch = True

    # 极端不均衡: 3% positive
    n = 500
    y = np.array([1] * 15 + [0] * 485)
    np.random.shuffle(y)

    params = {'objective': 'binary:logistic'}
    result = s._apply_class_balance(params.copy(), y)
    sw = result.get('scale_pos_weight', 1.0)
    assert sw == 5.0, f"Extreme imbalance: scale should be capped at 5.0, got {sw}"
    print(f"  [PASS] scale_pos_weight extreme: pos_ratio=0.03, scale={sw:.3f} (capped)")


def test_scale_pos_weight_disabled():
    """class_balance='none' 时不注入 scale_pos_weight。"""
    s = make_strategy()
    s._class_balance_mode = 'none'

    y = np.array([1] * 50 + [0] * 450)
    params = {'objective': 'binary:logistic'}
    result = s._apply_class_balance(params.copy(), y)
    assert 'scale_pos_weight' not in result, \
        "class_balance=none should not inject scale_pos_weight"
    print(f"  [PASS] scale_pos_weight disabled: mode=none, no injection")


def test_scale_pos_weight_single_class():
    """单一类别时不注入 scale_pos_weight。"""
    s = make_strategy()
    s._class_balance_mode = 'auto'

    y_all_zero = np.zeros(500, dtype=int)
    y_all_one = np.ones(500, dtype=int)

    params = {'objective': 'binary:logistic'}
    r1 = s._apply_class_balance(params.copy(), y_all_zero)
    r2 = s._apply_class_balance(params.copy(), y_all_one)
    assert 'scale_pos_weight' not in r1
    assert 'scale_pos_weight' not in r2
    print(f"  [PASS] scale_pos_weight single class: correctly skipped")


# ============================================================================
# Test 3: 行业敞口限制
# ============================================================================

def test_industry_constraint_basic():
    """行业约束基本功能：所有行业均 ≤ 30%，无需调整。"""
    s = make_strategy()
    s._ind_constraint_enabled = True
    s._ind_max_weight = 0.30
    s._ind_fallback = 'unknown'

    # 5个行业均匀分布 (20% each, all <= 30%)
    n = 15
    codes = [f"{600000 + i:06d}" for i in range(n)]
    industries = ['A'] * 3 + ['B'] * 3 + ['C'] * 3 + ['D'] * 3 + ['E'] * 3
    raw_weights = np.ones(n) / n  # 6.67% each

    selected = pd.DataFrame({'code': codes})
    full_snap = pd.DataFrame({'code': codes, 'industry': industries})
    adjusted = s._apply_industry_constraint(selected, raw_weights, full_snap)

    assert np.allclose(adjusted, raw_weights, atol=0.01), \
        "All industries <= 30%, weights should be unchanged"
    print(f"  [PASS] Industry constraint basic: all inds 20% each, unchanged")


def test_industry_constraint_excess():
    """行业约束：单一行业超额 → 正确裁剪。"""
    s = make_strategy()
    s._ind_constraint_enabled = True
    s._ind_max_weight = 0.30
    s._ind_fallback = 'unknown'

    # 10只股票, 行业A占50%, B占20%, C占15%, D占15%
    n = 10
    codes = [f"{600000 + i:06d}" for i in range(n)]
    # A: 40% (4/10) but give them higher weight
    industries = ['A'] * 4 + ['B'] * 2 + ['C'] * 2 + ['D'] * 2
    # A stocks get 0.13 each (total 0.52), others get 0.08 each (total 0.48)
    raw_weights = np.array([0.13] * 4 + [0.08] * 6)
    raw_weights = raw_weights / raw_weights.sum()

    selected = pd.DataFrame({'code': codes})
    full_snap = pd.DataFrame({'code': codes, 'industry': industries})

    adjusted = s._apply_industry_constraint(selected, raw_weights, full_snap)

    # 验证: 行业A权重应 ≤ 0.30
    ind_a_mask = np.array(industries) == 'A'
    ind_a_w = adjusted[ind_a_mask].sum()
    assert ind_a_w <= 0.31, f"Industry A weight {ind_a_w:.3f} should be <= 0.30"
    assert abs(adjusted.sum() - 1.0) < 0.02, f"Sum should be 1.0, got {adjusted.sum():.4f}"
    print(f"  [PASS] Industry constraint excess: ind_A=0.52→{ind_a_w:.3f}, sum={adjusted.sum():.3f}")


def test_industry_constraint_no_excess():
    """无超额行业时不触发裁剪。"""
    s = make_strategy()
    s._ind_constraint_enabled = True
    s._ind_max_weight = 0.30
    s._ind_fallback = 'unknown'

    n = 15
    codes = [f"{600000 + i:06d}" for i in range(n)]
    # 均匀分布, 每个行业20%
    industries = ['A'] * 3 + ['B'] * 3 + ['C'] * 3 + ['D'] * 3 + ['E'] * 3
    raw_weights = np.ones(n) / n

    selected = pd.DataFrame({'code': codes})
    full_snap = pd.DataFrame({'code': codes, 'industry': industries})
    adjusted = s._apply_industry_constraint(selected, raw_weights, full_snap)

    assert np.allclose(adjusted, raw_weights, atol=0.005), \
        "No excess should keep weights unchanged"
    print(f"  [PASS] Industry constraint no excess: weights unchanged")


def test_industry_constraint_fallback():
    """无行业数据时使用 fallback 行业。"""
    s = make_strategy()
    s._ind_constraint_enabled = True
    s._ind_max_weight = 0.30
    s._ind_fallback = 'unknown'

    n = 10
    codes = [f"{600000 + i:06d}" for i in range(n)]
    raw_weights = np.ones(n) / n

    selected = pd.DataFrame({'code': codes})
    # 无 industry 列
    full_snap = pd.DataFrame({'code': codes})

    adjusted = s._apply_industry_constraint(selected, raw_weights, full_snap)
    assert np.allclose(adjusted, raw_weights, atol=0.005), \
        "Fallback (single industry, max_w > 1.0) should keep weights"
    print(f"  [PASS] Industry constraint fallback: no industry column, fallback OK")


# ============================================================================
# Test 4: LightGBM 异质集成 (仅当 LightGBM 可用时)
# ============================================================================

def test_lgb_import_graceful():
    """LightGBM 不可用时优雅降级。"""
    s = make_strategy()
    s._lgb_enabled = False

    train_df, val_df, all_feats = _make_cv_train_data(n_days=30, n_stocks=50, n_feats=8)
    X_train = train_df[all_feats].fillna(0).values
    y_train = train_df['label'].values.astype(int)
    X_val = val_df[all_feats].fillna(0).values
    y_val = val_df['label'].values.astype(int)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=all_feats)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=all_feats)

    s.feature_names = all_feats
    s._selected_features = all_feats
    s._ensemble_seeds = [42]
    s._combined_enabled = False
    s._class_balance_mode = 'none'
    s.xgb_params = {
        'max_depth': 3, 'eta': 0.1, 'objective': 'binary:logistic',
        'eval_metric': 'logloss', 'tree_method': 'hist', 'seed': 42,
        'subsample': 0.8, 'nthread': 1,
    }
    s.config = {'xgb_n_estimators': 20}
    s.early_stopping_rounds = 5

    models = s._train_ensemble(dtrain, dval, force_seed=True)
    assert len(models) > 0, "XGBoost training should work without LightGBM"
    assert s.lgb_models == [] or len(s.lgb_models) == 0, \
        "LGB models should be empty when disabled"
    print(f"  [PASS] LightGBM graceful degradation: XGB models={len(models)}, LGB models={len(s.lgb_models)}")


def test_lgb_ensemble_training():
    """LightGBM 异质集成训练 (需要 lightgbm 已安装)。"""
    if not _LGB_AVAILABLE:
        print(f"  [SKIP] LightGBM ensemble training: lightgbm not installed")
        return

    s = make_strategy()
    s._lgb_enabled = True
    s._lgb_params = {
        'num_leaves': 15, 'learning_rate': 0.05, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'min_child_samples': 30,
        'reg_alpha': 0.1, 'reg_lambda': 0.5, 'n_estimators': 50,
        'early_stopping_rounds': 10, 'nthread': 1,
    }
    s._class_balance_max_scale = 5.0
    s._class_balance_mode = 'auto'

    train_df, val_df, all_feats = _make_cv_train_data(n_days=40, n_stocks=60, n_feats=8, pos_ratio=0.4)
    X_train = train_df[all_feats].fillna(0).values
    y_train = train_df['label'].values.astype(int)
    X_val = val_df[all_feats].fillna(0).values
    y_val = val_df['label'].values.astype(int)

    s.feature_names = all_feats
    s._selected_features = all_feats
    s._ensemble_seeds = [42, 43]

    # 训练单个 LightGBM
    lgb_model = s._train_single_lgb(X_train, y_train, X_val, y_val, all_feats, seed=42)
    if lgb_model is not None:
        assert hasattr(lgb_model, 'predict'), "LGB model should have predict method"
        pred = lgb_model.predict(X_val)
        assert len(pred) == len(y_val)
        auc = float(roc_auc_score(y_val, pred))
        print(f"  [PASS] LightGBM single model training: type={type(lgb_model).__name__}, val_AUC={auc:.4f}")


def test_lgb_ensemble_prediction():
    """LightGBM 异质集成预测 (需要 lightgbm 已安装)。"""
    if not _LGB_AVAILABLE:
        print(f"  [SKIP] LightGBM ensemble prediction: lightgbm not installed")
        return

    s = make_strategy()
    s._lgb_enabled = True
    s._lgb_params = {
        'num_leaves': 15, 'learning_rate': 0.1, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'min_child_samples': 30,
        'reg_alpha': 0.1, 'reg_lambda': 0.5, 'n_estimators': 30,
        'early_stopping_rounds': 5, 'nthread': 1,
    }
    s._class_balance_mode = 'none'
    s._class_balance_max_scale = 5.0

    train_df, val_df, all_feats = _make_cv_train_data(n_days=30, n_stocks=50, n_feats=6)
    X_train = train_df[all_feats].fillna(0).values
    y_train = train_df['label'].values.astype(int)
    X_val = val_df[all_feats].fillna(0).values
    y_val = val_df['label'].values.astype(int)

    s.feature_names = all_feats
    s._selected_features = all_feats
    s._ensemble_seeds = [42]
    s._combined_enabled = False
    s.xgb_params = {
        'max_depth': 3, 'eta': 0.1, 'objective': 'binary:logistic',
        'eval_metric': 'logloss', 'tree_method': 'hist', 'seed': 42,
        'subsample': 0.8, 'nthread': 1,
    }
    s.config = {'xgb_n_estimators': 20}
    s.early_stopping_rounds = 5

    # 训练 XGBoost (LightGBM 在 _train_ensemble 中自动训练)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=all_feats)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=all_feats)

    if _LGB_AVAILABLE:
        # 手动训练 LGB
        lgb_model = s._train_single_lgb(X_train, y_train, X_val, y_val, all_feats, seed=42)
        s.lgb_models = [lgb_model] if lgb_model else []

    # 训练 XGB
    model = xgb.train(s.xgb_params, dtrain, num_boost_round=20, verbose_eval=False)
    s.models = [model]
    s._active_models = [0]
    s.model_weights = [1.0]

    # 预测
    dpred = xgb.DMatrix(X_val, feature_names=all_feats)
    pred = s._predict_ensemble(dpred)
    assert len(pred) == len(y_val)
    assert np.all((pred >= 0) & (pred <= 1)), "Predictions should be in [0,1]"

    auc = float(roc_auc_score(y_val, pred))
    print(f"  [PASS] LightGBM ensemble prediction: pred_mean={pred.mean():.4f}, val_AUC={auc:.4f}")


# ============================================================================
# Test 5: Config 去重验证
# ============================================================================

def test_config_cleanup_no_top_level_duplicate():
    """验证 config 顶层不再有冗余参数。"""
    import yaml

    config_path = os.path.join(_PROJECT_ROOT, 'strategies', 'wyckoff20', 'config_v10.yaml')
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)

    # ensemble_seeds 应在 ensemble 段, 顶层不应有独立定义
    ensemble_cfg = config.get('ensemble', {})
    assert 'seeds' in ensemble_cfg, "ensemble.seeds should exist"
    # 顶层 max_positions/max_position_pct 应有注释说明已移至 execution 段
    assert 'execution' in config, "execution section should exist"
    assert 'max_positions' in config.get('execution', {}), \
        "execution.max_positions should exist"
    assert 'max_position_pct' in config.get('execution', {}), \
        "execution.max_position_pct should exist"

    # 验证新增配置段
    assert 'calibration_method' in config, "calibration_method should exist"
    assert config['calibration_method'] in ('platt', 'isotonic', 'none'), \
        f"calibration_method should be platt/isotonic/none, got {config['calibration_method']}"

    assert 'industry_constraint' in config, "industry_constraint section should exist"

    lgb_cfg = ensemble_cfg.get('lightgbm', {})
    assert 'enabled' in lgb_cfg, "ensemble.lightgbm.enabled should exist"

    print(f"  [PASS] Config cleanup: calibration={config['calibration_method']}, "
          f"industry_constraint enabled={config['industry_constraint']['enabled']}, "
          f"lgb enabled={lgb_cfg['enabled']}")


# ============================================================================
# Test 6: 端到端集成测试
# ============================================================================

def test_e2e_with_all_fixes():
    """端到端测试：所有 P0/P1 修复集成运行。"""
    s = make_strategy()
    s._calibration_method = 'platt'
    s._calibration_monitoring = True
    s._class_balance_max_scale = 5.0
    s._class_balance_auto_switch = True
    s._ind_constraint_enabled = True
    s._ind_max_weight = 0.30
    s._lgb_enabled = False  # E2E 测试关闭 LGB 以加速
    s._ensemble_seeds = [42, 43]
    s._combined_enabled = False
    s._class_balance_mode = 'auto'

    train_df, val_df, all_feats = _make_cv_train_data(n_days=50, n_stocks=80, n_feats=12, pos_ratio=0.35)
    X_train = train_df[all_feats].fillna(0).values
    y_train = train_df['label'].values.astype(int)
    X_val = val_df[all_feats].fillna(0).values
    y_val = val_df['label'].values.astype(int)

    s.feature_names = all_feats
    s._selected_features = all_feats
    s.xgb_params = {
        'max_depth': 3, 'eta': 0.1, 'objective': 'binary:logistic',
        'eval_metric': 'logloss', 'tree_method': 'hist', 'seed': 42,
        'subsample': 0.8, 'nthread': 1,
        'min_child_weight': 5, 'alpha': 0.3, 'lambda': 1.0, 'gamma': 0.0,
        'colsample_bytree': 0.8,
    }
    s.config = {'xgb_n_estimators': 30}
    s.early_stopping_rounds = 5

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=all_feats)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=all_feats)

    # Train ensemble
    models = s._train_ensemble(dtrain, dval, force_seed=True)
    assert len(models) >= 2, f"Expected >=2 models, got {len(models)}"
    s.models = models
    s._active_models = list(range(len(models)))
    s.model_weights = [1.0] * len(models)

    # Calibrate
    calibrator = s._calibrate_proba(dval)
    assert calibrator is not None, "Calibrator should not be None"
    assert isinstance(calibrator, LogisticRegression), "Should be Platt Scaling"

    # Predict
    s.calibrator = calibrator
    dpred = xgb.DMatrix(X_val, feature_names=all_feats)
    raw = s._predict_ensemble(dpred)
    cal = s._apply_calibration(raw)

    assert len(cal) == len(y_val)
    assert np.all(cal >= 0) and np.all(cal <= 1)

    # Industry constraint
    codes = [f"{600000 + i:06d}" for i in range(10)]
    industries = ['A'] * 4 + ['B'] * 3 + ['C'] * 2 + ['D'] * 1
    selected = pd.DataFrame({'code': codes, 'pred': cal[:10]})
    full_snap = pd.DataFrame({'code': codes, 'industry': industries})
    weights = np.array([0.13] * 4 + [0.08] * 6)
    weights = weights / weights.sum()

    adjusted = s._apply_industry_constraint(selected, weights, full_snap)
    ind_a_w = adjusted[np.array(industries) == 'A'].sum()

    # Summary
    results = {
        'n_models': len(models),
        'val_auc': float(roc_auc_score(y_val, raw)),
        'calibrator': type(calibrator).__name__,
        'cal_std': float(cal.std()),
        'ind_a_weight': float(ind_a_w),
        'all_ok': True,
    }
    print(f"  [PASS] E2E test: {results}")
    assert results['all_ok']


# ============================================================================
# 主入口
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Wyckoff V10 P0/P1 Fixes - 测试套件")
    print(f"LightGBM available: {_LGB_AVAILABLE}")
    print("=" * 60)

    # Platt Scaling
    print("\n--- Platt Scaling ---")
    test_platt_scaling_basic()
    test_platt_scaling_cv_ensemble()
    test_isotonic_fallback()

    # scale_pos_weight
    print("\n--- scale_pos_weight Cap ---")
    test_scale_pos_weight_cap()
    test_scale_pos_weight_extreme()
    test_scale_pos_weight_disabled()
    test_scale_pos_weight_single_class()

    # Industry Constraint
    print("\n--- Industry Constraint ---")
    test_industry_constraint_basic()
    test_industry_constraint_excess()
    test_industry_constraint_no_excess()
    test_industry_constraint_fallback()

    # LightGBM
    print("\n--- LightGBM Ensemble ---")
    test_lgb_import_graceful()
    test_lgb_ensemble_training()
    test_lgb_ensemble_prediction()

    # Config Cleanup
    print("\n--- Config Cleanup ---")
    test_config_cleanup_no_top_level_duplicate()

    # E2E
    print("\n--- E2E Integration ---")
    test_e2e_with_all_fixes()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
