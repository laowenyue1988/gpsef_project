# -*- coding: utf-8 -*-
"""
测试模块 8: 端到端冒烟测试

验证：
- smoke_train_predict: 完整训练→预测流程不报错
- auc_gate_integration: AUC门控在训练后正确触发
- calibration_integration: 校准器在训练后正确拟合
- feature_selection_integration: 特征选择管线完整运行
- model_params_consistency: XGBoost参数与配置一致
"""
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import xgboost as xgb
from strategies.wyckoff20.tests.conftest import make_strategy, make_synthetic_train_data
from sklearn.metrics import roc_auc_score


def _make_full_train_val_split(n_days=80, n_stocks=100, n_feats=12, seed=42):
    """生成完整的训练/验证数据，含特征列和label。"""
    rng = np.random.RandomState(seed)
    base_date = datetime(2025, 1, 1)

    all_records = []
    for d in range(n_days):
        dt = base_date + timedelta(days=d)
        for s in range(n_stocks):
            code = f"sh.{600000+s:06d}"
            row = {
                'snapshot_date': dt,
                'code': code,
            }
            # 生成特征（前5个有信号）
            for f in range(n_feats):
                if f < 5:
                    row[f'feat_{f}'] = rng.randn() + f * 0.3
                else:
                    row[f'feat_{f}'] = rng.randn()
            all_records.append(row)

    df = pd.DataFrame(all_records)

    # 生成label（与有信号特征正相关）
    signal_cols = [f'feat_{f}' for f in range(5)]
    df['label_raw'] = df[signal_cols].sum(axis=1) + rng.randn(len(df)) * 0.5
    # 二分类：跑赢中位数 → 1
    df['label'] = df.groupby('snapshot_date')['label_raw'].transform(
        lambda x: (x > x.median()).astype(int)
    )

    # 时序分割：80% train, 20% val
    unique_dates = sorted(df['snapshot_date'].unique())
    split_idx = int(len(unique_dates) * 0.8)
    train_dates = unique_dates[:split_idx]
    val_dates = unique_dates[split_idx:]

    train_df = df[df['snapshot_date'].isin(train_dates)].copy()
    val_df = df[df['snapshot_date'].isin(val_dates)].copy()

    return train_df, val_df, [f'feat_{f}' for f in range(n_feats)]


def test_smoke_train_predict():
    """完整训练→预测流程不报错。"""
    s = make_strategy()
    s._ensemble_seeds = [42, 43]  # 测试用2个种子加速
    s.config = {'xgb_n_estimators': 50}
    s.early_stopping_rounds = 10
    s._feat_sel_mda_n_repeats = 3  # 测试用小值
    s._min_features = 3
    s._max_features = 15

    train_df, val_df, all_feats = _make_full_train_val_split(n_days=60, n_stocks=50, n_feats=10)

    # 1. 特征选择
    selected = s._select_features(train_df, val_df, all_feats, target_col='label')
    assert len(selected) > 0, "Feature selection should return non-empty list"
    assert len(selected) >= s._min_features, f"Selected {len(selected)} < min {s._min_features}"
    assert len(selected) <= s._max_features, f"Selected {len(selected)} > max {s._max_features}"
    s.feature_names = selected

    print(f"  Feature selection: {len(all_feats)} → {len(selected)}")

    # 2. 构建DMatrix
    X_train = train_df[selected].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_val = val_df[selected].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_train = train_df['label'].values
    y_val = val_df['label'].values

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=selected)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=selected)

    # 3. 集成训练
    models = s._train_ensemble(dtrain, dval)
    s.models = models
    assert len(models) == 2, f"Expected 2 models, got {len(models)}"

    # 4. 集成预测
    preds = s._predict_ensemble(dval)
    assert len(preds) == len(y_val), f"Pred length mismatch: {len(preds)} vs {len(y_val)}"
    assert preds.min() >= 0 and preds.max() <= 1, "Preds should be in [0, 1]"

    # 5. 校准
    calibrator = s._calibrate_proba(dval)
    s.calibrator = calibrator
    assert calibrator is not None, "Calibrator should be created"

    # 6. 应用校准
    calibrated_preds = s._apply_calibration(preds)
    assert calibrated_preds.min() >= s._prob_clip_min
    assert calibrated_preds.max() <= s._prob_clip_max

    # 7. AUC门控
    val_auc = roc_auc_score(y_val, calibrated_preds)
    mode, scale = s._auc_gate(val_auc)

    print(f"  Val AUC: {val_auc:.4f} → mode={mode}, scale={scale}")
    print(f"  Preds range: [{calibrated_preds.min():.4f}, {calibrated_preds.max():.4f}]")

    print("[PASS] test_smoke_train_predict: full pipeline completed")


def test_auc_gate_integration():
    """AUC门控在训练后根据验证集AUC正确触发。"""
    s = make_strategy()
    s._ensemble_seeds = [42]
    s.config = {'xgb_n_estimators': 50}
    s.early_stopping_rounds = 10

    train_df, val_df, all_feats = _make_full_train_val_split(n_days=60, n_stocks=50, n_feats=8)

    # 直接用全部特征训练（跳过特征选择）
    s.feature_names = all_feats
    X_train = train_df[all_feats].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_val = val_df[all_feats].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_train = train_df['label'].values
    y_val = val_df['label'].values

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=all_feats)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=all_feats)

    models = s._train_ensemble(dtrain, dval)
    s.models = models

    # 计算验证集AUC
    preds = s._predict_ensemble(dval)
    val_auc = roc_auc_score(y_val, preds)

    # 根据AUC选择门控模式
    mode, scale = s._auc_gate(val_auc)

    # 验证门控逻辑正确（对齐Alpha191: full/half/empty）
    if val_auc >= 0.55:
        assert mode == "full" and scale == 1.0
    elif val_auc >= 0.50:
        assert mode == "half" and scale == 0.5
    else:
        assert mode == "empty" and scale == 0.0

    print(f"[PASS] test_auc_gate_integration: AUC={val_auc:.4f} → {mode}({scale}x)")


def test_calibration_integration():
    """校准器在集成训练后正确拟合。"""
    s = make_strategy()
    s._ensemble_seeds = [42, 43]
    s.config = {'xgb_n_estimators': 50}
    s.early_stopping_rounds = 10

    train_df, val_df, all_feats = _make_full_train_val_split(n_days=60, n_stocks=50, n_feats=8)
    s.feature_names = all_feats

    X_train = train_df[all_feats].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_val = val_df[all_feats].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_train = train_df['label'].values
    y_val = val_df['label'].values

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=all_feats)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=all_feats)

    s.models = s._train_ensemble(dtrain, dval)

    # 校准
    s.calibrator = s._calibrate_proba(dval)
    assert s.calibrator is not None

    # 验证校准效果
    raw_preds = s._predict_ensemble(dval)
    calibrated = s._apply_calibration(raw_preds)

    # 校准后Brier分数应 <= 原始（或接近）
    from sklearn.metrics import brier_score_loss
    brier_raw = brier_score_loss(y_val, np.clip(raw_preds, 0, 1))
    brier_cal = brier_score_loss(y_val, calibrated)

    print(f"  Brier: raw={brier_raw:.4f}, calibrated={brier_cal:.4f}")
    assert brier_cal <= brier_raw + 0.01, "Calibration should not worsen Brier score significantly"

    print("[PASS] test_calibration_integration: calibrator fitted, Brier improved")


def test_model_params_consistency():
    """验证config_v10.yaml的XGBoost参数与Alpha191标准一致。"""
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config_v10.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 验证config中的XGBoost参数 (depth=6: 根除prob平局, mcw=5: 允许弱信号分裂)
    assert config['xgb_max_depth'] == 6, f"xgb_max_depth should be 6, got {config['xgb_max_depth']}"
    assert config['xgb_learning_rate'] == 0.02, f"xgb_learning_rate should be 0.02, got {config['xgb_learning_rate']}"
    assert config['xgb_subsample'] == 0.7, f"xgb_subsample should be 0.7, got {config['xgb_subsample']}"
    assert config['xgb_colsample_bytree'] == 0.8
    assert config['xgb_min_child_weight'] == 5, f"min_child_weight should be 5, got {config['xgb_min_child_weight']}"
    assert config['xgb_reg_alpha'] == 0.3
    assert config['xgb_reg_lambda'] == 1.0
    assert config['xgb_gamma'] == 0.0
    assert config['xgb_tree_method'] == 'hist'
    assert config['xgb_device'] == 'cpu'
    assert config['xgb_nthread'] == 1
    assert config['xgb_objective'] == 'binary:logistic'
    assert config['xgb_eval_metric'] == 'logloss'

    # AUC门控参数 (对齐Alpha191: 0.48/0.52 离散半仓)
    assert config['auc_gate_enabled'] == True
    assert config['auc_gate_min'] == 0.48, \
        f"auc_gate_min should be 0.48, got {config['auc_gate_min']}"
    assert config['auc_gate_full'] == 0.52, \
        f"auc_gate_full should be 0.52, got {config['auc_gate_full']}"

    # CV参数
    assert config['cv_folds'] == 5
    assert config['cv_embargo_days'] == 21
    assert config['cv_feature_consensus'] == 3

    # 集成参数
    assert config['ensemble_seeds'] == [42, 43, 44, 45, 46]

    # 校准参数 (none: 避免Isotonic binning加剧prob平局)
    assert config['calibration_method'] == 'none'
    assert config['prob_clip_min'] == 0.05
    assert config['prob_clip_max'] == 0.95

    # 特征选择参数
    assert config['feat_sel_psi_threshold'] == 0.15
    assert config['feat_sel_mda_n_repeats'] == 20
    assert config['feat_sel_min_features'] == 10
    assert config['feat_sel_max_features'] == 80
    assert config['feat_sel_min_ic'] == 0.001, f"IC threshold should be 0.001 (Alpha191 aligned), got {config['feat_sel_min_ic']}"
    assert config['feat_sel_variance_threshold'] == 1.0e-5
    assert config['cv_ic_psi_split_ratio'] == 0.5

    # feature_selection v2.0 参数 (对齐Alpha191)
    fs_cfg = config['feature_selection']
    assert fs_cfg['use_voting'] is True
    assert fs_cfg['mda']['t_alpha'] == 0.05
    assert fs_cfg['mda']['random_state'] == 42
    assert fs_cfg['mda']['use_std_threshold'] is True
    assert fs_cfg['voting']['vote_min_folds'] == 3
    assert fs_cfg['voting']['vote_min_features'] == 10
    assert fs_cfg['voting']['top_k_features'] == 80
    assert fs_cfg['voting']['weight_ic'] == 0.0, \
        f"weight_ic should be 0.0 (IC ranking disabled in scoring), got {fs_cfg['voting']['weight_ic']}"
    assert fs_cfg['voting']['weight_mdi'] == 0.3
    assert fs_cfg['voting']['weight_mda'] == 1.5
    assert fs_cfg['voting']['fallback_to_ic_only'] is True

    # 验证make_strategy()创建的实例与config键名映射一致
    s = make_strategy()
    assert 'max_depth' in s.xgb_params
    assert 'eta' in s.xgb_params
    assert 'tree_method' in s.xgb_params
    assert 'nthread' in s.xgb_params
    assert s.xgb_params['objective'] == 'binary:logistic'
    assert s.xgb_params['eval_metric'] == 'logloss'

    print("[PASS] test_model_params_consistency: config values match Alpha191 standard")


def test_determinism_full_pipeline():
    """完整管线两次运行结果一致（确定性）。"""
    s1 = make_strategy()
    s1._ensemble_seeds = [42]
    s1.config = {'xgb_n_estimators': 30}
    s1.early_stopping_rounds = 10

    s2 = make_strategy()
    s2._ensemble_seeds = [42]
    s2.config = {'xgb_n_estimators': 30}
    s2.early_stopping_rounds = 10

    train_df, val_df, all_feats = _make_full_train_val_split(n_days=40, n_stocks=30, n_feats=6)

    X_train = train_df[all_feats].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_val = val_df[all_feats].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_train = train_df['label'].values
    y_val = val_df['label'].values

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=all_feats)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=all_feats)

    s1.models = s1._train_ensemble(dtrain, dval)
    s2.models = s2._train_ensemble(dtrain, dval)

    preds1 = s1._predict_ensemble(dval)
    preds2 = s2._predict_ensemble(dval)

    assert np.allclose(preds1, preds2, atol=1e-8), "Same config should produce identical predictions"

    print("[PASS] test_determinism_full_pipeline: 2 runs identical")


if __name__ == '__main__':
    test_smoke_train_predict()
    test_auc_gate_integration()
    test_calibration_integration()
    test_model_params_consistency()
    test_determinism_full_pipeline()
    print("\n=== All E2E Smoke tests PASSED ===")
