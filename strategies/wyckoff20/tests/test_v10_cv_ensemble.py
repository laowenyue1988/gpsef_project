# -*- coding: utf-8 -*-
"""
测试模块: CV 折模型集成 (对齐 Alpha191)

验证:
- cv_fold_defs_saved: _purged_kfold_cv 保存折定义
- cv_ensemble_trains_per_fold: 每折在独立验证集上训练
- cv_ensemble_auc_weighted: AUC加权投票
- cv_ensemble_diversity: 超参数多样性生效
- backward_compat_seed_ensemble: 无 ensemble 配置时回退种子集成
- cv_ensemble_calibration: CV 折校准 (合并所有折验证集)
- cv_ensemble_prediction: 加权预测正确
"""
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import xgboost as xgb
from sklearn.metrics import roc_auc_score
from strategies.wyckoff20.tests.conftest import make_strategy, make_synthetic_train_data


def _make_cv_train_data(n_days=80, n_stocks=50, seed=42):
    """生成带 CV 折定义的训练数据 (含 snapshot_date + label + 特征列)。"""
    rng = np.random.RandomState(seed)
    base_date = datetime(2025, 1, 1)
    feat_names = [f'feat_{i}' for i in range(8)]

    records = []
    for d in range(n_days):
        dt = base_date + timedelta(days=d)
        for s in range(n_stocks):
            row = {'snapshot_date': dt, 'code': f"sh.{600000 + s:06d}"}
            for f in range(8):
                row[f'feat_{f}'] = rng.randn() + f * 0.15
            records.append(row)

    df = pd.DataFrame(records)
    # 标签：截面中位数二分类
    df['label_raw'] = df[[f'feat_{f}' for f in range(3)]].sum(axis=1)
    df['label'] = df.groupby('snapshot_date')['label_raw'].transform(
        lambda x: (x > x.median()).astype(int))
    return df, feat_names


def test_cv_fold_defs_saved():
    """_purged_kfold_cv 应保存 CV 折定义到 self._cv_fold_defs。"""
    s = make_strategy()
    s.feature_names = [f'feat_{i}' for i in range(8)]
    df, feat_names = _make_cv_train_data()

    s._cv_folds = 5
    s._cv_embargo_days = 5  # 短 embargo 适配测试数据
    s._fs_vote_min_folds = 2
    s._fs_top_k_features = 10

    # 手动运行 _purged_kfold_cv (需要 ic_scores)
    ic_scores = {f: 0.01 for f in feat_names}
    result = s._purged_kfold_cv(df, feat_names, ic_scores)

    assert hasattr(s, '_cv_fold_defs'), "Should have _cv_fold_defs attribute"
    assert len(s._cv_fold_defs) >= 3, f"Expected >=3 fold defs, got {len(s._cv_fold_defs)}"

    # 验证每折定义格式: (train_dates, val_dates)
    for i, (train_dates, val_dates) in enumerate(s._cv_fold_defs):
        assert isinstance(train_dates, list), f"Fold {i}: train_dates should be list"
        assert isinstance(val_dates, list), f"Fold {i}: val_dates should be list"
        assert len(train_dates) > 0, f"Fold {i}: train_dates should not be empty"
        # 无日期重叠 (embargo 保证)
        overlap = set(train_dates) & set(val_dates)
        assert len(overlap) == 0, f"Fold {i}: train/val date overlap: {overlap}"

    print(f"[PASS] test_cv_fold_defs_saved: {len(s._cv_fold_defs)} folds saved")


def test_cv_ensemble_trains_per_fold():
    """CV 折集成: 每折在独立验证集上训练, 各产生一个模型。"""
    s = make_strategy()
    df, feat_names = _make_cv_train_data()

    s.feature_names = feat_names
    s._cv_folds = 5
    s._cv_embargo_days = 5
    s._fs_vote_min_folds = 2
    s._fs_top_k_features = 10

    # 模拟特征选择完成后保存折定义
    ic_scores = {f: 0.01 for f in feat_names}
    s._purged_kfold_cv(df, feat_names, ic_scores)

    # 设置 CV 折集成配置
    s._ensemble_method = 'cv_fold'
    s._ensemble_seeds = [42, 43, 44, 45, 46]
    s._ensemble_depth_list = [3, 4, 4, 5, 5]
    s._ensemble_subsample_list = [0.6, 0.7, 0.7, 0.8, 0.8]
    s._ensemble_metric_list = ['auc', 'auc', 'logloss', 'logloss', 'auc']
    s._ensemble_weight_method = 'auc_weighted'
    s._ensemble_min_auc = 0.50
    s.config = {'xgb_n_estimators': 30}
    s.early_stopping_rounds = 10

    models = s._train_cv_ensemble(df)

    assert models is not None, "CV ensemble should return models"
    assert len(models) >= 3, f"Expected >=3 models, got {len(models)}"

    # 验证 model_aucs / model_weights / _active_models 已设置
    assert len(s.model_aucs) == len(models), "model_aucs length mismatch"
    assert len(s.model_weights) == len(models), "model_weights length mismatch"
    assert len(s._active_models) > 0, "_active_models should not be empty"

    # 验证每对 (train_dates, val_dates) 没有重叠
    for i in range(len(models)):
        if i < len(s._cv_fold_defs):
            train_d, val_d = s._cv_fold_defs[i]
            overlap = set(train_d) & set(val_d)
            assert len(overlap) == 0, f"Fold {i}: date overlap"

    print(f"[PASS] test_cv_ensemble_trains_per_fold: {len(models)} models, "
          f"aucs={[f'{a:.4f}' for a in s.model_aucs]}")


def test_cv_ensemble_auc_weighted():
    """AUC加权投票: 高 AUC 模型权重 > 低 AUC 模型权重。"""
    s = make_strategy()
    df, feat_names = _make_cv_train_data(n_days=60)

    s.feature_names = feat_names
    s._cv_folds = 5
    s._cv_embargo_days = 3
    s._fs_vote_min_folds = 2
    s._fs_top_k_features = 10

    ic_scores = {f: 0.01 for f in feat_names}
    s._purged_kfold_cv(df, feat_names, ic_scores)

    s._ensemble_method = 'cv_fold'
    s._ensemble_seeds = [42, 43, 44, 45, 46]
    s._ensemble_depth_list = [3, 3, 3, 3, 3]       # 同深度, 排除结构差异
    s._ensemble_subsample_list = [0.7, 0.7, 0.7, 0.7, 0.7]  # 同采样率
    s._ensemble_metric_list = ['auc', 'auc', 'auc', 'auc', 'auc']
    s._ensemble_weight_method = 'auc_weighted'
    s._ensemble_min_auc = 0.0  # 不排除任何模型
    s.config = {'xgb_n_estimators': 30}
    s.early_stopping_rounds = 10

    s._train_cv_ensemble(df)

    # AUC 加权模式下, 权重应正比于 excess AUC
    aucs = s.model_aucs
    weights = s.model_weights
    # 高 AUC 模型应该有更高权重
    max_auc_idx = np.argmax(aucs)
    min_auc_idx = np.argmin(aucs)

    assert weights[max_auc_idx] > weights[min_auc_idx], \
        f"High-AUC model (idx={max_auc_idx}, auc={aucs[max_auc_idx]:.4f}, w={weights[max_auc_idx]:.3f}) " \
        f"should have higher weight than low-AUC model (idx={min_auc_idx}, auc={aucs[min_auc_idx]:.4f}, w={weights[min_auc_idx]:.3f})"

    print(f"[PASS] test_cv_ensemble_auc_weighted: "
          f"max_auc={aucs[max_auc_idx]:.4f} w={weights[max_auc_idx]:.3f}, "
          f"min_auc={aucs[min_auc_idx]:.4f} w={weights[min_auc_idx]:.3f}")


def test_cv_ensemble_diversity():
    """超参数多样性: 不同折使用不同的 depth/subsample/metric。"""
    s = make_strategy()
    df, feat_names = _make_cv_train_data(n_days=60)

    s.feature_names = feat_names
    s._cv_folds = 5
    s._cv_embargo_days = 3
    s._fs_vote_min_folds = 2
    s._fs_top_k_features = 10

    ic_scores = {f: 0.01 for f in feat_names}
    s._purged_kfold_cv(df, feat_names, ic_scores)

    s._ensemble_method = 'cv_fold'
    s._ensemble_seeds = [42, 43, 44, 45, 46]
    s._ensemble_depth_list = [3, 4, 4, 5, 5]       # 不同深度
    s._ensemble_subsample_list = [0.6, 0.7, 0.7, 0.8, 0.8]  # 不同采样率
    s._ensemble_metric_list = ['auc', 'auc', 'logloss', 'logloss', 'auc']  # 不同目标
    s._ensemble_weight_method = 'auc_weighted'
    s._ensemble_min_auc = 0.50
    s.config = {'xgb_n_estimators': 30}
    s.early_stopping_rounds = 10

    s._train_cv_ensemble(df)

    # 验证多样性: 通过检查不同模型的 best_iteration 不同来间接验证
    best_iters = [getattr(m, 'best_iteration', 0) for m in s.models]
    # 由于超参数不同, best_iter 应该不全相同
    unique_iters = len(set(best_iters))
    # 放宽验证: 至少有 2 个不同的 best_iter (不要求全部不同, 因为数据可能有噪声)
    assert unique_iters >= 2, \
        f"Hyperparameter diversity should produce different best_iters, got {unique_iters} unique out of {len(best_iters)}"

    print(f"[PASS] test_cv_ensemble_diversity: best_iters={best_iters} ({unique_iters} unique)")


def test_backward_compat_seed_ensemble():
    """向后兼容: 无 ensemble 配置段时, _train_ensemble 使用旧版种子集成。"""
    s = make_strategy()
    s._ensemble_method = 'seed_ensemble'
    s._ensemble_seeds = [42, 43]
    s.config = {'xgb_n_estimators': 30}
    s.early_stopping_rounds = 10

    # 构造 DMatrix (旧版种子集成接口)
    rng = np.random.RandomState(42)
    X_train = rng.randn(200, 6)
    y_train = (X_train[:, :2].sum(axis=1) > 0).astype(int)
    X_val = rng.randn(50, 6)
    y_val = (X_val[:, :2].sum(axis=1) > 0).astype(int)

    feat_names = [f'f{i}' for i in range(6)]
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feat_names)

    s.feature_names = feat_names
    s._class_balance_mode = 'none'

    models = s._train_ensemble(dtrain, dval)
    assert len(models) == 2, f"Expected 2 models, got {len(models)}"
    assert s._ensemble_method == 'seed_ensemble'

    # 预测测试
    s.models = models
    preds = s._predict_ensemble(dval)
    assert len(preds) == len(y_val)
    assert preds.min() >= 0 and preds.max() <= 1

    print(f"[PASS] test_backward_compat_seed_ensemble: {len(models)} seeds, range=[{preds.min():.4f}, {preds.max():.4f}]")


def test_cv_ensemble_prediction():
    """CV 折集成预测: 使用 AUC 加权投票。"""
    s = make_strategy()
    df, feat_names = _make_cv_train_data(n_days=60)

    s.feature_names = feat_names
    s._cv_folds = 5
    s._cv_embargo_days = 3
    s._fs_vote_min_folds = 2
    s._fs_top_k_features = 10

    ic_scores = {f: 0.01 for f in feat_names}
    s._purged_kfold_cv(df, feat_names, ic_scores)

    s._ensemble_method = 'cv_fold'
    s._ensemble_seeds = [42, 43, 44, 45, 46]
    s._ensemble_depth_list = [3, 4, 4, 5, 5]
    s._ensemble_subsample_list = [0.6, 0.7, 0.7, 0.8, 0.8]
    s._ensemble_metric_list = ['auc', 'auc', 'logloss', 'logloss', 'auc']
    s._ensemble_weight_method = 'auc_weighted'
    s._ensemble_min_auc = 0.50
    s.config = {'xgb_n_estimators': 30}
    s.early_stopping_rounds = 10

    s._train_cv_ensemble(df)

    # 构造测试数据
    rng = np.random.RandomState(99)
    X_test = rng.randn(100, 8)
    dtest = xgb.DMatrix(X_test, feature_names=feat_names)

    preds = s._predict_ensemble(dtest)
    assert len(preds) == 100
    assert preds.min() >= 0 and preds.max() <= 1

    # 加权平均应在各模型预测范围之间
    individual_preds = [m.predict(dtest) for m in s.models]
    min_pred = np.min([p.min() for p in individual_preds])
    max_pred = np.max([p.max() for p in individual_preds])
    assert preds.min() >= min_pred - 1e-10
    assert preds.max() <= max_pred + 1e-10

    print(f"[PASS] test_cv_ensemble_prediction: range=[{preds.min():.4f}, {preds.max():.4f}]")


def test_cv_ensemble_calibration():
    """CV 折校准: 合并所有折验证集预测, 拟合 Isotonic 回归。"""
    s = make_strategy()
    df, feat_names = _make_cv_train_data(n_days=60)

    s.feature_names = feat_names
    s._cv_folds = 5
    s._cv_embargo_days = 3
    s._fs_vote_min_folds = 2
    s._fs_top_k_features = 10
    s._calibration_method = 'isotonic'
    s._prob_clip_min = 0.05
    s._prob_clip_max = 0.95

    ic_scores = {f: 0.01 for f in feat_names}
    s._purged_kfold_cv(df, feat_names, ic_scores)

    s._ensemble_method = 'cv_fold'
    s._ensemble_seeds = [42, 43, 44, 45, 46]
    s._ensemble_depth_list = [3, 4, 4, 5, 5]
    s._ensemble_subsample_list = [0.6, 0.7, 0.7, 0.8, 0.8]
    s._ensemble_metric_list = ['auc', 'auc', 'logloss', 'logloss', 'auc']
    s._ensemble_weight_method = 'auc_weighted'
    s._ensemble_min_auc = 0.50
    s.config = {'xgb_n_estimators': 30}
    s.early_stopping_rounds = 10

    s._train_cv_ensemble(df)
    cal = s._calibrate_cv_ensemble(df)

    assert cal is not None, "Isotonic calibrator should not be None"

    # 校准后预测应在 [0, 1] 范围内
    rng = np.random.RandomState(99)
    X_test = rng.randn(50, 8)
    dtest = xgb.DMatrix(X_test, feature_names=feat_names)
    raw = s._predict_ensemble(dtest)
    cal_preds = cal.predict(raw)
    assert cal_preds.min() >= 0 and cal_preds.max() <= 1

    print(f"[PASS] test_cv_ensemble_calibration: raw range=[{raw.min():.4f}, {raw.max():.4f}], "
          f"cal range=[{cal_preds.min():.4f}, {cal_preds.max():.4f}]")


if __name__ == '__main__':
    test_cv_fold_defs_saved()
    test_cv_ensemble_trains_per_fold()
    test_cv_ensemble_auc_weighted()
    test_cv_ensemble_diversity()
    test_backward_compat_seed_ensemble()
    test_cv_ensemble_prediction()
    test_cv_ensemble_calibration()
    print("\n=== All CV Ensemble tests PASSED ===")
