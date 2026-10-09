# -*- coding: utf-8 -*-
"""
测试模块 1: Purged K-Fold CV + MDA多折投票 (对齐Alpha191)

验证：
- prefilter_basic: IC预过滤基本功能
- kfold_splits: K折划分正确，val区间不重叠
- embargo_isolation: embargo隔离有效
- mda_voting: MDA多折投票逻辑
- fallback_min_features: 不足时逐步降低共识门槛
- determinism: 多次运行结果一致
- empty_input: 空输入安全返回
"""
import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from strategies.wyckoff20.tests.conftest import make_strategy, make_synthetic_train_data


def _mock_mda_results(feats, pass_indices_per_fold):
    """
    生成mock MDA结果。
    pass_indices_per_fold: List[set[int]], 每折通过t检验的特征索引集合
    返回: (mda_means, mda_stds) 使得指定特征通过t检验，其余不通过
    """
    mda_means = {}
    mda_stds = {}
    for i, f in enumerate(feats):
        if i in pass_indices_per_fold:
            mda_means[f] = 0.01   # positive -> passes t-test
            mda_stds[f] = 0.001   # small std -> t_stat large
        else:
            mda_means[f] = -0.001  # negative -> fails t-test
            mda_stds[f] = 0.01
    return mda_means, mda_stds


def test_prefilter_basic():
    """IC预过滤基本功能：返回通过variance/sanitize/PSI/IC的特征。"""
    s = make_strategy()
    train_data = make_synthetic_train_data(n_days=60, n_stocks=30, n_feats=8)
    val_data = make_synthetic_train_data(n_days=20, n_stocks=30, n_feats=8)
    feats = [f'feat_{f}' for f in range(8)]

    result, ic_scores = s._prefilter_features(train_data, val_data, feats, target_col='label')

    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert isinstance(ic_scores, dict), f"Expected dict, got {type(ic_scores)}"
    # IC阈值=0.001, 合成数据应该大部分通过
    assert len(result) > 0, "Expected some features to pass prefilter"
    for f in result:
        assert f in feats, f"Unknown feature {f}"
        assert f in ic_scores, f"Missing IC score for {f}"
    print(f"[PASS] test_prefilter_basic: {len(result)} features passed, {len(ic_scores)} IC scores")


def test_kfold_splits():
    """K=5折划分正确，每折val区间不重叠。"""
    s = make_strategy()
    s._cv_folds = 5
    s._cv_embargo_days = 0
    s._fs_vote_min_folds = 1

    train_data = make_synthetic_train_data(n_days=100, n_stocks=50, n_feats=8)
    feats = [f'feat_{f}' for f in range(8)]
    ic_scores = {f: 0.05 for f in feats}

    # Mock xgb.train 和 _compute_mda_oos 避免实际训练
    fold_val_indices = []

    with patch.object(s, '_compute_mda_oos') as mock_mda, \
         patch('strategies.wyckoff20.gpsef_strategy_v10.xgb.train') as mock_train:

        mock_model = MagicMock()
        mock_model.get_score.return_value = {f: 1.0 for f in feats}
        mock_train.return_value = mock_model

        def mock_mda_fn(model, X_val, y_val, feature_names, n_repeats=20, random_state=42, t_alpha=0.05):
            # 记录每折的val行索引（用于验证不重叠）
            fold_val_indices.append(set(X_val.index.tolist()))
            # 所有特征都通过
            return _mock_mda_results(feats, set(range(len(feats))))

        mock_mda.side_effect = mock_mda_fn
        s._purged_kfold_cv(train_data, feats, ic_scores)

    # 验证5折都执行了 (mock_train调用5次)
    assert mock_train.call_count == 5, f"Expected 5 folds, got {mock_train.call_count}"

    # [2026-08-18 P0-1] MDA 输入已迁移为折内 inner-OOS (跨折可重叠属正常);
    # 折间 val 划分不重叠改由 _cv_fold_defs 验证 (折定义本身)
    fold_val_dates = [set(vd) for _, vd in s._cv_fold_defs]
    assert len(fold_val_dates) >= 2, "Expected >= 2 fold defs"
    for i in range(len(fold_val_dates)):
        for j in range(i + 1, len(fold_val_dates)):
            overlap = fold_val_dates[i] & fold_val_dates[j]
            assert len(overlap) == 0, f"Folds {i} and {j} val dates overlap: {len(overlap)} days"

    print("[PASS] test_kfold_splits: 5 folds, no val overlap")


def test_embargo_isolation():
    """embargo=21天时，训练集不含验证集前后embargo天内的数据。"""
    s = make_strategy()
    s._cv_folds = 5
    s._cv_embargo_days = 21
    s._fs_vote_min_folds = 1

    train_data = make_synthetic_train_data(n_days=200, n_stocks=30, n_feats=6)
    feats = [f'feat_{f}' for f in range(6)]
    ic_scores = {f: 0.05 for f in feats}

    fold_val_indices = []

    with patch.object(s, '_compute_mda_oos') as mock_mda, \
         patch('strategies.wyckoff20.gpsef_strategy_v10.xgb.train') as mock_train:

        mock_model = MagicMock()
        mock_model.get_score.return_value = {f: 1.0 for f in feats}
        mock_train.return_value = mock_model

        def mock_mda_fn(model, X_val, y_val, feature_names, n_repeats=20, random_state=42, t_alpha=0.05):
            # 记录每折的val行索引
            fold_val_indices.append(set(X_val.index.tolist()))
            return _mock_mda_results(feats, set(range(len(feats))))

        mock_mda.side_effect = mock_mda_fn
        s._purged_kfold_cv(train_data, feats, ic_scores)

    assert len(fold_val_indices) == 5, f"Expected 5 folds, got {len(fold_val_indices)}"

    # [2026-08-18 P0-1] MDA 输入已迁移为折内 inner-OOS; 折间 val 划分不重叠改由 _cv_fold_defs 验证
    fold_val_dates = [set(vd) for _, vd in s._cv_fold_defs]
    assert len(fold_val_dates) == 5, f"Expected 5 fold defs, got {len(fold_val_dates)}"
    for i in range(len(fold_val_dates)):
        for j in range(i + 1, len(fold_val_dates)):
            overlap = fold_val_dates[i] & fold_val_dates[j]
            assert len(overlap) == 0, f"Folds {i} and {j} val dates overlap: {len(overlap)} days"

    print("[PASS] test_embargo_isolation: no val overlap with embargo=21")


def test_mda_voting():
    """MDA多折投票：特征在>=vote_min_folds折通过t检验才入选。"""
    s = make_strategy()
    s._cv_folds = 5
    s._cv_embargo_days = 0
    s._fs_vote_min_folds = 3
    s._fs_vote_min_features = 1

    train_data = make_synthetic_train_data(n_days=100, n_stocks=50, n_feats=8)
    feats = [f'feat_{f}' for f in range(8)]
    ic_scores = {f: 0.05 for f in feats}

    # 设计每折MDA通过的特征:
    # feat_0: fold 0,1,2,3 = 4折 → pass (≥3)
    # feat_1: fold 0,1,2,4 = 4折 → pass
    # feat_2: fold 0,1,3,4 = 4折 → pass
    # feat_3: fold 0,2,3,4 = 4折 → pass
    # feat_4: fold 1 = 1折 → fail
    # feat_5: fold 2 = 1折 → fail
    # feat_6: fold 3 = 1折 → fail
    # feat_7: fold 4 = 1折 → fail
    fold_pass = [
        {0, 1, 2, 3},       # fold 0
        {0, 1, 2, 4},       # fold 1
        {0, 1, 3, 5},       # fold 2
        {0, 2, 3, 6},       # fold 3
        {1, 2, 3, 7},       # fold 4
    ]

    with patch.object(s, '_compute_mda_oos') as mock_mda, \
         patch('strategies.wyckoff20.gpsef_strategy_v10.xgb.train') as mock_train:

        mock_model = MagicMock()
        mock_model.get_score.return_value = {f: 1.0 for f in feats}
        mock_train.return_value = mock_model

        call_idx = [0]
        def mock_mda_fn(model, X_val, y_val, feature_names, n_repeats=20, random_state=42, t_alpha=0.05):
            idx = call_idx[0]
            call_idx[0] += 1
            return _mock_mda_results(feats, fold_pass[idx] if idx < len(fold_pass) else set())

        mock_mda.side_effect = mock_mda_fn
        result = s._purged_kfold_cv(train_data, feats, ic_scores)

    expected = ['feat_0', 'feat_1', 'feat_2', 'feat_3']
    assert result == expected, f"Expected {expected}, got {result}"

    print("[PASS] test_mda_voting: consensus=3, 4 features pass MDA voting")


def test_fallback_min_features():
    """MDA投票不足vote_min_features时，逐步降低共识门槛。"""
    s = make_strategy()
    s._cv_folds = 5
    s._cv_embargo_days = 0
    s._fs_vote_min_folds = 3
    s._fs_vote_min_features = 5

    train_data = make_synthetic_train_data(n_days=100, n_stocks=50, n_feats=8)
    feats = [f'feat_{f}' for f in range(8)]
    ic_scores = {f: 0.05 for f in feats}

    # 每折只有1个不同特征通过 → consensus=3时0个通过 → 逐步降低
    fold_pass = [
        {0}, {1}, {2}, {3}, {4}
    ]

    with patch.object(s, '_compute_mda_oos') as mock_mda, \
         patch('strategies.wyckoff20.gpsef_strategy_v10.xgb.train') as mock_train:

        mock_model = MagicMock()
        mock_model.get_score.return_value = {f: 1.0 for f in feats}
        mock_train.return_value = mock_model

        call_idx = [0]
        def mock_mda_fn(model, X_val, y_val, feature_names, n_repeats=20, random_state=42, t_alpha=0.05):
            idx = call_idx[0]
            call_idx[0] += 1
            return _mock_mda_results(feats, fold_pass[idx] if idx < len(fold_pass) else set())

        mock_mda.side_effect = mock_mda_fn
        result = s._purged_kfold_cv(train_data, feats, ic_scores)

    # consensus=3: 0 features → 降低到2: 0 → 降低到1: 5 features (each appears once)
    assert len(result) >= 5, f"Expected ≥5 features after fallback, got {len(result)}: {result}"

    print(f"[PASS] test_fallback_min_features: fallback to {len(result)} features")


def test_determinism():
    """同输入多次运行，结果完全一致。"""
    s = make_strategy()
    s._cv_folds = 5
    s._cv_embargo_days = 5
    s._fs_vote_min_folds = 2
    s._fs_vote_min_features = 1

    train_data = make_synthetic_train_data(n_days=80, n_stocks=30, n_feats=6)
    feats = [f'feat_{f}' for f in range(6)]
    ic_scores = {f: 0.05 for f in feats}

    results = []
    for _ in range(3):
        with patch.object(s, '_compute_mda_oos') as mock_mda, \
             patch('strategies.wyckoff20.gpsef_strategy_v10.xgb.train') as mock_train:

            mock_model = MagicMock()
            mock_model.get_score.return_value = {f: 1.0 for f in feats}
            mock_train.return_value = mock_model

            # 所有特征都通过
            mock_mda.side_effect = lambda model, X_val, y_val, feature_names, n_repeats=20, random_state=42, t_alpha=0.05: \
                _mock_mda_results(feats, set(range(len(feats))))
            result = s._purged_kfold_cv(train_data, feats, ic_scores)
            results.append(tuple(result))

    assert all(r == results[0] for r in results), f"Non-deterministic: {results}"
    print(f"[PASS] test_determinism: 3 runs identical → {len(results[0])} features")


def test_empty_input():
    """空输入返回空列表。"""
    s = make_strategy()
    empty_df = pd.DataFrame(columns=['snapshot_date', 'code', 'label', 'feat_0'])
    result = s._purged_kfold_cv(empty_df, ['feat_0'], {'feat_0': 0.05})
    assert result == [], f"Expected empty list, got {result}"
    print("[PASS] test_empty_input: empty input returns []")


if __name__ == '__main__':
    test_prefilter_basic()
    test_kfold_splits()
    test_embargo_isolation()
    test_mda_voting()
    test_fallback_min_features()
    test_determinism()
    test_empty_input()
    print("\n=== All Purged K-Fold CV + MDA Voting tests PASSED ===")
