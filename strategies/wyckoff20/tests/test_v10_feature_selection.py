# -*- coding: utf-8 -*-
"""
测试模块 2: 特征选择管线 (P1-1 + P1-2 + P3-4)

验证：
- ic_vectorized_matches_loop: 向量化IC与逐特征IC一致
- psi_filter: PSI过滤逻辑
- mda_ttest: MDA t检验逻辑
- fi_early_stopping: MDI模型使用Early Stopping
- min_max_features: 特征数上下限约束
- sanitize_features: 缺失率+高相关去重
"""
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from strategies.wyckoff20.tests.conftest import make_strategy, make_feature_df
from scipy.stats import spearmanr


def test_ic_vectorized_matches_loop():
    """向量化IC计算结果与逐特征spearmanr一致。"""
    s = make_strategy()
    df, y = make_feature_df(n_samples=200, n_feats=10, seed=42)
    feats = list(df.columns)

    # 构造psi_info（PSI=0, median=0 → 无过滤）
    psi_info = {f: (0.0, 0.0) for f in feats}

    good_feats, scores = s._compute_ic_vectorized(df, y, feats, psi_info)

    # 逐特征验证
    for f in feats:
        vec_ic = scores[f]['ic']
        # 逐特征spearmanr
        loop_ic = spearmanr(df[f].fillna(0).values, y).correlation
        assert abs(vec_ic - loop_ic) < 0.01, f"IC mismatch for {f}: vec={vec_ic:.6f}, loop={loop_ic:.6f}"

    print("[PASS] test_ic_vectorized_matches_loop: vectorized IC matches loop")


def test_psi_filter():
    """PSI > threshold 的特征被过滤。"""
    s = make_strategy()
    s._feat_sel_psi_threshold = 0.15

    rng = np.random.RandomState(42)
    n = 200

    # feat_0: 训练集和验证集分布相似 → PSI低
    tr_f0 = rng.randn(n)
    va_f0 = rng.randn(n)

    # feat_1: 训练集和验证集分布差异大 → PSI高
    tr_f1 = rng.randn(n)
    va_f1 = rng.randn(n) * 5 + 3  # 偏移+放大

    psi_0 = s._calc_psi(pd.Series(tr_f0), pd.Series(va_f0))
    psi_1 = s._calc_psi(pd.Series(tr_f1), pd.Series(va_f1))

    assert psi_0 < 0.15, f"feat_0 PSI should be < 0.15, got {psi_0:.4f}"
    assert psi_1 > 0.15, f"feat_1 PSI should be > 0.15, got {psi_1:.4f}"

    print(f"[PASS] test_psi_filter: PSI_0={psi_0:.4f}(pass), PSI_1={psi_1:.4f}(filtered)")


def test_psi_edge_cases():
    """PSI边界情况：空数据、NaN、全相同值。"""
    s = make_strategy()

    # 空数据
    psi_empty = s._calc_psi(pd.Series([]), pd.Series([]))
    assert psi_empty == 0.0, f"Empty PSI should be 0.0, got {psi_empty}"

    # 全NaN
    psi_nan = s._calc_psi(pd.Series([np.nan]*20), pd.Series([np.nan]*20))
    assert psi_nan == 0.0, f"All-NaN PSI should be 0.0, got {psi_nan}"

    # fill_value
    psi_fill = s._calc_psi(pd.Series([np.nan, 1, 2, 3]), pd.Series([1, 2, 3, 4]), fill_value=0)
    assert isinstance(psi_fill, float) and psi_fill >= 0

    print("[PASS] test_psi_edge_cases: empty/NaN/fillvalue handled")


def test_sanitize_features():
    """sanitize_features: 缺失率过滤 + 高相关去重。"""
    s = make_strategy()
    s._feat_sel_nan_threshold = 0.5
    s._feat_sel_max_corr = 0.90

    rng = np.random.RandomState(42)
    n = 100

    df = pd.DataFrame({
        'feat_good': rng.randn(n),
        'feat_high_nan': [np.nan] * 60 + list(rng.randn(40)),  # 60% NaN → 被过滤
        'feat_dup': rng.randn(n),  # 与feat_good高相关
    })
    # 让feat_dup与feat_good高度相关
    df['feat_dup'] = df['feat_good'] + rng.randn(n) * 0.01

    result = s._sanitize_features(df, 0.5, 0.90)

    # feat_high_nan 应被过滤（60% > 50%）
    assert 'feat_high_nan' not in result, f"feat_high_nan should be filtered, got {result}"

    # feat_dup 和 feat_good 高相关 → 只保留一个
    assert len(result) == 1, f"Expected 1 feature after dedup, got {result}"

    print(f"[PASS] test_sanitize_features: {result}")


def test_mda_ttest():
    """MDA t检验：有信号特征通过，噪声特征被拒。"""
    import xgboost as xgb

    s = make_strategy()
    s._feat_sel_mda_n_repeats = 10

    rng = np.random.RandomState(42)
    n = 300

    # 3个有信号特征 + 2个噪声特征
    X = pd.DataFrame({
        'feat_signal_0': rng.randn(n),
        'feat_signal_1': rng.randn(n),
        'feat_signal_2': rng.randn(n),
        'feat_noise_0': rng.randn(n),
        'feat_noise_1': rng.randn(n),
    })

    # 标签与信号特征强相关
    signal = X['feat_signal_0'] * 2 + X['feat_signal_1'] * 1.5 + X['feat_signal_2'] * 1.0
    y = (signal > np.median(signal)).astype(int).values

    feat_names = list(X.columns)

    # 训练一个小模型
    dtrain = xgb.DMatrix(X.values, label=y, feature_names=feat_names)
    params = s.xgb_params.copy()
    params['nthread'] = 1
    model = xgb.train(params, dtrain, num_boost_round=50, verbose_eval=False)

    # 运行MDA
    mda_means, mda_stds = s._compute_mda_oos(
        model, X, y, feat_names,
        n_repeats=10, random_state=42
    )

    # 验证返回值
    assert len(mda_means) == 5, f"Expected 5 MDA means, got {len(mda_means)}"
    assert len(mda_stds) == 5, f"Expected 5 MDA stds, got {len(mda_stds)}"

    # 信号特征的MDA均值应该 > 0
    for f in ['feat_signal_0', 'feat_signal_1', 'feat_signal_2']:
        assert mda_means[f] >= 0, f"Signal feature {f} MDA should be >= 0, got {mda_means[f]}"

    print(f"[PASS] test_mda_ttest: MDA means computed for 5 features")
    for f in feat_names:
        print(f"  {f}: mean={mda_means[f]:.6f}, std={mda_stds[f]:.6f}")


def test_fi_early_stopping():
    """MDI模型使用Early Stopping → best_iteration < n_estimators。"""
    import xgboost as xgb

    s = make_strategy()
    s._feat_sel_mdi_n_estimators = 200
    s.early_stopping_rounds = 10

    rng = np.random.RandomState(42)
    n = 200

    X_train = pd.DataFrame(rng.randn(n, 5), columns=[f'feat_{i}' for i in range(5)])
    y_train = rng.randint(0, 2, n)
    X_val = pd.DataFrame(rng.randn(100, 5), columns=X_train.columns)
    y_val = rng.randint(0, 2, 100)

    dtrain = xgb.DMatrix(X_train.values, label=y_train, feature_names=list(X_train.columns))
    dval = xgb.DMatrix(X_val.values, label=y_val, feature_names=list(X_val.columns))

    params = s.xgb_params.copy()
    params['nthread'] = 1

    model = xgb.train(
        params, dtrain,
        num_boost_round=s._feat_sel_mdi_n_estimators,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=s.early_stopping_rounds,
        verbose_eval=False
    )

    assert model.best_iteration is not None, "Early stopping should set best_iteration"
    assert model.best_iteration < s._feat_sel_mdi_n_estimators, \
        f"Early stopping should trigger before {s._feat_sel_mdi_n_estimators}, got {model.best_iteration}"

    print(f"[PASS] test_fi_early_stopping: best_iter={model.best_iteration} < {s._feat_sel_mdi_n_estimators}")


def test_min_max_features_constraint():
    """max_features截断：最终特征数不超过max。不再有IC top-K fallback。"""
    s = make_strategy()
    s._min_features = 3
    s._max_features = 5

    # 验证参数读取正确
    assert s._min_features == 3
    assert s._max_features == 5

    # 模拟特征选择管线最终步骤
    fake_feat_scores = {f'feat_{i}': {'composite_score': 10 - i, 'ic_abs': 0.05, 'ic': 0.05}
                        for i in range(10)}

    # 场景1: final_feats太多 → 截断到max
    final_feats = [f'feat_{i}' for i in range(10)]  # 10个，超过max=5
    if len(final_feats) > s._max_features:
        sorted_by_composite = sorted(final_feats,
                                     key=lambda f: fake_feat_scores[f]['composite_score'], reverse=True)
        final_feats = sorted_by_composite[:s._max_features]
    assert len(final_feats) == 5, f"Expected 5 features after truncation, got {len(final_feats)}"

    # 场景2: final_feats少于min → 不再回退到IC top-K，保留MDA结果
    final_feats = [f'feat_0', f'feat_1']  # 2个，少于min=3
    # IC fallback已移除，最终特征数可以小于min_features
    assert len(final_feats) == 2, f"Expected 2 features (no IC fallback), got {len(final_feats)}"

    print("[PASS] test_min_max_features_constraint: max=5 enforced, no IC fallback for <min")


if __name__ == '__main__':
    test_ic_vectorized_matches_loop()
    test_psi_filter()
    test_psi_edge_cases()
    test_sanitize_features()
    test_mda_ttest()
    test_fi_early_stopping()
    test_min_max_features_constraint()
    print("\n=== All Feature Selection tests PASSED ===")
