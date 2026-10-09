# -*- coding: utf-8 -*-
"""
V10 测试共享夹具：创建最小化策略实例 + 合成数据。
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

from strategies.wyckoff20.gpsef_strategy_v10 import (
    WyckoffV10Strategy,
    FactorNeutralizer,
    compute_sample_weights,
)

# ALL_FACTORS 是类属性，通过类访问
ALL_FACTORS = WyckoffV10Strategy.ALL_FACTORS


def make_strategy():
    """创建一个最小化 WyckoffV10Strategy 实例，绕过 IStrategy.initialize()。"""
    s = WyckoffV10Strategy.__new__(WyckoffV10Strategy)

    # 基本属性
    s.data_provider = None
    s.stock_pool = None
    s.config = {}
    s.target_stocks = []
    s.target_weights = {}
    s.trade_day_seq = 0
    s.models = []
    s.lgb_models = []                     # P1-11: LightGBM models
    s.feature_names = []
    s.val_auc = 0.0
    s.calibrator = None
    s._selected_features = []

    # 中性化
    s._neutralizer = FactorNeutralizer()
    s._industry_dummy_cols = []
    s._neutralize_enabled = True
    s._train_ols_betas = None
    s._industry_neutral = True
    s._market_cap_neutral = True
    s._full_market_neutralize = True

    # AUC门控 (对齐config_v10.yaml: full=0.52, min=0.48)
    s._auc_gate_enabled = True
    s._auc_gate_full = 0.52
    s._auc_gate_min = 0.48

    # CV参数
    s._cv_folds = 5
    s._cv_embargo_days = 21
    s._cv_feature_consensus = 3

    # 集成 (CV折集成 + 种子集成, 向后兼容)
    s._ensemble_enabled = True
    s._ensemble_method = 'seed_ensemble'  # 测试默认用种子集成
    s._ensemble_seeds = [42, 43, 44, 45, 46]
    s._ensemble_depth_list = [4, 4, 4, 4, 4]
    s._ensemble_subsample_list = [0.7, 0.7, 0.7, 0.7, 0.7]
    s._ensemble_colsample_list = [0.7, 0.7, 0.7, 0.7, 0.7]
    s._ensemble_metric_list = ['auc', 'auc', 'auc', 'auc', 'auc']
    s._ensemble_weight_method = 'uniform'
    s._ensemble_min_auc = 0.50
    s._ensemble_sample_weight = False
    s._cv_fold_defs = []
    s.model_aucs = []
    s.model_weights = []
    s._active_models = []

    # 校准
    s._calibration_method = 'platt'       # P0-2: 改为 Platt Scaling 默认
    s._prob_clip_min = 0.05
    s._prob_clip_max = 0.95
    s._calibration_monitoring = True     # P0-2: 校准质量监控

    # 训练配置
    s._class_balance_mode = 'auto'
    s._class_balance_max_scale = 5.0      # P0-3: 上限保护
    s._class_balance_auto_switch = True   # P0-3: 自动切换
    s._shap_enabled = False  # 测试中关闭 SHAP
    s._shap_monotonicity_threshold = 0.30
    s._shap_outlier_threshold = 0.30
    s._shap_sample_size = 1000
    s._shap_output_dir = './results/test_shap'
    s._combined_enabled = False
    s._combined_alpha = 0.5
    s._lr_warmup_cb = None                # LR warmup callback (test: disabled)
    s._lr_warmup_enabled = False

    # 特征选择参数
    s._feat_sel_nan_threshold = 0.5
    s._feat_sel_max_corr = 0.90
    s._feat_sel_psi_threshold = 0.15
    s._feat_sel_min_ic = 0.001
    s._feat_sel_mdi_n_estimators = 100  # 测试用小值
    s._feat_sel_mda_n_repeats = 5       # 测试用小值
    s._feat_sel_mda_n_jobs = -1
    s._feat_sel_mdi_nthread = 1
    s._min_features = 3
    s._max_features = 20
    s._feat_sel_variance_threshold = 1e-5
    s._cv_ic_psi_split_ratio = 0.5

    # 特征选择 v2.0 参数 (对齐Alpha191 feature_selection)
    s._fs_use_voting = True
    s._fs_mda_enabled = True
    s._fs_mda_n_repeats = 5
    s._fs_mda_threshold = 0.0
    s._fs_mda_use_std_threshold = True
    s._fs_mda_t_alpha = 0.05
    s._fs_mda_random_state = 42
    s._fs_vote_min_folds = 3
    s._fs_vote_min_features = 3
    s._fs_top_k_features = 20
    s._fs_weight_ic = 1.0
    s._fs_weight_mdi = 0.3
    s._fs_weight_mda = 1.5
    s._fs_fallback_to_ic_only = True

    # 预测参数
    s.pred_weight_temperature = 0.3
    s.early_stopping_rounds = 10

    # XGBoost参数（测试用小值）
    s.xgb_params = {
        'max_depth': 3,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'alpha': 0.3,
        'lambda': 1.0,
        'gamma': 0.0,
        'seed': 42,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'device': 'cpu',
        'nthread': 1,
    }

    # 训练参数
    s.rebalance_interval = 20
    s.train_window = 200
    s.pred_days = 20
    s.top_n = 300
    s.max_positions = 50
    s.max_position_pct = 0.10
    s.top_quantile = 0.10
    s.ref_date = datetime(2026, 5, 22)
    s._ref_date = datetime(2026, 5, 22)
    s.snapshot_dir = './data/wyckoff20_v5_snapshots'
    
    # 性能优化: 预构建标签查询索引
    s._daily_pivot_raw = None
    s._daily_pivot_adj = None
    s._daily_pivot_close = None
    s._validation_pipeline = None

    # IV
    s.iv_enabled = False
    s.iv_controller = None
    s._iv_scale_applied = 1.0
    s._iv_reduction_pending = {}

    s._label_threshold_method = 'median'
    s.full_df_daily = None
    s.calendar = None
    s.full_calendar = None

    # Barra 约束 (默认关闭)
    s._barra_enabled = False
    s._barra_method = 'factor_neutral'
    s._barra_lambda_dev = 0.5
    s._barra_lambda_l2 = 0.02
    s._barra_max_weight = 0.10
    s._barra_min_weight = 0.01
    s._barra_expected_ic = 0.05
    s._barra_vol_target = 0.15
    s._barra_min_positions = 20

    # 行业敞口限制 (P1-8)
    s._ind_constraint_enabled = True
    s._ind_max_weight = 0.30
    s._ind_fallback = 'unknown'

    s._lgb_enabled = False               # P1-11: LightGBM disabled in tests by default (avoid dep)
    s._lgb_ensemble_weight_xgb = 0.6
    s._lgb_ensemble_weight_lgb = 0.4
    s._lgb_params = {
        'num_leaves': 31, 'learning_rate': 0.03, 'subsample': 0.7,
        'colsample_bytree': 0.7, 'min_child_samples': 50,
        'reg_alpha': 0.1, 'reg_lambda': 0.5, 'n_estimators': 1000,
        'early_stopping_rounds': 100, 'nthread': 1,
    }

    return s


def make_synthetic_train_data(n_days=60, n_stocks=50, n_feats=10, seed=42):
    """
    生成合成训练数据（含 snapshot_date, code, label, 特征列）。
    """
    rng = np.random.RandomState(seed)
    base_date = datetime(2025, 1, 1)

    records = []
    for d in range(n_days):
        dt = base_date + timedelta(days=d)
        for s in range(n_stocks):
            code = f"sh.{600000 + s:06d}"
            row = {
                'snapshot_date': dt,
                'code': code,
                'label': rng.randint(0, 2),
            }
            for f in range(n_feats):
                row[f'feat_{f}'] = rng.randn() + f * 0.1
            records.append(row)

    df = pd.DataFrame(records)
    return df


def make_feature_df(n_samples=200, n_feats=10, seed=42, with_signal=True):
    """生成纯特征DataFrame（不含label），用于IC/MDA测试。"""
    rng = np.random.RandomState(seed)
    feats = {}
    for f in range(n_feats):
        if with_signal and f < 3:
            # 前3个特征有信号
            feats[f'feat_{f}'] = rng.randn(n_samples) + f * 0.5
        else:
            feats[f'feat_{f}'] = rng.randn(n_samples)

    df = pd.DataFrame(feats)
    # 生成标签：与有信号的特征正相关
    if with_signal:
        signal = sum(feats[f'feat_{f}'] for f in range(3)) / 3
        label = (signal > np.median(signal)).astype(int)
    else:
        label = rng.randint(0, 2, n_samples)

    return df, label
