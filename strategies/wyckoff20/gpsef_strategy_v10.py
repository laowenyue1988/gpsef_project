# -*- coding: utf-8 -*-
"""
Wyckoff V10 策略 — 独立版（V5已合并，不再继承V5）

V10 核心特性：
    1. 二分类标签（跑赢截面中位数→1）
    2. Purged K-Fold CV（K=5, embargo=21天, 跨折共识投票）
    3. CV折模型集成（每折独立验证集训练, 超参数多样性, AUC加权投票, 对齐Alpha191）
    4. Isotonic概率校准
    5. AUC门控（三档：full>=0.52/half>=0.48/empty<0.48, 对齐Alpha191）
    6. MDA t检验（n_repeats=20, t_crit自适应阈值）
    7. PSI过滤（threshold=0.15）
    8. 向量化IC计算（矩阵运算10x加速）
    9. 全市场中性化（训练-预测分布一致）
    10. 固定20日周期调仓 + 趋势过滤 + IV仓位控制

对齐Alpha191效率优化：向量化IC、MDA并行、DMatrix优化、全市场中性化

Author: BULLION GPSEF Framework
Date: 2026-06-24
Version: 10.1.0
"""

import os


import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict

import xgboost as xgb
from scipy.stats import spearmanr, t as t_dist
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from concurrent.futures import ThreadPoolExecutor

# LightGBM 异质集成 (P1-11: 可选依赖, 不可用时降级)
try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

# [CFI] 簇级置换重要性 (Prado MLAM Ch.6) — 复用 alpha191 系共享 ensemble 的纯函数
# (15 个 alpha191 策略 ensemble.py 同源; V10 特征选择/诊断与 alpha191 对齐)
try:
    from strategies.alpha191.ensemble import (
        cluster_features_global, compute_mda_oos_clustered,
        aggregate_cluster_votes, select_cluster_reprs, cfi_diagnose,
        cluster_features,  # [对齐 alpha191] 投票后聚类去冗余 (A 组流程)
        _objective_metric_fn,
        select_stable_features,  # [2026-08-20 对齐 D3] 折内 IC/PSI 再筛选
    )
    _CFI_AVAILABLE = True
except Exception as _cfi_imp_err:  # pragma: no cover
    _CFI_AVAILABLE = False
    _cfi_imp_err = _cfi_imp_err

from gpsef.core import IStrategy, StrategyContext
from gpsef.utils.neutralizer import FactorNeutralizer as _FrameworkNeutralizer

# Barra 约束优化（可降级，不强制依赖）
try:
    from strategies.wyckoff20.barra_constraint import solve_barra_weights, softmax_weights as _barra_softmax
    from strategies.wyckoff20.barra_builder import get_or_build_snapshot
    _BARRA_AVAILABLE = True
except ImportError:
    _BARRA_AVAILABLE = False

logger = logging.getLogger(__name__)



# ============================================================================
# XGBoost LR Warmup Callback (防止高学习率早期过拟合噪声)
# ============================================================================

class LRWarmupCallback(xgb.callback.TrainingCallback):
    """XGBoost 学习率 warmup callback (兼容 XGBoost >= 2.0)。

    在训练前 warmup_rounds 轮内，学习率从 start_lr 线性递增到 target_lr。
    之后保持 target_lr 不变。

    使用方式:
        cb = LRWarmupCallback(warmup_rounds=20, start_lr=0.005, target_lr=0.05)
        xgb.train(..., callbacks=[cb])

    原理:
        高 lr (如 0.05) 的前几棵树可能对噪声过度拟合，导致早停提前触发。
        warmup 让初期树以小步长拟合 → 避免早期噪声过度拟合 →
        让模型在 warmup 期后能稳定利用高 lr 的优势。
    """

    def __init__(self, warmup_rounds: int = 20, start_lr: float = 0.005,
                 target_lr: float = 0.05):
        super().__init__()
        self.warmup_rounds = max(1, warmup_rounds)
        self.start_lr = float(start_lr)
        self.target_lr = float(target_lr)

    def after_iteration(self, model: xgb.Booster, epoch: int,
                        evals_log: dict) -> bool:
        """每轮迭代后更新学习率。"""
        progress = min(1.0, epoch / max(1, self.warmup_rounds))
        lr = self.start_lr + (self.target_lr - self.start_lr) * progress
        model.set_param('eta', str(lr))
        return False  # 不提前停止训练


# ============================================================================
#  FactorNeutralizer（兼容层：Winsorize+Z-score预处理 + 框架OLS残差）
# ============================================================================

class FactorNeutralizer:
    """
    因子中性化器。

    处理流程：Winsorize(1%/99%) → Z-score → 框架OLS残差回归。
    """

    def __init__(self, winsorize_pct: Tuple[float, float] = (1, 99)):
        self.winsorize_pct = winsorize_pct
        self._framework_neutralizer = _FrameworkNeutralizer()
        # P0-1 FIX: stored training standardization params {col: {win_lower, win_upper, z_mean, z_std}}
        self._fit_params: Dict[str, Dict[str, float]] = {}

    def fit(self, df: pd.DataFrame, factor_cols: list):
        """P0-1 FIX (De Prado AFML Ch.5): fit standardization params from training data."""
        self._fit_params = {}
        for col in factor_cols:
            if col not in df.columns:
                continue
            raw = df[col].replace([np.inf, -np.inf], np.nan)
            valid = raw.dropna()
            if len(valid) < 10:
                self._fit_params[col] = {'win_lower': -np.inf, 'win_upper': np.inf, 'z_mean': 0.0, 'z_std': 1.0}
                continue
            lower, upper = np.percentile(valid, self.winsorize_pct)
            clipped = valid.clip(lower=lower, upper=upper)
            z_mean = float(clipped.mean())
            z_std = float(clipped.std())
            if z_std < 1e-8:
                z_std = 1.0
            self._fit_params[col] = {'win_lower': lower, 'win_upper': upper, 'z_mean': z_mean, 'z_std': z_std}
        logger.info(f"[FactorNeutralizer] fit: stored params for {len(self._fit_params)} factors")

    def _winsorize_apply(self, series: pd.Series, col: str) -> pd.Series:
        """P0-1: Winsorize using stored training params."""
        if col not in self._fit_params:
            return self._winsorize(series)
        p = self._fit_params[col]
        return series.clip(lower=p['win_lower'], upper=p['win_upper'])

    def _zscore_apply(self, series: pd.Series, col: str) -> pd.Series:
        """P0-1: Z-score using stored training params."""
        if col not in self._fit_params:
            return self._zscore(series)
        p = self._fit_params[col]
        return (series - p['z_mean']) / p['z_std']

    def _winsorize(self, series: pd.Series) -> pd.Series:
        valid = series.dropna()
        if len(valid) < 10:
            return series
        lower, upper = np.percentile(valid, self.winsorize_pct)
        return series.clip(lower=lower, upper=upper)

    def _zscore(self, series: pd.Series) -> pd.Series:
        mean = series.mean()
        std = series.std()
        if std < 1e-8:
            return series - mean
        return (series - mean) / std

    def neutralize(self, df: pd.DataFrame,
                   factor_cols: List[str], X_neutral: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in factor_cols:
            if col not in df.columns:
                continue
            raw = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = self._zscore_apply(self._winsorize_apply(raw, col), col)
        df = self._framework_neutralizer.neutralize(df, factor_cols, X_neutral)
        return df

    def neutralize_fit(self, df, factor_cols, X_neutral):
        """Fit neutralize: winsorize+zscore params from data, then regress out X_neutral.
        Returns (df_with_residuals, betas_info_dict).
        Uses _neutralize_vectorized (monkey-patched for determinism)."""
        df = df.copy()
        self.fit(df, factor_cols)
        for col in factor_cols:
            if col not in df.columns:
                continue
            raw = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = self._zscore_apply(self._winsorize_apply(raw, col), col)
        df, betas_info = self._framework_neutralizer._neutralize_vectorized(df, factor_cols, X_neutral, capture_betas=True)
        return df, betas_info


    def neutralize_apply(self, df: pd.DataFrame, factor_cols: list,
                         X_neutral: pd.DataFrame, betas_info: dict) -> pd.DataFrame:
        df = df.copy()
        for col in factor_cols:
            if col not in df.columns:
                continue
            raw = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = self._zscore_apply(self._winsorize_apply(raw, col), col)
        df = self._framework_neutralizer.neutralize_apply(
            df, factor_cols, X_neutral, betas_info)
        return df


# ============================================================================
#  XGBoost Booster sklearn兼容包装器
# ============================================================================

class XGBRegressorWrapper(RegressorMixin, BaseEstimator):
    """sklearn-compatible wrapper for xgb.Booster (regression task)."""

    def __init__(self, booster):
        self._booster = booster
        self.is_fitted_ = True
        self._feature_names = getattr(booster, 'feature_names', None)

    def predict(self, X):
        dmatrix = xgb.DMatrix(X, feature_names=self._feature_names)
        return self._booster.predict(dmatrix)

    def fit(self, X, y, **kwargs):
        return self


class XGBClassifierWrapper(ClassifierMixin, BaseEstimator):
    """sklearn-compatible wrapper for xgb.Booster (classification task)."""

    def __init__(self, booster):
        self._booster = booster
        self.is_fitted_ = True
        self._feature_names = getattr(booster, 'feature_names', None)
        self.classes_ = np.array([0, 1])

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        dmatrix = xgb.DMatrix(X, feature_names=self._feature_names)
        p1 = self._booster.predict(dmatrix)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def fit(self, X, y, **kwargs):
        return self


# ============================================================================
#  序贯采样权重计算（De Prado AFML Ch.7）
# ============================================================================

def compute_sample_weights(train_data, calendar, max_hold_days):
    """
    为每个训练样本计算基于平均唯一性的样本权重（高效版）。

    同一快照日的所有样本共享相同的窗口，按快照日分组计算，
    O(U*T) 复杂度，U=唯一快照日数。
    """
    if train_data.empty:
        return np.ones(len(train_data))

    N = len(train_data)
    snap_counts = train_data.groupby(train_data['snapshot_date']).size()
    unique_snaps = snap_counts.index

    cal_ts = pd.DatetimeIndex(sorted(calendar))
    T = len(cal_ts)

    snap_windows = {}
    for snap_date in unique_snaps:
        snap_ts = pd.Timestamp(snap_date)
        start_idx = cal_ts.searchsorted(snap_ts)
        if start_idx >= T:
            continue
        end_idx = min(start_idx + max_hold_days, T - 1)
        snap_windows[snap_date] = (start_idx, end_idx)

    concurrency = np.zeros(T, dtype=np.float64)
    for snap_date, (s, e) in snap_windows.items():
        n_samples = snap_counts[snap_date]
        concurrency[s:e + 1] += n_samples
    concurrency[concurrency == 0] = 1.0

    snap_avg_u = {}
    for snap_date, (s, e) in snap_windows.items():
        if s <= e:
            snap_avg_u[snap_date] = np.sum(1.0 / concurrency[s:e + 1]) / T
        else:
            snap_avg_u[snap_date] = 1.0 / T

    weights = np.array([snap_avg_u.get(sd, 1.0) for sd in train_data['snapshot_date']])

    w_mean = weights.mean()
    if w_mean > 0:
        weights = weights / w_mean

    return weights


# ============================================================================
#  主策略类
# ============================================================================

class WyckoffV10Strategy(IStrategy):
    """
    Wyckoff V10: 独立版（V5已合并）

    40因子 + 二分类标签 + Purged K-Fold CV + 5种子集成 +
    Isotonic校准 + AUC门控 + 趋势过滤 + IV仓位控制
    """

    # 40个因子名称
    ALL_FACTORS = [
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

    def __init__(self):
        self.data_provider = None
        self.stock_pool = None
        self.config = None
        self.target_stocks = []
        self.target_weights = {}
        self.trade_day_seq = 0

        # ML相关 — 集成模型
        self.models: List[xgb.Booster] = []
        self.lgb_models: List = []                    # P1-11: LightGBM 异质模型
        self.feature_names: List[str] = []
        self.val_auc: float = 0.0
        self.calibrator = None                         # Platt (sklearn) / Isotonic
        self._selected_features: List[str] = []

        # 数据
        self.full_df_daily: Optional[pd.DataFrame] = None
        self.calendar = None
        self.full_calendar = None
        self.snapshot_dir = None

        # 中性化
        self._neutralizer = FactorNeutralizer()
        self._industry_dummy_cols: List[str] = []
        self._neutralize_enabled: bool = True
        self._train_ols_betas = None
        self._score_neu_params = {}

        # IV仓位控制
        self.iv_enabled: bool = False
        self.iv_controller = None
        self._iv_scale_applied: float = 1.0
        self._iv_reduction_pending: Dict[str, float] = {}

        # 复权基准日
        self._ref_date: Optional[datetime] = None
        self._daily_pivot_raw = None     # 预构建标签索引 (initialize 中赋值)
        self._daily_pivot_adj = None
        self._daily_pivot_close = None

    # ================================================================
    #  初始化
    # ================================================================

    def initialize(self, context: StrategyContext):
        """策略初始化"""
        np.random.seed(42)
        config = context.config
        self.config = config
        self.data_provider = context.data_provider
        self.stock_pool = context.stock_pool
        self.output_dir = getattr(context, 'output_dir', './results/default')  # 回测输出目录
        self.ref_date = datetime.strptime(config['data']['reference_date'], '%Y%m%d')
        self._ref_date = self._get_fuquan_reference_date(context)

        # ML训练参数
        self.rebalance_interval = config.get('rebalance_interval', 20)
        # [重训间隔 2026-08-18 对齐 small] 每 N 次调仓重训一次, 期间沿用最近模型
        self.retrain_interval = config.get('retrain_interval', 1)
        self._rebalances_since_train = 0
        self.train_window = config.get('train_window_days', 200)
        self.pred_days = config.get('pred_days', 20)
        self.top_n = config.get('top_n', 300)
        # [P1-5 FIX] max_position_pct fallback 从 0.05 改为 0.10 (对齐 config_v10.yaml)
        # 同时优先读取顶层配置值, execution 段覆盖顶层
        _top_max_pct = config.get('max_position_pct', 0.10)
        self.max_positions = config['execution'].get('max_positions', 50)
        self.max_position_pct = config['execution'].get('max_position_pct', _top_max_pct)
        self.top_quantile = config.get('top_quantile', 0.10)

        # 中性化配置
        neu_cfg = config.get('neutralization', {})
        self._neutralize_enabled = neu_cfg.get('enabled', True)
        winsorize_pct = tuple(neu_cfg.get('winsorize_pct', [1, 99]))
        self._neutralizer = FactorNeutralizer(winsorize_pct=winsorize_pct)
        self._train_ols_betas = None
        self._industry_neutral = neu_cfg.get('industry_neutral', True)
        self._market_cap_neutral = neu_cfg.get('market_cap_neutral', True)
        self._full_market_neutralize = neu_cfg.get('full_market_neutralize', True)  # P1-4

        # AUC门控参数 (P0-2: 替换IC门控, 对齐Alpha191)
        # [P1-1 FIX] fallback 值从 0.50/0.55 改为 0.48/0.52, 与 config_v10.yaml 保持一致
        self._auc_gate_enabled = config.get('auc_gate_enabled', True)
        self._auc_gate_full = config.get('auc_gate_full', 0.52)   # was 0.55
        self._auc_gate_min = config.get('auc_gate_min', 0.48)     # was 0.50

        # 概率坍缩检测: 模型无区分力时自动跳过选股
        _collapse_threshold = config.get('prob_collapse_std_threshold', 0.003)
        self._prob_collapse_threshold = float(_collapse_threshold) if _collapse_threshold else 0.0
        if self._prob_collapse_threshold < 0:
            self._prob_collapse_threshold = 0.003

        # Purged K-Fold CV参数 (P0-1)
        self._cv_folds = config.get('cv_folds', 5)
        self._cv_embargo_days = config.get('cv_embargo_days', 21)
        self._cv_feature_consensus = config.get('cv_feature_consensus', 3)

        # 集成参数 (CV折集成 + 超参数多样性 + AUC加权 + LightGBM异质, 对齐Alpha191)
        ensemble_cfg = config.get('ensemble', {})
        if isinstance(ensemble_cfg, dict) and ensemble_cfg:
            self._ensemble_enabled = ensemble_cfg.get('enabled', True)
            self._ensemble_method = ensemble_cfg.get('method', 'cv_fold')
            self._ensemble_seeds = ensemble_cfg.get('seeds', [42, 43, 44, 45, 46])
            diversity_cfg = ensemble_cfg.get('diversity', {})
            self._ensemble_depth_list = diversity_cfg.get('max_depth_list', [4, 4, 4, 4, 4])
            self._ensemble_subsample_list = diversity_cfg.get('subsample_list', [0.7, 0.7, 0.7, 0.7, 0.7])
            self._ensemble_colsample_list = diversity_cfg.get('colsample_bytree_list', [0.7, 0.7, 0.7, 0.7, 0.7])
            self._ensemble_metric_list = diversity_cfg.get('eval_metric_list', ['auc', 'auc', 'auc', 'auc', 'auc'])
            weighting_cfg = ensemble_cfg.get('weighting', {})
            self._ensemble_weight_method = weighting_cfg.get('method', 'auc_weighted')
            self._ensemble_min_auc = weighting_cfg.get('min_auc_threshold', 0.50)
            self._ensemble_sample_weight = ensemble_cfg.get('sample_weight', False)
            # LightGBM 异质集成配置 (P1-11)
            lgb_cfg = ensemble_cfg.get('lightgbm', {})
            self._lgb_enabled = lgb_cfg.get('enabled', True) and _LGB_AVAILABLE
            if self._lgb_enabled and not _LGB_AVAILABLE:
                self._lgb_enabled = False
            self._lgb_params = {
                'num_leaves': lgb_cfg.get('num_leaves', 31),
                'learning_rate': lgb_cfg.get('learning_rate', 0.03),
                'subsample': lgb_cfg.get('subsample', 0.7),
                'colsample_bytree': lgb_cfg.get('colsample_bytree', 0.7),
                'min_child_samples': lgb_cfg.get('min_child_samples', 50),
                'reg_alpha': lgb_cfg.get('reg_alpha', 0.1),
                'reg_lambda': lgb_cfg.get('reg_lambda', 0.5),
                'n_estimators': lgb_cfg.get('n_estimators', 1000),
                'early_stopping_rounds': lgb_cfg.get('early_stopping_rounds', 100),
                'nthread': lgb_cfg.get('nthread', 1),
            }
            self._lgb_ensemble_weight_xgb = lgb_cfg.get('ensemble_weight_xgb', 0.6)
            self._lgb_ensemble_weight_lgb = lgb_cfg.get('ensemble_weight_lgb', 0.4)
        else:
            self._ensemble_enabled = True
            self._ensemble_method = 'seed_ensemble'
            self._ensemble_seeds = [42, 43, 44, 45, 46]
            self._ensemble_depth_list = []
            self._ensemble_subsample_list = []
            self._ensemble_colsample_list = []
            self._ensemble_metric_list = []
            self._ensemble_weight_method = 'uniform'
            self._ensemble_min_auc = 0.50
            self._ensemble_sample_weight = False
            self._lgb_enabled = _LGB_AVAILABLE

        # CV折定义 (由 _purged_kfold_cv 填充, 供 CV 折集成使用)
        self._cv_fold_defs: List[Tuple] = []

        # 集成模型元信息
        self.model_aucs: List[float] = []       # 每折/每种子验证AUC
        self.model_weights: List[float] = []    # 集成投票权重
        self._active_models: List[int] = []     # 有效模型索引

        # Barra 基准偏离约束 (组合优化层)
        barra_cfg = config.get('barra_constraint', {})
        if isinstance(barra_cfg, dict):
            self._barra_enabled = barra_cfg.get('enabled', False)
            self._barra_method = barra_cfg.get('method', 'factor_neutral')
            self._barra_lambda_dev = barra_cfg.get('lambda_deviation', 0.5)
            self._barra_lambda_l2 = barra_cfg.get('lambda_l2', 0.02)
            self._barra_max_weight = barra_cfg.get('max_weight', 0.10)
            self._barra_min_weight = barra_cfg.get('min_weight', 0.01)
            self._barra_expected_ic = barra_cfg.get('expected_ic', 0.05)
            self._barra_vol_target = barra_cfg.get('vol_target', 0.15)
            self._barra_min_positions = barra_cfg.get('min_positions_for_constraint', 20)
        else:
            self._barra_enabled = False

        # 概率校准与裁剪 (P0-2: platt/isotonic/none)
        self._calibration_method = config.get('calibration_method', 'platt')
        self._prob_clip_min = config.get('prob_clip_min', 0.05)
        self._prob_clip_max = config.get('prob_clip_max', 0.95)
        self._calibration_monitoring = config.get('calibration_monitoring', True)

        # 行业敞口限制 (P1-8: 防止单一行业集中度风险)
        ind_cfg = config.get('industry_constraint', {})
        self._ind_constraint_enabled = ind_cfg.get('enabled', True)
        self._ind_max_weight = ind_cfg.get('max_industry_weight', 0.30)
        self._ind_fallback = ind_cfg.get('fallback_industry', 'unknown')

        # XGBoost 类别平衡上限 (P0-3: 防止极端不均衡过拟合)
        self._class_balance_max_scale = config.get('xgb_class_balance_max_scale', 5.0)
        self._class_balance_auto_switch = config.get('xgb_class_balance_auto_switch', True)

        # 因子筛选参数
        self._feat_sel_nan_threshold = config.get('feat_sel_nan_threshold', 0.5)
        self._feat_sel_max_corr = config.get('feat_sel_max_corr', 0.90)
        self._feat_sel_psi_threshold = config.get('feat_sel_psi_threshold', 0.15)
        self._feat_sel_min_ic = config.get('feat_sel_min_ic', 0.001)
        self._feat_sel_mdi_n_estimators = config.get('feat_sel_mdi_n_estimators', 500)
        self._feat_sel_mda_n_repeats = config.get('feat_sel_mda_n_repeats', 20)
        self._feat_sel_mda_n_jobs = config.get('feat_sel_mda_n_jobs', -1)
        self._feat_sel_mdi_nthread = config.get('feat_sel_mdi_nthread', 1)
        self._min_features = config.get('feat_sel_min_features', 10)   # P3-4
        self._max_features = config.get('feat_sel_max_features', 80)   # P3-4
        self._feat_sel_variance_threshold = config.get('feat_sel_variance_threshold', 1e-5)
        self._cv_ic_psi_split_ratio = config.get('cv_ic_psi_split_ratio', 0.5)

        # 特征选择 v2.0 参数 (对齐Alpha191 feature_selection结构)
        fs_cfg = config.get('feature_selection', {})
        self._fs_use_voting = fs_cfg.get('use_voting', True)

        mda_cfg = fs_cfg.get('mda', {})
        self._fs_mda_enabled = mda_cfg.get('enabled', True)
        self._fs_mda_n_repeats = mda_cfg.get('n_repeats', 20)
        self._fs_mda_threshold = mda_cfg.get('threshold', 0.0)
        self._fs_mda_use_std_threshold = mda_cfg.get('use_std_threshold', True)
        self._fs_mda_t_alpha = mda_cfg.get('t_alpha', 0.05)
        self._fs_mda_random_state = mda_cfg.get('random_state', 42)

        vote_cfg = fs_cfg.get('voting', {})
        self._fs_vote_min_folds = vote_cfg.get('vote_min_folds', 3)
        self._fs_vote_min_features = vote_cfg.get('vote_min_features', 10)
        self._fs_top_k_features = vote_cfg.get('top_k_features', 80)
        self._fs_weight_ic = vote_cfg.get('weight_ic', 1.0)
        self._fs_weight_mdi = vote_cfg.get('weight_mdi', 0.3)
        self._fs_weight_mda = vote_cfg.get('weight_mda', 1.5)
        self._fs_fallback_to_ic_only = vote_cfg.get('fallback_to_ic_only', True)

        # [CFI] 簇级置换重要性 (Prado MLAM Ch.6, 对齐 alpha191):
        # cfi_mode=true 时特征选择改用簇级置换投票 (B组流程); diagnose=true 时训练后输出簇级重要性分布
        clust_cfg = fs_cfg.get('feature_clustering', {})
        self._cluster_cfi_enabled = bool(clust_cfg.get('enabled', False)) and \
                                    bool(clust_cfg.get('cfi_mode', False))
        self._cluster_legacy = bool(clust_cfg.get('legacy_threshold', False))
        self._cluster_diagnose = bool(clust_cfg.get('diagnose', False))
        self._cluster_corr_threshold = float(clust_cfg.get('corr_threshold', 0.70))
        self._cluster_corr_method = clust_cfg.get('corr_method', 'spearman')
        self._cluster_min_size = int(clust_cfg.get('min_cluster_size', 2))
        self._cluster_log = bool(clust_cfg.get('log_clusters', True))
        self._cluster_repr_method = clust_cfg.get('repr_method', 'sfi')

        # 预测值加权温度 (P1-7: 0.1→0.3)
        self.pred_weight_temperature = config.get('pred_weight_temperature', 0.3)

        # SHAP 因子质量验证
        shap_cfg = config.get('shap', {})
        self._shap_enabled = shap_cfg.get('enabled', True)
        self._shap_sample_size = shap_cfg.get('sample_size', 8000)
        self._shap_monotonicity_threshold = shap_cfg.get('monotonicity_threshold', 0.30)
        self._shap_outlier_threshold = shap_cfg.get('outlier_threshold', 0.30)
        self._shap_output_dir = os.path.join(self.output_dir, 'shap_analysis')

        # early_stopping
        self.early_stopping_rounds = config.get('xgb_early_stopping_rounds', 20)

        # LR Warmup 学习率调度
        self._lr_warmup_enabled = config.get('lr_warmup_enabled', False)
        if self._lr_warmup_enabled:
            warmup_rounds = config.get('lr_warmup_rounds', 20)
            warmup_start = config.get('lr_warmup_start', 0.005)
            warmup_target = config.get('lr_warmup_target',
                                       config.get('xgb_learning_rate', 0.02))
            self._lr_warmup_cb = LRWarmupCallback(
                warmup_rounds=int(warmup_rounds),
                start_lr=float(warmup_start),
                target_lr=float(warmup_target)
            )
            logger.info(f"[config] LR Warmup: rounds={warmup_rounds}, "
                        f"{warmup_start}→{warmup_target} (linear)")
        else:
            self._lr_warmup_cb = None

        # XGBoost参数 (对齐Alpha191最佳实践)
        self.xgb_params = {
            'max_depth': config.get('xgb_max_depth', 4),
            'eta': config.get('xgb_learning_rate', 0.02),
            'subsample': config.get('xgb_subsample', 0.7),
            'colsample_bytree': config.get('xgb_colsample_bytree', 0.7),
            'min_child_weight': config.get('xgb_min_child_weight', 10),
            'alpha': config.get('xgb_reg_alpha', 0.3),
            'lambda': config.get('xgb_reg_lambda', 1.0),
            'gamma': config.get('xgb_gamma', 0.0),
            'seed': config.get('xgb_random_state', 42),
            'objective': 'binary:logistic',
            'eval_metric': config.get('xgb_eval_metric', 'auc'),
            'tree_method': config.get('xgb_tree_method', 'hist'),   # P2-4
            'device': config.get('xgb_device', 'cpu'),              # P2-4
            'nthread': config.get('xgb_nthread', 1),
        }

        # 分类标签设定
        v10_cfg = config.get('v10', {})
        # 向后兼容：旧配置可能用 label_threshold 而非 label_method
        self._label_method = v10_cfg.get('label_method',
                                         v10_cfg.get('label_threshold', 'median'))
        self._label_quantile = float(v10_cfg.get('label_quantile', 0.30))
        self._label_absolute_threshold = float(v10_cfg.get('label_absolute_threshold', 0.10))
        # 合法性校验
        if self._label_method not in ('median', 'quantile', 'absolute'):
            logger.warning(f"[config] v10.label_method='{self._label_method}' 非法，"
                           f"回退为 'median'")
            self._label_method = 'median'
        if self._label_method == 'quantile' and not (0.0 < self._label_quantile < 1.0):
            logger.warning(f"[config] v10.label_quantile={self._label_quantile} 非法，"
                           f"回退为 0.30")
            self._label_quantile = 0.30
        if self._label_method == 'absolute' and not (-1.0 < self._label_absolute_threshold < 10.0):
            logger.warning(f"[config] v10.label_absolute_threshold={self._label_absolute_threshold} 非法，"
                           f"回退为 0.10")
            self._label_absolute_threshold = 0.10

        # 类别平衡 (对抗标签不均衡)
        _balance_cfg = config.get('xgb_class_balance', 'auto')
        self._class_balance_mode = _balance_cfg if _balance_cfg in ('auto', 'none') else 'auto'
        logger.info(f"[config] class_balance={self._class_balance_mode}")

        # 多目标联合优化 (evals_result + ntree_limit)
        self._combined_enabled = config.get('xgb_combined_enabled', False)
        self._combined_alpha = float(config.get('xgb_combined_alpha', 0.5))
        if self._combined_alpha < 0.0 or self._combined_alpha > 1.0:
            logger.warning(f"[config] xgb_combined_alpha={self._combined_alpha} 非法，回退 0.5")
            self._combined_alpha = 0.5
        # combined 模式下强制 eval_metric=['logloss','auc'] 以捕获两者到 evals_result
        # 不 override 原配置: 保持 self.xgb_params['eval_metric'] 不变用于日志
        if self._combined_enabled:
            logger.info(f"[config] combined enabled, alpha={self._combined_alpha}, "
                        f"eval_metric=['logloss','auc']")

        # 快照目录
        snapshot_cfg = config.get('snapshot', {})
        self.snapshot_dir = snapshot_cfg.get('factor_path', './data/wyckoff20_v5_snapshots')

        self.calendar = context.calendar
        self.full_calendar = context.full_calendar

        # IV仓位控制初始化
        iv_cfg = config.get('iv_control', {})
        self.iv_enabled = iv_cfg.get('enabled', False)
        if self.iv_enabled:
            from gpsef.position.iv_controller import IVPositionController
            self.iv_controller = IVPositionController({
                'enabled': True,
                'iv': {
                    'source': iv_cfg.get('source', 'parquet'),
                    'data_path': iv_cfg.get('data_path', './data/iv_index.parquet'),
                    'low_pct': iv_cfg.get('low_pct', 0.25),
                    'high_pct': iv_cfg.get('high_pct', 0.75),
                    'min_scale': iv_cfg.get('min_scale', 0.25),
                    'window': iv_cfg.get('window', 252),
                }
            })
            self.iv_controller.initialize_for_backtest()
            logger.info(f"  IV position control: ENABLED")
        else:
            logger.info(f"  IV position control: DISABLED")

        # recomd模式
        recomd_cfg = config.get('recomd', {})
        mode_from_config = config.get('mode', 'backtest')
        self._is_recomd = recomd_cfg.get('enabled', False) or (mode_from_config == 'recomd')
        if self._is_recomd:
            self.rebalance_interval = 1
            self.max_positions = recomd_cfg.get('max_positions', 300)
            self._recomd_snapshot_dir = recomd_cfg.get(
                'snapshot_dir', './data/wyckoff20_v10_recomd_snapshots')
            os.makedirs(self._recomd_snapshot_dir, exist_ok=True)
            self.iv_enabled = False

        # IV减仓冷却期 (P2-6)
        self._iv_cooldown_days = iv_cfg.get('cooldown_days', 5)

        # 预加载日线数据（必须在趋势过滤预计算之前）
        self._preload_daily_data(context)

        # 趋势过滤（依赖 full_df_daily，必须在 _preload_daily_data 之后）
        tf_cfg = config.get('trend_filter', {})
        if tf_cfg.get('enabled', False):
            self._precompute_trend_factors(tf_cfg)
        else:
            self._trend_factors_cache = None

        logger.info("=" * 60)
        logger.info("Wyckoff V10 Strategy initialized (Independent - V5 Merged)")
        logger.info(f"  objective: {self.xgb_params['objective']}")
        logger.info(f"  tree_method: {self.xgb_params['tree_method']}")
        logger.info(f"  max_depth={self.xgb_params['max_depth']}, lr={self.xgb_params['eta']}")
        logger.info(f"  min_child_weight={self.xgb_params['min_child_weight']}")
        logger.info(f"  CV: folds={self._cv_folds}, embargo={self._cv_embargo_days}d, consensus={self._cv_feature_consensus}")
        logger.info(f"  Ensemble: method={self._ensemble_method}, seeds={self._ensemble_seeds}")
        if self._ensemble_method == 'cv_fold':
            logger.info("[废弃 2026-08-18] config diversity 列表不再生效, "
                       "折集成多样性改为围绕 HP-Search 选中超参扰动 (对齐 small_alpha191)")
            logger.info(f"    weighting: {self._ensemble_weight_method}, "
                       f"min_auc={self._ensemble_min_auc}")
        logger.info(f"  AUC gate: enabled={self._auc_gate_enabled}, full>={self._auc_gate_full}, min>={self._auc_gate_min}")
        logger.info(f"  Calibration: {self._calibration_method}, clip=[{self._prob_clip_min},{self._prob_clip_max}]")
        logger.info(f"  Class Balance: auto_switch={self._class_balance_auto_switch}, max_scale={self._class_balance_max_scale}")
        logger.info(f"  Industry Constraint: {'ON' if self._ind_constraint_enabled else 'OFF'}, max_ind_weight={self._ind_max_weight}")
        logger.info(f"  LightGBM Ensemble: {'ON' if self._lgb_enabled else 'OFF'} (w_xgb={self._lgb_ensemble_weight_xgb}, w_lgb={self._lgb_ensemble_weight_lgb})")
        logger.info(f"  MDA: n_repeats={self._feat_sel_mda_n_repeats}, psi_threshold={self._feat_sel_psi_threshold}")
        logger.info(f"  Features: min={self._min_features}, max={self._max_features}, ic_min={self._feat_sel_min_ic}")
        logger.info(f"  Feature Selection: voting={self._fs_use_voting}, "
                    f"t_alpha={self._fs_mda_t_alpha}, weights=(ic={self._fs_weight_ic}, "
                    f"mdi={self._fs_weight_mdi}, mda={self._fs_weight_mda})")
        logger.info(f"  Voting: min_folds={self._fs_vote_min_folds}, "
                    f"min_features={self._fs_vote_min_features}, "
                    f"top_k={self._fs_top_k_features}, "
                    f"fallback_ic={self._fs_fallback_to_ic_only}")
        logger.info(f"  full_market_neutralize={self._full_market_neutralize}")
        logger.info(f"  iv_control: {'ON' if self.iv_enabled else 'OFF'}")
        logger.info("=" * 60)

    def _get_fuquan_reference_date(self, context: StrategyContext) -> datetime:
        """获取固定复权基准日"""
        data_cfg = context.config.get('data', {})
        backtest_cfg = context.config.get('backtest', {})
        ref_str = data_cfg.get('reference_date', backtest_cfg.get('end_date', ''))
        if ref_str:
            return datetime.strptime(ref_str, '%Y%m%d')
        return context.current_dt

    # ================================================================
    #  数据加载工具
    # ================================================================

    def _preload_daily_data(self, context: StrategyContext):
        """预加载全市场日线数据"""
        logger.info("Loading full market daily data...")
        dp = self.data_provider
        start = context.full_calendar[0]
        end = context.full_calendar[-1]
        codes = dp.get_stock_list()
        frames = []
        for code in codes:
            df = dp.get_daily_data(code, start, end, reference_date=self.ref_date)
            if df is None or df.empty:
                continue
            df = df[['trade_date', 'close', 'volume', 'amount', 'turn', 'raw_close', 'backAdjustFactor']].copy()
            df['ts_code'] = code
            frames.append(df)

        if not frames:
            raise RuntimeError("Failed to load any daily data")

        full_df = pd.concat(frames, ignore_index=True)
        full_df['trade_date'] = pd.to_datetime(full_df['trade_date'])
        full_df['close'] = pd.to_numeric(full_df['close'], errors='coerce')
        full_df = full_df.dropna(subset=['close'])
        full_df['code_digits'] = full_df['ts_code'].str.replace('sh.', '').str.replace('sz.', '')
        full_df.sort_values(['ts_code', 'trade_date'], inplace=True)
        full_df.reset_index(drop=True, inplace=True)
        self.full_df_daily = full_df
        logger.info(f"Full market data loaded: {len(full_df)} records")

        # 预构建标签查询索引: {date: {code: value}} 替代 O(N_row) 全表扫描
        # 性能: 5.6M 行扫描 → O(1) dict lookup (标签计算阶段节省 ~10分钟/回测)
        try:
            t0 = time.time()
            self._daily_pivot_raw = {}
            self._daily_pivot_adj = {}
            self._daily_pivot_close = {}
            pivot_fields = [('raw_close', '_daily_pivot_raw'),
                           ('backAdjustFactor', '_daily_pivot_adj'),
                           ('close', '_daily_pivot_close')]
            for field, attr in pivot_fields:
                pivot = full_df.pivot_table(
                    index='trade_date', columns='code_digits',
                    values=field, aggfunc='first')
                setattr(self, attr, {
                    pd.Timestamp(d): {str(c): float(v) for c, v in row.dropna().items()}
                    for d, row in pivot.iterrows()
                })
            logger.info(f"  Daily pivot index built: {len(self._daily_pivot_raw)} dates, {time.time()-t0:.1f}s")
        except Exception as e:
            logger.warning(f"  Daily pivot build failed ({e}), using fallback scan")
            self._daily_pivot_raw = None
            self._daily_pivot_adj = None
            self._daily_pivot_close = None

    def _get_trading_day_offset(self, base_date: datetime, offset: int) -> Optional[datetime]:
        cal = self.full_calendar
        try:
            idx = cal.index(base_date)
        except ValueError:
            for i, d in enumerate(cal):
                if d >= base_date:
                    idx = i
                    break
            else:
                return None
        target_idx = idx + offset
        if target_idx < len(cal):
            return cal[target_idx]
        return None

    def _get_previous_snapshot_date(self, current_dt: datetime) -> Optional[datetime]:
        if not os.path.exists(self.snapshot_dir):
            return None
        files = [f for f in os.listdir(self.snapshot_dir) if f.endswith('.parquet')]
        dates = []
        for f in files:
            try:
                d_str = f.replace('factors_', '').replace('.parquet', '')
                dates.append(datetime.strptime(d_str, '%Y%m%d'))
            except:
                pass
        dates.sort()
        for d in reversed(dates):
            if d <= current_dt:
                return d
        return None

    def _load_snapshot(self, date: datetime) -> Optional[pd.DataFrame]:
        fname = f'factors_{date.strftime("%Y%m%d")}.parquet'
        path = os.path.join(self.snapshot_dir, fname)
        if not os.path.exists(path):
            return None
        df = pd.read_parquet(path)
        if 'code' in df.columns:
            df['code'] = df['code'].astype(str).str.zfill(6)
        return df

    def _get_close_prices(self, codes, date):
        date = pd.Timestamp(date)
        codes = [str(c).zfill(6) for c in codes]
        # 优先使用预构建索引
        if (self._daily_pivot_close is not None and date in self._daily_pivot_close):
            close_map = self._daily_pivot_close[date]
            return {c: close_map.get(c, np.nan) for c in codes}
        # 回退: 全表扫描
        sub = self.full_df_daily[
            (self.full_df_daily['code_digits'].isin(codes)) &
            (self.full_df_daily['trade_date'] == date)]
        return dict(zip(sub['code_digits'], sub['close']))

    def _get_label_components(self, codes, date):
        """获取标签计算所需的前复权价格和调整因子。

        优化: 使用预构建的 dict 索引替代 full_df_daily 全表扫描。
        pivot index: O(1) per stock lookup vs O(N) table scan.
        """
        date = pd.Timestamp(date)
        codes = [str(c).zfill(6) for c in codes]

        # 优先使用预构建索引 (initialize 中构建, ~100MB 内存)
        if (self._daily_pivot_raw is not None
                and self._daily_pivot_adj is not None
                and date in self._daily_pivot_raw):
            raw_map = self._daily_pivot_raw[date]
            adj_map = self._daily_pivot_adj[date]
            return (
                {c: raw_map.get(c, np.nan) for c in codes},
                {c: adj_map.get(c, np.nan) for c in codes},
            )

        # 回退: 全表扫描 (pivot 构建失败或无此日期时)
        sub = self.full_df_daily[
            (self.full_df_daily['code_digits'].isin(codes)) &
            (self.full_df_daily['trade_date'] == date)]
        return (
            dict(zip(sub['code_digits'], sub['raw_close'])),
            dict(zip(sub['code_digits'], sub['backAdjustFactor']))
        )

    def _get_snapshot_dates(self):
        if not hasattr(self, '_cached_snap_dates'):
            if not os.path.exists(self.snapshot_dir):
                self._cached_snap_dates = []
            else:
                files = [f for f in os.listdir(self.snapshot_dir) if f.endswith('.parquet')]
                dates = []
                for f in files:
                    try:
                        d = f.replace('factors_', '').replace('.parquet', '')
                        dates.append(datetime.strptime(d, '%Y%m%d'))
                    except:
                        pass
                self._cached_snap_dates = sorted(dates)
        return self._cached_snap_dates

    # ================================================================
    #  中性化方法
    # ================================================================

    def _build_neutral_X(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        dummy_cols = []
        if self._industry_neutral and 'industry' in df.columns:
            categories = sorted(df['industry'].astype(str).unique())
            ind_dummies = pd.get_dummies(
                pd.Categorical(df['industry'].astype(str), categories=categories),
                prefix='ind')
            dummy_cols = ind_dummies.columns.tolist()
        else:
            ind_dummies = pd.DataFrame(index=df.index)

        if self._market_cap_neutral and 'total_mv' in df.columns:
            log_mv = np.log(df['total_mv'].clip(lower=1e-8))
            log_mv.name = 'log_mv'
        else:
            log_mv = pd.Series(0.0, index=df.index, name='log_mv')

        X_neutral = pd.concat([ind_dummies, log_mv], axis=1).fillna(0).astype(np.float64)
        return X_neutral, dummy_cols

    def _neutralize(self, df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
        if not self._neutralize_enabled:
            return df
        df = df.copy()
        if df.empty:
            return df

        # [Diag] X_neutral来源列MD5诊断
        import hashlib as _hl
        _ind_md5 = _hl.md5(np.array(df['industry'].astype(str)).tobytes()).hexdigest()[:12] if 'industry' in df.columns else 'N/A'
        _mv_md5 = _hl.md5(np.ascontiguousarray(df['total_mv'].astype(np.float64).values).tobytes()).hexdigest()[:12] if 'total_mv' in df.columns else 'N/A'
        # [Diag] 全量raw因子列MD5（覆盖所有列，不只前3列）
        _raw_cols_present = [c for c in factor_cols if c in df.columns]
        _all_raw_md5 = _hl.md5(np.ascontiguousarray(df[_raw_cols_present].values).tobytes()).hexdigest()[:16]
        logger.debug(f"[NeuDiag] industry_md5={_ind_md5}, total_mv_md5={_mv_md5}, all_raw_md5={_all_raw_md5}, rows={len(df)}, raw_cols={len(_raw_cols_present)}")

        X_neutral, _ = self._build_neutral_X(df)
        _xn_md5 = _hl.md5(np.ascontiguousarray(X_neutral.values).tobytes()).hexdigest()[:12]
        logger.debug(f"[NeuDiag] X_neutral_md5={_xn_md5}, cols={X_neutral.shape[1]}")

        if self._train_ols_betas is None:
            df, betas_info = self._neutralizer.neutralize_fit(df, factor_cols, X_neutral)
            self._train_ols_betas = betas_info
            # [Diag] OLS后中性化结果MD5 + betas MD5
            _neu_cols_present = [c + '_neu' for c in factor_cols if c + '_neu' in df.columns]
            if not _neu_cols_present:
                _neu_cols_present = [c for c in factor_cols if c in df.columns]
            _neu_md5 = _hl.md5(np.ascontiguousarray(df[_neu_cols_present].values.astype(np.float64)).tobytes()).hexdigest()[:16] if _neu_cols_present else 'N/A'
            _betas_md5 = _hl.md5(str(sorted(betas_info.get('betas', {}).keys())).encode() + str(sum(np.sum(np.abs(v)) for v in betas_info.get('betas', {}).values())).encode()).hexdigest()[:16] if betas_info.get('betas') else 'N/A'
            logger.debug(f"[NeuDiag/FIT] neu_result_md5={_neu_md5}, betas_detail_hash={_betas_md5}")
            logger.info(f"[neutralize/FIT] Captured OLS betas: "
                        f"{len(betas_info['betas'])} factors × {len(betas_info['X_cols'])} X columns")
        else:
            df = self._neutralizer.neutralize_apply(
                df, factor_cols, X_neutral, self._train_ols_betas)

        return df

    def _apply_score_neutralization(self, df: pd.DataFrame, score_cols: List[str], fit: bool = False) -> pd.DataFrame:
        df = df.copy()
        for col in score_cols:
            if col not in df.columns:
                continue
            series = df[col].replace([np.inf, -np.inf], np.nan)
            if not hasattr(self, '_score_neu_params'):
                self._score_neu_params = {}
            if fit:
                valid = series.dropna()
                if len(valid) < 10:
                    df[col] = series
                    continue
                lower, upper = np.percentile(valid, (1, 99))
                clipped = series.clip(lower=lower, upper=upper)
                mean = clipped.mean()
                std = clipped.std()
                if std < 1e-8:
                    std = 1.0
                self._score_neu_params[col] = {'win_lower': lower, 'win_upper': upper, 'z_mean': mean, 'z_std': std}
                series = (clipped - mean) / std
            elif col in self._score_neu_params:
                p = self._score_neu_params[col]
                clipped = series.clip(lower=p['win_lower'], upper=p['win_upper'])
                series = (clipped - p['z_mean']) / p['z_std']
            else:
                valid = series.dropna()
                if len(valid) < 10:
                    df[col] = series
                    continue
                lower, upper = np.percentile(valid, (1, 99))
                series = series.clip(lower=lower, upper=upper)
                mean = series.mean()
                std = series.std()
                if std > 1e-8:
                    series = (series - mean) / std
            df[col] = series
        return df

    # ================================================================
    #  因子筛选方法
    # ================================================================

    def _calc_psi(self, feat_tr: pd.Series, feat_va: pd.Series, fill_value=None) -> float:
        """计算 Population Stability Index (PSI)。

        P1修复: 在非NaN部分计算PSI (原始fillna导致退化的分位数 → 虚高PSI)。
        仅比较数据中有实际信号的部分, NaN率变化是单独的数据质量指标。
        """
        # 只对非NaN部分计算PSI
        feat_tr = feat_tr.dropna()
        feat_va = feat_va.dropna()

        if len(feat_tr) < 10 or len(feat_va) < 10:
            return 0.0

        def _psi(baseline, current):
            epsilon = 1e-4
            baseline, current = baseline + epsilon, current + epsilon
            return np.sum((baseline - current) * np.log(baseline / current))

        try:
            cuts = np.quantile(feat_tr, np.linspace(0, 1, 11))
            cuts[-1] += 1e-4
            base_cnt = np.histogram(feat_tr, bins=cuts)[0] / len(feat_tr)
            curr_cnt = np.histogram(feat_va, bins=cuts)[0] / len(feat_va)
            return _psi(base_cnt, curr_cnt)
        except Exception:
            return 0.0

    def _sanitize_features(self, df: pd.DataFrame, nan_threshold: float, max_corr: float) -> List[str]:
        sorted_cols = sorted(df.columns.tolist())
        df = df[sorted_cols].copy()

        nan_rate = df.isna().mean()
        keep_cols = nan_rate[nan_rate < nan_threshold].index.tolist()
        if not keep_cols:
            return []
        df = df[keep_cols]

        n_feat = len(keep_cols)
        if n_feat <= 1:
            return keep_cols

        # 一次性相关矩阵计算 (O(F²) via BLAS, 替代 O(F²) 逐对 corr 调用)
        # 性能对比: 1次矩阵运算 ~0.5s vs 逐对pandas corr ~15s (30x提速)
        corr_matrix = df[keep_cols].corr().values
        drop_flags = [False] * n_feat
        for i in range(n_feat):
            if drop_flags[i]:
                continue
            for j in range(i + 1, n_feat):
                if drop_flags[j]:
                    continue
                if abs(corr_matrix[i, j]) > max_corr:
                    drop_flags[j] = True

        final_cols = [col for col, drop in zip(keep_cols, drop_flags) if not drop]
        final_cols = sorted(final_cols)
        logger.info(f"Feature sanitization: {len(sorted_cols)} → {len(final_cols)}")
        return final_cols

    def _compute_ic_vectorized(self, tr: pd.DataFrame, y_train: np.ndarray,
                                feats: List[str], psi_info: dict) -> Tuple[List[str], dict]:
        """
        向量化IC计算（移植alpha191优化）。
        
        一次性 rank 所有特征 → Pearson on ranks 矩阵运算。
        预期比逐特征 spearmanr 快 10x。
        """
        # 批量 fillna + rank
        tr_filled = pd.DataFrame(index=tr.index, columns=feats, dtype=np.float64)
        for f in feats:
            _, med = psi_info[f]
            tr_filled[f] = tr[f].fillna(med)

        tr_ranked = tr_filled.rank()
        y_rank = pd.Series(y_train).rank()

        # 向量化 Pearson on ranks
        tr_arr = tr_ranked.values  # [N, F]
        y_arr = y_rank.values      # [N]
        tr_centered = tr_arr - tr_arr.mean(axis=0)
        y_centered = y_arr - y_arr.mean()
        numerator = tr_centered.T @ y_centered
        tr_ss = np.sqrt((tr_centered ** 2).sum(axis=0))
        y_ss = np.sqrt((y_centered ** 2).sum())
        ic_vector = numerator / (tr_ss * y_ss + 1e-12)
        ic_vector = np.where(np.isnan(ic_vector), 0.0, ic_vector)

        # 组装结果
        all_feat_scores = []
        for i, f in enumerate(feats):
            psi, _ = psi_info[f]
            ic_train = float(ic_vector[i])
            ic_abs = abs(ic_train)
            composite_score = ic_abs * (1 - min(psi, 1))
            all_feat_scores.append({
                'feature': f, 'psi': psi, 'ic': ic_train,
                'ic_abs': ic_abs, 'composite_score': composite_score,
            })

        good_feats = [fs['feature'] for fs in all_feat_scores
                      if fs['ic_abs'] >= self._feat_sel_min_ic]
        return good_feats, {fs['feature']: fs for fs in all_feat_scores}

    def _cluster_cfg_namespace(self):
        """构造 CFI 函数所需的 config SimpleNamespace (对齐 alpha191 cluster_* 属性名)"""
        import types as _types
        return _types.SimpleNamespace(
            cluster_enabled=True,
            cluster_cfi_mode=self._cluster_cfi_enabled,
            cluster_legacy_threshold=self._cluster_legacy,
            cluster_diagnose=self._cluster_diagnose,
            cluster_corr_threshold=self._cluster_corr_threshold,
            cluster_corr_method=self._cluster_corr_method,
            cluster_min_size=self._cluster_min_size,
            cluster_log=self._cluster_log,
            cluster_repr_method=self._cluster_repr_method,
            use_voting=self._fs_use_voting,
            mda_n_repeats=self._feat_sel_mda_n_repeats,
            mda_use_std_threshold=self._fs_mda_use_std_threshold,
            mda_t_alpha=self._fs_mda_t_alpha,
            mda_threshold=self._fs_mda_threshold,
            vote_min_folds=self._fs_vote_min_folds,
            vote_min_features=self._fs_vote_min_features,
            top_k_features=self._fs_top_k_features,
            weight_ic=self._fs_weight_ic,
            weight_mdi=self._fs_weight_mdi,
            weight_mda=self._fs_weight_mda,
            fallback_to_ic_only=self._fs_fallback_to_ic_only,
            XGB_MAX_DEPTH=self.xgb_params.get('max_depth', 4),
            XGB_LEARNING_RATE=self.xgb_params.get('eta', 0.05),
            XGB_SUBSAMPLE=self.xgb_params.get('subsample', 0.8),
            XGB_REG_ALPHA=self.xgb_params.get('alpha', 0.0),
            XGB_REG_LAMBDA=self.xgb_params.get('lambda', 1.0),
            XGB_MIN_CHILD_WEIGHT=self.xgb_params.get('min_child_weight', 1),
            XGB_GAMMA=self.xgb_params.get('gamma', 0.0),
            XGB_RANDOM_STATE=int(self.xgb_params.get('seed', 42)),
            XGB_TREE_METHOD='hist',
            XGB_DEVICE='cpu',
            XGB_N_ESTIMATORS=self._feat_sel_mdi_n_estimators,
            XGB_EARLY_STOPPING_ROUNDS=self.early_stopping_rounds,
        )

    def _fs_select_stable_namespace(self):
        """构造 select_stable_features 所需的 config SimpleNamespace (对齐 alpha191 属性名)。

        [2026-08-20 对齐 D3] 折内 IC/PSI 再筛选复用 alpha191 ensemble.select_stable_features,
        该函数以 config 属性方式访问参数 (config.MIN_FEAT_IC 等), 此处映射 V10 的 _feat_sel_* 配置。
        """
        import types as _types
        return _types.SimpleNamespace(
            NAN_THRESHOLD=self._feat_sel_nan_threshold,
            MAX_CORR=self._feat_sel_max_corr,
            PSI_THRESHOLD=self._feat_sel_psi_threshold,
            MIN_FEAT_IC=self._feat_sel_min_ic,
            MIN_FEATURES=self._min_features,
            MAX_FEATURES=self._max_features,
        )

    def _compute_mda_oos(self, model, X_val_df, y_val, feature_names,
                          n_repeats=20, random_state=42, t_alpha=0.05):
        """
        OOS MDA + t检验（移植alpha191优化版）。
        
        1. numpy预转换（避免每次迭代DataFrame copy）
        2. ThreadPoolExecutor并行predict（XGBoost释放GIL）
        3. t检验自适应阈值: mean > t_crit * SE
        """
        rng = np.random.RandomState(random_state)

        X_base = X_val_df[feature_names].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
        feat_to_idx = {f: i for i, f in enumerate(feature_names)}

        dval = xgb.DMatrix(X_base, label=y_val, missing=np.nan, feature_names=feature_names)

        try:
            baseline_proba = model.predict(dval)
            baseline_auc = float(roc_auc_score(y_val, baseline_proba))
        except ValueError:
            logger.warning("MDA: 基线AUC计算失败，跳过")
            return {f: 0.0 for f in feature_names}, {f: 0.0 for f in feature_names}

        total = len(feature_names)
        n_tasks = total * n_repeats
        logger.info(f"[MDA-OOS] 基线AUC={baseline_auc:.4f}, {total}特征 × {n_repeats}repeats = {n_tasks}任务")

        # 预生成所有置换（串行，RNG顺序确定）
        all_tasks = []
        for feat in feature_names:
            fidx = feat_to_idx[feat]
            for _ in range(n_repeats):
                col = X_base[:, fidx].copy()
                rng.shuffle(col)
                all_tasks.append((feat, fidx, col))

        import threading
        _thread_local = threading.local()

        def _eval_one(task):
            feat, fidx, shuffled_col = task
            if not hasattr(_thread_local, "X_ws"):
                _thread_local.X_ws = X_base.copy()
            _thread_local.X_ws[:] = X_base
            _thread_local.X_ws[:, fidx] = shuffled_col
            dval_perm = xgb.DMatrix(_thread_local.X_ws, label=y_val, missing=np.nan,
                                    feature_names=feature_names)
            try:
                perm_proba = model.predict(dval_perm)
                perm_auc = float(roc_auc_score(y_val, perm_proba))
                return feat, baseline_auc - perm_auc
            except ValueError:
                return feat, 0.0

        # Parallel execution (Perf Phase1: 4->8 workers)
        if n_tasks <= 20:
            results = [_eval_one(task) for task in all_tasks]
        else:
            max_workers = min(8, os.cpu_count() or 8)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                results = list(ex.map(_eval_one, all_tasks))

        # 聚合
        from collections import defaultdict
        feat_drops = defaultdict(list)
        for feat, drop in results:
            feat_drops[feat].append(drop)

        mda_means = {}
        mda_stds = {}
        for feat in feature_names:
            drops = feat_drops[feat]
            mda_means[feat] = float(np.mean(drops)) if drops else 0.0
            mda_stds[feat] = float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0

        # t检验
        df_n = n_repeats - 1
        t_crit = float(t_dist.ppf(1 - t_alpha, df_n))
        se_vals = np.array([mda_stds[f] / np.sqrt(n_repeats) for f in feature_names])
        mean_vals = np.array([mda_means[f] for f in feature_names])
        t_stats = np.divide(mean_vals, se_vals, out=np.zeros_like(mean_vals), where=(se_vals > 0))
        t_pass = t_stats > t_crit

        n_ttest_pass = int(np.sum(t_pass))
        logger.info(f"[MDA-OOS] t检验通过: {n_ttest_pass}/{total} (t_crit={t_crit:.3f}, df={df_n})")

        return mda_means, mda_stds

    def _prefilter_features(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
                            feats: List[str], target_col: str = 'label') -> Tuple[List[str], dict]:
        """
        IC预过滤（对齐Alpha191 select_stable_features）。
        
        IC阈值=0.001, 极宽松, 仅排除纯噪声。
        真正的特征选择由 _purged_kfold_cv 中的MDA多折投票完成。
        
        流程: 低方差 → 缺失率+高相关 → PSI → 向量化IC(0.001)
        
        Returns:
            (pre_filtered_features, ic_scores_dict)
        """
        feats = sorted(feats)
        if not feats:
            return [], {}

        y_train = df_train[target_col].values

        # 低方差过滤
        train_feat_df = df_train[feats].replace([np.inf, -np.inf], np.nan)
        selector = VarianceThreshold(threshold=self._feat_sel_variance_threshold)
        selector.fit(train_feat_df.fillna(0))
        n_before_var = len(feats)
        feats = train_feat_df.columns[selector.get_support()].tolist()
        n_var_dropped = n_before_var - len(feats)
        if n_var_dropped > 0:
            # 列出被移除特征的实际方差值 (诊断零方差是否为真)
            dropped_names = [c for c in train_feat_df.columns if c not in feats]
            sample_dropped = dropped_names[:3]
            sample_vars = [float(train_feat_df.fillna(0)[c].var()) for c in sample_dropped]
            sample_nans = [float(train_feat_df[c].isna().mean()) for c in sample_dropped]
            logger.info(f"[Prefilter] Variance (thresh={self._feat_sel_variance_threshold}): "
                       f"{n_before_var} → {len(feats)} "
                       f"({n_var_dropped} removed, samples: {list(zip(sample_dropped, sample_vars, sample_nans))})")
        if not feats:
            return [], {}

        # 缺失率+高相关
        feats = self._sanitize_features(df_train[feats], self._feat_sel_nan_threshold, self._feat_sel_max_corr)
        if not feats:
            return [], {}

        # PSI过滤 + 向量化IC
        tr = df_train[feats].replace([np.inf, -np.inf], np.nan)
        va = df_val[feats].replace([np.inf, -np.inf], np.nan)

        psi_info = {}
        psi_pass_feats = []
        for f in feats:
            try:
                tr_col = tr[f]
                va_col = va[f]
                tr_median = tr_col.median(skipna=True)
                if pd.isna(tr_median):
                    tr_median = 0.0
                psi = self._calc_psi(tr_col, va_col, fill_value=tr_median)
                if psi > self._feat_sel_psi_threshold:
                    continue
                psi_info[f] = (psi, tr_median)
                psi_pass_feats.append(f)
            except Exception:
                continue

        if not psi_pass_feats:
            return [], {}

        good_feats, feat_scores = self._compute_ic_vectorized(tr, y_train, psi_pass_feats, psi_info)
        ic_scores = {f: feat_scores[f]['ic_abs'] for f in good_feats}
        n_psi_dropped = len(feats) - len(psi_pass_feats)
        n_ic_dropped = len(psi_pass_feats) - len(good_feats)
        logger.info(f"[Prefilter] {len(feats)} → {len(good_feats)} features "
                    f"(PSI>{self._feat_sel_psi_threshold}: {n_psi_dropped}, "
                    f"|IC|<{self._feat_sel_min_ic}: {n_ic_dropped})")
        return good_feats, ic_scores

    def _select_features(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
                         feats: List[str], target_col: str = 'label') -> List[str]:
        """
        完整特征选择管线：
        低方差 → 缺失率+高相关 → PSI → 向量化IC → MDI(early stopping) → MDA(t检验) → min/max_features
        """
        rng = np.random.RandomState(self._fs_mda_random_state)  # P3-1: 局部RNG
        logger.info("=" * 60)
        logger.info("Feature Selection Pipeline (Full)")
        logger.info("=" * 60)

        feats = sorted(feats)
        if not feats:
            return []

        y_train = df_train[target_col].values
        y_val = df_val[target_col].values

        # Step 1: 低方差过滤
        logger.info("[Step 1] Low variance filter...")
        train_feat_df = df_train[feats].replace([np.inf, -np.inf], np.nan)
        selector = VarianceThreshold(threshold=self._feat_sel_variance_threshold)
        selector.fit(train_feat_df.fillna(0))
        feats = train_feat_df.columns[selector.get_support()].tolist()
        logger.info(f"After variance filter: {len(feats)} features")
        if not feats:
            return []

        # Step 2: 缺失率+高相关
        logger.info("[Step 2] NaN rate filter & high correlation dedup...")
        feats = self._sanitize_features(df_train[feats], self._feat_sel_nan_threshold, self._feat_sel_max_corr)
        logger.info(f"After sanitization: {len(feats)} features")
        if not feats:
            return []

        # Step 3: PSI过滤 (P1-2)
        logger.info(f"[Step 3] PSI filter (threshold={self._feat_sel_psi_threshold})...")
        tr = df_train[feats].replace([np.inf, -np.inf], np.nan)
        va = df_val[feats].replace([np.inf, -np.inf], np.nan)

        psi_info = {}
        psi_pass_feats = []
        psi_dropped = 0
        for f in feats:
            try:
                tr_col = tr[f]
                va_col = va[f]
                tr_median = tr_col.median(skipna=True)
                if pd.isna(tr_median):
                    tr_median = 0.0
                psi = self._calc_psi(tr_col, va_col, fill_value=tr_median)
                if psi > self._feat_sel_psi_threshold:
                    psi_dropped += 1
                    continue
                psi_info[f] = (psi, tr_median)
                psi_pass_feats.append(f)
            except Exception:
                continue
        logger.info(f"After PSI filter: {len(psi_pass_feats)} features (dropped {psi_dropped})")
        if not psi_pass_feats:
            return []

        # Step 4: 向量化IC过滤
        logger.info(f"[Step 4] Vectorized IC filter (threshold={self._feat_sel_min_ic})...")
        good_feats, feat_scores = self._compute_ic_vectorized(tr, y_train, psi_pass_feats, psi_info)
        logger.info(f"After IC filter: {len(good_feats)} features")
        if not good_feats:
            return []

        # Step 5: MDI过滤 (FI模型加Early Stopping, P1-6)
        logger.info(f"[Step 5] Train FI model for MDI (early_stopping={self.early_stopping_rounds})...")
        X_train_fi = df_train[good_feats].replace([np.inf, -np.inf], np.nan).fillna(0).values
        X_val_fi = df_val[good_feats].replace([np.inf, -np.inf], np.nan).fillna(0).values

        fi_params = self.xgb_params.copy()
        fi_params['nthread'] = self._feat_sel_mdi_nthread
        fi_params = self._apply_class_balance(fi_params, y_train)

        dtrain_fi = xgb.DMatrix(X_train_fi, label=y_train, feature_names=good_feats)
        dval_fi = xgb.DMatrix(X_val_fi, label=y_val, feature_names=good_feats)

        fi_model = xgb.train(
            fi_params, dtrain_fi,
            num_boost_round=self._feat_sel_mdi_n_estimators,
            evals=[(dtrain_fi, 'train'), (dval_fi, 'val')],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False
        )

        xgb_scores = fi_model.get_score(importance_type='gain')
        mdi_series = pd.Series([xgb_scores.get(f, 0) for f in good_feats], index=good_feats)
        mdi_sum = mdi_series.sum()
        if mdi_sum > 0:
            mdi_series = mdi_series / mdi_sum

        mdi_filtered = [f for f in good_feats if mdi_series[f] > 0]
        logger.info(f"MDI>0 features: {len(mdi_filtered)} / {len(good_feats)}")
        if not mdi_filtered:
            return good_feats

        # Step 6: MDA t检验 (P1-1)
        logger.info(f"[Step 6] MDA t-test (n_repeats={self._feat_sel_mda_n_repeats})...")
        mda_means, mda_stds = self._compute_mda_oos(
            fi_model, df_val[good_feats], y_val, good_feats,
            n_repeats=self._feat_sel_mda_n_repeats,
            random_state=self._fs_mda_random_state,
            t_alpha=self._fs_mda_t_alpha
        )

        df_n = self._feat_sel_mda_n_repeats - 1
        t_crit = float(t_dist.ppf(1 - self._fs_mda_t_alpha, df_n))
        final_feats = []
        for f in mdi_filtered:
            se = mda_stds[f] / np.sqrt(self._feat_sel_mda_n_repeats) if mda_stds[f] > 0 else 0
            t_stat = mda_means[f] / se if se > 0 else 0
            if t_stat > t_crit:
                final_feats.append(f)

        logger.info(f"MDA t-test passed: {len(final_feats)} / {len(mdi_filtered)}")

        if not final_feats:
            final_feats = [f for f in mdi_filtered if mda_means[f] > 0]
            if not final_feats:
                final_feats = mdi_filtered

        # Step 7: max_features截断 (P3-4)
        if len(final_feats) > self._max_features:
            logger.info(f"Too many features ({len(final_feats)}), truncating to top-{self._max_features}")
            sorted_by_composite = sorted(final_feats,
                                         key=lambda f: feat_scores[f]['composite_score'], reverse=True)
            final_feats = sorted_by_composite[:self._max_features]

        # 日志
        sorted_by_mdi = sorted(final_feats, key=lambda f: mdi_series[f], reverse=True)
        logger.info(f"Feature selection complete: {len(feats)} → {len(final_feats)}")
        for rank, f in enumerate(sorted_by_mdi[:20], 1):
            ic_val = feat_scores.get(f, {}).get('ic', float('nan'))
            logger.info(f"  #{rank:>3d} {f:<35s} IC={ic_val:+.4f}  MDI={mdi_series[f]:.6f}  MDA={mda_means.get(f, 0):.6f}")

        return sorted(final_feats)

    # ================================================================
    #  Purged K-Fold CV (P0-1)
    # ================================================================

    # ============================================================
    #  [超参选择] HP-Search (对齐 small_alpha191 A+B+C, 2026-08-18)
    # ============================================================
    def _select_hyperparams(self, train_data: pd.DataFrame,
                            feature_cols: List[str],
                            target_col: str = 'label') -> Tuple[Optional[str], Optional[float]]:
        """[超参选择] 多窗 expanding walk-forward 评估候选超参, 写回 self.xgb_params。

        对齐 small_alpha191 的 A+B+C 协议:
          - [A] 选择-评估隔离: 剔除与生产同规则的末 prod_val_size 天, 生产 val 零接触
          - [B] 多窗评估: W×inner_val_days 连续窗, expanding walk-forward (训练严格早于评估)
          - [C] 每候选每窗独立 early stopping + LRWarmup (与生产训练协议一致)
          - 评估特征 = pre_filtered (V10 折内 MDA 直接在 prefilter 全集上, 与 PKF 一致)
          - 防护: binary 全部候选平均 AUC<0.5 → 不写回, 保持原超参
          - 写回: self.xgb_params (max_depth/eta/subsample/colsample_bytree/
                  min_child_weight/alpha/lambda/gamma) + self.config['xgb_n_estimators']

        Returns:
            (best_name, best_score); 未启用/失败返回 (None, None)
        """
        hp_cfg = self.config.get('hyperparam_search', {})
        if not hp_cfg or not hp_cfg.get('enabled', False):
            return None, None
        candidates = hp_cfg.get('candidates', [])
        if not candidates:
            logger.warning("[HP-Search] hyperparam_search.candidates 为空, 跳过超参选择")
            return None, None

        es_rounds = int(hp_cfg.get('early_stopping_rounds', 20))
        use_es = bool(hp_cfg.get('use_early_stop', True))
        eval_rounds_cap = int(hp_cfg.get('eval_rounds', 1200))
        eval_windows = max(1, int(hp_cfg.get('eval_windows', 3)))
        inner_val_days = max(5, int(hp_cfg.get('inner_val_days', 20)))
        max_hold = int(getattr(self, 'pred_days', 20))

        # ===== [A] 选择-评估隔离: 剔除生产 val 段 (与生产同规则) =====
        all_dates = sorted(train_data['snapshot_date'].unique())
        n_dates = len(all_dates)
        prod_val_size = min(max_hold, max(n_dates // 5, 1))
        sel_dates = all_dates[:-prod_val_size] if prod_val_size > 0 else all_dates
        n_sel = len(sel_dates)
        if n_sel < 60:
            logger.warning("[HP-Search] 剔除生产val后训练数据不足 (%d 天), 跳过超参选择", n_sel)
            return None, None
        logger.info("[HP-Search] [A] 选择数据隔离: 剔除生产val %d 天, 仅用 %d 天 (生产val零接触)",
                    prod_val_size, n_sel)

        # 评估特征 = pre_filtered 全集 (与 PKF 折内 MDA 一致)
        eval_feats = list(feature_cols)
        if not eval_feats:
            logger.warning("[HP-Search] 无评估特征, 跳过超参选择")
            return None, None
        logger.info("[HP-Search] 候选评估特征: %d 个 (prefilter 全集, 与 PKF 一致)",
                    len(eval_feats))

        # ===== [B] 多窗 expanding walk-forward =====
        n_win_total = eval_windows * inner_val_days
        if n_sel < n_win_total:
            eval_windows = max(1, n_sel // (inner_val_days * 2))
            n_win_total = eval_windows * inner_val_days
            logger.warning("[HP-Search] 训练天数不足, 窗数降为 %d", eval_windows)
        win_dates = sel_dates[-n_win_total:]
        windows = []  # [(train_dates_set, val_dates_set)]
        for w in range(eval_windows):
            v_start = w * inner_val_days
            v_dates = set(win_dates[v_start:v_start + inner_val_days])
            # expanding: train = 该窗开始之前的所有 sel 数据 (训练严格早于评估)
            t_dates = set(sel_dates[:n_sel - n_win_total + v_start])
            windows.append((t_dates, v_dates))
        logger.info("[HP-Search] [B] 多窗评估: %d 窗 × %d 天 (expanding, 每窗train=%d~%d天)",
                    eval_windows, inner_val_days,
                    min(len(w[0]) for w in windows), max(len(w[0]) for w in windows))

        # 候选键(xgb_*前缀) → xgboost 参数名 / V10 xgb_params 键 映射
        # [2026-08-18] 必须用正确 xgboost 参数名 (xgb_max_depth→max_depth 等),
        # 否则 xgboost 静默忽略候选超参 (4组全同 bug 教训)
        cand_to_param = {
            'xgb_max_depth': 'max_depth',
            'xgb_learning_rate': 'eta',
            'xgb_subsample': 'subsample',
            'xgb_col_sample_by_tree': 'colsample_bytree',
            'xgb_reg_alpha': 'alpha',
            'xgb_reg_lambda': 'lambda',
            'xgb_min_child_weight': 'min_child_weight',
            'xgb_gamma': 'gamma',
        }
        hp_keys = list(cand_to_param.keys()) + ['xgb_n_estimators']
        int_keys = {'xgb_n_estimators', 'xgb_max_depth', 'xgb_min_child_weight'}
        base_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'seed': int(self.config.get('xgb_random_state', 42)),
            'tree_method': self.config.get('xgb_tree_method', 'hist'),
            'device': self.config.get('xgb_device', 'cpu'),
            'nthread': 1,
        }

        _t0 = time.time()
        results = []        # (name, mean_score, mean_rounds)
        detail = {}         # name -> {'scores': [...], 'rounds': [...]}
        _cal = self.full_calendar if self.full_calendar else []
        for cand in candidates:
            name = str(cand.get('name', 'c?'))
            missing_keys = [k for k in hp_keys if k not in cand]
            if missing_keys:
                logger.warning("[HP-Search] 候选 %s 缺参数 %s, 跳过", name, missing_keys)
                continue
            params = dict(base_params)
            for ck, pk in cand_to_param.items():
                params[pk] = int(cand[ck]) if ck in int_keys else float(cand[ck])
            cap = max(int(cand['xgb_n_estimators']), 50)
            cap = min(cap, eval_rounds_cap)
            win_scores = []
            win_rounds = []
            win_ok = True
            for (t_dates, v_dates) in windows:
                t_mask = train_data['snapshot_date'].isin(t_dates).values
                v_mask = train_data['snapshot_date'].isin(v_dates).values
                n_t, n_v = int(t_mask.sum()), int(v_mask.sum())
                if n_t < 100 or n_v < 50:
                    logger.warning("[HP-Search] 候选 %s 窗样本不足(train=%d,val=%d), 跳过",
                                   name, n_t, n_v)
                    win_ok = False
                    break
                X_t = train_data.loc[t_mask, eval_feats].replace(
                    [np.inf, -np.inf], np.nan).fillna(0).values
                y_t = train_data.loc[t_mask, target_col].values.astype(int)
                X_v = train_data.loc[v_mask, eval_feats].replace(
                    [np.inf, -np.inf], np.nan).fillna(0).values
                y_v = train_data.loc[v_mask, target_col].values.astype(int)
                # 每窗样本权重 (与生产折内同口径: 序贯唯一性)
                wt_t = None
                try:
                    wt_t = compute_sample_weights(train_data.loc[t_mask], _cal, max_hold)
                    if wt_t is None or len(wt_t) != n_t:
                        wt_t = None
                except Exception:
                    wt_t = None
                # [2026-08-20 对齐 alpha191 D1] 评估段 spw 注入受 hyperparam_search.eval_class_balance 控制
                #   true  -> 评估注入 spw (与部署同协议, V10 原行为; 实验用)
                #   false (默认) -> 评估不注入 (对齐 alpha191 收缩原则)
                # 理论依据 (alpha191 2026-08-19 实证): 弱信号下候选 AUC 差异 ~0.005-0.02 < 评估噪声,
                #   spw 对候选排序无信息量, 评估注入 spw 导致 19/41 期选择翻转 (Small 收益 109%->49%)
                #   → 默认关闭 = 收缩原则: 噪声大时向稳定评估先验收缩, 避免噪声驱动选择翻转
                # 注: _apply_class_balance 内部已守卫 class_balance_mode=='auto'
                if bool(hp_cfg.get('eval_class_balance', False)):
                    p_params = self._apply_class_balance(dict(params), y_t)
                else:
                    p_params = dict(params)
                dtrain = xgb.DMatrix(X_t, label=y_t, weight=wt_t,
                                     feature_names=eval_feats)
                dval = xgb.DMatrix(X_v, label=y_v, feature_names=eval_feats)
                try:
                    if use_es:
                        # [C] 每窗独立 early stopping + LRWarmup (与生产协议一致)
                        model = xgb.train(
                            p_params, dtrain, num_boost_round=cap,
                            evals=[(dtrain, 'train'), (dval, 'val')],
                            early_stopping_rounds=es_rounds, verbose_eval=False,
                            callbacks=[LRWarmupCallback(
                                target_lr=float(cand['xgb_learning_rate']))],
                        )
                        rounds = int(getattr(model, 'best_iteration', cap))
                    else:
                        # legacy fallback: 固定轮数
                        n_rounds = min(cap, 80)
                        model = xgb.train(p_params, dtrain, num_boost_round=n_rounds,
                                          verbose_eval=False,
                                          callbacks=[LRWarmupCallback(
                                              target_lr=float(cand['xgb_learning_rate']))])
                        rounds = model.num_boosted_rounds()
                    win_scores.append(float(roc_auc_score(y_v, model.predict(dval))))
                    win_rounds.append(rounds)
                except Exception as e:
                    logger.warning("[HP-Search] 候选 %s 窗训练失败: %s, 跳过", name, e)
                    win_ok = False
                    break
            if not win_ok or not win_scores:
                continue
            mean_score = float(np.mean(win_scores))
            mean_rounds = int(round(float(np.mean(win_rounds))))
            results.append((name, mean_score, mean_rounds))
            detail[name] = {'scores': win_scores, 'rounds': win_rounds}
            logger.info("[HP-Search] 候选 %s: val AUC=%.4f (W%d窗: %s, rounds=%s) "
                        "[depth=%d lr=%.4f ss=%.2f cs=%.2f l2=%.2f mcw=%d γ=%.3f]",
                        name, mean_score, len(win_scores),
                        ', '.join('%.4f' % s for s in win_scores),
                        win_rounds,
                        int(cand['xgb_max_depth']), float(cand['xgb_learning_rate']),
                        float(cand['xgb_subsample']), float(cand['xgb_col_sample_by_tree']),
                        float(cand['xgb_reg_lambda']), int(cand['xgb_min_child_weight']),
                        float(cand['xgb_gamma']))

        if not results:
            logger.warning("[HP-Search] 全部候选评估失败, 保持原超参")
            return None, None

        # [防护] binary 全部候选平均 AUC<0.5 → 评估段信号异常/特征反向 → 不写回
        if not any(s > 0.5 for _, s, _ in results):
            logger.warning("[HP-Search] 全部候选平均 val AUC<0.5 (%s), 评估段信号异常或特征反向, "
                           "不写回超参, 保持原配置",
                           ', '.join('%s=%.4f' % (n, s) for n, s, _ in results))
            self._hp_search_results = list(results)
            self._hp_search_detail = detail
            return None, None

        # 保存各组结果供诊断/测试 (生产无副作用)
        self._hp_search_results = list(results)
        self._hp_search_detail = detail

        best_name, best_score, _ = max(results, key=lambda r: r[1])
        best_cand = next(c for c in candidates if c.get('name') == best_name)
        for ck, pk in cand_to_param.items():
            if ck in best_cand:
                self.xgb_params[pk] = int(best_cand[ck]) if ck in int_keys else float(best_cand[ck])
        # n_estimators 写回 config (V10 训练轮数从 config 读, 与评估 cap 对齐)
        self.config['xgb_n_estimators'] = int(best_cand['xgb_n_estimators'])
        logger.info("[HP-Search] 选中候选 %s: AUC=%.4f (共%d组: %s), 耗时 %.1fs "
                    "→ 已写回 xgb_params + xgb_n_estimators",
                    best_name, best_score, len(results),
                    ', '.join('%s=%.4f' % (n, s) for n, s, _ in results),
                    time.time() - _t0)
        return best_name, best_score

    def _purged_kfold_cv(self, train_data: pd.DataFrame, pre_filtered_feats: List[str],
                         ic_scores: dict, cfi_clusters: Optional[List[List[str]]] = None) -> List[str]:
        """
        Purged K-Fold CV + MDA多折投票（对齐Alpha191 aggregate_fold_votes）。
        
        每折: 训练XGBoost模型 → MDA t检验 → 记录通过特征
        跨折: MDA投票(≥vote_min_folds折通过) → 综合评分 → top_k截断
        
        [CFI] cfi_clusters 非 None 时: 每折改做簇级整体置换 MDA (Prado MLAM Ch.6),
        跨折簇级投票 → 簇内选代表 (SFI/AFML §8.4.1)。
        
        Args:
            train_data: 训练数据
            pre_filtered_feats: IC预过滤后的特征列表
            ic_scores: IC分数字典 {feat: ic_abs}（来自_prefilter_features）
            cfi_clusters: [CFI] 全局簇结构 (cluster_features_global 输出); None=逐特征流程
        
        Returns:
            最终特征列表
        """
        from scipy.stats import rankdata as _rankdata

        K = self._cv_folds
        embargo = self._cv_embargo_days

        if not pre_filtered_feats or len(train_data) == 0:
            return []

        # 重置折定义 (每次特征选择时重新计算)
        self._cv_fold_defs = []

        unique_dates = sorted(train_data['snapshot_date'].unique())
        n_dates = len(unique_dates)
        block_size = max(1, n_dates // K)

        vote_min_folds = self._fs_vote_min_folds
        logger.info("=" * 60)
        logger.info(f"[Purged K-Fold CV + MDA Voting] K={K}, embargo={embargo}d, "
                    f"vote_min_folds={vote_min_folds}, "
                    f"t_alpha={self._fs_mda_t_alpha}")
        logger.info(f"  {n_dates} snapshot dates, block_size={block_size}, "
                    f"{len(pre_filtered_feats)} pre-filtered features")

        fold_mda_results = []   # [{feat: mda_mean}, ...]
        fold_mda_stds = []      # [{feat: mda_std}, ...]
        fold_mdi_results = []   # [{feat: mdi_score}, ...]
        fold_mda_pass = []      # [set of passing features, ...]
        fold_cluster_mda_results = []  # [CFI] [{cid: mda_mean}, ...]
        fold_cluster_mda_stds = []     # [CFI] [{cid: mda_std}, ...]

        for i in range(K):
            val_start = i * block_size
            val_end = min(val_start + block_size, n_dates)
            val_dates = unique_dates[val_start:val_end]

            purge_start = max(0, val_start - embargo)
            purge_end = min(val_end + embargo, n_dates)

            train_dates = [d for j, d in enumerate(unique_dates)
                           if j < purge_start or j >= purge_end]

            if len(train_dates) < 20 or len(val_dates) < 5:
                logger.warning(f"Fold {i+1}/{K}: insufficient data, skip")
                continue

            fold_train = train_data[train_data['snapshot_date'].isin(train_dates)]
            fold_val = train_data[train_data['snapshot_date'].isin(val_dates)]

            if len(fold_train) < 100 or len(fold_val) < 50:
                logger.warning(f"Fold {i+1}/{K}: insufficient samples, skip")
                continue

            logger.info(f"Fold {i+1}/{K}: train={len(train_dates)}d({len(fold_train)}) "
                        f"val={len(val_dates)}d({len(fold_val)}) [purge {embargo}d each side]")

            # 保存折定义: (train_dates, val_dates) 供生产模型 CV 折集成复用
            self._cv_fold_defs.append((train_dates, val_dates))

            # [P0-1 对齐 2026-08-18] 折内再切 inner-OOS (MDA 评估段; 验证块只评 AUC)
            # 嵌套原则 (Cawley-Talbot): 特征重要性(MDA)只接触训练部分, 验证块零接触
            # 原实现: MDA 在 fold_val 上算 → 验证集参与特征选择 → 选择偏倚 → meta_auc 乐观
            ft_unique = sorted(fold_train['snapshot_date'].unique())
            _io_frac = float(self.config.get('inner_oos_frac', 0.15))
            _io_min = int(self.config.get('inner_oos_min_days', 20))
            inner_oos_size = min(len(ft_unique) // 2,
                                 max(_io_min, int(len(ft_unique) * _io_frac)))
            inner_oos_dates = set(ft_unique[-inner_oos_size:])
            inner_train_dates = set(ft_unique[:-inner_oos_size])
            inner_train_set = fold_train[fold_train['snapshot_date'].isin(inner_train_dates)]
            inner_oos_set = fold_train[fold_train['snapshot_date'].isin(inner_oos_dates)]
            if len(inner_train_set) < 100 or len(inner_oos_set) < 30:
                logger.warning("Fold %d/%d: inner-OOS 切分样本不足(train=%d,oos=%d), 跳过",
                               i + 1, K, len(inner_train_set), len(inner_oos_set))
                continue
            logger.info("  [P0-1] inner-train=%dd(%d样本) inner-OOS=%dd(%d样本) (MDA评估段)",
                        len(inner_train_dates), len(inner_train_set),
                        len(inner_oos_dates), len(inner_oos_set))

            # ---- [2026-08-20 对齐 alpha191 D3] 折内 IC+PSI 再筛选 ----
            # alpha191 语义: 每折在折内 inner-train 上切 IC/PSI 子集 → select_stable_features 再筛
            # → 折内特征集 = 筛选后子集 (剔除该折内失效特征), 模型/MDA 均在该子集上进行
            # (仅用 inner-train; 验证块/inner-OOS 零接触, 嵌套原则 Cawley-Talbot)
            _ic_unique = sorted(inner_train_set['snapshot_date'].unique())
            _mid_raw = max(int(len(_ic_unique) * self._cv_ic_psi_split_ratio), 10)
            _mid = min(_mid_raw, len(_ic_unique) - 5)  # 保证 PSI 子集至少 5 天
            _ic_dates = _ic_unique[:_mid]
            _psi_dates = _ic_unique[_mid:]
            _ic_set = inner_train_set[inner_train_set['snapshot_date'].isin(_ic_dates)]
            _psi_set = inner_train_set[inner_train_set['snapshot_date'].isin(_psi_dates)]
            logger.info("  IC子集=%dd, PSI子集=%dd", len(_ic_dates), len(_psi_dates))
            try:
                _fs_cfg = self._fs_select_stable_namespace()
                fold_feats, _, _, _ic_scores_fold = select_stable_features(
                    _ic_set, _psi_set, pre_filtered_feats, _fs_cfg,
                    target_col='label', skip_preprocess=True)
            except Exception as _fs_e:
                logger.warning("Fold %d/%d: 折内 IC/PSI 再筛选失败: %s, 回退 prefilter 全集",
                               i + 1, K, _fs_e)
                fold_feats = list(pre_filtered_feats)
            if not fold_feats:
                logger.warning("Fold %d/%d: 折内再筛后无特征, 跳过", i + 1, K)
                continue
            fold_feats = sorted(fold_feats)
            logger.info("Fold %d/%d: 选入 %d 个特征", i + 1, K, len(fold_feats))

            # 训练XGBoost模型 (复用FI模型参数) — [P0-1] 用 inner-train, [D3] 用折内筛选特征
            X_train_fold = inner_train_set[fold_feats].replace(
                [np.inf, -np.inf], np.nan).fillna(0).values
            X_val_fold = fold_val[fold_feats].replace(
                [np.inf, -np.inf], np.nan).fillna(0).values
            y_train_fold = inner_train_set['label'].values.astype(int)
            y_val_fold = fold_val['label'].values.astype(int)

            fi_params = self.xgb_params.copy()
            fi_params['nthread'] = self._feat_sel_mdi_nthread
            fi_params = self._apply_class_balance(fi_params, y_train_fold)

            # [2026-08-05 修复] 折内模型带样本权重 (与生产/CV折集成同口径):
            # 平均唯一性(重叠惩罚) 应作用于特征选择与折内评估; [P0-1] 在 inner-train 上重算
            wt_fold = None
            try:
                _cal = self.full_calendar if self.full_calendar else []
                wt_fold = compute_sample_weights(inner_train_set, _cal, self.pred_days)
                if wt_fold is None or len(wt_fold) != len(y_train_fold):
                    wt_fold = None
            except Exception:
                wt_fold = None
            dtrain_fold = xgb.DMatrix(X_train_fold, label=y_train_fold, weight=wt_fold,
                                      feature_names=fold_feats)
            dval_fold = xgb.DMatrix(X_val_fold, label=y_val_fold,
                                    feature_names=fold_feats)

            try:
                fold_model = xgb.train(
                    fi_params, dtrain_fold,
                    num_boost_round=self._feat_sel_mdi_n_estimators,
                    evals=[(dtrain_fold, 'train'), (dval_fold, 'val')],
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose_eval=False
                )
            except Exception as e:
                logger.warning(f"Fold {i+1}/{K}: model training failed: {e}, skip")
                continue

            # MDI (归一化gain) — [D3] 折内特征集作用域
            xgb_scores = fold_model.get_score(importance_type='gain')
            mdi_dict = {f: xgb_scores.get(f, 0) for f in fold_feats}
            mdi_sum = sum(mdi_dict.values())
            if mdi_sum > 0:
                mdi_dict = {f: v / mdi_sum for f, v in mdi_dict.items()}
            fold_mdi_results.append(mdi_dict)

            # MDA t检验
            if cfi_clusters is not None and _CFI_AVAILABLE:
                # [CFI] 簇级整体置换 MDA (Prado MLAM Ch.6): 簇内全部特征同时打乱
                # [P0-1 对齐 2026-08-18] 在 inner-OOS 段上置换 (验证块零接触)
                fold_clusters = [[f for f in c if f in fold_feats] for c in cfi_clusters]
                _cfi_cfg = self._cluster_cfg_namespace()
                mda_means, mda_stds = compute_mda_oos_clustered(
                    fold_model, inner_oos_set[fold_feats],
                    inner_oos_set['label'].values.astype(int), fold_clusters,
                    n_repeats=self._feat_sel_mda_n_repeats,
                    random_state=self._fs_mda_random_state + i,
                    metric_fn=_objective_metric_fn('binary'),
                )
                fold_cluster_mda_results.append(mda_means)
                fold_cluster_mda_stds.append(mda_stds)
                # 簇级 t 检验通过 (计入投票)
                _df_n = self._feat_sel_mda_n_repeats - 1
                _t_crit = float(t_dist.ppf(1 - self._fs_mda_t_alpha, _df_n))
                _pass_clusters = set()
                for _cid in mda_means:
                    _se = mda_stds.get(_cid, 0) / np.sqrt(self._feat_sel_mda_n_repeats) \
                        if mda_stds.get(_cid, 0) > 0 else 0
                    if (_se > 0 and mda_means[_cid] > _t_crit * _se) or (_se == 0 and mda_means[_cid] > 0):
                        _pass_clusters.add(_cid)
                fold_mda_pass.append({f for cid in _pass_clusters
                                      for f in (cfi_clusters[cid] if cid < len(cfi_clusters) else [])})
                logger.info(f"  Fold {i+1} CFI cluster MDA t-test passed: "
                            f"{len(_pass_clusters)}/{len(fold_clusters)} clusters")
            else:
                # [P0-1 对齐 2026-08-18] MDA 在折内 inner-OOS 段上置换 (验证块只评 AUC)
                mda_means, mda_stds = self._compute_mda_oos(
                    fold_model, inner_oos_set[fold_feats],
                    inner_oos_set['label'].values.astype(int),
                    fold_feats,
                    n_repeats=self._feat_sel_mda_n_repeats,
                    random_state=self._fs_mda_random_state,
                    t_alpha=self._fs_mda_t_alpha
                )
                fold_mda_results.append(mda_means)
                fold_mda_stds.append(mda_stds)

                # t检验通过特征集 (作用域=折内筛选特征)
                df_n = self._feat_sel_mda_n_repeats - 1
                t_crit = float(t_dist.ppf(1 - self._fs_mda_t_alpha, df_n))
                pass_set = set()
                for f in fold_feats:
                    se = mda_stds[f] / np.sqrt(self._feat_sel_mda_n_repeats) if mda_stds[f] > 0 else 0
                    t_stat = mda_means[f] / se if se > 0 else 0
                    if t_stat > t_crit:
                        pass_set.add(f)
                    elif se == 0 and mda_means[f] > 0:
                        pass_set.add(f)
                fold_mda_pass.append(pass_set)
                logger.info(f"  Fold {i+1} MDA t-test passed: {len(pass_set)}/{len(fold_feats)}")

        if not fold_mda_pass:
            logger.warning("All folds failed in CV, returning pre-filtered features")
            return sorted(pre_filtered_feats[:self._fs_top_k_features])

        n_folds = len(fold_mda_pass)

        # [CFI] 簇级投票路径: 簇级 t 检验投票 → 簇内选代表 (Prado MLAM Ch.6 + AFML §8.4.1)
        if cfi_clusters is not None and fold_cluster_mda_results and _CFI_AVAILABLE:
            try:
                _cfi_cfg = self._cluster_cfg_namespace()
                # 每折 ic_scores 相同 (V10 IC 是全局预过滤口径)
                _n_folds_c = len(fold_cluster_mda_results)
                _fold_ic = [dict(ic_scores) for _ in range(_n_folds_c)]
                _sel_clusters, _cluster_report = aggregate_cluster_votes(
                    fold_cluster_mda_results, fold_cluster_mda_stds, cfi_clusters,
                    _cfi_cfg, fold_ic_results=_fold_ic,
                    consensus_features=sorted(pre_filtered_feats))
                # 簇内代表 (SFI 需 trade_date 列, V10 用 snapshot_date)
                _train_df = train_data.copy()
                if 'snapshot_date' in _train_df.columns and 'trade_date' not in _train_df.columns:
                    _train_df['trade_date'] = _train_df['snapshot_date']
                _final = select_cluster_reprs(
                    _train_df, _sel_clusters, cfi_clusters, _cfi_cfg,
                    target_col='label', objective_mode='binary')
                if _final:
                    logger.info(f"[CFI] Feature selection complete (cluster MDA voting): "
                                f"{len(cfi_clusters)}簇 → {len(_sel_clusters)}簇入选 → {len(_final)}特征")
                    return sorted(_final)
            except Exception as _cfi_e:
                logger.warning(f"[CFI] 簇级投票失败, 回退逐特征路径: {_cfi_e}")

        # ---- 逐特征 MDA 跨折投票 (原路径) ----
        mda_votes = {f: 0 for f in pre_filtered_feats}
        for pass_set in fold_mda_pass:
            for f in pass_set:
                mda_votes[f] += 1

        # 共识: ≥vote_min_folds折通过MDA t检验
        vote_min_folds = self._fs_vote_min_folds
        candidate_feats = [f for f in pre_filtered_feats if mda_votes[f] >= vote_min_folds]
        logger.info(f"MDA voting (≥{vote_min_folds}/{n_folds} folds): {len(candidate_feats)} features")

        # 投票分布日志
        vote_counts = [mda_votes[f] for f in pre_filtered_feats]
        n_exact = {i: sum(1 for v in vote_counts if v == i) for i in range(n_folds + 1)}
        dist_str = ", ".join([f"{i}folds={n_exact.get(i, 0)}" for i in range(n_folds, -1, -1)])
        logger.info(f"  Vote distribution: {dist_str}")

        # Fallback: 逐步降低共识门槛
        effective_min = vote_min_folds
        while len(candidate_feats) < self._fs_vote_min_features and effective_min > 1:
            effective_min -= 1
            candidate_feats = [f for f in pre_filtered_feats if mda_votes[f] >= effective_min]
            logger.warning(f"  Insufficient ({len(candidate_feats)}<{self._fs_vote_min_features}), "
                          f"lowering consensus to ≥{effective_min} → {len(candidate_feats)}")

        if not candidate_feats and self._fs_fallback_to_ic_only:
            logger.warning("No features passed MDA voting, fallback to IC top features")
            candidate_feats = sorted(pre_filtered_feats,
                                     key=lambda f: ic_scores.get(f, 0),
                                     reverse=True)[:self._fs_vote_min_features]

        # 综合评分: w_ic*avg_ic_rank + w_mdi*avg_mdi_rank + w_mda*avg_mda_rank
        w_ic = self._fs_weight_ic
        w_mdi = self._fs_weight_mdi
        w_mda = self._fs_weight_mda
        total_w = w_ic + w_mdi + w_mda

        # IC rank (全局, 各折相同)
        ic_vals = np.array([ic_scores.get(f, 0) for f in pre_filtered_feats])
        ic_rank_global = dict(zip(pre_filtered_feats,
                                  _rankdata(ic_vals) / max(len(pre_filtered_feats), 1)))

        scores = {}
        mda_rank_avg = {}   # [对齐 alpha191] 跨折平均 MDA 排名 (供 cluster_features 选代表)
        ic_rank_avg = {}    # [对齐 alpha191] 跨折平均 IC 排名
        for f in candidate_feats:
            avg_ic = ic_rank_global.get(f, 0)
            avg_mdi = np.mean([r.get(f, 0) for r in
                               [dict(zip(pre_filtered_feats,
                                         _rankdata(np.array([d.get(f, 0) for f in pre_filtered_feats]))
                                         / max(len(pre_filtered_feats), 1)))
                                for d in fold_mdi_results]]) if fold_mdi_results else 0
            avg_mda = np.mean([r.get(f, 0) for r in
                               [dict(zip(pre_filtered_feats,
                                         _rankdata(np.array([d.get(f, 0) for f in pre_filtered_feats]))
                                         / max(len(pre_filtered_feats), 1)))
                                for d in fold_mda_results]]) if fold_mda_results else 0
            scores[f] = (w_ic * avg_ic + w_mdi * avg_mdi + w_mda * avg_mda) / total_w
            mda_rank_avg[f] = avg_mda
            ic_rank_avg[f] = avg_ic

        # Top-K截断
        top_k = self._fs_top_k_features
        final_feats = sorted(candidate_feats, key=lambda f: scores[f], reverse=True)
        if len(final_feats) > top_k:
            final_feats = final_feats[:top_k]
            logger.info(f"Truncated to top-{top_k}")

        # [对齐 alpha191] 投票后聚类去冗余 (A 组流程: Purged K-Fold MDA 投票 →
        # cluster_features 每簇选代表; 与 alpha191 aggregate_fold_votes → cluster_features 同构)
        if _CFI_AVAILABLE and len(final_feats) >= 3:
            try:
                _vote_report = {
                    'scores': {f: float(scores.get(f, 0)) for f in final_feats},
                    'avg_mda_rank': mda_rank_avg,
                    'avg_ic_rank': ic_rank_avg,
                }
                _pre_cluster = list(final_feats)
                final_feats = cluster_features(
                    train_data, _pre_cluster, _vote_report,
                    self._cluster_cfg_namespace())
                logger.info(f"[FeatureClustering] 对齐alpha191 投票后聚类去冗余: "
                            f"{len(_pre_cluster)} → {len(final_feats)}")
            except Exception as _cf_e:
                logger.warning(f"[FeatureClustering] 聚类去冗余失败, 保持 top-k: {_cf_e}")

        logger.info(f"Feature selection complete (MDA voting + clustering): "
                    f"{len(pre_filtered_feats)} → {len(final_feats)}")
        for rank, f in enumerate(sorted(final_feats, key=lambda x: scores.get(x, 0), reverse=True)[:20], 1):
            logger.info(f"  #{rank:>3d} {f:<35s} votes={mda_votes[f]}/{n_folds}  "
                        f"IC={ic_scores.get(f, 0):.4f}  score={scores[f]:.4f}")

        return sorted(final_feats)

    # ================================================================
    #  多目标联合优化: evals_result + ntree_limit
    # ================================================================

    @staticmethod
    def _compute_combined_score(logloss_vals, auc_vals, alpha=0.5):
        """每轮的 combined 得分（越大越好）。
        
        公式: logloss_score = 1 - ll / 0.693 (0.693 = 随机猜测logloss)
              combined = alpha * logloss_score + (1-alpha) * auc
        """
        ll = np.array(logloss_vals, dtype=float)
        auc = np.array(auc_vals, dtype=float)
        ll_norm = np.clip(1.0 - ll / 0.693147, 0.0, 1.0)
        return alpha * ll_norm + (1.0 - alpha) * auc

    def _find_optimal_round(self, evals_result, alpha=None):
        """从验证集逐轮曲线中找到 combined 最优的轮次。
        
        Returns:
            optimal_round (1-indexed), or None if evals_result 不可用
        """
        if alpha is None:
            alpha = self._combined_alpha
        try:
            val_ll = evals_result['val']['logloss']
            val_auc = evals_result['val']['auc']
            if len(val_ll) != len(val_auc):
                return None
            scores = self._compute_combined_score(val_ll, val_auc, alpha)
            optimal_idx = int(np.argmax(scores))
            return optimal_idx + 1  # 1-indexed round number
        except (KeyError, ValueError, IndexError):
            return None

    # ================================================================
    #  CV折模型集成 (对齐Alpha191: 每折独立验证集 + 超参数多样性 + AUC加权)
    # ================================================================

    def _active_auc_metric(self) -> float:
        """[2026-08-20 对齐 alpha191 D5] 门控口径: 活跃模型 AUC 加权平均。

        alpha191 语义: 折集成 meta_AUC = np.average(active_aucs, weights=active_weights)
        (权重=纯超额归一化, 见 D4), 门控/校准共用同口径, 避免简单平均高估弱模型贡献。
        """
        model_aucs = getattr(self, 'model_aucs', None) or []
        active_models = getattr(self, '_active_models', None) or []
        model_weights = getattr(self, 'model_weights', None) or []
        if model_aucs and active_models:
            aucs = [model_aucs[i] for i in active_models]
            ws = [model_weights[i] for i in active_models]
        elif model_aucs:
            aucs = list(model_aucs)
            ws = list(model_weights) if model_weights else [1.0] * len(aucs)
        else:
            return 0.0
        if not aucs:
            return 0.0
        w_sum = float(sum(ws))
        if w_sum > 0:
            return float(np.average(aucs, weights=ws))
        return float(np.mean(aucs))

    def _train_cv_ensemble(self, train_data: pd.DataFrame) -> List[xgb.Booster]:
        """CV折集成训练: 复用 Purged K-Fold CV 的折定义,
        每折在各自独立的 train/val 分割上训练模型,
        应用超参数多样性, 按验证 AUC 加权投票。

        Args:
            train_data: 训练数据 DataFrame (含 feature_names 列 + label 列),
                        必须与 _purged_kfold_cv 使用的同一份数据。

        Returns:
            models: 每折训练的 XGBoost Booster 列表

        Side effects:
            设置 self.model_aucs, self.model_weights, self._active_models,
            self.models, self._optimal_rounds
        """
        if not self._cv_fold_defs:
            logger.warning("[CV Ensemble] 无 CV 折定义, 回退到种子集成")
            return None

        # [2026-08-20 D5] 防御: 重置门控口径缓存, 防上一期残留值被误用
        self._cv_meta_auc = None

        n_folds = len(self._cv_fold_defs)
        seeds = self._ensemble_seeds
        n_models = min(n_folds, len(seeds))

        # [对齐 small_alpha191 2026-08-18] 超参数多样性: 围绕 HP-Search 选中超参小扰动
        # 原硬编码 config diversity 列表 (如 [6,7,7,8,8]) 与超参选择结果无关 → 选择成果传导失效
        # 现: depth=[d,d,d+1,d+1,d+2], ss/cs=选中±0.05 (clip 0.3-1.0), metric 全 auc
        _sel_d = int(self.xgb_params.get('max_depth', 4))
        _sel_ss = float(self.xgb_params.get('subsample', 0.7))
        _sel_cs = float(self.xgb_params.get('colsample_bytree', 0.7))
        depth_list = [_sel_d, _sel_d, _sel_d + 1, _sel_d + 1, _sel_d + 2]
        depth_list = [max(2, d) for d in depth_list]
        ss_list = [max(0.3, min(1.0, _sel_ss - 0.05)), _sel_ss, _sel_ss,
                   max(0.3, min(1.0, _sel_ss + 0.05)), max(0.3, min(1.0, _sel_ss + 0.05))]
        cs_list = [max(0.3, min(1.0, _sel_cs - 0.05)), _sel_cs, max(0.3, min(1.0, _sel_cs + 0.05)),
                   _sel_cs, max(0.3, min(1.0, _sel_cs - 0.05))]
        metric_list = ['auc'] * 5

        logger.info("=" * 60)
        logger.info("[CV折集成] 使用 %d 折 CV 分割训练 %d 个模型", n_folds, n_models)
        logger.info("  超参数多样性: depth=%s, subsample=%s, colsample=%s, metric=%s",
                   [depth_list[i] if i < len(depth_list) else _sel_d for i in range(n_models)],
                   [ss_list[i] if i < len(ss_list) else _sel_ss for i in range(n_models)],
                   [cs_list[i] if i < len(cs_list) else _sel_cs for i in range(n_models)],
                   [metric_list[i] if i < len(metric_list) else 'auc' for i in range(n_models)])

        models = []
        model_aucs = []
        optimal_rounds = []
        t0 = time.time()

        # 预先准备 feature matrix
        X_all = train_data[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0).values

        for i in range(n_models):
            train_dates_i, val_dates_i = self._cv_fold_defs[i]
            seed_i = seeds[i]

            fold_train_mask = train_data['snapshot_date'].isin(train_dates_i).values
            fold_val_mask = train_data['snapshot_date'].isin(val_dates_i).values

            fold_train = train_data[fold_train_mask]
            fold_val = train_data[fold_val_mask]

            n_tr = len(fold_train)
            n_vl = len(fold_val)
            if n_tr < 100 or n_vl < 50:
                logger.warning("Fold %d/%d: 样本不足 (train=%d,val=%d), 跳过", i+1, n_folds, n_tr, n_vl)
                continue

            y_fold_train = fold_train['label'].values.astype(int)
            y_fold_val = fold_val['label'].values.astype(int)

            # 类别平衡
            # [P1-2 FIX] 尊重 xgb_class_balance 配置: 仅 mode='auto' 时注入 scale_pos_weight
            # 原代码无条件计算 spw 并注入, 与配置声明的 "none" 模式矛盾
            if self._class_balance_mode == 'auto':
                n_pos = int((y_fold_train == 1).sum())
                n_neg = int((y_fold_train == 0).sum())
                spw = float(n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
            else:
                spw = 1.0  # class_balance=none: 不注入, XGBoost 默认 1.0

            X_fold_train = X_all[fold_train_mask]
            X_fold_val = X_all[fold_val_mask]

            dtrain_fold = xgb.DMatrix(X_fold_train, label=y_fold_train,
                                      feature_names=self.feature_names)
            dval_fold = xgb.DMatrix(X_fold_val, label=y_fold_val,
                                    feature_names=self.feature_names)

            # P2: 折内样本权重 (序贯唯一性, De Prado AFML Ch.7)
            # [2026-08-05 修复] 原调用 compute_sample_weights(fold_train) 缺 calendar/
            # max_hold_days 参数 → TypeError 被静默吞掉 (权重从未生效) → 补全参数
            if self._ensemble_sample_weight and len(fold_train) > 100:
                try:
                    _cal = self.full_calendar if self.full_calendar else []
                    fold_weights = compute_sample_weights(fold_train, _cal, self.pred_days)
                    if fold_weights is not None and len(fold_weights) == len(y_fold_train):
                        dtrain_fold.set_weight(fold_weights)
                except Exception as e:
                    logger.warning(f"[CV Ensemble] Fold {i+1}: 折内权重计算失败: {e}, 使用等权")

            # 构建折特定超参数
            depth_i = depth_list[i] if i < len(depth_list) else 4
            ss_i = ss_list[i] if i < len(ss_list) else 0.7
            cs_i = cs_list[i] if i < len(cs_list) else 0.7
            metric_i = metric_list[i] if i < len(metric_list) else 'auc'

            fold_params = {
                'max_depth': depth_i,
                'eta': self.xgb_params['eta'],
                'subsample': ss_i,
                'colsample_bytree': cs_i,
                'colsample_bylevel': min(1.0, cs_i + 0.1),
                'colsample_bynode': min(1.0, cs_i),
                'alpha': self.xgb_params['alpha'],
                'lambda': self.xgb_params['lambda'],
                'min_child_weight': self.xgb_params['min_child_weight'],
                'gamma': self.xgb_params['gamma'],
                'scale_pos_weight': spw,
                'objective': 'binary:logistic',
                'eval_metric': metric_i,
                'seed': int(seed_i),
                'tree_method': self.xgb_params.get('tree_method', 'hist'),
                'device': self.xgb_params.get('device', 'cpu'),
                'nthread': 1,
            }

            model = xgb.train(
                fold_params, dtrain_fold,
                num_boost_round=self.config.get('xgb_n_estimators', 1000),  # [P2-3 FIX] was 500
                evals=[(dtrain_fold, 'train'), (dval_fold, 'val')],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False,
                callbacks=[self._lr_warmup_cb] if self._lr_warmup_cb else None,
            )
            models.append(model)
            optimal_rounds.append(None)  # CV折模式不使用 combined optimal_round
            best_iter = getattr(model, 'best_iteration', 0)

            # 折内验证 AUC
            fold_pred = model.predict(dval_fold)
            try:
                fold_auc = float(roc_auc_score(y_fold_val, fold_pred))
            except ValueError:
                fold_auc = 0.5
            model_aucs.append(fold_auc)

            logger.info("  Fold %d/%d (seed=%d, depth=%d, ss=%.1f, cs=%.1f, metric=%s): "
                       "train=%d, val=%d, best_iter=%d, val_AUC=%.4f",
                       i+1, n_folds, seed_i, depth_i, ss_i, cs_i, metric_i,
                       n_tr, n_vl, best_iter, fold_auc)

        if not models:
            logger.error("[CV Ensemble] 所有折均训练失败")
            return None

        train_time = time.time() - t0

        # --- AUC 加权投票 [2026-08-20 对齐 alpha191 D4] ---
        # alpha191 语义: 纯超额归一化 max(0, AUC-0.5)/Σ → AUC≤0.5 模型权重=0 (弱模型零权重)
        # 原 V10: base_w=0.5/n 兜底 → 弱模型仍有正权重 (对弱模型更宽容)
        if self._ensemble_weight_method == 'uniform':
            weights = [1.0] * len(models)
        else:  # auc_weighted (默认, 对齐 alpha191)
            excess = [max(0.0, a - 0.50) for a in model_aucs]
            total_e = sum(excess) + 1e-10
            weights = [e / total_e for e in excess]

        # 排除弱模型
        min_auc = self._ensemble_min_auc
        active = [j for j, a in enumerate(model_aucs) if a >= min_auc]
        if not active:
            active = list(range(len(models)))  # 降级: 全用

        self.models = models
        self.model_aucs = model_aucs
        self.model_weights = weights
        self._active_models = active
        self._optimal_rounds = optimal_rounds

        # 门控用 AUC: 仅活跃模型 (排除弱模型, 避免被bad split拖低门控)
        # [2026-08-20 对齐 alpha191 D5] 加权平均口径 (权重=纯超额归一化, 与预测/校准共用)
        meta_auc = self._active_auc_metric()
        self._cv_meta_auc = meta_auc

        logger.info("CV折集成完成: %d 模型 (%d active), 耗时 %.1fs, meta_AUC=%.4f",
                   len(models), len(active), train_time, meta_auc)
        logger.info("  集成权重: %s", [
            f"fold{i+1} auc={model_aucs[i]:.4f} w={weights[i]:.3f}"
            + (" EXCLUDED" if i not in active else "")
            for i in range(len(models))
        ])

        # P1-11: LightGBM 异质集成 (CV折模式)
        self.lgb_models = []
        if self._lgb_enabled:
            try:
                lgb_t0 = time.time()
                # 收集所有折的训练/验证数据 (用 XGBoost 的同一数据)
                all_train_preds = []
                all_train_labels = []
                for i in range(n_models):
                    train_dates_i, val_dates_i = self._cv_fold_defs[i]
                    fold_train_mask = train_data['snapshot_date'].isin(train_dates_i).values
                    fold_val_mask = train_data['snapshot_date'].isin(val_dates_i).values
                    fold_train = train_data[fold_train_mask]
                    fold_val = train_data[fold_val_mask]
                    if len(fold_train) < 100 or len(fold_val) < 50:
                        continue
                    y_fi = fold_train['label'].values.astype(int)
                    y_vi = fold_val['label'].values.astype(int)
                    X_fi = X_all[fold_train_mask]
                    X_vi = X_all[fold_val_mask]

                    # 每折训练一个 LightGBM 模型
                    seed_i = seeds[i] if i < len(seeds) else 42
                    # [2026-08-05] 折内 LGBM 权重 (与 XGB 折内权重同源)
                    _lgb_wt = None
                    try:
                        _cal = self.full_calendar if self.full_calendar else []
                        _lgb_wt = compute_sample_weights(fold_train, _cal, self.pred_days)
                    except Exception:
                        _lgb_wt = None
                    lgb_model = self._train_single_lgb(
                        X_fi, y_fi, X_vi, y_vi, self.feature_names, seed_i,
                        sample_weight=_lgb_wt)
                    if lgb_model is not None:
                        self.lgb_models.append(lgb_model)

                lgb_time = time.time() - lgb_t0
                logger.info(f"[LGB CV Ensemble] {len(self.lgb_models)} models trained, "
                           f"time: {lgb_time:.1f}s")
            except Exception as e:
                logger.warning(f"[LGB CV Ensemble] Training failed: {e}, "
                             "falling back to XGBoost-only")

        return models

    def _train_single_lgb(self, X_train, y_train, X_val, y_val,
                           feature_names: List[str], seed: int,
                           sample_weight=None):
        """训练单个 LightGBM 模型。

        Args:
            X_train, y_train: 训练数据
            X_val, y_val: 验证数据
            feature_names: 特征名列表
            seed: 随机种子
            sample_weight: [2026-08-05] 样本权重 (平均唯一性, 与 XGBoost 同口径)

        Returns:
            trained LightGBM Booster or None on failure
        """
        if not _LGB_AVAILABLE:
            return None

        try:
            lgb_params = self._lgb_params
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'num_leaves': lgb_params['num_leaves'],
                'learning_rate': lgb_params['learning_rate'],
                'subsample': lgb_params['subsample'],
                'colsample_bytree': lgb_params['colsample_bytree'],
                'min_child_samples': lgb_params['min_child_samples'],
                'reg_alpha': lgb_params['reg_alpha'],
                'reg_lambda': lgb_params['reg_lambda'],
                'seed': seed,
                'nthread': lgb_params['nthread'],
                'verbosity': -1,
            }

            if self._class_balance_mode == 'auto':
                n_pos = int((y_train == 1).sum())
                n_neg = int((y_train == 0).sum())
                if n_pos > 0 and n_neg > 0:
                    spw = float(n_neg) / float(n_pos)
                    params['scale_pos_weight'] = min(spw, self._class_balance_max_scale)

            # [2026-08-05 修复] LGBM 带样本权重 (与 XGBoost 同口径, 平均唯一性)
            wt_tr = None
            if sample_weight is not None:
                wt_arr = np.asarray(sample_weight)
                if len(wt_arr) == len(y_train):
                    wt_tr = wt_arr
            dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names,
                                 weight=wt_tr)
            dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_names,
                               reference=dtrain)

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=lgb_params['n_estimators'],
                valid_sets=[dtrain, dval],
                valid_names=['train', 'val'],
                callbacks=[
                    lgb.early_stopping(lgb_params['early_stopping_rounds']),
                    lgb.log_evaluation(period=0),
                ],
            )
            return model
        except Exception as e:
            logger.warning(f"[LGB] Single model training failed: {e}")
            return None

    # ================================================================
    #  种子模型集成 (旧版, 向后兼容)
    # ================================================================

    def _train_ensemble(self, dtrain: xgb.DMatrix, dval: xgb.DMatrix,
                         force_seed: bool = False) -> List[xgb.Booster]:
        """
        模型集成训练。自动选择 CV 折集成或种子集成模式。

        当 force_seed=True 或 ensemble 方法为 seed_ensemble 或 CV 折定义不足时,
        使用旧版种子集成 (仅 seed 不同, 同一验证集早停)。

        当 ensemble 方法为 cv_fold 且 force_seed=False 时, 由调用方决定
        (此方法接收 DMatrix, 仅供 seed_ensemble 和 SHAP 调用)。

        当 xgb_combined_enabled=true 时，捕获 evals_result 并选出
        combined 最优轮次，预测时用 iteration_range 截断。

        P1-11: 同时训练 LightGBM 异质模型 (通过不同 tree-growing 策略
        增加模型多样性, 降低单一模型族的系统性偏差风险)。
        """
        models = []
        optimal_rounds = []  # combined 最优轮次（1-indexed）
        t0 = time.time()
        y_train = dtrain.get_label().astype(int)
        X_train_np = dtrain.get_data()._proxy if hasattr(dtrain.get_data(), '_proxy') else dtrain.get_data()
        feature_names = dtrain.feature_names

        for seed in self._ensemble_seeds:
            params = self.xgb_params.copy()
            params['seed'] = seed
            params = self._apply_class_balance(params, y_train)

            if self._combined_enabled:
                # 强制双指标: auc 在前(仅记录), logloss 在后(驱动早停)
                params['eval_metric'] = ['auc', 'logloss']
                evals_result = {}
                model = xgb.train(
                    params, dtrain,
                    num_boost_round=self.config.get('xgb_n_estimators', 1000),
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    evals_result=evals_result,
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose_eval=False,
                    callbacks=[self._lr_warmup_cb] if self._lr_warmup_cb else None,
                )
                opt_round = self._find_optimal_round(evals_result)
                optimal_rounds.append(opt_round)
                if opt_round is not None:
                    logger.info(f"  Model {len(models)} (seed={seed}): "
                                f"best_iter={model.best_iteration}, "
                                f"combined_optimal_round={opt_round}")
                else:
                    logger.info(f"  Model {len(models)} (seed={seed}): "
                                f"best_iter={model.best_iteration}, combined fallback to best_iter")
            else:
                model = xgb.train(
                    params, dtrain,
                    num_boost_round=self.config.get('xgb_n_estimators', 1000),
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose_eval=False,
                    callbacks=[self._lr_warmup_cb] if self._lr_warmup_cb else None,
                )
                optimal_rounds.append(None)

            models.append(model)

        train_time = time.time() - t0
        self._optimal_rounds = optimal_rounds

        # 种子模式下, 均等权重 (所有模型共用同一验证集, 无 AUC 加权依据)
        self.model_aucs = [0.5] * len(models)
        self.model_weights = [1.0] * len(models)
        self._active_models = list(range(len(models)))

        # P1-11: LightGBM 异质集成 (种子模式)
        self.lgb_models = []
        if self._lgb_enabled:
            try:
                lgb_t0 = time.time()
                X_val_np = dval.get_data() if hasattr(dval, 'get_data') else None
                y_val = dval.get_label().astype(int) if hasattr(dval, 'get_label') else None
                self.lgb_models = self._train_lgb_ensemble(
                    X_train_np, y_train, X_val_np, y_val, feature_names, mode='seed')
                lgb_time = time.time() - lgb_t0
                logger.info(f"[LGB Seed Ensemble] {len(self.lgb_models)} models trained, "
                           f"time: {lgb_time:.1f}s")
                train_time += lgb_time
            except Exception as e:
                logger.warning(f"[LGB Seed Ensemble] Training failed: {e}, "
                             "falling back to XGBoost-only")

        logger.info(f"Ensemble training complete ({self._ensemble_method}): "
                    f"{len(models)} XGBoost + {len(self.lgb_models)} LightGBM models, "
                    f"time: {train_time:.1f}s")
        return models

    def _train_lgb_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray,
                             feature_names: List[str], mode: str = 'seed') -> List:
        """训练 LightGBM 异质模型集合。

        LightGBM 与 XGBoost 的关键差异:
        - 叶子生长策略: leaf-wise (best-first) vs level-wise
          → LightGBM 倾向于更深的不平衡树, XGBoost 倾向于均衡的树
        - 分裂策略: 基于梯度的单边采样 (GOSS) + 互斥特征捆绑 (EFB)
          → 对高维稀疏特征更友好
        - 叶子节点: 可配置更多叶子数 (num_leaves), 配合 min_child_samples 防止过拟合

        这些差异使得 LightGBM + XGBoost 异质集成能有效降低单一模型的系统性偏差。

        Args:
            X_train, y_train: 训练数据
            X_val, y_val: 验证数据
            feature_names: 特征名列表
            mode: 'seed' (种子集成) 或 'cv_fold' (CV折集成)

        Returns:
            List of trained LightGBM Booster objects
        """
        if not _LGB_AVAILABLE:
            logger.warning("[LGB] lightgbm not installed, skipping")
            return []

        models = []
        lgb_params = self._lgb_params.copy()

        for i, seed in enumerate(self._ensemble_seeds):
            fold_params = {
                'objective': 'binary',
                'metric': 'auc',
                'num_leaves': lgb_params['num_leaves'],
                'learning_rate': lgb_params['learning_rate'],
                'subsample': lgb_params['subsample'],
                'colsample_bytree': lgb_params['colsample_bytree'],
                'min_child_samples': lgb_params['min_child_samples'],
                'reg_alpha': lgb_params['reg_alpha'],
                'reg_lambda': lgb_params['reg_lambda'],
                'seed': seed,
                'nthread': lgb_params['nthread'],
                'verbosity': -1,
            }

            # 类别平衡: LightGBM 使用 is_unbalance 或 scale_pos_weight
            if self._class_balance_mode == 'auto':
                n_pos = int((y_train == 1).sum())
                n_neg = int((y_train == 0).sum())
                if n_pos > 0 and n_neg > 0:
                    spw = float(n_neg) / float(n_pos)
                    fold_params['scale_pos_weight'] = min(spw, self._class_balance_max_scale)

            dtrain_lgb = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
            dval_lgb = lgb.Dataset(X_val, label=y_val, feature_name=feature_names,
                                   reference=dtrain_lgb)

            model = lgb.train(
                fold_params,
                dtrain_lgb,
                num_boost_round=lgb_params['n_estimators'],
                valid_sets=[dtrain_lgb, dval_lgb],
                valid_names=['train', 'val'],
                callbacks=[
                    lgb.early_stopping(lgb_params['early_stopping_rounds']),
                    lgb.log_evaluation(period=0),
                ],
            )
            models.append(model)

        # 计算 LightGBM 验证 AUC
        lgb_aucs = []
        for model in models:
            try:
                pred = model.predict(X_val)
                auc = float(roc_auc_score(y_val, pred))
                lgb_aucs.append(auc)
            except Exception:
                lgb_aucs.append(0.5)

        if lgb_aucs:
            logger.info(f"[LGB] Valid AUCs: {[f'{a:.4f}' for a in lgb_aucs]}, "
                       f"mean={np.mean(lgb_aucs):.4f}")

        return models

    def _predict_ensemble(self, dmatrix: xgb.DMatrix) -> np.ndarray:
        """集成预测：XGBoost AUC加权平均 + LightGBM 异质集成。

        当 combined 模式时，每个 XGBoost 模型使用各自的 optimal_round 截断。
        XGBoost >=2.0 使用 iteration_range=(0, n) 替代 ntree_limit。

        P1-11: 同时使用 LightGBM 模型预测并加权融合。
        融合公式: final_pred = w_xgb * xgb_pred + w_lgb * lgb_pred
        其中 w_xgb + w_lgb = 1.0 (由 ensemble.lightgbm.ensemble_weight_xgb/lgb 控制)
        """
        if not self.models:
            raise ValueError("模型未训练, 无法预测")

        # === XGBoost 预测 (现有逻辑) ===
        # 确定活跃模型和权重
        if (hasattr(self, '_active_models') and self._active_models
                and hasattr(self, 'model_weights') and self.model_weights):
            active_idx = self._active_models
            weights = [self.model_weights[i] for i in active_idx]
            active_models = [self.models[i] for i in active_idx]
            active_rounds = ([self._optimal_rounds[i] for i in active_idx]
                             if hasattr(self, '_optimal_rounds') and self._optimal_rounds
                             else [None] * len(active_idx))
        else:
            active_models = self.models
            active_rounds = (self._optimal_rounds
                             if hasattr(self, '_optimal_rounds') and self._optimal_rounds
                             else [None] * len(self.models))
            weights = [1.0] * len(active_models)

        # 逐 XGBoost 模型预测
        all_preds = []
        for model, opt_round in zip(active_models, active_rounds):
            if self._combined_enabled and opt_round is not None and opt_round > 0:
                all_preds.append(model.predict(dmatrix, iteration_range=(0, opt_round)))
            else:
                all_preds.append(model.predict(dmatrix))

        # XGBoost 加权平均
        if len(all_preds) == 1:
            xgb_pred = all_preds[0]
        else:
            w_sum = sum(weights)
            if w_sum > 0:
                xgb_pred = np.average(np.column_stack(all_preds), axis=1, weights=weights)
            else:
                xgb_pred = np.column_stack(all_preds).mean(axis=1)

        # === LightGBM 预测 (P1-11: 异质集成) ===
        if self._lgb_enabled and self.lgb_models:
            # 获取 DMatrix 中的原始数据用于 LightGBM 预测
            # 注意: XGBoost DMatrix → numpy array
            X_data = dmatrix.get_data()._proxy if hasattr(dmatrix.get_data(), '_proxy') else dmatrix.get_data()
            if isinstance(X_data, np.ndarray):
                # 使用 DMatrix 的 feature_names 子集
                if dmatrix.feature_names and self.feature_names:
                    try:
                        # 构建列索引映射
                        feature_idx = [dmatrix.feature_names.index(f) for f in self.feature_names
                                       if f in dmatrix.feature_names]
                        X_lgb = X_data[:, feature_idx] if feature_idx else X_data
                    except (ValueError, IndexError):
                        X_lgb = X_data
                else:
                    X_lgb = X_data

                lgb_preds = []
                for m in self.lgb_models:
                    try:
                        lgb_preds.append(m.predict(X_lgb))
                    except Exception:
                        continue

                if lgb_preds:
                    lgb_pred = np.column_stack(lgb_preds).mean(axis=1)
                    # 异质集成权重融合
                    w_xgb = self._lgb_ensemble_weight_xgb
                    w_lgb = self._lgb_ensemble_weight_lgb
                    final_pred = w_xgb * xgb_pred + w_lgb * lgb_pred
                    return final_pred
                else:
                    return xgb_pred
            else:
                return xgb_pred
        else:
            return xgb_pred

    # ================================================================
    #  Isotonic概率校准 (P0-4)
    # ================================================================

    def _calibrate_proba(self, dval: xgb.DMatrix):
        """在验证集预测上拟合概率校准器。

        支持三种方法:
        - platt: Platt Scaling (LogisticRegression on raw predictions)
                比 Isotonic 更平滑, 不会加剧 binning 导致的概率平局
        - isotonic: Isotonic Regression (保序回归, 更灵活但可能过拟合)
        - none: 不使用校准

        Returns:
            calibrator object or None
        """
        if self._calibration_method not in ('platt', 'isotonic'):
            return None

        val_preds = self._predict_ensemble(dval)
        y_val = dval.get_label().astype(int)

        # 确保至少有两类样本
        if len(np.unique(y_val)) < 2:
            logger.warning("[Calibrate] 验证集单一类别, 跳过校准")
            return None

        # 确保验证集有足够样本
        if len(val_preds) < 50:
            logger.warning(f"[Calibrate] 验证集样本不足 ({len(val_preds)} < 50), 跳过校准")
            return None

        if self._calibration_method == 'platt':
            # Platt Scaling: 使用无正则化的 LogisticRegression 拟合 sigmoid
            # 以下方法等同于 Platt (1999) 论文的原始方法:
            #   P(y=1|f) = 1 / (1 + exp(A * f + B))
            # Platt Scaling: C=1e8 ≈ 无正则化 (与 penalty=None 数学等价, 零警告)
            calibrator = LogisticRegression(C=1e8, solver='lbfgs', max_iter=1000)
            calibrator.fit(val_preds.reshape(-1, 1), y_val)
            calibrated = calibrator.predict_proba(val_preds.reshape(-1, 1))[:, 1]
        elif self._calibration_method == 'isotonic':
            calibrator = IsotonicRegression(y_min=self._prob_clip_min,
                                            y_max=self._prob_clip_max,
                                            out_of_bounds='clip')
            calibrator.fit(val_preds, y_val)
            calibrated = calibrator.predict(val_preds)

        # 退化检测
        prob_range = calibrated.max() - calibrated.min()
        if prob_range < 0.01:
            logger.warning("概率校准退化: 模型可能无区分力 (range < 0.01)")

        # 校准质量监控 (P0-2: ECE + AUC对比)
        if self._calibration_monitoring:
            try:
                auc_raw = float(roc_auc_score(y_val, val_preds))
                auc_cal = float(roc_auc_score(y_val, calibrated))
                # 计算 Expected Calibration Error (ECE)
                ece_raw = self._compute_ece(y_val, val_preds, n_bins=10)
                ece_cal = self._compute_ece(y_val, calibrated, n_bins=10)
                logger.info(
                    f"[Calibrate/{self._calibration_method}] "
                    f"raw_AUC={auc_raw:.4f} cal_AUC={auc_cal:.4f} "
                    f"(Δ={auc_cal - auc_raw:+.4f}) | "
                    f"raw_ECE={ece_raw:.4f} cal_ECE={ece_cal:.4f} "
                    f"(Δ={ece_cal - ece_raw:+.4f})"
                )
            except ValueError:
                pass

        return calibrator

    def _calibrate_cv_ensemble(self, train_data: pd.DataFrame):
        """CV 折集成校准: 仅收集活跃模型的验证集预测, 合并后拟合校准器。

        支持 platt/isotonic/none 三种校准方法。
        每折活跃模型在其自身验证集上预测, 所有预测拼接后作为校准器的训练数据。
        排除模型不参与校准 —— 因为预测时它们也不参与加权平均。
        """
        if self._calibration_method not in ('platt', 'isotonic'):
            return None

        if not self._cv_fold_defs or not self.models:
            logger.warning("[CV Calibrate] 无 CV 折定义或模型, 跳过校准")
            return None

        # 确定活跃模型索引 (对齐 _predict_ensemble)
        active_set = set(self._active_models) if (hasattr(self, '_active_models')
                         and self._active_models) else set(range(len(self.models)))

        X_all = train_data[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0).values
        all_preds = []
        all_labels = []

        # [2026-08-05 修复] 校准器拟合"融合 OOS" = XGB折集成×w_xgb + LGBM×w_lgb
        # (业界: calibrate the final ensemble output, 校准器输入分布=预测分布)
        w_xgb = self._lgb_ensemble_weight_xgb if self._lgb_enabled else 1.0
        w_lgb = self._lgb_ensemble_weight_lgb if self._lgb_enabled else 0.0

        for i, model in enumerate(self.models):
            if i >= len(self._cv_fold_defs):
                break
            # P1: 跳过排除模型 (预测时不参与加权, 校准也不应包含)
            if i not in active_set:
                continue
            _, val_dates_i = self._cv_fold_defs[i]
            fold_val_mask = train_data['snapshot_date'].isin(val_dates_i).values
            fold_val = train_data[fold_val_mask]
            if len(fold_val) == 0:
                continue

            X_val_i = X_all[fold_val_mask]
            y_val_i = fold_val['label'].values.astype(int)
            dval_i = xgb.DMatrix(X_val_i, feature_names=self.feature_names)
            pred_i = model.predict(dval_i)

            # LGBM 融合 (与 _predict_ensemble 同一公式)
            if self._lgb_enabled and self.lgb_models:
                try:
                    lgb_preds_i = []
                    for m in self.lgb_models:
                        try:
                            lgb_preds_i.append(m.predict(X_val_i))
                        except Exception:
                            continue
                    if lgb_preds_i:
                        lgb_pred_i = np.column_stack(lgb_preds_i).mean(axis=1)
                        pred_i = w_xgb * pred_i + w_lgb * lgb_pred_i
                except Exception as e:
                    logger.warning(f"[CV Calibrate] LGBM OOS 融合失败 ({e}), 该折用 XGB-only")

            all_preds.append(pred_i)
            all_labels.append(y_val_i)

        if not all_preds:
            logger.warning("[CV Calibrate] 无有效验证集预测")
            return None

        val_preds = np.concatenate(all_preds)
        y_val = np.concatenate(all_labels)

        if len(np.unique(y_val)) < 2 or len(val_preds) < 50:
            logger.warning(f"[CV Calibrate] 样本不足, 跳过校准")
            return None

        # 拟合校准器 (platt or isotonic)
        if self._calibration_method == 'platt':
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning)
                warnings.filterwarnings('ignore', category=UserWarning)
                calibrator = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
            calibrator.fit(val_preds.reshape(-1, 1), y_val)
            calibrated = calibrator.predict_proba(val_preds.reshape(-1, 1))[:, 1]
        else:  # isotonic
            calibrator = IsotonicRegression(y_min=self._prob_clip_min,
                                            y_max=self._prob_clip_max,
                                            out_of_bounds='clip')
            calibrator.fit(val_preds, y_val)
            calibrated = calibrator.predict(val_preds)

        # 退化检测
        prob_range = calibrated.max() - calibrated.min()
        if prob_range < 0.01:
            logger.warning("概率校准退化: 模型可能无区分力 (range < 0.01)")

        # 校准质量监控
        if self._calibration_monitoring:
            try:
                auc_raw = float(roc_auc_score(y_val, val_preds))
                auc_cal = float(roc_auc_score(y_val, calibrated))
                ece_raw = self._compute_ece(y_val, val_preds, n_bins=10)
                ece_cal = self._compute_ece(y_val, calibrated, n_bins=10)
                logger.info(
                    f"[CV Calibrate/{self._calibration_method}] "
                    f"({len(y_val)} samples, {len(active_set)}/{len(self.models)} active models) "
                    f"raw_AUC={auc_raw:.4f} cal_AUC={auc_cal:.4f} "
                    f"(Δ={auc_cal - auc_raw:+.4f}) | "
                    f"raw_ECE={ece_raw:.4f} cal_ECE={ece_cal:.4f} "
                    f"(Δ={ece_cal - ece_raw:+.4f})"
                )
            except ValueError:
                pass

        return calibrator

    @staticmethod
    def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """计算 Expected Calibration Error (ECE)。

        ECE = Σ (|B_m| / N) * |acc(B_m) - conf(B_m)|

        其中 B_m 是第 m 个概率区间的样本, acc 是区间内的真实准确率,
        conf 是区间内的平均预测概率。越低越好 (0 = 完美校准)。

        这是业界标准的校准质量指标 (Guo et al., 2017 "On Calibration of Modern Neural Networks")。
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(np.clip(y_prob, 0, 1), bins) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)
        ece = 0.0
        for m in range(n_bins):
            idx = bin_ids == m
            if idx.sum() == 0:
                continue
            acc_m = y_true[idx].mean()
            conf_m = y_prob[idx].mean()
            ece += (idx.sum() / len(y_true)) * abs(acc_m - conf_m)
        return float(ece)

    def _apply_calibration(self, raw_preds: np.ndarray) -> np.ndarray:
        """应用校准 + 概率裁剪。

        支持三种校准器类型:
        - LogisticRegression (Platt): 调用 predict_proba[:, 1]
        - IsotonicRegression: 调用 predict()
        - None: 跳过校准, 仅做裁剪
        """
        preds = raw_preds.copy()
        if self.calibrator is not None:
            if isinstance(self.calibrator, LogisticRegression):
                # Platt Scaling: predict_proba returns [p0, p1] per sample
                preds = self.calibrator.predict_proba(preds.reshape(-1, 1))[:, 1]
            else:
                # IsotonicRegression (and any other .predict()-based calibrator)
                preds = self.calibrator.predict(preds)
        # P3-2: 概率裁剪
        preds = np.clip(preds, self._prob_clip_min, self._prob_clip_max)
        return preds

    # ================================================================
    #  AUC门控 (P0-2: 替换IC门控)
    # ================================================================

    def _auc_gate(self, val_auc: float) -> Tuple[str, float]:
        """
        AUC门控三档（对齐Alpha191）：
        AUC >= auc_gate_full → full, 仓位1.0x
        auc_gate_min <= AUC < auc_gate_full → half, 仓位0.5x
        AUC < auc_gate_min → empty, 仓位0.0x（空仓）
        """
        if val_auc >= self._auc_gate_full:
            return "full", 1.0
        elif val_auc >= self._auc_gate_min:
            return "half", 0.5
        else:
            return "empty", 0.0

    def _check_prob_collapse(self, preds: np.ndarray, n_select: int = None) -> bool:
        """检测选中股票的预测概率是否坍缩（模型无区分力）。
        
        XGBoost best_iter=0 + isotonic 校准时，选出的 top-N 股票
        概率几乎完全相同，模型实际上没有选股能力。
        
        检测维度：
        1. 若 n_select 给定：检查 top-N 的 (max-min) < threshold
        2. 备选：全局 std < threshold
        
        Returns:
            True  = 选出的股票无区分力，应跳过
            False = 区分力正常
        """
        if self._prob_collapse_threshold <= 0:
            return False
        
        if n_select and n_select < len(preds):
            # 核心检测: 选中股票的分数范围
            top_n = np.sort(preds)[-n_select:]
            top_range = float(top_n[-1] - top_n[0])
            if top_range < self._prob_collapse_threshold:
                logger.warning(
                    f"[ProbCollapse] selected top-{n_select} range={top_range:.8f} < threshold={self._prob_collapse_threshold}, "
                    f"model has no discrimination, skip selection"
                )
                return True
            # 选中股票区分度正常
            return False
        
        # 回退: 全局标准差
        pred_std = float(np.std(preds))
        if pred_std < self._prob_collapse_threshold:
            logger.warning(
                f"[ProbCollapse] global std={pred_std:.6f} < threshold={self._prob_collapse_threshold}, "
                f"model has no discrimination, skip selection"
            )
            return True
        return False

    # ================================================================
    #  构建训练数据集
    # ================================================================

    def _build_training_dataset(self, end_date: datetime):
        """
        构建训练数据集 — V10独立版：二分类标签 + Purged K-Fold CV + 全市场中性化
        """
        rng = np.random.RandomState(42)  # P3-1: 局部RNG
        _orig_full_cal = self.full_calendar
        _truncated_cal = [d for d in self.full_calendar if d < end_date]
        if _truncated_cal:
            self.full_calendar = _truncated_cal

        all_snaps = self._get_snapshot_dates()

        valid_snaps = []
        for d in all_snaps:
            if d >= end_date:
                continue
            # [2026-08-07 PIT 对齐] 预检与主循环同语义: 标签 base = d-1 收盘, future = d-1+pred_days
            base_d = self._get_trading_day_offset(d, -1)
            if base_d is None or base_d >= d:
                continue
            future = self._get_trading_day_offset(base_d, self.pred_days)
            if future is None or future >= end_date:
                continue
            valid_snaps.append(d)

        if not valid_snaps:
            logger.warning("No valid snapshots (insufficient future data for labels)")
            self.full_calendar = _orig_full_cal
            return None, None, None, None, None

        valid_snaps.sort()
        train_snaps = (valid_snaps[-self.train_window:]
                       if len(valid_snaps) >= self.train_window
                       else valid_snaps)

        # [P2-2 FIX] 防泄露断言: 训练数据快照日期必须早于当前调仓日期
        if train_snaps and max(train_snaps) >= end_date:
            logger.error(f"[LEAK CHECK FAILED] Latest training snapshot {max(train_snaps)} "
                        f">= rebalance date {end_date}, data leakage!")
            # 不抛异常而是返回 None 触发空仓 (生产安全)
            self.full_calendar = _orig_full_cal
            return None, None, None, None, None

        if not train_snaps:
            self.full_calendar = _orig_full_cal
            return None, None, None, None, None

        logger.info(f"Training snapshots: {train_snaps[0].date()} ~ {train_snaps[-1].date()}, "
                    f"total {len(train_snaps)} days")

        # 构建样本
        samples_with_ret = []
        for dt in train_snaps:
            snap = self._load_snapshot(dt)
            if snap is None or snap.empty:
                continue

            # [2026-08-07 PIT 对齐] 标签 base 日 = 特征因子的信息时点 (dt-1 收盘):
            #   V10 factors_{dt} 内容截至 dt-1 收盘(reference_date=prev) → 标签从 dt-1 close 模拟入场
            #   future = base_date + pred_days = dt+19 收盘 (持有 20 交易日不变)
            #   修复前: base = dt close (晚特征 1 天, PIT 不对齐)
            base_date = self._get_trading_day_offset(dt, -1)
            if base_date is None or base_date >= dt:
                continue
            future_date = self._get_trading_day_offset(base_date, self.pred_days)
            if future_date is None or future_date >= end_date:
                continue

            base_raw, base_adj = self._get_label_components(snap['code'], base_date)
            fut_raw, fut_adj = self._get_label_components(snap['code'], future_date)

            if len(base_raw) == 0 or len(fut_raw) == 0:
                continue

            returns = {}
            for code in snap['code'].unique():
                r_now = base_raw.get(code, np.nan)
                r_fut = fut_raw.get(code, np.nan)
                a_now = base_adj.get(code, np.nan)
                a_fut = fut_adj.get(code, np.nan)
                if (pd.notna(r_now) and pd.notna(r_fut) and r_now != 0
                        and pd.notna(a_now) and pd.notna(a_fut)):
                    adj_now = r_now * a_now
                    adj_fut = r_fut * a_fut
                    returns[code] = adj_fut / adj_now - 1

            if not returns:
                continue

            snap = snap.copy()
            snap['label_raw'] = snap['code'].map(returns)
            snap['snapshot_date'] = dt
            snap.dropna(subset=['label_raw'], inplace=True)
            if not snap.empty:
                samples_with_ret.append(snap)

        self.full_calendar = _orig_full_cal

        if not samples_with_ret:
            logger.warning("No valid training samples")
            return None, None, None, None, None

        data = pd.concat(samples_with_ret, ignore_index=True)

        # 确定性修复: 去除重复 (snapshot_date, code) 行
        # 不同进程加载顺序不同 → _row_idx 不同 → sort后行序不同 → OLS betas 不同
        _before_dedup = len(data)
        data = data.drop_duplicates(subset=['snapshot_date', 'code'], keep='first')
        _n_dropped = _before_dedup - len(data)
        if _n_dropped > 0:
            logger.warning(f"[Dedup] Removed {_n_dropped} duplicate (snapshot_date, code) rows "
                          f"({_before_dedup} → {len(data)})")
        data = data.reset_index(drop=True)

        # 方案B: 添加唯一行号索引作为排序 tiebreaker，确保跨进程行序完全确定
        data['_row_idx'] = np.arange(len(data))
        data = data.sort_values(['snapshot_date', 'code', '_row_idx'], kind='mergesort').reset_index(drop=True)
        logger.info(f"Raw samples: {len(data)}")

        # 二分类标签：按配置选择阈值方式
        if self._label_method == 'absolute':
            # absolute 模式：20日收益率(前复权) > 固定阈值 → label=1
            # 收益率已使用 raw_close × backAdjustFactor 正确计算
            # 正样本比例由市场环境自然决定（大牛市多，熊市少甚至可能为0）
            threshold = self._label_absolute_threshold
            data['label'] = (data['label_raw'] > threshold).astype(int)
        elif self._label_method == 'quantile':
            # quantile 模式：收益率截面排名前 label_quantile 分位 → 1
            q = self._label_quantile
            data['label'] = (
                data.groupby('snapshot_date')['label_raw']
                .transform(lambda x: (x >= x.quantile(1 - q)).astype(int))
            )
        else:
            # median 模式（默认）：跑赢截面中位数 → 1
            data['label'] = (
                data.groupby('snapshot_date')['label_raw']
                .transform(lambda x: (x > x.median()).astype(int))
            )

        n_pos = int(data['label'].sum())
        method_label = f"method={self._label_method}"
        if self._label_method == 'quantile':
            method_label += f", q={self._label_quantile}"
        elif self._label_method == 'absolute':
            method_label += f", threshold={self._label_absolute_threshold}"
        logger.info(f"V10 Classification labels ({method_label}):"
                    f" pos={n_pos} ({100*n_pos/len(data):.1f}%), "
                    f"neg={len(data)-n_pos} ({100*(len(data)-n_pos)/len(data):.1f}%)")

        # 时序验证集划分 (80/20) — 用于最终模型训练和AUC门控
        n_snaps = len(train_snaps)
        val_split_idx = int(n_snaps * 0.8)
        train_snap_dates = train_snaps[:val_split_idx]
        val_snap_dates = train_snaps[val_split_idx:]

        train_data = data[data['snapshot_date'].isin(train_snap_dates)].copy()
        val_data = data[data['snapshot_date'].isin(val_snap_dates)].copy()

        logger.info(f"Time-series split: train {len(train_snap_dates)} days, "
                    f"val {len(val_snap_dates)} days")
        logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

        # === 中性化处理 ===
        if self._neutralize_enabled:
            logger.info("Applying neutralization to base factors...")

            raw_cols = [f'{fname}_raw' for fname in self.ALL_FACTORS
                        if f'{fname}_raw' in train_data.columns]
            score_cols = [f'{fname}_score' for fname in self.ALL_FACTORS
                          if f'{fname}_score' in train_data.columns]
            conf_cols = [f'{fname}_conf' for fname in self.ALL_FACTORS
                         if f'{fname}_conf' in train_data.columns]

            # 方案B: mergesort 确定性排序 + _row_idx tiebreaker
            # 不同进程 hash seed 不同导致 pd.isin/groupby 内部行序不同 → OLS betas 不同
            # 添加 _row_idx 作为第三排序键，确保即使 snapshot_date+code 有重复也完全确定
            train_data = train_data.sort_values(
                ['snapshot_date', 'code', '_row_idx'], kind='mergesort'
            ).reset_index(drop=True)
            val_data = val_data.sort_values(
                ['snapshot_date', 'code', '_row_idx'], kind='mergesort'
            ).reset_index(drop=True)

            # 诊断: 前5行排序顺序确认
            top5 = train_data[['snapshot_date', 'code']].head(5)
            logger.info(f"[SortCheck] first5: {list(zip(top5['snapshot_date'].astype(str), top5['code']))}")

            # 诊断: 训练数据全量指纹 (验证跨进程数据一致性)
            _fp_cols = ['snapshot_date', 'code'] + raw_cols[:3]  # 日期+代码+前3个raw因子
            _fp_data = train_data[_fp_cols].head(100).to_csv(index=False)
            import hashlib as _hashlib
            _data_fingerprint = _hashlib.md5(_fp_data.encode()).hexdigest()[:12]
            logger.info(f"[DataFingerprint] train_rows={len(train_data)}, "
                       f"fp100={_data_fingerprint}, "
                       f"raw_sum={float(train_data[raw_cols].replace([np.inf, -np.inf], np.nan).sum().sum()):.2f}")

            self._train_ols_betas = None  # 每次重新fit
            # 诊断: 记录中性化前的原始数据摘要 (定位 OLS 之前是否已存在数值差异)
            if raw_cols:
                raw_summary = float(train_data[raw_cols].replace([np.inf, -np.inf], np.nan).sum().sum())
                logger.info(f"[PreNeutralize] raw_data_sum={raw_summary:.2f}, "
                           f"train_rows={len(train_data)}, raw_cols={len(raw_cols)}")
            self._score_neu_params = {}   # P0-1 FIX: reset score neutralization params
            train_data = self._neutralize(train_data, raw_cols)
            val_data = self._neutralize(val_data, raw_cols)

            train_data = self._apply_score_neutralization(train_data, score_cols + conf_cols, fit=True)
            val_data = self._apply_score_neutralization(val_data, score_cols + conf_cols)

            # 诊断: 记录中性化后的数据摘要 + OLS beta hash
            neu_cols = [c for c in train_data.columns if c.endswith('_neu')]
            if neu_cols:
                neu_summary = float(train_data[neu_cols].replace([np.inf, -np.inf], np.nan).sum().sum())
                beta_hash = 0
                if isinstance(self._train_ols_betas, dict):
                    # 确定性修复: 用 md5(round(beta, 10)) 替代 hash(float)
                    # hash(float) 对 1e-17 级别的浮点差异产生完全不同的值，是误报
                    import hashlib as _hashlib
                    _beta_bytes = b''
                    for k, v in sorted(self._train_ols_betas.get('betas', {}).items()):
                        if isinstance(v, np.ndarray):
                            _beta_bytes += np.round(v[:5], 10).tobytes()
                    beta_hash = int(_hashlib.md5(_beta_bytes).hexdigest()[:16], 16)
                logger.info(f"[PostNeutralize] neu_sum={neu_summary:.2f}, "
                           f"beta_hash={abs(beta_hash)}, neu_cols={len(neu_cols)}")

            self._industry_dummy_cols = []
            logger.info("Neutralization complete")
        else:
            train_data = train_data.reset_index(drop=True)
            val_data = val_data.reset_index(drop=True)
            self._industry_dummy_cols = []

        # 方案B: 清理 _row_idx 辅助列（仅用于排序 tiebreaker，不进入特征或模型）
        train_data.drop(columns=['_row_idx'], inplace=True, errors='ignore')
        val_data.drop(columns=['_row_idx'], inplace=True, errors='ignore')

        # 构建候选特征列表
        feats = []
        for fname in self.ALL_FACTORS:
            raw_neu_col = f'{fname}_raw_neu'
            score_col = f'{fname}_score'
            conf_col = f'{fname}_conf'

            if self._neutralize_enabled:
                if raw_neu_col in train_data.columns:
                    feats.append(raw_neu_col)
            else:
                raw_col = f'{fname}_raw'
                if raw_col in train_data.columns:
                    feats.append(raw_col)

            if score_col in train_data.columns:
                feats.append(score_col)
            if conf_col in train_data.columns:
                feats.append(conf_col)

        # 派生特征
        if 'effort_result_raw' in train_data.columns:
            if 'effort_result_raw' not in feats:
                feats.append('effort_result_raw')
        if self._neutralize_enabled and 'effort_result_raw_neu' in train_data.columns:
            if 'effort_result_raw_neu' not in feats:
                feats.append('effort_result_raw_neu')

        feats = sorted(set(f for f in feats if f in train_data.columns))
        logger.info(f"Candidate features: {len(feats)}")

        # === 特征选择: IC预过滤 + CV/MDA多折投票 (对齐Alpha191) ===
        # Step 1: IC预过滤 (极宽松, 仅除噪声)
        logger.info(f"[Feature Selection] Step 1: IC pre-filter (threshold={self._feat_sel_min_ic})...")
        pre_filtered, ic_scores = self._prefilter_features(
            train_data, val_data, feats, target_col='label')

        if not pre_filtered or len(pre_filtered) < self._min_features:
            logger.warning(f"Prefilter insufficient ({len(pre_filtered)}), using all {len(feats)} candidates")
            pre_filtered = feats
            # 重新为所有候选特征计算 IC (避免 fallback 后 ic_scores 为空的展示问题)
            if 'label' in train_data.columns and len(feats) > 0:
                tr_ic = train_data[feats].replace([np.inf, -np.inf], np.nan)
                psi_info_ic = {}
                y_fallback = train_data['label'].values.astype(float)
                for f in feats:
                    m = tr_ic[f].median(skipna=True)
                    psi_info_ic[f] = (0.0, 0.0 if pd.isna(m) else m)
                _, full_fallback_scores = self._compute_ic_vectorized(tr_ic, y_fallback, feats, psi_info_ic)
                ic_scores = {f: full_fallback_scores[f]['ic_abs'] for f in full_fallback_scores}
            else:
                ic_scores = {}

        # [2026-08-18 对齐 small] HP-Search 超参选择 (A+B+C, 写回 self.xgb_params)
        # 评估特征 = pre_filtered (与 PKF 折内 MDA 同一特征集), 生产 val 段零接触
        self._hp_best_name, self._hp_best_score = self._select_hyperparams(
            train_data, pre_filtered, target_col='label')

        # Step 2: CV + MDA多折投票 (主选择器)
        # [CFI] cfi_mode: CV 前全局聚类 → 簇级置换 MDA 投票 (Prado MLAM Ch.6)
        logger.info("[Feature Selection] Step 2: Purged K-Fold CV + MDA voting...")
        if self._cluster_cfi_enabled and _CFI_AVAILABLE:
            _cfi_cfg = self._cluster_cfg_namespace()
            _clusters = cluster_features_global(train_data, pre_filtered, _cfi_cfg)
            self._selected_features = self._purged_kfold_cv(
                train_data, pre_filtered, ic_scores, cfi_clusters=_clusters)
            logger.info(f"[CFI] 全局聚类: {len(pre_filtered)}特征 → {len(_clusters)}簇")
        else:
            self._selected_features = self._purged_kfold_cv(
                train_data, pre_filtered, ic_scores)

        self.feature_names = self._selected_features
        logger.info(f"Final selected features: {len(self.feature_names)}")

        if not self.feature_names:
            return None, None, None, None, None

        # === [SHAP] 因子质量验证: 单调性 + 离群点 ===
        n_features_before = len(self.feature_names)
        if self._shap_enabled and self.feature_names:
            try:
                logger.info("[SHAP] Starting factor quality verification...")
                # 先训练轻量生产模型用于 SHAP 解释
                X_full = train_data[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0).values
                y_full = train_data['label'].values.astype(int)

                dtrain_tmp = xgb.DMatrix(X_full, label=y_full, feature_names=self.feature_names)
                dval_tmp = xgb.DMatrix(
                    val_data[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0).values,
                    label=val_data['label'].values.astype(int),
                    feature_names=self.feature_names)

                # [P0-1 FIX] 使用局部变量存储 SHAP 临时训练的种子集成模型,
                # 避免覆写 self.models (可能是 CV 折集成模型, 在 before_market_open 中
                # _train_cv_ensemble() 之后调用 _build_training_dataset 时已设置)
                shap_models = self._train_ensemble(dtrain_tmp, dval_tmp, force_seed=True)
                _orig_models_for_shap = self.models
                try:
                    self.models = shap_models
                    shap_vals, X_sample = self._compute_shap_values(X_full)
                finally:
                    self.models = _orig_models_for_shap  # 必须恢复, 即使 SHAP 异常
                if shap_vals is not None:
                    passed, removed = self._check_shap_monotonicity(shap_vals, X_sample)
                    outliers = self._detect_shap_outliers(shap_vals)

                    # 剔除噪音特征
                    removed_names = {f for f, _ in removed}
                    self._selected_features = [f for f in self._selected_features
                                               if f not in removed_names]
                    self.feature_names = self._selected_features

                    # 空特征保护: 若无可用特征, 返回 None 触发空仓 (对齐 AUC<0.48 行为)
                    if not self.feature_names:
                        logger.warning("[SHAP] All features removed as noise → "
                                       "no usable features this period (same as AUC<0.48)")
                        self._save_shap_report(
                            end_date, passed, removed, outliers,
                            n_features_before, 0)
                        return None, None, None, None, None

                    self._save_shap_report(
                        end_date if not hasattr(self, '_current_rebalance_date')
                        else self._current_rebalance_date,
                        passed, removed, outliers,
                        n_features_before, len(self._selected_features))
            except ImportError:
                logger.warning("[SHAP] shap package not installed, skipping")
            except Exception as e:
                logger.warning(f"[SHAP] Analysis failed ({e}), keeping all features")

        # 构建DMatrix
        X_train = train_data[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_val = val_data[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)

        y_train = train_data['label'].values.astype(int)
        y_val = val_data['label'].values.astype(int)

        # [CFI-诊断] 不在本函数执行: self.models 此时尚未训练(需 CV 折集成完成后),
        # 诊断调用已移至 before_market_open 模型训练完成后 (见 before_market_open)。

        # 序贯采样权重
        truncated_cal = [d for d in _orig_full_cal if d < end_date]
        try:
            sample_weights = compute_sample_weights(train_data, truncated_cal, self.pred_days)
            if len(sample_weights) != len(train_data):
                sample_weights = np.ones(len(train_data))
        except Exception as e:
            logger.warning(f"Sample weights failed: {e}, using uniform")
            sample_weights = np.ones(len(train_data))

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights,
                             feature_names=self.feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)

        return dtrain, dval, sample_weights, train_data, val_data

    def _apply_class_balance(self, params: dict, y_train: np.ndarray) -> dict:
        """按 XGBoost 最佳实践注入 scale_pos_weight，自动平衡类别（带上限保护）。

        XGBoost 文档推荐: scale_pos_weight = count(negative) / count(positive)
        这是 XGBoost 处理不均衡分类的标准方法——对少数类的梯度乘以该权重，
        等价于在损失函数 L = Σ w_i * l(y_i, ŷ_i) 中赋予少数类更高的错误惩罚。

        P0-3 FIX: 添加上限保护 (max_scale), 防止极端不均衡 (正样本 < 5%) 时
        scale_pos_weight 过大 (可达 20+) 导致严重过拟合。

        对比其它方案的优劣:
        - SMOTE/过采样: 增加内存和训练时间，可能引入合成噪声
        - 欠采样: 丢弃多数类信息，损失建模潜力
        - class_weight: sklearn 方案，但 XGBoost 不支持此参数
        → scale_pos_weight: 零额外开销，XGBoost 原生支持，业界标准
        → 上限保护: 防止对少数类的过度放大

        Returns:
            params 的浅拷贝，注入 scale_pos_weight (如果 mode="auto")
            返回额外标志 _spw_was_capped=True 表示触发了上限
        """
        if self._class_balance_mode != 'auto':
            return params

        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())

        if n_pos == 0 or n_neg == 0:
            logger.warning("[class_balance] 单一类别, 跳过 scale_pos_weight")
            return params

        scale = float(n_neg) / float(n_pos)
        pos_ratio = n_pos / (n_pos + n_neg)

        # P0-3: 上限保护
        max_scale = self._class_balance_max_scale
        was_capped = False
        if scale > max_scale:
            logger.warning(
                f"[class_balance] ⚠ scale_pos_weight={scale:.2f} > max={max_scale}, "
                f"触发上限保护 (pos_ratio={pos_ratio:.1%}), capped to {max_scale}"
            )
            scale = max_scale
            was_capped = True

        params = params.copy()
        params['scale_pos_weight'] = scale

        logger.info(f"[class_balance] auto: pos={n_pos}, neg={n_neg}, "
                    f"ratio={pos_ratio:.2%}, scale_pos_weight={scale:.3f}"
                    + (" (CAPPED)" if was_capped else ""))

        # P0-3: 自动切换到 quantile 标签模式 (由调用方在训练前检查)
        if (self._class_balance_auto_switch and
                pos_ratio < 0.10 and
                hasattr(self, '_label_method')):
            logger.warning(
                f"[class_balance] ⚠ 正样本比例 {pos_ratio:.1%} < 10%, "
                f"强烈建议切换 label_method 为 quantile 模式以改善样本均衡"
            )

        return params

    # ================================================================
    #  特征工程 - 派生特征
    # ================================================================

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建派生特征"""
        df = df.copy()

        if 'effort_result_raw' in df.columns:
            raw_vals = df['effort_result_raw'].copy()
            df['effort_result_raw_abs'] = raw_vals.abs()
            z_score = (raw_vals - raw_vals.mean()) / (raw_vals.std() + 1e-8)
            df['effort_result_extreme'] = (z_score.abs() > 2.0).astype(float)

        if all(c in df.columns for c in ['spring_quality_score', 'sos_strength_score']):
            df['spring_sos_proximity'] = df['spring_quality_score'] * df['sos_strength_score']

        return df

    # ================================================================
    #  特征矩阵构建（预测时使用）
    # ================================================================

    def _build_feature_matrix(self, snap: pd.DataFrame, pre_neutralized: bool = False) -> pd.DataFrame:
        """从快照数据构建特征矩阵"""
        if not pre_neutralized:
            if self._neutralize_enabled:
                raw_cols = [f'{fname}_raw' for fname in self.ALL_FACTORS if f'{fname}_raw' in snap.columns]
                score_cols = [f'{fname}_score' for fname in self.ALL_FACTORS if f'{fname}_score' in snap.columns]
                conf_cols = [f'{fname}_conf' for fname in self.ALL_FACTORS if f'{fname}_conf' in snap.columns]

                snap = self._neutralize(snap, raw_cols)
                snap = self._apply_score_neutralization(snap, score_cols + conf_cols)

        snap = self._prepare_features(snap)

        available_feats = []
        for fname in self.ALL_FACTORS:
            if self._neutralize_enabled:
                raw_neu = f'{fname}_raw_neu'
                if raw_neu in snap.columns:
                    available_feats.append(raw_neu)
            else:
                raw_col = f'{fname}_raw'
                if raw_col in snap.columns:
                    available_feats.append(raw_col)

            score_col = f'{fname}_score'
            if score_col in snap.columns:
                available_feats.append(score_col)
            conf_col = f'{fname}_conf'
            if conf_col in snap.columns:
                available_feats.append(conf_col)

        # [2026-08-05 修复] 与训练侧 _build_training_dataset (L2840-2842) 对齐:
        # 训练侧候选特征无条件含 effort_result_raw (raw 原值, 与 neutralized 版并存)
        # 此处原缺失该列 → 特征选择选中它时预测缺列 fail-fast 抛错 (此前静默补0掩盖)
        # 现在补上 effort_result_raw, 训练/预测特征集完全一致
        derived_cols = ['effort_result_raw', 'effort_result_raw_abs',
                        'effort_result_extreme', 'spring_sos_proximity']
        for col in derived_cols:
            if col in snap.columns:
                available_feats.append(col)

        available_feats = sorted(set(available_feats))
        X = snap[available_feats].replace([np.inf, -np.inf], np.nan).fillna(0)

        # [2026-08-05 修复] 业界最佳实践: 训练/预测特征集必须严格一致。
        # 缺失列直接 fail-fast 抛错 (显式暴露数据管线问题), 而非静默补 0
        # (静默补 0 会让模型在"未知值"上预测, 与训练分布不一致, 结果不可信)
        missing_cols = [c for c in self.feature_names if c not in X.columns]
        if missing_cols:
            raise ValueError(
                f'预测数据缺少训练特征列: {sorted(missing_cols)[:8]}... '
                f'— 训练/预测特征集必须一致 (共缺 {len(missing_cols)} 列)'
            )
        X = X[self.feature_names]

        return X

    def _neutralize_full_market(self, snap: pd.DataFrame) -> pd.DataFrame:
        """
        P1-4: 全市场中性化。
        先对全市场快照做中性化，再由调用方过滤候选池。
        """
        if not self._neutralize_enabled or not self._full_market_neutralize:
            return snap

        raw_cols = [f'{fname}_raw' for fname in self.ALL_FACTORS if f'{fname}_raw' in snap.columns]
        score_cols = [f'{fname}_score' for fname in self.ALL_FACTORS if f'{fname}_score' in snap.columns]
        conf_cols = [f'{fname}_conf' for fname in self.ALL_FACTORS if f'{fname}_conf' in snap.columns]

        snap = self._neutralize(snap, raw_cols)
        snap = self._apply_score_neutralization(snap, score_cols + conf_cols)
        return snap

    # ================================================================
    #  模型评估
    # ================================================================

    def _calc_model_performance(self, dtrain: xgb.DMatrix, dval: xgb.DMatrix = None) -> dict:
        """计算分类模型性能指标（集成版）"""
        if not self.models:
            return {}

        perf = {}

        preds_train = self._predict_ensemble(dtrain)
        labels_train = dtrain.get_label().astype(int)

        ll_train = log_loss(labels_train, preds_train, labels=[0, 1])
        try:
            auc_train = roc_auc_score(labels_train, preds_train)
        except ValueError:
            auc_train = float('nan')
        spearman_train = spearmanr(preds_train, labels_train)[0] if len(labels_train) > 1 else np.nan

        perf['train'] = {
            'logloss': ll_train, 'auc': auc_train,
            'spearman_ic': spearman_train, 'n_samples': len(labels_train)
        }

        if dval is not None:
            preds_val = self._predict_ensemble(dval)
            labels_val = dval.get_label().astype(int)
            ll_val = log_loss(labels_val, preds_val, labels=[0, 1])
            try:
                auc_val = roc_auc_score(labels_val, preds_val)
            except ValueError:
                auc_val = float('nan')
            spearman_val = spearmanr(preds_val, labels_val)[0] if len(labels_val) > 1 else np.nan

            perf['val'] = {
                'logloss': ll_val, 'auc': auc_val,
                'spearman_ic': spearman_val, 'n_samples': len(labels_val)
            }

        return perf

    def _log_feature_importance(self, top_n: int = None):
        """记录特征重要性（使用集成中第一个模型）"""
        if not self.models:
            return
        model = self.models[0]
        importance = model.get_score(importance_type='gain')
        if not importance:
            return
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        shown = sorted_items[:top_n] if top_n is not None else sorted_items

        total_feats = len(self.feature_names) if self.feature_names else len(sorted_items)
        logger.info("Feature Importance Gain (%d/%d features with gain>0):",
                    len(sorted_items), total_feats)
        for rank, (feat, score) in enumerate(shown, 1):
            logger.info(f"  #{rank:>3d} {feat:<35s}  gain={score:.2f}")

    # ================================================================
    #  SHAP 因子质量验证 (单调性 + 离群点 + 报告)
    # ================================================================

    def _compute_shap_values(self, X_train: np.ndarray):
        """在训练集子样本上计算 SHAP 值 (TreeExplainer)。

        Args:
            X_train: 训练特征矩阵 (numpy array, 完整训练集)
        Returns:
            (shap_values, X_sample): SHAP 值和对应的子样本特征矩阵
        """
        if not self.models:
            logger.warning("[SHAP] No trained models, skipping")
            return None, None

        import shap
        model = self.models[0]
        n = min(self._shap_sample_size, len(X_train))
        idx = np.random.RandomState(42).choice(len(X_train), n, replace=False)
        X_sample = X_train[idx]
        explainer = shap.TreeExplainer(model, feature_names=self.feature_names)
        shap_vals = explainer.shap_values(X_sample)
        logger.info(f"[SHAP] Computed on {n} samples, {len(self.feature_names)} features")
        return shap_vals, X_sample

    def _check_shap_monotonicity(self, shap_values: np.ndarray,
                                  X_sample: np.ndarray):
        """SHAP 单调性验证：检测无规律噪音特征。

        Returns:
            passed: [(fname, |r|, pattern), ...]
            removed: [(fname, |r|), ...]
        """
        from scipy.stats import spearmanr
        threshold = self._shap_monotonicity_threshold
        passed = []
        removed = []

        for i, fname in enumerate(self.feature_names):
            x = X_sample[:, i]
            sv = shap_values[:, i]
            valid = np.isfinite(x) & np.isfinite(sv)
            if valid.sum() < 100:
                removed.append((fname, 0.0))
                continue

            r, p = spearmanr(x[valid], sv[valid])
            abs_r = abs(r)

            pattern = 'monotonic'
            if abs_r >= threshold:
                pattern = 'negative' if r < 0 else 'positive'
                passed.append((fname, abs_r, pattern))
            else:
                # 检测倒U型
                import pandas as pd
                try:
                    bins = pd.qcut(x[valid], 15, duplicates='drop')
                    bin_means = pd.Series(sv[valid]).groupby(bins, observed=False).mean()
                    peaks = (bin_means.diff().shift(-1) < 0) & (bin_means.diff() > 0)
                    if peaks.sum() == 1:
                        pattern = 'inverted-U'
                        passed.append((fname, abs_r, pattern))
                    else:
                        removed.append((fname, abs_r))
                except Exception:
                    removed.append((fname, abs_r))

        logger.info(f"[SHAP] Monotonicity check: {len(passed)} passed, "
                    f"{len(removed)} removed (|r|<{threshold})")
        return passed, removed

    def _detect_shap_outliers(self, shap_values: np.ndarray):
        """检测 SHAP 离群点：前 1% 样本贡献 > threshold 总 SHAP 绝对值。

        Returns:
            {fname: concentration_ratio, ...}
        """
        threshold = self._shap_outlier_threshold
        outliers = {}
        for i, fname in enumerate(self.feature_names):
            abs_shap = np.abs(shap_values[:, i])
            top_n = max(1, int(len(abs_shap) * 0.01))
            top_idx = np.argpartition(abs_shap, -top_n)[-top_n:]
            concentration = abs_shap[top_idx].sum() / max(abs_shap.sum(), 1e-10)
            if concentration > threshold:
                outliers[fname] = round(float(concentration), 4)
        if outliers:
            logger.warning(f"[SHAP] Outlier-dependent features ({threshold=}): {len(outliers)}")
        return outliers

    def _save_shap_report(self, rebalance_date, passed, removed, outliers,
                          n_features_before, n_features_after):
        """保存 SHAP 分析报告到 output_dir。

        输出: report.json (分析结论) + 日志摘要
        """
        import json, os
        dt_str = rebalance_date.strftime('%Y%m%d') if hasattr(rebalance_date, 'strftime') else str(rebalance_date)
        out_dir = os.path.join(self._shap_output_dir, f"rebalance_{dt_str}")
        os.makedirs(out_dir, exist_ok=True)

        report = {
            "rebalance_date": dt_str,
            "features_before": n_features_before,
            "features_after": n_features_after,
            "passed": [{"feature": f, "monotonicity": round(r, 4), "pattern": p}
                       for f, r, p in passed],
            "removed": [{"feature": f, "monotonicity": round(r, 4), "reason": "noise"}
                        for f, r in removed],
            "outliers": outliers,
        }

        path = os.path.join(out_dir, "report.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"[SHAP] Report saved to {path} "
                    f"({n_features_before}→{n_features_after} features)")
        if removed:
            names = [f[0] for f in removed]
            logger.warning(f"[SHAP] Removed {len(removed)} noise features: {names}")

    # ================================================================
    #  趋势过滤 (V10保留)
    # ================================================================

    def _precompute_trend_factors(self, tf_cfg: dict):
        """预计算全市场趋势因子"""
        lookback = tf_cfg.get('lookback', 5)
        ma_period = tf_cfg.get('ma_period', 20)

        df = self.full_df_daily.copy()
        df = df.sort_values(['code_digits', 'trade_date']).reset_index(drop=True)
        df['adj_close'] = df['raw_close'] * df['backAdjustFactor']

        # 避免 lambda: shift 后向量化计算 (Cython 加速路径)
        _shifted = df.groupby('code_digits')['adj_close'].shift(lookback)
        df['price_decline_factor'] = df['adj_close'] / _shifted - 1

        if 'volume' in df.columns:
            vol = df.groupby('code_digits')['volume'].transform(lambda s: s.astype(float))
            v0 = vol.groupby(df['code_digits']).shift(lookback - 1)
            v1 = vol.groupby(df['code_digits']).shift(lookback - 2)
            v3 = vol.groupby(df['code_digits']).shift(1)
            v4 = vol
            vol_mean = vol.groupby(df['code_digits']).transform(
                lambda s: s.rolling(lookback, min_periods=lookback).mean())
            slope_denom = lookback * (lookback**2 - 1) / 12.0
            slope = (-2 * v0 - v1 + v3 + 2 * v4) / slope_denom
            df['volume_rise_factor'] = slope / vol_mean.replace(0, np.nan)
        else:
            df['volume_rise_factor'] = np.nan

        ma20 = df.groupby('code_digits')['adj_close'].transform(
            lambda s: s.rolling(ma_period, min_periods=ma_period).mean())
        df['ma20_distance'] = (ma20 - df['adj_close']) / ma20.replace(0, np.nan)

        cache = df[['code_digits', 'trade_date',
                     'price_decline_factor', 'volume_rise_factor', 'ma20_distance']].copy()
        cache = cache.set_index(['code_digits', 'trade_date'])
        cache = cache.dropna(subset=['price_decline_factor', 'volume_rise_factor', 'ma20_distance'],
                             how='all')
        self._trend_factors_cache = cache
        logger.info(f"[TrendFilter] Precomputed: {len(cache)} records")

    def _apply_trend_filter(self, snap, current_dt):
        """趋势过滤：基于min评分过滤弱势共振股票"""
        tf_cfg = self.config.get('trend_filter', {})
        if not tf_cfg.get('enabled', False):
            return snap

        filter_pct = tf_cfg.get('filter_pct', 0.20)
        n_before = len(snap)
        codes_in_snap = snap['code'].astype(str).str.zfill(6).unique()

        cache = self._trend_factors_cache
        if cache is None:
            return snap

        cal = self.full_calendar
        try:
            idx = cal.index(current_dt)
        except ValueError:
            for i, d in enumerate(cal):
                if d >= current_dt:
                    idx = i
                    break
            else:
                return snap

        if idx < 1:
            return snap

        prev_date = cal[idx - 1]

        try:
            factors = cache.loc[(slice(None), prev_date), :].copy()
        except KeyError:
            return snap

        factors = factors.reset_index()
        factors = factors[factors['code_digits'].isin(codes_in_snap)]

        if factors.empty:
            return snap

        valid = factors.dropna().copy()
        n_valid = len(valid)
        if n_valid < 20:
            return snap

        valid['price_score'] = 1.0 - valid['price_decline_factor'].rank(pct=True)
        valid['volume_score'] = valid['volume_rise_factor'].rank(pct=True)
        valid['ma20_score'] = valid['ma20_distance'].rank(pct=True)
        valid['weakness_score'] = valid[['price_score', 'volume_score', 'ma20_score']].min(axis=1)

        threshold = valid['weakness_score'].quantile(1 - filter_pct)
        to_filter = valid[valid['weakness_score'] >= threshold]
        filter_codes = set(to_filter['code_digits'].tolist())

        snap_codes = snap['code'].astype(str).str.zfill(6)
        result = snap[~snap_codes.isin(filter_codes)].copy()

        n_filtered = n_before - len(result)
        logger.info(f"[TrendFilter] {current_dt.date()}: "
                    f"candidates={n_before}, filtered={n_filtered}, remaining={len(result)}")
        return result

    # ================================================================
    #  盘前选股
    # ================================================================

    def _should_retrain_model(self) -> Tuple[bool, str]:
        """[重训间隔 2026-08-18 对齐 small] 判断本次调仓是否需要重训模型。

        规则 (与 small_alpha191 一致):
          - 模型未就绪 (无折定义/无模型) → 必须重训
          - retrain_interval <= 1 → 每次调仓都重训
          - _rebalances_since_train >= retrain_interval → 重训并重置
          - 否则 → 沿用最近模型 (特征集/门控 AUC/校准器全部沿用)
        """
        if self.retrain_interval <= 1:
            return True, 'interval<=1 每次重训'
        if not self._cv_fold_defs or not self.models:
            return True, '模型未就绪'
        if self._rebalances_since_train >= self.retrain_interval:
            return True, f'已达间隔{self.retrain_interval}'
        return False, f'未达间隔({self._rebalances_since_train}/{self.retrain_interval})'

    def before_market_open(self, context: StrategyContext):
        """调仓日执行：Purged K-Fold CV → 集成训练 → 校准 → AUC门控 → 选股"""
        if getattr(self, '_is_recomd', False):
            self._recomd_daily_select(context)
            return

        self.trade_day_seq += 1
        is_rebalance = ((self.trade_day_seq - 1) % self.rebalance_interval == 0)

        # IV仓位控制
        if self.iv_enabled and self.iv_controller is not None:
            today_scale = self.iv_controller.get_scaling_factor(context.current_dt)
            if not is_rebalance:
                if (self._iv_scale_applied is not None
                        and today_scale < self._iv_scale_applied - 0.005):
                    reduction = today_scale / self._iv_scale_applied
                    self._iv_reduction_pending = {}
                    portfolio = context.portfolio
                    tv = portfolio.total_value if portfolio.total_value > 0 else 1.0
                    for code, pos in portfolio.positions.items():
                        current_w = pos.market_value / tv
                        self._iv_reduction_pending[code] = current_w * reduction
                    logger.info(f"[IV] {context.current_dt.date()}: "
                                f"scale {self._iv_scale_applied:.3f}→{today_scale:.3f}")
                self._iv_scale_applied = today_scale

        if not is_rebalance:
            return

        current_dt = context.current_dt
        logger.info(f"[{current_dt.date()}] ========== Rebalance day ==========")

        # [重训间隔 2026-08-18 对齐 small] 每 N 次调仓重训一次, 期间沿用最近模型
        self._rebalances_since_train += 1
        retrain_now, retrain_reason = self._should_retrain_model()
        if not retrain_now:
            logger.info(f"[重训间隔] 沿用最近模型 ({retrain_reason}), 跳过训练直接选股")
        else:
            # 1. 构建训练数据
            train_res, val_res, sample_weights, raw_train_data, raw_val_data = \
                self._build_training_dataset(current_dt)

            if train_res is None:
                logger.warning("Insufficient training data, clearing positions")
                self.target_weights = {}
                self.target_stocks = []
                return

            n_samples = train_res.num_row()
            logger.info(f"Train samples: {n_samples}, Val samples: {val_res.num_row()}, "
                        f"Features: {len(self.feature_names)}")

            # 2. 模型集成训练: CV 折集成 (优先) / 种子集成 (降级)
            use_cv_ensemble = (self._ensemble_method == 'cv_fold'
                               and self._cv_fold_defs
                               and len(self._cv_fold_defs) >= 3
                               and raw_train_data is not None)

            if use_cv_ensemble:
                logger.info("[Ensemble] 使用 CV 折集成模式 (每折独立验证集 + 超参数多样性 + AUC加权)")
                cv_models = self._train_cv_ensemble(raw_train_data)
                if cv_models is None:
                    logger.warning("[Ensemble] CV折集成失败, 回退到种子集成")
                    self.models = self._train_ensemble(train_res, val_res, force_seed=True)
                    self.calibrator = self._calibrate_proba(val_res)
                    perf = self._calc_model_performance(train_res, val_res)
                    val_auc = perf.get('val', {}).get('auc', 0.0) if perf else 0.0
                else:
                    # CV 折集成成功
                    # 校准: 仅收集活跃模型验证集预测, 合并拟合
                    self.calibrator = self._calibrate_cv_ensemble(raw_train_data)
                    # 门控 AUC: [2026-08-20 D5] 加权平均口径 (与 _train_cv_ensemble 内 meta_auc 同源)
                    val_auc = getattr(self, '_cv_meta_auc', self._active_auc_metric())
                    perf = {'train': {}, 'val': {'auc': val_auc}}

                    # P2: OOT 外推期验证 (20% holdout, 独立于 CV 折)
                    if raw_val_data is not None and len(raw_val_data) > 100:
                        try:
                            X_oot = raw_val_data[self.feature_names].replace(
                                [np.inf, -np.inf], np.nan).fillna(0).values
                            y_oot = raw_val_data['label'].values.astype(int)
                            doot = xgb.DMatrix(X_oot, feature_names=self.feature_names)
                            oot_preds = self._predict_ensemble(doot)
                            oot_auc = float(roc_auc_score(y_oot, oot_preds))
                            logger.info(f"[OOT Holdout] {len(raw_val_data)} samples, AUC={oot_auc:.4f} "
                                        f"(vs CV meta_AUC={val_auc:.4f}, "
                                        f"gap={oot_auc - val_auc:+.4f})")
                        except Exception:
                            pass
            else:
                if self._ensemble_method == 'cv_fold' and not self._cv_fold_defs:
                    logger.info("[Ensemble] 无 CV 折定义 (特征选择未生成), 回退到种子集成")
                elif self._ensemble_method == 'cv_fold' and len(self._cv_fold_defs) < 3:
                    logger.info("[Ensemble] CV 折不足 (%d < 3), 回退到种子集成", len(self._cv_fold_defs))
                else:
                    logger.info("[Ensemble] 种子集成模式")
                self.models = self._train_ensemble(train_res, val_res)
                self.calibrator = self._calibrate_proba(val_res)
                perf = self._calc_model_performance(train_res, val_res)
                val_auc = perf.get('val', {}).get('auc', 0.0) if perf else 0.0

            # [CFI-诊断] 簇级重要性分布输出 (生产模型训练后, 纯增量, 零行为影响)
            # self.models 此时已就绪 (CV 折集成 _train_cv_ensemble 内部赋值 / 种子集成赋值)
            # 输出到回测结果目录下的 cfi_importance/ 单独文件夹
            if (self._cluster_diagnose and _CFI_AVAILABLE and self.feature_names
                    and self.models and raw_val_data is not None
                    and len(raw_val_data) > 0 and 'label' in raw_val_data.columns):
                try:
                    _cfi_cfg = self._cluster_cfg_namespace()
                    _diag_dir = os.path.join(self.output_dir, 'cfi_importance')
                    os.makedirs(_diag_dir, exist_ok=True)
                    _diag_end = getattr(self, '_current_rebalance_date', current_dt)
                    _fname = f'cfi_importance_{pd.Timestamp(_diag_end).strftime("%Y%m%d")}.csv'
                    cfi_diagnose(
                        self.models, raw_val_data[self.feature_names],
                        raw_val_data['label'].values.astype(int),
                        self.feature_names, _cfi_cfg,
                        output_path=os.path.join(_diag_dir, _fname),
                        metric_fn=_objective_metric_fn('binary'),
                        n_repeats=self._feat_sel_mda_n_repeats,
                    )
                except Exception as _diag_e:
                    logger.warning(f"[CFI-Diag] 诊断输出失败: {_diag_e}")
            elif self._cluster_diagnose and _CFI_AVAILABLE:
                logger.debug("[CFI-Diag] 模型/验证数据不可用, 跳过诊断")

            if val_auc is None or np.isnan(val_auc):
                val_auc = 0.0
            self.val_auc = val_auc

            if perf:
                t = perf.get('train', {})
                if t and 'logloss' in t and 'auc' in t:
                    logger.info(f"Train: LogLoss={t['logloss']:.4f}, AUC={t['auc']:.4f}")
                if 'val' in perf:
                    v = perf['val']
                    if v and 'auc' in v:
                        if 'logloss' in v:
                            logger.info(f"Val:   LogLoss={v['logloss']:.4f}, AUC={v['auc']:.4f}")
                        else:
                            logger.info(f"Val:   AUC={v['auc']:.4f} (CV折集成meta AUC)")

            self._rebalances_since_train = 0  # 训练成功重置 (失败路径 return 不重置)
        self._log_feature_importance(top_n=15)

        confidence_level, position_multiplier = self._auc_gate(val_auc)
        logger.info(f"AUC Gate: {confidence_level} (val_auc={val_auc:.4f}, multiplier={position_multiplier})")

        # AUC门控: empty模式直接空仓返回（对齐Alpha191）
        if self._auc_gate_enabled and position_multiplier == 0.0:
            logger.warning(f"[AUC Gate] val_auc={val_auc:.4f} < {self._auc_gate_min} → empty, "
                          f"clear all positions, skip selection")
            self.target_weights = {}
            self.target_stocks = []
            return

        # 5. 获取快照 + P1-4 全市场中性化
        prev = self._get_previous_snapshot_date(current_dt)
        if prev is None:
            self.target_weights = {}
            self.target_stocks = []
            return

        full_snap = self._load_snapshot(prev)
        if full_snap is None or full_snap.empty:
            self.target_weights = {}
            self.target_stocks = []
            return

        # 全市场中性化 → 再过滤候选池
        if self._full_market_neutralize:
            full_snap = self._neutralize_full_market(full_snap)

        candidates = self.stock_pool.get_candidates(current_dt, context)
        cand_digits = {c.replace('sh.', '').replace('sz.', '') for c in candidates}
        snap = full_snap[full_snap['code'].isin(cand_digits)]

        if snap.empty:
            self.target_weights = {}
            self.target_stocks = []
            return

        snap = self._apply_trend_filter(snap, current_dt)
        if snap.empty:
            self.target_weights = {}
            self.target_stocks = []
            return

        # 6. 特征工程 + 集成预测 + 校准
        X_pred = self._build_feature_matrix(snap, pre_neutralized=self._full_market_neutralize)
        dpred = xgb.DMatrix(X_pred, feature_names=self.feature_names)
        raw_preds = self._predict_ensemble(dpred)
        preds = self._apply_calibration(raw_preds)

        # 概率坍缩检测: 固定检查 top-300 的区分度
        # 300 是候选池上限, 与 max_positions(实际持仓数) 无关
        # 当 top-300 无区分力时模型无效, 即使 top-50 有区分力也可能只是噪声
        n_candidates = len(snap)
        if self._check_prob_collapse(preds, 300):
            self.target_weights = {}
            self.target_stocks = []
            return

        snap = snap.copy()
        snap['pred'] = preds
        snap['raw_pred'] = raw_preds  # 用于 Barra 优化中打破 calibrated 值的平局

        logger.info(f"Prediction stats: mean={preds.mean():.4f}, std={preds.std():.4f}, "
                    f"min={preds.min():.4f}, max={preds.max():.4f}")

        # 7. 选股（不因AUC门控改变选股数量，仅影响仓位权重，对齐Alpha191）
        n_select = max(1, int(np.ceil(n_candidates * self.top_quantile)))
        n_select = min(n_select, self.max_positions)
        selected = snap.sort_values('pred', ascending=False).head(n_select)

        # 8. 权重分配: Barra 约束优化 (优先) / softmax (降级)
        # alpha 用 raw_preds（精度无限无平局），calibrated preds 仅用于门控/日志
        preds_values = selected['pred'].values
        raw_preds_values = selected['raw_pred'].values if 'raw_pred' in selected.columns else None
        alpha_preds = raw_preds_values if raw_preds_values is not None else preds_values
        sel_codes = [str(c).zfill(6) for c in selected['code'].values]
        max_pct = self.max_position_pct

        use_barra = (_BARRA_AVAILABLE and self._barra_enabled
                     and len(selected) >= self._barra_min_positions)

        if use_barra:
            try:
                constraint_date = self._get_trading_day_offset(current_dt, -1)
                if constraint_date is None:
                    constraint_date = current_dt - timedelta(days=1)
                barra_expo = get_or_build_snapshot(constraint_date)
                if barra_expo is not None and len(barra_expo) > 100:
                    weights, barra_ok = solve_barra_weights(
                        alpha_preds, sel_codes, barra_expo,
                        max_weight=self._barra_max_weight,
                        constraint_date=constraint_date,
                    )
                    if barra_ok:
                        logger.info(f"[Barra] Constrained optimization applied "
                                    f"({int((weights > 0).sum())} positions)")
                    else:
                        weights, use_barra = None, False
                else:
                    use_barra = False
            except Exception as e:
                logger.warning(f"[Barra] Exception, falling back to softmax: {e}")
                use_barra = False

        if not use_barra:
            # 降级: softmax
            if len(preds_values) > 0:
                exp_preds = np.exp((preds_values - preds_values.max()) / self.pred_weight_temperature)
                weights = exp_preds / exp_preds.sum()
                weights = np.minimum(weights, max_pct)
                weights = weights / weights.sum()
            else:
                weights = np.array([])
            if _BARRA_AVAILABLE and self._barra_enabled:
                logger.info("[Barra] Fallback to softmax (insufficient data or solver failed)")

        # P1-8: 行业敞口约束 (在 softmax/Barra 之后应用, 作为后处理硬性约束)
        if len(weights) > 0 and self._ind_constraint_enabled:
            weights = self._apply_industry_constraint(selected, weights, full_snap)

        logger.info(f"Selection ({len(selected)} stocks, mode: {confidence_level}):")
        for i, (_, row) in enumerate(selected.head(10).iterrows()):
            code = str(row['code']).zfill(6)
            w = weights[i] if i < len(weights) else 0
            logger.info(f"  #{i+1} {code} | prob={row['pred']:.4f} | weight={w:.4f}")

        # 9. 构建目标权重
        self.target_weights = {}
        self.target_stocks = []
        for i, (_, row) in enumerate(selected.iterrows()):
            digits = str(row['code']).zfill(6)
            full_code = ('sh.' + digits if digits.startswith(('6', '9'))
                         else 'sz.' + digits)
            w = weights[i] if i < len(weights) else 0
            self.target_weights[full_code] = w
            self.target_stocks.append(full_code)

        # AUC门控: half模式仓位减半（对齐Alpha191 gate_mult）
        if self._auc_gate_enabled and position_multiplier < 1.0:
            self.target_weights = {k: v * position_multiplier
                                   for k, v in self.target_weights.items()}
            logger.info(f"[AUC Gate] {confidence_level} mode, weights scaled by {position_multiplier}")

        # IV仓位缩放
        if self.iv_enabled and self.iv_controller is not None:
            scale = self.iv_controller.get_scaling_factor(context.current_dt)
            self.target_weights = {k: v * scale for k, v in self.target_weights.items()}
            self._iv_scale_applied = scale
            self._iv_reduction_pending = {}
            logger.info(f"IV scale applied: {scale:.3f}")

        total_weight = sum(self.target_weights.values())
        logger.info(f"Rebalance target: {len(self.target_weights)} stocks, total weight: {total_weight:.4f}")

    # ================================================================
    #  行业敞口限制 (P1-8: 防止单一行业集中度风险)
    # ================================================================

    def _apply_industry_constraint(self, selected: pd.DataFrame, weights: np.ndarray,
                                    full_snap: pd.DataFrame) -> np.ndarray:
        """对权重施加行业敞口上限约束（业界标准 posterior clipping）。

        遵循机构级最佳实践:
        1. 从 full_snap 获取每只股票的行业分类（来自 snapshot 的 'industry' 列）
        2. 按行业聚合权重，检测超额
        3. 对超额行业按比例缩回 (pro rata clipping)
        4. 回收的权重按比例重新分配给未触及上限的行业
        5. 迭代直到所有行业 <= 上限，或达到最大迭代次数

        与 Barra 约束的区别:
        - Barra 在优化层面控制因子暴露 (事前)
        - 行业约束在权重分配后做硬性裁剪 (事后, 简单有效)

        Args:
            selected: 选股结果 DataFrame (必须含 'code' 列)
            weights: 原始权重数组 (已归一化)
            full_snap: 全市场快照 (含 'code' 和 'industry' 列)

        Returns:
            adjusted_weights: 调整后的权重数组
        """
        if not self._ind_constraint_enabled:
            return weights

        # 检查行业数据可用性
        if 'industry' not in full_snap.columns:
            logger.warning("[IndConstraint] 快照无 'industry' 列, 无法应用行业约束, 将使用 fallback 行业")
            # 给每只股票分配默认行业
            codes = [str(c).zfill(6) for c in selected['code'].values]
            industries = np.array([self._ind_fallback] * len(codes))
        else:
            # 通过 code 列匹配行业
            codes = [str(c).zfill(6) for c in selected['code'].values]
            code_to_ind = dict(zip(
                full_snap['code'].astype(str).str.zfill(6),
                full_snap['industry'].fillna(self._ind_fallback).astype(str)
            ))
            industries = np.array([code_to_ind.get(c, self._ind_fallback) for c in codes])

        # 计算各行业总权重
        unique_inds = sorted(set(industries))
        ind_weights = {}
        for ind in unique_inds:
            ind_weights[ind] = weights[industries == ind].sum()

        # 检查是否有行业超额
        max_w = self._ind_max_weight
        excess_inds = {ind: w - max_w for ind, w in ind_weights.items() if w > max_w}

        if not excess_inds:
            logger.debug(f"[IndConstraint] ✓ 所有行业权重均 ≤ {max_w:.1%}, "
                         f"无需调整 ({len(unique_inds)} industries)")
            return weights

        # 迭代裁剪 (最多3轮, 业界标准)
        adjusted_weights = weights.copy()
        max_iter = 3
        for iteration in range(max_iter):
            ind_weights = {}
            for ind in unique_inds:
                ind_weights[ind] = adjusted_weights[industries == ind].sum()

            excess_inds = {ind: w - max_w for ind, w in ind_weights.items() if w > max_w}
            if not excess_inds:
                break

            # 对超额行业做比例缩放 (pro rata)
            for ind, excess in excess_inds.items():
                ind_mask = industries == ind
                current_ind_w = ind_weights[ind]
                # 缩放因子: 将行业权重从 current_ind_w 缩到 max_w
                scale = max_w / current_ind_w
                adjusted_weights[ind_mask] *= scale

            # 回收的权重重新分配
            total_after_clip = adjusted_weights.sum()
            if total_after_clip > 0 and abs(total_after_clip - 1.0) > 1e-6:
                # 将回收的权重按比例分配给未触及上限的行业
                unclipped_inds = [ind for ind in unique_inds
                                  if ind_weights.get(ind, 0) <= max_w]
                if unclipped_inds:
                    unclipped_mask = np.isin(industries, unclipped_inds)
                    if unclipped_mask.sum() > 0:
                        excess_to_redistribute = 1.0 - total_after_clip
                        current_unclipped_w = adjusted_weights[unclipped_mask].sum()
                        if current_unclipped_w > 0:
                            ratio = excess_to_redistribute / current_unclipped_w
                            adjusted_weights[unclipped_mask] *= (1.0 + ratio)

            # 最终归一化
            w_sum = adjusted_weights.sum()
            if w_sum > 0:
                adjusted_weights /= w_sum

        # 日志输出
        adjusted_ind_weights = {}
        for ind in unique_inds:
            adjusted_ind_weights[ind] = adjusted_weights[industries == ind].sum()
        excess_after = {ind: f"{w:.3f}" for ind, w in adjusted_ind_weights.items() if w > max_w}

        if excess_after:
            logger.warning(
                f"[IndConstraint] ⚠ Iteration {iteration+1} 后仍有行业超额: "
                f"{excess_after}, 最大约束={max_w:.1%}"
            )
        else:
            n_adjusted = sum(1 for ind in unique_inds
                           if abs(ind_weights.get(ind, 0) - adjusted_ind_weights.get(ind, 0)) > 0.001)
            if n_adjusted > 0:
                logger.info(
                    f"[IndConstraint] ✓ 行业约束: {n_adjusted} 个行业被裁剪 "
                    f"(max={max_w:.1%}, {len(unique_inds)} 行业总计, "
                    f"{len(excess_inds)} 个超额)"
                )

        return adjusted_weights

    # ================================================================
    #  盘中交易
    # ================================================================

    def handle_bar(self, context: StrategyContext):
        """盘中交易执行"""
        if getattr(self, '_is_recomd', False):
            return

        broker = context.broker
        portfolio = context.portfolio
        date = context.current_dt

        is_rebalance = ((self.trade_day_seq - 1) % self.rebalance_interval == 0)

        # IV减仓执行
        if not is_rebalance and self._iv_reduction_pending:
            for code, weight in list(self._iv_reduction_pending.items()):
                broker.order_target_percent(code, weight)
            logger.info(f"[{date.date()}] IV reduction: {len(self._iv_reduction_pending)} positions")
            self._iv_reduction_pending = {}

        if is_rebalance:
            for code in list(portfolio.positions.keys()):
                if code not in self.target_weights:
                    broker.order_target_percent(code, 0.0)

            held = [c for c in portfolio.positions if c in self.target_weights]
            available = self.max_positions - len(held)

            sorted_targets = sorted(self.target_weights.items(), key=lambda x: x[1], reverse=True)
            for code, weight in sorted_targets:
                if code in portfolio.positions:
                    broker.order_target_percent(code, weight)
                elif available > 0:
                    broker.order_target_percent(code, weight)
                    available -= 1

        total_value = portfolio.total_value
        logger.info(f"[{date.date()}] {'Rebalance' if is_rebalance else 'Hold'}: "
                    f"total={total_value:,.0f}, cash={portfolio.cash:,.0f}, "
                    f"positions={len(portfolio.positions)}")

    # ================================================================
    #  recommend接口
    # ================================================================

    def recommend(self, context: StrategyContext,
                  max_n: int = None) -> Tuple[List[str], Dict[str, float]]:
        """返回选股结果供外部调用"""
        rng = np.random.RandomState(42)  # P3-1: 局部RNG

        if max_n is None:
            max_n = self.top_n

        current_dt = context.current_dt
        logger.info(f"[WyckoffV10 recommend] {current_dt.date()} target: {max_n} stocks")

        # 1. 构建训练数据
        train_res, val_res, sample_weights, raw_train_data, raw_val_data = self._build_training_dataset(current_dt)

        if train_res is None:
            return ([], {})

        # 2. 集成训练 + 校准
        # [P2-4 FIX] 复用 before_market_open 的 CV 折/种子集成选择逻辑,
        # 确保 recommend() 与回测使用相同的模型训练路径
        use_cv_ensemble = (self._ensemble_method == 'cv_fold'
                           and self._cv_fold_defs
                           and len(self._cv_fold_defs) >= 3
                           and raw_train_data is not None)

        if use_cv_ensemble:
            logger.info("[recommend] 使用 CV 折集成模式")
            cv_models = self._train_cv_ensemble(raw_train_data)
            if cv_models is None:
                logger.warning("[recommend] CV折集成失败, 回退到种子集成")
                self.models = self._train_ensemble(train_res, val_res)
                self.calibrator = self._calibrate_proba(val_res)
            else:
                self.calibrator = self._calibrate_cv_ensemble(raw_train_data)
                val_auc = getattr(self, '_cv_meta_auc', self._active_auc_metric())
                self.val_auc = val_auc
        else:
            self.models = self._train_ensemble(train_res, val_res)
            self.calibrator = self._calibrate_proba(val_res)

        # 3. AUC门控 (CV折模式已在上面计算val_auc, 种子模式需重新计算)
        if not use_cv_ensemble or self.models is None or cv_models is None:
            perf = self._calc_model_performance(train_res, val_res)
            val_auc = perf.get('val', {}).get('auc', 0.0) if perf else 0.0
        if val_auc is None or np.isnan(val_auc):
            val_auc = 0.0
        self.val_auc = val_auc

        confidence_level, position_multiplier = self._auc_gate(val_auc)
        logger.info(f"[recommend] AUC Gate: {confidence_level} (val_auc={val_auc:.4f}, multiplier={position_multiplier})")

        # AUC门控: empty模式直接返回空（对齐Alpha191）
        if self._auc_gate_enabled and position_multiplier == 0.0:
            logger.warning(f"[recommend] val_auc={val_auc:.4f} < {self._auc_gate_min} → empty, return []")
            return ([], {})

        # 4. 获取快照 + 全市场中性化
        prev = self._get_previous_snapshot_date(current_dt)
        if prev is None:
            return ([], {})

        full_snap = self._load_snapshot(prev)
        if full_snap is None or full_snap.empty:
            return ([], {})

        if self._full_market_neutralize:
            full_snap = self._neutralize_full_market(full_snap)

        candidates = self.stock_pool.get_candidates(current_dt, context)
        cand_digits = {c.replace('sh.', '').replace('sz.', '') for c in candidates}
        snap = full_snap[full_snap['code'].isin(cand_digits)]

        if snap.empty:
            return ([], {})

        # 趋势过滤（对齐回测流程, 确保recmd与backtest选股一致）
        snap = self._apply_trend_filter(snap, current_dt)
        if snap.empty:
            return ([], {})

        # 5. 特征工程 + 集成预测 + 校准
        X_pred = self._build_feature_matrix(snap, pre_neutralized=self._full_market_neutralize)
        dpred = xgb.DMatrix(X_pred, feature_names=self.feature_names)
        raw_preds = self._predict_ensemble(dpred)
        preds = self._apply_calibration(raw_preds)

        # 概率坍缩检测：以 top-300 为基准
        if self._check_prob_collapse(preds, 300):
            logger.warning(f"[recommend] Prob collapse detected, return empty")
            return ([], {})

        snap = snap.copy()
        snap['pred'] = preds
        selected = snap.sort_values('pred', ascending=False).head(max_n)

        stocks = []
        rank_pcts = {}
        for _, row in selected.iterrows():
            digits = str(row['code']).zfill(6)
            full_code = ('sh.' + digits if digits.startswith(('6', '9'))
                         else 'sz.' + digits)
            stocks.append(full_code)
            rank_pcts[full_code] = float(row['pred'])

        logger.info(f"[WyckoffV10 recommend] {current_dt.date()}: "
                    f"{len(stocks)} stocks, val_auc={self.val_auc:.4f}")
        return (stocks, rank_pcts)

    # ================================================================
    #  recomd 每日选股
    # ================================================================

    def _recomd_daily_select(self, context: StrategyContext):
        """recomd模式每日选股：集成训练 + 校准 + AUC门控 + 快照保存"""
        self.trade_day_seq += 1
        is_rebalance = ((self.trade_day_seq - 1) % self.rebalance_interval == 0)
        if not is_rebalance:
            return

        current_dt = context.current_dt
        logger.info(f"[V10 recomd] {current_dt.date()} ========== Daily selection ==========")

        # 1. 构建训练数据 (与 before_market_open 完全一致)
        train_res, val_res, sample_weights, train_data, val_data = self._build_training_dataset(current_dt)
        if train_res is None:
            logger.warning("[V10 recomd] Insufficient training data, skipping")
            return

        logger.info(f"[V10 recomd] Train: {train_res.num_row()}, "
                    f"Val: {val_res.num_row()}, Features: {len(self.feature_names)}")

        # 2. 模型训练 + 校准 (与 before_market_open 完全一致: CV折集成优先)
        t0 = time.time()
        use_cv_ensemble = (self._ensemble_method == 'cv_fold'
                           and self._cv_fold_defs
                           and len(self._cv_fold_defs) >= 3
                           and train_data is not None)

        if use_cv_ensemble:
            cv_models = self._train_cv_ensemble(train_data)
            if cv_models is not None:
                self.calibrator = self._calibrate_cv_ensemble(train_data)
                # [2026-08-20 D5] 加权平均口径 (与 _train_cv_ensemble 内 meta_auc 同源)
                val_auc = getattr(self, '_cv_meta_auc', self._active_auc_metric())
            else:
                logger.warning("[V10 recomd] CV折集成失败, 回退到种子集成")
                self.models = self._train_ensemble(train_res, val_res)
                self.calibrator = self._calibrate_proba(val_res)
                val_auc = float(self._calc_model_performance(train_res, val_res).get('val', {}).get('auc', 0.0))
        else:
            self.models = self._train_ensemble(train_res, val_res)
            self.calibrator = self._calibrate_proba(val_res)
            val_auc = float(self._calc_model_performance(train_res, val_res).get('val', {}).get('auc', 0.0))

        logger.info(f"[V10 recomd] Training done in {time.time()-t0:.1f}s")

        # 3. AUC门控 (val_auc 已在训练块中计算)
        if val_auc is None or np.isnan(val_auc):
            val_auc = 0.0
        self.val_auc = val_auc
        logger.info(f"[V10 recomd] Val AUC={val_auc:.4f}")

        self._log_feature_importance(top_n=15)

        confidence_level, position_multiplier = self._auc_gate(val_auc)
        logger.info(f"[V10 recomd] AUC Gate: {confidence_level} (val_auc={val_auc:.4f}, multiplier={position_multiplier})")

        # AUC门控: empty模式直接返回（对齐Alpha191）
        if self._auc_gate_enabled and position_multiplier == 0.0:
            logger.warning(f"[V10 recomd] val_auc={val_auc:.4f} < {self._auc_gate_min} → empty, skip selection")
            self._save_recomd_snapshot(current_dt, pd.DataFrame(columns=['code', 'pred', 'rank_pct']))
            return

        # 选股数量不因AUC门控改变（对齐Alpha191, 仅影响仓位权重）
        n_select = self.max_positions

        # 4. 获取快照 + 全市场中性化
        prev = self._get_previous_snapshot_date(current_dt)
        if prev is None:
            logger.warning("[V10 recomd] No snapshot, skipping")
            return

        full_snap = self._load_snapshot(prev)
        if full_snap is None or full_snap.empty:
            return

        if self._full_market_neutralize:
            full_snap = self._neutralize_full_market(full_snap)

        if self.stock_pool is not None:
            candidates = self.stock_pool.get_candidates(current_dt, context)
            cand_digits = {c.replace('sh.', '').replace('sz.', '') for c in candidates}
            snap = full_snap[full_snap['code'].isin(cand_digits)]
        else:
            snap = full_snap

        if snap.empty:
            return

        snap = self._apply_trend_filter(snap, current_dt)
        if snap.empty:
            return

        # 5. 预测
        X_pred = self._build_feature_matrix(snap, pre_neutralized=self._full_market_neutralize)
        dpred = xgb.DMatrix(X_pred, feature_names=self.feature_names)
        raw_preds = self._predict_ensemble(dpred)
        preds = self._apply_calibration(raw_preds)

        # 概率坍缩检测: 检查 top-300 的选股区分度
        if self._check_prob_collapse(preds, 300):
            self._save_recomd_snapshot(current_dt, pd.DataFrame(columns=['code', 'pred', 'rank_pct']))
            logger.warning(f"[V10 recomd] {current_dt.date()}: prob collapse, saved empty snapshot")
            return

        snap = snap.copy()
        snap['pred'] = preds

        logger.info(f"[V10 recomd] Pred: mean={preds.mean():.4f}, std={preds.std():.4f}")

        # 6. 排序选股
        all_rank = snap['pred'].rank(ascending=False, method='min')
        total = len(all_rank)
        snap['rank_pct'] = ((total - all_rank + 1) / total) * 100.0
        selected = snap.sort_values('pred', ascending=False).head(min(n_select, len(snap)))

        for i, (_, row) in enumerate(selected.head(10).iterrows()):
            code = str(row['code']).zfill(6)
            logger.info(f"  #{i+1} {code} | prob={row['pred']:.4f} | rank_pct={row['rank_pct']:.1f}")

        # 7. 保存快照
        self._save_recomd_snapshot(current_dt, selected[['code', 'pred', 'rank_pct']])
        logger.info(f"[V10 recomd] {current_dt.date()}: {len(selected)} stocks, val_auc={self.val_auc:.4f}")

    def _save_recomd_snapshot(self, date: datetime, result_df: pd.DataFrame):
        """保存选股结果到parquet"""
        fname = f'recomd_{date.strftime("%Y%m%d")}.parquet'
        path = os.path.join(self._recomd_snapshot_dir, fname)
        out = result_df.copy()
        out['code'] = out['code'].astype(str).str.zfill(6)
        out['pred'] = out['pred'].astype(float)
        out['rank_pct'] = out['rank_pct'].astype(float)
        out.to_parquet(path, index=False)
        logger.debug(f"[V10 recomd] Saved {len(out)} rows → {path}")
