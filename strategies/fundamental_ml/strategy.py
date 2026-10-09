# strategies/fundamental_ml/strategy.py

"""
基本面ML策略 - v6优化版

【v6 优化记录 - 2026-05-22】
基于v5回测深度诊断的6项关键优化：

1. 🔴 PSI过滤改为仅对rank因子生效
   - 问题：价值因子(PE/PB)在牛熊市分布差异巨大，PSI必然高，但不代表失效
   - 修复：原始值因子(pe_neu, pb_neu等)跳过PSI检查；rank因子已做截面排序，PSI检查仍适用
   - pb_pct_neu是MDI第一名(gain=208.90)，证明原始值因子极有价值

2. 🔴 RankIC阈值从0.01提升到0.03
   - 问题：噪音因子与有效因子混在一起(IC=0.009 vs IC=0.13)
   - 修复：min_ic从0.01提升到0.03，减少噪音因子进入MDI/MDA环节

3. 🔴 置信度门控改为三档保守持仓
   - 问题：IC<0.05时清仓，45%时间空仓，A股长期向上导致巨大机会成本
   - 修复：IC>=0.10正常、0.05~0.10保守(0.5x)、<0.05防御(0.25x，最低10只)
   - 长期持仓好于空仓，防御持仓保留市场参与度

4. 🔴 预测值加权持仓
   - 问题：预测范围0.3882~0.6097，top1和top60差0.01，等权无区分度
   - 修复：softmax加权(temperature=0.1)，高预测值给更高权重
   - 单只最大权重5%限制

5. 🔴 缩短调仓周期至20天
   - 问题：rebalance_interval=40与pred_days=40完全匹配，信号衰减期未被捕捉
   - 修复：rebalance_interval从40降至20，半预测窗口调仓
   - 基本面信号半衰期约20-40天，更高频率能更好捕捉信号衰减

6. 🟡 early_stopping_rounds从30降至20
   - 减少在小IC轮次中的过拟合训练，best_iteration更快收敛

【v5 优化记录 - 2026-05-20】
1. 🔴 因子深加工：从已有快照数据计算8个交互因子
   - Quality-Value: 高ROE + 低PB = 便宜的好公司
   - Growth Consistency: 营收和利润同时增长
   - Earnings Quality: ROE × 毛利率
   - Value-Momentum: 低PE分位 + 价格动量
   - Growth Acceleration: 季度环比动量
   - Cash Flow Quality: 现金流 × ROE
   - Value Composite: 低PE + 低PB
   - GARP: 高ROE + 低PE（合理价格成长）

2. 🔴 因子筛选机制：PSI+IC初筛 + MDI/MDA复筛
   - 低方差过滤 → 缺失率过滤+高相关去重 → PSI过滤 → RankIC过滤
   - MDI(XGBoost gain) → MDA(permutation importance)
   - 跳过PCA一致性检验（基本面特征数少）

3. 🔴 模型置信度门控：根据验证集Spearman IC动态调整持仓
   - val_ic >= 0.10：正常交易
   - 0.05 <= val_ic < 0.10：持仓数减半
   - val_ic < 0.05：清仓/保持空仓

4. 🟡 配置参数优化
   - max_positions: 300 → 60
   - top_quantile: 0.10 → 0.02
   - pred_days: 60 → 40
   - rebalance_interval: 60 → 40
   - train_window_days: 120 → 200

【v4 关键优化】
1. 截面排名变换目标：y = fp/bp - 1 → rank(pct=True)
2. 时序验证集 + 真正的early stopping
3. XGBoost超参数优化
4. 评估指标升级：新增Spearman IC
5. 修复持仓数超过max_positions
"""

import os, logging, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict

import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
from scipy.special import softmax

from gpsef.core import IStrategy, StrategyContext
from gpsef.utils.neutralizer import FactorNeutralizer

logger = logging.getLogger(__name__)


# ============================================================================
# XGBoost Booster sklearn兼容包装器（回归任务）
# ============================================================================

class XGBRegressorWrapper(BaseEstimator, RegressorMixin):
    """
    sklearn-compatible wrapper for xgb.Booster (regression task).

    解决 sklearn.inspection.permutation_importance 与 xgb.Booster 的接口不兼容问题。
    包装器将 numpy 输入自动转为 DMatrix，使 Booster 可直接用于 sklearn 的置换重要性计算。

    设计原则：
    1. 零训练开销：直接复用已训练好的 Booster，不重新训练
    2. 精确等价：predict 输出与 Booster.predict(DMatrix) 完全一致
    3. 符合 sklearn 约定：predict 返回预测值
    """

    def __init__(self, booster):
        """
        Args:
            booster: xgb.Booster，已训练好的XGBoost模型
        """
        self._booster = booster
        self.is_fitted_ = True
        # 保存训练时的特征名；permutation_importance 内部打乱列后传入 numpy array（无列名），
        # 必须在 predict 时显式传回特征名，否则 Booster 因特征名不匹配而报 ValueError
        self._feature_names = getattr(booster, 'feature_names', None)

    def predict(self, X):
        """
        预测回归值（sklearn API 约定）。

        Args:
            X: numpy ndarray 或 pandas DataFrame

        Returns:
            numpy array，回归预测值
        """
        dmatrix = xgb.DMatrix(X, feature_names=self._feature_names)
        return self._booster.predict(dmatrix)

    def fit(self, X, y, **kwargs):
        """
        占位方法。Booster 已训练，无需再次拟合。
        sklearn 某些内部流程会调用 fit，此处直接跳过。
        """
        return self


# ============================================================================
# 序贯采样权重计算（与 alpha191/ensemble.py 中 compute_sample_weights 一致）
# ============================================================================

def compute_sample_weights(train_data, calendar, max_hold_days):
    """
    为每个训练样本计算基于平均唯一性的样本权重（高效版）。

    算法来源: De Prado《Advances in Financial Machine Learning》第7章

    【v2 性能优化】
    原实现对每个样本逐一构建指示矩阵行，O(N*T) 时空复杂度。
    优化核心：同一快照日的所有样本共享相同的 [事件日, 退出日] 窗口，
    因此其平均唯一性完全相同。只需按快照日分组，O(U*T) 复杂度。

    参数:
        train_data: DataFrame，必须包含 'snapshot_date' 列（快照日期/事件日）。
        calendar:   交易日列表（datetime 类型），已排序。
        max_hold_days: 标签前瞻天数（垂直屏障宽度），基本面策略中 = pred_days。

    返回:
        np.array，长度与 train_data 行数相同，表示每个样本的训练权重。
    """
    if train_data.empty:
        return np.ones(len(train_data))

    N = len(train_data)

    # 1. 按快照日分组，统计每个快照日的样本数
    snap_counts = train_data.groupby(train_data['snapshot_date']).size()
    unique_snaps = snap_counts.index

    # 将 calendar 转为 DatetimeIndex，用 searchsorted 快速定位
    cal_ts = pd.DatetimeIndex(sorted(calendar))
    T = len(cal_ts)

    # 2. 对每个唯一快照日，计算其在日历中的窗口 [start_idx, end_idx]
    snap_windows = {}
    for snap_date in unique_snaps:
        snap_ts = pd.Timestamp(snap_date)
        start_idx = cal_ts.searchsorted(snap_ts)
        if start_idx >= T:
            continue  # 快照日在日历范围外，跳过
        end_idx = min(start_idx + max_hold_days, T - 1)
        snap_windows[snap_date] = (start_idx, end_idx)

    # 3. 计算每天的并发样本数
    concurrency = np.zeros(T, dtype=np.float64)
    for snap_date, (s, e) in snap_windows.items():
        n_samples = snap_counts[snap_date]
        concurrency[s:e + 1] += n_samples

    # 避免除零
    concurrency[concurrency == 0] = 1.0

    # 4. 对每个唯一快照日，计算该组样本的平均唯一性
    snap_avg_u = {}
    for snap_date, (s, e) in snap_windows.items():
        if s <= e:
            snap_avg_u[snap_date] = np.sum(1.0 / concurrency[s:e + 1]) / T
        else:
            snap_avg_u[snap_date] = 1.0 / T

    # 5. 将权重分配到每个样本
    weights = np.array([snap_avg_u.get(sd, 1.0) for sd in train_data['snapshot_date']])

    # 6. 归一化到均值为1
    w_mean = weights.mean()
    if w_mean > 0:
        weights = weights / w_mean

    return weights


class FundamentalMLStrategy(IStrategy):

    def initialize(self, context: StrategyContext):
        config = context.config
        self.config = config
        self.data_provider = context.data_provider
        self.stock_pool = context.stock_pool

        self.ref_date = datetime.strptime(config['data']['reference_date'], '%Y%m%d')

        # 【v6优化】调仓周期从40降至20
        self.rebalance_interval = config.get('rebalance_interval', 20)
        self.train_window = config.get('train_window_days', 200)  # 交易日数
        self.pred_days = config.get('pred_days', 40)  # 交易日数
        logger.info(f"[初始化] rebalance_interval={self.rebalance_interval}, "
                    f"train_window={self.train_window}, pred_days={self.pred_days}")
        self.top_n = config.get('top_n', 60)
        self.max_positions = config['execution'].get('max_positions', 60)
        self.max_position_pct = config['execution'].get('max_position_pct', 0.05)

        # 【v6优化】置信度门控三档阈值
        self.val_ic_trade_threshold = config.get('val_ic_trade_threshold', 0.10)
        self.val_ic_reduce_threshold = config.get('val_ic_reduce_threshold', 0.05)
        self.val_ic_defend_threshold = config.get('val_ic_defend_threshold', 0.05)  # 【v6新增】防御持仓阈值
        logger.info(f"[初始化] 置信度门控: trade={self.val_ic_trade_threshold}, "
                    f"reduce={self.val_ic_reduce_threshold}, defend={self.val_ic_defend_threshold}")

        # 【v6优化】RankIC阈值从0.01提升到0.03
        self._feat_sel_nan_threshold = config.get('feat_sel_nan_threshold', 0.5)
        self._feat_sel_max_corr = config.get('feat_sel_max_corr', 0.95)
        self._feat_sel_psi_threshold = config.get('feat_sel_psi_threshold', 0.25)
        self._feat_sel_min_ic = config.get('feat_sel_min_ic', 0.03)  # 【v6优化】0.01 → 0.03
        self._feat_sel_mdi_n_estimators = config.get('feat_sel_mdi_n_estimators', 500)
        self._feat_sel_mda_n_repeats = config.get('feat_sel_mda_n_repeats', 10)

        # 【v6新增】预测值加权温度参数
        self.pred_weight_temperature = config.get('pred_weight_temperature', 0.1)

        self.snapshot_dir = os.path.join('data', 'fundamental_ml_snapshots')

        self.calendar = context.calendar
        self.full_calendar = context.full_calendar

        self.model: Optional[xgb.Booster] = None
        self.feature_names: List[str] = []
        self.val_ic: float = 0.0

        self.target_weights = {}
        self.trade_day_seq = 0

        # 【v3】行业哑变量列名缓存
        self._industry_dummy_cols: List[str] = []
        # 【v3】缺失指示特征列名缓存
        self._missing_indicator_cols: List[str] = []

        # 【v5新增】筛选后的因子列表（训练时确定，预测时使用）
        self._selected_base_features: List[str] = []

        # 【v6新增】early_stopping_rounds从30降至20
        self.early_stopping_rounds = config.get('xgb_early_stopping_rounds', 20)

        self.xgb_params = {
            'max_depth': config.get('xgb_max_depth', 3),
            'eta': config.get('xgb_learning_rate', 0.05),
            'subsample': config.get('xgb_subsample', 0.8),
            'colsample_bytree': config.get('xgb_colsample_bytree', 0.6),
            'min_child_weight': config.get('xgb_min_child_weight', 50),
            'alpha': config.get('xgb_reg_alpha', 0.1),
            'lambda': config.get('xgb_reg_lambda', 1.0),
            'seed': config.get('xgb_random_state', 42),
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'nthread': 1,
        }

        self._preload_daily_data(context)

    # --------------- 数据加载与工具函数 ---------------

    def _preload_daily_data(self, context: StrategyContext):
        logger.info("正在预加载全市场日线数据（固定基准日，使用扩展日历）...")
        dp = self.data_provider
        start = context.full_calendar[0]
        end = context.full_calendar[-1]
        codes = dp.get_stock_list()
        frames = []
        for code in codes:
            df = dp.get_daily_data(code, start, end, reference_date=self.ref_date)
            if df is None or df.empty:
                continue
            df = df[['trade_date', 'close', 'amount', 'turn']].copy()
            df['ts_code'] = code
            frames.append(df)

        if not frames:
            raise RuntimeError("未能加载任何有效日线数据")

        full_df = pd.concat(frames, ignore_index=True)
        full_df['trade_date'] = pd.to_datetime(full_df['trade_date'])
        full_df['close'] = pd.to_numeric(full_df['close'], errors='coerce')
        full_df = full_df.dropna(subset=['close'])
        full_df['code_digits'] = full_df['ts_code'].str.replace('sh.', '').str.replace('sz.', '')
        full_df.sort_values(['ts_code', 'trade_date'], inplace=True)
        full_df.reset_index(drop=True, inplace=True)
        self.full_df_daily = full_df
        logger.info(f"全市场日线预加载完成，共 {len(full_df)} 条有效记录")

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
            if d < current_dt:
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
        sub = self.full_df_daily[
            (self.full_df_daily['code_digits'].isin(codes)) &
            (self.full_df_daily['trade_date'] == date)
            ]
        return dict(zip(sub['code_digits'], sub['close']))

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

    # ====================================================================
    # 【v5新增】因子深加工：计算交互因子
    # ====================================================================

    def _add_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        从已有截面排名计算交互因子（经济逻辑驱动的非线性特征）。

        在neutralizing之前调用，输入数据已包含_rank列。

        返回: 添加了交互因子列的DataFrame
        """
        if data is None or data.empty:
            return data

        data = data.copy()
        interactions = {}

        # 1. Quality-Value: 高ROE + 低PB = 便宜的好公司
        if 'roe_rank' in data.columns and 'pb_rank' in data.columns:
            interactions['quality_value'] = data['roe_rank'] * (1 - data['pb_rank'])

        # 2. Growth Consistency: 营收和利润同时增长
        if 'rev_growth_rank' in data.columns and 'profit_growth_rank' in data.columns:
            interactions['growth_consistency'] = data['rev_growth_rank'] * data['profit_growth_rank']

        # 3. Earnings Quality: ROE × 毛利率
        if 'roe_rank' in data.columns and 'gross_margin_rank' in data.columns:
            interactions['earnings_quality'] = data['roe_rank'] * data['gross_margin_rank']

        # 4. Value-Momentum: 低PE分位 + 价格动量
        if 'pe_pct' in data.columns and 'close_pct' in data.columns:
            interactions['value_momentum'] = (1 - data['pe_pct']) * data['close_pct']

        # 5. Growth Acceleration: 季度环比动量
        if 'rev_qoq_rank' in data.columns and 'profit_qoq_rank' in data.columns:
            interactions['growth_accel'] = data['rev_qoq_rank'] * data['profit_qoq_rank']

        # 6. Cash Flow Quality: 现金流 × ROE
        if 'ocf_per_share_rank' in data.columns and 'roe_rank' in data.columns:
            interactions['cf_quality'] = data['ocf_per_share_rank'] * data['roe_rank']

        # 7. Value Composite: 低PE + 低PB
        if 'pe_rank' in data.columns and 'pb_rank' in data.columns:
            interactions['value_composite'] = (1 - data['pe_rank']) * (1 - data['pb_rank'])

        # 8. GARP: 高ROE + 低PE（合理价格成长）
        if 'roe_rank' in data.columns and 'pe_rank' in data.columns:
            interactions['garp'] = data['roe_rank'] * (1 - data['pe_rank'])

        for name, series in interactions.items():
            data[name] = series

        if interactions:
            logger.debug(f"交互因子计算完成: {list(interactions.keys())}")

        return data

    # ====================================================================
    # 【v5新增】因子筛选机制（v6优化PSI过滤逻辑）
    # ====================================================================

    def _calc_psi(self, feat_tr: pd.Series, feat_va: pd.Series, fill_value=None) -> float:
        """
        计算PSI（Population Stability Index）。

        与ensemble.py的calc_psi相同逻辑。

        Args:
            feat_tr: 训练集特征
            feat_va: 验证集特征
            fill_value: 缺失值填充值

        Returns:
            PSI值
        """
        if fill_value is not None:
            feat_tr = feat_tr.fillna(fill_value)
            feat_va = feat_va.fillna(fill_value)
        else:
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
        """
        特征预处理：缺失率过滤 + 高相关去重。

        与ensemble.py的sanitize_features逻辑一致。

        Args:
            df: 特征DataFrame
            nan_threshold: 缺失率阈值，超过该值则剔除
            max_corr: 相关系数阈值，超过该值则去重

        Returns:
            筛选后的特征名列表
        """
        sorted_cols = sorted(df.columns.tolist())
        df = df[sorted_cols].copy()

        # 缺失率过滤
        nan_rate = df.isna().mean()
        keep_cols = nan_rate[nan_rate < nan_threshold].index.tolist()
        if not keep_cols:
            return []
        df = df[keep_cols]

        n_feat = len(keep_cols)
        if n_feat <= 1:
            return keep_cols

        drop_flags = [False] * n_feat
        batch_size = 500
        for start in range(0, n_feat, batch_size):
            end = min(start + batch_size, n_feat)
            batch_cols = keep_cols[start:end]
            for i, col_i in enumerate(batch_cols):
                global_idx = start + i
                if drop_flags[global_idx]:
                    continue
                for j in range(global_idx + 1, n_feat):
                    if drop_flags[j]:
                        continue
                    col_j = keep_cols[j]
                    corr = df[col_i].corr(df[col_j])
                    if pd.notna(corr) and abs(corr) > max_corr:
                        drop_flags[j] = True

        final_cols = [col for col, drop in zip(keep_cols, drop_flags) if not drop]
        final_cols = sorted(final_cols)
        logger.info(f"特征预处理：原始特征数 {len(sorted_cols)}，剔除后保留 {len(final_cols)} 个")
        return final_cols

    def _select_features(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
                         feats: List[str], target_col: str = 'label') -> List[str]:
        """
        因子筛选：PSI+IC初筛 + MDI/MDA复筛。

        【v6优化】PSI过滤改为仅对rank因子生效：
        - 原始值因子(pe_neu, pb_neu等)天然分布不稳定，PSI高不代表失效
        - rank因子已做截面排序，分布应稳定，PSI过滤有意义
        - pb_pct_neu是MDI第一名(gain=208.90)，证明原始值因子极有价值

        流程：
        1. 低方差过滤
        2. 缺失率过滤 + 高相关去重
        3. PSI过滤（仅对rank因子，PSI > 0.25移除；原始值因子用宽松阈值2.0）
        4. RankIC过滤（|IC| < 0.03 的特征移除）【v6优化：0.01 → 0.03】
        5. 训练临时fi_model（reg:squarederror，跑满n_estimators，无早停）
        6. MDI过滤（XGBoost gain = 0 的特征移除）
        7. MDA过滤（permutation importance <= 0 的特征移除）
        8. 跳过PCA一致性检验

        Args:
            df_train: 训练集DataFrame
            df_val: 验证集DataFrame
            feats: 需要筛选的特征名列表（不含行业哑变量和缺失指示器）
            target_col: 目标列名

        Returns:
            筛选后的特征名列表
        """
        logger.info("=" * 60)
        logger.info("【v6因子筛选】PSI+IC初筛（PSI仅对rank因子）+ MDI/MDA复筛")
        logger.info("=" * 60)

        feats = sorted(feats)
        if not feats:
            logger.warning("候选特征为空")
            return []

        if target_col not in df_train.columns:
            raise ValueError(f"训练集必须包含 '{target_col}' 列")

        y_train = df_train[target_col].values
        y_val = df_val[target_col].values

        # ---------- 步骤1：低方差过滤 ----------
        logger.info("【步骤1】低方差过滤...")
        train_feat_df = df_train[feats].replace([np.inf, -np.inf], np.nan)
        selector = VarianceThreshold(threshold=1e-5)
        selector.fit(train_feat_df.fillna(0))
        low_variance_cols = train_feat_df.columns[selector.get_support()].tolist()
        feats = low_variance_cols
        logger.info(f"低方差过滤后保留特征数：{len(feats)}")

        if not feats:
            logger.warning("低方差过滤后无特征")
            return []

        # ---------- 步骤2：缺失率过滤 + 高相关去重 ----------
        logger.info("【步骤2】缺失率过滤 & 高相关去重...")
        kept_feats = self._sanitize_features(df_train[feats], self._feat_sel_nan_threshold, self._feat_sel_max_corr)
        feats = kept_feats
        logger.info(f"预处理后保留特征数：{len(feats)}")

        if not feats:
            logger.warning("预处理后无特征")
            return []

        # ---------- 步骤3&4：PSI过滤 + RankIC过滤 ----------
        # 【v6优化】PSI过滤改为仅对rank因子生效
        logger.info("【步骤3-4】PSI过滤（仅rank因子）& RankIC过滤（阈值0.03）...")
        tr = df_train[feats].replace([np.inf, -np.inf], np.nan)
        va = df_val[feats].replace([np.inf, -np.inf], np.nan)
        logger.info(f"训练集样本: {len(tr)}条, 验证集样本: {len(va)}条, 候选特征数: {len(feats)}")

        all_feat_scores = []
        for f in feats:
            try:
                tr_col = tr[f]
                va_col = va[f]
                tr_median = tr_col.median(skipna=True)
                if pd.isna(tr_median):
                    tr_median = 0.0

                # PSI计算 - 【v6优化】仅对rank因子严格PSI过滤
                is_rank_factor = '_rank_' in f  # 【v6新增】判断是否为rank因子
                psi = self._calc_psi(tr_col, va_col, fill_value=tr_median)

                if is_rank_factor:
                    # rank因子：使用标准PSI阈值0.25
                    if psi > self._feat_sel_psi_threshold:
                        logger.debug(f"特征 {f} 被PSI过滤(rank因子): PSI={psi:.4f} > {self._feat_sel_psi_threshold}")
                        continue
                else:
                    # 原始值因子：使用宽松PSI阈值2.0（仅在极端分布偏移时过滤）
                    # 【v6优化原因】价值因子(PE/PB)在牛熊市分布差异大，PSI高不代表失效
                    if psi > 2.0:
                        logger.debug(f"特征 {f} 被PSI过滤(原始值宽松): PSI={psi:.4f} > 2.0")
                        continue

                # RankIC计算 - 【v6优化】阈值从0.01提升到0.03
                tr_filled = tr_col.fillna(tr_median).values
                ic_train = spearmanr(tr_filled, y_train)[0]
                if np.isnan(ic_train):
                    ic_train = 0

                ic_abs = abs(ic_train)
                if ic_abs < self._feat_sel_min_ic:
                    logger.debug(f"特征 {f} 被IC过滤: |IC|={ic_abs:.4f} < {self._feat_sel_min_ic}")
                    continue

                all_feat_scores.append({
                    'feature': f,
                    'psi': psi,
                    'ic': ic_train,
                    'ic_abs': ic_abs,
                    'is_rank': is_rank_factor,
                })
            except Exception as e:
                logger.debug(f"特征 {f} 计算异常: {e}")
                continue

        if not all_feat_scores:
            logger.warning("没有特征通过PSI/IC检查，返回空特征列表")
            return []

        good_feats = [fs['feature'] for fs in all_feat_scores]
        logger.info(f"PSI/IC过滤后保留特征数：{len(good_feats)} (其中rank因子: {sum(1 for fs in all_feat_scores if fs['is_rank'])}, 原始值因子: {sum(1 for fs in all_feat_scores if not fs['is_rank'])})")

        if len(good_feats) < 10:
            logger.warning(f"有效特征数 {len(good_feats)} 小于最低要求 10，特征筛选失败")
            return good_feats  # 返回现有结果

        # ---------- 步骤5：训练临时fi_model ----------
        logger.info("【步骤5】训练临时fi_model（MDI/MDA共用, %d 棵树）...", self._feat_sel_mdi_n_estimators)

        X_train_fi = df_train[good_feats].replace([np.inf, -np.inf], np.nan).fillna(0).values
        X_val_fi = df_val[good_feats].replace([np.inf, -np.inf], np.nan).fillna(0).values

        fi_params = {
            'max_depth': self.config.get('xgb_max_depth', 3),
            'eta': self.config.get('xgb_learning_rate', 0.05),
            'subsample': self.config.get('xgb_subsample', 0.8),
            'colsample_bytree': self.config.get('xgb_colsample_bytree', 0.6),
            'alpha': self.config.get('xgb_reg_alpha', 0.1),
            'lambda': self.config.get('xgb_reg_lambda', 1.0),
            'min_child_weight': self.config.get('xgb_min_child_weight', 50),
            'objective': 'reg:squarederror',
            'seed': self.config.get('xgb_random_state', 42),
            'nthread': 1,
        }

        dtrain_fi = xgb.DMatrix(X_train_fi, label=y_train, feature_names=good_feats)
        dval_fi = xgb.DMatrix(X_val_fi, label=y_val, feature_names=good_feats)

        fi_model = xgb.train(
            fi_params,
            dtrain_fi,
            num_boost_round=self._feat_sel_mdi_n_estimators,
            verbose_eval=False
        )

        # ---------- 步骤6：MDI过滤（XGBoost gain） ----------
        logger.info("【步骤6】MDI过滤（XGBoost gain）...")
        xgb_scores = fi_model.get_score(importance_type='gain')
        mdi_series = pd.Series(
            [xgb_scores.get(f, 0) for f in good_feats],
            index=good_feats
        )

        # 归一化
        mdi_sum = mdi_series.sum()
        if mdi_sum > 0:
            mdi_series = mdi_series / mdi_sum

        # MDI > 0 过滤
        mdi_filtered = [f for f in good_feats if mdi_series[f] > 0]
        logger.info(f"MDI>0 的特征数: {len(mdi_filtered)} / {len(good_feats)}")

        if not mdi_filtered:
            logger.warning("无特征通过MDI过滤")
            return good_feats  # 返回IC过滤结果

        # ---------- 步骤7：MDA过滤（Permutation Importance） ----------
        logger.info("【步骤7】MDA过滤（Permutation Importance, %d 次）...", self._feat_sel_mda_n_repeats)

        perm_model = XGBRegressorWrapper(fi_model)

        perm_imp = permutation_importance(
            perm_model, X_val_fi, y_val,
            n_repeats=self._feat_sel_mda_n_repeats,
            random_state=42,
            scoring='neg_mean_squared_error'
        )

        mda_series = pd.Series(
            np.maximum(perm_imp.importances_mean, 0),
            index=good_feats
        )

        # MDA > 0 过滤
        final_feats = [f for f in mdi_filtered if mda_series[f] > 0]
        logger.info(f"MDA>0 的特征数: {len(final_feats)} / {len(mdi_filtered)}")

        if not final_feats:
            logger.warning("无特征通过MDA过滤，返回MDI过滤结果")
            return mdi_filtered

        # ---------- 汇总 ----------
        logger.info(f"因子筛选完成: {len(feats)} → {len(final_feats)} 个")
        if all_feat_scores:
            passed_scores = [fs for fs in all_feat_scores if fs['feature'] in final_feats]
            if passed_scores:
                avg_ic = np.mean([fs['ic_abs'] for fs in passed_scores])
                avg_psi = np.mean([fs['psi'] for fs in passed_scores])
                max_ic = max([fs['ic_abs'] for fs in passed_scores])
                rank_count = sum(1 for fs in passed_scores if fs['is_rank'])
                raw_count = len(passed_scores) - rank_count
                logger.info(f"  平均|RankIC|: {avg_ic:.4f}, 平均PSI: {avg_psi:.4f}, 最高|IC|: {max_ic:.4f}")
                logger.info(f"  因子类型: rank因子 {rank_count}个, 原始值因子 {raw_count}个")

        # 记录Top10特征重要性
        if final_feats:
            sorted_by_mdi = sorted(final_feats, key=lambda f: mdi_series[f], reverse=True)[:10]
            logger.info("Top 10 MDI特征:")
            for rank, f in enumerate(sorted_by_mdi, 1):
                is_rank_str = "(rank)" if '_rank_' in f else "(raw)"
                logger.info(f"  #{rank} {f}: MDI={mdi_series[f]:.6f}, MDA={mda_series[f]:.6f} {is_rank_str}")

        return sorted(final_feats)

    # ====================================================================
    # 中性化方法
    # ====================================================================

    def _neutralize(self, df, factor_cols):
        """
        使用通用中性化工具进行行业+市值中性化。
        """
        df = df.copy()
        if df.empty:
            return df

        for col in factor_cols:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        if 'industry' in df.columns:
            ind_dummies = pd.get_dummies(df['industry'].astype(str), prefix='ind')
        else:
            ind_dummies = pd.DataFrame(index=df.index)

        if 'total_mv' in df.columns:
            log_mv = np.log(df['total_mv'].clip(lower=1e-8))
        else:
            log_mv = pd.Series(0.0, index=df.index)

        X_neutral = pd.concat([ind_dummies, log_mv.rename('log_mv')], axis=1).fillna(0)

        neutralizer = FactorNeutralizer()
        df = neutralizer.neutralize(df, factor_cols, X_neutral)

        return df

    # ====================================================================
    # 【v5优化】构建训练数据 - 含因子筛选
    # ====================================================================

    def _build_training_dataset(self, end_date: datetime):
        """
        构建训练数据集（含因子筛选）。

        【v5 优化说明】
        1. 因子深加工：计算8个交互因子
        2. 因子筛选：PSI+IC初筛 + MDI/MDA复筛
        3. 时序验证集划分

        返回: (dtrain, dval, sample_weights_train, df_train, df_val) 五元组
              如果训练数据不足，返回 (None, None, None, None, None)
        """
        _orig_full_cal = self.full_calendar
        _truncated_cal = [d for d in self.full_calendar if d < end_date]
        if _truncated_cal:
            self.full_calendar = _truncated_cal

        all_snaps = self._get_snapshot_dates()

        valid_snaps = []
        for d in all_snaps:
            if d >= end_date:
                continue
            future = self._get_trading_day_offset(d, self.pred_days)
            if future is None:
                continue
            valid_snaps.append(d)

        if not valid_snaps:
            logger.warning("无可用快照（缺少足够未来交易日）")
            self.full_calendar = _orig_full_cal
            return None, None, None, None, None

        valid_snaps.sort()
        train_snaps = valid_snaps[-self.train_window:] if len(valid_snaps) >= self.train_window else valid_snaps

        if not train_snaps:
            logger.warning("无有效训练快照")
            self.full_calendar = _orig_full_cal
            return None, None, None, None, None

        logger.info(f"训练快照日期：{train_snaps[0].date()} ~ {train_snaps[-1].date()}，共 {len(train_snaps)} 天")

        samples = []
        for dt in train_snaps:
            snap = self._load_snapshot(dt)
            if snap is None or snap.empty:
                continue

            future_date = self._get_trading_day_offset(dt, self.pred_days)
            if future_date is None:
                continue

            base_px = self._get_close_prices(snap['code'], dt)
            future_px = self._get_close_prices(snap['code'], future_date)

            if len(base_px) == 0 or len(future_px) == 0:
                continue

            labels = {}
            for code in snap['code'].unique():
                bp = base_px.get(code, np.nan)
                fp = future_px.get(code, np.nan)
                if pd.notna(bp) and pd.notna(fp) and bp != 0:
                    labels[code] = fp / bp - 1

            if not labels:
                continue

            snap = snap.copy()
            snap['label'] = snap['code'].map(labels)
            snap.dropna(subset=['label'], inplace=True)

            if snap.empty:
                continue

            snap['snapshot_date'] = dt
            samples.append(snap)

        self.full_calendar = _orig_full_cal

        if not samples:
            logger.warning("无有效训练样本")
            return None, None, None, None, None

        data = pd.concat(samples, ignore_index=True)
        logger.info(f"原始样本总数：{len(data)}")

        # ---------- 截面排名变换目标 ----------
        data['label_raw'] = data['label'].copy()
        data['label'] = data.groupby('snapshot_date')['label'].rank(pct=True)
        logger.info(
            f"标签变换完成：原始收益率 → 截面排名百分位，"
            f"排名分布: min={data['label'].min():.4f}, max={data['label'].max():.4f}, "
            f"mean={data['label'].mean():.4f}"
        )

        # ---------- 时序验证集划分 ----------
        n_snaps = len(train_snaps)
        val_split_idx = int(n_snaps * 0.8)
        train_snap_dates = train_snaps[:val_split_idx]
        val_snap_dates = train_snaps[val_split_idx:]

        train_data = data[data['snapshot_date'].isin(train_snap_dates)].copy()
        val_data = data[data['snapshot_date'].isin(val_snap_dates)].copy()

        logger.info(
            f"时序验证集划分：训练集 {len(train_snap_dates)} 天 ({train_snap_dates[0].date()} ~ {train_snap_dates[-1].date()}), "
            f"验证集 {len(val_snap_dates)} 天 ({val_snap_dates[0].date()} ~ {val_snap_dates[-1].date()})"
        )
        logger.info(f"训练样本数: {len(train_data)}, 验证样本数: {len(val_data)}")

        # ---------- 特征工程（基础特征） ----------
        raw = ['pe', 'pb', 'roe', 'gross_margin', 'ocf_per_share',
               'rev_growth', 'profit_growth', 'rev_qoq', 'profit_qoq']
        features = [c for c in raw + [f'{c}_rank' for c in raw] +
                    ['pe_pct', 'pb_pct', 'close_pct'] if c in train_data.columns]

        if 'total_mv' in train_data.columns:
            train_data['log_mv'] = np.log(train_data['total_mv'].clip(lower=1e-8))
            val_data['log_mv'] = np.log(val_data['total_mv'].clip(lower=1e-8))
            features.append('log_mv')
        else:
            train_data['log_mv'] = 0
            val_data['log_mv'] = 0

        # ---------- 【v5新增】交互因子 ----------
        logger.info("【v5因子深加工】计算交互因子...")
        train_data = self._add_interaction_features(train_data)
        val_data = self._add_interaction_features(val_data)

        # 交互因子列表（用于中性化）
        interaction_feats = [
            'quality_value', 'growth_consistency', 'earnings_quality',
            'value_momentum', 'growth_accel', 'cf_quality',
            'value_composite', 'garp'
        ]
        available_interactions = [f for f in interaction_feats if f in train_data.columns]
        features.extend(available_interactions)
        logger.info(f"交互因子加入特征列表: {available_interactions}")

        # ---------- 中性化 ----------
        train_neu = self._neutralize(train_data, features)
        val_neu = self._neutralize(val_data, features)

        # ---------- 【v5新增】因子筛选（仅对基本面因子，不含行业哑变量和缺失指示器） ----------
        # 筛选候选：原始因子_neu + 交互因子_neu
        base_features_to_select = [c + '_neu' for c in features if c + '_neu' in train_neu.columns]
        logger.info(f"【v6因子筛选】候选因子数: {len(base_features_to_select)}")

        if len(base_features_to_select) >= 10:
            selected_base_feats = self._select_features(
                train_neu, val_neu, base_features_to_select, target_col='label'
            )
            self._selected_base_features = selected_base_feats
            logger.info(f"因子筛选完成，保留 {len(selected_base_feats)} 个基本面因子")
        else:
            # 特征太少，跳过筛选
            self._selected_base_features = base_features_to_select
            logger.info(f"特征数过少({len(base_features_to_select)})，跳过筛选")

        # ---------- 特征矩阵构建 ----------
        # 1. 筛选后的基本面因子（中性化）
        feat_neu = [c for c in self._selected_base_features if c in train_neu.columns]
        X_train_neu = train_neu[feat_neu].copy()
        X_val_neu = val_neu[feat_neu].copy()

        # 2. 缺失指示特征（仅对筛选后保留的因子创建）
        # 反推筛选后因子对应的原始列名：pb_pct_neu → pb_pct
        selected_raw_cols = [c.replace('_neu', '') for c in self._selected_base_features]
        self._missing_indicator_cols = []
        for col in selected_raw_cols:
            if col in train_data.columns:
                nan_rate = train_data[col].isna().sum() / len(train_data)
                if nan_rate > 0.01:
                    missing_col = f'{col}_missing'
                    train_data[missing_col] = (
                            train_data[col].isna() | train_data[col].isin([np.inf, -np.inf])
                    ).astype(int)
                    val_data[missing_col] = (
                            val_data[col].isna() | val_data[col].isin([np.inf, -np.inf])
                    ).astype(int)
                    self._missing_indicator_cols.append(missing_col)

        X_train_missing = train_data[
            self._missing_indicator_cols].copy() if self._missing_indicator_cols else pd.DataFrame(
            index=train_data.index)
        X_val_missing = val_data[self._missing_indicator_cols].copy() \
            if self._missing_indicator_cols else pd.DataFrame(
            index=val_data.index)

        # 3. 【v5修复】不再添加行业哑变量
        # 原因：_neu特征已通过行业+市值中性化（残差化），行业效应已被移除。
        # 若再添加行业哑变量，模型会用行业作为最易分裂路径，压制基本面因子，
        # 导致v4"行业哑变量主导55%特征重要性"的问题在v5中重现。
        # 行业信息已在中性化阶段被正确使用，无需在模型训练阶段重复引入。
        self._industry_dummy_cols = []

        # 4. 合并特征矩阵（仅 _neu因子 + 缺失指示器）
        X_train = pd.concat([X_train_neu, X_train_missing], axis=1)
        X_val = pd.concat([X_val_neu, X_val_missing], axis=1)

        self.feature_names = list(X_train.columns)

        y_train = train_data['label'].values
        y_val = val_data['label'].values

        # ---------- 序贯采样权重 ----------
        logger.info("计算序贯采样权重（平均唯一性）...")
        truncated_cal = [d for d in _orig_full_cal if d < end_date]
        try:
            sample_weights = compute_sample_weights(
                train_data=train_data,
                calendar=truncated_cal,
                max_hold_days=self.pred_days
            )
            if len(sample_weights) != len(train_data):
                logger.warning("序贯采样权重长度不一致，使用均匀权重")
                sample_weights = np.ones(len(train_data))
            else:
                logger.info(
                    f"序贯采样权重统计: min={sample_weights.min():.4f}, "
                    f"max={sample_weights.max():.4f}, mean={sample_weights.mean():.4f}"
                )
        except Exception as e:
            logger.warning(f"序贯采样权重计算失败: {e}，使用均匀权重")
            sample_weights = np.ones(len(train_data))

        # ---------- 构建 DMatrix ----------
        dtrain = xgb.DMatrix(
            X_train, label=y_train, weight=sample_weights,
            feature_names=self.feature_names
        )
        dval = xgb.DMatrix(
            X_val, label=y_val,
            feature_names=self.feature_names
        )

        return dtrain, dval, sample_weights, train_data, val_data

    # ====================================================================
    # 【v5优化】特征工程辅助方法（预测时使用）
    # ====================================================================

    def _build_feature_matrix(self, snap: pd.DataFrame) -> pd.DataFrame:
        """
        从快照数据构建特征矩阵（训练/预测共用）。
        特征矩阵 = 筛选后的中性化因子 + 缺失指示器 + 行业哑变量
        """
        raw = ['pe', 'pb', 'roe', 'gross_margin', 'ocf_per_share',
               'rev_growth', 'profit_growth', 'rev_qoq', 'profit_qoq']
        features = [c for c in raw + [f'{c}_rank' for c in raw] +
                    ['pe_pct', 'pb_pct', 'close_pct'] if c in snap.columns]

        if 'total_mv' in snap.columns:
            snap = snap.copy()
            snap['log_mv'] = np.log(snap['total_mv'].clip(lower=1e-8))
            features.append('log_mv')
        else:
            snap = snap.copy()
            snap['log_mv'] = 0

        # ---------- 【v5新增】交互因子 ----------
        snap = self._add_interaction_features(snap)

        interaction_feats = [
            'quality_value', 'growth_consistency', 'earnings_quality',
            'value_momentum', 'growth_accel', 'cf_quality',
            'value_composite', 'garp'
        ]
        available_interactions = [f for f in interaction_feats if f in snap.columns]
        features.extend(available_interactions)

        # 中性化
        snap_neu = self._neutralize(snap, features)

        # 1. 筛选后的中性化因子（与训练时一致）
        feat_neu = [c for c in self._selected_base_features if c in snap_neu.columns]
        X_neu = snap_neu[feat_neu].copy()

        # 2. 缺失指示特征
        X_missing = pd.DataFrame(index=snap_neu.index)
        for missing_col in self._missing_indicator_cols:
            raw_col = missing_col.replace('_missing', '')
            if raw_col in snap.columns:
                X_missing[missing_col] = (
                        snap[raw_col].isna() | snap[raw_col].isin([np.inf, -np.inf])
                ).astype(int)
            else:
                X_missing[missing_col] = 1

        # 3. 【v5修复】不再添加行业哑变量（与训练时一致）
        # 行业信息已在中性化阶段使用，模型输入不含行业哑变量

        # 4. 合并（仅 _neu因子 + 缺失指示器）
        X = pd.concat([X_neu, X_missing], axis=1)

        # 对齐特征列
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = np.nan
        X = X[self.feature_names]

        return X

    # ====================================================================
    # 模型评估与特征重要性
    # ====================================================================

    def _calc_model_performance(self, dtrain: xgb.DMatrix, dval: xgb.DMatrix = None) -> dict:
        """
        计算模型性能指标。
        """
        if self.model is None:
            return {}

        perf = {}

        # 训练集性能
        preds_train = self.model.predict(dtrain)
        labels_train = dtrain.get_label()
        rmse_train = np.sqrt(mean_squared_error(labels_train, preds_train))
        r2_train = r2_score(labels_train, preds_train)
        mae_train = np.mean(np.abs(preds_train - labels_train))
        corr_train = np.corrcoef(preds_train, labels_train)[0, 1] if len(labels_train) > 1 else np.nan
        if len(labels_train) > 1:
            spearman_train, _ = spearmanr(preds_train, labels_train)
        else:
            spearman_train = np.nan

        perf['train'] = {
            'rmse': rmse_train, 'r2': r2_train, 'mae': mae_train,
            'corr': corr_train, 'spearman_ic': spearman_train,
            'n_samples': len(labels_train)
        }

        # 验证集性能
        if dval is not None:
            preds_val = self.model.predict(dval)
            labels_val = dval.get_label()
            rmse_val = np.sqrt(mean_squared_error(labels_val, preds_val))
            r2_val = r2_score(labels_val, preds_val)
            mae_val = np.mean(np.abs(preds_val - labels_val))
            corr_val = np.corrcoef(preds_val, labels_val)[0, 1] if len(labels_val) > 1 else np.nan
            if len(labels_val) > 1:
                spearman_val, _ = spearmanr(preds_val, labels_val)
            else:
                spearman_val = np.nan

            perf['val'] = {
                'rmse': rmse_val, 'r2': r2_val, 'mae': mae_val,
                'corr': corr_val, 'spearman_ic': spearman_val,
                'n_samples': len(labels_val)
            }

        return perf

    def _log_feature_importance(self, top_n: int = 15):
        """
        记录特征重要性。
        """
        if self.model is None:
            return
        importance = self.model.get_score(importance_type='gain')
        if not importance:
            return
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        logger.info("特征重要性 Top %d (gain):", len(sorted_items))
        for feat, score in sorted_items:
            logger.info(f"  {feat}: {score:.2f}")

    # ====================================================================
    # recommend 方法 - 供 alpha191 元标注模型调用
    # ====================================================================

    def recommend(self, context: StrategyContext, max_n: int = None) -> Tuple[List[str], Dict[str, float]]:
        """
        完全复刻 before_market_open 中的训练与选股流程。

        【v6更新】
        1. 支持因子筛选后的特征（PSI仅对rank因子生效）
        2. 支持模型置信度门控三档持仓
        3. 支持预测值加权

        Returns:
            Tuple[List[str], Dict[str, float]]: (推荐股票代码列表, {code: 预测排名百分位})
        """
        if max_n is None:
            max_n = self.top_n

        current_dt = context.current_dt
        logger.info(f"【基本面主模型】开始为 {current_dt.date()} 选股，目标数量 {max_n}")

        # 1. 训练数据
        train_res, val_res, sample_weights, _, _ = self._build_training_dataset(current_dt)

        if train_res is None:
            logger.warning("基本面主模型：训练数据不足")
            return ([], {})

        dtrain = train_res
        dval = val_res

        n_samples = dtrain.num_row()
        n_features = len(self.feature_names)

        logger.info(f"基本面主模型训练样本数: {n_samples}，验证样本数: {dval.num_row()}，特征数: {n_features}")

        t0 = time.time()

        # 训练 - 【v6优化】early_stopping_rounds从30降至20
        self.model = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=self.config.get('xgb_n_estimators', 500),
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False
        )

        train_time = time.time() - t0
        logger.info(f"基本面主模型训练完成，耗时 {train_time:.1f} 秒，最佳迭代: {self.model.best_iteration}")

        # 性能评估
        perf = self._calc_model_performance(dtrain, dval)
        val_ic = perf.get('val', {}).get('spearman_ic', 0.0) if perf else 0.0
        self.val_ic = val_ic if val_ic is not None else 0.0

        if perf:
            logger.info(
                f"基本面主模型训练集: RMSE={perf['train']['rmse']:.4f}, R²={perf['train']['r2']:.4f}, "
                f"Corr={perf['train']['corr']:.4f}, Spearman IC={perf['train']['spearman_ic']:.4f}"
            )
            if 'val' in perf:
                logger.info(
                    f"基本面主模型验证集: RMSE={perf['val']['rmse']:.4f}, R²={perf['val']['r2']:.4f}, "
                    f"Corr={perf['val']['corr']:.4f}, Spearman IC={perf['val']['spearman_ic']:.4f}"
                )

        # 2. 获取最新快照
        prev = self._get_previous_snapshot_date(current_dt)
        if prev is None:
            logger.warning("基本面主模型：无上一交易日快照")
            return ([], {})

        snap = self._load_snapshot(prev)
        if snap is None or snap.empty:
            return ([], {})

        # 3. 候选池过滤
        # 【修复】使用 self.stock_pool 而非 context.stock_pool，
        # 与 before_market_open 保持一致，避免被外部调用者的 context 污染
        candidates = self.stock_pool.get_candidates(current_dt, context)
        cand_digits = {c.replace('sh.', '').replace('sz.', '') for c in candidates}
        snap = snap[snap['code'].isin(cand_digits)]

        if snap.empty:
            logger.warning("基本面主模型：快照中无候选股票")
            return ([], {})

        # 4. 特征工程与预测
        X_pred = self._build_feature_matrix(snap)
        dpred = xgb.DMatrix(X_pred, feature_names=self.feature_names)
        preds = self.model.predict(dpred)

        snap = snap.copy()
        snap['pred'] = preds

        logger.info(f"基本面主模型预测统计: 均值={preds.mean():.4f}, 标准差={preds.std():.4f}, "
                    f"最小={preds.min():.4f}, 最大={preds.max():.4f}")

        # 【v6优化】置信度门控三档持仓（不再清仓）
        confidence_level = "正常"
        position_multiplier = 1.0
        if self.val_ic < self.val_ic_defend_threshold:  # IC < 0.05
            confidence_level = "防御"
            position_multiplier = 0.25
            max_n = max(10, int(max_n * position_multiplier))  # 最低10只，不再清仓
            logger.warning(f"验证集IC={self.val_ic:.4f} < {self.val_ic_defend_threshold}，"
                           f"防御置信度，持仓数调整为: {max_n}")
        elif self.val_ic < self.val_ic_trade_threshold:  # 0.05 ≤ IC < 0.10
            confidence_level = "保守"
            position_multiplier = 0.5
            max_n = max(1, int(max_n * position_multiplier))
            logger.warning(f"验证集IC={self.val_ic:.4f} < {self.val_ic_trade_threshold}，"
                           f"保守置信度，持仓数调整为: {max_n}")
        else:  # IC ≥ 0.10
            logger.info(f"验证集IC={self.val_ic:.4f} >= {self.val_ic_trade_threshold}，正常置信度")

        # 5. 选出前 max_n 只
        n_select = min(max_n, len(snap))
        selected = snap.sort_values('pred', ascending=False).head(n_select)

        # 【v6新增】预测值加权
        temperature = self.pred_weight_temperature
        max_pct = self.max_position_pct
        preds_values = selected['pred'].values
        # softmax加权
        exp_preds = np.exp((preds_values - preds_values.max()) / temperature)
        weights = exp_preds / exp_preds.sum()
        # 限制单只最大权重
        weights = np.minimum(weights, max_pct)
        # 重新归一化
        weights = weights / weights.sum()

        # 6. 输出
        logger.info(f"基本面主模型选出 {n_select} 只股票（置信度: {confidence_level}），前5只：")
        for i, (_, row) in enumerate(selected.head(5).iterrows()):
            pe_val = row.get('pe', np.nan)
            pb_val = row.get('pb', np.nan)
            roe_val = row.get('roe', np.nan)
            w = weights[i] if i < len(weights) else 0
            logger.info(f"  #{i+1} {row['code']} | pred={row['pred']:.4f} | weight={w:.4f} | PE:{pe_val:.2f} PB:{pb_val:.2f} ROE:{roe_val:.3f}")

        # 7. 转为 sh/sz 格式
        rec_codes = []
        pred_dict = {}
        for i, (_, row) in enumerate(selected.iterrows()):
            digits = str(row['code']).zfill(6)
            if digits.startswith(('6', '9')):
                full_code = 'sh.' + digits
            else:
                full_code = 'sz.' + digits
            rec_codes.append(full_code)
            pred_dict[full_code] = float(row['pred'])

        return rec_codes, pred_dict

    # ====================================================================
    # 主驱动：盘前选股
    # ====================================================================

    def before_market_open(self, context: StrategyContext):
        self.trade_day_seq += 1
        if self.trade_day_seq % self.rebalance_interval != 1:
            return

        current_dt = context.current_dt
        logger.info(f"[{current_dt.date()}] ========== 调仓日，开始训练与选股 ==========")

        # 1. 构建训练数据
        train_res, val_res, sample_weights, _, _ = self._build_training_dataset(current_dt)

        if train_res is None:
            logger.warning("训练数据不足，清仓")
            self.target_weights = {}
            return

        n_samples = train_res.num_row()
        n_features = len(self.feature_names)

        logger.info(f"训练样本数: {n_samples}，验证样本数: {val_res.num_row()}，特征数: {n_features}")

        t0 = time.time()

        # 2. 模型训练 - 【v6优化】early_stopping_rounds从30降至20
        self.model = xgb.train(
            self.xgb_params,
            train_res,
            num_boost_round=self.config.get('xgb_n_estimators', 500),
            evals=[(train_res, 'train'), (val_res, 'val')],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=20
        )

        train_time = time.time() - t0
        logger.info(f"XGBoost 训练完成，耗时 {train_time:.1f} 秒，最佳迭代: {self.model.best_iteration}")

        # 3. 模型性能
        perf = self._calc_model_performance(train_res, val_res)

        # 【v5新增】记录验证集IC用于置信度门控
        val_ic = perf.get('val', {}).get('spearman_ic', 0.0) if perf else 0.0
        self.val_ic = val_ic if val_ic is not None else 0.0

        if perf:
            logger.info(
                f"训练集性能: RMSE={perf['train']['rmse']:.4f}, R²={perf['train']['r2']:.4f}, "
                f"MAE={perf['train']['mae']:.4f}, Corr={perf['train']['corr']:.4f}, "
                f"Spearman IC={perf['train']['spearman_ic']:.4f}"
            )
            if 'val' in perf:
                logger.info(
                    f"验证集性能: RMSE={perf['val']['rmse']:.4f}, R²={perf['val']['r2']:.4f}, "
                    f"MAE={perf['val']['mae']:.4f}, Corr={perf['val']['corr']:.4f}, "
                    f"Spearman IC={perf['val']['spearman_ic']:.4f}"
                )

        self._log_feature_importance(top_n=15)

        # 【v6优化】置信度门控三档持仓（不再清仓，A股长期向上，空仓机会成本过高）
        confidence_level = "正常"
        position_multiplier = 1.0
        if self.val_ic < self.val_ic_defend_threshold:  # IC < 0.05
            confidence_level = "防御"
            position_multiplier = 0.25
            logger.warning(
                f"⚠️ 验证集IC={self.val_ic:.4f} < {self.val_ic_defend_threshold}，"
                f"防御置信度，持仓数减至1/4（最低10只），不再清仓"
            )
        elif self.val_ic < self.val_ic_trade_threshold:  # 0.05 ≤ IC < 0.10
            confidence_level = "保守"
            position_multiplier = 0.5
            logger.warning(
                f"⚠️ 验证集IC={self.val_ic:.4f} < {self.val_ic_trade_threshold}，"
                f"保守置信度，持仓数减半"
            )
        else:  # IC ≥ 0.10
            logger.info(f"✅ 验证集IC={self.val_ic:.4f} >= {self.val_ic_trade_threshold}，正常置信度")

        # 4. 获取最新因子快照
        prev = self._get_previous_snapshot_date(current_dt)
        if prev is None:
            logger.warning("无上一交易日快照，清仓")
            self.target_weights = {}
            return

        snap = self._load_snapshot(prev)
        if snap is None or snap.empty:
            self.target_weights = {}
            return

        # 5. 候选池过滤
        candidates = self.stock_pool.get_candidates(current_dt, context)
        cand_digits = {c.replace('sh.', '').replace('sz.', '') for c in candidates}
        snap = snap[snap['code'].isin(cand_digits)]

        if snap.empty:
            self.target_weights = {}
            return

        # 6. 特征工程与预测
        X_pred = self._build_feature_matrix(snap)
        dpred = xgb.DMatrix(X_pred, feature_names=self.feature_names)
        preds = self.model.predict(dpred)

        snap = snap.copy()
        snap['pred'] = preds

        logger.info(f"预测统计: 均值={preds.mean():.4f}, 标准差={preds.std():.4f}, "
                    f"最小={preds.min():.4f}, 最大={preds.max():.4f}")

        # 7. 选股
        # 【v6优化】防御置信度时持仓数降至1/4（最低10只），不再清仓
        # 【v6优化】偏低置信度时持仓数减半
        top_quantile = self.config.get('top_quantile', 0.02)
        n_candidates = len(snap)
        n_select = max(1, int(np.ceil(n_candidates * top_quantile)))
        n_select = min(n_select, self.max_positions)

        if position_multiplier < 1.0:
            n_select = max(10 if position_multiplier <= 0.25 else 1, int(n_select * position_multiplier))
            logger.info(f"{confidence_level}置信度，持仓数调整为: {n_select}")

        selected = snap.sort_values('pred', ascending=False).head(n_select)

        # 【v6新增】预测值加权持仓
        temperature = self.pred_weight_temperature
        max_pct = self.max_position_pct
        preds_values = selected['pred'].values
        # softmax加权
        exp_preds = np.exp((preds_values - preds_values.max()) / temperature)
        weights = exp_preds / exp_preds.sum()
        # 限制单只最大权重
        weights = np.minimum(weights, max_pct)
        # 重新归一化
        weights = weights / weights.sum()

        # 详细选股日志
        logger.info(f"选股结果（共 {len(selected)} 只，置信度: {confidence_level}）：")
        for i, (_, row) in enumerate(selected.iterrows()):
            code = str(row['code']).zfill(6)
            pe_val = row.get('pe', np.nan)
            pb_val = row.get('pb', np.nan)
            roe_val = row.get('roe', np.nan)
            w = weights[i] if i < len(weights) else 0
            logger.info(f"  #{i+1} {code}  | 预测排名百分位: {row['pred']:.4f} | 权重: {w:.4f} | "
                        f"PE: {pe_val:.2f} | PB: {pb_val:.2f} | ROE: {roe_val:.3f}")

        # 构建目标权重 - 【v6优化】使用softmax加权权重
        self.target_weights = {}
        for i, (_, row) in enumerate(selected.iterrows()):
            digits = str(row['code']).zfill(6)
            full_code = 'sh.' + digits if digits.startswith(('6', '9')) else 'sz.' + digits
            w = weights[i] if i < len(weights) else 0
            self.target_weights[full_code] = w

        total_weight = sum(self.target_weights.values())
        logger.info(f"本次调仓目标：{len(self.target_weights)} 只股票，总权重 {total_weight:.4f}，置信度: {confidence_level}")

    # ====================================================================
    # 盘中交易执行
    # ====================================================================

    def handle_bar(self, context: StrategyContext):
        broker = context.broker
        portfolio = context.portfolio
        date = context.current_dt

        is_rebalance = (self.trade_day_seq % self.rebalance_interval == 1)

        if is_rebalance:
            logger.debug(f"[handle_bar] 调仓日触发: trade_day_seq={self.trade_day_seq}, "
                         f"rebalance_interval={self.rebalance_interval}")

        if is_rebalance:
            # Step 1: 卖出不在目标中的旧仓
            for code in list(portfolio.positions.keys()):
                if code not in self.target_weights:
                    broker.order_target_percent(code, 0.0)

            # Step 2: 【v6优化】按预测值排序目标权重，买入可用槽位数量的新股票
            # 目标权重已通过softmax加权，不再需要重新等权分配
            sorted_targets = sorted(
                self.target_weights.items(), key=lambda x: x[1], reverse=True
            )

            # 计算实际可用槽位
            held_targets = [c for c in portfolio.positions if c in self.target_weights]
            residual = [c for c in portfolio.positions if c not in self.target_weights]
            available_slots = self.max_positions - len(held_targets) - len(residual)

            buy_targets = {}
            for code, weight in sorted_targets:
                if code in portfolio.positions:
                    buy_targets[code] = weight
                elif available_slots > 0:
                    buy_targets[code] = weight
                    available_slots -= 1

            # Step 3: 【v6优化】直接使用softmax加权权重下单
            if buy_targets:
                for code, weight in buy_targets.items():
                    broker.order_target_percent(code, weight)

        # 每日资产记录
        total_value = portfolio.total_value
        logger.info(
            f"[{date.date()}] {'调仓' if is_rebalance else '持仓'}后总资产: "
            f"{total_value:,.0f}，现金: {portfolio.cash:,.0f}，"
            f"持仓数: {len(portfolio.positions)}")
