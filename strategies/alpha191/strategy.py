# -*- coding: utf-8 -*-
"""
Alpha191 策略 - 元标注版本（完整迁移 + 基本面主模型集成 + 因子中性化 + 真实分钟路径标签）
【2026-05-08 改造】
1. 修复中性化fillna问题（移除.fillna(0)）
2. 将概率调整止盈止损改为基本面排名百分位调整止盈止损
3. 保留时序划分训练和验证设计（无需修改）
4. 保留序贯采样权重设计（无需修改）
5. 两阶段模型合并为单阶段
【2026-05-16 修正】
6. rank_pcts直接使用基本面主策略的预测值（rank(pct=True)），不再重新排名
7. _meta_select_stocks移除固定阈值，改为按P(profit)排序取top MAX_POSITIONS
8. min_win_prob参数废弃
9. recommend()直接返回(选股列表, 预测值字典)，alpha191不再重新计算，
   从根源确保预测排名百分位与单独运行基本面策略完全一致
10. handle_bar止盈止损直接从config读取pt_factor/sl_factor（self._pt_factor/self._sl_factor），
    不依赖meta_labeler属性，避免属性值与config不一致导致公式计算错误
11. 【修正11】标签止盈止损与交易完全一致：
    - compute_labels 的 fund_rank_pct 参数改为 {(date, code): rank_pct} 格式，
      支持按日期+股票精确查找，避免不同日期的同一股票rank值被覆盖
    - _assign_meta_labels 构建 {(date, code): rank_pct} 格式的字典传给 compute_labels
    - 标签生成时的 pt/sl 公式与 handle_bar 完全一致
12. 【修正12】交易止盈止损改为分钟K线跟踪，到期开盘价退出：
    - handle_bar 中检查持仓时，使用 get_minute_data 获取当日分钟K线
    - 逐分钟检查 high/low 是否触及 PT/SL 阈值（与 compute_labels 遍历顺序一致：先PT后SL）
    - 分钟数据不可用时回退到日线检查
    - 到期退出条件：held_trade_days >= max_hold_days
      设计逻辑（2026-08-07 标签时点修复）：
        标签：信号日(D-1)收盘入场 → PT/SL在D~D+max_hold-1内检测 → 到期取D+max_hold-1日收盘
        交易：D日开盘入场 → 持有至D+max_hold日开盘退出（held_trade_days=max_hold_days触发）
        标签-交易偏差：D-1 close → D open = 隔夜跳空，作为实盘可行性主动接受
      持仓天数与标签完全一致（均为 max_hold_days 个交易日）
    - 到期日以开盘价退出（在 before_market_open 中标记，handle_bar 中优先处理）

【2026-05-19 BUG修复】
BUG-3: 波动率计算注释说明 - 标签端截止日期(event_date)与交易端截止日期(date)的1天差异是设计上的滞后
BUG-4: 调仓续期时清除volatility_cache - 避免旧的波动率缓存与新的建仓价格/日期不匹配
BUG-7: 优化训练日期过滤循环 - 从O(N²)改为O(N)

【2026-06-01 BUG修复 - 本次修复】
BUG-A: self.reference_date 初始化为 None → 从 config 正确读取
BUG-B: _fund_params 硬编码 v3 参数 → 更新为 v4 参数
BUG-C: _check_pt_sl_minute 分钟数据空时无回退 → 已添加日线回退
BUG-D: LinearRegression 未导入 → 已添加导入
BUG-E: _get_cached_factor_snapshot 调用不存在的方法 load_snapshot_for_date → 重写为从 data_provider 获取日线数据计算 alpha191 因子
BUG-F: _load_select_mode_data 返回值未被使用 → 删除 before_market_open 中的无效调用
BUG-G: _get_cached_factor_snapshot 概念混乱 → 重写为从 data_provider 获取日线+计算alpha因子
BUG-H: self.feature_cols 训练后未更新 → 训练成功后保存 feature_cols_neu 到 self.feature_cols
BUG-I: _meta_select_stocks 预测前未中性化 → 添加 _neutralize_factors 调用生成 _neu 特征
BUG-J: _get_primary_recommendations*_with_rank 不从本地快照读取 → 优先从snapshot_manager读取stocks+rank_pcts，未命中时才回退实时训练
BUG-K: initialize()重复创建snapshot_manager覆盖已加载快照 → 改为条件判断(hasattr检查)
BUG-L: _get_cached_factor_snapshot用get_daily_data(None,...)报错 → 改用data_provider.get_factor_snapshot(date)
BUG-M: _neutralize_factors使用FactorNeutralizer(group_col=...)报错 → 恢复原版API: FactorNeutralizer()+neutralize(df,factor_cols,X_neutral)
BUG-N: _merge_fundamental_info使用_get_cached_factor_snapshot合并基本面 → 恢复原版: 使用load_fundamental_snapshot补充industry和total_mv列
BUG-O: _meta_select_stocks使用错误的预测流程 → 恢复原版: data_provider.get_factor_snapshot(t_minus_1)+ensemble_manager.selected_features+predict_proba
"""

# ===== [v3.0] Cross-mode Determinism Guarantee =====
# PYTHONHASHSEED=0: eliminate dict/set iteration order randomness
# OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1: eliminate BLAS floating-point non-determinism
import os
import json
os.environ.setdefault('PYTHONHASHSEED', '0')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import random
random.seed(42)

import time
import logging
import numpy as np
np.random.seed(42)
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from .ensemble import compute_sample_weights, compute_sample_weights_cached
from concurrent.futures import ThreadPoolExecutor, as_completed
from gpsef.core import IStrategy, StrategyContext
# [R6-E4 2026-09-10] 已移除死导入 (原 .pool 模块的 StockPoolManager, 导入后从未使用;
#   运行时股票池 = 引擎注入的 DynamicStockPool。pool.py 已归档 strategies/_deprecated_20260910/)
from .labels import TripleBarrierLabeler
from .barriers import compute_barriers
from .ensemble import EnsembleModelManager
from .data_utils import (get_trade_calendar, get_multi_index_stocks_by_akshare,
                         map_code_to_ts_code,
                         map_ts_code_to_digits,
                         build_industry_map_from_snapshots)
from gpsef.utils.neutralizer import FactorNeutralizer
from sklearn.linear_model import LinearRegression

# [v3.0] New modules
from .garch_position import GARCHPositionController

logger = logging.getLogger(__name__)


class GlobalConfig:
    """用于传递给 EnsembleModelManager 的配置容器"""
    # 特征重要性分析
    FI_KENDALL_TAU_THRESHOLD_LOW = 0.4  # MDI-MDA低限：低于此值走严格保守模式
    # [v3.1 P1-4] 温度缩放：放大 XGBoost 概率离散度
    # T=0.7 温和放大, 避免 T=0.3/0.5 过度集中 (Part C v4.0)
    PROB_TEMPERATURE = 0.5  # [2026-08-05] fallback, 实际值由 config meta_labeling.prob_temperature 注入


# ===== 模块级辅助函数（多进程市值缓存构建） =====
def _compute_total_mv_chunk(file_paths):
    """并行计算一批股票的 total_mv (模块级, 供 ProcessPoolExecutor 用)。

    向量化: total_mv = amount * 100 / turn, 逐文件批量计算。
    [2026-09-10 统一] 只做**逐文件纯计算**: 缓存的去重/排序由调用方在合并后统一处理
    (按 (trade_date, ts_code, total_mv) 排序后 keep='first'), 与并行完成顺序解耦。
    """
    import pandas as pd
    records = []
    skipped = 0
    for fp in file_paths:
        try:
            df = pd.read_parquet(fp, columns=['ts_code', 'trade_date', 'amount', 'turn'])
            if df.empty:
                skipped += 1
                continue
            valid = df.dropna(subset=['amount', 'turn'])
            valid = valid[valid['turn'] > 0]
            if valid.empty:
                skipped += 1
                continue
            valid = valid.copy()
            valid['total_mv'] = valid['amount'] * 100.0 / valid['turn']
            valid['trade_date'] = pd.to_datetime(valid['trade_date']).dt.normalize()
            records.append(valid[['trade_date', 'ts_code', 'total_mv']])
        except Exception:
            skipped += 1
    return records, skipped


# ═══════════ 分阶段止盈止损 v1.0 — StageTracker & StageManager ═══════════
# STAGE_CONFIG 在 initialize() 中从 config.yaml 读取构建 (见 staged_exit 配置段)
HARD_SL_SIGMA_DEFAULT = -6.0   # 兜底默认值 (配置缺失时使用)


@dataclass
class StageTracker:
    code: str
    entry_date: object   # datetime
    entry_adj: float      # 入场价（前复权）
    entry_raw: float      # 入场价（真实）
    entry_qty: float      # 当前股数（再投资/送股后可变）
    original_entry_qty: float  # 建仓时股数（固定基准，分阶段卖出按此计算）
    daily_vol: float       # 入场时 EWMA daily_vol
    hard_sl: float = -6.0
    stage1_done: bool = False
    stage2_done: bool = False
    stage3_done: bool = False
    stage4_done: bool = False
    s1_pt_hit: bool = False   # S1 由止盈触发（用于利润锁定SL上移）
    s2_pt_hit: bool = False   # S2 由止盈触发
    s3_pt_hit: bool = False   # S3 由止盈触发
    pt_after_sl: bool = False  # SL触发后PT下移(S2→1.5σ, S3→2σ)
    # [R6-③ 2026-09-10] 利润锁定/PT下移参数 (StageManager 注入, 默认与原硬编码等价)
    pl_sl_s1: float = 0.0
    pl_sl_s2: float = 1.0
    pl_sl_s3: float = 2.0
    pt_after_sl_s2: float = 1.5
    pt_after_sl_s3: float = 2.0
    cum_max_up: float = float('-inf')
    cum_max_dn: float = float('inf')

    def profit_lock_sl(self) -> float:
        """利润锁定 SL：PT 触发后 SL 上移（业界标准 breakeven/profit-locking）
        S3 PT → +2σ, S2 PT → +1σ, S1 PT → 0σ(breakeven), 无 PT → -inf(不覆盖)
        [R6-③ 2026-09-10] 上移阈值读 self.pl_sl_s1/s2/s3 (config 注入, 默认 0/1/2 不变)

        [2026-09-02 方案A] 阈值/比例统一读 self.STAGE_CONFIG (Strategy 层),
        StageTracker 不再持有 stage_cfg/_cfg (死代码已清理)。
        """
        if self.s3_pt_hit: return self.pl_sl_s3
        if self.s2_pt_hit: return self.pl_sl_s2
        if self.s1_pt_hit: return self.pl_sl_s1
        return float('-inf')


class StageManager:
    def __init__(self, stage_cfg=None, hard_sl=-6.0,
                 pl_sl_s1=0.0, pl_sl_s2=1.0, pl_sl_s3=2.0,
                 pt_after_sl_s2=1.5, pt_after_sl_s3=2.0):
        self.trackers: Dict[str, StageTracker] = {}
        self.freed_cash_pool: float = 0.0
        # [2026-09-02 方案A] stage_cfg 参数保留仅为接口兼容 (外部 StageManager(stage_cfg=...) 调用);
        #   实际阈值/比例由 Strategy 层 self.STAGE_CONFIG 直接读取, tracker 不再持有。
        self.stage_cfg = stage_cfg
        self.hard_sl = hard_sl
        # [R6-③ 2026-09-10] 利润锁定/PT下移参数 (Strategy 层 config 注入)
        self.pl_sl_s1 = pl_sl_s1
        self.pl_sl_s2 = pl_sl_s2
        self.pl_sl_s3 = pl_sl_s3
        self.pt_after_sl_s2 = pt_after_sl_s2
        self.pt_after_sl_s3 = pt_after_sl_s3

    def register(self, code, entry_date, entry_adj, entry_raw, qty, daily_vol):
        self.trackers[code] = StageTracker(
            code=code, entry_date=entry_date,
            entry_adj=entry_adj, entry_raw=entry_raw,
            entry_qty=qty, original_entry_qty=qty,  # 初始二者相同
            daily_vol=daily_vol,
            hard_sl=self.hard_sl,
            pl_sl_s1=self.pl_sl_s1, pl_sl_s2=self.pl_sl_s2, pl_sl_s3=self.pl_sl_s3,
            pt_after_sl_s2=self.pt_after_sl_s2, pt_after_sl_s3=self.pt_after_sl_s3)

    def reset(self):
        self.trackers.clear()
        self.freed_cash_pool = 0.0

    def adjust_qty(self, code: str, multiplier: float):
        """送股/转增后同步 qty（基准和当前都缩放）"""
        if code in self.trackers:
            self.trackers[code].entry_qty *= multiplier
            self.trackers[code].original_entry_qty *= multiplier
            # [R6-E3-fix-20260910] entry_adj **不得**随送转缩放 (原实现是"把 entry_adj 除以 multiplier", 已删)。
            #   根因: 价格源 (provider 复权序列, reference_date 固定) 在除权日**连续**, 不含该
            #   跳变 → entry_adj 与价格源必须同尺度, 否则阈值瞬间塌陷 → 幻象触发 + 假价成交:
            #   实证 sz.300963 2023-06-15 (10送5.09): ÷1.509 后出现 +57.8% 幻象 (σ_up=25.71)
            #   → 四档同日清仓, 成交价仅 0.67×市价 (97.7K 市值记账 67.4K)。
            #   qty 缩放保留 (真实价 × 新股数 配套; 买入成本/盈亏对比见 avg_cost)。


class Alpha191Strategy(IStrategy):
    """Alpha191 因子选股策略 - 元标注版本（集成基本面主模型）

    【改造说明】
    1. 修复中性化fillna问题（移除.fillna(0)，让XGBoost原生处理NaN）
    2. 将概率调整止盈止损改为基本面排名百分位调整止盈止损
    3. 保留时序划分训练和验证设计（客观评价，无需修改）
    4. 保留序贯采样权重设计（客观评价，无需修改）
    5. 两阶段模型合并为单阶段（直接预测强势盈利bin==2）
    """

    def __init__(self):
        # 【V7】第一层选股：直读 Wyckoff20-V10 recomd parquet 快照
        self._recomd_snapshot_dir = './data/wyckoff20_v10_recomd_snapshots'
        
    def initialize(self, context: StrategyContext):
        config = context.config
        self.data_provider = context.data_provider
        self.config = config
        self.stock_pool = context.stock_pool
        # [2026-08-07] 优先引擎回测结果目录 (context.output_dir), 回退 config
        # 修复: 此前一直用 config 默认 ./results → model_dir=./results/meta_models →
        #      cfi_importance/ 等分析产物落在 results 根而非回测目录
        self.output_dir = (getattr(context, 'output_dir', None)
                           or config.get('output_dir', './results'))
        self.index_code = config.get('index_code', 'all')
        self.local_data_path = config['data']['base_path']

        # [C2-20260905] 因子快照目录隔离: config data.factor_snapshot_dir 指定时,
        #   实例级覆盖 provider.snapshot_path (alpha191 → factor_snapshots_test,
        #   零框架改动; 该目录快照由 snapshot_fast.py (A1-A12 语义) 生成)
        _snap_dir = config.get('data', {}).get('factor_snapshot_dir')
        if _snap_dir and self.data_provider is not None:
            self.data_provider.snapshot_path = os.path.join(
                os.path.abspath(self.local_data_path), _snap_dir)
            os.makedirs(self.data_provider.snapshot_path, exist_ok=True)
            logger.info(f"[C2] 因子快照目录隔离: {self.data_provider.snapshot_path}")

        # ========== 策略参数 ==========
        self.REBALANCE_INTERVAL = config.get('rebalance_interval', 5)
        self.RETRAIN_INTERVAL = config.get('retrain_interval', 1)
        self.MIN_TRAIN_SAMPLES = config.get('min_train_sample', 100)
        self.TRAIN_WINDOW_DAYS = config.get('train_window_days', 50)
        self.VAL_SPLIT_RATIO = config.get('val_split_ratio', 0.1)
        self.AUC_THRESHOLD = config.get('auc_threshold', 0.55)
        # [L2] staged_rank 模式门控: 验证集 Spearman IC 阈值 (替代 AUC 门控)
        self.IC_GATE_THRESHOLD = float(config.get('ic_gate_threshold', 0.02))
        self.PSI_THRESHOLD = config.get('psi_threshold', 0.15)
        self.MIN_FEAT_IC = config.get('min_feat_ic', 0.03)
        self.NAN_THRESHOLD = config.get('nan_threshold', 0.30)
        self.MAX_CORR = config.get('max_corr', 0.85)
        self.MIN_FEATURES = config.get('min_features', 15)
        self.MAX_FEATURES = config.get('max_features', 100)
        self.THRESHOLD_SEARCH_POINTS = config.get('threshold_search_points', 50)
        self.USE_CALIBRATION = config.get('use_calibration', False)
        self.CALIBRATION_METHOD = config.get('meta_labeling', {}).get('calibration_method', 'none')  # P1-B

        # 特征选择 v2.0 配置
        fs_cfg = config.get('feature_selection', {})
        self.USE_VOTING = fs_cfg.get('use_voting', True)
        mda_cfg = fs_cfg.get('mda', {})
        self.MDA_ENABLED = mda_cfg.get('enabled', True)
        self.MDA_N_REPEATS = mda_cfg.get('n_repeats', 5)
        self.MDA_THRESHOLD = mda_cfg.get('threshold', 0.001)
        self.MDA_USE_STD_THRESHOLD = mda_cfg.get('use_std_threshold', False)
        self.MDA_T_ALPHA = mda_cfg.get('t_alpha', 0.05)
        # [P0-1] 折内 inner-OOS 段参数 (MDA 评估段, 验证块只评 AUC)
        self.MDA_INNER_OOS_FRAC = mda_cfg.get('inner_oos_frac', 0.15)
        self.MDA_INNER_OOS_MIN_DAYS = mda_cfg.get('inner_oos_min_days', 20)
        vote_cfg = fs_cfg.get('voting', {})
        self.VOTE_MIN_FOLDS = vote_cfg.get('vote_min_folds', 3)
        self.VOTE_MIN_FEATURES = vote_cfg.get('vote_min_features', 10)
        self.TOP_K_FEATURES = vote_cfg.get('top_k_features', 40)
        self.WEIGHT_IC = vote_cfg.get('weight_ic', 1.0)
        self.WEIGHT_MDI = vote_cfg.get('weight_mdi', 0.3)
        self.WEIGHT_MDA = vote_cfg.get('weight_mda', 1.5)
        self.FALLBACK_TO_IC_ONLY = vote_cfg.get('fallback_to_ic_only', True)
        clust_cfg = fs_cfg.get('feature_clustering', {})
        self.CLUSTER_ENABLED = clust_cfg.get('enabled', False)
        self.CLUSTER_METHOD = clust_cfg.get('method', 'hierarchical')
        self.CLUSTER_CORR_THRESHOLD = clust_cfg.get('corr_threshold', 0.70)
        self.CLUSTER_CORR_METHOD = clust_cfg.get('corr_method', 'spearman')
        self.CLUSTER_REPR_METHOD = clust_cfg.get('repr_method', 'score_max')
        self.CLUSTER_MIN_SIZE = clust_cfg.get('min_cluster_size', 2)
        self.CLUSTER_LOG = clust_cfg.get('log_clusters', True)
        self.CLUSTER_CFI_MODE = clust_cfg.get('cfi_mode', False)  # [CFI] 簇级置换重要性投票 (Prado MLAM Ch.6)
        self.CLUSTER_LEGACY_THRESHOLD = clust_cfg.get('legacy_threshold', False)  # [C组] legacy 阈值语义 (|corr|>=0.30归簇)
        self.CLUSTER_DIAGNOSE = clust_cfg.get('diagnose', False)  # [CFI-诊断] 生产模型训练后输出簇级重要性分布

        # XGBoost 参数
        self.XGB_N_ESTIMATORS = config.get('xgb_n_estimators', 300)
        self.XGB_MAX_DEPTH = config.get('xgb_max_depth', 5)
        self.XGB_LEARNING_RATE = config.get('xgb_learning_rate', 0.01)
        self.XGB_SUBSAMPLE = config.get('xgb_subsample', 0.8)
        self.XGB_COL_SAMPLE_BY_TREE = config.get('xgb_col_sample_by_tree', 0.8)
        self.XGB_REG_ALPHA = config.get('xgb_reg_alpha', 0.1)
        self.XGB_REG_LAMBDA = config.get('xgb_reg_lambda', 1.0)
        self.XGB_MIN_CHILD_WEIGHT = config.get('xgb_min_child_weight', 1)
        self.XGB_GAMMA = config.get('xgb_gamma', 0.00)
        self.XGB_RANDOM_STATE = config.get('xgb_random_state', 42)
        self.XGB_EARLY_STOPPING_ROUNDS = config.get('xgb_early_stopping_rounds', 50)
        self.XGB_EVAL_METRIC = config.get('xgb_eval_metric', 'auc')
        self.XGB_TREE_METHOD = config.get('xgb_tree_method', 'hist')
        self.XGB_DEVICE = config.get('xgb_device', 'cpu')

        # Alpha191 Score 体系配置
        score_cfg = config.get('alpha191_score', {})
        self._alpha191_score_enabled = score_cfg.get('enabled', False)
        self._score_tanh_amplitude = score_cfg.get('tanh_amplitude', 0.4)
        self._score_tanh_k = score_cfg.get('tanh_k', 3.0)
        self._score_oversold_threshold = score_cfg.get('oversold_threshold', 0.20)
        self._score_overbought_threshold = score_cfg.get('overbought_threshold', 0.80)
        self.ENSEMBLE_SEEDS = config.get('ensemble_seeds', [42])  # [Step2a] 预测用种子

        # Purged K-Fold CV 参数
        self.CV_FOLDS = config.get('cv_folds', 5)
        self.CV_EMBARGO_DAYS = config.get('cv_embargo_days', 6)
        self.CV_FEATURE_CONSENSUS = config.get('cv_feature_consensus', 3)
        self.CV_IC_PSI_SPLIT_RATIO = config.get('cv_ic_psi_split_ratio', 0.5)

        # 资金与风控参数
        self.INITIAL_CASH = config['backtest']['initial_cash']
        self.COMMISSION_RATE = config['execution'].get('commission_rate', 0.0003)
        self.TAX_RATE = config['execution'].get('tax_rate', 0.0005)
        self.SLIPPAGE = config['execution'].get('slippage', 0.002)
        self.MAX_POSITIONS = config['execution'].get('max_positions', 20)
        self.MAX_POSITION_PCT = config['execution'].get('max_position_pct', 0.10)
        self.BENCHMARK_CODE = config.get('benchmark_code', 'sh.000300')
        self.OPTIMIZE_MODE = config.get('optimize_mode', 1)
        self.RISK_PARITY_WINDOW = config.get('risk_parity_window', 20)

        # ----- 真实元标注标签器 -----
        meta_cfg = config.get('meta_labeling', {})
        # 【修正10】兼容嵌套和展平两种config格式
        if isinstance(meta_cfg, dict) and 'pt_mult' in meta_cfg:
            _pt_mult = meta_cfg.get('pt_mult', 2.0)
            _sl_mult = meta_cfg.get('sl_mult', 1.5)
            _max_hold = meta_cfg.get('max_hold_days', 5)
            _minute_freq = meta_cfg.get('minute_freq', '5')
            _daily_vol_window = meta_cfg.get('daily_vol_window', 20)
            _vol_multiplier = meta_cfg.get('vol_multiplier', 1.0)
            _vol_method = meta_cfg.get('vol_method', 'rolling')
            _vol_ewma_lambda = meta_cfg.get('vol_ewma_lambda', 0.94)
            _vol_ewma_seed_days = meta_cfg.get('vol_ewma_seed_days', 60)
        else:
            _pt_mult = config.get('pt_mult', 2.0)
            _sl_mult = config.get('sl_mult', 1.5)
            _max_hold = config.get('max_hold_days', 5)
            _minute_freq = config.get('minute_freq', '5')
            _daily_vol_window = config.get('daily_vol_window', 20)
            _vol_multiplier = config.get('vol_multiplier', 1.0)
            _vol_method = config.get('vol_method', 'rolling')
            _vol_ewma_lambda = config.get('vol_ewma_lambda', 0.94)
            _vol_ewma_seed_days = config.get('vol_ewma_seed_days', 60)
        self.use_meta_labeling = meta_cfg.get('enabled', True) if isinstance(meta_cfg, dict) else config.get(
            'meta_labeling_enabled', True)
        # [L1/L2] 标签模式: relative_barrier(旧三重屏障) | staged_binary(L1分阶段二分类) | staged_rank(L2分位回归)
        self._label_mode = meta_cfg.get('label_mode', 'relative_barrier') if isinstance(meta_cfg, dict) else 'relative_barrier'
        self._label_min_abs_ret = float(meta_cfg.get('label_min_abs_ret', 0.0)) if isinstance(meta_cfg, dict) else 0.0
        if self.use_meta_labeling:
            # 【问题3/4/5修复】不再传入 reference_date，
            # compute_labels 中使用 event_date 作为 reference_date
            # [L1] staged 模式: 从 config.staged_exit 读取与交易端完全同源的
            # 分阶段参数 (S1-S4 阈值/卖出比例/硬兜底), 传入标签器
            _staged_cfg = None
            _staged_hard_sl = -6.0
            _profit_lock = None  # [R6-③] staged 模式外保持 None (构造器默认等价)
            _pt_after_sl = None  # [R6-③]
            if self._label_mode in ('staged_binary', 'staged_rank'):
                _se = config.get('staged_exit', {})
                _defaults = {1: (2.0, -2.0, 0.25), 2: (3.0, -3.0, 0.35),
                             3: (4.0, -4.0, 0.20), 4: (9.0, -4.0, 0.20)}
                _staged_cfg = {}
                for _si in (1, 2, 3, 4):
                    _sc = _se.get(f'stage{_si}', {})
                    _d = _defaults[_si]
                    _staged_cfg[_si] = {
                        'pt': float(_sc.get('pt_sigma', _d[0])),
                        'sl': float(_sc.get('sl_sigma', _d[1])),
                        'pct': float(_sc.get('sell_pct', _d[2])),
                    }
                _staged_hard_sl = float(_se.get('hard_sl_sigma', -6.0))
                # [R6-③ 2026-09-10] 利润锁定/PT下移参数 (与交易端同源, config 单点定义; 默认值等价)
                _plk_lbl = _se.get('profit_lock', {}) or {}
                _profit_lock = {
                    's1_sl': float(_plk_lbl.get('s1_sl', 0.0)),
                    's2_sl': float(_plk_lbl.get('s2_sl', 1.0)),
                    's3_sl': float(_plk_lbl.get('s3_sl', 2.0)),
                }
                _pasl_lbl = _se.get('pt_after_sl', {}) or {}
                _pt_after_sl = {
                    's2_pt': float(_pasl_lbl.get('s2_pt', 1.5)),
                    's3_pt': float(_pasl_lbl.get('s3_pt', 2.0)),
                }
            self.meta_labeler = TripleBarrierLabeler(
                data_provider=self.data_provider,
                pt_mult=_pt_mult,
                sl_mult=_sl_mult,
                max_hold_days=_max_hold,
                minute_freq=_minute_freq,
                daily_vol_window=_daily_vol_window,
                vol_multiplier=_vol_multiplier,
                vol_method=_vol_method,
                vol_ewma_lambda=_vol_ewma_lambda,
                vol_ewma_seed_days=_vol_ewma_seed_days,
                staged_cfg=_staged_cfg,
                hard_sl=_staged_hard_sl,
                profit_lock=_profit_lock,
                pt_after_sl=_pt_after_sl
            )
            if self._label_mode in ('staged_binary', 'staged_rank'):
                logger.info(f"[L1/L2] 标签模式={self._label_mode}: 分阶段退出模拟 "
                            f"(S1={_staged_cfg[1]['pt']}σ/{_staged_cfg[1]['pct']:.0%}, "
                            f"S2={_staged_cfg[2]['pt']}σ/{_staged_cfg[2]['pct']:.0%}, "
                            f"S3={_staged_cfg[3]['pt']}σ/{_staged_cfg[3]['pct']:.0%}, "
                            f"S4={_staged_cfg[4]['pt']}σ/{_staged_cfg[4]['pct']:.0%}, "
                            f"hard={_staged_hard_sl}σ, min_abs_ret={self._label_min_abs_ret})")
        else:
            self.meta_labeler = None

        self.primary_recommendations = None

        # ----- 第一层选股快照 + 因子中性化辅助数据 -----
        self.fund_snapshot_dir = os.path.join(
            os.path.abspath(self.local_data_path),
            'wyckoff20_v5_snapshots'
        )

        # 行业映射缓存（因子中性化需要）
        self.industry_map = {}
        if config.get('neutralization', {}).get('enabled', True):
            # [v3.2 PIT fix] 仅使用回测开始日期之前的快照，防止未来行业重分类泄露
            start_date = config.get('backtest', {}).get('start_date', '20200101')
            self.industry_map = build_industry_map_from_snapshots(
                self.fund_snapshot_dir, as_of_date=start_date
            )
            logger.info(f"行业映射加载完成，共 {len(self.industry_map)} 只股票"
                        f" (使用 <= {start_date} 快照防前视偏差, 后续从BARRA快照动态补充)")
        self._train_ols_betas = None  # fit/transform: 训练时捕获OLS系数，预测时复用（对齐Wyckoff）

        # ----- 第一层选股: 读 Wyckoff V10 recomd 快照 (alpha191 生产设计保留) -----
        recomd_dir = os.path.join(
            os.path.abspath(self.local_data_path),
            'wyckoff20_v10_recomd_snapshots'
        )
        if not os.path.isdir(recomd_dir):
            raise RuntimeError(f"[V7] 快照目录不存在: {recomd_dir}")
        self._v7_recomd_dir = recomd_dir
        self._v7_snapshot_cache = {}
        self._v7_available_dates = set()
        for fn in os.listdir(recomd_dir):
            if fn.startswith('recomd_') and fn.endswith('.parquet'):
                self._v7_available_dates.add(fn.replace('recomd_', '').replace('.parquet', ''))
        logger.info(f"[V7] 加载 {len(self._v7_available_dates)} 个 V7 Wyckoff 选股快照")

        # ----- Alpha191 主模型 -----
        self.ensemble_manager = None
        self.feature_cols = []
        self.trade_day_seq = 0
        self.val_auc = 0.0           # 模型验证AUC, 门控和日志使用
        self.holdings = {}  # {code: {'date': datetime, 'price': float}}

        # ── 分阶段退出配置 (从 config.yaml [staged_exit] 段读取) ──
        se_cfg = config.get('staged_exit', {})
        self.staged_exit_enabled = se_cfg.get('enabled', True)
        self.STAGE_CONFIG = {
            1: dict(
                pt=float(se_cfg.get('stage1', {}).get('pt_sigma', 2.0)),
                sl=float(se_cfg.get('stage1', {}).get('sl_sigma', -2.0)),
                pct=float(se_cfg.get('stage1', {}).get('sell_pct', 0.25)),
                max_days=int(se_cfg.get('stage1', {}).get('max_hold_days', 5)),
            ),
            2: dict(
                pt=float(se_cfg.get('stage2', {}).get('pt_sigma', 3.0)),
                sl=float(se_cfg.get('stage2', {}).get('sl_sigma', -3.0)),
                pct=float(se_cfg.get('stage2', {}).get('sell_pct', 0.35)),
                max_days=int(se_cfg.get('stage2', {}).get('max_hold_days', 10)),
            ),
            3: dict(
                pt=float(se_cfg.get('stage3', {}).get('pt_sigma', 4.0)),
                sl=float(se_cfg.get('stage3', {}).get('sl_sigma', -4.0)),
                pct=float(se_cfg.get('stage3', {}).get('sell_pct', 0.5)),
                max_days=int(se_cfg.get('stage3', {}).get('max_hold_days', 15)),
            ),
            4: dict(
                pt=float(se_cfg.get('stage4', {}).get('pt_sigma', 9.0)),
                sl=float(se_cfg.get('stage4', {}).get('sl_sigma', -4.0)),
                pct=float(se_cfg.get('stage4', {}).get('sell_pct', 0.20)),
                max_days=int(se_cfg.get('stage4', {}).get('max_hold_days', 20)),
            ),
        }
        self.hard_sl_sigma = float(se_cfg.get('hard_sl_sigma', -6.0))
        self.reinvest_enabled = se_cfg.get('reinvest_enabled', True)
        self.reinvest_min_amount = float(se_cfg.get('reinvest_min_amount', 5000))
        logger.info("[v3.6] 分阶段退出: enabled=%s, S1=%.1fσ/%.0f%%, S2=%.1fσ/%.0f%%, S3=%.1fσ/%.0f%%, hard=%.1fσ, reinvest=%s",
                    self.staged_exit_enabled,
                    self.STAGE_CONFIG[1]['pt'], self.STAGE_CONFIG[1]['pct']*100,
                    self.STAGE_CONFIG[2]['pt'], self.STAGE_CONFIG[2]['pct']*100,
                    self.STAGE_CONFIG[3]['pt'], self.STAGE_CONFIG[3]['pct']*100,
                    self.hard_sl_sigma, self.reinvest_enabled)
        # [R6-③ 2026-09-10] 利润锁定/PT下移参数化 (config 单点定义, 默认值与原硬编码等价),
        #   与标签端 TripleBarrierLabeler 同源读取同一 config 段 (交易-标签对齐)
        _se_plk = (config.get('staged_exit', {}) or {})
        _plk_cfg = _se_plk.get('profit_lock', {}) or {}
        _pasl_cfg = _se_plk.get('pt_after_sl', {}) or {}
        self._pl_sl_s1 = float(_plk_cfg.get('s1_sl', 0.0))
        self._pl_sl_s2 = float(_plk_cfg.get('s2_sl', 1.0))
        self._pl_sl_s3 = float(_plk_cfg.get('s3_sl', 2.0))
        self._pt_after_sl_s2 = float(_pasl_cfg.get('s2_pt', 1.5))
        self._pt_after_sl_s3 = float(_pasl_cfg.get('s3_pt', 2.0))
        self.stage_mgr = StageManager(stage_cfg=self.STAGE_CONFIG,
                                       hard_sl=self.hard_sl_sigma,
                                       pl_sl_s1=self._pl_sl_s1, pl_sl_s2=self._pl_sl_s2,
                                       pl_sl_s3=self._pl_sl_s3,
                                       pt_after_sl_s2=self._pt_after_sl_s2,
                                       pt_after_sl_s3=self._pt_after_sl_s3)
        self.volatility_cache = {}  # {code: daily_vol}

        # ============================================================
        # [2026-08-29 方案A-1] 组合级回撤熔断 (drawdown breaker)
        #   设计: 三态 0=NORMAL(满仓建仓) 1=HALF(建仓减半) 2=FLAT(不建仓)
        #   触发: 净值回撤 > dd_flat → FLAT; > dd_half → HALF (保守优先)
        #   恢复: FLAT→HALF 用市场信号(指数≥MA 连续 trend_confirm_days 日)+基线重置
        #         [2026-09-02] 基线重置至 FLAT 触发净值 (原峰值重置至当前净值会赦免历史亏损)
        #         (空仓净值恒定, 净值恢复条件会死锁 — grid_lowvol 教训);
        #         HALF→NORMAL 用净值回撤收窄 < dd_recover_full (半仓净值会动, 可达)
        #   只影响"建仓规模", 不清仓现有持仓 (grid_lowvol 教训: 风控=限建仓非强清)
        # ============================================================
        dc_cfg = self.config.get('drawdown_breaker', {})
        self._dd_cfg = {
            'enabled': bool(dc_cfg.get('enabled', True)),
            'dd_half': float(dc_cfg.get('dd_half', 0.10)),
            'dd_flat': float(dc_cfg.get('dd_flat', 0.15)),
            'dd_recover_full': float(dc_cfg.get('dd_recover_full', 0.05)),
            'trend_confirm_days': int(dc_cfg.get('trend_confirm_days', 3)),
            # [2026-09-02] index_code 由顶层 risk_index_code 统一控制 (默认中证2000); 段内值保留为回退
            'index_code': str(config.get('risk_index_code') or dc_cfg.get('index_code') or 'sh000001'),
            'index_ma': int(dc_cfg.get('index_ma', 20)),
        }
        self._dd_state = 0              # 0=NORMAL 1=HALF 2=FLAT
        self._dd_scale = 1.0            # 建仓规模因子 (1.0/0.5/0.0)
        self._dd_above_days = 0         # 指数≥MA 连续天数 (FLAT 恢复确认)
        self._prev_nav = float(config.get('backtest', {}).get('initial_cash', 1e6))
        self._dd_peak = self._prev_nav
        # [2026-09-02 修复] FLAT 触发时的净值快照 (恢复基线):
        #   原实现 FLAT→HALF 恢复时 _dd_peak = 当前净值 → 赦免全部历史亏损 →
        #   熔断失效 (2026-07 回撤 -19.37% 实证, 详见 deliverables/回撤熔断峰值重置缺陷分析_20260902.md)。
        #   现改为重置到 FLAT 触发时净值: 接受已发生亏损, 但 FLAT 期间继续下跌
        #   纳入回撤计算, 熔断门槛持续有效。 (与 alpha191 同步修复)
        self._dd_flat_nav = None
        self._dd_index_close = self._load_breaker_index_close(self._dd_cfg['index_code'])

        # ============================================================
        # [2026-08-29 方案A-2] 调仓日趋势过滤 (rebalance trend gate)
        #   调仓日指数收盘 < MA(ma) → 建仓减半 (只影响当日建仓规模, 不强清)
        #   依据: 2024-01-22 调仓日满仓接盘 -6.4% (回测日志实证)
        # ============================================================
        tg_cfg = self.config.get('rebalance_trend_gate', {})
        self._trend_gate_cfg = {
            'enabled': bool(tg_cfg.get('enabled', True)),
            # [2026-09-02] 与回撤熔断共用 risk_index_code (默认中证2000); 段内值保留为回退
            'index_code': str(config.get('risk_index_code') or tg_cfg.get('index_code') or 'sh000001'),
            'ma': int(tg_cfg.get('ma', 10)),
            # [2026-09-01] 趋势×AUC 联合门控参数 (四档决策矩阵, 取代旧 half_position/require_ma_down)
            'auc_threshold_up': float(tg_cfg.get('auc_threshold_up', 0.50)),
            'auc_threshold_dn': float(tg_cfg.get('auc_threshold_dn', 0.54)),
            'scale_up_ok': float(tg_cfg.get('scale_up_ok', 1.0)),
            'scale_up_bad': float(tg_cfg.get('scale_up_bad', 0.5)),
            'scale_dn_ok': float(tg_cfg.get('scale_dn_ok', 0.5)),
            'scale_dn_bad': float(tg_cfg.get('scale_dn_bad', 0.0)),
        }
        self._trend_scale = 1.0         # 调仓日建仓因子 (0.5/1.0)

        self.target_stocks = []
        self.target_weights = {}
        self.force_retrain = True
        self.rebalance_interval = self.REBALANCE_INTERVAL
        # [重训间隔] 距上次训练已执行的调仓次数 (>= RETRAIN_INTERVAL 时重训, 期间沿用最近模型)
        self._rebalances_since_train = 0
        # 【BUG-A修复】从config读取reference_date，确保handle_bar和_check_pt_sl_minute使用正确值
        # [REF-AUTO-2026-09-10] 'auto'/缺省 → **回测结束日**；与 engine.resolve_reference_date
        #   共用同一解析 → 策略端与引擎/经纪商使用的基准日严格一致。
        #   （原实现被各配置硬编码 '20220908' 覆盖，与"默认为回测结束日"的设计意图不符）
        from gpsef.backtest.engine import resolve_reference_date as _resolve_ref_date
        self.reference_date = _resolve_ref_date(
            config.get('data') or {},
            str((config.get('backtest') or {}).get('end_date', '') or ''))

        # 【修正10】止盈止损参数直接存储，避免从meta_labeler读取不一致
        self._pt_mult = _pt_mult
        self._sl_mult = _sl_mult
        # 【改造2】基本面排名百分位缓存
        
        # ----- 缓存 -----
        self._factor_snapshot_cache = {}


        # ----- 冷启动标识 -----
        self._cold_start = True
        self._last_train_date = None

        # ----- V10上游信号门控 -----
        self._v10_gate_enabled = config.get('data', {}).get('v10_gate_enabled', False)
        self._v10_snapshot_dir = config.get('data', {}).get('recomd_snapshot_dir', './data/wyckoff20_v10_recomd_snapshots')
        self._v10_max_defer_days = config.get('data', {}).get('v10_gate_max_defer_days', 0)
        self._v10_deferred = False        # 当前是否处于顺延状态
        self._v10_defer_start_day = 0     # 开始顺延时的trade_day_seq

        # 构建/加载全市场市值缓存（按日计算并存储，避免训练时逐股读parquet）
        self._ensure_total_mv_cache()

        # ---- [v3.0] GARCH 仓位控制 ----
        gc_cfg = self.config.get('garch_control', {})
        self._garch_enabled = gc_cfg.get('enabled', True)
        if self._garch_enabled:
            self._garch_ctrl = GARCHPositionController(gc_cfg)
            logger.info(f"[v3.0] GARCH 波动率目标仓位控制: 已启用 "
                        f"(target={gc_cfg.get('vol_target', 0.15):.0%}, "
                        f"lookback={gc_cfg.get('lookback', 250)}d)")
        else:
            self._garch_ctrl = None
            logger.info("[v3.0] GARCH 波动率目标仓位控制: 已禁用")

        # ---- [2026-08-18] 因子验证三件套 (SHAP时序/ADV/CIC) 已完全移除 ----
        # 依据: 审查报告结论 (SHAP 0票空转/ADV冗余/CIC 86%误杀) + 用户决策
        # 粗筛职能由 ensemble 内 Purged K-Fold (低方差+IC+PSI+MDA投票) 承担

        # ---- [v3.0] BARRA 约束优化 ----
        barra_cfg = self.config.get('barra', {})
        self._barra_enabled = barra_cfg.get('enabled', True)
        self._barra_solve = None       # function reference
        self._barra_softmax = None     # function reference
        self._barra_build_alpha = None # function reference
        if self._barra_enabled:
            try:
                from strategies.wyckoff20.barra_constraint import (
                    solve_barra_weights, softmax_weights, build_alpha_from_preds
                )
                self._barra_cfg = barra_cfg
                self._barra_solve = solve_barra_weights           # store for handle_bar use
                self._barra_softmax = softmax_weights              # store for fallback
                self._barra_build_alpha = build_alpha_from_preds   # store for alpha construction
                logger.info(f"[v3.0] BARRA 约束优化: 已启用 "
                            f"(max_weight={barra_cfg.get('max_weight', 0.10)}, "
                            f"risk_aversion={barra_cfg.get('risk_aversion', 1.0)})")
            except ImportError as e:
                self._barra_cfg = None
                self._barra_enabled = False
                logger.warning(f"[v3.0] BARRA 约束优化: CVXOPT/V10 模块不可用 ({e})，已禁用")
        else:
            self._barra_cfg = None
            logger.info("[v3.0] BARRA 约束优化: 已禁用")

        # ---- [2026-08-18] 全市场日线预加载 + 市场条件缓存 已随三件套移除 ----
        # (原用途: ConditionalICAnalyzer 条件IC分析; 三件套移除后无消费方)

        # ==================== 初始化完成 ====================
        logger.info("=" * 60)
        logger.info("Alpha191 策略初始化完成")
        logger.info(f"元标注: {'启用' if self.use_meta_labeling else '禁用'}")
        logger.info(f"调仓周期: {self.rebalance_interval} 天")
        logger.info(f"最大持仓: {self.MAX_POSITIONS}")
        logger.info(f"PT/SL: pt_mult={self._pt_mult}, sl_mult={self._sl_mult} (统一波动率归一化)")
        logger.info(f"V10上游门控: {'启用' if self._v10_gate_enabled else '禁用'}, "
                    f"snapshot_dir={self._v10_snapshot_dir}, max_defer={self._v10_max_defer_days}d")
        logger.info("=" * 60)

    def _create_model_config(self):
        """创建模型配置（返回 GlobalConfig 对象，支持属性访问）

        【BUG-P修复】EnsembleModelManager 内部使用 config.AUC_THRESHOLD 属性访问，
        不能返回 dict，必须返回 GlobalConfig 对象。
        """
        cfg = GlobalConfig()
        cfg.XGB_N_ESTIMATORS = self.XGB_N_ESTIMATORS
        cfg.XGB_MAX_DEPTH = self.XGB_MAX_DEPTH
        cfg.XGB_LEARNING_RATE = self.XGB_LEARNING_RATE
        cfg.XGB_SUBSAMPLE = self.XGB_SUBSAMPLE
        cfg.XGB_COL_SAMPLE_BY_TREE = self.XGB_COL_SAMPLE_BY_TREE
        cfg.XGB_REG_ALPHA = self.XGB_REG_ALPHA
        cfg.XGB_REG_LAMBDA = self.XGB_REG_LAMBDA
        cfg.XGB_MIN_CHILD_WEIGHT = self.XGB_MIN_CHILD_WEIGHT
        cfg.XGB_GAMMA = self.XGB_GAMMA
        cfg.XGB_RANDOM_STATE = self.XGB_RANDOM_STATE
        cfg.XGB_EARLY_STOPPING_ROUNDS = self.XGB_EARLY_STOPPING_ROUNDS
        cfg.XGB_EVAL_METRIC = self.XGB_EVAL_METRIC
        cfg.XGB_TREE_METHOD = self.XGB_TREE_METHOD
        cfg.XGB_DEVICE = self.XGB_DEVICE
        cfg.ENSEMBLE_SEEDS = list(self.ENSEMBLE_SEEDS)     # [Step2a] 预测用种子
        cfg.MIN_FEAT_IC = self.MIN_FEAT_IC
        cfg.NAN_THRESHOLD = self.NAN_THRESHOLD
        cfg.MAX_CORR = self.MAX_CORR
        cfg.MIN_FEATURES = self.MIN_FEATURES
        cfg.MAX_FEATURES = self.MAX_FEATURES
        cfg.THRESHOLD_SEARCH_POINTS = self.THRESHOLD_SEARCH_POINTS
        cfg.USE_CALIBRATION = self.USE_CALIBRATION
        cfg.CALIBRATION_METHOD = self.CALIBRATION_METHOD  # P1-B
        cfg.AUC_THRESHOLD = self.AUC_THRESHOLD
        cfg.VAL_SPLIT_RATIO = self.VAL_SPLIT_RATIO

        # Purged K-Fold CV 参数
        cfg.cv_folds = self.CV_FOLDS
        cfg.cv_embargo_days = self.CV_EMBARGO_DAYS
        cfg.cv_feature_consensus = self.CV_FEATURE_CONSENSUS
        cfg.cv_ic_psi_split_ratio = self.CV_IC_PSI_SPLIT_RATIO
        cfg.PSI_THRESHOLD = self.PSI_THRESHOLD

        cfg.MIN_FEATURES = self.MIN_FEATURES  # 保守回退最少特征数

        # 特征选择 v2.0 参数
        cfg.use_voting = self.USE_VOTING
        cfg.mda_enabled = self.MDA_ENABLED
        cfg.mda_n_repeats = self.MDA_N_REPEATS
        cfg.mda_threshold = self.MDA_THRESHOLD
        cfg.mda_use_std_threshold = self.MDA_USE_STD_THRESHOLD
        cfg.mda_t_alpha = self.MDA_T_ALPHA
        cfg.mda_inner_oos_frac = self.MDA_INNER_OOS_FRAC   # [P0-1] inner-OOS 占比
        cfg.mda_inner_oos_min_days = self.MDA_INNER_OOS_MIN_DAYS  # [P0-1] inner-OOS 最小天数
        cfg.vote_min_folds = self.VOTE_MIN_FOLDS
        cfg.vote_min_features = self.VOTE_MIN_FEATURES
        cfg.top_k_features = self.TOP_K_FEATURES
        cfg.weight_ic = self.WEIGHT_IC
        cfg.weight_mdi = self.WEIGHT_MDI
        cfg.weight_mda = self.WEIGHT_MDA
        cfg.fallback_to_ic_only = self.FALLBACK_TO_IC_ONLY
        cfg.cluster_enabled = self.CLUSTER_ENABLED
        cfg.cluster_method = self.CLUSTER_METHOD
        cfg.cluster_corr_threshold = self.CLUSTER_CORR_THRESHOLD
        cfg.cluster_corr_method = self.CLUSTER_CORR_METHOD
        cfg.cluster_repr_method = self.CLUSTER_REPR_METHOD
        cfg.cluster_min_size = self.CLUSTER_MIN_SIZE
        cfg.cluster_log = self.CLUSTER_LOG
        cfg.cluster_cfi_mode = self.CLUSTER_CFI_MODE  # [CFI] 簇级置换重要性投票
        cfg.cluster_legacy_threshold = self.CLUSTER_LEGACY_THRESHOLD  # [C组] legacy 阈值语义
        cfg.cluster_diagnose = self.CLUSTER_DIAGNOSE  # [CFI-诊断] 簇级重要性分布输出
        # [v3.1 P1-4] 温度缩放: 放大 XGBoost 概率离散度, T=0.5 增加差异化
        cfg.PROB_TEMPERATURE = float(self.config.get('meta_labeling', {}).get('prob_temperature', 0.5))
        # [2026-08-05] LGBM 异质集成配置注入: config 的 lightgbm 段 → ensemble._lgb_enabled
        # (修复前 GlobalConfig 无 lightgbm 属性 → lgb_cfg={} → LGBM 从未启用)
        cfg.lightgbm = self.config.get('lightgbm', {})
        # [P0-0] 超参候选选择配置注入 (ensemble._select_hyperparams 读取)
        cfg.hyperparam_search = self.config.get('hyperparam_search', {})
        # [2026-08-18] 注入 MAX_HOLD_DAYS (HP-Search 选择-评估隔离用, 与生产 prod_val_size 同规则)
        cfg.MAX_HOLD_DAYS = getattr(self.meta_labeler, 'max_hold_days', 20)
        return cfg

    # ==================== 因子中性化（对齐 Wyckoff20 V5 管线） ====================

    @staticmethod
    def _winsorize(series: pd.Series, pct: tuple = (1, 99)) -> pd.Series:
        """单变量Winsorize缩尾"""
        valid = series.dropna()
        if len(valid) < 10:
            return series
        lower, upper = np.percentile(valid, pct)
        return series.clip(lower=lower, upper=upper)

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        """Z-score标准化"""
        m = series.mean()
        s = series.std()
        if s < 1e-8:
            return series - m
        return (series - m) / s

    def _build_neutral_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建中性化解释变量矩阵：行业哑变量 + 对数市值。
        对齐 Wyckoff20 V5 _build_neutral_X：
        - 行业按字母排序，消除 get_dummies 列序非确定性
        - .fillna(0).astype(np.float64) 确保 numpy lstsq 兼容
        """
        cfg = self.config.get('neutralization', {})
        if cfg.get('use_industry', True) and 'industry' in df.columns:
            categories = sorted(df['industry'].astype(str).unique())
            ind_dummies = pd.get_dummies(
                pd.Categorical(df['industry'].astype(str), categories=categories),
                prefix='ind'
            )
        else:
            ind_dummies = pd.DataFrame(index=df.index)

        if cfg.get('use_market_cap', True) and 'total_mv' in df.columns:
            log_mv = np.log(df['total_mv'].clip(lower=1e-8))
            log_mv.name = 'log_mv'
        else:
            log_mv = pd.Series(0.0, index=df.index, name='log_mv')

        X_neutral = pd.concat([ind_dummies, log_mv], axis=1).fillna(0).astype(np.float64)
        return X_neutral

    def _ensure_total_mv_cache(self):
        """确保市值缓存存在**且与原始数据输入一致**（派生缓存 = 输入的函数）。

        [FIX-DET4-2026-09-10 定稿 · 业界标准派生缓存设计]

        设计原则：缓存不是"真相"，只是 `raw_daily/*` 的**物化视图**：

            total_mv_cache = f(raw_daily/*)      # 纯函数、幂等

        由此得到三条性质（这正是"无论缓存如何更新，结果都应可复现"的成立条件）：

          ① **输入未变 → 零计算复用**：输入指纹一致则直接使用缓存，不重算不写盘；
          ② **输入变了 → 全量确定性重建**：重建内容只由当前输入决定，
             与"上一次缓存是什么/被谁怎么改过"**完全无关** ⇒
             **只要原始数据与过滤规则不变，重建后的缓存内容必然相同，结果必然可复现**；
          ③ **原子替换**（tmp + os.replace）：读者永远看到完整版本，不会读到半成品。

        ⚠️ 指纹中含 mtime，但它**只决定"要不要重算"**；因为重算是幂等的，
        mtime 抖动至多多做一次无用功，**绝不改变结果语义**。
        （反例，即 2026-09-09 事故：用 mtime 决定**缓存复用**且重算**非幂等**。）

        性能：指纹 = 5441 次 stat（毫秒级）；仅当 raw_daily 真变化时才全量重建（约 10-20s）。
        """
        import os, glob, json as _json, logging
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from multiprocessing import cpu_count
        from gpsef.data.derived_cache import files_fingerprint

        logger = logging.getLogger(__name__)

        cache_path = os.path.join(self.data_provider.base_path, 'total_mv_cache.parquet')
        meta_path = cache_path.replace('.parquet', '_meta.json')
        raw_path = self.data_provider.raw_daily_path
        # [FIX-DET-2026-09-10] 必须 sorted: glob 顺序依赖文件系统 → chunk 组成漂移
        all_files = sorted(glob.glob(os.path.join(raw_path, '*.parquet')))
        cur_fp = files_fingerprint(all_files)

        # ── ① 输入指纹一致 → 复用（零计算、零写盘）──
        if os.path.exists(cache_path) and os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = _json.load(f)
                if meta.get('fingerprint') == cur_fp and meta.get('n_rows'):
                    logger.info('[市值缓存] 输入指纹一致 → 复用缓存: %s 行 / %s 只 / %s~%s '
                                '(零计算, 未写盘; 指纹=%s)',
                                f"{meta['n_rows']:,}", meta.get('n_stocks'),
                                meta.get('min_date'), meta.get('max_date'), cur_fp[:8])
                    return
                logger.info('[市值缓存] 输入指纹已变化 (旧=%s 新=%s) → 全量确定性重建 '
                            '(重建结果只由当前输入决定, 与旧缓存状态无关)',
                            str(meta.get('fingerprint'))[:8], cur_fp[:8])
            except Exception as e:
                logger.warning('[市值缓存] 元数据读取失败 (%s) → 全量重建', e)
        elif os.path.exists(cache_path):
            logger.info('[市值缓存] 缺少输入指纹元数据 → 全量确定性重建一次以建立指纹')

        # ── ② 全量确定性重建 ──
        logger.info('[市值缓存] 开始并行构建全市场每日市值缓存 (全量, 确定性)...')
        # 过滤指数文件
        INDEX_PREFIXES = ('sh_000', 'sz_399')  # [Fix-20260908] sz_000=深市主板股票非指数(原误过滤461只)
        stock_files = [
            f for f in all_files
            if not os.path.basename(f).lower().startswith(INDEX_PREFIXES)
        ]
        logger.info(f'[市值缓存] 全市场 {len(stock_files)} 只股票（跳过 {len(all_files)-len(stock_files)} 个指数）')

        # 分批，避免单次并行开销过大
        n_workers = min(cpu_count(), 8)
        chunk_size = max(1, len(stock_files) // n_workers)
        chunks = [stock_files[i:i+chunk_size] for i in range(0, len(stock_files), chunk_size)]
        logger.info(f'[市值缓存] 并行构建: {n_workers} workers × {len(chunks)} chunks')

        all_records = []
        skipped = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_compute_total_mv_chunk, chunk): idx
                       for idx, chunk in enumerate(chunks)}
            for future in as_completed(futures):
                try:
                    records, n_skip = future.result()
                    all_records.extend(records)
                    skipped += n_skip
                except Exception as e:
                    logger.warning(f'[市值缓存] chunk {futures[future]} 失败: {e}')
                    skipped += len(chunks[futures[future]])

        if not all_records:
            logger.warning('[市值缓存] 无有效数据！中性化将回退到实时计算')
            return

        cache = pd.concat(all_records, ignore_index=True)
        # 确定性去重: 先按 (date, code, value) 排序, 再 keep='first'
        #   → 与并行完成顺序完全无关（同一 (date,code) 的值本就唯一）
        cache = cache.sort_values(['trade_date', 'ts_code', 'total_mv']).reset_index(drop=True)
        cache = cache.drop_duplicates(subset=['trade_date', 'ts_code'], keep='first')
        cache = cache.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)

        # ── ③ 原子替换（读者永远看到完整版本）──
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = f'{cache_path}.tmp_{os.getpid()}'
        cache.to_parquet(tmp_path, index=False, compression='snappy')
        os.replace(tmp_path, cache_path)

        _dates = pd.to_datetime(cache['trade_date'])
        with open(meta_path, 'w', encoding='utf-8') as f:
            _json.dump({
                'fingerprint': cur_fp,
                'n_rows': int(cache.shape[0]),
                'n_dates': int(_dates.nunique()),
                'n_stocks': int(cache['ts_code'].nunique()),
                'min_date': str(_dates.min().date()),
                'max_date': str(_dates.max().date()),
                'built_at': str(datetime.now()),
                'source': 'raw_daily/*.parquet  (total_mv = amount*100/turn, turn>0)',
                'idempotent': True,
            }, f, ensure_ascii=False)
        logger.info(
            f'[市值缓存] 全量重建完成并原子替换: {cache.shape[0]:,} 行, '
            f'{cache["ts_code"].nunique()} 只股票, {_dates.min().date()}~{_dates.max().date()}, '
            f'{skipped} 文件跳过, 文件大小 {os.path.getsize(cache_path)/1024/1024:.1f} MB, '
            f'指纹 {cur_fp[:8]}'
        )

    def _load_total_mv_cache(self) -> pd.DataFrame:
        """加载全市场市值缓存（首次加载后缓存在内存）"""
        if not hasattr(self, '_total_mv_cache_df') or self._total_mv_cache_df is None:
            cache_path = os.path.join(self.data_provider.base_path, 'total_mv_cache.parquet')
            if not os.path.exists(cache_path):
                return None
            self._total_mv_cache_df = pd.read_parquet(cache_path)
            logger = logging.getLogger('strategies.alpha191.strategy')
            logger.info(f'[市值缓存] 已加载: {self._total_mv_cache_df.shape[0]:,} 行')
        return self._total_mv_cache_df

    def _realtime_total_mv(self, code: str, date_values) -> pd.Series:
        """[2026-08-19] 缓存缺失股票的市值实时重算（与缓存构建相同逻辑）。

        与 _compute_total_mv_chunk 完全同源:
            total_mv = amount * 100 / turn  (raw_daily/{code}.parquet)
        日期语义与缓存路径一致: merge_asof(direction='backward') = 每行取 ≤date 最新市值。

        Returns:
            pd.Series: 与 date_values 对齐的 total_mv; 无数据/读取失败 → NaN。
        """
        import os
        logger = logging.getLogger('strategies.alpha191.strategy')
        dates = pd.to_datetime(pd.Series(list(date_values))).dt.normalize()
        safe = code.replace('.', '_')
        dfile = os.path.join(self.data_provider.raw_daily_path, f'{safe}.parquet')
        if not os.path.exists(dfile):
            logger.warning(f'[市值填充/实时重算] {code}: raw_daily 文件不存在, 保持 NaN(中位数兜底)')
            return pd.Series(np.nan, index=dates.index)
        try:
            raw = pd.read_parquet(dfile, columns=['ts_code', 'trade_date', 'amount', 'turn'])
            raw = raw.dropna(subset=['amount', 'turn'])
            raw = raw[raw['turn'] > 0]
            if raw.empty:
                logger.warning(f'[市值填充/实时重算] {code}: 无有效 amount/turn, 保持 NaN')
                return pd.Series(np.nan, index=dates.index)
            raw['total_mv'] = raw['amount'] * 100.0 / raw['turn']
            raw['trade_date'] = pd.to_datetime(raw['trade_date']).dt.normalize()
            cg = raw[['trade_date', 'total_mv']].sort_values('trade_date') \
                .drop_duplicates('trade_date', keep='last')
            left = pd.DataFrame({'_d': dates.values, '_i': np.arange(len(dates))})
            m = pd.merge_asof(left.sort_values('_d'), cg,
                              left_on='_d', right_on='trade_date', direction='backward')
            m = m.sort_values('_i').set_index('_i')
            return m['total_mv'].reset_index(drop=True)
        except Exception as e:
            logger.warning(f'[市值填充/实时重算] {code} 失败: {e}, 保持 NaN(中位数兜底)')
            return pd.Series(np.nan, index=dates.index)

    def _fill_total_mv_per_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """[市值时点统一修复 2026-08-05] 按 ≤snapshot_date 最新市值填充。

        与预测侧 _merge_fundamental_info(ref_date=t_minus_1) 完全一致（同一
        '≤date 最新' 语义）→ 训练/预测使用同一时点口径的市值。
        无未来: snapshot_date = T-1（特征时点），其市值 ≤ 标签开始日 T+1。
        实现: merge_asof(direction='backward') 等价于 groupby.last()，全向量化。
        """
        if 'total_mv' in df.columns and not df['total_mv'].isna().all():
            return df
        if 'ts_code' not in df.columns:
            return df
        # 优先用真实快照日期(snapshot_date=T-1); 无则回退 trade_date
        date_col = 'snapshot_date' if 'snapshot_date' in df.columns else 'trade_date'
        if date_col not in df.columns:
            return df

        logger = logging.getLogger('strategies.alpha191.strategy')
        cache = self._load_total_mv_cache()
        if cache is None:
            logger.warning('[市值填充] 缓存不存在，回退到实时计算（慢）')
            return self._fill_total_mv_fallback(df)

        n_before = len(df)
        df = df.copy()
        df['_mv_date'] = pd.to_datetime(df[date_col]).dt.normalize()
        cache_mv = cache.copy()
        cache_mv['trade_date'] = pd.to_datetime(cache_mv['trade_date']).dt.normalize()
        cache_mv = cache_mv.drop_duplicates(subset=['ts_code', 'trade_date'])

        # 逐 ts_code 组内 asof: 对每行取 cache 中 ≤_mv_date 的最新市值
        # （与预测侧 groupby.last 语义等价; 全局 by 版 merge_asof 在 pandas 3.0 大表不可靠）
        parts = []
        for code, g in df.groupby('ts_code'):
            cg = cache_mv[cache_mv['ts_code'] == code][['trade_date', 'total_mv']].sort_values('trade_date')
            if cg.empty:
                # [2026-08-19 增强] 缓存缺失的股票: 用与缓存构建相同逻辑实时重算
                #   (total_mv = amount*100/turn, ≤date 最新; 原实现直接 NaN → 中位数兜底,
                #   该股票特征中性化失真 — hs300 实测 33 只成分缺市值)
                g = g.copy()
                _dt = g[date_col] if date_col in g.columns else g['trade_date']
                g['total_mv'] = self._realtime_total_mv(
                    g['ts_code'].iloc[0], _dt).to_numpy()
                parts.append(g)
                continue
            # 去掉 left 侧的 total_mv 占位列, 避免 merge_asof 列冲突(否则 left 的 NaN 覆盖 right 值)
            g = g.drop(columns=['total_mv'], errors='ignore')
            m = pd.merge_asof(g.sort_values('_mv_date'), cg, left_on='_mv_date',
                              right_on='trade_date', direction='backward')
            # [2026-08-19 修复 hs300 全 0.5 bug] 在 part 层统一列名:
            #   merge_asof 把 left 的 trade_date(标签日) 自动改名 trade_date_x,
            #   right 的市值日期为 trade_date_y; 而 cg.empty 分支保留原名 'trade_date'。
            #   原实现: concat 后 merged 同时含 'trade_date_x' 与 'trade_date' 两列 →
            #   rename(trade_date_x→trade_date) 产生重复列 → .loc[:,~duplicated] 保留
            #   第一个(merge 分支列) → cg.empty 分支行(cache 缺该股票)的 trade_date 变 NaN
            #   (hs300 实测 7920 行 = 33 只缺市值股票 × 240 天 → 训练全链路 AUC=0.5 崩溃)
            if 'trade_date_x' in m.columns:
                m = m.rename(columns={'trade_date_x': 'trade_date'})
                m = m.drop(columns=['trade_date_y'], errors='ignore')
            parts.append(m)
        merged = pd.concat(parts, ignore_index=True)
        merged.drop(columns=['_mv_date'], inplace=True, errors='ignore')
        # merge_asof 列冲突: left 的 trade_date(标签日) 变 trade_date_x, right 的市值日期变 trade_date_y
        # 保留 left 的 trade_date（训练样本标签日），删除 right 的市值日期列
        if 'trade_date_x' in merged.columns:
            merged.rename(columns={'trade_date_x': 'trade_date'}, inplace=True)
            merged.drop(columns=['trade_date_y'], inplace=True, errors='ignore')
        elif 'trade_date_y' in merged.columns:
            merged.drop(columns=['trade_date_y'], inplace=True, errors='ignore')
        # [2026-08-11 修复] cg 空分支(cache 无该股票)保留原名 'trade_date',
        # 与 merge 组的 'trade_date_x' rename 后 concat 成重复列 → 去重
        # (重复列会使 df['trade_date'] 返回 DataFrame → groupby Grouper 非 1 维报错)
        merged = merged.loc[:, ~merged.columns.duplicated()].copy()
        n_matched = merged['total_mv'].notna().sum()

        # 缺失值用全局中位数填充
        median_mv = merged['total_mv'].median()
        if pd.notna(median_mv) and median_mv > 0:
            merged['total_mv'] = merged['total_mv'].fillna(median_mv)
        else:
            merged['total_mv'] = merged['total_mv'].fillna(1e10)

        logger.info(f'[市值填充] 匹配 {n_matched}/{n_before} 行 (≤snapshot_date 最新, 训练/预测同口径)')
        return merged
    def _fill_total_mv_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """回退方案：向量化批量读取并 merge（缓存不存在时使用，极少触发）

        将逐股逐日的嵌套循环替换为：批量读全部 parquet → concat → merge。
        """
        import os
        logger = logging.getLogger('strategies.alpha191.strategy')
        logger.warning('[市值填充/FALLBACK] 缓存缺失，使用向量化批量读取（约需5-15秒）')

        unique_codes = df['ts_code'].unique()
        raw_path = self.data_provider.raw_daily_path

        all_records = []
        for ts_code in unique_codes:
            safe = ts_code.replace('.', '_')
            dfile = os.path.join(raw_path, f'{safe}.parquet')
            if not os.path.exists(dfile):
                continue
            try:
                raw = pd.read_parquet(dfile, columns=['ts_code', 'trade_date', 'amount', 'turn'])
                raw = raw.dropna(subset=['amount', 'turn'])
                raw = raw[raw['turn'] > 0]
                if raw.empty:
                    continue
                raw['total_mv'] = raw['amount'] * 100.0 / raw['turn']
                raw['trade_date'] = pd.to_datetime(raw['trade_date']).dt.normalize()
                all_records.append(raw[['trade_date', 'ts_code', 'total_mv']])
            except Exception:
                pass

        if not all_records:
            logger.error('[市值填充/FALLBACK] 无法读取任何数据')
            df['total_mv'] = np.nan
            return df

        cache = pd.concat(all_records, ignore_index=True)
        # 向量化 merge
        df_dates = pd.to_datetime(df['trade_date']).dt.normalize()
        df = pd.merge(df, cache, left_on=['ts_code', df_dates],
                       right_on=['ts_code', 'trade_date'],
                       how='left', suffixes=('', '_fb'))
        if 'trade_date_fb' in df.columns:
            df = df.drop(columns=['trade_date_fb'])

        median_mv = df['total_mv'].median()
        if pd.notna(median_mv) and median_mv > 0:
            df['total_mv'] = df['total_mv'].fillna(median_mv)
        else:
            df['total_mv'] = df['total_mv'].fillna(1e10)
        return df

    def _neutralize_factors(self, df, factor_cols, ref_date=None):
        """
        因子中性化：Winsorize → Z-score → OLS残差回归。
        对齐 Wyckoff20 V5 _neutralize 管线：
        - Winsorize(1%/99%) + Z-score 预处理
        - fit/transform 分离：训练期 neutralize_fit 捕获 coeff，预测期 neutralize_apply 复用
        - 保留 _neu 列 NaN，由 XGBoost 原生处理
        """
        cfg = self.config.get('neutralization', {})
        if not cfg.get('enabled', True):
            return df

        df = df.copy()
        if df.empty:
            return df

        if ref_date is not None and ('total_mv' not in df.columns or df['total_mv'].isna().all()):
            df = self._merge_fundamental_info(df, ref_date)

        # Step 1+2: Winsorize + Z-score 预处理（截面按日分组，消除跨日信息混合）
        # 【性能优化 v2】向量化 Z-score：用 transform('mean'/'std'/'quantile') 替代 transform(lambda)
        # 输出与原始 _zscore(_winsorize(x)) 完全一致
        if 'trade_date' in df.columns:
            td = df['trade_date']
            for col in factor_cols:
                if col not in df.columns:
                    continue
                raw = df[col].replace([np.inf, -np.inf], np.nan)

                # 向量化 Winsorize: 按日分组计算 1%/99% 分位数并 clip
                valid_counts = raw.notna().groupby(td).transform('sum')
                lo = raw.groupby(td).transform('quantile', 0.01)
                hi = raw.groupby(td).transform('quantile', 0.99)
                # _winsorize: len(valid) < 10 时不缩尾（返回原值）
                lo = lo.where(valid_counts >= 10, -np.inf)
                hi = hi.where(valid_counts >= 10, np.inf)
                winsorized = raw.clip(lower=lo, upper=hi)

                # 向量化 Z-score: 按日分组 mean/std
                means = winsorized.groupby(td).transform('mean')
                stds = winsorized.groupby(td).transform('std')
                # _zscore: s < 1e-8 时返回 series - m（即除以1）
                safe_stds = stds.where(stds >= 1e-8, 1.0)
                df[col] = (winsorized - means) / safe_stds
        else:
            for col in factor_cols:
                if col not in df.columns:
                    continue
                raw = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = self._zscore(self._winsorize(raw))

        # Step 3: 构建解释变量矩阵
        X_neutral = self._build_neutral_X(df)

        # Step 4: fit/transform OLS残差回归
        framework = FactorNeutralizer()
        if self._train_ols_betas is None:
            # 训练期：拟合OLS并缓存系数
            df, betas_info = framework.neutralize_fit(df, factor_cols, X_neutral)
            self._train_ols_betas = betas_info
            logger.info(
                f"[neutralize/FIT] Captured OLS betas: "
                f"{len(betas_info['betas'])} factors x {len(betas_info['X_cols'])} X columns"
            )
        else:
            # 预测期：复用训练期系数（纯矩阵乘法，跨机器确定性）
            df = framework.neutralize_apply(df, factor_cols, X_neutral, self._train_ols_betas)

        return df

    # ==================== V7 Wyckoff 选股（直读预生成快照） ====================

    @staticmethod
    def _code_to_ts(code: str) -> str:
        """6位数字代码转 ts_code 格式: 0/3→sz.XXX, 6/8→sh.XXX"""
        if code.startswith(('0', '3')):
            return f'sz.{code}'
        elif code.startswith(('6', '8')):
            return f'sh.{code}'
        return code

    def _read_v7_snapshot(self, date_str: str):
        """[V7] 运行期/训练期第一层选股: 读 V10 Wyckoff recomd 快照。

        返回 (codes, pred_dict):
          - codes:     V10 推荐池 (pred 降序, 截断 candidate_pool_size, ts_code)
          - pred_dict: {code: V10 pred} (v10_pred_raw/rank 特征用)
        """
        if date_str in self._v7_snapshot_cache:
            return self._v7_snapshot_cache[date_str]
        fpath = os.path.join(self._v7_recomd_dir, f"recomd_{date_str}.parquet")
        if not os.path.exists(fpath):
            return None
        df = pd.read_parquet(fpath)
        if df.empty:
            return None
        df = df.sort_values("pred", ascending=False)
        max_pool = self.config.get("meta_labeling", {}).get("candidate_pool_size", 300)
        df_pool = df.head(max_pool)
        codes = [self._code_to_ts(str(c)) for c in df_pool["code"].tolist()]
        pred_dict = {self._code_to_ts(str(row["code"])): float(row["pred"])
                     for _, row in df_pool.iterrows()}
        self._v7_snapshot_cache[date_str] = (codes, pred_dict)
        logger.debug(f"[V7] {date_str} stocks={len(codes)}")
        return (codes, pred_dict)
    def _get_primary_recommendations(self, context: StrategyContext, max_n=None):
        """[V7] 运行期选股：从 V7 快照取股票。
        max_n=None 时返回全量（供元标注筛选用），
        max_n=数字时返回前N只（降级回退用）。
        """
        date_str = context.current_dt.strftime('%Y%m%d')
        result = self._read_v7_snapshot(date_str)
        if result is None:
            logger.warning(f'[V7] 快照缺失: {date_str}')
            return []
        codes, _ = result  # [Phase B1] 解包元组
        if max_n is None:
            return codes  # 全量，供元标注筛选
        return codes[:max_n]

    def _get_primary_recommendations_for_date_with_rank(self, sample_date: datetime, context: StrategyContext):
        """[V7] 训练期历史回放：返回全部 V7 股票 + 均匀 rank_pct。"""
        date_str = sample_date.strftime('%Y%m%d')
        result = self._read_v7_snapshot(date_str)
        if result is None:
            return [], {}
        codes, pred_dict = result
        return codes, pred_dict  # [Phase B1] 不再硬编码 0.5

    # ==================== 数据预加载 ====================

    def _get_previous_trading_day(self, current_date, context_or_calendar=None):
        """
        获取当前日期之前的最近交易日。

        Args:
            current_date: 当前日期
            context_or_calendar: StrategyContext 或 日历列表（可选）

        Returns:
            datetime: 前一交易日
        """
        # 兼容两种调用方式：传入 context 或传入 calendar 列表
        if context_or_calendar is not None:
            if hasattr(context_or_calendar, 'full_calendar'):
                # 传入的是 context 对象
                calendar = getattr(context_or_calendar, 'full_calendar', context_or_calendar.calendar)
            elif isinstance(context_or_calendar, (list, tuple)):
                # 传入的是日历列表
                calendar = context_or_calendar
            else:
                calendar = []
        else:
            calendar = getattr(self, 'calendar', [])

        if not calendar:
            raise ValueError(
                f"无法找到 {current_date.date() if hasattr(current_date, 'date') else current_date} 之前的交易日")

        current_ts = pd.Timestamp(current_date)
        for d in reversed(calendar):
            if d < current_ts:
                return d

        raise ValueError(f"无法找到 {current_date.date()} 之前的交易日")

    def _merge_fundamental_info(self, df, date):
        """补充 industry 和 total_mv 列用于因子中性化。

        BUG-6 修复: 使用 _load_full_daily_data（有缓存）替代逐只读 parquet，
        减少 I/O。turn 列仍需读原始 parquet（_load_full_daily_data 不含 turn）。
        """
        need_industry = 'industry' not in df.columns
        need_total_mv = 'total_mv' not in df.columns
        if not need_industry and not need_total_mv:
            return df
        code_digits = df['ts_code'].str.extract(r'(\d{6})', expand=False)
        if need_industry and self.industry_map:
            df['industry'] = code_digits.map(self.industry_map)
            # [v3.4 FIX] 对静态映射缺失的股票(如IPO新股)，从BARRA快照动态补充行业信息
            missing = df['industry'].isna()
            if missing.any():
                try:
                    from strategies.wyckoff20.barra_builder import get_or_build_snapshot
                    barra_snap = get_or_build_snapshot(date)
                    if barra_snap is not None and 'industry' in barra_snap.columns:
                        barra_ind = dict(zip(barra_snap['code'].astype(str).str.zfill(6),
                                             barra_snap['industry']))
                        df.loc[missing, 'industry'] = code_digits[missing].map(barra_ind)
                        n_filled = missing.sum() - df.loc[missing, 'industry'].isna().sum()
                        if n_filled > 0:
                            logger.debug(f'行业映射: 从BARRA快照补充{n_filled}只股票')
                except Exception:
                    pass
            df['industry'] = df['industry'].fillna('OTHER')
        if need_total_mv:
            cache = self._load_total_mv_cache()
            if cache is not None:
                # 从缓存取 <=date 的最新 total_mv
                date_ts = pd.Timestamp(date)
                mv_snapshot = (
                    cache[cache['trade_date'] <= date_ts]
                    .sort_values('trade_date')
                    .groupby('ts_code', as_index=False)
                    .last()[['ts_code', 'total_mv']]
                )
                df = pd.merge(df, mv_snapshot, on='ts_code', how='left', suffixes=('', '_mv'))
                # 清理 merge 产生的多余列（原 df 没有 total_mv，不需要处理后缀）
                if 'total_mv_mv' in df.columns:
                    df = df.drop(columns=['total_mv_mv'])
                if df['total_mv'].isna().all():
                    df['total_mv'] = np.nan
            else:
                df['total_mv'] = np.nan
        return df

    def _get_primary_recommendations_for_date(self, sample_date, context):
        """[V7] 训练期历史回放：返回全部 V7 选股。"""
        rec_codes, _ = self._get_primary_recommendations_for_date_with_rank(sample_date, context)
        return rec_codes

    # ==================== Alpha191 因子 Score 计算 ====================

    # 六种算子语义 → Score 模板的因子分类（基于算子层先验知识）
    _SCORE_FACTORS_TANH = [
        # 动量方向: 正值看多
        'alpha_014', 'alpha_015', 'alpha_018', 'alpha_020', 'alpha_021',
        'alpha_024', 'alpha_027', 'alpha_029', 'alpha_088',
        # 效率/强度: 高值看多
        'alpha_009', 'alpha_013', 'alpha_069', 'alpha_080',
        # VWAP/量价: 溢价看多
        'alpha_006', 'alpha_008', 'alpha_013', 'alpha_017',
        # 量价背离(取反因子): 高值=背离, 看多(修复)
        'alpha_001', 'alpha_005', 'alpha_016', 'alpha_032', 'alpha_042',
    ]

    _SCORE_FACTORS_OVERSOLD_BULLISH = [
        # 价格位置类: 极端低位=超卖反弹
        'alpha_031', 'alpha_034', 'alpha_046', 'alpha_047', 'alpha_057',
        'alpha_065', 'alpha_066',
    ]

    _SCORE_FACTORS_OVERBOUGHT_BEARISH = [
        # 价格位置类: 极端高位=超买回落
        'alpha_072', 'alpha_082',
    ]

    _SCORE_FACTORS_DOMINANCE = [
        # 趋势主导度: 高值=买方主导看多
        'alpha_023', 'alpha_040', 'alpha_053', 'alpha_058',
        'alpha_063', 'alpha_067', 'alpha_079',
    ]

    _SCORE_FACTORS_ACCUMULATION = [
        # 资金流累积: 正值=买入主导看多
        'alpha_011', 'alpha_043', 'alpha_060', 'alpha_084', 'alpha_094',
    ]

    _SCORE_FACTORS_BREAKOUT = [
        # 突破信号
        'alpha_004', 'alpha_038', 'alpha_086',
    ]

    def _compute_alpha191_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对已中性化的 `_neu` 列计算 Score 和 Confidence。

        核心设计:
          - Score 在 `_neu`（已去行业+市值偏差）上计算，避免双重中性化
          - 5种模板覆盖约60个因子，按算子语义分类
          - Conf = 截面有效率(非NaN占比)，反映因子在当前快照中的可用性
          - 不做截面的 Winsorize/Sigmoid（由后续 _neutralize_factors 处理）

        返回: DataFrame with added {factor}_score and {factor}_conf columns
        """
        if not self._alpha191_score_enabled:
            return df

        df = df.copy()
        n_stocks = len(df)
        if n_stocks < 10:
            return df

        # ---------- 模板1: tanh 连续映射 ----------
        for fname in self._SCORE_FACTORS_TANH:
            neu_col = f'{fname}_neu'
            score_col = f'{fname}_score'
            conf_col = f'{fname}_conf'
            if neu_col not in df.columns:
                continue

            series = df[neu_col].replace([np.inf, -np.inf], np.nan)
            valid = series.dropna()
            if len(valid) < 10:
                df[score_col] = 0.5
                df[conf_col] = len(valid) / max(n_stocks, 1)
                continue

            std = valid.std()
            if std < 1e-8:
                df[score_col] = 0.5
            else:
                z = np.clip(series / (self._score_tanh_k * std), -4, 4)
                df[score_col] = np.clip(0.5 + self._score_tanh_amplitude * np.tanh(z), 0, 1)

            df[conf_col] = len(valid) / n_stocks

        # ---------- 模板2: 超卖看多 (极端低位→高分) ----------
        for fname in self._SCORE_FACTORS_OVERSOLD_BULLISH:
            neu_col = f'{fname}_neu'
            score_col = f'{fname}_score'
            conf_col = f'{fname}_conf'
            if neu_col not in df.columns:
                continue

            series = df[neu_col].replace([np.inf, -np.inf], np.nan)
            valid = series.dropna()
            if len(valid) < 10:
                df[score_col] = 0.5
                df[conf_col] = len(valid) / max(n_stocks, 1)
                continue

            q_low = valid.quantile(self._score_oversold_threshold)
            q_mid_low = valid.quantile(0.40)
            q_mid_high = valid.quantile(0.60)

            scores = pd.Series(0.5, index=series.index)
            scores.loc[series <= q_low] = np.clip(
                0.5 + 0.35 * (q_low - series.loc[series <= q_low]) / (q_low - valid.min() + 1e-8), 0.5, 0.95)
            scores.loc[(series > q_low) & (series <= q_mid_low)] = 0.60
            scores.loc[(series > q_mid_high)] = 0.35

            df[score_col] = scores
            df[conf_col] = len(valid) / n_stocks

        # ---------- 模板3: 超买看空 (极端高位→低分) ----------
        for fname in self._SCORE_FACTORS_OVERBOUGHT_BEARISH:
            neu_col = f'{fname}_neu'
            score_col = f'{fname}_score'
            conf_col = f'{fname}_conf'
            if neu_col not in df.columns:
                continue

            series = df[neu_col].replace([np.inf, -np.inf], np.nan)
            valid = series.dropna()
            if len(valid) < 10:
                df[score_col] = 0.5
                df[conf_col] = len(valid) / max(n_stocks, 1)
                continue

            q_high = valid.quantile(self._score_overbought_threshold)
            q_mid_low = valid.quantile(0.40)
            scores = pd.Series(0.5, index=series.index)
            scores.loc[series >= q_high] = np.clip(
                0.5 - 0.35 * (series.loc[series >= q_high] - q_high) / (valid.max() - q_high + 1e-8), 0.05, 0.5)
            scores.loc[(series < q_high) & (series > q_mid_low)] = 0.40

            df[score_col] = scores
            df[conf_col] = len(valid) / n_stocks

        # ---------- 模板4: 主导度 (高值=看多) ----------
        for fname in self._SCORE_FACTORS_DOMINANCE:
            neu_col = f'{fname}_neu'
            score_col = f'{fname}_score'
            conf_col = f'{fname}_conf'
            if neu_col not in df.columns:
                continue

            series = df[neu_col].replace([np.inf, -np.inf], np.nan)
            valid = series.dropna()
            if len(valid) < 10:
                df[score_col] = 0.5
                df[conf_col] = len(valid) / max(n_stocks, 1)
                continue

            q_high = valid.quantile(0.70)
            q_low = valid.quantile(0.30)
            scores = pd.Series(0.5, index=series.index)
            scores.loc[series >= q_high] = 0.78
            scores.loc[(series > valid.median())] = 0.64
            scores.loc[series <= q_low] = 0.22

            df[score_col] = scores
            df[conf_col] = len(valid) / n_stocks

        # ---------- 模板5: 资金流累积 ----------
        for fname in self._SCORE_FACTORS_ACCUMULATION:
            neu_col = f'{fname}_neu'
            score_col = f'{fname}_score'
            conf_col = f'{fname}_conf'
            if neu_col not in df.columns:
                continue

            series = df[neu_col].replace([np.inf, -np.inf], np.nan)
            valid = series.dropna()
            if len(valid) < 10:
                df[score_col] = 0.5
                df[conf_col] = len(valid) / max(n_stocks, 1)
                continue

            std = valid.std()
            if std < 1e-8:
                df[score_col] = 0.5
            else:
                z = np.clip(series / std, -3, 3)
                df[score_col] = np.clip(0.5 + 0.4 * np.tanh(z), 0, 1)

            df[conf_col] = len(valid) / n_stocks

        # ---------- 模板6: 突破信号 (条件判断) ----------
        for fname in self._SCORE_FACTORS_BREAKOUT:
            neu_col = f'{fname}_neu'
            score_col = f'{fname}_score'
            conf_col = f'{fname}_conf'
            if neu_col not in df.columns:
                continue

            series = df[neu_col].replace([np.inf, -np.inf], np.nan)
            valid = series.dropna()
            if len(valid) < 10:
                df[score_col] = 0.5
                df[conf_col] = len(valid) / max(n_stocks, 1)
                continue

            q_high = valid.quantile(0.80)
            q_low = valid.quantile(0.20)
            scores = pd.Series(0.5, index=series.index)
            scores.loc[series >= q_high] = 0.75
            scores.loc[series <= q_low] = 0.30

            df[score_col] = scores
            df[conf_col] = len(valid) / n_stocks

        return df

    # ==================== Alpha191 因子计算 ====================

    def _get_alpha_factors(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """计算 Alpha191 因子"""
        df = df_daily.copy()

        alpha_list = []

        # ---------- 短周期因子 ----------
        for lag in [1, 3, 5]:
            col = f'close_Lag{lag}'
            if col in df.columns:
                df[f'alpha_ret_{lag}d'] = df['close'] / df[col] - 1
                alpha_list.append(f'alpha_ret_{lag}d')

        # ---------- 波动率因子 ----------
        for window in [5, 10, 20]:
            col = f'close_W{window}'
            if col in df.columns:
                ret = np.log(df['close'] / df[col])
                df[f'alpha_vol_W{window}'] = ret.rolling(window, min_periods=window // 2).std()
                alpha_list.append(f'alpha_vol_W{window}')

        # ---------- 量价相关性 ----------
        if 'volume' in df.columns and 'close' in df.columns:
            for window in [5, 10]:
                vol_ma = df['volume'].rolling(window, min_periods=window // 2).mean()
                close_ma = df['close'].rolling(window, min_periods=window // 2).mean()
                corr = df['volume'].rolling(window, min_periods=window // 2).corr(df['close'])
                df[f'alpha_vp_corr_W{window}'] = corr
                alpha_list.append(f'alpha_vp_corr_W{window}')

        # ---------- 动量因子 ----------
        for period in [5, 10, 20]:
            col = f'close_W{period}'
            if col in df.columns:
                df[f'alpha_momentum_W{period}'] = df['close'] / df[col] - 1
                alpha_list.append(f'alpha_momentum_W{period}')

        # ---------- 趋势因子 ----------
        for period in [5, 10, 20]:
            close_ma = df['close'].rolling(period, min_periods=period // 2).mean()
            df[f'alpha_trend_W{period}'] = df['close'] / close_ma - 1
            alpha_list.append(f'alpha_trend_W{period}')

        # ---------- 波动率比值 ----------
        for w1, w2 in [(5, 20), (10, 20)]:
            c1 = f'alpha_vol_W{w1}'
            c2 = f'alpha_vol_W{w2}'
            if c1 in df.columns and c2 in df.columns:
                df[f'alpha_vol_ratio_W{w1}_{w2}'] = df[c1] / (df[c2] + 1e-8)
                alpha_list.append(f'alpha_vol_ratio_W{w1}_{w2}')

        # ---------- 行业中性化（如果有行业信息） ----------
        if 'industry' in df.columns:
            industry_dummies = pd.get_dummies(df['industry'], prefix='ind')
            for alpha_col in alpha_list[:]:
                if alpha_col in df.columns:
                    try:
                        X_ind = industry_dummies.values
                        X_alpha = df[[alpha_col]].values
                        mask = ~np.isnan(X_alpha.flatten())
                        if mask.sum() > 10:
                            reg = LinearRegression()
                            reg.fit(X_ind[mask], X_alpha[mask])
                            df[alpha_col + '_indneutral'] = df[alpha_col] - reg.predict(X_ind)
                            alpha_list.append(alpha_col + '_indneutral')
                    except Exception:
                        pass

        return df, alpha_list

    # ==================== 日内行为特征 (R5 IPBS) ====================
    # 【R5】IPBS: Intraday Price Behavior Score
    #   三特征：CPR_5（收盘位置）、VWAP_Dev_5（成本偏离）、DistrRatio_5（派发比率）
    #   信号来源：OHLCV日内结构 — 与模型close-only因子正交

    def _compute_ipbs(self, code, t_minus_1, context):
        """【R5】计算单只股票的IPBS（Intraday Price Behavior Score）

        三个日内行为特征：
          1. CPR_5: 5日平均收盘位置 = mean((C-L)/(H-L), 5d)
          2. VWAP_Dev_5: 收盘价vs5日成交量加权均价偏差
          3. DistrRatio_5: 近5日派发日/(派发日+吸筹日)比率

        OHLCV结构信息与close-only因子正交——知道了日收益率不能推断日内收盘位置。

        Returns:
            dict: {'cpr_5','vwap_dev_5','distr_ratio_5','raw_ipbs'} 或 None(数据不足)
        """
        tech_cfg = self.config.get('tech_filter', {})
        cpr_window = tech_cfg.get('cpr_window', 5)
        vwap_window = tech_cfg.get('vwap_window', 5)
        vol_thresh = tech_cfg.get('distr_volume_threshold', 1.2)
        vol_ma_window = tech_cfg.get('distr_volume_ma_window', 20)

        # 需要至少 max(cpr_window, vwap_window, vol_ma_window) + 5 个交易日
        lookback = max(cpr_window, vwap_window, vol_ma_window) + 5
        lookback_start = t_minus_1 - timedelta(days=int(lookback * 1.5) + 10)

        try:
            df = self.data_provider.get_daily_data(
                code, lookback_start, t_minus_1,
                reference_date=self.reference_date
            )
        except Exception as e:
            logger.warning(f"[R5-IPBS] {code} 数据获取失败: {e}")
            return None

        if df.empty or len(df) < lookback:
            logger.warning(f"[R5-IPBS] {code} 数据不足({len(df)}行)")
            return None

        df = df.sort_values('trade_date').reset_index(drop=True)

        # ====== 特征1: CPR_5 = 5日平均收盘位置 ======
        # CPR → 1: 收盘价接近最高价（买方主导收盘）
        # CPR → 0: 收盘价接近最低价（卖方主导收盘）
        df['cpr'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        cpr_5 = df['cpr'].iloc[-cpr_window:].mean()

        # ====== 特征2: VWAP_Dev_5 = 收盘价vs5日量加权均价偏差 ======
        # >0: 价格高于近期平均交易成本 → 近期买入者盈利 → 卖压低
        # <0: 价格低于近期平均交易成本 → 近期买入者被套 → 止损卖压高
        recent = df.iloc[-vwap_window:]
        vwap_5 = (recent['close'] * recent['volume']).sum() / (recent['volume'].sum() + 1e-8)
        vwap_dev_5 = (df['close'].iloc[-1] - vwap_5) / (vwap_5 + 1e-8)

        # ====== 特征3: DistrRatio_5 = 近5日派发日比率 ======
        # 派发日(Distribution): Close<Open AND Volume>1.2×MA(Vol,20) → 机构出货
        # 吸筹日(Accumulation): Close>Open AND Volume>1.2×MA(Vol,20) → 机构建仓
        # DistrRatio → 1: 派发主导; → 0: 吸筹主导
        vol_ma = df['volume'].rolling(vol_ma_window, min_periods=vol_ma_window // 2).mean()
        is_high_vol = df['volume'] > vol_ma * vol_thresh
        is_down = df['close'] < df['open']
        is_up = df['close'] > df['open']

        recent_mask = df.index[-cpr_window:]
        n_dist = (is_high_vol.loc[recent_mask] & is_down.loc[recent_mask]).sum()
        n_accum = (is_high_vol.loc[recent_mask] & is_up.loc[recent_mask]).sum()
        distr_ratio_5 = n_dist / (n_dist + n_accum + 1e-8)

        # 原始分（未排名，用于日志）
        # vwap_dev乘10缩放到与CPR/DistrRatio类似的[0,1]范围
        # （vwap_dev典型值±0.05，×10后±0.5，1+后[0.5, 1.5]，与CPR同量级）
        raw_ipbs = (cpr_5 + (1 - distr_ratio_5) + min(max(1 + vwap_dev_5 * 10, 0), 2)) / 3

        return {
            'cpr_5': cpr_5,
            'vwap_dev_5': vwap_dev_5,
            'distr_ratio_5': distr_ratio_5,
            'raw_ipbs': raw_ipbs,
        }

    def _check_ipbs_exit(self, code, t_minus_1, context):
        """【R5】持仓IPBS退出检查（三重确认）

        三条件全部满足才退出：
          ① CPR_5 < threshold (默认0.30)    → 近5日持续收盘偏低
          ② DistrRatio_5 > threshold (默认0.70) → 近5日以派发日为主
          ③ VWAP_Dev_5 < threshold (默认-0.01) → 价格低于近期成本1%以上

        Returns:
            (should_exit: bool, reason: str or None)
        """
        tech_cfg = self.config.get('tech_filter', {})

        result = self._compute_ipbs(code, t_minus_1, context)
        if result is None:
            return False, None

        cpr_thresh = tech_cfg.get('ipbs_exit_cpr_threshold', 0.30)
        distr_thresh = tech_cfg.get('ipbs_exit_distr_threshold', 0.70)
        vwap_thresh = tech_cfg.get('ipbs_exit_vwap_threshold', -0.01)

        cond_cpr = result['cpr_5'] < cpr_thresh
        cond_distr = result['distr_ratio_5'] > distr_thresh
        cond_vwap = result['vwap_dev_5'] < vwap_thresh

        if cond_cpr and cond_distr and cond_vwap:
            reason = (
                f"IPBS-退出(CPR5={result['cpr_5']:.3f}<{cpr_thresh}, "
                f"DistrR={result['distr_ratio_5']:.3f}>{distr_thresh}, "
                f"VWAPDev={result['vwap_dev_5']:.4f}<{vwap_thresh})"
            )
            return True, reason

        return False, None

    # ==================== IPBS软重排 ====================

    def _ipbs_soft_rerank(self, selected_codes, prob_dict, context, tech_cfg):
        """【R5】IPBS软重排核心逻辑

        流程：
        1. 取元标注top-extended_pool_size（默认30）作为扩池
        2. 计算扩池内每只股票的IPBS
        3. 横截面排名归一化
        4. 软重排：adjusted_prob = prob × (1 + λ × IPBS_rank)
        5. 取调整后top-20

        关键性质：λ=0.15确保模型概率始终主导，技术评分仅在边际区间(排名15-30)翻盘。
        """
        t_minus_1 = self._get_previous_trading_day(context.current_dt, context)
        extended_pool_size = tech_cfg.get('extended_pool_size', 30)
        ipbs_lambda = tech_cfg.get('ipbs_lambda', 0.15)
        max_positions = self.config.get('execution', {}).get('max_positions', 20)

        # Step 1: 扩池
        pool = selected_codes
        pool_size = min(extended_pool_size, len(pool))
        pool = pool[:pool_size]

        # Step 2: 计算IPBS
        ipbs_data = {}
        for code in pool:
            result = self._compute_ipbs(code, t_minus_1, context)
            if result is not None:
                ipbs_data[code] = result

        if len(ipbs_data) < max_positions:
            logger.warning(f"[R5-IPBS] 有效IPBS数据不足({len(ipbs_data)}只), 跳过重排")
            return selected_codes[:max_positions], prob_dict

        # Step 3: 横截面排名归一化
        codes_with_ipbs = list(ipbs_data.keys())
        cpr_ranks = pd.Series([ipbs_data[c]['cpr_5'] for c in codes_with_ipbs],
                              index=codes_with_ipbs).rank(pct=True)
        distr_ranks = pd.Series([1 - ipbs_data[c]['distr_ratio_5'] for c in codes_with_ipbs],
                                index=codes_with_ipbs).rank(pct=True)
        vwap_ranks = pd.Series([ipbs_data[c]['vwap_dev_5'] for c in codes_with_ipbs],
                               index=codes_with_ipbs).rank(pct=True)

        ipbs_rank = (cpr_ranks + distr_ranks + vwap_ranks) / 3  # [0, 1]

        # Step 4: 软重排 adjusted_prob = prob × (1 + λ × IPBS)
        adjusted_prob = {}
        for code in pool:
            base_prob = prob_dict.get(code, 0.5)
            tech_score = ipbs_rank.get(code, 0.5)
            adjusted_prob[code] = base_prob * (1 + ipbs_lambda * tech_score)

        # Step 5: 按调整后概率排序，取top-max_positions
        sorted_codes = sorted(pool, key=lambda c: adjusted_prob.get(c, 0), reverse=True)
        final_codes = sorted_codes[:max_positions]
        final_prob = {c: prob_dict.get(c, 0.5) for c in final_codes}

        # 日志
        n_changed = len(set(selected_codes[:max_positions]) - set(final_codes))
        if n_changed > 0:
            logger.info(f"[R5-IPBS] 软重排: {n_changed}只股票被替换")
            old_top = set(selected_codes[:max_positions])
            new_top = set(final_codes)
            for code in old_top - new_top:
                logger.info(f"  降级 {code}: prob={prob_dict.get(code, 0):.3f}, "
                           f"IPBS={ipbs_rank.get(code, 0.5):.3f}")
            for code in new_top - old_top:
                logger.info(f"  升级 {code}: prob={prob_dict.get(code, 0):.3f}, "
                           f"IPBS={ipbs_rank.get(code, 0.5):.3f}")
        else:
            logger.info(f"[R5-IPBS] 软重排: 排名无变化")

        return final_codes, final_prob

    def _pre_market_tech_filter(self, selected_codes, prob_dict, context):
        """【R5】盘前技术面筛选入口：IPBS软重排

        根据 config.method 分派：
          - "ipbs_soft_rerank" → IPBS软重排（默认）
          - 其他 → 直接返回（兼容回退）
        """
        tech_cfg = self.config.get('tech_filter', {})
        if not tech_cfg.get('enabled', True):
            return selected_codes, prob_dict

        method = tech_cfg.get('method', 'ipbs_soft_rerank')
        if method == 'ipbs_soft_rerank':
            return self._ipbs_soft_rerank(selected_codes, prob_dict, context, tech_cfg)
        else:
            return selected_codes, prob_dict

    # ==================== 上下文特征（P1-1） ====================

    def _meta_select_stocks(self, recommended, context):
        """
        【改造5】使用元标注模型进行二分类选股：直接预测盈利概率 P(profit)。
        移除固定概率阈值，改为按概率排序取 top MAX_POSITIONS。

        【BUG-O修复】恢复原版选股流程：
        - 使用 data_provider.get_factor_snapshot(t_minus_1) 获取因子快照
        - 先中性化生成 _neu 列，再使用 ensemble_manager.selected_features + predict_proba
        - 不再使用 self._get_cached_factor_snapshot + self.ensemble_manager.predict
        """
        if not recommended:
            logger.warning("[_meta_select_stocks] 推荐股票列表为空")
            return []

        t_minus_1 = self._get_previous_trading_day(context.current_dt, context)

        features_df = context.data_provider.get_factor_snapshot(t_minus_1)

        if features_df is None or features_df.empty:
            logger.warning(f"[_meta_select_stocks] {t_minus_1.date()} 因子快照为空")
            return []

        features_df = features_df[features_df['ts_code'].isin(recommended)].copy()

        if features_df.empty:
            logger.warning("[_meta_select_stocks] 推荐股票在因子快照中无数据")
            return []

        logger.info(f"[_meta_select_stocks] 候选股票数: {len(features_df)}")

        ts_code_col = features_df['ts_code'].copy()

        # [2026-08-12 用户决策] 特征池预过滤: alpha_095_FFD 从候选特征中排除
        # (策略层面过滤, 不动因子脚本 factors.py 与 data/factor_snapshots/ 快照)
        feature_cols = [c for c in features_df.columns if c.startswith('alpha_')]

        # 中性化前的诊断日志
        logger.info(f"[_meta_select_stocks] 中性化前特征数: {len(feature_cols)}")

        features_df = self._neutralize_factors(features_df, feature_cols, t_minus_1)

        features_df['ts_code'] = ts_code_col.values

        # [R3-1 A2] 附加 V10 pred_rank（替代 z-score，截面 rank percentile）
        date_str = context.current_dt.strftime('%Y%m%d')
        result = self._read_v7_snapshot(date_str)
        if result is not None:
            _, pred_dict = result
            features_df['v10_pred_raw'] = features_df['ts_code'].map(pred_dict).fillna(0.5)
        else:
            features_df['v10_pred_raw'] = 0.5

        # 截面 rank percentile（替代 z-score，50 只候选池内相对信号强度）
        v10_vals = features_df['v10_pred_raw']
        if v10_vals.std() > 1e-6:
            features_df['v10_pred_rank'] = v10_vals.rank(pct=True)
        else:
            features_df['v10_pred_rank'] = 0.5

        logger.info(f"[R3-1 A2] V10 pred_raw: min={v10_vals.min():.4f}, max={v10_vals.max():.4f}, "
                    f"std={v10_vals.std():.4f}  |  rank: min={features_df['v10_pred_rank'].min():.2f}, "
                    f"max={features_df['v10_pred_rank'].max():.2f}")

        feature_cols_neu = [c for c in features_df.columns if c.endswith('_neu') and c.startswith('alpha_')]

        # [Alpha191 Score] 在中性化后的 _neu 列上计算 Score 和 Conf
        if self._alpha191_score_enabled:
            features_df = self._compute_alpha191_scores(features_df)
            # 将 score/conf 列纳入特征列表
            score_cols = [c for c in features_df.columns if c.endswith('_score')]
            conf_cols = [c for c in features_df.columns if c.endswith('_conf')]
            feature_cols_neu.extend(score_cols + conf_cols)

        feature_cols_neu.append('v10_pred_rank')  # [R3-1 A2] 排名百分位
        feature_cols_neu.append('v10_pred_raw')   # [Part C] 原始预测值（连续信号）

        keep_cols = ['ts_code'] + feature_cols_neu
        keep_cols = [c for c in keep_cols if c in features_df.columns]
        features_df = features_df[keep_cols]

        features_df.set_index('ts_code', inplace=True)

        # 【改造5】单阶段模型：直接预测强势盈利概率 P(profit)

        if self.ensemble_manager is None:
            logger.warning("[_meta_select_stocks] 模型未就绪")
            return []

        # 使用训练时筛选出的特征
        selected_feats = self.ensemble_manager.selected_features

        missing = [f for f in selected_feats if f not in features_df.columns]

        if missing:
            logger.warning(f"[_meta_select_stocks] 预测缺少特征: {missing[:5]}...")

        # 只使用存在的特征
        selected_feats = [f for f in selected_feats if f in features_df.columns]

        if not selected_feats:
            logger.warning("[_meta_select_stocks] 无可用特征进行预测")
            return []

        features_for_pred = features_df[selected_feats]

        logger.info(f"[_meta_select_stocks] 预测输入形状: {features_for_pred.shape}")

        # 直接预测强势盈利概率
        prob_profit = self.ensemble_manager.predict_proba(features_for_pred)

        if hasattr(prob_profit, 'flatten'):
            prob_profit = prob_profit.flatten()

        # 概率分布诊断日志
        prob_stats = {
            'count': len(prob_profit),
            'min': float(prob_profit.min()) if len(prob_profit) > 0 else 0,
            'max': float(prob_profit.max()) if len(prob_profit) > 0 else 0,
            'mean': float(prob_profit.mean()) if len(prob_profit) > 0 else 0,
            'median': float(np.median(prob_profit)) if len(prob_profit) > 0 else 0,
        }
        _em_obj = getattr(self.ensemble_manager, 'objective_mode', 'binary') if self.ensemble_manager is not None else 'binary'
        _dist_name = '回归预测分布(截面分位)' if _em_obj == 'regression' else '盈利概率分布(二分类)'
        logger.info(f"[_meta_select_stocks] {_dist_name}: {prob_stats}")

        # 直接按预测概率排序取 Top-N，不做阈值筛选
        MAX_META_SELECTED = int(self.config.get('meta_labeling', {}).get('max_meta_selected', 20))

        # 【R5】当IPBS软重排启用时，扩大选股池到 extended_pool_size
        tech_cfg = self.config.get('tech_filter', {})
        if tech_cfg.get('enabled', False) and tech_cfg.get('method', '') == 'ipbs_soft_rerank':
            extended_pool = tech_cfg.get('extended_pool_size', 30)
            MAX_META_SELECTED = min(MAX_META_SELECTED, extended_pool)

        n_select = min(MAX_META_SELECTED, len(prob_profit))
        top_idx = np.argsort(prob_profit)[::-1][:n_select]
        selected = list(features_df.index[top_idx])

        n_total = len(features_df)
        logger.info(f"[_meta_select_stocks] Top-{n_select} 精选模式：{n_select}/{n_total} 只入选（按概率排序，无阈值筛选）")

        # 日志：前5只股票的概率
        sorted_for_log = np.argsort(prob_profit)[::-1]
        top5_probs = [(features_df.index[i], float(prob_profit[i])) for i in sorted_for_log[:5]]
        logger.info(f"[_meta_select_stocks] 精选结果: {len(selected)}/{len(features_df)} 只入选")
        for rank, (code, prob) in enumerate(top5_probs, 1):
            logger.info(f"  #{rank} {code} P(profit)={prob:.4f}")

        # 构建选中股票的概率字典（用于仓位控制）
        prob_dict = {}
        for i, code in enumerate(features_df.index):
            if code in set(selected):
                prob_dict[code] = float(prob_profit[i])

        return selected, prob_dict

    def _load_select_mode_data(self, current_dt, context):
        """加载选股所需数据（保留方法但返回值未被使用）

        【BUG-F修复说明】
        此方法的返回值在 before_market_open 中未被使用，
        故调用已被删除。此方法保留以防其他地方调用。
        """
        try:
            end_date = current_dt - timedelta(days=1)
            start_date = end_date - timedelta(days=365)

            df_daily = self.data_provider.get_daily_data(
                None, start_date, end_date, reference_date=self.reference_date
            )
            if df_daily.empty:
                logger.warning("日线数据为空")
                return None, None

            # 计算因子
            df_daily, alpha_list = self._get_alpha_factors(df_daily)

            # 合并基本面快照
            snap = self._get_cached_factor_snapshot(current_dt, context)
            if snap is not None and not snap.empty:
                df_daily = df_daily.merge(
                    snap[['ts_code', 'trade_date'] + [c for c in snap.columns
                                                      if
                                                      c not in ['ts_code', 'trade_date', 'close', 'open', 'high', 'low',
                                                                'volume', 'amount']]],
                    on=['ts_code', 'trade_date'],
                    how='left'
                )

            # 过滤停牌股
            df_daily = df_daily[df_daily['volume'] > 0].copy()

            df_factors = df_daily[['ts_code', 'trade_date'] + alpha_list].copy()

            return df_daily, df_factors

        except Exception as e:
            logger.error(f"加载选股数据失败: {e}")
            return None, None

    def _get_cached_factor_snapshot(self, date, context):
        """获取Alpha191因子快照（带缓存）

        【BUG-L修复】原修复中使用 get_daily_data(None,...) 导致
        'NoneType' object has no attribute 'replace' 报错。
        恢复原始实现方式：调用 data_provider.get_factor_snapshot(date)，
        这是 data_provider 专门提供的因子快照接口。

        Args:
            date: 日期（datetime）
            context: 策略上下文

        Returns:
            DataFrame: 包含 alpha191 因子的当日截面数据，或 None
        """
        date_key = date.strftime('%Y%m%d')
        if date_key in self._factor_snapshot_cache:
            return self._factor_snapshot_cache[date_key]

        try:
            # 使用 data_provider 专门的因子快照接口
            dp = self.data_provider if hasattr(self, 'data_provider') else context.data_provider
            df = dp.get_factor_snapshot(date)
            if df is None or df.empty:
                self._factor_snapshot_cache[date_key] = None
                return None
            self._factor_snapshot_cache[date_key] = df.copy()
            return df.copy()
        except Exception as e:
            logger.debug(f"获取因子快照失败 {date_key}: {e}")
            return None

    # ==================== 真实分钟路径标签 ====================
    def _assign_meta_labels(self, sample_snaps, calendar, recommended_by_date=None):
        """[R3-1 A1] 窄池内相对排名标签

        标签含义：在 V10 已选50只候选中，未来 rebalance_interval 天收益率
        排进 Top 40% → bin=1（强势），否则 → bin=0（弱势）。

        核心改进：标签空间 = 预测空间（都是50只窄池内排名），解决全市场
        标签与窄池排序任务不对齐的结构性问题。

        Args:
            sample_snaps: 因子快照列表（保留参数兼容性，A1模式不使用）
            calendar: 交易日历
            recommended_by_date: {date: [code_list]}，每个训练日 V10 推荐的50只
        """
        if recommended_by_date is None or not recommended_by_date:
            logger.warning("[A1] recommended_by_date 为空，fallback 到全市场 Triple Barrier 标签")
            return self._assign_meta_labels_legacy(sample_snaps, calendar)

        rebal_days = self.rebalance_interval
        label_top_pct = self.config.get('meta_labeling', {}).get('label_top_pct', 0.4)
        cal_ts = pd.DatetimeIndex(sorted(calendar))

        all_records = []
        for sample_date, codes in recommended_by_date.items():
            sample_ts = pd.Timestamp(sample_date)
            d_idx = cal_ts.searchsorted(sample_ts)
            exit_idx = d_idx + rebal_days
            if exit_idx >= len(cal_ts):
                continue

            entry_date = cal_ts[d_idx]
            exit_date = cal_ts[exit_idx]

            for code in codes:
                # [FIX-1] 使用 data_provider 获取前复权价（原 raw_daily_path 为未复权价，
                # 除权日附近收益率被跳空缺口扭曲，产生系统性标签噪声）
                try:
                    df = self.data_provider._load_full_daily_data(code, self.reference_date)
                except Exception:
                    continue
                if df is None or df.empty:
                    continue
                df['trade_date'] = pd.to_datetime(df['trade_date']).dt.normalize()
                close_map = dict(zip(df['trade_date'], df['close']))

                entry_price = close_map.get(pd.Timestamp(entry_date))
                exit_price = close_map.get(pd.Timestamp(exit_date))

                if entry_price is not None and exit_price is not None and entry_price > 0:
                    ret = (exit_price / entry_price) - 1
                    all_records.append({
                        'date': sample_date,
                        'code': code,
                        'ret': ret,
                        'entry_price': entry_price,
                        'exit_price': exit_price
                    })

        if not all_records:
            logger.warning("[A1] 无有效标签记录，返回空 DataFrame")
            return pd.DataFrame(columns=['date', 'code', 'bin', 'ret', 'exit_type'])

        df_all = pd.DataFrame(all_records)

        def _rank_label(group):
            n = len(group)
            if n == 0:
                return group
            cutoff = max(1, int(np.ceil(n * label_top_pct)))
            group = group.sort_values('ret', ascending=False)
            group['bin'] = 0
            group.iloc[:cutoff, group.columns.get_loc('bin')] = 1
            group['exit_type'] = group['bin'].map({1: 'profit', 0: 'loss'})
            return group

        # pandas 2.x group_keys=False + apply 会丢失分组列 → 用循环替代
        result_parts = []
        for date_key, grp in df_all.groupby('date'):
            result_parts.append(_rank_label(grp))
        labels_df = pd.concat(result_parts, ignore_index=True) if result_parts else pd.DataFrame(columns=['date', 'code', 'bin', 'ret', 'exit_type'])

        n_dates = labels_df['date'].nunique()
        avg_pool = len(labels_df) / max(n_dates, 1)
        pos_rate = labels_df['bin'].mean()
        logger.info(f"[A1] 窄池相对标签: {n_dates} 个训练日, "
                    f"平均每组 {avg_pool:.1f} 只, "
                    f"正样本率={pos_rate:.3f} (top {label_top_pct*100:.0f}%), "
                    f"收益范围=[{labels_df['ret'].min():.4f}, {labels_df['ret'].max():.4f}]")

        return labels_df[['date', 'code', 'bin', 'ret', 'exit_type']]

    def _assign_meta_labels_legacy(self, sample_snaps, calendar):
        """全市场 Triple Barrier 标签（旧版，A1 的 fallback）"""
        code_date_map = {}
        for snap_df in sample_snaps:
            for _, row in snap_df[['trade_date', 'ts_code']].drop_duplicates().iterrows():
                code = row['ts_code']
                dt = row['trade_date']
                code_date_map.setdefault(code, set()).add(dt)

        all_events = []
        for code, dates in code_date_map.items():
            try:
                df = self.data_provider._load_full_daily_data(code, max(dates))
            except Exception:
                continue
            if df is None or df.empty:
                continue
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.normalize()
            close_map = dict(zip(df['trade_date'], df['close']))
            for dt in dates:
                if dt in close_map:
                    all_events.append({
                        'trade_date': dt,
                        'ts_code': code,
                        'entry_price': close_map[dt]
                    })
        if not all_events:
            return pd.DataFrame(columns=['date', 'code', 'bin', 'ret', 'exit_type'])
        events_all = pd.DataFrame(all_events)
        events_all = events_all.set_index('trade_date')
        labels_df = self.meta_labeler.compute_labels(events_all, calendar)
        return labels_df

    def _assign_meta_labels_v2(self, sample_snaps, calendar, recommended_by_date=None):
        """[A2] 窄池 + 三重屏障标签（Prado Meta-Label对齐版）

        标签定义（Lopez de Prado Ch.3）：
        y_meta = 1  if Primary 预测正确（交易盈利）
        y_meta = 0  if Primary 预测错误（交易亏损）

        通过 Triple Barrier 退出结果判定：
        - profit_take 触发 → bin=1（触达止盈，一定盈利）
        - stop_loss 触发   → bin=0（触达止损，一定亏损）
        - expire 到期      → bin=1 if ret>0 else 0（到期时盈亏判断）

        关键区别 vs 旧版：
        - 不再强制 top-40% 排名 → 正样本率由市场/策略自然决定
        - 不再按截面收益率排序覆盖 bin → 保留屏障语义
        - Meta 模型学习的是"V10信号在当前环境下是否会盈利"
          而非"哪只股票相对排名更高"

        时序对齐：
        - 标签入场：D-1日close（信号产生时点收盘）→ compute_labels_fast 覆盖为前复权 D-1日close
        - 交易入场：D日open（实际成交价，信号次日开盘执行）
        - 偏差：D-1 close → D open = 隔夜跳空，作为实盘可行性主动接受
        """
        if recommended_by_date is None or not recommended_by_date:
            logger.warning("[A2] recommended_by_date 为空，fallback 到 A1 标签")
            return self._assign_meta_labels(sample_snaps, calendar)

        cal_ts = pd.DatetimeIndex(sorted(calendar))

        # Phase 1: 为窄池内每只股票构建事件（信号日 D-1 close 入场）
        # [2026-08-07 标签时点修复] 事件日 = 信号产生日 D-1:
        #   信号基于 recomd_{D}（≤D-1 收盘信息, D 盘前可得）→ 标签从 D-1 收盘模拟入场
        #   交易 D 开盘执行 → 隔夜跳空偏差(D-1 close→D open)作为实盘可行性主动接受
        all_events = []
        for sample_date, codes in recommended_by_date.items():
            sample_ts = pd.Timestamp(sample_date)
            d_idx = cal_ts.searchsorted(sample_ts)
            if d_idx <= 0 or d_idx >= len(cal_ts):
                continue
            event_ts = cal_ts[d_idx - 1]  # 信号日 = 执行日前一交易日 (D-1)
            for code in codes:
                # [FIX-2] 不再读 raw_daily_path（原始未复权价），
                # compute_labels_fast Phase 2.5 会用 dp._load_full_daily_data()
                # 覆盖为前复权 D日close，此处仅注册事件（entry_price 占位）
                all_events.append({
                    'trade_date': event_ts,
                    'ts_code': code,
                    'entry_price': 1.0,  # 占位，Phase 2.5 覆盖为前复权价
                })

        if not all_events:
            logger.warning("[A2] 无有效事件，fallback 到 A1")
            return self._assign_meta_labels(sample_snaps, calendar)

        events_df = pd.DataFrame(all_events)
        events_df = events_df.set_index('trade_date')

        # Phase 2: 使用 TripleBarrierLabeler 计算屏障退出结果
        # align_entry_mode='event_date_close' 使入场价使用D日close（与A1/V10一致）
        # [2026-08-30 修复] 不传全局 reference_date: 标签计算内部用事件窗口末日作局部基准,
        #   消除"复权基准日不同 → 标签分布翻转"问题 (跨基准日回测不可比的根因)
        barrier_labels = self.meta_labeler.compute_labels_fast(
            events_df, calendar, align_entry_mode='event_date_close')

        if barrier_labels is None or barrier_labels.empty:
            logger.warning("[A2] 三重屏障标签计算返回空，fallback 到 A1")
            return self._assign_meta_labels(sample_snaps, calendar)

        # Phase 3: 直接使用屏障退出判定 bin 标签（Prado Meta-Label 定义）
        # labels.py compute_labels_fast Phase 8 已正确计算：
        #   profit_take → bin=1, stop_loss → bin=0, expire → bin=1 if ret>0 else 0
        # 不再强制 top-40% 排名覆盖 —— 那会破坏 Meta-Label 语义
        labels_df = barrier_labels.copy()

        # Phase 4: 诊断日志
        n_dates = labels_df['date'].nunique()
        avg_pool = len(labels_df) / max(n_dates, 1)
        pos_rate = labels_df['bin'].mean()
        exit_dist = labels_df['exit_type'].value_counts().to_dict()
        # 按退出类型统计正样本率
        exit_pos_rates = {}
        for et in ['profit_take', 'stop_loss', 'expire']:
            et_mask = labels_df['exit_type'] == et
            if et_mask.any():
                exit_pos_rates[et] = labels_df.loc[et_mask, 'bin'].mean()
        logger.info(
            f"[A2] 窄池+三重屏障标签（Meta-Label）: {n_dates} 个训练日, "
            f"平均每组 {avg_pool:.1f} 只, "
            f"正样本率={pos_rate:.3f} (barrier-based, 自然分布), "
            f"退出分布={exit_dist}, "
            f"各退出正样本率={exit_pos_rates}, "
            f"收益范围=[{labels_df['ret'].min():.4f}, {labels_df['ret'].max():.4f}]"
        )

        return labels_df[['date', 'code', 'bin', 'ret', 'exit_type']]

    def _assign_meta_labels_staged(self, sample_snaps, calendar, recommended_by_date=None,
                                   mode='staged_binary'):
        """[L1/L2] 窄池 + 分阶段退出标签（标签-交易同规则, de Prado 标签-执行对齐）

        标签定义（L1）：
        y_meta = 1  if 分阶段实现收益 realized_ret > 0（交易盈利）
        y_meta = 0  otherwise

        标签定义（L2, staged_rank）：
        y = realized_ret 在事件日截面上的分位 (0,1]（排序/回归目标）

        与 _assign_meta_labels_v2 的区别：三重屏障 PT/SL/expire 判定 →
        分阶段退出模拟 (labels.compute_labels_staged), 输出 realized_ret。
        """
        if recommended_by_date is None or not recommended_by_date:
            logger.warning("[L1/L2] recommended_by_date 为空，fallback 到 A1 标签")
            return self._assign_meta_labels(sample_snaps, calendar)

        cal_ts = pd.DatetimeIndex(sorted(calendar))

        # Phase 1: 构建事件 (与 v2 相同, entry_price 占位, 标签器覆盖为 D-1日close)
        # [2026-08-07 标签时点修复] 事件日 = 信号日 D-1 (信号基于 ≤D-1 收盘的 recomd_{D})
        # 标签从 D-1 收盘模拟入场 → 交易 D 开盘执行 → 隔夜跳空主动接受
        all_events = []
        for sample_date, codes in recommended_by_date.items():
            sample_ts = pd.Timestamp(sample_date)
            d_idx = cal_ts.searchsorted(sample_ts)
            if d_idx <= 0 or d_idx >= len(cal_ts):
                continue
            event_ts = cal_ts[d_idx - 1]  # 信号日 = 执行日前一交易日 (D-1)
            for code in codes:
                all_events.append({
                    'trade_date': event_ts,
                    'ts_code': code,
                    'entry_price': 1.0,
                })

        if not all_events:
            logger.warning("[L1/L2] 无有效事件，fallback 到 A1")
            return self._assign_meta_labels(sample_snaps, calendar)

        events_df = pd.DataFrame(all_events).set_index('trade_date')

        # Phase 2: 分阶段退出模拟
        # [2026-08-30 修复] 不传全局 reference_date: 标签计算内部用事件窗口末日作局部基准,
        #   消除"复权基准日不同 → 样本分布翻转"问题 (跨基准日回测不可比的根因)
        staged_labels = self.meta_labeler.compute_labels_staged(
            events_df, calendar)

        if staged_labels is None or staged_labels.empty:
            logger.warning("[L1/L2] 分阶段标签计算返回空，fallback 到 A1")
            return self._assign_meta_labels(sample_snaps, calendar)

        labels_df = staged_labels.copy()

        # Phase 3: 按模式生成 bin
        if mode == 'staged_rank':
            # [L2] 截面分位: 每个事件日内 realized_ret 的 rank pct (0,1]
            labels_df['bin'] = labels_df.groupby('date')['realized_ret'].rank(pct=True)
        else:
            # [L1] 二分类: sign(realized_ret); min_abs_ret>0 时丢弃小盈亏样本
            if self._label_min_abs_ret > 0:
                n_before = len(labels_df)
                labels_df = labels_df[
                    labels_df['realized_ret'].abs() >= self._label_min_abs_ret]
                logger.info(f"[L1] min_abs_ret={self._label_min_abs_ret:.4f}: "
                            f"丢弃小盈亏样本 {n_before - len(labels_df)}/{n_before}")
            labels_df['bin'] = (labels_df['realized_ret'] > 0).astype(int)

        # Phase 4: 诊断日志
        n_dates = labels_df['date'].nunique()
        avg_pool = len(labels_df) / max(n_dates, 1)
        exit_dist = labels_df['exit_type'].value_counts().to_dict()
        rr = labels_df['realized_ret']
        if mode == 'staged_rank':
            logger.info(
                f"[L2] 窄池+分阶段标签（排序/回归目标）: {n_dates} 个训练日, "
                f"平均每组 {avg_pool:.1f} 只, "
                f"bin(分位) mean={labels_df['bin'].mean():.3f}, "
                f"realized_ret mean={rr.mean():+.4f} median={rr.median():+.4f} "
                f"范围=[{rr.min():.4f}, {rr.max():.4f}], "
                f"退出分布={exit_dist}"
            )
        else:
            pos_rate = labels_df['bin'].mean()
            exit_pos_rates = {}
            for et in ['profit_take', 'stop_loss', 'expire']:
                et_mask = labels_df['exit_type'] == et
                if et_mask.any():
                    exit_pos_rates[et] = round(float(labels_df.loc[et_mask, 'bin'].mean()), 4)
            logger.info(
                f"[L1] 窄池+分阶段标签（Meta-Label, 标签-交易同规则）: {n_dates} 个训练日, "
                f"平均每组 {avg_pool:.1f} 只, "
                f"正样本率={pos_rate:.3f}, "
                f"退出分布={exit_dist}, "
                f"各退出正样本率={exit_pos_rates}, "
                f"realized_ret mean={rr.mean():+.4f} median={rr.median():+.4f} "
                f"范围=[{rr.min():.4f}, {rr.max():.4f}]"
            )

        return labels_df[['date', 'code', 'bin', 'ret', 'exit_type', 'realized_ret']]

    # ==================== 元标注模型训练 ====================
    # 【改造5】合并为单阶段模型
    def _should_retrain_model(self):
        """[重训间隔] 判断本次调仓是否应重训模型。

        规则 (调用方需先 self._rebalances_since_train += 1):
          1. force_retrain=True (冷启动/外部强制) → 重训
          2. 模型未就绪 (ensemble_manager 为空或无折模型) → 重训
          3. RETRAIN_INTERVAL <= 1 → 每次调仓都重训 (原行为)
          4. _rebalances_since_train >= RETRAIN_INTERVAL → 重训并重置计数
          5. 其余 → 沿用最近模型 (不重训, 特征集/meta_auc/门控一并沿用)

        Returns:
            (should_retrain: bool, reason: str)
        """
        model_ready = (self.ensemble_manager is not None
                       and getattr(self.ensemble_manager, 'xgb_models', None))
        if self.force_retrain:
            return True, 'force_retrain (冷启动/强制)'
        if not model_ready:
            return True, '模型未就绪 (首次训练或上次训练失败)'
        if self.RETRAIN_INTERVAL <= 1:
            return True, f'retrain_interval={self.RETRAIN_INTERVAL}<=1 (每次调仓都重训)'
        if self._rebalances_since_train >= self.RETRAIN_INTERVAL:
            return True, (f'已达重训间隔 ({self._rebalances_since_train}'
                          f'/{self.RETRAIN_INTERVAL} 次调仓)')
        return False, (f'沿用最近模型 (距上次重训 {self._rebalances_since_train}'
                       f'/{self.RETRAIN_INTERVAL} 次调仓)')

    def _train_meta_model(self, current_date, context, recommended, df_daily, df_factors):

        logger.info("=" * 60)

        logger.info(f"【开始元标注模型训练（单阶段：直接预测强势盈利）】当前日期: {current_date.date()}")

        try:
            cal_df = get_trade_calendar('2010-01-01', current_date.strftime('%Y-%m-%d'))
            trade_days_all = sorted(cal_df['calendar_date'].tolist())
            trade_days_all = [pd.Timestamp(d) for d in trade_days_all if d < current_date]

            logger.info(
                f"历史交易日范围: {trade_days_all[0].date()} ~ {trade_days_all[-1].date()} (共 {len(trade_days_all)} 天)")
        except Exception as e:
            logger.error(f"获取日历失败: {e}")
            return False

        # ---------- 1. 过滤出标签封闭的交易日 ----------

        last_hist_day = trade_days_all[-1]

        eligible_train_dates = []

        # 【BUG-7修复】使用enumerate替代index()，O(N²) → O(N)
        for d_idx, d in enumerate(trade_days_all):
            label_end_idx = d_idx + self.meta_labeler.max_hold_days
            if label_end_idx < len(trade_days_all) and trade_days_all[label_end_idx] <= last_hist_day:
                eligible_train_dates.append(d)

        logger.info(
            f"标签封闭的可用交易日: {len(eligible_train_dates)} 天 (max_hold_days={self.meta_labeler.max_hold_days})")

        if len(eligible_train_dates) < self.TRAIN_WINDOW_DAYS:
            logger.warning(f"满足标签封闭条件的交易日不足 {self.TRAIN_WINDOW_DAYS} 天")

            return False

        train_dates = eligible_train_dates[-self.TRAIN_WINDOW_DAYS:]

        logger.info(f"训练窗口: {train_dates[0].date()} ~ {train_dates[-1].date()} (共 {len(train_dates)} 天)")

        # ---------- 2. 历史回放：每个训练日重新运行主模型选股 ----------

        logger.info(f"【历史回放】对 {len(train_dates)} 个训练日逐日运行主模型选股...")

        # 【Phase2优化】批量预加载因子快照（并发读取100天parquet → ~5s）
        all_snap_dates = []
        snap_date_set = set()
        for sample_date in train_dates:
            sd = self._get_previous_trading_day(sample_date, context)
            if sd is not None and sd not in snap_date_set:
                all_snap_dates.append(sd)
                snap_date_set.add(sd)

        logger.info(f"【历史回放】批量预加载 {len(all_snap_dates)} 天因子快照（并发）...")
        dp_snap = self.data_provider if hasattr(self, 'data_provider') else context.data_provider

        def _load_one_snap(date):
            dk = date.strftime('%Y%m%d')
            if dk in self._factor_snapshot_cache:
                return
            try:
                df = dp_snap.get_factor_snapshot(date)
                self._factor_snapshot_cache[dk] = df.copy() if df is not None else None
            except Exception:
                self._factor_snapshot_cache[dk] = None

        max_workers = min(8, len(all_snap_dates))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(_load_one_snap, all_snap_dates))
        loaded = sum(1 for dk in (d.strftime('%Y%m%d') for d in all_snap_dates)
                     if self._factor_snapshot_cache.get(dk) is not None)
        logger.info(f"【历史回放】快照预加载完成: {loaded}/{len(all_snap_dates)} 有效")

        all_samples = []
        recommended_by_date = {}  # [R3-1 A1] {sample_date: [code_list]}
        skipped_count = 0

        for i, sample_date in enumerate(train_dates):
            if i % 10 == 0:
                logger.info(f"历史回放进度: {i + 1}/{len(train_dates)}")

            try:
                daily_recommended = self._get_primary_recommendations_for_date(sample_date, context)
            except Exception as e:
                logger.debug(f"获取 {sample_date.date()} 主模型推荐失败: {e}")
                skipped_count += 1
                continue

            if not daily_recommended:
                skipped_count += 1
                continue

            # [R3-1 A1] 收集每日推荐列表
            recommended_by_date[sample_date] = daily_recommended

            # 【Phase2】已预加载到缓存，直接命中
            sd = self._get_previous_trading_day(sample_date, context)
            if sd is None:
                skipped_count += 1
                continue
            snap = self._get_cached_factor_snapshot(sd, context)
            if snap is None or snap.empty:
                skipped_count += 1
                continue

            daily_recommended_set = set(daily_recommended)
            snap = snap[snap['ts_code'].isin(daily_recommended_set)].copy()

            if snap.empty:
                skipped_count += 1
                continue

            snap['snapshot_date'] = sd  # [FIX P1-1] 保留真实快照日期用于上下文特征
            snap['trade_date'] = sample_date  # 重标签为 T 用于标签合并
            all_samples.append(snap)

        logger.info(f"【历史回放完成】{len(all_samples)}/{len(train_dates)} 个训练日有样本，跳过 {skipped_count} 个")

        if not all_samples:
            logger.warning("无可用历史因子快照，训练失败")
            return False

        # 【修复】增加样本拼接诊断日志
        train_data = pd.concat(all_samples, ignore_index=True)
        logger.info(f"合并后数据形状: {train_data.shape}, 列数: {len(train_data.columns)}")

        # ---------- 3. 生成三分类标签 ----------

        logger.info("开始生成三分类标签...")

        try:
            label_mode = self.config.get('meta_labeling', {}).get('label_mode', 'relative')
            if label_mode in ('staged_binary', 'staged_rank'):
                # [L1/L2] 分阶段退出标签 (标签-交易同规则)
                labels_df = self._assign_meta_labels_staged(
                    all_samples, trade_days_all,
                    recommended_by_date=recommended_by_date, mode=label_mode)
            elif label_mode == 'relative_barrier':
                labels_df = self._assign_meta_labels_v2(all_samples, trade_days_all,
                                                         recommended_by_date=recommended_by_date)
            elif label_mode == 'absolute':
                labels_df = self._assign_meta_labels_legacy(all_samples, trade_days_all)
            else:
                labels_df = self._assign_meta_labels(all_samples, trade_days_all,
                                                      recommended_by_date=recommended_by_date)

            if labels_df is None or labels_df.empty:
                logger.error("标签生成返回空数据")
                return False

            logger.info(f"标签数据形状: {labels_df.shape}, 标签分布:\n{labels_df['bin'].value_counts().to_dict()}")
        except Exception as e:
            import traceback
            logger.error(f"生成标签异常: {e}")
            logger.error(f"堆栈:\n{traceback.format_exc()}")
            return False

        # ---------- 4. 合并标签 ----------

        # 标准化日期格式以便合并
        train_data['trade_date_std'] = pd.to_datetime(train_data['trade_date'])
        labels_df['date_std'] = pd.to_datetime(labels_df['date'])

        try:
            # [L1/L2] staged 模式额外携带 realized_ret (诊断/未来收益归因权重)
            _label_cols = ['date', 'code', 'bin', 'exit_type']
            if 'realized_ret' in labels_df.columns:
                _label_cols.append('realized_ret')
            # [2026-08-07 标签时点修复] 合并键对齐 PIT:
            #   标签事件日 = 信号日 D-1 (entry=D-1 close) ↔ 样本特征快照日 snapshot_date=D-1
            #   (修复前: 标签 date=D(执行日) ↔ 样本 trade_date=D; 事件日前移后必须改 snapshot_date)
            train_data = train_data.merge(
                labels_df[_label_cols],
                left_on=['snapshot_date', 'ts_code'],
                right_on=['date', 'code'],
                how='left'
            )
        except Exception as e:
            logger.error(f"合并标签数据失败: {e}")
            return False

        train_data.drop(columns=['date', 'code'], inplace=True, errors='ignore')
        # [L2] staged_rank: bin 为 (0,1] 浮点分位, 保留浮点; 其余模式为 0/1 整数
        if self._label_mode == 'staged_rank':
            train_data['bin'] = train_data['bin'].fillna(0.5).astype(float)
        else:
            train_data['bin'] = train_data['bin'].fillna(0).astype(int)

        if self._label_mode == 'staged_rank':
            _b = train_data['bin']
            logger.info(f"训练样本总数: {len(train_data)}, 标签(分位)统计: "
                        f"mean={_b.mean():.3f} std={_b.std():.3f} "
                        f"min={_b.min():.3f} max={_b.max():.3f}")
        else:
            logger.info(f"训练样本总数: {len(train_data)}, 标签分布: {train_data['bin'].value_counts().to_dict()}")

        if len(train_data) < self.MIN_TRAIN_SAMPLES:
            logger.warning(f"训练样本不足 ({len(train_data)} < {self.MIN_TRAIN_SAMPLES})")
            return False

        # ---------- 5. 因子中性化 ----------

        logger.info("开始因子中性化...")

        try:
            # [2026-08-12 用户决策] 特征池预过滤: alpha_095_FFD 从候选特征中排除 (策略层面, 不动因子脚本/快照)
            feature_cols = [c for c in train_data.columns if c.startswith('alpha_')]
            logger.info(f"中性化前因子数: {len(feature_cols)}")

            # 【fit/transform】训练期强制重新拟合OLS系数（对齐Wyckoff）
            self._train_ols_betas = None
            # [数据泄露修复] 按 trade_date 填充 total_mv，消除市值时序泄露
            train_data = self._fill_total_mv_per_date(train_data)
            # 市值已按日填充，ref_date=None 避免二次填充覆盖
            train_data = self._neutralize_factors(train_data, feature_cols, ref_date=None)

            logger.info(f"中性化后数据形状: {train_data.shape}")

        except Exception as e:
            import traceback
            logger.error(f"因子中性化异常: {e}")
            logger.error(f"堆栈:\n{traceback.format_exc()}")
            return False

        # [R3-1 A2] 附加 V10 pred_raw，然后截面 rank percentile（替代 z-score）
        # Prado Ch.3: 二级模型特征必须包含一级模型预测值
        # 【性能优化 v2】预转换日期 + 向量化 rank，避免逐日循环
        v10_col_raw = 'v10_pred_raw'
        v10_col_rank = 'v10_pred_rank'
        train_data[v10_col_raw] = 0.5  # 默认值

        # 预转换 trade_date 为 datetime（避免循环内重复转换）
        trade_date_ts = pd.to_datetime(train_data['trade_date'])

        for sample_date in train_dates:
            result = self._read_v7_snapshot(sample_date.strftime('%Y%m%d'))
            if result is not None:
                _, pred_dict = result
                date_mask = trade_date_ts == pd.Timestamp(sample_date)
                train_data.loc[date_mask, v10_col_raw] = (
                    train_data.loc[date_mask, 'ts_code'].map(pred_dict).fillna(0.5)
                )

        # [R3-1 A2] 截面 rank percentile（向量化：groupby rank 替代逐日循环）
        train_data[v10_col_rank] = 0.5  # 默认值
        group_stds = train_data.groupby(trade_date_ts)[v10_col_raw].transform('std')
        mask_std = group_stds > 1e-6
        if mask_std.any():
            ranks = train_data.groupby(trade_date_ts)[v10_col_raw].rank(pct=True)
            train_data.loc[mask_std, v10_col_rank] = ranks[mask_std]

        # V10 pred 分布诊断
        v10_rank_vals = train_data.loc[train_data[v10_col_rank] != 0.5, v10_col_rank]
        if len(v10_rank_vals) > 0:
            logger.info(f"[R3-1 A2] V10 pred_rank: min={v10_rank_vals.min():.4f}, "
                        f"max={v10_rank_vals.max():.4f}, mean={v10_rank_vals.mean():.4f}")

        # RankIC 诊断（用 rank 而非 raw）
        valid_mask = train_data[v10_col_rank].notna() & train_data['bin'].notna()
        if valid_mask.sum() > 10:
            from scipy.stats import spearmanr
            v10_ic = spearmanr(
                train_data.loc[valid_mask, v10_col_rank],
                train_data.loc[valid_mask, 'bin']
            )[0]
            logger.info(f"[R3-1 A2] V10 pred_rank RankIC vs bin = {v10_ic:.4f}")

        feature_cols_neu = [c for c in train_data.columns if c.endswith('_neu') and c.startswith('alpha_')]

        # [Alpha191 Score] 在中性化后的 _neu 列上计算 Score 和 Conf
        if self._alpha191_score_enabled:
            logger.info("[Alpha191 Score] Computing factor scores on neutralized features...")
            train_data = self._compute_alpha191_scores(train_data)
            n_score = sum(1 for c in train_data.columns if c.endswith('_score'))
            n_conf = sum(1 for c in train_data.columns if c.endswith('_conf'))
            logger.info(f"  Added {n_score} score + {n_conf} conf columns")
            # 将 score/conf 列纳入特征列表，与 _neu 列平等竞争特征选择
            score_cols = [c for c in train_data.columns if c.endswith('_score')]
            conf_cols = [c for c in train_data.columns if c.endswith('_conf')]
            feature_cols_neu.extend(score_cols + conf_cols)

        feature_cols_neu.append(v10_col_rank)  # [R3-1 A2] 排名百分位
        feature_cols_neu.append(v10_col_raw)   # [Part C] 原始预测值（连续信号）

        if len(feature_cols_neu) <= 1:  # 至少需要 v10_pred_rank + 至少1个 _neu 特征
            logger.error(f"中性化后无有效特征 (含v10_pred_rank共{len(feature_cols_neu)}个)")
            return False

        # [Phase A] 直接使用二分类盈利标签（labels.py 已输出 {0, 1}）
        # [L2] staged_rank: bin 为 (0,1] 分位浮点 (回归目标), 日志改为统计量
        train_data['label_profit'] = train_data['bin']
        if self._label_mode == 'staged_rank':
            _lp = train_data['label_profit']
            logger.info(f"[Phase A] 回归标签(分位): mean={_lp.mean():.3f} std={_lp.std():.3f} "
                        f"P10={_lp.quantile(0.1):.3f} P90={_lp.quantile(0.9):.3f}")
        else:
            profit_rate = train_data['label_profit'].mean()
            label_dist = train_data['label_profit'].value_counts().to_dict()
            logger.info(f"[Phase A] 二分类标签分布: 0(亏损)={label_dist.get(0, 0)}, "
                        f"1(盈利)={label_dist.get(1, 0)}, 正样本比例={profit_rate:.3f}")

        logger.info("开始训练单阶段模型...")
        logger.info(f"训练数据形状: {train_data.shape}")
        logger.info(f"特征数: {len(feature_cols_neu)}")

        model_cfg = self._create_model_config()

        model_dir = os.path.join(self.output_dir, 'meta_models')

        os.makedirs(model_dir, exist_ok=True)

        # 【改造5】只有一个 ensemble_manager
        self.ensemble_manager = EnsembleModelManager(model_dir, model_cfg)

        # 类别平衡配置传入ensemble_manager（对齐Wyckoff V10）
        _balance_cfg = self.config.get('xgb_class_balance', 'auto')
        self._class_balance_mode = _balance_cfg if _balance_cfg in ('auto', 'none') else 'auto'
        self.ensemble_manager.class_balance_mode = self._class_balance_mode
        logger.info(f"[class_balance] mode={self._class_balance_mode}")

        # 保存特征列供后续选股使用（仅训练成功后才更新）
        # 注意: 在 train() 调用之前不更新 self.feature_cols，
        #         避免训练失败时 feature_cols 指向过期特征

        # 【Phase2优化】样本权重带跨周期缓存
        # [v3.7] 标签质量加权: PT/SL 强信号样本 ×2, expire 弱信号 ×1
        train_end_key = train_dates[-1].strftime('%Y%m%d')
        _pt_sl_mult = self.config.get('meta_labeling', {}).get(
            'sample_weight_pt_sl_multiplier', 2.0)
        sample_w = compute_sample_weights_cached(
            train_data, trade_days_all, self.meta_labeler.max_hold_days,
            cache_key=train_end_key,
            exit_type=train_data['exit_type'] if 'exit_type' in train_data.columns else None,
            pt_sl_multiplier=_pt_sl_mult
        )

        # ================================================================
        # [2026-08-18] 因子验证三件套调用段已移除 (SHAP时序/ADV/CIC 全删)
        # 原功能: ensemble 训练前 run_rebalance() 投票剔除劣质因子
        # 移除依据: 审查结论 (0票空转/冗余/86%误杀) + 用户决策
        # 特征筛选完全交由 ensemble 内 Purged K-Fold (低方差+IC+PSI+MDA投票)
        # ================================================================

        try:
            success = self.ensemble_manager.train(
                df_train=train_data,
                current_date=current_date.date(),
                feature_cols=feature_cols_neu,
                target_col='label_profit',
                sample_weight=sample_w
            )
        except Exception as e:
            import traceback
            logger.error(f"训练执行异常: {e}")
            logger.error(f"堆栈:\n{traceback.format_exc()}")
            return False

        if not success:
            logger.warning("模型训练失败")
            return False

        # 训练成功 → 更新特征列供后续选股使用
        self.feature_cols = feature_cols_neu

        # [R3-1 A2] 检查 v10_pred_rank 是否通过特征筛选
        if 'v10_pred_rank' in self.ensemble_manager.selected_features:
            v10_feat_rank = sorted(self.ensemble_manager.selected_features).index('v10_pred_rank') + 1
            selected_n = len(self.ensemble_manager.selected_features)
            logger.info(f"[R3-1 A2] ✅ v10_pred_rank 入选特征集，排名 {v10_feat_rank}/{selected_n}")
        else:
            logger.warning("[R3-1 A2] ⚠️ v10_pred_rank 未通过特征筛选（PSI豁免后，唯一可能是 IC 不足）")
            if 'v10_pred_rank' in self.ensemble_manager.feature_scores:
                fs = self.ensemble_manager.feature_scores['v10_pred_rank']
                logger.warning(f"[R3-1 A2] v10_pred_rank IC={fs.get('ic', 0):.4f}, "
                              f"IC_abs={fs.get('ic_abs', 0):.4f}, "
                              f"composite_score={fs.get('composite_score', 0):.4f}")

        # 【模型门控值】存储折集成 meta（D 修复 2026-08-19: 口径=折集成 OOS 平均, 与预测模型一致）
        # [Fix-20260907] rank 模式度量是 Spearman IC, 日志名区分 AUC/IC 避免误导
        _em_obj = getattr(self.ensemble_manager, 'objective_mode', 'binary')
        _metric_name = 'IC' if _em_obj == 'regression' else 'AUC'
        if hasattr(self.ensemble_manager, 'meta_auc') and self.ensemble_manager.meta_auc is not None:
            self.meta_auc = float(self.ensemble_manager.meta_auc)
            logger.info(f'[模型门控] 折集成 meta_{_metric_name}={self.meta_auc:.4f}')
        elif hasattr(self.ensemble_manager, 'xgb_val_auc') and self.ensemble_manager.xgb_val_auc is not None:
            self.meta_auc = float(self.ensemble_manager.xgb_val_auc)
            logger.info(f'[模型门控] XGB val {_metric_name}={self.meta_auc:.4f}')
        elif hasattr(self.ensemble_manager, 'best_val_auc') and self.ensemble_manager.best_val_auc is not None:
            self.meta_auc = float(self.ensemble_manager.best_val_auc)
            logger.info(f'[模型门控] 模型最佳{_metric_name}={self.meta_auc:.4f}')

        # 【BUG-H修复】保存中性化后的特征列名到 self.feature_cols
        # 否则 _meta_select_stocks 中 self.feature_cols 永远为空列表，
        # 导致预测时 X = snap[self.feature_cols] 为空 DataFrame
        self.feature_cols = feature_cols_neu

        self.force_retrain = False

        logger.info(f"【元标注模型训练完成】特征数: {len(self.feature_cols)}")
        logger.info("=" * 60)

        return True

    # ==================== V10上游门控 ====================

    def _check_v10_snapshot_empty(self, current_date) -> bool:
        """检查V10 recomd选股快照是否为空。

        Args:
            current_date: 当前交易日

        Returns:
            True: 快照为空或不存在 → 上层无选股, 应跳过
            False: 快照有股票 → 可以正常选股
        """
        import os
        date_str = current_date.strftime('%Y%m%d')
        fname = f'recomd_{date_str}.parquet'
        path = os.path.join(self._v10_snapshot_dir, fname)

        if not os.path.exists(path):
            logger.warning(f"[V10 Gate] 快照文件不存在: {path}, 视为空")
            return True

        try:
            df = pd.read_parquet(path)
            if len(df) == 0:
                logger.info(f"[V10 Gate] 快照为空 (0行): {path}")
                return True
            logger.info(f"[V10 Gate] 快照正常: {len(df)} 只股票")
            return False
        except Exception as e:
            logger.error(f"[V10 Gate] 读取快照异常: {path}, {e}, 视为空")
            return True

    # ==================== 盘前选股 ====================

    def before_market_open(self, context):

        current_date = context.current_dt

        self.trade_day_seq += 1

        # 初始化每周期目标（清零避免过期权重残留）
        self.target_stocks = []
        self.target_weights = {}

        # [2026-08-29 方案A-1 修复] 回撤熔断状态机: 每日盘前更新 (必须在 need_rebalance_check
        #   早退 return 之前) —— 原实现放在 before_market_open 尾部, 非调仓日提前 return
        #   导致状态机每 20 日才更新一次, 2024-02-07 最大回撤日(非调仓日)从未被检测,
        #   R0/R1 熔断 0 触发、结果逐位相同 (2026-08-29 批0 发现)
        self._update_dd_breaker(current_date)

        # 【修正12】每日检查到期持仓，标记为开盘价退出
        # 到期持仓在 handle_bar 中优先以开盘价卖出，不再检查当日PT/SL
        self._expiry_sells_today = []
        if self.meta_labeler is not None and self.holdings:
            for code in list(self.holdings.keys()):
                entry_date = self.holdings[code]['date']
                try:
                    idx_entry = context.calendar.index(pd.Timestamp(entry_date))
                    idx_today = context.calendar.index(current_date)
                    held_trade_days = idx_today - idx_entry
                except (ValueError, KeyError):
                    held_trade_days = (current_date - entry_date).days
                # 【修正12】到期条件：held_trade_days >= max_hold_days
                # 设计逻辑（2026-08-07 标签时点修复）：
                #   标签：信号日(D-1)close入场 → PT/SL在D~D+max_hold_days-1内检测 → 到期取D+max_hold_days-1日close
                #   交易：D日open入场 → 持有至D+max_hold_days日open退出（held_trade_days=max_hold_days触发）
                #   标签-交易偏差：D-1 close → D open = 隔夜跳空，作为实盘可行性主动接受
                # held_trade_days在D+max_hold_days日=max_hold_days，条件恰好触发
                if held_trade_days >= self.meta_labeler.max_hold_days:
                    self._expiry_sells_today.append(code)
                    logger.info(f"[{current_date.date()}] {code} 到期退出标记（持有{held_trade_days}天，"
                                f"到期日开盘价退出，不再检查当日PT/SL）")

        # 【R5】每日IPBS持仓退出检查（每个交易日都执行，非仅再平衡日）
        # 用T-1的OHLCV数据判定，不依赖当日分钟线
        self._ipbs_exit_list = []
        tech_cfg_exit = self.config.get('tech_filter', {})
        if tech_cfg_exit.get('enabled', True) and tech_cfg_exit.get('ipbs_exit_enabled', True):
            t_minus_1 = self._get_previous_trading_day(current_date, context)
            ipbs_exit_list = []

            for code in list(context.portfolio.positions.keys()):
                if code not in self.holdings:
                    continue
                try:
                    should_exit, reason = self._check_ipbs_exit(code, t_minus_1, context)
                    if should_exit and reason:
                        ipbs_exit_list.append((code, reason))
                except Exception as e:
                    logger.warning(f"[{current_date.date()}] {code} IPBS退出检查异常: {e}")

            self._ipbs_exit_list = ipbs_exit_list
            if ipbs_exit_list:
                logger.info(f"[R5-持仓退出] {len(ipbs_exit_list)}只触发IPBS退出")
                for code, reason in ipbs_exit_list:
                    logger.info(f"  {code}: {reason}")

        # ===== V10上游信号门控 + 调仓日判断 =====
        is_rebalance_day = (self.trade_day_seq % self.rebalance_interval == 1)
        need_rebalance_check = is_rebalance_day or self._v10_deferred

        if not need_rebalance_check:
            return  # 非调仓日, 非顺延 → 正常跳过

        # V10上游门控检查
        if self._v10_gate_enabled:
            v10_empty = self._check_v10_snapshot_empty(current_date)

            # 检查最大顺延天数
            if self._v10_deferred and self._v10_max_defer_days > 0:
                defer_days = self.trade_day_seq - self._v10_defer_start_day
                if defer_days > self._v10_max_defer_days:
                    v10_empty = False  # 超过最大顺延天数, 强制恢复（不再等）
                    logger.warning(f"[V10 Gate] 顺延已超{self._v10_max_defer_days}天 ({defer_days}d), "
                                   f"强制恢复选股")

            if v10_empty:
                # V10上层无选股 → 清仓, 进入顺延状态
                self.target_stocks = []
                self.target_weights = {}
                if not self._v10_deferred:
                    self._v10_deferred = True
                    self._v10_defer_start_day = self.trade_day_seq
                    logger.warning(f"[V10 Gate] 当日上层无选股 (V10快照为空), "
                                   f"跳过选股, 进入顺延状态 (day={self.trade_day_seq})")
                else:
                    defer_days_soft = self.trade_day_seq - self._v10_defer_start_day
                    logger.info(f"[V10 Gate] 当日上层仍无选股, 继续顺延 "
                                f"(已顺延{defer_days_soft}d)")
                    # 顺延超过60个交易日时输出WARNING（模型可能长期失效）
                    if defer_days_soft > 60:
                        logger.warning(f"[V10 Gate] ⚠️ V10顺延已超过60个交易日 ({defer_days_soft}d), "
                                       f"上游模型可能长期失效, 建议检查V10 recomd快照")
                return  # 只清仓, 不训练不选股

            # V10快照有股票 → 恢复选股
            if self._v10_deferred:
                logger.info(f"[V10 Gate] V10快照恢复 ({self._v10_defer_start_day}→{self.trade_day_seq}), "
                            f"顺延{self.trade_day_seq - self._v10_defer_start_day}天后恢复选股")
                self._v10_deferred = False
                self._v10_defer_start_day = 0
                # 重置调仓周期计数器, 使下次调仓在rebalance_interval天后
                self.trade_day_seq = 1

        logger.info("=" * 60)

        logger.info(
            f"[{current_date.date()}] 调仓日 - 选股流程 (第 {self.trade_day_seq} 个交易日, 调仓周期={self.rebalance_interval})")

        # 【V7】元标注时取候选池股票（由 candidate_pool_size 控制），非元标注时取 top MAX_POSITIONS
        if self.use_meta_labeling:
            recommended = self._get_primary_recommendations(context, max_n=None)  # 候选池全量
        else:
            recommended = self._get_primary_recommendations(context, max_n=self.MAX_POSITIONS)

        if not recommended:
            self.target_stocks = []

            logger.warning(f"[{current_date.date()}] 主模型未选出任何股票")

            return

        logger.info(f"主模型推荐数量: {len(recommended)}")

        if self.use_meta_labeling:

            # ===== [重训间隔] 每 RETRAIN_INTERVAL 次调仓重训一次, 期间沿用最近模型 =====
            self._rebalances_since_train += 1
            retrain_now, retrain_reason = self._should_retrain_model()
            logger.info(f"[重训间隔] {retrain_reason} → "
                        f"{'执行重训' if retrain_now else '跳过重训, 沿用现有模型'}")

            if not retrain_now:
                # 沿用最近模型: 特征集/meta_auc/AUC门控均沿用上次训练结果, 直接进入选股
                pass
            else:
                # 【修复】增加异常捕获，捕获训练过程中的所有异常
                try:
                    logger.info(f"[{current_date.date()}] 开始训练元标注模型...")

                    train_success = self._train_meta_model(current_date, context, recommended, None, None)

                    if train_success:
                        self._rebalances_since_train = 0
                    else:
                        logger.warning(f"[{current_date.date()}] 元标注模型训练失败，将使用主模型推荐结果")
                        self.target_stocks = recommended[:self.MAX_POSITIONS]
                        self.target_weights = {}  # [v15 P2-2] 训练失败时重置权重，避免使用上期陈旧概率
                        logger.info(f"降级策略: 使用主模型推荐的 {len(self.target_stocks)} 只股票")
                        return

                except Exception as e:
                    import traceback
                    logger.error(f"[{current_date.date()}] 元标注模型训练异常: {e}")
                    logger.error(f"完整堆栈:\n{traceback.format_exc()}")
                    logger.warning(f"[{current_date.date()}] 元标注训练异常，降级使用主模型推荐")
                    self.target_stocks = recommended[:self.MAX_POSITIONS]
                    self.target_weights = {}  # [v15 P2-2] 训练异常时重置权重
                    logger.info(f"降级策略: 使用主模型推荐的 {len(self.target_stocks)} 只股票")
                    return

        # 【修复】增加选股异常捕获
        if self.use_meta_labeling and self.ensemble_manager is not None:
            try:
                result = self._meta_select_stocks(recommended, context)
                if isinstance(result, tuple):
                    self.target_stocks, self.target_weights = result
                else:
                    # 兼容旧返回值
                    self.target_stocks = result
                    self.target_weights = {}
            except Exception as e:
                import traceback
                logger.error(f"[{current_date.date()}] 元标注选股异常: {e}")
                logger.error(f"完整堆栈:\n{traceback.format_exc()}")
                logger.warning(f"[{current_date.date()}] 元标注选股异常，降级使用主模型推荐")
                self.target_stocks = recommended[:self.MAX_POSITIONS]
                self.target_weights = {}
        else:
            self.target_stocks = recommended[:self.MAX_POSITIONS]
            self.target_weights = {}  # 非元标注模式用等权

        # 【R5】IPBS软重排：从扩池中重新排序（λ=0.15，技术评分仅在边际区间翻盘）
        # 构建prob_dict（兼容元标注和降级两种分支）
        if self.use_meta_labeling and self.ensemble_manager is not None:
            prob_dict = self.target_weights if self.target_weights else {c: 0.5 for c in self.target_stocks}
        else:
            prob_dict = {c: 0.5 for c in self.target_stocks}

        if self.target_stocks:
            self.target_stocks, prob_filtered = self._pre_market_tech_filter(
                self.target_stocks, prob_dict, context
            )
            self.target_weights = prob_filtered

        logger.info(f"最终选中 {len(self.target_stocks)} 只股票")

        # ============================================================
        # [v3.3] 机构级仓位控制（BARRA→GARCH 顺序修正版）
        #   执行顺序: AUC 二元门控 → BARRA 个股权重(Σw=1.0) → GARCH 总敞口(用BARRA权重) → 缩放
        #   设计原则:
        #     - AUC 仅回答"模型是否有效"（二元门控）
        #     - BARRA 先运行: 在 Σw=1.0 全投资约束下优化个股权重
        #     - GARCH 用 BARRA 优化权重合成静态持仓历史收益, 估计"真实持仓的波动率"
        #     - 消除"代理组合"与"实际组合"脱节: GARCH建模的波动率精确反映BARRA敞口结构
        # ============================================================

        # ---- 1) 模型有效性门控 ----
        # [2026-09-01 用户裁决] 原 AUC 二元门控废弃: staged_binary 模式由趋势×AUC
        #   联合门控取代; staged_rank 回归模式用 Spearman IC 门控。
        # [Fix-20260907] rank 模式 IC 门控**强制启用**(不再依赖 auc_gate_enabled,
        #   避免用户仅切换 label_mode 后失去模型质量把关); binary 分支行为不变。
        total_exposure = 1.0
        meta_cfg = self.config.get('meta_labeling', {})
        _is_rank_mode = self._is_rank_mode_active()
        _rank_ic_gate_on = _is_rank_mode and (meta_cfg.get('auc_gate_enabled', False)
                                              or getattr(self, '_label_mode', '') == 'staged_rank')
        if _rank_ic_gate_on:
            meta_ic_val = None
            if self.ensemble_manager is not None:
                meta_ic_val = getattr(self.ensemble_manager, 'meta_auc', None)
            if meta_ic_val is None:
                meta_ic_val = getattr(self, 'meta_auc', 0.0)
            # [Fix-20260907] merged-OOS 对照（更真实跨期口径, 仅供诊断）
            _merged_ic = None
            if self.ensemble_manager is not None:
                _merged_ic = getattr(self.ensemble_manager, 'xgb_val_auc', None)
            if meta_ic_val is not None and meta_ic_val < self.IC_GATE_THRESHOLD:
                self.target_stocks = []
                self.target_weights = {}
                logger.warning(f'[策略分配-IC门控] 折集成 IC={meta_ic_val:.4f} < '
                               f'{self.IC_GATE_THRESHOLD} (merged_OOS={_merged_ic if _merged_ic is not None else float("nan"):.4f}) '
                               f'→ 空仓')
                return
            logger.info(f'[策略分配-IC门控] 折集成 IC={meta_ic_val:.4f} >= '
                        f'{self.IC_GATE_THRESHOLD} (merged_OOS={_merged_ic if _merged_ic is not None else float("nan"):.4f}) '
                        f'→ 放行')

        # ---- 2) BARRA 约束优化（先运行，Σw=1.0，得到优化后权重供 GARCH 使用） ----
        barra_weights_raw = None  # 存储 BARRA 优化权重(Σw=1.0)，供 GARCH 使用
        if self._barra_enabled and self._barra_solve is not None and self.target_weights:
            try:
                from strategies.wyckoff20.barra_builder import get_or_build_snapshot
                codes = list(self.target_weights.keys())
                probs = np.array([self.target_weights[c] for c in codes])
                barra_expo = get_or_build_snapshot(current_date)
                if barra_expo is not None and len(barra_expo) > 100:
                    barra_weights_raw, barra_ok = self._barra_solve(
                        preds=probs,
                        selected_codes=codes,
                        barra_expo=barra_expo,
                        risk_aversion=self._barra_cfg.get('risk_aversion', 1.0),
                        max_weight=self._barra_cfg.get('max_weight', 0.10),
                        max_ind_dev=self._barra_cfg.get('max_ind_dev', 0.12),
                        constraint_date=current_date,
                        total_exposure=1.0,  # BARRA 始终全投资约束 Σw=1.0
                        lambda_dev=self._barra_cfg.get('lambda_dev', 0.5),
                        lambda_l2=self._barra_cfg.get('lambda_l2', 0.5),
                        lambda_ind=self._barra_cfg.get('lambda_ind', 0.3),
                    )
                    if barra_ok:
                        logger.info(f"[BARRA/Pre-GARCH] 优化完成: {len(codes)} 只股票, "
                                    f"max_w={max(barra_weights_raw):.3f}")
                    else:
                        barra_weights_raw = None
                        raise RuntimeError("BARRA returned barra_ok=False")
                else:
                    raise RuntimeError(f"BARRA snapshot unavailable")
            except Exception as e:
                logger.warning(f"[BARRA] 优化失败, 降级使用预测概率权重: {e}")
                barra_weights_raw = None

        # ---- 3) GARCH 波动率目标 → 总敞口比例 (使用 BARRA 优化权重) ----
        if self._garch_enabled and self._garch_ctrl is not None and self.target_weights:
            try:
                codes = list(self.target_weights.keys())
                # [v3.3] 优先使用 BARRA 优化权重; 降级时使用预测概率权重
                if barra_weights_raw is not None and len(barra_weights_raw) == len(codes):
                    w_array = np.array(barra_weights_raw)
                    garch_weight_source = 'barra'
                else:
                    w_array = np.array([self.target_weights[c] for c in codes])
                    w_array = w_array / w_array.sum()
                    garch_weight_source = 'pred_probs'

                ret_matrix = self._build_historical_return_matrix(
                    codes, self._garch_ctrl.lookback, current_date, context
                )
                if ret_matrix is not None and ret_matrix.shape[1] >= 3:
                    garch_scale, annual_vol, info = self._garch_ctrl.compute(
                        w_array, ret_matrix, context.portfolio.total_value
                    )
                    total_exposure = max(0.0, min(1.0, garch_scale))
                    logger.info(f'[策略分配] GARCH total_exposure={total_exposure:.3f} '
                                f'annual_vol={annual_vol:.2%} '
                                f'method={info.get("method", "?")} '
                                f'weights={garch_weight_source} '
                                f'dd={info.get("current_drawdown", 0):.2%}')
                else:
                    n_stocks = ret_matrix.shape[1] if ret_matrix is not None else 0
                    logger.debug(f'[策略分配] GARCH 跳过 (有效股票={n_stocks}, 需要≥3)')
            except Exception as e:
                logger.warning(f'[策略分配] GARCH 异常 (保持 total_exposure=1.0): {e}')

        # ---- 4) 缩放 BARRA 权重: w_final = w_barra(Σ=1.0) × total_exposure ----
        if barra_weights_raw is not None and self.target_weights:
            codes = list(self.target_weights.keys())
            if len(barra_weights_raw) == len(codes):
                barra_weights = barra_weights_raw * total_exposure
                self.target_weights = dict(zip(codes, barra_weights))
                logger.info(f"[BARRA] 最终权重: sum_w={sum(barra_weights):.3f}, "
                            f"total_exposure={total_exposure:.3f}, "
                            f"max_w={max(barra_weights):.3f}, "
                            f"HHI={sum(w**2 for w in barra_weights):.3f}")
            else:
                logger.warning(f"[BARRA] 权重-代码数量不匹配, 跳过最终缩放")
        elif self.target_weights:
            # BARRA 降级路径: 使用预测概率权重 × total_exposure
            codes = list(self.target_weights.keys())
            probs = np.array([self.target_weights[c] for c in codes])
            probs = probs / probs.sum()
            w_final = probs * total_exposure
            self.target_weights = dict(zip(codes, w_final))
            logger.info(f"[BARRA/fallback] 预测概率权重缩放: sum_w={sum(w_final):.3f}")

        # ---- [2026-08-29 方案A-1] 组合级回撤熔断 FLAT 清空目标 (调仓日执行; 状态机已每日盘前更新) ----
        if self._dd_state == 2 and self.target_stocks:
            self.target_stocks = []
            self.target_weights = {}
            logger.warning(f'[方案A 回撤熔断] FLAT 状态 → 清空本次调仓目标 (不建仓)')

        if self.target_stocks:
            logger.info(f"选中股票示例: {self.target_stocks[:5]}...")

    # ==================== 执行调仓 ====================
    # 【改造2 + 修正12】使用基本面排名百分位调整止盈止损 + 分钟线跟踪
    def handle_bar(self, context: StrategyContext):
        broker = context.broker
        portfolio = context.portfolio
        date = context.current_dt

        # ---------- 0. 处理到期退出 + IPBS退出（开盘价卖出） ----------
        # 到期退出和IPBS退出均在 before_market_open 中预标记，此处以开盘价执行
        expiry_sells = getattr(self, '_expiry_sells_today', [])

        # 【R5】IPBS持仓退出（去重：跳过已在到期退出列表中的股票）
        expiry_set = set(expiry_sells)
        ipbs_exit_sells = [(c, r) for c, r in getattr(self, '_ipbs_exit_list', []) if c not in expiry_set]

        all_premarket_sells = [(code, '到期退出') for code in expiry_sells] + ipbs_exit_sells
        for code, exit_reason in all_premarket_sells:
            if code in portfolio.positions:
                # 获取买入价和卖出价（真实成交价），计算盈亏
                entry_price_raw = self.holdings.get(code, {}).get('price_raw', None)
                entry_price_adj = self.holdings.get(code, {}).get('price', None)
                sell_price_raw = None
                sell_price_adj = None
                daily = self.data_provider.get_daily_data(code, date, date, reference_date=self.reference_date)
                if not daily.empty:
                    sell_price_adj = daily['open'].iloc[0]
                    sell_price_raw = daily['raw_open'].iloc[0] if 'raw_open' in daily.columns else sell_price_adj
                broker.order_target_percent(code, 0.0)
                # [v13.2] 保存分红历史（需在 del holdings 前读取）
                _saved_hold = self.holdings.get(code, {}).copy() if code in self.holdings else {}
                if code in self.holdings:
                    del self.holdings[code]
                if code in self.volatility_cache:
                    del self.volatility_cache[code]
                # 打印盈亏（真实价×复权价双显示）[v13.2] 价格盈亏=前复权, 含分红=真实净值
                if entry_price_raw is not None and sell_price_raw is not None and entry_price_raw > 0:
                    # 价格盈亏 = 前复权价比较（纯市场涨跌）
                    pnl_pct_adj = (sell_price_adj / entry_price_adj - 1) * 100 if entry_price_adj and sell_price_adj else None
                    pnl_sign = '+' if (pnl_pct_adj or 0) >= 0 else ''
                    # 含分红 = 真实净值比较（前复权收益 - 税收拖累）
                    div_hist = _saved_hold.get('dividend_history', [])
                    pnl_pct_real = pnl_pct_adj  # 默认与前复权相同（无分红时）
                    if div_hist and pnl_pct_adj is not None:
                        buy_qty = _saved_hold.get('quantity', 0)
                        buy_cost = entry_price_raw * buy_qty if buy_qty > 0 else 0
                        if buy_cost > 0:
                            total_tax = sum(d.get('cash_tax', 0) + d.get('bonus_tax_amount', 0) for d in div_hist)
                            tax_drag_pct = total_tax / buy_cost * 100
                            pnl_pct_real = pnl_pct_adj - tax_drag_pct
                    real_sign = '+' if (pnl_pct_real or 0) >= 0 else ''
                    adj_str = f' 价格盈亏={pnl_sign}{pnl_pct_adj:.2f}%' if pnl_pct_adj is not None else ''
                    real_str = f' 含分红={real_sign}{pnl_pct_real:.2f}%' if pnl_pct_real is not None else ''
                    div_summary = self._build_dividend_summary(div_hist)
                    div_lines = chr(10) + div_summary if div_summary else ''
                    logger.info(f"[{date.date()}] {code} {exit_reason}（开盘价）"
                                f" 买入={entry_price_raw:.2f}(复权{entry_price_adj:.2f})"
                                f" 卖出={sell_price_raw:.2f}(复权{sell_price_adj:.2f})"
                                f"{adj_str}{real_str}{div_lines}")
                else:
                    logger.info(f"[{date.date()}] {code} {exit_reason}（开盘价）")
            elif code in self.holdings:
                # 【P0修复】幽灵持仓清理：holdings有但positions无 → 买入订单未成交
                logger.warning(f"[{date.date()}] {code} 幽灵持仓清理"
                               f"（holdings有记录但positions无持仓，可能买入未成交），"
                               f"清理 holdings 记录")
                del self.holdings[code]
                if code in self.volatility_cache:
                    del self.volatility_cache[code]

        # 清理IPBS退出列表（避免跨日残留）
        self._ipbs_exit_list = []

        # ---------- 1. 分阶段止盈止损（替换旧单一屏障） ----------
        is_rebalance = (self.trade_day_seq % self.rebalance_interval == 1)

        if not is_rebalance and self.staged_exit_enabled:
            # 预加载今日所有持仓分钟线（批量利用 provider 缓存）
            minute_map = {}
            for code in list(portfolio.positions.keys()):
                try:
                    mdf = self.data_provider.get_minute_data(
                        code, datetime.combine(date, datetime.min.time()),
                        datetime.combine(date, datetime.max.time()),
                        freq='5', reference_date=self.reference_date)
                    minute_map[code] = mdf if not mdf.empty else None
                except Exception:
                    minute_map[code] = None

            for code in list(portfolio.positions.keys()):
                if code not in self.holdings:
                    # 孤儿清理
                    logger.warning(f"[{date.date()}] {code} 孤儿持仓，清理")
                    broker.order_target_percent(code, 0.0, trade_reason='orphan_cleanup')
                    if code in self.volatility_cache: del self.volatility_cache[code]
                    continue

                tracker = self.stage_mgr.trackers.get(code)
                if tracker is None:
                    # 无 tracker（防御：可能调仓日未注册，或旧持仓残留）
                    # 直接用旧单一屏障兜底（安全网）
                    entry_price = self.holdings[code].get('price', 0)
                    entry_price_raw = self.holdings[code].get('price_raw', entry_price)
                    vol = self.holdings[code].get('daily_vol', 0.02)
                    if vol <= 0: vol = 0.02
                    pt_th = entry_price * (1 + 6.3 * vol)
                    sl_th = entry_price * (1 - 4.7 * vol)
                    ttype, tprice, _ttime = self._check_pt_sl_minute(code, date, pt_th, sl_th)
                    if ttype is not None:
                        self._sell_at_trigger(code, date, ttype, tprice, tprice / entry_price * entry_price_raw, broker)
                    continue

                # [2026-09-02 方案A] S1-S4 由 stageX_done 标记无条件同时检查 (σ事件驱动,
                #   无持有期门槛, 单日大跌可击穿全清 — 经回测验证为合理设计)。
                #   current_stage/held_days 参数已废弃 (函数体内从未使用), 清理冗余。
                mdf = minute_map.get(code)
                if mdf is None:
                    events = self._check_staged_daily_fallback(code, date, tracker)
                else:
                    events = self._check_staged_minute(code, date, tracker, mdf)

                for (stg, is_pt, trig_price_adj, trig_time) in events:
                    # 换算真实价
                    trig_price_raw = self._adj_to_raw_price(code, date, trig_price_adj)
                    sell_pct = self.STAGE_CONFIG[stg]['pct']
                    sell_qty = tracker.original_entry_qty * sell_pct  # 始终基于建仓基准
                    broker.order_value(
                        code, -sell_qty * trig_price_raw,
                        trade_time=trig_time, exec_price_override=trig_price_raw,
                        apply_slippage=True,
                        trade_reason=f'stage{stg}_{"pt" if is_pt else "sl"}')
                    # 标记阶段完成 + PT hit 标记（利润锁定SL上移用）
                    if stg == 1:
                        tracker.stage1_done = True
                        if is_pt: tracker.s1_pt_hit = True
                        else: tracker.pt_after_sl = True  # SL触发→PT下移
                    elif stg == 2:
                        tracker.stage2_done = True
                        if is_pt: tracker.s2_pt_hit = True
                        else: tracker.pt_after_sl = True
                    elif stg == 3:
                        tracker.stage3_done = True
                        if is_pt: tracker.s3_pt_hit = True
                        else: tracker.pt_after_sl = True
                        if is_pt: tracker.s3_pt_hit = True
                    elif stg == 4:
                        tracker.stage4_done = True
                    # 回收现金到池
                    self.stage_mgr.freed_cash_pool += sell_qty * trig_price_raw * 0.999
                    # PT 触发后记录 SL 上移日志
                    if is_pt and stg <= 3:
                        new_sl = tracker.profit_lock_sl()
                        if new_sl > float('-inf'):
                            logger.info(
                                f"[{date.date()}] {code} Stage{stg} 止盈触发 → "
                                f"SL 上移至 {new_sl:+.1f}σ (利润锁定)")
                    # SL 触发后记录 PT 下移日志
                    if not is_pt and stg <= 2 and tracker.pt_after_sl:
                        logger.info(
                            f"[{date.date()}] {code} Stage{stg} 止损触发 → "
                            f"PT 下移至 1.5σ/2.0σ (快速退出)")
                    logger.info(
                        f"[{date.date()}] {code} Stage{stg} "
                        f"{'止盈' if is_pt else '止损'}触发 "
                        f"卖{int(sell_pct*100)}% @ {trig_price_raw:.2f}(真实) "
                        f"σ_up={tracker.cum_max_up/tracker.daily_vol:.2f} "
                        f"σ_dn={tracker.cum_max_dn/tracker.daily_vol:.2f}")

                # S4 完成后不立即删 tracker/holdings：剩余零股/再投资后仍可能在持仓中
                # 由调仓日 reset() 统一清理，避免提前删除造成"孤儿持仓"误判

            # 闲置现金再投资
            if self.reinvest_enabled and self.stage_mgr.freed_cash_pool > self.reinvest_min_amount:
                self._reinvest_freed_cash(broker, portfolio)

        elif is_rebalance:
            # 调仓日: 分阶段跳过, 所有持仓在下方调仓环节统一全清
            pass

        # ---------- 2. 调仓逻辑（仅在调仓日执行） ----------
        if is_rebalance:
            # [v3.6] 分阶段退出: 调仓日全量清仓（不留任何部分卖出残留）

            # Step A: 强制清空所有持仓
            for code in list(portfolio.positions.keys()):
                broker.order_target_percent(code, 0.0, trade_reason='rebalance_clear_all')
            # 清理 holdings + volatility 缓存
            for code in list(self.holdings.keys()):
                if code in self.holdings: del self.holdings[code]
            for code in list(self.volatility_cache.keys()):
                del self.volatility_cache[code]
            # 清除分阶段状态
            self.stage_mgr.reset()

            # Step B: 按目标权重重新建仓
            # [2026-08-29 方案A] 建仓规模 = 回撤熔断 scale × 趋势过滤 scale (只限建仓, 不强清)
            self._update_rebalance_trend_gate(date)
            build_scale = self._current_build_scale()
            if build_scale < 1.0:
                logger.info(f'[{date.date()}] [方案A] 建仓规模 x{build_scale:.2f} '
                            f'(回撤熔断={self._dd_scale}, 趋势={self._trend_scale})')
            # [Fix-20260907] build_scale<=0 → 明确空仓语义: 跳过建仓循环,
            #   避免对 target 下 0 仓单产生"买入未成交回滚"误报 (x0.00 应直接空仓)
            if build_scale <= 0:
                logger.info(f'[{date.date()}] [方案A] 建仓规模=0 (回撤熔断={self._dd_scale}, '
                            f'趋势={self._trend_scale}) → 本次调仓保持空仓 (不建仓)')
                return
            held_codes = set()
            for code, target_w in sorted(self.target_weights.items(), key=lambda x: x[1], reverse=True):
                if target_w <= 0: continue
                open_p_adj, open_p_raw = self._get_open_price(code, date)
                if open_p_adj is None: continue

                # 计算 EWMA vol
                dv = self._compute_entry_vol(code, date)
                if dv <= 0: dv = 0.02

                from .barriers import compute_barriers as _cb
                pt, sl = _cb(open_p_adj, dv, pt_mult=self.meta_labeler.pt_mult,
                             sl_mult=self.meta_labeler.sl_mult, hold_days=1)

                self.holdings[code] = {
                    'date': date, 'price': open_p_adj, 'price_raw': open_p_raw,
                    'pt': pt, 'sl': sl, 'daily_vol': dv, 'dividend_history': [],
                }

                logger.info(f'[{date.date()}] {code} 建仓 买入={open_p_raw:.2f}'
                            f'(复权{open_p_adj:.2f}) PT={pt:.2f} SL={sl:.2f} '
                            f'仓位={target_w:.1%}')
                broker.order_target_percent(code, target_w * build_scale, trade_reason='new_open')
                if code not in portfolio.positions:
                    if code in self.holdings: del self.holdings[code]
                    logger.warning(f"[{date.date()}] {code} 买入订单未成交，回滚")
                else:
                    if code in self.holdings:
                        self.holdings[code]['quantity'] = float(portfolio.positions[code].quantity)

            # Step C: 为所有新持仓初始化 StageTracker
            self.stage_mgr.reset()
            for code in list(portfolio.positions.keys()):
                h = self.holdings.get(code, {})
                self.stage_mgr.register(
                    code=code, entry_date=date,
                    entry_adj=h.get('price', 0),
                    entry_raw=h.get('price_raw', 0),
                    qty=float(portfolio.positions[code].quantity),
                    daily_vol=h.get('daily_vol', 0.02))

            logger.info(f"[{date.date()}] 调仓完成，持仓数: {len(portfolio.positions)} "
                        f"staged: {len(self.stage_mgr.trackers)}")

        # ---------- 3. 每日资产日志 ----------
        logger.info(
            f"[{date.date()}] 总资产: {portfolio.total_value:,.0f}, "
            f"现金: {portfolio.cash:,.0f}, 持仓数: {len(portfolio.positions)}"
        )

        # [2026-08-29 方案A] 回撤跟踪 (日终, T+1 盘前状态机使用, 无未来)
        self._prev_nav = portfolio.total_value
        self._dd_peak = max(self._dd_peak, portfolio.total_value)

    # ==================== 方案A: 组合级回撤熔断 + 调仓日趋势过滤 ====================

    def _load_breaker_index_close(self, index_code: str):
        """预加载指数日线收盘序列 (data/index_daily/{code}.parquet, ≤T-1 用, 严格无未来)。"""
        try:
            path = os.path.join('data', 'index_daily', f'{index_code}.parquet')
            df = pd.read_parquet(path)
            date_col = 'trade_date' if 'trade_date' in df.columns else 'date'
            c = df[[date_col, 'close']].copy()
            c[date_col] = pd.to_datetime(c[date_col])
            return c.set_index(date_col).sort_index()['close']
        except Exception as e:
            logger.warning(f'[方案A] 指数 {index_code} 加载失败: {e} → 趋势判断恒 False')
            return None

    def _breaker_index_down(self, date, ma) -> bool:
        """指数收盘 < MA(ma) (≤T-1, 严格无未来)。"""
        if self._dd_index_close is None or ma <= 0:
            return False
        px = self._dd_index_close[self._dd_index_close.index < pd.Timestamp(date)]
        if len(px) < ma + 3:
            return False
        return float(px.iloc[-1]) < float(px.iloc[-ma:].mean())

    def _update_dd_breaker(self, date):
        """组合级回撤熔断状态机 (触发保守优先 + 市场信号恢复 + 基线重置)。
        [2026-09-02] FLAT→HALF 恢复基线重置至 FLAT 触发净值 (原峰值重置至当前净值
        会赦免历史亏损 → 熔断失效, 2026-07 回撤 -19.37% 实证)。

        触发: 回撤>dd_flat→FLAT(scale 0); 回撤>dd_half→HALF(scale 0.5)。
        恢复: FLAT→HALF 需指数≥MA 连续 trend_confirm_days 日 + 基线重置
              [2026-09-02] 基线 = FLAT 触发净值 (原"峰值重置"至当前净值会赦免亏损)
              ——空仓净值恒定, 净值恢复条件死锁(grid_lowvol 教训);
              HALF→NORMAL 需回撤收窄 < dd_recover_full (半仓净值会动, 可达)。
        触发守卫: 已在 FLAT 不重复触发 (否则 dd 恒定-15% 永远锁死, 恢复不可达)。
        """
        c = self._dd_cfg
        if not c['enabled']:
            self._dd_state, self._dd_scale = 0, 1.0
            return
        dd = self._prev_nav / self._dd_peak - 1.0 if self._dd_peak > 0 else 0.0
        idx_down = self._breaker_index_down(date, c['index_ma'])
        self._dd_above_days = 0 if idx_down else self._dd_above_days + 1

        if dd <= -c['dd_flat'] + 1e-9 and self._dd_state != 2:
            self._dd_state = 2
            self._dd_scale = 0.0
            self._dd_above_days = 0          # FLAT 后重新累计 (防恢复过快)
            # [2026-09-02 修复] 记录 FLAT 触发时净值, 作为恢复回撤基线 (与 alpha191 同步)
            self._dd_flat_nav = self._prev_nav
            logger.warning(f'[方案A 回撤熔断] 回撤={dd:.2%} ≤ {c["dd_flat"]:.0%} → FLAT 不建仓 '
                           f'(基线 {self._dd_flat_nav:.0f})')
        elif dd <= -c['dd_half'] + 1e-9 and self._dd_state == 0:
            self._dd_state = 1
            self._dd_scale = 0.5
            logger.warning(f'[方案A 回撤熔断] 回撤={dd:.2%} ≤ {c["dd_half"]:.0%} → HALF 建仓减半')
        else:
            if self._dd_state == 2 and self._dd_above_days >= c['trend_confirm_days']:
                self._dd_state = 1
                self._dd_scale = 0.5
                # [2026-09-02 修复] 原实现 _dd_peak = self._prev_nav (当前净值) →
                #   赦免全部历史亏损 → 熔断失效。现重置到 FLAT 触发时净值:
                #   - 接受 FLAT 前的已发生亏损 (回撤基线 = FLAT 入场净值)
                #   - FLAT 期间的继续下跌仍计入回撤 → 熔断门槛持续有效
                #   - HALF 期间回撤收窄 > -dd_recover_full (相对此基线) 才 NORMAL, 可达 (半仓净值会动)
                #   (与 alpha191 同步修复)
                self._dd_peak = self._dd_flat_nav if self._dd_flat_nav else self._prev_nav
                self._dd_above_days = 0
                logger.warning(f'[方案A 回撤熔断] 指数确认{c["trend_confirm_days"]}日 → FLAT→HALF '
                               f'(回撤基线重置至 FLAT 触发净值 {self._dd_peak:.0f}, 历史亏损已确认)')
            elif self._dd_state == 1 and dd > -c['dd_recover_full'] + 1e-9:
                self._dd_state = 0
                self._dd_scale = 1.0
                logger.info(f'[方案A 回撤熔断] 回撤收窄 {dd:.2%} > -{c["dd_recover_full"]:.0%} → NORMAL 满仓')

    def _is_rank_mode_active(self) -> bool:
        """staged_rank（分位回归）模式判定：策略 label_mode 或模型 objective_mode 任一命中。

        rank 模式下模型度量为 Spearman IC（值域远小于 AUC 0.5+），
        因此 AUC 语义的趋势门控/二元门控一律不适用，改由 IC 门控裁决模型质量。
        """
        if getattr(self, '_label_mode', '') == 'staged_rank':
            return True
        em = getattr(self, 'ensemble_manager', None)
        if em is not None and getattr(em, 'objective_mode', 'binary') == 'regression':
            return True
        return False

    def _update_rebalance_trend_gate(self, date):
        """调仓日趋势 × AUC 联合门控 (2026-09-01 用户裁决, 取代原趋势过滤+AUC二元门控)。

        决策矩阵 (参数全部来自 config rebalance_trend_gate 段):
          指数收盘 >= MA(趋势强): meta_AUC > auc_threshold_up → scale_up_ok; 否则 scale_up_bad
          指数收盘 <  MA(趋势弱): meta_AUC > auc_threshold_dn → scale_dn_ok; 否则 scale_dn_bad
        只影响调仓日建仓 (不强清); 与回撤熔断 dd_scale 相乘合成最终建仓规模。
        AUC = 调仓日重训后的折集成 meta_AUC (重训跳过时沿用上次值, 与原 AUC 门控同口径)。
        """
        tg = self._trend_gate_cfg
        if not tg['enabled']:
            self._trend_scale = 1.0
            return
        # [Fix-20260907] rank(分位回归)模式: meta 度量为 Spearman IC(≈0~0.1),
        #   与 AUC 阈值(0.52/0.54)不可比 → 恒触发 bad 档 → 误空仓/半仓。
        #   rank 模式下模型质量由策略端 IC 门控(ic_gate_threshold)裁决，
        #   此处趋势门控跳过 AUC 比较(不强加降仓)，回撤熔断 dd_scale 仍全局生效。
        if self._is_rank_mode_active():
            self._trend_scale = 1.0
            logger.info('[方案A 趋势门控] staged_rank 模式: 跳过 AUC 语义门控 '
                        '(模型质量由 IC 门控裁决, 仅回撤熔断生效)')
            return
        down = self._breaker_index_down(date, tg['ma'])
        thr = tg['auc_threshold_dn'] if down else tg['auc_threshold_up']
        ok_scale = tg['scale_dn_ok'] if down else tg['scale_up_ok']
        bad_scale = tg['scale_dn_bad'] if down else tg['scale_up_bad']
        auc = getattr(self, 'meta_auc', 0.5)
        if auc is None:
            auc = 0.5
        self._trend_scale = ok_scale if auc > thr else bad_scale
        logger.info(f'[方案A 趋势AUC门控] 指数{"<" if down else ">="}MA{tg["ma"]} '
                    f'AUC={auc:.4f} vs 阈值{thr:.2f} → 建仓 x{self._trend_scale}')

    def _current_build_scale(self) -> float:
        """组合建仓规模因子 = 回撤熔断 scale × 趋势过滤 scale。"""
        return float(self._dd_scale) * float(self._trend_scale)

    # ==================== 分阶段止盈止损 辅助方法 ====================

    def _get_open_price(self, code, date):
        daily = self.data_provider.get_daily_data(code, date, date, reference_date=self.reference_date)
        if not daily.empty:
            adj = daily['open'].iloc[0]
            raw = daily['raw_open'].iloc[0] if 'raw_open' in daily.columns else adj
            return adj, raw
        return None, None

    def _compute_entry_vol(self, code, date):
        """EWMA daily vol (λ=0.94, seed=60d) — 与持仓端 volatile 缓存一致"""
        if not self.meta_labeler: return 0.02
        vol_method = self.config.get('meta_labeling', {}).get('vol_method', 'rolling')
        lookback = max(self.meta_labeler.daily_vol_window, self.meta_labeler.vol_ewma_seed_days)
        hist = self.data_provider.get_daily_data(
            code, date - timedelta(days=lookback * 2), date, reference_date=self.reference_date)
        if hist.empty: return 0.02
        if 'raw_close' in hist.columns and 'backAdjustFactor' in hist.columns:
            adj = hist['raw_close'] * hist['backAdjustFactor']; rets = adj.pct_change().dropna()
        else:
            rets = hist['close'].pct_change().dropna()
        if len(rets) < 2: return 0.02
        if vol_method == 'ewma':
            lambd = self.config.get('meta_labeling', {}).get('vol_ewma_lambda', 0.94)
            ev = rets.ewm(alpha=1-lambd, adjust=False).var().iloc[-1]
            return float(np.sqrt(ev)) if pd.notna(ev) and ev > 0 else 0.02
        dv = rets.std(); return dv if not (np.isnan(dv) or dv == 0) else 0.02

    def _adj_to_raw_price(self, code, date, adj_price):
        """复权价→真实价（方案A: raw_close/adj_close 比率）"""
        try:
            dd = self.data_provider.get_daily_data(code, date, date, reference_date=self.reference_date)
            if not dd.empty:
                row = dd.iloc[0]
                rc = float(row.get('raw_close', np.nan))
                ac = float(row.get('close', np.nan))
                if pd.notna(rc) and pd.notna(ac) and rc > 0 and ac > 0:
                    return adj_price * rc / ac
        except Exception: pass
        return adj_price

    def _check_staged_minute(self, code, date, tracker, minute_df):
        """分阶段触发检查（5min 逐 bar）。

        [2026-09-02 方案A] stage/held_days 参数已废弃移除: S1-S4 由 stageX_done 标记
        无条件同时检查 (σ 事件驱动, 无持有期门槛; 单日大跌可击穿全清 — 回测验证合理)。
        """
        events = []
        entry = tracker.entry_adj; vol = tracker.daily_vol
        if vol <= 0: return events
        cum_u, cum_d = tracker.cum_max_up, tracker.cum_max_dn

        for _, bar in minute_df.iterrows():
            # [E6-2026-09-10] 事件驱动逐 bar 生效: pl_sl 于每根 bar 起点按 flags 刷新
            #   (上一根 bar 内的 PT 事件自本根 bar 起抬高止损线; 同 bar 内保持保守)
            pl_sl = tracker.profit_lock_sl()  # 利润锁定SL（PT触发后上移）
            hi = bar['high']; lo = bar['low']
            if not np.isnan(hi): cum_u = max(cum_u, hi / entry - 1)
            if not np.isnan(lo): cum_d = min(cum_d, lo / entry - 1)
            up_s = (hi / entry - 1) / vol if not np.isnan(hi) else float('-inf')  # [PTFIX] 当根触达
            # SL用当前bar的low收益率（不用累计最大回撤）
            # 原因: profit_lock_sl为正数时, dn_s(累计回撤,常为负) <= eff_sl(正数)恒为True
            cur_s = (lo / entry - 1) / vol if not np.isnan(lo) and entry > 0 else float('inf')

            # 硬兜底（任意阶段, 未全清时触发）
            if cur_s <= self.hard_sl_sigma and not tracker.stage4_done:
                tt = str(bar.get('trade_time', ''))
                hard_price = entry * (1 + self.hard_sl_sigma * vol)  # 阈值价而非 bar low
                events.append((4, False, hard_price, tt))
                tracker.cum_max_up, tracker.cum_max_dn = cum_u, cum_d
                return events

            # 利润锁定SL：取 max(staged_sl, profit_lock_sl)
            # PT触发后SL上移，保护已有利润
            eff_sl_1 = max(self.STAGE_CONFIG[1]['sl'], pl_sl)
            eff_sl_2 = max(self.STAGE_CONFIG[2]['sl'], pl_sl)
            eff_sl_3 = max(self.STAGE_CONFIG[3]['sl'], pl_sl)
            eff_sl_4 = max(self.STAGE_CONFIG[4]['sl'], pl_sl)

            # SL后PT下移: 跌到-2σ后剩余仓位降低PT (分钟线实证: 15%先反弹到+1.5σ)
            if tracker.pt_after_sl:
                eff_pt_2 = tracker.pt_after_sl_s2; eff_pt_3 = tracker.pt_after_sl_s3
            else:
                eff_pt_2 = self.STAGE_CONFIG[2]['pt']; eff_pt_3 = self.STAGE_CONFIG[3]['pt']

            # S1 — [O3-20260904] 同bar PT/SL歧义→SL优先(保守, 与标签端同步);
            #        触发后立即设 flag，防止同一日内后续 bar 重复触发
            if not tracker.stage1_done:
                if cur_s <= eff_sl_1:
                    sl_price = entry * (1 + eff_sl_1 * vol)  # 阈值价而非 bar low
                    events.append((1, False, sl_price, str(bar.get('trade_time', ''))))
                    tracker.stage1_done = True
                    tracker.pt_after_sl = True  # [E6] SL事件自下一根bar起下移PT
                elif up_s >= self.STAGE_CONFIG[1]['pt']:
                    pt_price = entry * (1 + self.STAGE_CONFIG[1]['pt'] * vol)  # 阈值价而非 bar high
                    events.append((1, True, pt_price, str(bar.get('trade_time', ''))))
                    tracker.stage1_done = True
                    tracker.s1_pt_hit = True

            # S2 — 同上
            if not tracker.stage2_done:
                if cur_s <= eff_sl_2:
                    sl_price = entry * (1 + eff_sl_2 * vol)
                    events.append((2, False, sl_price, str(bar.get('trade_time', ''))))
                    tracker.stage2_done = True
                    tracker.pt_after_sl = True  # [E6] SL事件自下一根bar起下移PT
                elif up_s >= eff_pt_2:
                    pt_price = entry * (1 + eff_pt_2 * vol)
                    events.append((2, True, pt_price, str(bar.get('trade_time', ''))))
                    tracker.stage2_done = True
                    tracker.s2_pt_hit = True

            # S3 (不终止，继续检查 S4)
            if not tracker.stage3_done:
                if cur_s <= eff_sl_3:
                    sl_price = entry * (1 + eff_sl_3 * vol)
                    events.append((3, False, sl_price, str(bar.get('trade_time', ''))))
                    tracker.stage3_done = True
                    tracker.pt_after_sl = True  # [E6] SL事件自下一根bar起下移PT
                elif up_s >= eff_pt_3:
                    pt_price = entry * (1 + eff_pt_3 * vol)
                    events.append((3, True, pt_price, str(bar.get('trade_time', ''))))
                    tracker.stage3_done = True
                    tracker.s3_pt_hit = True

            # S4 (全清最后 20%)
            if not tracker.stage4_done:
                if cur_s <= eff_sl_4:
                    sl_price = entry * (1 + eff_sl_4 * vol)
                    tt = str(bar.get('trade_time', ''))
                    events.append((4, False, sl_price, tt))
                    tracker.cum_max_up, tracker.cum_max_dn = cum_u, cum_d
                    return events
                elif up_s >= self.STAGE_CONFIG[4]['pt']:
                    pt_price = entry * (1 + self.STAGE_CONFIG[4]['pt'] * vol)
                    tt = str(bar.get('trade_time', ''))
                    events.append((4, True, pt_price, tt))
                    tracker.cum_max_up, tracker.cum_max_dn = cum_u, cum_d
                    return events

        tracker.cum_max_up, tracker.cum_max_dn = cum_u, cum_d
        return events

    def _check_staged_daily_fallback(self, code, date, tracker):
        """分钟线不可用时回退日线 high/low 近似检查。

        [2026-09-02 方案A] stage/held_days 参数已废弃移除 (同 _check_staged_minute)。
        """
        entry = tracker.entry_adj; vol = tracker.daily_vol
        if vol <= 0: return []
        pl_sl = tracker.profit_lock_sl()  # 利润锁定SL
        try:
            dd = self.data_provider.get_daily_data(code, date, date, reference_date=self.reference_date)
            if dd.empty: return []
            h = dd['high'].iloc[0]; l = dd['low'].iloc[0]
            up_s = (h / entry - 1) / vol  # [PTFIX] 当根触达
            # SL用当日low收益率（不用累计最大回撤，同 minute 版修复）
            cur_s = (l / entry - 1) / vol if entry > 0 else float('inf')
            events = []
            if cur_s <= self.hard_sl_sigma and not tracker.stage4_done:
                hard_price = entry * (1 + self.hard_sl_sigma * vol)
                events.append((4, False, hard_price, f"{date.date()} 09:30:00"))
                tracker.stage4_done = True
                return events
            eff_sl_1 = max(self.STAGE_CONFIG[1]['sl'], pl_sl)
            eff_sl_2 = max(self.STAGE_CONFIG[2]['sl'], pl_sl)
            eff_sl_3 = max(self.STAGE_CONFIG[3]['sl'], pl_sl)
            eff_sl_4 = max(self.STAGE_CONFIG[4]['sl'], pl_sl)
            # SL后PT下移
            if tracker.pt_after_sl:
                eff_pt_2 = tracker.pt_after_sl_s2; eff_pt_3 = tracker.pt_after_sl_s3
            else:
                eff_pt_2 = self.STAGE_CONFIG[2]['pt']; eff_pt_3 = self.STAGE_CONFIG[3]['pt']
            if not tracker.stage1_done:
                if cur_s <= eff_sl_1:
                    sl_price = entry * (1 + eff_sl_1 * vol)
                    events.append((1, False, sl_price, f"{date.date()} 09:30:00"))
                    tracker.stage1_done = True
                    tracker.pt_after_sl = True  # [E6] SL触发→PT下移 (单bar/日: 下次调用生效)
                elif up_s >= self.STAGE_CONFIG[1]['pt']:
                    pt_price = entry * (1 + self.STAGE_CONFIG[1]['pt'] * vol)
                    events.append((1, True, pt_price, f"{date.date()} 09:30:00"))
                    tracker.stage1_done = True
                    tracker.s1_pt_hit = True
            if not tracker.stage2_done:
                if cur_s <= eff_sl_2:
                    sl_price = entry * (1 + eff_sl_2 * vol)
                    events.append((2, False, sl_price, f"{date.date()} 09:30:00"))
                    tracker.stage2_done = True
                    tracker.pt_after_sl = True  # [E6] SL触发→PT下移 (单bar/日: 下次调用生效)
                elif up_s >= eff_pt_2:
                    pt_price = entry * (1 + eff_pt_2 * vol)
                    events.append((2, True, pt_price, f"{date.date()} 09:30:00"))
                    tracker.stage2_done = True
                    tracker.s2_pt_hit = True
            if not tracker.stage3_done:
                if cur_s <= eff_sl_3:
                    sl_price = entry * (1 + eff_sl_3 * vol)
                    events.append((3, False, sl_price, f"{date.date()} 09:30:00"))
                    tracker.stage3_done = True
                    tracker.pt_after_sl = True  # [E6] SL触发→PT下移 (单bar/日: 下次调用生效)
                elif up_s >= eff_pt_3:
                    pt_price = entry * (1 + eff_pt_3 * vol)
                    events.append((3, True, pt_price, f"{date.date()} 09:30:00"))
                    tracker.stage3_done = True
                    tracker.s3_pt_hit = True
            if not tracker.stage4_done:
                if cur_s <= eff_sl_4:
                    sl_price = entry * (1 + eff_sl_4 * vol)
                    events.append((4, False, sl_price, f"{date.date()} 09:30:00"))
                    tracker.stage4_done = True
                elif up_s >= self.STAGE_CONFIG[4]['pt']:
                    pt_price = entry * (1 + self.STAGE_CONFIG[4]['pt'] * vol)
                    events.append((4, True, pt_price, f"{date.date()} 09:30:00"))
                    tracker.stage4_done = True
            return events
        except Exception:
            return []

    def _sell_at_trigger(self, code, date, trigger_type, trigger_price_adj, trigger_price_raw, broker):
        """兜底: 旧单一屏障逻辑触发（用于无 tracker 的持仓）"""
        trade_reason = 'profit_take' if trigger_type == 'profit_take' else 'stop_loss'
        broker.order_target_percent(code, 0.0, exec_price_override=trigger_price_raw,
                                    apply_slippage=True, trade_reason=trade_reason)
        if code in self.holdings: del self.holdings[code]
        if code in self.volatility_cache: del self.volatility_cache[code]

    def _reinvest_freed_cash(self, broker, portfolio):
        """闲置现金按目标权重比例再投资于剩余持仓"""
        pool = self.stage_mgr.freed_cash_pool
        if pool <= self.reinvest_min_amount: return
        active = [c for c in portfolio.positions if c in self.target_weights
                  and c in self.stage_mgr.trackers
                  and not self.stage_mgr.trackers[c].stage3_done]
        if not active:
            self.stage_mgr.freed_cash_pool = 0.0; return
        total_w = sum(self.target_weights.get(c, 0) for c in active)
        for code in active:
            add_val = pool * self.target_weights[code] / total_w
            if add_val < self.reinvest_min_amount: continue
            success = broker.order_value(code, add_val, trade_reason='reinvest')
            if success:
                # 更新 entry_qty 反映当前持仓股数（original_entry_qty 不变 = 基准固定）
                t = self.stage_mgr.trackers[code]
                t.entry_qty = float(portfolio.positions[code].quantity)
                pool -= add_val
        self.stage_mgr.freed_cash_pool = max(pool, 0.0)

    # ==================== 除权事件回调 ====================

    def on_ex_dividend(self, code: str, date, stock_ratio: float,
                       cash_per_share: float, bonus_ps: float,
                       net_dividend: float = 0.0, bonus_tax_amount: float = 0.0,
                       cash_tax: float = 0.0):
        """
        引擎回调 [v13]：除权事件发生后更新策略内部追踪。

        PT/SL 阈值基于前复权价，不受除权影响，无需调整。
        price_raw 不调整（保留原始买入价用于含分红盈亏对比）。
        """
        if code in self.holdings:
            h = self.holdings[code]
            if 'dividend_history' not in h:
                h['dividend_history'] = []
            h['dividend_history'].append({
                'date': date,
                'stock_ratio': stock_ratio,
                'cash_per_share': cash_per_share,
                'bonus_ps': bonus_ps,
                'net_dividend': net_dividend,
                'bonus_tax_amount': bonus_tax_amount,
                'cash_tax': cash_tax,
            })
        # [v3.6] 同步 StageTracker qty (送股/转增后股数变化)
        if stock_ratio > 0:
            self.stage_mgr.adjust_qty(code, 1.0 + stock_ratio)

    # ==================== 分红历史展示辅助 ====================

    @staticmethod
    def _build_dividend_summary(dividend_history):
        """构建分红历史展示文本（多行）[v13.2] 含税收拖累"""
        if not dividend_history:
            return ''
        total_net_div = sum(d.get('net_dividend', 0) for d in dividend_history)
        total_btax = sum(d.get('bonus_tax_amount', 0) for d in dividend_history)
        total_cash_tax = sum(d.get('cash_tax', 0) for d in dividend_history)
        total_tax = total_cash_tax + total_btax
        div_net = total_net_div - total_btax  # 税后分红净收入
        parts = []
        for d in dividend_history:
            desc_parts = []
            if d.get('cash_per_share', 0) > 0:
                desc_parts.append('派' + str(round(d['cash_per_share'], 3)) + '元')
            if d.get('stock_ratio', 0) > 0:
                desc_parts.append('送转' + str(int(d['stock_ratio'] * 100)) + '%')
            parts.append(d['date'].strftime('%m-%d') + '(' + '+'.join(desc_parts) + ')')
        event_line = '  期间除权: ' + ' | '.join(parts) if parts else ''
        income_line = '  分红净收入: {:+,.2f} (税后分红{:,.2f} - 送股税{:,.2f})'.format(
            div_net, total_net_div, total_btax)
        tax_line = '  税收拖累: {:,.2f} (现金税{:,.2f} + 送股税{:,.2f})'.format(
            total_tax, total_cash_tax, total_btax)
        return event_line + chr(10) + income_line + chr(10) + tax_line

    # ==================== [v3.0] 辅助方法：历史收益率矩阵构建 ====================
    def _build_historical_return_matrix(self, codes, lookback, current_date, context):
        """
        构建过去 lookback 个交易日的个股前复权收益率矩阵。
        用于 GARCH 虚拟净值合成。

        Args:
            codes: 股票代码列表
            lookback: 回看交易日数
            current_date: 当前回测日期
            context: StrategyContext
        Returns:
            (T, N) ndarray 或 None
        """
        n_codes = len(codes)
        fail_reasons = {}  # {code: reason} for diagnostic logging

        try:
            # 获取交易日历
            trade_dates_df = get_trade_calendar(
                start_date=(current_date - timedelta(days=lookback * 3)).strftime('%Y-%m-%d'),
                end_date=current_date.strftime('%Y-%m-%d'),
            )
            if trade_dates_df is None or len(trade_dates_df) < lookback:
                logger.debug(f"[GARCH] 交易日历不足: need={lookback}, "
                            f"got={len(trade_dates_df) if trade_dates_df is not None else 0}")
                return None

            # Extract calendar_date column from DataFrame -> sorted list
            trade_dates_list = sorted(trade_dates_df['calendar_date'].tolist())
            hist_dates = trade_dates_list[-lookback:]
            start_dt = hist_dates[0]
            end_dt = hist_dates[-1]

            ret_dict = {}
            for code in codes:
                try:
                    daily = self.data_provider.get_daily_data(
                        code, start_dt, end_dt, reference_date=self.reference_date
                    )
                    if daily is None:
                        fail_reasons[code] = 'daily is None'
                        continue
                    if daily.empty:
                        fail_reasons[code] = 'daily empty'
                        continue
                    if len(daily) < lookback // 2:
                        fail_reasons[code] = f'len={len(daily)} < {lookback // 2}'
                        continue

                    daily = daily.set_index('trade_date').sort_index()
                    daily['ret'] = daily['close'].pct_change()
                    rets = daily['ret'].dropna().values
                    if len(rets) < lookback // 2:
                        fail_reasons[code] = f'valid_ret={len(rets)} < {lookback // 2}'
                        continue
                    ret_dict[code] = rets[-lookback:]
                except Exception as e:
                    fail_reasons[code] = f'{type(e).__name__}: {str(e)[:40]}'

            # Diagnostic: report failure distribution
            n_success = len(ret_dict)
            n_fail = n_codes - n_success
            if n_fail > 0:
                reason_counts = {}
                for reason in fail_reasons.values():
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                top_reasons = sorted(reason_counts.items(), key=lambda x: -x[1])[:3]
                reason_str = ' | '.join(f'{r}({c}x)' for r, c in top_reasons)
                logger.warning(f"[GARCH] 收益率矩阵: {n_success}/{n_codes} 成功, "
                               f"{n_fail} 失败 | {reason_str}")

            if n_success < 3:
                logger.debug(f"[GARCH] 有效股票不足: {n_success} < 3, 跳过")
                return None

            # Build aligned matrix
            ret_matrix = np.zeros((lookback, n_success))
            for i, (code, rets) in enumerate(ret_dict.items()):
                n = min(lookback, len(rets))
                ret_matrix[-n:, i] = rets[-n:]

            return ret_matrix
        except Exception as e:
            logger.warning(f"[GARCH] 收益率矩阵构建异常: {type(e).__name__}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    # ==================== 分钟K线止盈止损检查 ====================
    # 【修正12】新增
    def _check_pt_sl_minute(self, code, date, pt_threshold, sl_threshold):
        """检查分钟K线是否触发止盈止损

        【修正12】与 compute_labels 完全一致的遍历顺序：先检查止盈(PT)，后检查止损(SL)
        这样当日内同时触及PT和SL时，以先检查的为准（PT优先）。

        M6 修改: 返回值从 (type, price) 扩展为 (type, price, time)

        Args:
            code: 股票代码
            date: 当前交易日
            pt_threshold: 止盈阈值
            sl_threshold: 止损阈值

        Returns:
            Tuple[str, float, str]:
                (触发类型, 触发价格, 触发时间 'YYYY-MM-DD HH:MM:SS')
                或 (None, None, None)
        """
        start_time = datetime.combine(date, datetime.min.time())
        end_time = datetime.combine(date, datetime.max.time())
        # 【问题3/4/5修复】使用 date 作为 reference_date（交易时的"当前时点"）
        minute_df = self.data_provider.get_minute_data(
            code, start_time, end_time,
            freq=self.meta_labeler.minute_freq if self.meta_labeler else '5',
            reference_date=self.reference_date
        )
        if minute_df.empty:
            # 【BUG-C修复】分钟数据不可用时，回退到日线检查high/low
            try:
                # 【问题3/4/5修复】使用 date 作为 reference_date
                daily_df = self.data_provider.get_daily_data(
                    code, date, date, reference_date=self.reference_date
                )
                if not daily_df.empty:
                    daily_high = daily_df['high'].iloc[0]
                    daily_low = daily_df['low'].iloc[0]
                    open_time = f"{date.strftime('%Y-%m-%d')} 09:30:00"
                    if daily_high >= pt_threshold:
                        return 'profit_take', pt_threshold, open_time
                    if daily_low <= sl_threshold:
                        return 'stop_loss', sl_threshold, open_time
            except Exception as e:
                logger.debug(f'[{date.date()}] {code} 日线回退也失败: {e}')
            return None, None, None

        # M6: 防御性检查 trade_time 列
        if 'trade_time' not in minute_df.columns:
            logger.warning(f'[{date.date()}] {code} 分钟数据无 trade_time 列，fallback 到开盘时间')
            times = None
        else:
            times = minute_df['trade_time'].values

        # M6: 辅助函数 - 获取触发时间
        def _get_trigger_time(idx):
            if times is not None:
                return pd.Timestamp(times[idx]).strftime('%Y-%m-%d %H:%M:%S')
            return f"{date.strftime('%Y-%m-%d')} 09:30:00"

        # 【VECTORISED】numpy 向量化——先看当天最高/最低是否触及，再比先后
        highs = minute_df['high'].values
        lows = minute_df['low'].values
        pt_hit = highs >= pt_threshold
        sl_hit = lows <= sl_threshold
        pt_any = pt_hit.any()
        sl_any = sl_hit.any()

        if pt_any and sl_any:
            pt_idx = np.argmax(pt_hit)
            sl_idx = np.argmax(sl_hit)
            if pt_idx <= sl_idx:
                return 'profit_take', pt_threshold, _get_trigger_time(pt_idx)
            else:
                return 'stop_loss', sl_threshold, _get_trigger_time(sl_idx)
        elif pt_any:
            idx = np.argmax(pt_hit)
            return 'profit_take', pt_threshold, _get_trigger_time(idx)
        elif sl_any:
            idx = np.argmax(sl_hit)
            return 'stop_loss', sl_threshold, _get_trigger_time(idx)
        return None, None, None

# [Small] 兼容别名: 原类名指向微盘版 (config 已指向 SmallAlpha191Strategy)
SmallAlpha191Strategy = Alpha191Strategy  # 兼容别名
