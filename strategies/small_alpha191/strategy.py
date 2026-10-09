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
    cum_max_up: float = float('-inf')
    cum_max_dn: float = float('inf')
    # [E5a-2 profit_protect] 早期保本保护: 无PT触发但累计浮盈峰值σ≥be_trigger_sigma
    #   → SL 抬升至 be_sl_sigma (默认0σ=保本)。默认 inf=禁用 (与原行为一致)。
    be_trigger_sigma: float = float('inf')
    be_sl_sigma: float = 0.0
    # [2026-09-09 参数化] 利润锁定 SL 上移阈值 (PT触发后), 由 config.staged_exit.profit_lock
    #   注入; 默认值与历史硬编码逐位一致 (0.0/1.0/2.0), 保证行为不变。
    pl_sl_s1: float = 0.0
    pl_sl_s2: float = 1.0
    pl_sl_s3: float = 2.0
    # [2026-09-09 参数化] SL触发后 PT 下移阈值, 由 config.staged_exit.pt_after_sl 注入
    #   (默认 1.5/2.0, 与历史硬编码一致)。
    pt_after_sl_s2: float = 1.5
    pt_after_sl_s3: float = 2.0

    def profit_lock_sl(self) -> float:
        """利润锁定 SL：PT 触发后 SL 上移（业界标准 breakeven/profit-locking）
        S3 PT → +2σ, S2 PT → +1σ, S1 PT → 0σ(breakeven), 无 PT → -inf(不覆盖)

        [2026-09-02 方案A] 阈值/比例统一读 self.STAGE_CONFIG (Strategy 层),
        StageTracker 不再持有 stage_cfg/_cfg (死代码已清理)。
        [2026-09-09] 上移阈值改为读 config 注入的 pl_sl_s1/s2/s3 (默认 0/1/2 不变)。
        [E5a-2 profit_protect] 无 PT 触发时, 若累计浮盈峰值 σ≥be_trigger_sigma
        (cum_max_up 为截至昨日的峰值 → 当日生效, 与标签端"日开盘刷新次日生效"
        对齐), 则返回 be_sl_sigma (保本/锁定线), 防止浮盈回吐为亏损。
        """
        if self.s3_pt_hit: return self.pl_sl_s3
        if self.s2_pt_hit: return self.pl_sl_s2
        if self.s1_pt_hit: return self.pl_sl_s1
        # [E5a-2] 早期保本保护 (无 PT 触发)
        if self.daily_vol > 0 and self.cum_max_up / self.daily_vol >= self.be_trigger_sigma:
            return self.be_sl_sigma
        return float('-inf')


class StageManager:
    def __init__(self, stage_cfg=None, hard_sl=-6.0,
                 be_trigger_sigma=float('inf'), be_sl_sigma=0.0,
                 pl_sl_s1=0.0, pl_sl_s2=1.0, pl_sl_s3=2.0,
                 pt_after_sl_s2=1.5, pt_after_sl_s3=2.0):
        self.trackers: Dict[str, StageTracker] = {}
        self.freed_cash_pool: float = 0.0
        # [2026-09-02 方案A] stage_cfg 参数保留仅为接口兼容 (外部 StageManager(stage_cfg=...) 调用);
        #   实际阈值/比例由 Strategy 层 self.STAGE_CONFIG 直接读取, tracker 不再持有。
        self.stage_cfg = stage_cfg
        self.hard_sl = hard_sl
        # [E5a-2 profit_protect] 早期保本保护参数 (Strategy.initialize 注入, 与标签端同源)
        self.be_trigger_sigma = be_trigger_sigma
        self.be_sl_sigma = be_sl_sigma
        # [2026-09-09 参数化] 利润锁定 SL 上移 / SL后PT下移 (config 注入, 默认=历史硬编码)
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
            be_trigger_sigma=self.be_trigger_sigma,
            be_sl_sigma=self.be_sl_sigma,
            pl_sl_s1=self.pl_sl_s1, pl_sl_s2=self.pl_sl_s2,
            pl_sl_s3=self.pl_sl_s3,
            pt_after_sl_s2=self.pt_after_sl_s2,
            pt_after_sl_s3=self.pt_after_sl_s3)

    def reset(self):
        self.trackers.clear()
        self.freed_cash_pool = 0.0

    def drop(self, code: str):
        """[O1-20260904] 定向移除单个持仓的 tracker (错位轮动: 只清当前批次,
        保留其余批次 tracker; 原 reset() 全清会在分批场景误杀其它批次)"""
        if code in self.trackers:
            del self.trackers[code]

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


class SmallAlpha191Strategy(IStrategy):
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
        #   实例级覆盖 provider.snapshot_path (test3 → factor_snapshots_test,
        #   零框架改动、不影响 test2/生产目录的默认 factor_snapshots)
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
        # [B2-20260905] P2-2: 特征选择 IC 口径 ('daily'=逐日截面RankIC均值 | 'pooled'=旧全面板)
        self.IC_MODE = config.get('ic_mode', 'daily')
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
        # [2026-09-06 修复] 标签端尊重 staged_exit.enabled: 分阶段总开关关闭时,
        #   分阶段标签模式降级为 relative_barrier, 与交易端停用分阶段保持一致 (消除割裂)
        if self._label_mode in ('staged_binary', 'staged_rank') \
                and not config.get('staged_exit', {}).get('enabled', True):
            logger.warning(f"[L1/L2] staged_exit.enabled=false → 标签模式 {self._label_mode} "
                           f"降级为 relative_barrier (与交易端一致)")
            self._label_mode = 'relative_barrier'
        self._label_min_abs_ret = float(meta_cfg.get('label_min_abs_ret', 0.0)) if isinstance(meta_cfg, dict) else 0.0
        # [Q5N] 目标截面变换: none(现行) | orth_vol(N1) | fwd_ret_orth(N2)
        #   ★唯一变量 = 目标的截面变换; staged_exit 逐键不变 ⇒ 不违反 D-225
        #   ★非法值直接 raise (禁止静默降级, R-108 同族)
        self._target_transform = str(
            meta_cfg.get('target_transform', 'none')) if isinstance(meta_cfg, dict) else 'none'
        if self._target_transform not in ('none', 'orth_vol', 'fwd_ret_orth'):
            raise ValueError('[Q5N] 非法 meta_labeling.target_transform=%r'
                             % self._target_transform)
        if self._target_transform != 'none':
            logger.info('[Q5N] 目标截面变换已启用: %s (标签端额外产出 vol/fwd_ret_5)',
                        self._target_transform)
        if self.use_meta_labeling:
            # 【问题3/4/5修复】不再传入 reference_date，
            # compute_labels 中使用 event_date 作为 reference_date
            # [L1] staged 模式: 从 config.staged_exit 读取与交易端完全同源的
            # 分阶段参数 (S1-S4 阈值/卖出比例/硬兜底), 传入标签器
            _staged_cfg = None
            _staged_hard_sl = -6.0
            # [2026-09-09 加固] 原实现 _profit_protect 仅在 staged 分支内定义, 非 staged
            #   模式下第447行引用会 NameError (当前配置未触发, 属潜伏缺陷) → 补默认值。
            _profit_protect = None
            # [2026-09-09 参数化] 利润锁定 SL 上移 / SL后PT下移: 标签端与交易端同源
            #   (原硬编码于 labels.py 的 Python/numba 两版, 现由 config 单点定义)
            #   [E9-WIRE-2026-09-10] 修复: test3 此前遗漏下方接线 → 标签端恒用默认值;
            #   补齐后与 17 个家族策略一致 (默认值等价, 行为不变)。
            _profit_lock = None  # 非 staged 模式保持 None (构造器默认等价)
            _pt_after_sl = None  # 非 staged 模式保持 None (构造器默认等价)
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
                # [E5a-2 profit_protect] 与交易端 StageManager 同源 (同 config 段), 标签同步保本保护
                _pp_cfg_lbl = _se.get('profit_protect', {})
                _profit_protect = {
                    'enabled': bool(_pp_cfg_lbl.get('enabled', False)),
                    'trigger_sigma': float(_pp_cfg_lbl.get('trigger_sigma', 1.0)),
                    'sl_sigma': float(_pp_cfg_lbl.get('sl_sigma', 0.0)),
                }
                # [E9-WIRE-2026-09-10] 利润锁定/PT下移: 标签端与交易端同源 (config 单点定义);
                #   test3 此前遗漏此接线 (其余 17 策略均已有) → 补齐。默认值等价, 行为不变;
                #   非默认值场景 (E9 lock/pas 标定臂) 下保证标签-交易语义一致。
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
                profit_protect=_profit_protect,
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

        # ----- [Small] 第一层选股池参数化: V7快照 → pool_mode (smallcap/bigcap/index_component/code_prefix) -----
        # 原 alpha191: 加载 V7 Wyckoff 选股快照 (recomd_*.parquet)
        # pool_mode 分支 (全部截至 D-1, 无未来数据):
        #   smallcap        : DynamicStockPool 候选池 → T-1 市值最低 N 只
        #   bigcap          : DynamicStockPool 候选池 → T-1 市值最高 N 只
        #   index_component : 指数成分股(index_stocks_cache, ≤D最近) ∩ DynamicStockPool
        #   code_prefix     : DynamicStockPool 候选池 ∩ 代码前缀 (如 300/301=创业板, 688=科创板)
        self._v7_recomd_dir = None
        self._v7_snapshot_cache = {}
        self._v7_available_dates = set()
        self._pool_mode = str(config.get('data', {}).get('pool_mode', 'smallcap'))
        self._pool_index = str(config.get('data', {}).get('pool_index', 'hs300'))
        self._pool_prefixes = [str(p) for p in (config.get('data', {}).get('pool_prefixes', []) or [])]
        self._smallcap_pool_size = int(config.get('data', {}).get('smallcap_pool_size', 300))
        if self.stock_pool is None:
            raise RuntimeError(f"[Pool] context.stock_pool 未注入 (DynamicStockPool), 无法构建选股池")
        # [O2-fix-20260910] 原"实例级遮蔽"补丁已移除 —— 框架层 DynamicStockPool.VALID_PREFIXES
        #   已补全 sh.689/sz.003/sz.302 (gpsef/pool/dynamic_pool.py = 唯一真值源, 与下载器/
        #   data_utils 一致, 且对家族全部策略生效)。此处仅保留一致性守卫: 若未来框架表再漂移
        #   → 立即告警, 不得再以实例遮蔽临时绕过 (R6-E2 纪律)。
        _dp_prefixes = tuple(getattr(self.stock_pool, 'VALID_PREFIXES', ()) or ())
        _expected_prefixes = ('sh.600', 'sh.601', 'sh.603', 'sh.605', 'sh.688', 'sh.689',
                              'sz.000', 'sz.001', 'sz.002', 'sz.003', 'sz.300', 'sz.301', 'sz.302')
        if _dp_prefixes and set(_dp_prefixes) != set(_expected_prefixes):
            logger.warning('[Pool][O2-fix] ⚠️ 框架 VALID_PREFIXES 与预期不一致: 缺失=%s '
                           '多出=%s → 请检查 gpsef/pool/dynamic_pool.py (不得再以实例遮蔽临时绕过)',
                           sorted(set(_expected_prefixes) - set(_dp_prefixes)),
                           sorted(set(_dp_prefixes) - set(_expected_prefixes)))
        # 市值查找加速结构: 按 trade_date 排序 + 每日起止索引 (O(log N + k) 查询)
        try:
            self._ensure_total_mv_cache()   # 幂等: 缓存文件不存在时先构建
            _mv_df = self._load_total_mv_cache()
            _mv_df = _mv_df.sort_values('trade_date').reset_index(drop=True)
            _mv_df['_td'] = pd.to_datetime(_mv_df['trade_date']).values
            _td_arr = _mv_df['_td'].values
            self._mv_uniq_dates, self._mv_starts = np.unique(_td_arr, return_index=True)
            self._mv_ends = np.r_[self._mv_starts[1:], len(_td_arr)]
            self._mv_sorted = _mv_df[['ts_code', 'total_mv', '_td']]
            logger.info(f"[Pool] 市值查找索引构建完成: {len(self._mv_uniq_dates)} 个交易日")
        except Exception as e:
            self._mv_sorted = None
            logger.warning(f"[Pool] 市值索引构建失败 ({e}), 市值类池将不可用")
        # 指数成分股缓存 (index_component 模式): data/index_stocks_cache.parquet
        self._index_components = {}   # {index: {date_str: [codes]}}
        self._index_dates = {}        # {index: sorted(date_str)}
        if self._pool_mode == 'index_component':
            try:
                _ic_path = os.path.join(self.data_provider.base_path, 'index_stocks_cache.parquet')
                _ic_df = pd.read_parquet(_ic_path)
                for _k, _lst in zip(_ic_df['cache_key'], _ic_df['stock_list']):
                    _idx, _d = _k.split('_')[0], _k.split('_')[1]
                    self._index_components.setdefault(_idx, {})[_d] = list(_lst)
                for _idx in self._index_components:
                    self._index_dates[_idx] = sorted(self._index_components[_idx].keys())
                _n_dates = len(self._index_dates.get(self._pool_index, []))
                logger.info(f"[Pool] 指数成分股缓存加载: {self._pool_index} {_n_dates} 个日期")
                if _n_dates == 0:
                    logger.warning(f"[Pool] 指数 {self._pool_index} 无成分股数据")
            except Exception as e:
                logger.warning(f"[Pool] 指数成分股缓存加载失败 ({e})")
        logger.info(f"[Pool] 第一层选股池 mode={self._pool_mode} "
                    f"(index={self._pool_index}, prefixes={self._pool_prefixes}, "
                    f"size={self._smallcap_pool_size})")

        # ----- [Small] 选股池快照落盘缓存 (按池隔离) -----
        # 语义: 缓存 {date_str: [codes]}, 跨回测复用, 优先读本地,
        #       无缓存/指纹失效(池配置或市值口径变化)才重算并落盘。
        if self._pool_mode == 'smallcap':
            # [O2-20260904] pool_cache_tag 隔离: 前缀名单改动后旧缓存(旧前缀)必须失效,
            #   tag 非空时写独立文件, 避免跨目录共享 smallcap_pool_cache.parquet 命中旧名单。
            _pool_cache_tag = str(config.get('data', {}).get('pool_cache_tag', '') or '').strip()
            if _pool_cache_tag:
                self._smallcap_cache_path = os.path.join(
                    os.path.abspath(self.local_data_path),
                    f'smallcap_pool_cache_{_pool_cache_tag}.parquet')
            else:
                self._smallcap_cache_path = os.path.join(
                    os.path.abspath(self.local_data_path), 'smallcap_pool_cache.parquet')
        else:
            _tag = self._pool_mode + ('_' + self._pool_index if self._pool_index
                                      else '_' + '_'.join(self._pool_prefixes))
            self._smallcap_cache_path = os.path.join(
                os.path.abspath(self.local_data_path), f'pool_{_tag}.parquet')
        self._smallcap_meta_path = self._smallcap_cache_path.replace('.parquet', '_meta.json')
        self._smallcap_cache = {}       # {date_str: [codes]} 已加载或新计算
        self._smallcap_new_dates = {}   # {date_str: [(code, mv)]} 待落盘
        self._smallcap_dirty = 0        # 未落盘新日期计数
        # [FIX-DET-2026-09-10] 池缓存状态: loaded | recomputed | missing
        #   用于溯源落盘 (provenance.json) 与"结果是否与既有实验可比"的判定。
        self._pool_cache_status = 'unknown'
        self._pool_cache_loaded_dates = 0
        self._pool_cache_recomputed = False
        # [FIX-DET2-2026-09-10] 已冻结历史日期中, 重算结果与落盘内容不一致的日期数
        self._pool_cache_conflicts = 0
        # [FIX-DET9][R7] 逐日期输入指纹映射 与 因输入变化被判"陈旧"的日期数
        self._pool_cache_per_date = {}
        self._pool_cache_stale_dates = 0
        # [FIX-DET10][R4] BARRA 快照溯源: 使用过的快照日期集合 / 是否发生运行期重建
        self._barra_snap_used = set()
        self._barra_snap_files = None
        self._barra_rebuilt = False
        self._load_smallcap_cache()

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
            ),
            2: dict(
                pt=float(se_cfg.get('stage2', {}).get('pt_sigma', 3.0)),
                sl=float(se_cfg.get('stage2', {}).get('sl_sigma', -3.0)),
                pct=float(se_cfg.get('stage2', {}).get('sell_pct', 0.35)),
            ),
            3: dict(
                pt=float(se_cfg.get('stage3', {}).get('pt_sigma', 4.0)),
                sl=float(se_cfg.get('stage3', {}).get('sl_sigma', -4.0)),
                pct=float(se_cfg.get('stage3', {}).get('sell_pct', 0.20)),
            ),
            4: dict(
                pt=float(se_cfg.get('stage4', {}).get('pt_sigma', 9.0)),
                sl=float(se_cfg.get('stage4', {}).get('sl_sigma', -4.0)),
                pct=float(se_cfg.get('stage4', {}).get('sell_pct', 0.20)),
            ),
        }
        self.hard_sl_sigma = float(se_cfg.get('hard_sl_sigma', -6.0))
        self.reinvest_enabled = se_cfg.get('reinvest_enabled', True)
        self.reinvest_min_amount = float(se_cfg.get('reinvest_min_amount', 5000))
        # [E5a-2 profit_protect] 早期保本保护: 无PT触发但累计浮盈峰值σ≥trigger_sigma
        #   → 每日开盘将 SL 抬至 sl_sigma (默认0σ=保本)。与标签端 TripleBarrierLabeler
        #   同源读取同一 config 段 (交易-标签对齐)。enabled=false → inf=禁用 (原行为不变)。
        _pp_cfg = se_cfg.get('profit_protect', {})
        self._pp_enabled = bool(_pp_cfg.get('enabled', False))
        self._pp_trigger_sigma = (float(_pp_cfg.get('trigger_sigma', 1.0))
                                  if self._pp_enabled else float('inf'))
        self._pp_sl_sigma = float(_pp_cfg.get('sl_sigma', 0.0))
        # [2026-09-09 参数化] 利润锁定 SL 上移 (PT触发后): S1→0σ / S2→+1σ / S3→+2σ
        #   原硬编码在 profit_lock_sl() 内且 labels.py 另存一份 (有标签-交易不同步风险),
        #   现由 config.staged_exit.profit_lock 单点定义, 与标签端同源读取。
        _plk_cfg = se_cfg.get('profit_lock', {})
        self._pl_sl_s1 = float(_plk_cfg.get('s1_sl', 0.0))
        self._pl_sl_s2 = float(_plk_cfg.get('s2_sl', 1.0))
        self._pl_sl_s3 = float(_plk_cfg.get('s3_sl', 2.0))
        # [2026-09-09 参数化] SL触发后 PT 下移 (快速退出): 默认 S2→1.5σ / S3→2.0σ
        _pasl_cfg = se_cfg.get('pt_after_sl', {})
        self._pt_after_sl_s2 = float(_pasl_cfg.get('s2_pt', 1.5))
        self._pt_after_sl_s3 = float(_pasl_cfg.get('s3_pt', 2.0))
        # 检查层统一读取 (须在上述两行之后定义)
        self.PT_AFTER_SL = {2: self._pt_after_sl_s2, 3: self._pt_after_sl_s3}
        if self._pp_enabled:
            logger.info(f"[E5a-2 profit_protect] 早期保本保护启用: "
                        f"浮盈峰值>={self._pp_trigger_sigma}σ → SL 抬至 {self._pp_sl_sigma}σ")
        logger.info("[v3.6] 分阶段退出: enabled=%s, S1=%.1fσ/%.0f%%, S2=%.1fσ/%.0f%%, S3=%.1fσ/%.0f%%, hard=%.1fσ, reinvest=%s",
                    self.staged_exit_enabled,
                    self.STAGE_CONFIG[1]['pt'], self.STAGE_CONFIG[1]['pct']*100,
                    self.STAGE_CONFIG[2]['pt'], self.STAGE_CONFIG[2]['pct']*100,
                    self.STAGE_CONFIG[3]['pt'], self.STAGE_CONFIG[3]['pct']*100,
                    self.hard_sl_sigma, self.reinvest_enabled)
        self.stage_mgr = StageManager(stage_cfg=self.STAGE_CONFIG,
                                       hard_sl=self.hard_sl_sigma,
                                       be_trigger_sigma=self._pp_trigger_sigma,
                                       be_sl_sigma=self._pp_sl_sigma,
                                       pl_sl_s1=self._pl_sl_s1,
                                       pl_sl_s2=self._pl_sl_s2,
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
        #   纳入回撤计算, 熔断门槛持续有效。 (与 small_alpha191_test 同步修复)
        self._dd_flat_nav = None
        self._dd_index_close = self._load_breaker_index_close(self._dd_cfg['index_code'])

        # [E3 2026-09-09] 目标敞口闭环状态变量
        self._daily_vol_scale = 1.0     # 日频 vol_scale (池指数 EWMA + 不对称恢复)
        self._sl_locked = set()         # 止损锁定名单 (本周期内禁再部署回补)
        self._redeploy_list = []        # 欠配再部署预标记 [(code, amt)]
        self._derisk_list = []          # 超配减仓预标记 [(code, amt)]

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
        # [O1-20260904] 错位分批调仓: rebalance_batches 个批次, 每 rebalance_interval
        #   交易日轮动 1 批 (全周期 = interval×batches 交易日 ≈ 原 20 日周期)。
        #   batches<=1 → 旧行为 (全清重建)。
        self.rebalance_batches = max(1, int(config.get('rebalance_batches', 1)))
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
        self._garch_cfg = gc_cfg     # [E2] 保存配置引用 (use_pool_index 分支读取)
        self._garch_enabled = gc_cfg.get('enabled', True)
        if self._garch_enabled:
            # [AUDIT-FIX 2026-09-14 D-176 P0-1b] GARCH 配置互斥 fail-fast 校验
            #   背景: use_daily_scale=true 时策略走日频分支并令 ret_matrix=None, 而
            #   GARCHPositionController.compute() 受 `ret_matrix is not None` 守卫 ⇒
            #   四级回撤阶梯 / 恢复缓冲 / kill-switch (三者唯一执行入口 = compute) 全部失效。
            #   ⇒ 同配置共启 = 纯无用功(约 12h/臂), 故硬失败, 禁止静默无效。
            _mutex_keys = [k for k in ('use_drawdown_control', 'use_recovery_buffer',
                                       'use_kill_switch') if gc_cfg.get(k, False)]
            if gc_cfg.get('use_daily_scale', False) and _mutex_keys:
                raise ValueError(
                    "[GARCH 配置互斥] use_daily_scale=true 会使 "
                    "GARCHPositionController.compute() 不被调用, 届时 %s 全部失效。"
                    "如需启用回撤机制, 请设 use_daily_scale=false。" % _mutex_keys)
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
                # [2026-09-08] 基线指数可切换 (csi800 默认, 可切 hs300/zz500/sh000001/sh932000)
                self._barra_baseline = barra_cfg.get('baseline_index_code', 'csi800')
                logger.info(f"[v3.0] BARRA 约束优化: 已启用 "
                            f"(max_weight={barra_cfg.get('max_weight', 0.10)}, "
                            f"risk_aversion={barra_cfg.get('risk_aversion', 1.0)}, "
                            f"baseline={self._barra_baseline})")
            except ImportError as e:
                self._barra_cfg = None
                self._barra_enabled = False
                logger.warning(f"[v3.0] BARRA 约束优化: CVXOPT/V10 模块不可用 ({e})，已禁用")
        else:
            self._barra_cfg = None
            logger.info("[v3.0] BARRA 约束优化: 已禁用")

        # ---- [Q5BAR 2026-10-08] BARRA **风险预算式降权/剔除**（与上面"求解器配权"机制独立、互斥） ----
        #   背景：D-243 结案的是「求解器配权」路径（改权重 ⇒ 天然倾向集中，HHI 与等权不可区分）；
        #        其 §3.3 明确点名「用暴露做降权/剔除（风险预算式）」**从未试过** ⇒ 本波检验该机制。
        #   语义：候选集内对指定暴露做稳健 z（中位/MAD）⇒ R_i = Σ_f z_{i,f}
        #        ⇒ score'_i = score_i × exp(−λ·max(0,R_i)) ⇒ 照常排序取 Top-N。
        #   ★只改"选哪 N 只"，**完全不动权重层** ⇒ 从设计上排除"求解器把权重集中"的通道。
        #   ★默认 mode='off' ⇒ 缺键时零影响闭合（与采纳配置逐位一致）。
        _rb_cfg = {}
        try:
            _rb_cfg = (barra_cfg or {}).get('risk_budget') or {}
            if not isinstance(_rb_cfg, dict):
                _rb_cfg = {}
        except Exception:
            _rb_cfg = {}
        self._barra_rb_cfg = _rb_cfg
        self._barra_rb_mode = str(_rb_cfg.get('mode', 'off') or 'off').strip().lower()
        self._barra_rb_expo_cache = {}          # {YYYYMMDD(str): DataFrame|None} 只读缓存
        self._barra_rb_used = set()             # 本次运行实际使用过的快照日期（进 provenance）
        if self._barra_rb_mode == 'risk_budget':
            logger.info(
                "[Q5BAR] BARRA 风险预算式降权: 已启用 mode=%s factors=%s strength=%s "
                "exclude_quantile=%s min_coverage=%s shuffle_seed=%s",
                self._barra_rb_mode, _rb_cfg.get('factors'),
                _rb_cfg.get('strength'), _rb_cfg.get('exclude_quantile'),
                _rb_cfg.get('min_coverage', 0.8), _rb_cfg.get('shuffle_seed'))
        else:
            logger.info("[Q5BAR] BARRA 风险预算式降权: 已禁用（mode=%s ⇒ 零影响闭合）",
                        self._barra_rb_mode)

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
        cfg.IC_MODE = self.IC_MODE   # [B2-20260905] 逐日截面RankIC开关
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
        from .determinism_provenance import files_fingerprint

        logger = logging.getLogger('strategies.small_alpha191_test3.strategy')

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
            logger = logging.getLogger('strategies.small_alpha191_test3.strategy')
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
        logger = logging.getLogger('strategies.hs300_alpha191.strategy')
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

        logger = logging.getLogger('strategies.small_alpha191_test3.strategy')
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

        # [FIX-P-20260915] 缺失市值先按【当日截面】中位数填充（D-217）。
        #   原实现用面板全局中位数 ⇒ 与生产端单日截面口径不一致，属 D-216 同类
        #   （训练/生产口径错位）。全局中位数保留为第二级兜底。
        _dc = next((c for c in ('trade_date', 'trade_date_x', 'date')
                    if c in merged.columns), None)
        if _dc is not None:
            try:
                _per_med = merged.groupby(
                    pd.to_datetime(merged[_dc]))['total_mv'].transform('median')
                merged['total_mv'] = merged['total_mv'].fillna(_per_med)
            except Exception:
                pass
        # 缺失值用全局中位数兜底填充
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
        logger = logging.getLogger('strategies.small_alpha191_test3.strategy')
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

        # [FIX-P-20260915] 同 FIX-P1：先按当日截面中位数，再全局兜底（D-217）
        if 'trade_date' in df.columns:
            try:
                _per_med = df.groupby(
                    pd.to_datetime(df['trade_date']))['total_mv'].transform('median')
                df['total_mv'] = df['total_mv'].fillna(_per_med)
            except Exception:
                pass
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
            # [B1-20260905] P2-1: fit 面板剔除生产 val 段 (与 ensemble prod_val 切割同规则:
            #   val_size = min(max_hold_days, max(n_dates//5, 1)))，消除 O7 排除的不对称残留。
            #   beta 仅由非 val 段估计, 再 apply 到全面板 (prod_val 的 OOS AUC 不再见过该段因子)。
            fit_df, X_fit = df, X_neutral
            if (cfg.get('exclude_prod_val_in_fit', True)
                    and 'trade_date' in df.columns):
                uniq_dates = sorted(df['trade_date'].unique())
                max_hold = int(self.config.get('meta_labeling', {}).get('max_hold_days', 20))
                val_size = min(max_hold, max(len(uniq_dates) // 5, 1))
                if 0 < val_size < len(uniq_dates):
                    cutoff = uniq_dates[-val_size]
                    fit_mask = df['trade_date'] < cutoff
                    fit_df = df.loc[fit_mask]
                    X_fit = X_neutral.loc[fit_mask]
                    logger.info(
                        f"[B1] 中性化 fit 剔除生产 val 段: 全 {len(uniq_dates)} 天 → "
                        f"fit 用 {fit_df['trade_date'].nunique()} 天 (cutoff={pd.Timestamp(cutoff).date()}, "
                        f"剔除 {len(uniq_dates) - fit_df['trade_date'].nunique()} 天)")
            _, betas_info = framework.neutralize_fit(fit_df, factor_cols, X_fit)
            df = framework.neutralize_apply(df, factor_cols, X_neutral, betas_info)
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
        """第一层选股池 (pool_mode 参数化, 全部截至 D-1, 无未来数据)。

        返回 (codes, pred_dict) 与 alpha191 同构:
          - codes:     池内股票 (ts_code 格式 'sh.600000')
          - pred_dict: 空 dict (无 V10 pred, 下游 v10_pred_raw 特征中性 0.5)
        池模式:
          smallcap        : DynamicStockPool 候选池 → T-1 市值最低 N 只
          bigcap          : DynamicStockPool 候选池 → T-1 市值最高 N 只
          index_component : 指数成分股(index_stocks_cache ≤D 最近) ∩ DynamicStockPool
          code_prefix     : DynamicStockPool 候选池 ∩ 代码前缀
        日期边界: dynamic_pool 内部仅用 T-1/T-2; 成分股取 cache_key ≤ D 最近; 市值严格 < D。
        缓存: 进程内 _v7_snapshot_cache + 落盘 pool cache (按池隔离), 优先读, 无才计算。
        """
        if date_str in self._v7_snapshot_cache:
            return self._v7_snapshot_cache[date_str]
        # 落盘缓存命中 (跨回测复用)
        if date_str in self._smallcap_cache:
            codes = self._smallcap_cache[date_str][:self._smallcap_pool_size]
            self._v7_snapshot_cache[date_str] = (codes, {})
            return self._v7_snapshot_cache[date_str]
        if self.stock_pool is None:
            return None
        date = pd.Timestamp(date_str)

        # ===== [Pool] index_component: 指数成分股 ∩ DynamicStockPool (T-1 可交易过滤) =====
        if self._pool_mode == 'index_component':
            codes = self._get_index_components(date)
            if codes is None:
                logger.warning(f"[Pool] {date_str} 指数 {self._pool_index} 无成分股数据")
                return None
            try:
                cand_set = set(self.stock_pool.get_candidates(date, None) or [])
                if cand_set:
                    codes = [c for c in codes if c in cand_set]
            except Exception as e:
                logger.warning(f"[Pool] dynamic_pool 异常 {date_str}: {e}")
            if not codes:
                logger.warning(f"[Pool] {date_str} 成分股过滤后为空")
                return None
            self._cache_smallcap_result(date_str, codes)
            logger.debug(f'[Pool] {date_str} 成分股={len(codes)}只 (index={self._pool_index})')
            return self._v7_snapshot_cache[date_str]

        # ===== [Pool] code_prefix: DynamicStockPool 候选池 ∩ 代码前缀 =====
        if self._pool_mode == 'code_prefix':
            try:
                candidates = self.stock_pool.get_candidates(date, None) or []
            except Exception as e:
                logger.warning(f"[Pool] dynamic_pool 异常 {date_str}: {e}")
                return None
            codes = [c for c in candidates
                     if any(c.split('.')[1].startswith(p) for p in self._pool_prefixes)]
            if not codes:
                logger.warning(f"[Pool] {date_str} 前缀过滤后为空 (prefixes={self._pool_prefixes})")
                return None
            self._cache_smallcap_result(date_str, codes)
            logger.debug(f'[Pool] {date_str} 前缀池={len(codes)}只 (prefixes={self._pool_prefixes})')
            return self._v7_snapshot_cache[date_str]

        # ===== [Pool] smallcap/bigcap: DynamicStockPool 候选池 → T-1 市值排序取 N =====
        if getattr(self, '_mv_sorted', None) is None:
            return None
        try:
            candidates = self.stock_pool.get_candidates(date, None)
        except Exception as e:
            logger.warning(f"[Small] dynamic_pool 异常 {date_str}: {e}")
            return None
        if not candidates:
            logger.warning(f"[Small] 候选池为空: {date_str}")
            return None
        # T-1 市值: 定位 trade_date < date 的最近交易日 (D日盘前 D日市值未生成, 严格取 T-1)
        # 与 dynamic_pool._get_prev_trade_dates 的 side='left' 边界一致 (均不含 date 当日)
        pos = int(np.searchsorted(self._mv_uniq_dates,
                                  np.datetime64(date), side='left')) - 1
        if pos < 0:
            return None
        s, e = int(self._mv_starts[pos]), int(self._mv_ends[pos])
        sec = self._mv_sorted.iloc[s:e]
        mv_map = dict(zip(sec['ts_code'], sec['total_mv']))
        pooled = [c for c in candidates if c in mv_map]
        if not pooled:
            logger.warning(f"[Small] 候选池无市值数据: {date_str}")
            return None
        # [FIX-DET3-2026-09-10] 确定性排序: 原实现 `sort(key=mv_map[c], reverse=...)` 是稳定排序,
        #   市值**完全并列**时名次由 `candidates` 的顺序决定 → 若 candidates 顺序不确定
        #   (或候选集组成变化), 边界成员归属就会抖动。
        #   实测: 池中确实存在并列对 (sz.300114 / sz.302132 同一公司双代码, 窗口内每日一组且全程重复)。
        #   修复: 显式二级键 ts_code (升序), 使结果与输入顺序彻底解耦。逻辑见 rank_pool_codes。
        from .determinism_provenance import rank_pool_codes
        codes, _bnd_tie = rank_pool_codes(
            candidates, mv_map, self._smallcap_pool_size, self._pool_mode)
        if not codes:
            logger.warning(f"[Small] 候选池无市值数据: {date_str}")
            return None
        if _bnd_tie is not None:
            logger.warning(
                '[Pool][FIX-DET3] ⚠️ 池边界存在市值并列 (%s): 第%d名 %s 与 第%d名 %s 市值均为 %.6g '
                '→ 成员归属由 ts_code 二级键决定 (确定性, 与输入顺序无关)',
                date_str, int(self._smallcap_pool_size), _bnd_tie[0],
                int(self._smallcap_pool_size) + 1, _bnd_tie[1], _bnd_tie[2])
        self._cache_smallcap_result(date_str, codes, mv_map)
        logger.debug(f'[Small] {date_str} 候选池={len(candidates)} → 市值最低{len(codes)}只')
        return self._v7_snapshot_cache[date_str]

    def _get_index_components(self, date) -> list:
        """[Pool] 指数成分股: index_stocks_cache 中 cache_key ≤ date 的最近一条 (严格无未来)。

        成分股名单每周更新(缓存), D 日盘前用最近一次已公布名单 —— 与真实指数调仓时点一致。
        """
        dates = self._index_dates.get(self._pool_index)
        if not dates:
            return None
        dstr = pd.Timestamp(date).strftime('%Y%m%d')
        pos = int(np.searchsorted(np.array(dates), dstr, side='right')) - 1
        if pos < 0:
            return None
        return list(self._index_components[self._pool_index][dates[pos]])

    def _cache_smallcap_result(self, date_str, codes, mv_map=None):
        """统一缓存/落盘: 写入进程内缓存 + 待落盘 (每 50 个新日期自动 flush)。"""
        self._v7_snapshot_cache[date_str] = (codes, {})
        self._smallcap_cache[date_str] = codes
        self._smallcap_new_dates[date_str] = [(c, mv_map[c] if mv_map is not None else 0.0)
                                              for c in codes]
        self._smallcap_dirty += 1
        if self._smallcap_dirty >= 50:
            self.flush_smallcap_cache()

    def _smallcap_cache_fingerprint(self) -> str:
        """缓存指纹: 池过滤配置 + smallcap_pool_size + 市值缓存【内容哈希】。

        [FIX-DET-2026-09-10] 原实现取 total_mv_cache.parquet 的 **文件 mtime** 作为
        指纹成分, 导致一个可复现性缺陷:
          只要该文件被重写 (哪怕内容一字未变, 例如增量补日/重跑 _ensure_total_mv_cache),
          mtime 变化 → 指纹变化 → 选股池缓存被判失效 → 回测内**全量重算选股池** →
          结果取决于"文件何时被写过"而非"数据内容是什么"。
        实证: 2026-09-09 E0-full 两次运行的差异 (+6.90% vs +6.27%) 即由此类
        共享输入改写造成 (池缓存于 08:53:52 被重写, 两次运行分别读到了改写前/后的名单)。

        修复: 改为对文件**内容**做 md5。内容相同 → 指纹相同 → 缓存复用 → 结果可复现;
              内容不同 → 指纹变化 → 仍会正确失效重算 (原防护语义不变)。
        另引入 fp_version: 指纹算法变更时强制一次性失效, 避免新旧算法哈希被误判为"相同"。
        """
        from .determinism_provenance import md5_file, pool_cache_fingerprint
        filters = list(getattr(self.stock_pool, 'filters', []) or [])
        mv_path = os.path.join(self.data_provider.base_path, 'total_mv_cache.parquet')
        # 内容哈希 (进程内记忆化, 只读一次); 文件缺失时退化为 None
        try:
            mv_digest = md5_file(mv_path) if os.path.exists(mv_path) else None
        except Exception:
            mv_digest = None
        # [R6-E2-2026-09-10] 纳入**框架级生效前缀表**: 前缀表（哪些代码段进入股票宇宙）
        #   是决定池成员的关键规则之一, 必须先于候选集进入指纹 —— 否则框架前缀表修复后
        #   旧缓存（缺新段股票）仍会被复用。与家族 13 个池策略统一为同一算法 (fp_version=3)。
        eff_prefixes = sorted(tuple(getattr(self.stock_pool, 'VALID_PREFIXES', ()) or ()))
        return pool_cache_fingerprint(
            filters=filters,
            pool_size=self._smallcap_pool_size,
            mv_digest=mv_digest,
            pool_mode=self._pool_mode,
            pool_index=self._pool_index,
            pool_prefixes=self._pool_prefixes,
            eff_prefixes=eff_prefixes,
        )

    def _load_smallcap_cache(self):
        """加载本地选股池快照缓存 (指纹校验); 无缓存/失效时静默置空待计算。

        [FIX-DET-2026-09-10] 增加状态标记与显式告警:
        原先"失效即静默重算", 使**结果口径在无人察觉的情况下改变**。
        现在凡发生重算, 都记录 _pool_cache_status='recomputed' 并 WARNING 提示
        "本次结果与既有实验不可直接对比", 同时由 provenance.json 固化留档。
        """
        try:
            if not (os.path.exists(self._smallcap_cache_path)
                    and os.path.exists(self._smallcap_meta_path)):
                self._pool_cache_status = 'missing'
                logger.warning('[Pool][FIX-DET] 选股池快照缓存不存在, 本次回测将计算并落盘; '
                               '若既有实验已使用该池, 则本次结果与其**不可直接对比**')
                return
            meta = json.load(open(self._smallcap_meta_path, encoding='utf-8'))
            cur_fp = self._smallcap_cache_fingerprint()
            per_date = meta.get('per_date') or {}
            if meta.get('fingerprint') != cur_fp and not per_date:
                # 旧格式元数据（无逐日期指纹）→ 保守处理: 整表视为失效
                self._pool_cache_status = 'recomputed'
                self._pool_cache_recomputed = True
                logger.warning('[Pool][FIX-DET] 选股池缓存指纹不匹配且无逐日期指纹 → 本次重新计算。'
                               '(meta指纹=%s, 当前指纹=%s) ⚠ 本次结果与既有实验不可直接对比, '
                               '已记入 provenance.json',
                               str(meta.get('fingerprint'))[:8], cur_fp[:8])
                return
            df = pd.read_parquet(self._smallcap_cache_path)
            if df.empty:
                self._pool_cache_status = 'missing'
                return
            # [FIX-DET9-2026-09-10][R7] **逐日期有效性**:
            #   每个日期记录其写入时的输入指纹; 与当前指纹不同的日期视为"缺失",
            #   将在本次回测中按需重算 → 从机制上消除"文件里部分新/部分旧"的一致性窗口。
            _cur_short = cur_fp[:16]
            df = df.sort_values(['trade_date', 'mv_rank'])
            n_stale = 0
            self._pool_cache_per_date = {}
            for d, g in df.groupby('trade_date'):
                ds = str(d)
                fp_d = per_date.get(ds)
                if fp_d is not None and fp_d != _cur_short:
                    n_stale += 1
                    continue          # 旧输入下的行 → 按需重算
                self._smallcap_cache[ds] = g['ts_code'].tolist()
                if fp_d:
                    self._pool_cache_per_date[ds] = fp_d
            self._pool_cache_stale_dates = n_stale
            self._pool_cache_loaded_dates = len(self._smallcap_cache)
            if n_stale:
                self._pool_cache_status = 'partial'
                logger.warning('[Pool][R7] 逐日期指纹: %d/%d 个日期的输入已变化 → 这些日期将重算 '
                               '(其余 %d 日直接复用)', n_stale, n_stale + len(self._smallcap_cache),
                               len(self._smallcap_cache))
            else:
                self._pool_cache_status = 'loaded'
                logger.info(f'[Pool] 选股池快照缓存加载: {len(self._smallcap_cache)} 个交易日 '
                            f'(逐日期指纹全部有效) -> {self._smallcap_cache_path}')
        except Exception as e:
            logger.warning(f'[Pool] 选股池快照缓存加载失败 ({e}), 本次重新计算')
            self._pool_cache_status = 'recomputed'
            self._pool_cache_recomputed = True
            self._smallcap_cache = {}

    def flush_smallcap_cache(self):
        """把新计算的日期合并写入本地缓存 (回测结束由 run_backtest 调用; 累计50日自动触发)。

        [FIX-DET2-2026-09-10] 语义由"**覆盖**同日期旧行"改为"**只追加, 历史日期冻结**"。

        原缺陷（2026-09-09 事故根因）::
            写回时会把【本次重算到的日期】的既有行先剔除, 再以新行替换
            (即"覆盖同日期旧行"语义)
            → 一次口径变更 / 长跑实验的中间写入, 就会让**同一份配置**的两次运行
              读到不同的历史名单。后果链:
              09-08 22:28 total_mv_cache 被重建（[Fix-20260908] sz_000 误过滤修复,
                       4979 → 5440 只, "跳过指数" 462 → 1）
              → 池缓存指纹失效（原指纹含该文件 mtime）→ 相关运行【全量重算】
              → EB1d 长跑 9h42m、flush 31 次, 在 E0-full 的
                 A(04:21 加载) 与 B(15:43 加载) **之间**覆盖了 2023-04~2024-04 的行
              → A 用旧口径池(无 sz.000820 等深市主板股) / B 用新口径池
              → 分歧起点 2024-03-01, 差异股票 sz.001238 ↔ sz.000820（均 sz.000/001 段,
                 正是 sz_000 修复的指纹）→ +6.90% vs +6.27%

        新语义: **历史日期一旦落盘即冻结**。
          · 已存在日期 → 不再覆盖;
          · 本次新算结果与该日期已落盘内容【不一致】→ 视为"口径变更"或"非确定性",
            写入 `*__conflict.parquet` 旁路文件 + WARNING, 交人工裁决;
          · 口径确需升级 → **显式提升 `data.pool_cache_tag`**（如 o2_20260904 → o3_<日期>）
            新建版本文件, 而不是原地改写历史。
        """
        if not self._smallcap_new_dates:
            return
        try:
            rows = []
            for d, lst in self._smallcap_new_dates.items():
                for rank, (code, _mv) in enumerate(lst, start=1):
                    rows.append({'trade_date': d, 'ts_code': code, 'mv_rank': rank})
            new_df = pd.DataFrame(rows)

            # [FIX-DET9-2026-09-10][R8] **锁内"读-改-写"**:
            #   并发写者（如长跑实验）可能已在我方加载后改写了文件。
            #   故写入前【在排他锁内重新读取】，与新行合并后再原子替换 ——
            #   既不覆盖他人成果，也不产生混合版本。
            from .determinism_provenance import merge_pool_cache, file_lock, FileLockTimeout
            import os as _os
            lock_path = f'{self._smallcap_cache_path}.lock'
            fp_short = self._smallcap_cache_fingerprint()[:16]
            try:
                with file_lock(lock_path, timeout=180):
                    old = None
                    meta_old = {}
                    if os.path.exists(self._smallcap_cache_path):
                        try:
                            old = pd.read_parquet(self._smallcap_cache_path)
                        except Exception:
                            old = None
                    if os.path.exists(self._smallcap_meta_path):
                        try:
                            meta_old = json.load(open(self._smallcap_meta_path, encoding='utf-8'))
                        except Exception:
                            meta_old = {}

                    df_out, stats, conflicts = merge_pool_cache(old, new_df)

                    _tmp = f'{self._smallcap_cache_path}.tmp_{_os.getpid()}'
                    df_out.to_parquet(_tmp, index=False)
                    _os.replace(_tmp, self._smallcap_cache_path)

                    # 逐日期指纹: 本次写入的日期记录当前输入指纹（R7）
                    per_date = dict(meta_old.get('per_date') or {})
                    for _d in new_df['trade_date'].astype(str).unique():
                        per_date[_d] = fp_short
                    _keep = set(df_out['trade_date'].astype(str).unique())
                    per_date = {k: v for k, v in per_date.items() if k in _keep}

                    with open(self._smallcap_meta_path, 'w', encoding='utf-8') as f:
                        json.dump({'fingerprint': self._smallcap_cache_fingerprint(),
                                   'per_date': per_date,
                                   'n_dates': len(_keep),
                                   'n_replaced_dates': stats['n_replaced_dates'],
                                   'n_append_dates': stats['n_append_dates'],
                                   'n_diffs_vs_cached': stats['n_conflict_dates'],
                                   'updated': str(datetime.now())}, f, ensure_ascii=False)

                logger.info('[Pool] 选股池快照缓存落盘(锁内合并且原子替换): 替换 %d 日 / 新追加 %d 日 / '
                            '逐日期指纹 %d 条 / 总 %d 日',
                            stats['n_replaced_dates'], stats['n_append_dates'],
                            len(per_date), len(_keep))

                if conflicts:
                    cpath = self._smallcap_cache_path.replace('.parquet', '__conflict.parquet')
                    try:
                        pd.DataFrame([
                            {'trade_date': d, 'n_new': i['n_new'], 'n_old': i['n_old'],
                             'n_overlap': i['n_overlap']}
                            for d, i in sorted(conflicts.items())
                        ]).to_parquet(cpath, index=False)
                    except Exception:
                        cpath = '(旁路文件写入失败)'
                    logger.warning(
                        '[Pool][R7] %d 个日期的重算结果与既有缓存行不同 (如 %s)，差异明细: %s。'
                        '判定: (a)若池配置/市值等输入确有变化 → 属正常重算, **须同步重跑既有实验基线**; '
                        '(b)若输入未变而结果不同 → 表明存在**非确定性**, 须立即排查。',
                        len(conflicts), sorted(conflicts.keys())[:3], cpath)

                self._pool_cache_conflicts = stats['n_conflict_dates']
            except FileLockTimeout as _e:
                logger.warning('[Pool][R8] 获取缓存写锁超时 (%s) → 本次**跳过落盘**; '
                               '已算结果本次有效, 下次运行会重算缺失日期。', _e)
            self._smallcap_new_dates = {}
            self._smallcap_dirty = 0
        except Exception as e:
            logger.warning(f'[Pool] 选股池快照缓存落盘失败 ({e}), 下次回测将重算这些日期')

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
            # [B4-20260905] P3-2: BARRA 已禁用时不触发 get_or_build_snapshot (实证 28.1s/日冗余构建),
            #   缺失行业直接置 OTHER (中性化器原生支持)。
            missing = df['industry'].isna()
            if missing.any() and self._barra_enabled:
                try:
                    from strategies.wyckoff20.barra_builder import get_or_build_snapshot
                    barra_snap = get_or_build_snapshot(date, baseline=self._barra_baseline)
                    if barra_snap is not None and 'industry' in barra_snap.columns:
                        barra_ind = dict(zip(barra_snap['code'].astype(str).str.zfill(6),
                                             barra_snap['industry']))
                        df.loc[missing, 'industry'] = code_digits[missing].map(barra_ind)
                        n_filled = missing.sum() - df.loc[missing, 'industry'].isna().sum()
                        if n_filled > 0:
                            logger.debug(f'行业映射: 从BARRA快照补充{n_filled}只股票')
                except Exception:
                    pass
            elif missing.any():
                logger.debug(f'[B4] BARRA已禁用, {int(missing.sum())}只缺失行业置OTHER (跳过28s快照构建)')
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
        """[FIX-O-20260915] 按 trade_date **逐日截面**计算 Score/Conf（D-216）。

        ★ 原缺陷：原实现对传入 df 整体取分位。而训练路径传入的是**多日训练面板**
          （strategy.py:3233 的 train_data，≈240 日 × ~200 股），于是
          `valid.quantile(...)` / `valid.min()/max()/median()` 全部**跨日池化** ⇒
            ① 训练/生产口径错位：生产路径（:2437）传单日 T−1 快照，按日截面取分位。
               实测同一因子两种口径下 **73.6%~76.8% 的样本落入不同分数桶**
               ⇒ 模型学的是"面板分布"下的特征，推理时见到的是"单日分布"的特征。
            ② 验证集信息泄漏：池化面板含生产 val 段 ⇒ val 的因子值参与了训练样本的
               特征变换，破坏 `exclude_prod_val_in_fit` 声称的"生产 val 零接触"。
          修法：按 trade_date 分组逐日调用（单日输入走同一函数 ⇒ 训练 ≡ 生产）。
          发现方式：`_audit_lookahead_enum.py` 的 R2（跨日聚合机械枚举）规则。
        """
        if 'trade_date' not in df.columns:
            return self._compute_alpha191_scores_single(df)
        _dts = pd.to_datetime(df['trade_date'])
        if _dts.nunique() <= 1:
            return self._compute_alpha191_scores_single(df)
        _idx = df.index
        parts = [self._compute_alpha191_scores_single(g)
                 for _, g in df.groupby(_dts, sort=False)]
        _out = pd.concat(parts, axis=0)
        try:
            return _out.reindex(_idx)
        except Exception:
            return _out

    def _compute_alpha191_scores_single(self, df: pd.DataFrame) -> pd.DataFrame:
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

        # [Q5R-OPT′ 2026-09-23] preds_mode=replay：**免训练重放**（与训练版 random 臂逐位等价）
        #   依据：deliverables/Q5R波1批次设计与启动_20260921.md §10.2
        #   语义：载入 dump 缓存的【候选集 codes + 位移前分数向量】，用与 random 分支**完全相同的
        #        RNG 流**（同 seed_base + 同 date）做同一置换 ⇒ Top-N 与 prob_dict 逐位相同。
        #   零影响：仅当 experiment.preds_mode == 'replay' 时进入；默认 normal ⇒ 不进入本块。
        _exp_cfg_early = self.config.get('experiment') or {}
        if str(_exp_cfg_early.get('preds_mode', 'normal')).strip().lower() == 'replay':
            _q5r_early = self._q5r_replay_select(context, _exp_cfg_early)
            if _q5r_early is not None:
                return _q5r_early
            logger.warning(
                '[Q5R-OPT] ★replay 缓存缺失/异常 ⇒ 回退常规训练路径（该轮**不可比**，属降级而非静默）')

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
        _, pred_dict = result if result is not None else (None, {})
        # [B5-20260905] P3-1: pred_dict 为空(smallcap 无 V10)时 v10 特征恒为常数 0.5,
        #   零方差无信息 → 不入特征池 (原先依赖低方差过滤, 污染特征叙事)
        v10_has_signal = bool(pred_dict)
        if v10_has_signal:
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

        if v10_has_signal:
            feature_cols_neu.append('v10_pred_rank')  # [R3-1 A2] 排名百分位
            feature_cols_neu.append('v10_pred_raw')   # [Part C] 原始预测值（连续信号）
        else:
            logger.info("[B5] pred_dict 为空(smallcap 无 V10), v10_pred 常数特征不入特征池")

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
        # [Fix-20260907 同步 alpha191] 概率分布日志名区分回归(截面分位)与二分类(盈利概率)
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

        # [Q5R-preds_mode] 零信息对照：置换「预测值 ↔ 股票」的对应关系
        #   语义：取置换后的 Top-N ≡ 「池内随机 N 只」；唯一变量 = 选股是否含信息。
        #   关键：预测值的多重集逐位不变 ⇒ 只破坏"谁对应哪个分数"，不改变分数分布。
        #   默认 preds_mode=normal ⇒ 本分支不执行 ⇒ 与改造前逐位一致（零影响闭合）。
        _exp_cfg = self.config.get('experiment') or {}
        _preds_mode = str(_exp_cfg.get('preds_mode', 'normal')).strip().lower()
        if _preds_mode == 'dump':
            # [Q5R-OPT′] 零行为改变：不置换 prob_profit，只把【候选集 + 位移前分数】落盘。
            self._q5r_dump_cache(context, features_df, prob_profit, _exp_cfg)
        elif _preds_mode == 'random':
            _seed_base = int(_exp_cfg.get('random_seed', 0))
            _date_int = int(context.current_dt.strftime('%Y%m%d'))
            # ★内联式抽为方法（行为逐位不变，见 _test_q5r_opt.py 的行为级断言）
            prob_profit = self._q5r_permute(prob_profit, _seed_base, _date_int)
            logger.info(
                '[Q5R] preds_mode=random: seed_base=%d date=%d 已置换预测值↔股票对应（%d 只）',
                _seed_base, _date_int, len(prob_profit))
        n_select = min(MAX_META_SELECTED, len(prob_profit))

        # ---- [Q5BAR 2026-10-08] BARRA 风险预算式降权（mode='off' ⇒ 零影响闭合） ----
        #   ★只改"排序用分数" _score_for_rank，**不改 prob_dict**（下方仍用原 prob_profit）
        #     ⇒ 权重层完全不动 ⇒ 从设计上排除"求解器把权重集中到少数几只"的通道。
        #   ★信息性可证：日志的 Top-5 仍打印**模型原概率**（prob_profit），
        #     而 selected 由 _score_for_rank 决定 ⇒ 二者差异即 BARRA 的实际作用。
        _score_for_rank = np.asarray(prob_profit, dtype=float)
        if self._barra_rb_mode == 'risk_budget':
            try:
                _rb_expo = self._load_barra_expo_pit(context.current_dt)
                _rb_sd = self._barra_rb_cfg.get('shuffle_seed')
                if _rb_sd is not None:
                    _rb_sd = int(_rb_sd) + int(context.current_dt.strftime('%Y%m%d'))
                _rb_new, _rb_info = self._apply_barra_risk_budget(
                    [str(c) for c in features_df.index], _score_for_rank, _rb_expo,
                    self._barra_rb_cfg.get('factors') or ['volatility', 'beta'],
                    float(self._barra_rb_cfg.get('strength', 0.0) or 0.0),
                    float(self._barra_rb_cfg.get('exclude_quantile', 0.0) or 0.0),
                    _rb_sd,
                    float(self._barra_rb_cfg.get('min_coverage', 0.80) or 0.80))
                if _rb_new is not None:
                    _score_for_rank = np.asarray(_rb_new, dtype=float)
                    logger.info(
                        '[Q5BAR] 风险预算降权生效: n=%d 有效=%d 覆盖=%.1f%% factors=%s λ=%.2f '
                        'R中位=%.3f p90=%.3f 乘子=[%.3f,%.3f] 剔除=%d 改变分数=%d shuffle=%s',
                        _rb_info['n'], _rb_info['n_valid'], _rb_info['coverage'] * 100.0,
                        _rb_info['factors_used'], _rb_info['lam'], _rb_info['R_median'],
                        _rb_info['R_p90'], _rb_info['penalty_min'], _rb_info['penalty_max'],
                        _rb_info['n_excluded'], _rb_info['score_changed'],
                        _rb_info['shuffle_seed'])
                else:
                    logger.warning('[Q5BAR] 风险预算降权未生效（保持原排序）: %s', _rb_info)
            except Exception as _rb_e:                            # noqa: BLE001
                logger.warning('[Q5BAR] 风险预算降权异常（保持原排序）: %s', _rb_e)

        top_idx = np.argsort(_score_for_rank)[::-1][:n_select]
        selected = list(features_df.index[top_idx])
        if not np.all(np.isfinite(_score_for_rank[top_idx])):
            logger.warning('[Q5BAR] ⚠️ Top-%d 内含 %d 只被剔除标的（候选不足，需降 exclude_quantile）',
                           n_select, int(np.sum(~np.isfinite(_score_for_rank[top_idx]))))

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

    # ─────────────────────────────────────────────────────────────────────
    # [Q5R-OPT′ 2026-09-23] dump / replay 两个方法
    #   设计档：deliverables/Q5R波1批次设计与启动_20260921.md §10.2
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _q5r_permute(prob_profit, seed_base, date_int):
        """★唯一定义的置换：random 与 replay **共用本函数** ⇒ 等价性由构造保证。

        与改造前内联式**逐位相同**：
            _rng = np.random.default_rng(seed_base + date_int)
            prob_profit = _rng.permutation(prob_profit)
        """
        _rng = np.random.default_rng(int(seed_base) + int(date_int))
        return _rng.permutation(prob_profit)

    def _q5r_cache_dir(self, exp_cfg):
        d = str(exp_cfg.get('cache_dir', '_q5r_cache') or '_q5r_cache')
        if not os.path.isabs(d):
            d = os.path.join(os.getcwd(), d)
        return d

    def _q5r_dump_cache(self, context, features_df, prob_profit, exp_cfg):
        """落盘本轮【候选集 codes（与 prob_profit 同序）+ 位移前分数向量】。"""
        d = self._q5r_cache_dir(exp_cfg)
        os.makedirs(d, exist_ok=True)
        date_int = int(context.current_dt.strftime('%Y%m%d'))
        codes = np.asarray([str(c) for c in features_df.index])
        scores = np.asarray(prob_profit, dtype='float64')
        p = os.path.join(d, 'round_%d.npz' % date_int)
        np.savez_compressed(p, codes=codes, scores=scores,
                            date_int=np.asarray([date_int], dtype='int64'))
        logger.info('[Q5R-OPT] dump: date=%d 候选=%d 已写 %s', date_int, len(codes), p)

    def _q5r_replay_select(self, context, exp_cfg):
        """免训练重放：载缓存 → 同一置换 → Top-N + prob_dict。失败返回 None（交回常规路径）。"""
        try:
            d = self._q5r_cache_dir(exp_cfg)
            date_int = int(context.current_dt.strftime('%Y%m%d'))
            p = os.path.join(d, 'round_%d.npz' % date_int)
            if not os.path.exists(p):
                # ★R-159：外部依赖缺失**禁止静默降级** ⇒ 落 MISSING 标记 + ERROR 日志。
                #   落臂审计据此判该臂 INVALID（缓存不全 ⇒ 部分轮次已降级为训练版）。
                # ★21 臂共用同一 cache_dir ⇒ 标记文件必须**按 seed 隔离**，否则并发混写
                _q5r_sd = int(exp_cfg.get('random_seed', 0) or 0)
                try:
                    with open(os.path.join(d, '_q5r_replay_MISSING_s%d.txt' % _q5r_sd),
                              'a', encoding='utf-8') as _mf:
                        _mf.write('date=%d at=%s\n'
                                  % (date_int, time.strftime('%Y-%m-%d %H:%M:%S')))
                except Exception:                                  # noqa: BLE001
                    pass
                logger.error('[Q5R-OPT] ★replay 缓存缺失：date=%d ⇒ 本轮将降级为训练路径'
                             '（该臂判 INVALID，禁止进判据）', date_int)
                return None
            z = np.load(p, allow_pickle=False)
            codes = [str(c) for c in z['codes']]
            scores = np.asarray(z['scores'], dtype='float64')
            if len(codes) != len(scores) or len(codes) == 0:
                logger.warning('[Q5R-OPT] ★replay 缓存自洽失败：codes=%d scores=%d',
                               len(codes), len(scores))
                return None
            # [Q5R-OPT′ v4 · 2026-09-24 · D-289] ★恢复 dump 臂当轮 meta_auc
            #   免训练 ⇒ ensemble_manager=None 且 self.meta_auc 缺省 0.0
            #   ⇒ IC 门控（本文件 3929 行）判 0.0 < 0.02 ⇒ **空仓**（v3 短窗实测 5/5 轮空仓）。
            #   修法：从 sidecar 恢复该轮真实 meta_auc（只用于阈值比较 ⇒ 4dp 精度足够）。
            _sd = os.path.join(d, '_meta_auc_series.json')
            _ok_ic = False
            if os.path.exists(_sd):
                try:
                    _ser = json.load(open(_sd, encoding='utf-8')).get('series', {})
                    _k = str(date_int)
                    if _k in _ser:
                        self.meta_auc = float(_ser[_k])
                        _ok_ic = True
                except Exception:                                   # noqa: BLE001
                    _ok_ic = False
            if not _ok_ic:
                try:
                    with open(os.path.join(d, '_q5r_replay_MISSING_s%d.txt'
                                           % int(exp_cfg.get('random_seed', 0) or 0)),
                              'a', encoding='utf-8') as _mf:
                        _mf.write('date=%d MISSING_META_AUC at=%s\n'
                                  % (date_int, time.strftime('%Y-%m-%d %H:%M:%S')))
                except Exception:                                   # noqa: BLE001
                    pass
                logger.error('[Q5R-OPT] ★replay 缺少 meta_auc（sidecar）：date=%d ⇒ 拒绝降级（R-159）',
                             date_int)
                return None
            seed_base = int(exp_cfg.get('random_seed', 0))
            prob = self._q5r_permute(scores, seed_base, date_int)
            n_select = min(int(self.config.get('meta_labeling', {}).get('max_meta_selected', 20)),
                           len(prob))
            top_idx = np.argsort(prob)[::-1][:n_select]
            selected = [codes[i] for i in top_idx]
            prob_dict = {}
            sel_set = set(selected)
            for i, code in enumerate(codes):
                if code in sel_set:
                    prob_dict[code] = float(prob[i])
            logger.info('[Q5R-OPT] replay: date=%d seed_base=%d 候选=%d 选中=%d（免训练）',
                        date_int, seed_base, len(codes), len(selected))
            # ★运行期自证：追加记录命中轮次 ⇒ 落臂后审计「命中轮数 == 预期轮数」
            try:
                with open(os.path.join(d, '_q5r_replay_HITS_s%d.txt' % seed_base),
                          'a', encoding='utf-8') as _hf:
                    _hf.write('%d\n' % date_int)
            except Exception:                                      # noqa: BLE001
                pass
            return selected, prob_dict
        except Exception as e:                                   # noqa: BLE001
            logger.warning('[Q5R-OPT] ★replay 异常：%s', e)
            return None


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

    def _attach_total_mv(self, labels_df):
        """[Q5N-fix 2026-09-19] 按 (date, code) 从策略内市值缓存关联 total_mv。

        为什么必须在此处补：`labels_df` 产生于**特征面板合并之前**，其列仅
        [date, code, ret, realized_ret, exit_type, n_fills(, vol, fwd_ret_5)]
        —— **不含 total_mv**（total_mv 是后续 `_merge_fundamental_info` 才并入 train_data 的）。
        初版 Q5N 直接在 labels_df 上做 `fwd_ret_orth` ⇒ 每次重训抛
        `缺少必需列 ['total_mv']` ⇒ 元标注模型训练失败、退化为"主模型推荐结果"
        ⇒ **整臂无有效交易**（R-126 事故）。

        数据源：`self._mv_sorted = [ts_code, total_mv, _td]`（`_ensure_total_mv_cache` 已构建，
        与选股池所用市值同源 ⇒ 口径一致、无前视：事件日 ≤ 交易日）。
        """
        mv = getattr(self, '_mv_sorted', None)
        if mv is None or len(mv) == 0:
            try:
                _mv_df = self._load_total_mv_cache()
                _mv_df = _mv_df.sort_values('trade_date').reset_index(drop=True)
                _mv_df['_td'] = pd.to_datetime(_mv_df['trade_date']).values
                mv = _mv_df[['ts_code', 'total_mv', '_td']]
            except Exception as _e:
                raise RuntimeError(
                    '[Q5N] 市值缓存不可用（_mv_sorted 为空）⇒ 无法为 fwd_ret_orth 提供 total_mv: %r'
                    % _e)
        _m = mv[['ts_code', 'total_mv', '_td']].copy()
        _m['date'] = pd.to_datetime(_m['_td']).dt.normalize()
        _m = _m.rename(columns={'ts_code': 'code'})[['date', 'code', 'total_mv']]
        _m = _m.drop_duplicates(subset=['date', 'code'], keep='first')
        _out = labels_df.copy()
        _out['date'] = pd.to_datetime(_out['date']).dt.normalize()
        _out = _out.merge(_m, on=['date', 'code'], how='left')
        _n_miss = int(_out['total_mv'].isna().sum())
        logger.info('[Q5N] total_mv 关联: 命中 %d/%d，缺失 %d（缺失行将由变换层丢弃并计数）',
                    len(_out) - _n_miss, len(_out), _n_miss)
        return _out

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
            # [Q5N 闸⑦] 禁止静默口径切换（R-108 同族）
            #   此分支会使标签静默降级为 legacy 三重屏障 (pt_mult 6.3/sl_mult 4.7),
            #   而交易端仍用 staged_exit (S1/S2 = 4.5σ/−1.8σ) ⇒ D-225 同源性被无声破坏。
            #   实证：S17b_SL060PT150 与 Q5C_BARRA_sh932000 各 66 次重训中该分支触发 0 次
            #   ⇒ 改为显式失败不改变既有行为，只把"静默"变成"响"。
            raise RuntimeError(
                "[L1/L2] recommended_by_date 为空 ⇒ 拒绝静默降级为 A1/legacy 标签 "
                "(会使 标签屏障 ≠ 交易出场规则，破坏 D-225 同源性)。"
                "若确需 A1 标签，请显式设置 meta_labeling.label_mode=relative_barrier。")

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
            events_df, calendar,
            align_entry_mode=str(self.config.get('meta_labeling', {}).get(
                'label_entry_align', 'next_day_open')),
            # [Q5N] 仅当启用目标变换时才额外产出 vol/fwd_ret_5 ⇒ none 路径逐位不变
            extra_label_cols=(self._target_transform != 'none'))

        if staged_labels is None or staged_labels.empty:
            logger.warning("[L1/L2] 分阶段标签计算返回空，fallback 到 A1")
            return self._assign_meta_labels(sample_snaps, calendar)

        labels_df = staged_labels.copy()

        # Phase 3: 按模式生成 bin
        if mode == 'staged_rank':
            # [L2] 截面分位: 每个事件日内 realized_ret 的 rank pct (0,1]
            labels_df['bin'] = labels_df.groupby('date')['realized_ret'].rank(pct=True)
            # [Q5N] 目标截面变换（唯一被检验变量; staged_exit 逐键不变 ⇒ 不违反 D-225）
            #   none ⇒ 本分支不执行 ⇒ 与改造前逐位一致
            if self._target_transform != 'none':
                from strategies.small_alpha191_test3.target_transforms import (
                    apply_target_transform)
                # [Q5N-fix 2026-09-19] fwd_ret_orth 需要 total_mv，而 labels_df 产生于
                #   特征面板合并**之前** ⇒ 必须在此按 (date, code) 从策略内市值缓存关联。
                #   （初版遗漏此步 ⇒ N2 每次重训抛"缺少必需列 ['total_mv']" ⇒ 整臂无交易，见 R-126）
                if (self._target_transform == 'fwd_ret_orth'
                        and 'total_mv' not in labels_df.columns):
                    labels_df = self._attach_total_mv(labels_df)
                labels_df = apply_target_transform(
                    labels_df, self._target_transform, logger=logger)
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
        # [Q5R-OPT′ v2 2026-09-23 · D-289] preds_mode=replay ⇒ ★真正免训练
        #   v1 的短路点位于 `_meta_select_stocks` 内（本文件 2447 行），
        #   而训练发生在 3803 行（更早）⇒ 训练早已完成 ⇒ 无效。
        #   此处置于「是否重训」判定的最前，使 3795 的 `if not retrain_now` 直接放行。
        _q5r_exp = self.config.get('experiment') or {}
        if str(_q5r_exp.get('preds_mode', '') or '').lower() == 'replay':
            return False, 'preds_mode=replay ⇒ ★免训练（缓存重放，跳过标签/特征/训练）'
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
            label_mode = self._label_mode
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

        # [B5-20260905] P3-1: 全训练面板 v10_pred_raw 零方差(smallcap 无 V10, 恒 0.5) → 两列均不入池
        v10_train_has_signal = bool((train_data[v10_col_raw].std() or 0) > 1e-6)
        if v10_train_has_signal:
            feature_cols_neu.append(v10_col_rank)  # [R3-1 A2] 排名百分位
            feature_cols_neu.append(v10_col_raw)   # [Part C] 原始预测值（连续信号）
        else:
            logger.info("[B5] 训练面板 v10_pred_raw 零方差(常数0.5), v10_pred 特征不入池")

        if len(feature_cols_neu) < 1:  # [B5] 至少1个有效特征
            logger.error(f"中性化后无有效特征 (共{len(feature_cols_neu)}个)")
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
        _ml_cfg = self.config.get('meta_labeling', {})
        _pt_sl_mult = _ml_cfg.get('sample_weight_pt_sl_multiplier', 2.0)
        # ★[波4-波C′] 两子轴拆分（D-296）+ sham 种子（R-146）。
        #   三者缺省均为 None ⇒ 严格走原合并掩码分支 ⇒ **零影响闭合**。
        _pt_mult = _ml_cfg.get('sample_weight_pt_multiplier', None)
        _sl_mult = _ml_cfg.get('sample_weight_sl_multiplier', None)
        _sham_seed = _ml_cfg.get('sample_weight_sham_seed', None)
        # ★E-1 自证行（R-178：须覆盖该波**全部**被改变变量 pt/sl/sham）
        logger.info(f"[Q5T-波C'] sample_weight 口径: pt_sl_mult={_pt_sl_mult} "
                    f"pt_mult={_pt_mult} sl_mult={_sl_mult} sham_seed={_sham_seed}")
        _sw_cache_key = f"{train_end_key}|pt{_pt_mult}|sl{_sl_mult}|sham{_sham_seed}"
        sample_w = compute_sample_weights_cached(
            train_data, trade_days_all, self.meta_labeler.max_hold_days,
            cache_key=_sw_cache_key,
            exit_type=train_data['exit_type'] if 'exit_type' in train_data.columns else None,
            pt_sl_multiplier=_pt_sl_mult,
            pt_multiplier=_pt_mult,
            sl_multiplier=_sl_mult,
            sham_seed=_sham_seed,
        )

        # ================================================================
        # [2026-08-18] 因子验证三件套调用段已移除 (SHAP时序/ADV/CIC 全删)
        # 原功能: ensemble 训练前 run_rebalance() 投票剔除劣质因子
        # 移除依据: 审查结论 (0票空转/冗余/86%误杀) + 用户决策
        # 特征筛选完全交由 ensemble 内 Purged K-Fold (低方差+IC+PSI+MDA投票)
        # ================================================================

        # ★[Z23] 同日标签置换泄漏检验（D-306 预注册；默认 None ⇒ 严格走原路径 ⇒ 零影响闭合）
        #   做法：在【同日截面内】置换 target 列 label_profit（保持每日标签分布不变）
        #   判据（预注册 R-160）：种子 x3；通过 = 三轮 meta_IC 全部 ∈ [-0.03, +0.03]
        #        且三轮 95% CI 均含 0（锚 0.2481 ⇒ 塌幅 >=87%）；任一 >0.03 且 CI 不含 0 ⇒ 泄漏事故调查
        _lp_seed = _ml_cfg.get('label_permutation_seed', None)
        _lp_groups = 0
        _lp_rows = 0
        if _lp_seed is not None:
            if 'trade_date' not in train_data.columns:
                raise KeyError("[Z23] label_permutation: train_data 缺 trade_date 列 ⇒ fail-fast (R-159)")
            if 'label_profit' not in train_data.columns:
                raise KeyError("[Z23] label_permutation: train_data 缺 label_profit 列 ⇒ fail-fast (R-159)")
            _lp_rng = np.random.default_rng(int(_lp_seed))
            _lp_new = train_data['label_profit'].to_numpy(copy=True)
            for _lp_td, _lp_idx in train_data.groupby('trade_date').indices.items():
                _lp_ii = np.asarray(_lp_idx)
                _lp_new[_lp_ii] = _lp_new[_lp_ii][_lp_rng.permutation(_lp_ii.size)]
                _lp_groups += 1
                _lp_rows += int(_lp_ii.size)
            train_data = train_data.copy()
            train_data['label_profit'] = _lp_new
        # ★E-1 自证行（R-178：须覆盖该波**全部**被改变变量 seed / groups / rows）
        logger.info(f"[Z23] label_permutation: seed={_lp_seed} groups={_lp_groups} rows={_lp_rows}")

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
        # [Fix-20260907 同步 alpha191] rank 模式度量是 Spearman IC, 日志名区分 AUC/IC 避免误导
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

    def _barra_trace(self, current_date):
        """[FIX-DET10][R4] 记录 BARRA 快照选取, 并检出运行期按需重建。

        返回本次选取的快照日期字符串（YYYYMMDD）或 None。
        选取语义与 barra_builder.get_or_build_snapshot 一致: 目录中**严格 < 调仓日**的最大日期。
        """
        import os as _os
        bdir = (self.config.get('barra', {}) or {}).get('snapshot_dir') or './data/barra_exposures'
        if not _os.path.isabs(bdir):
            bdir = _os.path.abspath(bdir)
        if not _os.path.isdir(bdir):
            return None
        dates = []
        for fn in _os.listdir(bdir):
            if fn.startswith('barra_expo_') and fn.endswith('.parquet'):
                stem = fn[len('barra_expo_'):-len('.parquet')]
                try:
                    dates.append(datetime.strptime(stem, '%Y%m%d'))
                except ValueError:
                    continue
        if self._barra_snap_files is None:
            self._barra_snap_files = set(dates)
        else:
            _new = set(dates) - self._barra_snap_files
            if _new:
                logger.warning('[BARRA][R4] ⚠️ 本次调用**新增了 %d 个快照文件** (如 %s) —— '
                               '说明发生了运行期按需重建, 会改变后续调仓日的快照选取口径。'
                               '本次结果与既有实验可能不可比, 已记入 provenance。',
                               len(_new), sorted(str(d.date()) for d in list(_new)[:3]))
                self._barra_snap_files |= _new
                self._barra_rebuilt = True
        valid = [d for d in dates if d < pd.Timestamp(current_date)]
        if not valid:
            return None
        return max(valid).strftime('%Y%m%d')

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

        # [E3 2026-09-09] 目标敞口闭环: 日频 vol_scale(池指数+不对称恢复) + 欠配再部署/超配减仓预标记
        self._update_daily_vol_scale(current_date)
        self._update_exposure_gap(context)

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
        # [Q5R-OPT′ v3 · 2026-09-24 · D-289] ★replay 模式必须进入 _meta_select_stocks：
        #   免训练路径下 `ensemble_manager` **不会被创建**（=None），若沿用原条件会落到 else 分支
        #   ⇒ `self.target_stocks = recommended[:20]` **回退主模型推荐** ⇒ replay 缓存选股**完全失效**。
        #   实测：v2 短窗臂 `[Q5R-OPT] replay` 命中 0 次（应 5 次）。
        #   非 replay 路径不受影响（`_q5r_is_replay` 恒为 False ⇒ 条件等价于原来）。
        _q5r_exp_v3 = self.config.get('experiment') or {}
        _q5r_is_replay = str(_q5r_exp_v3.get('preds_mode', '') or '').strip().lower() == 'replay'
        if self.use_meta_labeling and (self.ensemble_manager is not None or _q5r_is_replay):
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
        # [FIX-DET6-2026-09-10] 全量落日志（**仅日志, 不改变任何行为**）:
        #   用于验证"同一调仓日在不同回测区间下选股是否一致"（跨区间可复现性验收）。
        #   原先仅记录 5 只示例, 无法做逐日全量比对。格式固定为一行, 便于 grep/比对。
        logger.info(f"[选股快照] {current_date.strftime('%Y%m%d')} | n={len(self.target_stocks)} | "
                    f"{','.join(str(c) for c in self.target_stocks)}")

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
        # [Fix-20260907 同步 alpha191] rank 模式 IC 门控**强制启用**(不再依赖
        #   auc_gate_enabled, 避免用户仅切换 label_mode 后失去模型质量把关); binary 分支行为不变。
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
                barra_expo = get_or_build_snapshot(current_date, baseline=self._barra_baseline)
                # [FIX-DET10-2026-09-10][R4] BARRA 快照可追溯 + 运行期重建检出
                #   缺陷: get_or_build_snapshot 选取"最近 ≤ 调仓日的**已存在**快照", 且缺伴随文件时
                #   会**按需重建并写盘** ⇒ 结果依赖"目录当前有哪些文件", 重建还会改变后续选取。
                #   护栏(不修改共享 barra_builder): ① 记录本次选取的快照日期 → 进 provenance;
                #   ② 检出"本次调用新增了快照文件" → 显式 WARNING; ③ 目录清单摘要由 provenance 前后比对。
                try:
                    _sel = self._barra_trace(current_date)
                    if _sel:
                        self._barra_snap_used.add(_sel)
                except Exception:
                    pass
                if barra_expo is not None and len(barra_expo) > 100:
                    barra_weights_raw, barra_ok = self._barra_solve(
                        preds=probs,
                        selected_codes=codes,
                        barra_expo=barra_expo,
                        risk_aversion=self._barra_cfg.get('risk_aversion', 1.0),
                        max_weight=self._barra_cfg.get('max_weight', 0.10),
                        constraint_date=current_date,
                        total_exposure=1.0,  # BARRA 始终全投资约束 Σw=1.0
                        lambda_dev=self._barra_cfg.get('lambda_dev', 0.5),
                        lambda_l2=self._barra_cfg.get('lambda_l2', 0.5),
                        lambda_ind=self._barra_cfg.get('lambda_ind', 0.3),
                        baseline=self._barra_baseline,
                        expected_ic=float(getattr(self, 'meta_auc', 0.05) or 0.05),
                        alpha_scale=float(self._barra_cfg.get('alpha_scale', 10.0)),
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

                # [E3 2026-09-09] 日频敞口优先: 使用每日盘前已算好的 _daily_vol_scale
                #   (池指数 EWMA + 不对称恢复), 取代调仓日一算管 20 天的粗粒度
                if self._garch_cfg.get('use_daily_scale', False):
                    total_exposure = max(0.0, min(1.0, float(getattr(self, '_daily_vol_scale', 1.0))))
                    logger.info(f'[策略分配] GARCH(E3日频) total_exposure={total_exposure:.3f} '
                                f'(vol_scale 每日盘前更新, 不对称恢复)')
                    ret_matrix = None
                elif self._garch_cfg.get('use_pool_index', False):
                    ret_matrix = self._build_pool_index_return_matrix(
                        self._garch_ctrl.lookback, current_date)
                    if ret_matrix is not None and ret_matrix.shape[1] >= 1:
                        garch_scale, annual_vol, info = self._garch_ctrl.compute(
                            np.array([1.0]), ret_matrix, context.portfolio.total_value
                        )
                        total_exposure = max(0.0, min(1.0, garch_scale))
                        logger.info(f'[策略分配] GARCH(B池指数) total_exposure={total_exposure:.3f} '
                                    f'annual_vol={annual_vol:.2%} '
                                    f'method={info.get("method", "?")} '
                                    f'dd={info.get("current_drawdown", 0):.2%}')
                    else:
                        logger.debug('[策略分配] GARCH(方案B) 池指数矩阵构建失败, 保持 total_exposure=1.0')
                    ret_matrix = None  # 防止进入下方方案A分支重复计算
                else:
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

        # ---- 4b) [Q5P′ 2026-09-24] 选股层波动率惩罚（默认关闭 ⇒ 零影响闭合） ----
        #   w'_i ∝ w_i × (σ_med/σ_i)**λ；Σw 守恒（总敞口不变，D-277 不碰敞口层）
        #   σ 取生产口径 self._compute_entry_vol（与标签端同源，R-145）
        _vp_cfg = {}
        try:
            _vp_cfg = self.config.get('vol_penalty') or {}
        except Exception:
            _vp_cfg = {}
        _vp_lam = float(_vp_cfg.get('strength', 0.0) or 0.0)
        if bool(_vp_cfg.get('enabled', False)) and _vp_lam > 0.0 and self.target_weights:
            try:
                _vp_codes = list(self.target_weights.keys())
                _vp_w = [float(self.target_weights[c]) for c in _vp_codes]
                _vp_sig = [float(self._compute_entry_vol(c, current_date)) for c in _vp_codes]
                _vp_new, _vp_info = self._apply_vol_penalty_weights(
                    _vp_codes, _vp_w, _vp_sig, _vp_lam)
                if _vp_new is not None:
                    self.target_weights = _vp_new
                    logger.info(
                        f"[Q5P′] vol_penalty λ={_vp_info['lam']:.2f} n={_vp_info['n']} "
                        f"σ_sel中位={_vp_info['sigma_ref']:.4%} "
                        f"σ比=[{_vp_info['sigma_ratio_min']:.2f},{_vp_info['sigma_ratio_max']:.2f}] "
                        f"Σw {_vp_info['sum_w_before']:.4f}→{_vp_info['sum_w_after']:.4f} "
                        f"max_w {_vp_info['max_w_before']:.4f}→{_vp_info['max_w_after']:.4f} "
                        f"HHI {_vp_info['hhi_before']:.4f}→{_vp_info['hhi_after']:.4f}")
                else:
                    logger.warning(f"[Q5P′] vol_penalty 跳过（保持原权重）: {_vp_info}")
            except Exception as _e:
                logger.warning(f"[Q5P′] vol_penalty 异常 (保持原权重): {_e}")

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
                # [O1-20260904] 到期/IPBS清仓后同步 drop tracker (错位分批下残留
                #   tracker 会成为孤儿: 不在 positions 不会再被 staged 循环触碰,
                #   但 trackers 字典永久累积 → 内存脏状态 + len(trackers) 统计虚高)
                self.stage_mgr.drop(code)
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
                self.stage_mgr.drop(code)  # [O1] 同步 drop 幽灵 tracker

        # 清理IPBS退出列表（避免跨日残留）
        self._ipbs_exit_list = []

        # ---------- [E3 2026-09-09] 目标敞口闭环执行（开盘价） ----------
        # 欠配再部署: 按预标记金额买入现存持仓 (未触发S2、不在锁定名单);
        # 超配减仓: 按预标记金额卖出锁定名单持仓。
        # 仅在非调仓日执行 (调仓日由 Step B 统一部署, 避免与全清重建冲突)。
        _is_reb_e3 = (self.trade_day_seq % self.rebalance_interval == 1)
        if not _is_reb_e3 and getattr(self, '_garch_enabled', False):
            for code, amt in getattr(self, '_redeploy_list', []):
                if code in portfolio.positions:
                    try:
                        daily_d = self.data_provider.get_daily_data(
                            code, date, date, reference_date=self.reference_date)
                        if not daily_d.empty and 'open' in daily_d.columns:
                            px_open = daily_d['open'].iloc[0]
                            broker.order_value(
                                code, min(amt, float(portfolio.cash) * 0.95),
                                trade_time=f'{date.date()} 09:31:00',
                                exec_price_override=px_open,
                                apply_slippage=True,
                                trade_reason='e3_redeploy')
                    except Exception as e:
                        logger.warning(f'[E3-敞口] 再部署 {code} 异常: {e}')
            for code, amt in getattr(self, '_derisk_list', []):
                if code in portfolio.positions:
                    try:
                        daily_d = self.data_provider.get_daily_data(
                            code, date, date, reference_date=self.reference_date)
                        if not daily_d.empty and 'open' in daily_d.columns:
                            px_open = daily_d['open'].iloc[0]
                            broker.order_value(
                                code, -min(amt, float(portfolio.positions[code].market_value or 0)),
                                trade_time=f'{date.date()} 09:31:00',
                                exec_price_override=px_open,
                                apply_slippage=True,
                                trade_reason='e3_derisk')
                    except Exception as e:
                        logger.warning(f'[E3-敞口] 减仓 {code} 异常: {e}')
            self._redeploy_list = []
            self._derisk_list = []

        # ---------- 1. 分阶段止盈止损（替换旧单一屏障） ----------
        is_rebalance = (self.trade_day_seq % self.rebalance_interval == 1)
        _batches = getattr(self, 'rebalance_batches', 1)
        _stagger = _batches > 1
        _rot_cohort = None
        if _stagger and is_rebalance:
            # [O1] 轮动日: 本批次即将全清重建 → 不对其再做分级退出 (避免重复卖);
            #     其余批次仍照常分级退出 (原实现整日跳过 → 其余批次少 1 日保护, 已修正)
            _rot_cohort = ((self.trade_day_seq - 1) // self.rebalance_interval) % _batches
        _staged_run = self.staged_exit_enabled and (not is_rebalance or _rot_cohort is not None)

        if _staged_run:
            # 预加载今日所有持仓分钟线（批量利用 provider 缓存）
            minute_map = {}
            for code in list(portfolio.positions.keys()):
                if _rot_cohort is not None and code in self.holdings \
                        and self.holdings[code].get('cohort', -1) == _rot_cohort:
                    continue  # 轮动批次持仓由 Step A 统一开盘价清仓, 不预载
                try:
                    mdf = self.data_provider.get_minute_data(
                        code, datetime.combine(date, datetime.min.time()),
                        datetime.combine(date, datetime.max.time()),
                        freq='5', reference_date=self.reference_date)
                    minute_map[code] = mdf if not mdf.empty else None
                except Exception:
                    minute_map[code] = None

            for code in list(portfolio.positions.keys()):
                if _rot_cohort is not None and code in self.holdings \
                        and self.holdings[code].get('cohort', -1) == _rot_cohort:
                    continue  # 同上: 跳过轮动批次
                if code not in self.holdings:
                    # 孤儿清理
                    logger.warning(f"[{date.date()}] {code} 孤儿持仓，清理")
                    broker.order_target_percent(code, 0.0, trade_reason='orphan_cleanup')
                    if code in self.volatility_cache: del self.volatility_cache[code]
                    continue

                tracker = self.stage_mgr.trackers.get(code)
                if tracker is None:
                    # 无 tracker（防御：可能调仓日未注册，或旧持仓残留）
                    # 用分阶段 stage1 阈值兜底（与 STAGE_CONFIG 同源, 替代旧硬编码 6.3σ/4.7σ）
                    entry_price = self.holdings[code].get('price', 0)
                    entry_price_raw = self.holdings[code].get('price_raw', entry_price)
                    vol = self.holdings[code].get('daily_vol', 0.02)
                    if vol <= 0: vol = 0.02
                    pt_th = entry_price * (1 + self.STAGE_CONFIG[1]['pt'] * vol)
                    sl_th = entry_price * (1 + self.STAGE_CONFIG[1]['sl'] * vol)
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
                    # [FIX-N-20260915] 拒单回滚: 原实现忽略 order_value 返回值 ⇒ 当卖出被拒
                    #   (跌停/停牌/无价) 时仍标记阶段完成、置 pt_after_sl、并把
                    #   sell_qty*price*0.999 虚拟计入 freed_cash_pool —— 即"以为卖了其实没卖",
                    #   且该阶段永不重试。实测被拒占触发 0.24%(Q2c) ~ 0.97%(P0_Bare)。
                    #   现: 仅在成交成功时消费阶段标记; 失败则回滚, 由后续 bar 重新判定。
                    _st_pre = (tracker.stage1_done, tracker.stage2_done,
                               tracker.stage3_done, tracker.stage4_done,
                               tracker.s1_pt_hit, tracker.s2_pt_hit,
                               tracker.s3_pt_hit, tracker.pt_after_sl)
                    _filled = broker.order_value(
                        code, -sell_qty * trig_price_raw,
                        trade_time=trig_time, exec_price_override=trig_price_raw,
                        apply_slippage=True,
                        trade_reason=f'stage{stg}_{"pt" if is_pt else "sl"}')
                    if not _filled:
                        (tracker.stage1_done, tracker.stage2_done, tracker.stage3_done,
                         tracker.stage4_done, tracker.s1_pt_hit, tracker.s2_pt_hit,
                         tracker.s3_pt_hit, tracker.pt_after_sl) = _st_pre
                        logger.info(f"[{date.date()}] {code} Stage{stg} "
                                    f"{'止盈' if is_pt else '止损'}卖出被拒（跌停/停牌/无价）"
                                    f" ⇒ 回滚阶段标记，待后续 bar 重试")
                        continue
                    # [E3 2026-09-09] 止损锁定名单: SL 触发标的本周期内禁止再部署回补
                    #   (v5 质询: 机械回补=抵消止损尾部截断+变相摊低成本; PT 不锁定)
                    if not is_pt:
                        self._sl_locked.add(code)
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

            # 闲置现金再投资 (仅非调仓日; 调仓日/轮动日资金在 Step B 统一部署,
            # 避免与批次清仓逻辑冲突)
            if (not is_rebalance and self.reinvest_enabled
                    and self.stage_mgr.freed_cash_pool > self.reinvest_min_amount):
                self._reinvest_freed_cash(broker, portfolio)

        elif is_rebalance:
            # 调仓日: 分阶段跳过, 所有持仓在下方调仓环节统一全清
            pass

        # ---------- 2. 调仓逻辑（仅在调仓日执行） ----------
        if is_rebalance:
            # [O1-20260904] 错位分批调仓: batches>1 时每 rebalance_interval 日轮动 1 批,
            #   批次由调仓序列号取模决定 (全周期 interval×batches ≈ 原 20 日)。
            #   设计要点:
            #     - 仅清空"当前批次"持仓 → 其余批次继续持有 (错位重叠, 利用率↑)
            #     - 每批次仅部署 ~1/batches × 目标权重 (不空槽补仓: 中间日释放现金
            #       不追买, 静默等待本批次轮动日统一再部署)
            #     - FLAT 熔断态 → 不清仓不建仓 (只靠每日到期/分级退出自然降险)
            batches = getattr(self, 'rebalance_batches', 1)
            staggered = batches > 1
            if staggered:
                cohort_c = ((self.trade_day_seq - 1) // self.rebalance_interval) % batches
                logger.info(f"[{date.date()}] [O1-错位轮动] 批次 {cohort_c + 1}/{batches} "
                            f"(每{self.rebalance_interval}日轮动, 全周期≈"
                            f"{self.rebalance_interval * batches}日)")
            else:
                cohort_c = None

            # ---------- Step A: 清仓 (分批: 仅当前批次; 旧行为: 全清) ----------
            if staggered and getattr(self, '_dd_state', 0) == 2:
                # FLAT: 不强清 (风控=限建仓), 到期/分级退出自然降险
                logger.warning(f"[{date.date()}] [O1-错位轮动] FLAT 熔断态 → 跳过本批次轮动 "
                               f"(不清仓不建仓, 等待自然到期)")
            else:
                if staggered:
                    sell_codes = [code for code in list(portfolio.positions.keys())
                                  if code in self.holdings
                                  and self.holdings[code].get('cohort', -1) == cohort_c]
                else:
                    sell_codes = list(portfolio.positions.keys())
                for code in sell_codes:
                    broker.order_target_percent(
                        code, 0.0,
                        trade_reason='rebalance_clear_all' if not staggered
                        else f'cohort{cohort_c}_rotate')
                # 清理 holdings + volatility 缓存 + tracker (分批: 仅当前批次)
                for code in sell_codes:
                    if code in self.holdings: del self.holdings[code]
                    if code in self.volatility_cache: del self.volatility_cache[code]
                    if staggered:
                        self.stage_mgr.drop(code)
                    else:
                        pass
                if not staggered:
                    self.stage_mgr.reset()
                    # [E3 2026-09-09] 调仓日全清重建 → 止损锁定名单清空 (新周期重新计)
                    self._sl_locked = set()

                # ---------- Step B: 建仓 (分批: 目标权重 × 1/batches) ----------
                # [2026-08-29 方案A] 建仓规模 = 回撤熔断 scale × 趋势过滤 scale (只限建仓, 不强清)
                self._update_rebalance_trend_gate(date)
                build_scale = self._current_build_scale()
                if build_scale < 1.0:
                    logger.info(f'[{date.date()}] [方案A] 建仓规模 x{build_scale:.2f} '
                                f'(回撤熔断={self._dd_scale}, 趋势={self._trend_scale})')
                # [Fix-20260907 同步 alpha191] build_scale<=0 → 明确空仓语义: 跳过建仓循环,
                #   避免对 target 下 0 仓单产生"买入未成交回滚"误报 (x0.00 应直接空仓)
                if build_scale <= 0:
                    logger.info(f'[{date.date()}] [方案A] 建仓规模=0 (回撤熔断={self._dd_scale}, '
                                f'趋势={self._trend_scale}) → 本次调仓保持空仓 (不建仓)')
                    return
                held_codes = set()
                # 分批: 每批次只建仓 n_cohort 只 (总持仓≈MAX_POSITIONS), 批次内权重
                #   归一 × (全局目标敞口 S_raw / batches):
                #   → 4 批各持 S_raw/batches, 错位重叠期总敞口≈S_raw (保留 GARCH/BARRA 全局敞口语义)
                _tw_items = sorted(self.target_weights.items(), key=lambda x: x[1], reverse=True)
                if staggered:
                    _n_cohort = max(1, self.MAX_POSITIONS // batches)
                    _cand = [(c, w) for c, w in _tw_items
                             if w > 0 and c not in portfolio.positions]
                    _cand = _cand[:_n_cohort]
                    _S_raw = sum(w for _, w in _tw_items if w > 0) or 1.0
                    _wsum = sum(w for _, w in _cand) or 1.0
                    _tw_items = [(c, w / _wsum * (_S_raw / batches)) for c, w in _cand]
                for code, target_w in _tw_items:
                    if target_w <= 0: continue
                    open_p_adj, open_p_raw = self._get_open_price(code, date)
                    if open_p_adj is None: continue

                    # 计算 EWMA vol
                    dv = self._compute_entry_vol(code, date)
                    if dv <= 0: dv = 0.02

                    self.holdings[code] = {
                        'date': date, 'price': open_p_adj, 'price_raw': open_p_raw,
                        'daily_vol': dv, 'dividend_history': [],
                        'cohort': cohort_c if staggered else None,
                    }

                    _order_w = target_w * build_scale
                    logger.info(f'[{date.date()}] {code} 建仓 买入={open_p_raw:.2f}'
                                f'(复权{open_p_adj:.2f}) '
                                f'仓位={_order_w:.1%}'
                                + (f' [批次{cohort_c + 1}/{batches}]' if staggered else ''))
                    broker.order_target_percent(code, _order_w, trade_reason='new_open')
                    if code not in portfolio.positions:
                        if code in self.holdings: del self.holdings[code]
                        logger.warning(f"[{date.date()}] {code} 买入订单未成交，回滚")
                    else:
                        if code in self.holdings:
                            self.holdings[code]['quantity'] = float(portfolio.positions[code].quantity)

                # ---------- Step C: 为新持仓初始化 StageTracker (分批: 仅本批次新仓) ----------
                if staggered:
                    for code in list(portfolio.positions.keys()):
                        h = self.holdings.get(code, {})
                        if h.get('cohort', -1) != cohort_c:
                            continue  # 其他批次 tracker 保留
                        if code in self.stage_mgr.trackers:
                            self.stage_mgr.drop(code)
                        self.stage_mgr.register(
                            code=code, entry_date=date,
                            entry_adj=h.get('price', 0),
                            entry_raw=h.get('price_raw', 0),
                            qty=float(portfolio.positions[code].quantity),
                            daily_vol=h.get('daily_vol', 0.02))
                else:
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

    # ==================== [E3] 目标敞口闭环 ====================
    def _update_daily_vol_scale(self, current_date):
        """[E3] 日频波动率目标敞口 (池指数信号源 + 不对称恢复)。

        vol_scale_raw = clip(vol_target / vol_ann, min_scale, max_scale)。
        不对称恢复: 指数收盘 > MA5 时允许 scale 上升; 否则只降不升
        (防 V 型反弹前过早恢复 — E2-A 失败教训)。
        数据: data/index_daily/{risk_index_code}.parquet, ≤T-1 严格无未来。
        """
        if not self._garch_enabled or self._garch_ctrl is None:
            self._daily_vol_scale = 1.0
            return
        try:
            gc = self._garch_cfg
            vol_target = float(gc.get('vol_target', 0.24))
            min_scale = float(gc.get('min_scale', 0.15))
            max_scale = float(gc.get('max_scale', 1.0))
            ewma_lambda = float(gc.get('ewma_lambda', 0.94))
            idx_code = getattr(self, 'risk_index_code', 'sh932000')
            path = os.path.join('data', 'index_daily', f'{idx_code}.parquet')
            if not os.path.exists(path):
                self._daily_vol_scale = getattr(self, '_daily_vol_scale', 1.0)
                return
            df = pd.read_parquet(path)
            dcol = 'trade_date' if 'trade_date' in df.columns else 'date'
            df[dcol] = pd.to_datetime(df[dcol])
            df = df.set_index(dcol).sort_index()
            px_hist = df['close']
            px_hist = px_hist[px_hist.index < pd.Timestamp(current_date)]  # ≤T-1
            if len(px_hist) < 30:
                self._daily_vol_scale = getattr(self, '_daily_vol_scale', 1.0)
                return
            rets = px_hist.pct_change().dropna()
            # [2026-09-21 Q5D-T] 信号类型分支：trend（趋势/均线）| vol（默认，原行为完全不变）
            _signal_type = str(gc.get('signal_type', 'vol')).strip().lower()
            if _signal_type == 'trend':
                _ma_days = int(gc.get('trend_ma_days', 60))
                _floor = float(gc.get('trend_floor', 0.5))
                if len(px_hist) < _ma_days + 1:
                    self._daily_vol_scale = getattr(self, '_daily_vol_scale', 1.0)
                    return
                _ma_v = float(px_hist.iloc[-_ma_days:].mean())
                _px_last = float(px_hist.iloc[-1])
                new_scale = 1.0 if _px_last > _ma_v else _floor
                prev = float(getattr(self, '_daily_vol_scale', 1.0))
                self._daily_vol_scale = new_scale
                if abs(new_scale - prev) > 1e-3:
                    _va = float(np.sqrt(max(float(rets.ewm(alpha=1 - ewma_lambda, adjust=False).var().iloc[-1]), 1e-12) * 244))
                    logger.info(f'[E3-敞口] {current_date.date()} trend ma{_ma_days}={_ma_v:.4f} '
                                f'px={_px_last:.4f} vol_ann={_va:.1%} scale={new_scale:.2f} '
                                f'({"above>MA" if _px_last > _ma_v else "below<=MA(floor=%.2f)" % _floor})')
                return
            ewma_var = float(rets.ewm(alpha=1 - ewma_lambda, adjust=False).var().iloc[-1])
            vol_ann = float(np.sqrt(max(ewma_var, 1e-12) * 244))
            raw_scale = float(np.clip(vol_target / vol_ann, min_scale, max_scale))
            # 不对称恢复: 仅当指数站上 MA_N 允许恢复; 否则只降不升
            # [2026-09-09 审计] MA 周期配置化 (原硬编码 5, 属魔法数字); 默认 5 保持原行为
            _recovery_ma = int(gc.get('recovery_ma_days', 5))
            _ma_n = float(px_hist.iloc[-_recovery_ma:].mean())
            above_ma = float(px_hist.iloc[-1]) > _ma_n
            prev = float(getattr(self, '_daily_vol_scale', 1.0))
            new_scale = raw_scale if above_ma else min(raw_scale, prev)
            self._daily_vol_scale = new_scale
            if abs(new_scale - prev) > 1e-3:
                logger.info(f'[E3-敞口] {current_date.date()} vol_ann={vol_ann:.1%} '
                            f'raw={raw_scale:.2f} scale={new_scale:.2f} '
                            f'({"恢复允许(>MA%d)" % _recovery_ma if above_ma else "只降不升(<MA%d)" % _recovery_ma})')
        except Exception as e:
            logger.warning(f'[E3-敞口] vol_scale 计算异常: {e}')
            self._daily_vol_scale = getattr(self, '_daily_vol_scale', 1.0)

    def _update_exposure_gap(self, context):
        """[E3] 目标敞口闭环: |E_act - E_tgt| > band → 预标记调节 (handle_bar 开盘执行)。

        欠配 → 再部署到现存持仓 (未触发 S2、不在止损锁定名单);
        超配 → 减仓 (锁定名单优先)。
        """
        self._redeploy_list = []
        self._derisk_list = []
        if not self._garch_enabled:
            return
        try:
            portfolio = context.portfolio
            nav = float(portfolio.total_value)
            if nav <= 0:
                return
            # [E3-Fix] Portfolio 无 position_value 属性 → 持仓市值手动求和
            pos_val = sum(float(p.market_value) for p in portfolio.positions.values())
            e_act = pos_val / nav
            l3 = 0.0 if getattr(self, '_dd_state', 0) == 2 else 1.0
            e_tgt = float(np.clip(getattr(self, '_daily_vol_scale', 1.0) * l3, 0.0, 1.0))
            band = float(self._garch_cfg.get('exposure_band', 0.05))
            gap = e_tgt - e_act
            if abs(gap) <= band:
                return
            locked = getattr(self, '_sl_locked', set())
            # [E3-fix2 2026-09-09] 欠配再部署默认关闭 (redeploy_enabled=false):
            #   E3-a half 实证: 再部署=机械加仓"未触发止损但正在阴跌"的现存持仓
            #   (2023H2 微盘普跌) → 平均仓位 91% + 单位敞口 -29.3% + MaxDD -18.7%
            #   严重劣化。欠配由调仓日正常建仓流程解决 (建仓规模=vol_scale 缩放)。
            if gap > 0:
                if not self._garch_cfg.get('redeploy_enabled', False):
                    return
                # 欠配 → 再部署
                deploy = min(float(portfolio.cash), gap * nav)
                candidates = []
                for code in list(self.holdings.keys()):
                    if code in locked:
                        continue
                    tr = self.stage_mgr.trackers.get(code) if self.stage_mgr else None
                    if tr is not None and getattr(tr, 'stage2_done', False):
                        continue
                    if code in portfolio.positions:
                        candidates.append(code)
                if candidates and deploy > self.reinvest_min_amount:
                    amt_each = deploy / len(candidates)
                    self._redeploy_list = [(c, amt_each) for c in candidates]
                    logger.info(f'[E3-敞口] 欠配 {gap:+.1%} (act={e_act:.1%}, tgt={e_tgt:.1%}) '
                                f'→ 再部署 {len(candidates)} 只 × {amt_each:,.0f}元')
                else:
                    logger.debug(f'[E3-敞口] 欠配 {gap:+.1%} 但无可再部署候选'
                                 f'({len(candidates)} 候选, deploy={deploy:,.0f} < min)')
            else:
                # 超配 → 减仓 (锁定名单优先)
                sell_codes = [c for c in list(portfolio.positions.keys())
                              if c in locked]
                over = -gap * nav
                if sell_codes and over > self.reinvest_min_amount:
                    amt_each = over / len(sell_codes)
                    self._derisk_list = [(c, amt_each) for c in sell_codes]
                    logger.info(f'[E3-敞口] 超配 {gap:+.1%} (act={e_act:.1%}, tgt={e_tgt:.1%}) '
                                f'→ 减仓 {len(sell_codes)} 只 (锁定名单) × {amt_each:,.0f}元')
        except Exception as e:
            logger.warning(f'[E3-敞口] 闭环计算异常: {e}')

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
            # [2026-09-02 修复] 记录 FLAT 触发时净值, 作为恢复回撤基线 (与 small_alpha191_test 同步)
            self._dd_flat_nav = self._prev_nav
            logger.warning(f'[方案A 回撤熔断] 回撤={dd:.2%} ≤ {c["dd_flat"]:.0%} → FLAT 不建仓 '
                           f'(基线 {self._dd_flat_nav:.0f})')
        else:
            # [E3 2026-09-09] 删除 HALF 态 (dd_half 分支已移除): 中度降仓职能由日频
            #   vol_scale 连续承担; 熔断仅保留 FLAT 极端尾部保险丝。
            #   FLAT→NORMAL 直接恢复 (原 FLAT→HALF→NORMAL 两步), 恢复后的实际建仓规模
            #   由 vol_scale 自然决定 (危机后 vol 高 → 自然低仓, 消除鞭打与长尾)。
            if self._dd_state == 2 and self._dd_above_days >= c['trend_confirm_days']:
                self._dd_state = 0
                self._dd_scale = 1.0
                # [2026-09-02 修复] 基线重置至 FLAT 触发净值 (原"峰值重置至当前净值"赦免历史亏损)
                self._dd_peak = self._dd_flat_nav if self._dd_flat_nav else self._prev_nav
                self._dd_above_days = 0
                logger.warning(f'[方案A 回撤熔断] 指数确认{c["trend_confirm_days"]}日 → FLAT→NORMAL '
                               f'(回撤基线重置至 FLAT 触发净值 {self._dd_peak:.0f}; '
                               f'实际规模由 vol_scale 决定)')

    def _is_rank_mode_active(self) -> bool:
        """[Fix-20260907 同步 alpha191] staged_rank（分位回归）模式判定：
        策略 label_mode 或模型 objective_mode 任一命中。

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
        # [Fix-20260907 同步 alpha191] rank(分位回归)模式: meta 度量为 Spearman IC(≈0~0.1),
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

    @staticmethod
    def _apply_vol_penalty_weights(codes, weights, sigmas, lam):
        """[Q5P′ 2026-09-24] 选股层波动率惩罚（纯函数，便于行为级断言 R-127）。

            w'_i ∝ w_i × (σ_ref / σ_i) ** lam,   σ_ref = 有效 σ 的**中位**
        返回 (new_weights_dict_or_None, info)。**None 表示不生效**（调用方保持原权重）。

        保证：
          ① Σw **守恒**（乘子归一 ⇒ 总敞口不变）
          ② lam <= 0 或 有效 σ < 2 只 ⇒ 返回 None（**零影响**）
          ③ 非有限 / 非正的 σ 视为无效，其乘子取 1.0
        """
        try:
            n = len(codes)
            w = np.asarray(weights, dtype=float)
            s = np.asarray(sigmas, dtype=float)
            if n == 0 or w.shape[0] != n or s.shape[0] != n or float(lam) <= 0.0:
                return None, {}
            ok = np.isfinite(s) & (s > 0)
            if int(ok.sum()) < 2:
                return None, {'reason': 'sigma_valid_lt2', 'n_valid': int(ok.sum())}
            s_ref = float(np.median(s[ok]))
            mult = np.ones(n, dtype=float)
            mult[ok] = (s_ref / s[ok]) ** float(lam)
            w_before = float(w.sum())
            w2 = w * mult
            s2 = float(w2.sum())
            if not np.isfinite(s2) or s2 <= 0.0:
                return None, {'reason': 'sum_nonpositive'}
            w2 = w2 * (w_before / s2)
            info = {
                'n': n, 'n_valid': int(ok.sum()), 'lam': float(lam),
                'sigma_ref': s_ref,
                'sigma_ratio_min': float(np.min(s[ok]) / s_ref),
                'sigma_ratio_max': float(np.max(s[ok]) / s_ref),
                'sum_w_before': w_before, 'sum_w_after': float(w2.sum()),
                'max_w_before': float(np.max(w)), 'max_w_after': float(np.max(w2)),
                'hhi_before': float(np.sum(w ** 2)), 'hhi_after': float(np.sum(w2 ** 2)),
            }
            return dict(zip(codes, w2)), info
        except Exception:
            return None, {'reason': 'exception'}

    @staticmethod
    def _apply_barra_risk_budget(codes, scores, exposures_df, factors,
                                 lam, exclude_quantile=0.0, shuffle_seed=None,
                                 min_coverage=0.80):
        """[Q5BAR 2026-10-08] BARRA **风险预算式降权/剔除**（纯函数，便于行为级断言 R-127）。

            score'_i = score_i × exp(−λ · max(0, R_i))
            R_i      = Σ_f z_{i,f}，z 用**稳健 z**（(x − median) / (1.4826·MAD)）

        可选剔除：R_i 最高 `exclude_quantile` 分位的股票 ⇒ score'_i = −inf（等价不入选）。

        返回 (new_scores_list_or_None, info)。**None 表示不生效**（调用方保持原分数）。

        零影响闭合（任一满足即返回 None）：
          · lam <= 0 或入参为空 / 长度不一致
          · 无暴露快照 / 请求因子列全部缺失
          · 有效覆盖占比 < min_coverage（默认 0.80）或有效样本 < 3

        设计要点（对照 `_apply_vol_penalty_weights` 范式）：
          · **只罚"风险高"的一侧**（max(0,R)）—— 不做"低风险加分"，避免改变分数另一端语义
          · **稳健 z**（中位/MAD）抗离群 —— 微盘池离群极多
          · **不做 scores 归一化**：本函数改的是"排序分数"而非"权重"，Top-N 只依赖序
          · shuffle_seed 非 None ⇒ 对 R 在**有效子集内做同分布随机置换**（R-146 sham 语义）
          · 代码归一：候选 `sh.600000`/`600000.SH` 与快照 `600000` 统一为 6 位数字
        """
        try:
            n = len(codes)
            sc = np.asarray(scores, dtype=float)
            if n == 0 or sc.shape[0] != n or float(lam) <= 0.0:
                return None, {'reason': 'lam_nonpositive_or_empty'}
            if exposures_df is None or len(exposures_df) == 0:
                return None, {'reason': 'no_exposure'}
            facs = [str(f) for f in (factors or [])]
            cols = [f + '_exp' for f in facs]
            cols = [c for c in cols if c in exposures_df.columns]
            if not cols:
                return None, {
                    'reason': 'factor_cols_missing', 'requested': facs,
                    'available': sorted(c for c in exposures_df.columns if c.endswith('_exp'))}
            c6 = pd.Series([str(c).split('.')[-1].zfill(6) for c in codes])
            ex = exposures_df.copy()
            ex['_c6'] = ex['code'].astype(str).str.split('.').str[-1].str.zfill(6)
            ex = ex.drop_duplicates('_c6').set_index('_c6')
            sub = ex.reindex(c6.values)
            cov = int(sub[cols[0]].notna().sum())
            coverage = cov / float(n)
            if coverage < float(min_coverage):
                return None, {'reason': 'coverage_below_min', 'coverage': coverage,
                              'n_valid': cov, 'n': n, 'min_coverage': float(min_coverage)}
            if cov < 3:
                return None, {'reason': 'n_valid_lt3', 'n_valid': cov}
            X = sub[cols].to_numpy(dtype=float)
            Z = np.zeros_like(X, dtype=float)
            n_used = 0
            for j in range(X.shape[1]):
                xj = X[:, j]
                m = np.isfinite(xj)
                if int(m.sum()) < 3:
                    continue
                med = float(np.median(xj[m]))
                mad = float(np.median(np.abs(xj[m] - med)))
                scale = 1.4826 * mad
                if not np.isfinite(scale) or scale <= 0.0:
                    mad2 = float(np.mean(np.abs(xj[m] - med)))   # 退化兜底
                    scale = mad2 if (np.isfinite(mad2) and mad2 > 0.0) else 0.0
                if scale <= 0.0:
                    continue
                Z[:, j] = np.where(m, (xj - med) / scale, 0.0)
                n_used += 1
            if n_used == 0:
                return None, {'reason': 'all_factor_cols_degenerate'}
            R = Z.sum(axis=1)
            if shuffle_seed is not None:                          # R-146 sham
                rng = np.random.default_rng(int(shuffle_seed))
                R = rng.permutation(R)
            penalty = np.exp(-float(lam) * np.maximum(0.0, R))
            sc2 = sc * penalty
            n_excl = 0
            eq = float(exclude_quantile or 0.0)
            if eq > 0.0:
                k = int(np.floor(eq * n))
                if k > 0:
                    thr = float(np.partition(R, n - k)[n - k])
                    mask = R >= thr
                    sc2 = np.where(mask, -np.inf, sc2)
                    n_excl = int(mask.sum())
            info = {
                'n': n, 'n_valid': cov, 'coverage': coverage,
                'factors_used': [c[:-4] for c in cols], 'lam': float(lam),
                'R_median': float(np.median(R)), 'R_p90': float(np.percentile(R, 90)),
                'R_max': float(np.max(R)),
                'penalty_min': float(np.min(penalty)),
                'penalty_median': float(np.median(penalty)),
                'penalty_max': float(np.max(penalty)),
                'shuffle_seed': (int(shuffle_seed) if shuffle_seed is not None else None),
                'exclude_quantile': eq, 'n_excluded': n_excl,
                'score_changed': int(np.sum(sc2 != sc)),
            }
            return list(sc2), info
        except Exception as e:                                    # noqa: BLE001
            return None, {'reason': 'exception', 'err': str(e)[:120]}

    def _load_barra_expo_pit(self, current_date):
        """[Q5BAR 2026-10-08] **只读**载入 PIT 暴露快照（严格 < current_date 的最大日期）。

        与旧机制的关键区别（R-151/R-156 快照门）：
          · 旧路径 `get_or_build_snapshot` 在缺伴随文件时会**按需重建并写盘** ⇒ 结果依赖
            "目录当前有哪些文件"，且重建会改变后续调仓日的快照选取口径。
          · 本方法 **绝不写盘、绝不重建** ⇒ 选取只由"截至当前已存在的文件"决定；缺失即返回
            None（调用方判为**不生效**，非静默降级）。
        缓存：同一快照日期只读一次（`self._barra_rb_expo_cache`）。
        选取语义与 `_barra_trace` 一致（严格 < 调仓日）。
        """
        import os as _os
        try:
            bdir = (self._barra_rb_cfg.get('snapshot_dir')
                    or (self.config.get('barra', {}) or {}).get('snapshot_dir')
                    or './data/barra_exposures')
            if not _os.path.isabs(bdir):
                bdir = _os.path.abspath(bdir)
            if not _os.path.isdir(bdir):
                return None
            dates = []
            for fn in _os.listdir(bdir):
                if fn.startswith('barra_expo_') and fn.endswith('.parquet'):
                    stem = fn[len('barra_expo_'):-len('.parquet')]
                    try:
                        dates.append(datetime.strptime(stem, '%Y%m%d'))
                    except ValueError:
                        continue
            valid = [d for d in dates if d < pd.Timestamp(current_date)]
            if not valid:
                return None
            sel = max(valid).strftime('%Y%m%d')
            if sel in self._barra_rb_expo_cache:
                return self._barra_rb_expo_cache[sel]
            df = pd.read_parquet(_os.path.join(bdir, 'barra_expo_%s.parquet' % sel))
            self._barra_rb_expo_cache[sel] = df
            self._barra_rb_used.add(sel)
            return df
        except Exception as e:                                    # noqa: BLE001
            logger.warning('[Q5BAR] 读取 BARRA 暴露快照失败（视为不生效）: %s', e)
            return None

    def _compute_entry_vol(self, code, date):
        """EWMA daily vol (λ=0.94, seed=60d) — 与持仓端 volatile 缓存一致"""
        if not self.meta_labeler: return 0.02
        vol_method = self.config.get('meta_labeling', {}).get('vol_method', 'rolling')
        lookback = max(self.meta_labeler.daily_vol_window, self.meta_labeler.vol_ewma_seed_days)
        hist = self.data_provider.get_daily_data(
            code, date - timedelta(days=lookback * 2), date, reference_date=self.reference_date)
        # [FIX-B-20260915] 排除建仓日 D 自身: σ 是屏障宽度, 必须在【D-1 收盘(决策时点)】可知。
        #   原实现端点 = D ⇒ ①相对 D 日开盘建仓构成 1 日前视 (D 日 close 在建仓时不可知)
        #   ②与标签端 labels._batch_daily_volatility (窗口截到事件日 D-1) 口径不对称
        #   ⇒ 训练屏障与执行屏障不同源。实测影响: PT1 屏障位移 mean +0.077% 价 /
        #   p90 +0.563% / max +11.4%; |位移|>0.5% 的建仓占 13.5%。
        _dcol = 'trade_date' if 'trade_date' in hist.columns else (
            'date' if 'date' in hist.columns else None)
        if _dcol is not None:
            hist = hist[pd.to_datetime(hist[_dcol]).dt.normalize()
                        < pd.Timestamp(date).normalize()]
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
        [2026-09-10] PT 判定改为当根 bar high 触达 (与 SL 当根 low 对称; 见循环内注释)。
        [2026-09-10 E6] 事件驱动逐 bar 生效: pl_sl 于每根 bar 起点刷新 (上一根 bar 的
        PT 事件自本根起抬高 SL; SL 事件在检查内即时置 pt_after_sl, 自下一根 bar 起下移 PT);
        同 bar 内保持保守 (本根的全部检查使用"截至上一根 bar"的状态)。
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
            # [FIX-A-20260915] 零值 bar 守卫: raw_minute/5min 对停牌日会填 high=low=0 的
            #   占位 bar (实测 68/400 只样本存在, 合计 669 个零值交易日)。若不排除:
            #   cur_s = (0/entry-1)/vol ≈ -30σ ⇒ 幻影硬止损, 并以阈值价"成交"不可成交的股票
            #   (实测 Q2c 胜出包 7207 笔订单中有 5 笔落在停牌日)。日线数据无此问题
            #   (6209 条停牌行中 0 条 high<=0), 故只需守卫分钟路径。
            hi_ok = (not np.isnan(hi)) and hi > 0
            lo_ok = (not np.isnan(lo)) and lo > 0
            if hi_ok: cum_u = max(cum_u, hi / entry - 1)
            if lo_ok: cum_d = min(cum_d, lo / entry - 1)
            # [2026-09-10 修复] PT 判定用【当根 bar 的 high】(不再用累计峰值 cum_u), 与 SL 同口径:
            #   原累计峰值口径在 SL后PT下移换档后, 会把换档前的历史高点误判为新目标的触达
            #   → 以非可执行价格成交 (实证 sz.301120: 4/20 “止盈”@20.50 而当日真实 high=18.55)。
            #   当根口径与累计口径在所有“阈值固定”场景逐位等价 (首触即发), 仅修正换档场景。
            up_s = (hi / entry - 1) / vol if hi_ok else float('-inf')
            # SL用当前bar的low收益率（不用累计最大回撤）
            # 原因: profit_lock_sl为正数时, dn_s(累计回撤,常为负) <= eff_sl(正数)恒为True
            cur_s = (lo / entry - 1) / vol if lo_ok and entry > 0 else float('inf')

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
                eff_pt_2 = self.PT_AFTER_SL[2]; eff_pt_3 = self.PT_AFTER_SL[3]
            else:
                eff_pt_2 = self.STAGE_CONFIG[2]['pt']; eff_pt_3 = self.STAGE_CONFIG[3]['pt']

            # S1 — [O3-20260904] 同bar PT/SL歧义→SL优先(保守, 与标签端同步);
            #        触发后立即设 flag，防止同一日内后续 bar 重复触发
            if not tracker.stage1_done:
                if cur_s <= eff_sl_1:
                    sl_price = entry * (1 + eff_sl_1 * vol)  # 阈值价而非 bar low
                    events.append((1, False, sl_price, str(bar.get('trade_time', ''))))
                    tracker.stage1_done = True
                    tracker.pt_after_sl = True  # [E6] SL事件自下一根bar起下移PT (applier 冗余置位保留)
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
        [2026-09-10 E6] 单 bar/日粒度: 逐 bar 生效语义与分钟版同义 (下一根 bar = 次日);
        SL 事件同样在检查内即时置 pt_after_sl (applier 冗余置位保留)。
        """
        entry = tracker.entry_adj; vol = tracker.daily_vol
        if vol <= 0: return []
        pl_sl = tracker.profit_lock_sl()  # 利润锁定SL
        try:
            dd = self.data_provider.get_daily_data(code, date, date, reference_date=self.reference_date)
            if dd.empty: return []
            h = dd['high'].iloc[0]; l = dd['low'].iloc[0]
            # [2026-09-10 修复] PT 判定用当日 high (同 minute 版: 不再用累计峰值)
            up_s = (h / entry - 1) / vol
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
                eff_pt_2 = self.PT_AFTER_SL[2]; eff_pt_3 = self.PT_AFTER_SL[3]
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
                    tracker.pt_after_sl = True  # [E6] SL触发→PT下移
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
                    tracker.pt_after_sl = True  # [E6] SL触发→PT下移
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
    def _build_pool_index_return_matrix(self, lookback, current_date):
        """[E2 方案B] 池指数收益率矩阵 (T,1) — 用中证2000 收益代理组合波动。

        绕过个股矩阵的日期对齐/新股零填充问题 (R1 的替代路径);
        缺点: 指数 ≠ 实际持仓组合 (R3 代理偏差, v1 可接受须标注)。
        数据: data/index_daily/{risk_index_code}.parquet, ≤T-1 严格无未来。
        """
        try:
            idx_code = getattr(self, 'risk_index_code', 'sh932000')
            path = os.path.join('data', 'index_daily', f'{idx_code}.parquet')
            if not os.path.exists(path):
                logger.warning(f'[GARCH-B] 池指数文件不存在: {path}')
                return None
            df = pd.read_parquet(path)
            dcol = 'trade_date' if 'trade_date' in df.columns else 'date'
            df[dcol] = pd.to_datetime(df[dcol])
            df = df.set_index(dcol).sort_index()
            px = df['close']
            px = px[px.index < pd.Timestamp(current_date)]  # 严格 ≤T-1
            if len(px) < lookback:
                logger.debug(f'[GARCH-B] 指数历史不足: {len(px)} < {lookback}')
                return None
            rets = px.pct_change().dropna().iloc[-lookback:]
            return rets.values.reshape(-1, 1)
        except Exception as e:
            logger.warning(f'[GARCH-B] 池指数矩阵构建异常: {type(e).__name__}: {e}')
            return None

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
            # [AUDIT-FIX 2026-09-14 D-179 / S-49] T-1 截断（消除前视偏差）:
            #   原实现取 trade_dates_list[-lookback:]，会包含 current_date 当日,
            #   经 get_daily_data(闭区间) 取到当日收盘 ⇒ close.pct_change() 末项 = 当日收益；
            #   而建仓使用当日开盘价 (_get_open_price) ⇒ 用当日收盘定当日开盘的仓位 = look-ahead。
            #   姊妹函数 _build_pool_index_return_matrix 已有 `px.index < current_date` 保护,
            #   此处与之对齐（严格 ≤T-1）。
            _prev_dates = [d for d in trade_dates_list
                           if pd.Timestamp(d) < pd.Timestamp(current_date)]
            if len(_prev_dates) < lookback:
                logger.debug(f"[GARCH] T-1 交易日历不足: {len(_prev_dates)} < {lookback}")
                return None
            hist_dates = _prev_dates[-lookback:]
            start_dt = hist_dates[0]
            end_dt = hist_dates[-1]

            ret_dict = {}
            # [E2-R1 2026-09-09] 修复: ret_dict 存带日期的收益 Series (原存裸数组无法对齐)
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
                    rets = daily['ret'].dropna()
                    if len(rets) < lookback // 2:
                        fail_reasons[code] = f'valid_ret={len(rets)} < {lookback // 2}'
                        continue
                    ret_dict[code] = rets
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
            # [E2-R1 2026-09-09] 修复: 按交易日历 hist_dates 对齐, 缺失日 NaN
            #   (原实现 np.zeros 零填充 + 每股尾部对齐 → 新股/停牌股波动系统性低估)
            ret_matrix = np.full((lookback, n_success), np.nan)
            hist_idx = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in hist_dates])
            for i, (code, ret_series) in enumerate(ret_dict.items()):
                aligned = ret_series.reindex(hist_idx)
                ret_matrix[:, i] = aligned.values

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
Alpha191Strategy = SmallAlpha191Strategy
