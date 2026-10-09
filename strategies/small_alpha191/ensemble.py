# -*- coding: utf-8 -*-
"""
重构版集成学习模块 - XGBoost 元标注模型（v3.0）

修复历史：
- 【2026-06-23 v3.0】生产模型时序分割+early stopping+OOS校准+验证集AUC门控
- 【2026-06-23 v3.0】旧版FI实现删除，统一MDA为compute_mda_oos（种子固定）
- 【2026-06-22 v2.0】Purged K-Fold CV + 跨折投票(IC+MDA) + 特征聚类
- 【2026-06-06 v1.1】移除PCA-Kendall，改为MDI-MDA加权Kendall一致性检验
- 【2026-05-18 v1.0】MDI用XGBoost gain，MDA用XGBBoosterWrapper包装器

工作流（v3.0）：
1. train() 调用 → 预处理（nan去重 + 低方差过滤）
2. _purged_kfold_cv()：5折 Purged K-Fold CV（embargo=21）
   每折内：IC+PSI特征选择 → XGBoost训练(early stopping) → 验证集AUC
   每折内：MDI(gain) + OOS MDA(compute_mda_oos, seed固定)
3. aggregate_fold_votes()：跨折投票（IC共识预滤 → MDA投票准入 → 综合评分 → Top-K）
4. cluster_features()：特征聚类去冗余（当前关闭）
5. 生产模型训练：时序末尾train/val分割 → 多种子XGBoost(early stopping) → val AUC门控
6. Isotonic校准：使用生产模型验证集OOS概率（非in-sample）
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import spearmanr, binom
import warnings
import logging
import gc
import os
import time
import pickle
import hashlib
from typing import Dict, List, Optional, Tuple, Union

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logging.warning("xgboost未安装，无法使用XGBoost")

# [v3.5] LightGBM 异质集成 (对齐V10)
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    lgb = None

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)






def compute_sample_weights(train_data, calendar, max_hold_days,
                           exit_type=None, pt_sl_multiplier=2.0,
                           pt_multiplier=None, sl_multiplier=None,
                           sham_seed=None):
    """
    为每个训练样本计算基于平均唯一性的样本权重（高效版）。
    算法来源: De Prado《Advances in Financial Machine Learning》第7章
    【v2 性能优化】
    原实现对每个样本逐一构建指示矩阵行，O(N*T) 时空复杂度。
    当 N 较大时，指示矩阵占用大量内存，逐行 Pandas 赋值极慢。
    优化核心：同一 trade_date 的所有样本共享相同的 [事件日, 退出日] 窗口，
    因此其平均唯一性完全相同。只需按 trade_date 分组，O(U*T) 复杂度，
    U=唯一交易日数（通常远小于N），加速数百至数千倍。
    数学等价性证明：
    设样本 i, j 的事件日相同（trade_date_i = trade_date_j），
    则其窗口均为 [s, e]，indM[t,i] = indM[t,j] ∀t，
    故 uniqueness[t,i] = uniqueness[t,j] = 1/concurrency[t]，
    故 avg_u[i] = avg_u[j]。按 trade_date 分组计算结果完全一致。

    【v3.7 标签质量加权】
    在时序窗口权重基础上乘以标签质量因子：
    - 触及 PT/SL 屏障的强信号样本：× pt_sl_multiplier（默认2.0）
    - 到期退出的弱信号样本：× 1.0（保留，不丢弃）

    参数：
        train_data: DataFrame，必须包含 'trade_date' 列（事件日）。
        calendar:   交易日列表（datetime 类型），已排序。
        max_hold_days: 最大持有天数（垂直屏障）。
        exit_type:  可选，Series/array，取值 'profit_take'/'stop_loss'/'expire'，
                    长度与 train_data 行数一致。
        pt_sl_multiplier: PT/SL 样本的额外权重倍数（默认 2.0）。
    返回：
        np.array，长度与 train_data 行数相同，表示每个样本的训练权重。
    """
    if train_data.empty:
        return np.ones(len(train_data))
    N = len(train_data)
    # 1. 按 trade_date 分组，统计每个事件日的样本数
    event_counts = train_data.groupby(train_data['trade_date']).size()
    unique_dates = event_counts.index
    # 将 calendar 转为 DatetimeIndex，用 searchsorted 快速定位
    cal_ts = pd.DatetimeIndex(sorted(calendar))
    T = len(cal_ts)
    # 2. 对每个唯一事件日，计算其在日历中的窗口 [start_idx, end_idx]
    date_windows = {}
    for event_date in unique_dates:
        event_ts = pd.Timestamp(event_date)
        start_idx = cal_ts.searchsorted(event_ts)
        if start_idx >= T:
            continue  # 事件日在日历范围外，跳过
        end_idx = min(start_idx + max_hold_days, T - 1)
        date_windows[event_date] = (start_idx, end_idx)
    # 3. 计算每天的并发样本数：concurrency[t] = 当天被多少样本覆盖
    #    用 numpy 数组代替 DataFrame，仅遍历唯一事件日（U次，非N次）
    concurrency = np.zeros(T, dtype=np.float64)
    for event_date, (s, e) in date_windows.items():
        n_samples = event_counts[event_date]
        concurrency[s:e + 1] += n_samples
    # 避免除零
    concurrency[concurrency == 0] = 1.0
    # 4. 对每个唯一事件日，计算该组样本的平均唯一性
    #    严格等价于原方法：avg_u[i] = (1/T) * Σ_{t∈window_i}(1/concurrency[t])
    #    原方法 uniqueness.mean(axis=0) 对所有T行求均值（窗口外=0参与除法），
    #    等价于 sum(窗口内1/concurrency) / T，而非 sum/|window|。
    date_avg_u = {}
    for event_date, (s, e) in date_windows.items():
        if s <= e:
            # 窗口内每天的 unique 比例求和，再除以总日历天数 T
            # 这和原方法 indM.div(concurrency).mean(axis=0) 完全等价
            date_avg_u[event_date] = np.sum(1.0 / concurrency[s:e + 1]) / T
        else:
            date_avg_u[event_date] = 1.0 / T
    # 5. 将权重分配到每个样本（同事件日的样本权重相同）
    weights = np.array([date_avg_u.get(td, 1.0) for td in train_data['trade_date']])
    # 6. 归一化到均值为1
    w_mean = weights.mean()
    if w_mean > 0:
        weights = weights / w_mean

    # 7. [v3.7] 标签质量加权：PT/SL 强信号样本 × pt_sl_multiplier
    #    ★[波4-波C′] 拆分两子轴：pt_multiplier / sl_multiplier（D-296）
    #    ★零影响闭合：两者**同时为 None** ⇒ 严格走原合并掩码分支（与 G15 逐位等价）
    _use_split = (pt_multiplier is not None) or (sl_multiplier is not None)
    if exit_type is not None and _use_split:
        exit_arr = np.asarray(exit_type)
        _pt_m = 1.0 if pt_multiplier is None else float(pt_multiplier)
        _sl_m = 1.0 if sl_multiplier is None else float(sl_multiplier)
        weights = weights.copy()  # 避免修改传入数组
        _m_pt = (exit_arr == 'profit_take')
        weights[_m_pt] = weights[_m_pt] * _pt_m
        _m_sl = (exit_arr == 'stop_loss')
        weights[_m_sl] = weights[_m_sl] * _sl_m
        # 重新归一化到均值为1
        w_mean2 = weights.mean()
        if w_mean2 > 0:
            weights = weights / w_mean2
    elif exit_type is not None and pt_sl_multiplier > 1.0:
        exit_arr = np.asarray(exit_type)
        pt_sl_mask = (exit_arr == 'profit_take') | (exit_arr == 'stop_loss')
        weights = weights.copy()  # 避免修改传入数组
        weights[pt_sl_mask] = weights[pt_sl_mask] * pt_sl_multiplier
        # 重新归一化到均值为1
        w_mean2 = weights.mean()
        if w_mean2 > 0:
            weights = weights / w_mean2

    # ★[波4-波C′ · R-146] sham 对照：对**最终权重向量**按固定 seed 随机置换到样本。
    #   权重的**多重集逐位不变** ⇒ 均值/ESS 完全一致，仅破坏"样本↔权重"对应关系
    #   ⇒ 用于区分"机制效应"与"仅改变有效样本量的机械效应"。
    if sham_seed is not None:
        _rng = np.random.default_rng(int(sham_seed))
        weights = weights[_rng.permutation(len(weights))]

    return weights


# 【Phase2优化】样本权重跨周期缓存
_sample_weight_cache = {}  # {cache_key: (fingerprint, unique_dates_tuple, weights_map)}
                           # [O7-20260904] 引入指纹: 权重(唯一性)取决于样本日期全集,
                           #   仅 key 相同但窗口组成不同(增删日期)时旧缓存不可复用
def _sw_window_fingerprint(unique_dates_in_data):
    """窗口指纹: (起始日, 结束日, 日期数, 有序日期哈希)。唯一性权重依赖全体样本日期,
    集合任一变动 → 指纹变化 → 缓存失效, 杜绝"同 key 异窗口"错配。"""
    _srt = tuple(sorted(unique_dates_in_data))
    _t0 = str(_srt[0]) if _srt else ''
    _t1 = str(_srt[-1]) if _srt else ''
    _h = hash(_srt) if _srt else 0
    return (_t0, _t1, len(_srt), _h)

def compute_sample_weights_cached(train_data, calendar, max_hold_days,
                                   cache_key=None, exit_type=None,
                                   pt_sl_multiplier=2.0,
                                   pt_multiplier=None, sl_multiplier=None,
                                   sham_seed=None):
    """
    带跨周期缓存的样本权重计算。

    与 compute_sample_weights 结果完全一致，但当训练窗口滑动时，
    大部分 trade_date 的权重已在上周期计算过，直接复用。

    缓存策略：以 cache_key 标识训练窗口，存储 unique_dates 的权重映射。
    下一周期查询时，命中过的 event_date 直接取缓存值。

    参数：
        train_data: DataFrame，必须包含 'trade_date' 列
        calendar: 交易日列表
        max_hold_days: 最大持有天数
        cache_key: 可选缓存键（如训练窗口结束日期），用于区分不同周期
        exit_type:  可选，Series/array，取值 'profit_take'/'stop_loss'/'expire'
        pt_sl_multiplier: PT/SL 样本的额外权重倍数

    返回：
        np.array, 与 compute_sample_weights 完全一致
    """
    if train_data.empty:
        return np.ones(len(train_data))

    # 收集需要计算权重的唯一日期
    unique_dates_in_data = set(train_data['trade_date'])
    fpr = _sw_window_fingerprint(unique_dates_in_data)
    # ★[波4-波C′ · D-303 同批必修] 权重参数指纹：倍数/种子必须进入缓存身份，
    #   否则**同进程内换倍数会静默复用旧 wmap**（当前每臂恒定 ⇒ 未触发，但必须修）。
    _wparams = (pt_sl_multiplier, pt_multiplier, sl_multiplier, sham_seed)

    # 检查缓存命中
    cached_map = None
    if cache_key is not None and cache_key in _sample_weight_cache:
        cached_fpr, cached_wparams, cached_unique, cached_wmap = _sample_weight_cache[cache_key]
        # [O7-20260904] 指纹一致才可复用 (窗口日期集合完全相同); 否则重算
        # ★[波4-波C′] 权重参数也须一致（口径变更 ⇒ 缓存失效）
        if cached_fpr == fpr and cached_wparams == _wparams:
            cached_map = cached_wmap
            logger.debug(
                f"[样本权重缓存] 命中! {len(unique_dates_in_data)} 个日期全部复用 (指纹一致)"
            )
        else:
            logger.debug("[样本权重缓存] 指纹/参数不一致 (窗口或权重口径变化), 重新计算")

    # 如果缓存未命中，调用原始函数计算
    if cached_map is None:
        if cache_key is not None:
            logger.debug("[样本权重缓存] 未命中，重新计算")
        weights = compute_sample_weights(train_data, calendar, max_hold_days,
                                          exit_type=exit_type,
                                          pt_sl_multiplier=pt_sl_multiplier,
                                          pt_multiplier=pt_multiplier,
                                          sl_multiplier=sl_multiplier,
                                          sham_seed=sham_seed)
        # 缓存结果
        if cache_key is not None:
            # ★[波4-波C′] wmap 键改为 (trade_date, exit_type)：
            #   旧实现 wmap[trade_date]=w 是"后写覆盖"，而 PT/SL/expire 三类样本权重
            #   **在同一日期内并不相同** ⇒ 命中路径无法忠实还原（T5 暴露的既存隐性缺陷）。
            #   按 (日期, 退出类型) 建键后，命中路径与直算路径**逐位一致**。
            _et_arr = None if exit_type is None else np.asarray(exit_type)
            wmap = {}
            for _i, (td, w) in enumerate(zip(train_data['trade_date'], weights)):
                _et = None if _et_arr is None else _et_arr[_i]
                wmap[(td, _et)] = w
            _sample_weight_cache[cache_key] = (fpr, _wparams,
                                               list(unique_dates_in_data), wmap)
            # [O7] 防无界增长: 仅保留最近 MAX_SW_CACHE 个窗口
            _MAX_SW_CACHE = 8
            while len(_sample_weight_cache) > _MAX_SW_CACHE:
                _sample_weight_cache.pop(next(iter(_sample_weight_cache)))
        return weights

    # 从缓存映射权重（★键为 (trade_date, exit_type)，见上方 wmap 说明）
    _et_arr_hit = None if exit_type is None else np.asarray(exit_type)
    weights = np.array([
        cached_map.get((td, None if _et_arr_hit is None else _et_arr_hit[_i]), 1.0)
        for _i, td in enumerate(train_data['trade_date'])
    ])
    w_mean = weights.mean()
    if w_mean > 0:
        weights = weights / w_mean
    return weights


def calc_psi(feat_tr, feat_va, fill_value=None):
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


def sanitize_features(df: pd.DataFrame, nan_thold: float, max_corr: float) -> List[str]:
    sorted_cols = sorted(df.columns.tolist())
    df = df[sorted_cols].copy()
    nan_rate = df.isna().mean()
    keep_cols = nan_rate[nan_rate < nan_thold].index.tolist()
    if not keep_cols:
        return []
    df = df[keep_cols]
    n_feat = len(keep_cols)
    if n_feat <= 1:
        return keep_cols
    import numpy as np
    arr = df.fillna(0).values.astype(np.float64)
    corr_mat = np.corrcoef(arr, rowvar=False)
    drop_flags = [False] * n_feat
    for i in range(n_feat):
        if drop_flags[i]:
            continue
        for j in range(i + 1, n_feat):
            if drop_flags[j]:
                continue
            v = corr_mat[i, j]
            if not np.isnan(v) and abs(v) > max_corr:
                drop_flags[j] = True
    final_cols = [col for col, drop in zip(keep_cols, drop_flags) if not drop]
    final_cols = sorted(final_cols)
    logger.info(
        f"特征预处理：原始特征数 {len(sorted_cols)}，剔除后保留 {len(final_cols)} 个（缺失率 > {nan_thold} 或 |corr| > {max_corr}）")
    return final_cols


def align_sample_weight_to_sort(df_train, sample_weight):
    """[2026-08-18] 排序前记录原始行序, 排序后同步重排 sample_weight。

    BUG 修复: train() 原实现 sort_values(['trade_date','ts_code']).reset_index() 后
    sample_weight 未重排 → 折内/生产/HP-Search 用位置索引取权重
    (sample_weight[train_set.index.values]) → 权重与样本错位 (张冠李戴)。

    返回 (df_train, sample_weight_new):
      - sample_weight 为 None → (df_train, None) 原样
      - 长度不匹配 → 警告并原样返回 (防御)
      - 否则 df_train 排序 (附 _orig_pos 列) + sample_weight 重排到新行序
    """
    if sample_weight is None:
        return df_train, None
    sw_orig = np.asarray(sample_weight)
    if len(sw_orig) != len(df_train):
        logger.warning("[sample_weight] 长度不匹配 (%d vs %d), 跳过重排", len(sw_orig), len(df_train))
        return df_train, sample_weight
    df_train['_orig_pos'] = np.arange(len(df_train))
    df_train = df_train.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
    sample_weight = sw_orig[df_train['_orig_pos'].values].copy()
    logger.info("[sample_weight] 已按 (trade_date, ts_code) 排序重排 (%d 行, 与样本对齐)",
                len(sample_weight))
    return df_train, sample_weight


def select_stable_features(df_train, df_val, feats, config, target_col='label', current_date=None, sample_weight=None, skip_preprocess=False, skip_ic=False):
    logger.debug("=" * 60)
    logger.debug("[FS:select_stable] IC+PSI feature screening (RankIC)")
    logger.debug("=" * 60)
    feats = sorted(feats)
    if target_col not in df_train.columns:
        raise ValueError(f"训练集必须包含 '{target_col}' 列")
    y_train = df_train[target_col].values
    # 新增：低方差过滤（仅移除全常量列，threshold=0）
    train_feat_df = df_train[feats].replace([np.inf, -np.inf], np.nan)
    selector = VarianceThreshold(threshold=0.0)  # scikit-learn默认，仅移除方差=0的列
    selector.fit(train_feat_df.fillna(0))
    low_variance_cols = train_feat_df.columns[selector.get_support()]
    feats = low_variance_cols.tolist()
    logger.info(f"低方差过滤后保留特征数：{len(feats)}")
    # [R3-1 A3] v10_pred_rank 作为先验特征（来自 V10 模型），PSI 豁免但不跳过 IC 筛选
    PSI_EXEMPT = {'v10_pred_rank'}
    if not skip_preprocess:
        logger.info("【步骤1】特征预处理（缺失率过滤 & 高相关去重）")
        exempt_feats = [f for f in feats if f in PSI_EXEMPT]
        non_exempt_feats = [f for f in feats if f not in PSI_EXEMPT]
        kept_feats = sanitize_features(train_feat_df[non_exempt_feats], config.NAN_THRESHOLD, config.MAX_CORR)
        kept_feats = sorted(kept_feats + exempt_feats)
        logger.info(f"预处理后保留特征数：{len(kept_feats)} (含 PSI豁免: {exempt_feats})")
        feats = kept_feats
    tr = df_train[feats].replace([np.inf, -np.inf], np.nan)
    va = df_val[feats].replace([np.inf, -np.inf], np.nan)
    logger.info(f"训练集样本: {len(tr)}条, 验证集样本: {len(va)}条")
    logger.info(f"候选特征数: {len(feats)}")

    # Phase 1全局预处理时跳过IC筛选，仅做清理
    if skip_ic:
        logger.info(f"[Phase1] 跳过IC/PSI筛选，保留全部 {len(feats)} 特征进入CV")
        # 只打一次IC统计日志供参考
        ics = []
        for f in feats[:20]:  # 采样前20个做快速诊断
            try:
                tr_c = tr[f].fillna(tr[f].median(skipna=True) if not pd.isna(tr[f].median(skipna=True)) else 0).values
                ic = spearmanr(tr_c, y_train)[0]
                if not np.isnan(ic): ics.append(abs(ic))
            except Exception:
                pass
        if ics:
            logger.info(f"  IC抽样(前20): min={min(ics):.4f} max={max(ics):.4f} mean={np.mean(ics):.4f}")
        return feats, {}, 0.0, {}

    # === 性能优化：向量化 IC 计算 ===
    # 第一遍：计算 PSI 和 median，筛选通过 PSI 的特征
    psi_info = {}  # {feat: (psi, tr_median)}
    psi_pass_feats = []
    for f in feats:
        try:
            tr_col = tr[f]
            va_col = va[f]
            tr_median = tr_col.median(skipna=True)
            if pd.isna(tr_median):
                tr_median = 0.0
            if f in PSI_EXEMPT:
                psi = calc_psi(tr_col, va_col, fill_value=tr_median)
                logger.info(f"[R3-1 A3] {f} PSI={psi:.4f} (先验特征豁免PSI过滤，但需通过IC筛选)")
                psi_info[f] = (psi, tr_median)
                psi_pass_feats.append(f)
            else:
                psi = calc_psi(tr_col, va_col, fill_value=tr_median)
                if psi > config.PSI_THRESHOLD:
                    continue
                psi_info[f] = (psi, tr_median)
                psi_pass_feats.append(f)
        except Exception:
            continue

    if not psi_pass_feats:
        logger.warning("没有特征通过PSI检查，返回空特征列表")
        return [], {}, 0.0, {}

    # 第二遍：批量 fillna + rank + 向量化 Pearson（Spearman = Pearson on ranks）
    # 一次性填充所有通过特征
    tr_filled_df = pd.DataFrame(index=tr.index, columns=psi_pass_feats, dtype=np.float64)
    for f in psi_pass_feats:
        _, med = psi_info[f]
        tr_filled_df[f] = tr[f].fillna(med)

    # [B2-20260905] P2-2: IC 口径切换 (config.IC_MODE, 默认 'daily'):
    #   'daily'  — 逐日截面 Spearman 后取均值 (业界标准 RankIC, 消除跨日混排混入的"区分日期"成分,
    #              与 MDA 口径一致; 每日 <30 样本的日子剔除防小样本噪声)
    #   'pooled' — 旧行为: 全面板跨 ~220 天整体 rank 的 Spearman
    ic_mode = getattr(config, 'IC_MODE', 'daily')
    y_s = pd.Series(y_train, index=tr_filled_df.index, dtype=np.float64)

    if ic_mode == 'daily' and 'trade_date' in df_train.columns:
        td = df_train['trade_date']
        # 逐日截面 rank (groupby.rank 一次性完成)
        tr_ranked_d = tr_filled_df.groupby(td).rank()
        y_rank_d = y_s.groupby(td).rank()
        # 逐日 Pearson on ranks (向量化协方差公式)
        mx = tr_ranked_d.groupby(td).transform('mean')
        my = y_rank_d.groupby(td).transform('mean')
        xc = tr_ranked_d - mx
        yc = y_rank_d - my
        cov_d = xc.mul(yc, axis=0).groupby(td).sum()          # [n_days, n_feats]
        vx_d = (xc ** 2).groupby(td).sum()
        vy_d = (yc ** 2).groupby(td).sum()
        cnt_d = tr_filled_df.groupby(td).size()
        ic_daily = cov_d / (np.sqrt(vx_d.mul(vy_d, axis=0)) + 1e-12)
        ic_daily = ic_daily[cnt_d >= 30]                       # [B2] 小样本日守卫
        if len(ic_daily) == 0:
            logger.warning("[B2] 逐日IC: 无 ≥30 样本的交易日, 回退 pooled 口径")
            ic_mode = 'pooled'
        else:
            ic_vector = ic_daily.mean(axis=0).values
            ic_vector = np.where(np.isnan(ic_vector), 0.0, ic_vector)
            logger.info(f"[B2] 逐日截面RankIC: {len(ic_daily)} 个有效交易日 (≥30样本守卫), "
                        f"日均样本={cnt_d[cnt_d >= 30].mean():.0f}")

    if ic_mode != 'daily' or 'trade_date' not in df_train.columns:
        if ic_mode == 'daily':
            logger.warning("[B2] df_train 无 trade_date 列, 回退 pooled 口径")
        # 一次性 rank 所有特征（pandas C 层实现，比逐特征 spearmanr 快 10x+）
        tr_ranked = tr_filled_df.rank()
        y_rank = y_s.rank()

        # 向量化 Pearson on ranks
        tr_arr = tr_ranked.values  # [n_samples, n_feats]
        y_arr = y_rank.values      # [n_samples]
        tr_mean = tr_arr.mean(axis=0)
        y_mean = y_arr.mean()
        tr_centered = tr_arr - tr_mean
        y_centered = y_arr - y_mean
        numerator = tr_centered.T @ y_centered       # [n_feats]
        tr_ss = np.sqrt((tr_centered ** 2).sum(axis=0))
        y_ss = np.sqrt((y_centered ** 2).sum())
        ic_vector = numerator / (tr_ss * y_ss + 1e-12)
        # 处理 NaN（常数列 → tr_ss=0 → ic=0）
        ic_vector = np.where(np.isnan(ic_vector), 0.0, ic_vector)

    # 组装结果（与原代码格式完全一致）
    all_feat_scores = []
    for i, f in enumerate(psi_pass_feats):
        psi, _ = psi_info[f]
        ic_train = float(ic_vector[i])
        ic_abs = abs(ic_train)
        composite_score = ic_abs * (1 - min(psi, 1))
        all_feat_scores.append({
            'feature': f,
            'psi': psi,
            'ic': ic_train,
            'ic_abs': ic_abs,
            'composite_score': composite_score,
        })
    if not all_feat_scores:
        logger.warning("没有特征通过基础检查，返回空特征列表")
        return [], {}, 0.0, {}
    max_ic = max([fs['ic_abs'] for fs in all_feat_scores])
    logger.info(f"特征RankIC绝对值范围: {min([fs['ic_abs'] for fs in all_feat_scores]):.4f} - {max_ic:.4f}")
    good_feats = [fs['feature'] for fs in all_feat_scores if fs['ic_abs'] >= config.MIN_FEAT_IC]
    logger.info(f"RankIC≥{config.MIN_FEAT_IC} 通过: {len(good_feats)}个特征")
    MIN_VALID_FEATURES = 10
    if len(good_feats) < MIN_VALID_FEATURES:
        logger.warning(
            f"有效特征数 {len(good_feats)} 小于硬性最低要求 {MIN_VALID_FEATURES}，特征筛选失败，跳过当前预测目标")
        return [], {}, 0.0, {}
    if len(good_feats) < config.MIN_FEATURES and len(all_feat_scores) > 0:
        feat_scores_sorted = sorted(all_feat_scores, key=lambda x: x['ic_abs'], reverse=True)
        additional = [fs['feature'] for fs in feat_scores_sorted[:config.MIN_FEATURES]
                      if fs['feature'] not in good_feats]
        good_feats.extend(additional)
        logger.warning(f"特征不足 MIN_FEATURES({config.MIN_FEATURES})，补充 {len(additional)} 个高RankIC特征")
    if len(good_feats) > config.MAX_FEATURES:
        feat_score_dict = {fs['feature']: fs['composite_score'] for fs in all_feat_scores}
        good_feats_sorted = sorted(good_feats, key=lambda x: feat_score_dict.get(x, 0), reverse=True)
        good_feats = good_feats_sorted[:config.MAX_FEATURES]
        logger.info(f"特征过多，保留前 {config.MAX_FEATURES} 个")
    good_feats = sorted(good_feats)
    logger.info(f"特征筛选结果: {len(good_feats)}/{len(feats)} 通过")
    if all_feat_scores:
        passed_scores = [fs for fs in all_feat_scores if fs['feature'] in good_feats]
        avg_ic = np.mean([fs['ic_abs'] for fs in passed_scores]) if passed_scores else 0
        avg_psi = np.mean([fs['psi'] for fs in passed_scores]) if passed_scores else 0
        logger.info(f"  平均|RankIC|: {avg_ic:.4f}, 平均PSI: {avg_psi:.4f}")
        logger.info(f"  最高|RankIC|: {max_ic:.4f}")
    feat_scores = {fs['feature']: fs for fs in all_feat_scores}
    # ===== 特征重要性分析 =====
    # 说明：此时主模型尚未训练，_feature_importance_analysis 内部会自行训练
    # 一个临时 fi_model 供 MDI/MDA 共用，训练后即丢弃
    # [R3-1 A3] 废除 force_include：v10_pred_rank 如未通过 IC 筛选，说明当期无效，不强制追加
    # 注：PSI_EXEMPT 已确保 v10_pred_rank 不被 PSI 过滤，IC 筛选由 composite_score 自然决定
    return good_feats, feat_scores, max_ic, {fs['feature']: fs['ic_abs'] for fs in all_feat_scores}


def _fmt_date_str(d):
    """将日期对象或字符串格式化为 YYYYMMDD"""
    if d is None:
        return 'unknown'
    if hasattr(d, 'strftime'):
        return d.strftime('%Y%m%d')
    return str(d)[:10].replace('-', '')








def prepare_dmatrix(df, features, label=None, weights=None, group_sizes=None):
    """
    构建 XGBoost DMatrix，支持样本权重。
    【2026-05-08 改造】移除 fillna(0)，改由 XGBoost 原生处理 NaN。
    XGBoost DMatrix 在创建时设置 missing=np.nan，可原生支持 NaN 值。
    【R3-1 M5】binary:logistic 不需要 group_sizes，参数保留向后兼容。
    """
    X = df[features].replace([np.inf, -np.inf], np.nan)
    X = X.astype(np.float32)
    if label is not None:
        if weights is not None:
            return xgb.DMatrix(X, label=label, weight=weights, missing=np.nan, feature_names=features)
        else:
            return xgb.DMatrix(X, label=label, missing=np.nan, feature_names=features)
    else:
        return xgb.DMatrix(X, missing=np.nan, feature_names=features)


# ==============================
# [Part C] LR Warmup Callback
# 学习率从 warmup_lr 线性递增至 target_lr
# 参考: Wyckoff V10, De Prado AFML
# XGBoost 2.x compatible: 必须继承 TrainingCallback
# ==============================
from xgboost.callback import TrainingCallback

class LRWarmupCallback(TrainingCallback):
    """XGBoost 2.x 兼容 LR Warmup 回调。
    前 warmup_rounds 轮学习率从 warmup_lr 线性递增至 target_lr，
    之后维持 target_lr。
    """
    def __init__(self, warmup_lr: float = 0.005, target_lr: float = 0.05,
                 warmup_rounds: int = 20):
        super().__init__()
        self.warmup_lr = warmup_lr
        self.target_lr = target_lr
        self.warmup_rounds = warmup_rounds

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        if epoch < self.warmup_rounds:
            ratio = (epoch + 1) / self.warmup_rounds
            new_lr = self.warmup_lr + (self.target_lr - self.warmup_lr) * ratio
            model.set_param('learning_rate', new_lr)
            if epoch == 0:
                logger.debug(f"[LRWarmup] start lr={self.warmup_lr:.4f} -> target={self.target_lr}")
            if epoch == self.warmup_rounds - 1:
                logger.debug(f"[LRWarmup] end lr={new_lr:.4f} (warmup complete)")
        return False  # 不停止训练
class ProbabilityCalibrator:
    """校准原始预测分数为 [0,1] 概率。

    支持两种方法:
    - 'isotonic': IsotonicRegression 单调映射 (可能产生 binning 平台效应)
    - 'platt': LogisticRegression sigmoid 拟合 (平滑, 推荐)

    参考文献: Prado AFML Ch.10 Step 2 "Calibrate the probabilities"
    """
    def __init__(self, method: str = 'isotonic'):
        self.calibrator = None
        self.is_fitted = False
        self.method = method

    def fit(self, raw_scores, true_labels):
        if self.method == 'platt':
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression(penalty=None, solver='lbfgs')
            self.calibrator.fit(raw_scores.ravel().reshape(-1, 1), true_labels.ravel())
            calibrated = self.calibrator.predict_proba(raw_scores.ravel().reshape(-1, 1))[:, 1]
        else:
            from sklearn.isotonic import IsotonicRegression
            self.calibrator = IsotonicRegression(y_min=0.05, y_max=0.95, out_of_bounds='clip')
            self.calibrator.fit(raw_scores.ravel(), true_labels.ravel())
            calibrated = self.calibrator.predict(raw_scores.ravel())

        self.is_fitted = True

        # 退化检测
        prob_range = calibrated.max() - calibrated.min()
        if prob_range < 0.01:
            logger.warning(
                f"[ProbabilityCalibrator] 概率校准退化: range={prob_range:.4f} (method={self.method})"
            )
        return calibrated

    def predict(self, raw_scores):
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        if self.method == 'platt':
            calibrated = self.calibrator.predict_proba(raw_scores.ravel().reshape(-1, 1))[:, 1]
        else:
            calibrated = self.calibrator.predict(raw_scores.ravel())
        return np.clip(calibrated, 0.05, 0.95)

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.calibrator, f)

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            self.calibrator = pickle.load(f)
        self.is_fitted = True


# ============================================================================
#  [L2] 目标模式 (binary / regression) 检测与统一度量
# ============================================================================

def _objective_metric_fn(mode):
    """[L2] 按目标模式返回评估函数: binary→ROC AUC, regression→Spearman rank IC。

    回归模式用 Spearman IC 是业界标准 (选股模型的排序质量度量),
    与选股业务目标 (按预测分位排序取 Top-N) 直接对齐。
    """
    if mode == 'regression':
        def _ic(y_true, y_pred):
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)
            if len(y_true) < 3 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
                return 0.0
            from scipy.stats import spearmanr
            r = spearmanr(y_true, y_pred)[0]
            return float(r) if np.isfinite(r) else 0.0
        return _ic

    def _auc(y_true, y_pred):
        try:
            return float(roc_auc_score(y_true, y_pred))
        except ValueError:
            return 0.5
    return _auc


def _detect_objective_mode(y, config=None):
    """[L2] 目标模式检测: config.OBJECTIVE_MODE 显式指定优先
    ('binary'/'regression'), 'auto' 时按目标取值自动判定:
    唯一取值 ⊆ {0,1} → binary, 否则 → regression。"""
    forced = getattr(config, 'OBJECTIVE_MODE', 'auto') if config is not None else 'auto'
    if forced in ('binary', 'regression'):
        return forced
    y = np.asarray(y)
    y = y[~pd.isna(y)]
    if len(y) == 0:
        return 'binary'
    uniq = np.unique(y.astype(float))
    if len(uniq) <= 2 and set(uniq.tolist()) <= {0.0, 1.0}:
        return 'binary'
    return 'regression'


def compute_mda_oos(model, X_val_df, y_val, feature_names, n_repeats=5, random_state=42,
                    metric_fn=None):
    """在验证集上计算每个特征的 OOS (Out-Of-Sample) MDA。

    对每个特征 f：
    1. 基线：model.predict(X_val) → AUC_baseline
    2. 将 X_val[f] 随机置换
    3. model.predict(X_val_permuted) → AUC_permuted
    4. MDA(f) = AUC_baseline - AUC_permuted（重复 n_repeats 次取均值）

    MDA(f) > 0：特征有正向贡献
    MDA(f) ≈ 0：特征无效
    MDA(f) < 0：特征产生噪声

    Args:
        model: XGBoost Booster
        X_val_df: 验证集特征 DataFrame
        y_val: 验证集标签 array
        feature_names: 要评估的特征名列表
        n_repeats: 每个特征置换次数（建议≥10以支持t检验）
        random_state: 随机种子（固定保证可复现）

    Returns:
        Tuple[Dict[str, float], Dict[str, float]]: (特征名→MDA均值, 特征名→MDA标准差)
        标准差用于后续 t 检验自适应阈值（mean > t_crit * se）
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from scipy import stats as sp_stats
    import logging
    logger = logging.getLogger(__name__)

    rng = np.random.RandomState(random_state)
    # [L2] 统一度量: 默认 AUC, regression 模式传入 Spearman IC
    if metric_fn is None:
        metric_fn = _objective_metric_fn('binary')

    # === 性能优化：预转换为 numpy float32 数组 ===
    # 与原代码 X_val_df[feature_names].replace([np.inf, -np.inf], np.nan).astype(np.float32) 完全等价
    # 跳过每次迭代的 DataFrame copy/replace/astype 开销
    X_base = X_val_df[feature_names].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
    feat_to_idx = {f: i for i, f in enumerate(feature_names)}

    # 基线 DMatrix（复用 prepare_dmatrix，保证与训练路径一致）
    dval = prepare_dmatrix(X_val_df, feature_names, label=y_val)

    # 基线 AUC/IC
    try:
        baseline_proba = model.predict(dval)
        baseline_auc = float(metric_fn(y_val, baseline_proba))
    except ValueError:
        logger.warning("MDA: 基线度量计算失败，跳过")
        return {f: 0.0 for f in feature_names}, {f: 0.0 for f in feature_names}

    total = len(feature_names)
    n_tasks = total * n_repeats
    logger.info("[MDA-OOS] 基线 Val 度量=%.4f，开始评估 %d 个特征 (n_repeats=%d, total_tasks=%d)",
                baseline_auc, total, n_repeats, n_tasks)

    # === 预生成所有置换（串行，RNG 调用顺序与原代码完全一致）===
    # 原代码顺序：for feat: for rep: col = X[feat].copy(); rng.shuffle(col)
    # 预生成顺序完全相同，保证可复现
    all_tasks = []  # [(feat_name, feat_idx, shuffled_col), ...]
    for feat in feature_names:
        fidx = feat_to_idx[feat]
        for _ in range(n_repeats):
            col = X_base[:, fidx].copy()
            rng.shuffle(col)
            all_tasks.append((feat, fidx, col))

    # === 并行执行 predict + AUC ===
    # [Perf] Thread-local workspace: 每线程分配一次, [:] = X_base 批量恢复
    import threading
    _thread_local = threading.local()

    def _eval_one(task):
        """单任务评估：创建置换DMatrix -> predict -> AUC -> drop"""
        feat, fidx, shuffled_col = task
        if not hasattr(_thread_local, 'X_ws'):
            _thread_local.X_ws = X_base.copy()
            _thread_local.X_ws_fnames = feature_names
        _thread_local.X_ws[:] = X_base
        _thread_local.X_ws[:, fidx] = shuffled_col
        dval_perm = xgb.DMatrix(_thread_local.X_ws, label=y_val, missing=np.nan,
                                feature_names=_thread_local.X_ws_fnames)
        try:
            perm_proba = model.predict(dval_perm)
            perm_auc = float(metric_fn(y_val, perm_proba))
            return feat, baseline_auc - perm_auc
        except ValueError:
            return feat, 0.0

    # 小规模任务串行执行（避免线程开销）
    if n_tasks <= 20:
        results = [_eval_one(task) for task in all_tasks]
    else:
        from concurrent.futures import ThreadPoolExecutor
        max_workers = min(8, os.cpu_count() or 8)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_eval_one, all_tasks))

    # === 按特征聚合结果（顺序与原代码一致）===
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

    # t 检验自适应阈值统计
    df_n = n_repeats - 1
    t_crit = float(sp_stats.t.ppf(1 - 0.05, df_n))  # 默认 alpha=0.05 单侧
    se_vals = np.array([mda_stds[f] / np.sqrt(n_repeats) for f in feature_names])
    mean_vals = np.array([mda_means[f] for f in feature_names])
    t_stats = np.where(se_vals > 0, mean_vals / se_vals, 0.0)
    t_pass = t_stats > t_crit

    # 详尽MDA分布日志（诊断 MDA>0 vs t检验通过 的差异）
    mda_vals = mean_vals
    n_pos = int(np.sum(mda_vals > 0))
    n_neg = int(np.sum(mda_vals < 0))
    n_zero = int(np.sum(mda_vals == 0))
    n_t001 = int(np.sum(mda_vals > 0.001))
    n_t005 = int(np.sum(mda_vals > 0.005))
    n_t01 = int(np.sum(mda_vals > 0.01))
    n_ttest_pass = int(np.sum(t_pass))

    logger.info("[MDA-OOS] 完成 (基线AUC=%.4f, n=%d, n_repeats=%d, t_crit=%.3f @df=%d):",
                baseline_auc, total, n_repeats, t_crit, df_n)
    logger.info("  MDA均值分布: min=%.6f, max=%.6f, mean=%.6f, median=%.6f",
                float(np.min(mda_vals)), float(np.max(mda_vals)),
                float(np.mean(mda_vals)), float(np.median(mda_vals)))
    logger.info("  MDA标准差分布: min=%.6f, max=%.6f, mean=%.6f, median=%.6f",
                float(np.min(se_vals * np.sqrt(n_repeats))), float(np.max(se_vals * np.sqrt(n_repeats))),
                float(np.mean(se_vals * np.sqrt(n_repeats))), float(np.median(se_vals * np.sqrt(n_repeats))))
    logger.info("  MDA分位: P10=%.6f, P25=%.6f, P50=%.6f, P75=%.6f, P90=%.6f, P95=%.6f",
                float(np.percentile(mda_vals, 10)), float(np.percentile(mda_vals, 25)),
                float(np.percentile(mda_vals, 50)), float(np.percentile(mda_vals, 75)),
                float(np.percentile(mda_vals, 90)), float(np.percentile(mda_vals, 95)))
    logger.info("  t统计量分位: P10=%.3f, P25=%.3f, P50=%.3f, P75=%.3f, P90=%.3f, P95=%.3f (临界=%.3f)",
                float(np.percentile(t_stats, 10)), float(np.percentile(t_stats, 25)),
                float(np.percentile(t_stats, 50)), float(np.percentile(t_stats, 75)),
                float(np.percentile(t_stats, 90)), float(np.percentile(t_stats, 95)), t_crit)
    logger.info("  阈值分布: MDA>0=%d, MDA>0.001=%d, MDA>0.005=%d, MDA>0.01=%d | t检验通过(>%.3f)=%d | MDA<0=%d, MDA==0=%d",
                n_pos, n_t001, n_t005, n_t01, t_crit, n_ttest_pass, n_neg, n_zero)

    # Top-5 / Bottom-5 特征 MDA 值（含 t 统计量）
    feat_stats = [(f, mda_means[f], mda_stds[f], t_stats[i]) for i, f in enumerate(feature_names)]
    feat_stats.sort(key=lambda x: x[1], reverse=True)
    top5_str = ", ".join([f"{f}={m:.6f}(t={t:.2f})" for f, m, s, t in feat_stats[:5]])
    bot5_str = ", ".join([f"{f}={m:.6f}(t={t:.2f})" for f, m, s, t in feat_stats[-5:]])
    logger.info("  Top-5 MDA: %s", top5_str)
    logger.info("  Bottom-5 MDA: %s", bot5_str)
    return mda_means, mda_stds


def compute_mda_oos_clustered(model, X_val_df, y_val, clusters, n_repeats=5, random_state=42,
                              metric_fn=None):
    """[Test-CFI] 簇级 OOS MDA（Prado MLAM Ch.6 Clustered MDA）: 置换单元=特征簇。

    对每个簇: 同时置换簇内全部特征列（逐列独立 shuffle, 与 compute_mda_oos 同机制）,
    度量性能下降 → 簇级 MDA。簇内特征同时被打乱 → 相关替代特征无法填补 →
    簇级重要性不受簇内线性替代效应污染。
    单例簇等价于 AFML §8.3.2 逐特征 MDA（完全同语义, 同 RNG 顺序）。

    Args:
        model: XGBoost Booster
        X_val_df: 验证集特征 DataFrame
        y_val: 验证集标签 array
        clusters: List[List[str]] 簇结构（每簇成员仅保留验证集中存在的列）
        n_repeats: 每簇置换次数
        random_state: 随机种子
        metric_fn: 度量函数 (默认 AUC; regression 传 Spearman IC)

    Returns:
        Tuple[Dict[int, float], Dict[int, float]]: 簇索引 → (MDA均值, MDA标准差)
        簇索引 = clusters 下标; 某折无可见成员的簇在该折无记录（调用方按缺失处理）
    """
    import numpy as np
    import threading
    import logging
    from collections import defaultdict
    logger = logging.getLogger(__name__)

    if metric_fn is None:
        metric_fn = _objective_metric_fn('binary')

    # members 必须保持 X_val_df 的列顺序（= 模型训练特征顺序）:
    # XGBoost predict 会校验 DMatrix feature_names 与模型一致, 若按簇展平顺序
    # (相关性分组序 ≠ 特征名排序) 会抛 ValueError: feature_names mismatch →
    # 基线度量失败 → 整折簇级 MDA 失效 (2026-08-06 生产回测暴露, 测试数据恰好未触发)
    cluster_feat_set = set(f for c in clusters for f in c)
    members = [f for f in X_val_df.columns if f in cluster_feat_set]
    if not members:
        return {i: 0.0 for i in range(len(clusters))}, {i: 0.0 for i in range(len(clusters))}

    X_base = X_val_df[members].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
    feat_to_idx = {f: i for i, f in enumerate(members)}

    dval = prepare_dmatrix(X_val_df, members, label=y_val)
    try:
        baseline_proba = model.predict(dval)
        baseline_metric = float(metric_fn(y_val, baseline_proba))
    except ValueError:
        logger.warning("[CFI-MDA] 基线度量计算失败，跳过")
        return {i: 0.0 for i in range(len(clusters))}, {i: 0.0 for i in range(len(clusters))}

    rng = np.random.RandomState(random_state)

    # 任务生成: 每簇 n_repeats 次, 每次为该簇所有成员列生成独立 shuffle 后的列
    all_tasks = []
    for cid, cluster in enumerate(clusters):
        idxs = [feat_to_idx[f] for f in cluster if f in feat_to_idx]
        if not idxs:
            continue  # 该折此簇无可见成员 → 无记录（投票按缺失处理）
        for _ in range(n_repeats):
            cols = []
            for idx in idxs:
                col = X_base[:, idx].copy()
                rng.shuffle(col)
                cols.append((idx, col))
            all_tasks.append((cid, cols))

    _thread_local = threading.local()

    def _eval_one(task):
        cid, cols = task
        if not hasattr(_thread_local, 'X_ws'):
            _thread_local.X_ws = X_base.copy()
            _thread_local.X_ws_fnames = members
        _thread_local.X_ws[:] = X_base
        for idx, col in cols:
            _thread_local.X_ws[:, idx] = col
        dval_perm = xgb.DMatrix(_thread_local.X_ws, label=y_val, missing=np.nan,
                                feature_names=_thread_local.X_ws_fnames)
        try:
            perm_proba = model.predict(dval_perm)
            perm_metric = float(metric_fn(y_val, perm_proba))
            return cid, baseline_metric - perm_metric
        except ValueError:
            return cid, 0.0

    if len(all_tasks) <= 20:
        results = [_eval_one(t) for t in all_tasks]
    else:
        from concurrent.futures import ThreadPoolExecutor
        max_workers = min(8, os.cpu_count() or 8)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_eval_one, all_tasks))

    cluster_drops = defaultdict(list)
    for cid, drop in results:
        cluster_drops[cid].append(drop)

    mda_means = {}
    mda_stds = {}
    for cid in range(len(clusters)):
        drops = cluster_drops.get(cid, [])
        mda_means[cid] = float(np.mean(drops)) if drops else 0.0
        mda_stds[cid] = float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0

    logger.info("[CFI-MDA] 簇级 OOS MDA 完成 (基线=%.4f, %d簇, n_repeats=%d, tasks=%d)",
                baseline_metric, len(clusters), n_repeats, len(all_tasks))
    return mda_means, mda_stds


def aggregate_fold_votes(fold_ic_results, fold_mdi_results, fold_mda_results, config,
                       consensus_features=None, fold_mda_stds=None):
    """多折投票聚合：IC 共识 → MDA 投票 → 综合评分 → Top-K 截断。

    三层递进（Prado AFML 范式）：
    1. 折内排名归一化（百分位）
    2. 跨折均值排名
    3. 综合评分 = w_ic*avg_ic_rank + w_mdi*avg_mdi_rank + w_mda*avg_mda_rank
    4. MDA 投票准入（≥vote_min_folds 折通过 t检验或固定阈值）
    5. Top-K 截断

    Args:
        fold_ic_results: List[Dict[str, float]]  各折特征的 |IC| 绝对值
        fold_mdi_results: List[Dict[str, float]] 各折特征的 MDI 值
        fold_mda_results: List[Dict[str, float]] 各折特征的 OOS MDA 均值
        config: 配置对象
        consensus_features: List[str]  IC 共识特征（预筛选层），若提供则仅在此子集内
                           做 MDA 投票；若为 None 则在全部特征上做（向后兼容）
        fold_mda_stds: List[Dict[str, float]]  各折特征的 OOS MDA 标准差（用于 t 检验）

    Returns:
        final_features: List[str]  按综合评分排序的最终特征集
        vote_report: dict  包含各特征详细评分，供日志
    """
    import numpy as np
    import pandas as pd
    import logging
    logger = logging.getLogger(__name__)

    use_voting = getattr(config, 'use_voting', True)
    mda_threshold = getattr(config, 'mda_threshold', 0.001)
    mda_use_std = getattr(config, 'mda_use_std_threshold', False)
    mda_t_alpha = getattr(config, 'mda_t_alpha', 0.05)
    mda_n_repeats = getattr(config, 'mda_n_repeats', 5)
    vote_min_folds = getattr(config, 'vote_min_folds', 3)
    vote_min_features = getattr(config, 'vote_min_features', 10)
    top_k = getattr(config, 'top_k_features', 40)
    w_ic = getattr(config, 'weight_ic', 1.0)
    w_mdi = getattr(config, 'weight_mdi', 0.3)
    w_mda = getattr(config, 'weight_mda', 1.5)
    fallback_to_ic = getattr(config, 'fallback_to_ic_only', True)

    # t 检验临界值（单侧）
    if mda_use_std and mda_n_repeats > 1:
        from scipy import stats as sp_stats
        t_crit = float(sp_stats.t.ppf(1 - mda_t_alpha, mda_n_repeats - 1))
        logger.info("[FeatureVoting] MDA投票模式: t检验 (alpha=%.2f, t_crit=%.3f, df=%d, n_repeats=%d)",
                    mda_t_alpha, t_crit, mda_n_repeats - 1, mda_n_repeats)
    else:
        t_crit = None
        logger.info("[FeatureVoting] MDA投票模式: 固定阈值 (%.6f)", mda_threshold)

    n_folds = len(fold_ic_results)
    if n_folds == 0:
        logger.warning("[FeatureVoting] 无有效折，返回空特征集")
        return [], {}

    # 收集所有出现过的特征
    all_features = set()
    for fd in fold_ic_results:
        all_features.update(fd.keys())

    # 所有特征平等参与 MDA 投票（不再豁免 v10_pred_rank）
    # Stage 0: IC 共识预筛选（Prado AFML 三层递进第一层）
    if consensus_features is not None and len(consensus_features) > 0:
        base_set = set(consensus_features)
        all_count = len(all_features)
        normal_feats = sorted(base_set)
        consensus_n = getattr(config, 'cv_feature_consensus', n_folds)
        logger.info("[FeatureVoting] IC共识预筛选: %d → %d 特征 (共识阈值=≥%d折/%d折)",
                    all_count, len(normal_feats), consensus_n, n_folds)
    else:
        normal_feats = sorted(all_features)
    logger.info("[FeatureVoting] 跨折聚合: %d 折, %d 个IC共识候选 (所有特征平等参与)",
                n_folds, len(normal_feats))

    # --- 第一步：各折内排名归一化 ---
    def percentile_rank(values_dict, feats):
        """将 dict 值转为 [0,1] 百分位排名"""
        vals = np.array([values_dict.get(f, 0.0) for f in feats])
        from scipy.stats import rankdata
        ranks = rankdata(vals) / max(len(vals), 1)
        return dict(zip(feats, ranks))

    ic_ranks = []
    mdi_ranks = []
    mda_ranks = []
    mda_votes = {f: 0 for f in normal_feats}

    for k in range(n_folds):
        ic_dict = fold_ic_results[k]
        mdi_dict = fold_mdi_results[k] if k < len(fold_mdi_results) else {}
        mda_dict = fold_mda_results[k] if k < len(fold_mda_results) else {}
        std_dict = fold_mda_stds[k] if (fold_mda_stds and k < len(fold_mda_stds)) else {}

        ic_ranks.append(percentile_rank(ic_dict, normal_feats))
        if mdi_dict:
            mdi_ranks.append(percentile_rank(mdi_dict, normal_feats))
        if mda_dict:
            mda_ranks.append(percentile_rank(mda_dict, normal_feats))
            # MDA 投票计数：t检验或固定阈值
            for f in normal_feats:
                mda_mean = mda_dict.get(f, -999)
                if mda_use_std and t_crit is not None and std_dict:
                    # t 检验: mean > t_crit * SE, SE = std / sqrt(n_repeats)
                    mda_std = std_dict.get(f, 0.0)
                    se = mda_std / np.sqrt(mda_n_repeats) if mda_n_repeats > 0 else 0.0
                    if se > 0 and mda_mean > t_crit * se:
                        mda_votes[f] += 1
                    elif se == 0 and mda_mean > 0:
                        # std=0 时退化为 mean>0（所有重复结果一致）
                        mda_votes[f] += 1
                else:
                    # 固定阈值模式（向后兼容）
                    if mda_mean > mda_threshold:
                        mda_votes[f] += 1

    # --- 第二步：跨折均值排名 ---
    avg_ic_rank = {}
    avg_mdi_rank = {}
    avg_mda_rank = {}

    for f in normal_feats:
        avg_ic_rank[f] = np.mean([r.get(f, 0.0) for r in ic_ranks]) if ic_ranks else 0.0
        avg_mdi_rank[f] = np.mean([r.get(f, 0.0) for r in mdi_ranks]) if mdi_ranks else 0.0
        avg_mda_rank[f] = np.mean([r.get(f, 0.0) for r in mda_ranks]) if mda_ranks else 0.0

    # --- 第三步：综合评分 ---
    total_w = w_ic + w_mdi + w_mda
    if total_w == 0:
        total_w = 1.0

    scores = {}
    for f in normal_feats:
        scores[f] = (
            w_ic * avg_ic_rank[f] +
            w_mdi * avg_mdi_rank[f] +
            w_mda * avg_mda_rank[f]
        ) / total_w

    # --- 第四步：MDA 投票准入 ---
    has_mda = len(fold_mda_results) > 0 and any(d for d in fold_mda_results if d)
    if has_mda and use_voting:
        candidate_feats = [f for f in normal_feats if mda_votes[f] >= vote_min_folds]
    else:
        # 无 MDA 或 use_voting=false: 只按 IC 排序
        candidate_feats = list(normal_feats)

    # --- 降级保护 ---
    effective_vote_min = vote_min_folds
    while len(candidate_feats) < vote_min_features and effective_vote_min > 1:
        prev_count = len(candidate_feats)
        prev_min = effective_vote_min
        effective_vote_min -= 1
        candidate_feats = [f for f in normal_feats if mda_votes[f] >= effective_vote_min]
        logger.warning("[FeatureVoting] 通过特征不足(%d<%d @≥%d折)，放宽至 ≥%d折 → %d 个候选",
                       prev_count, vote_min_features, prev_min, effective_vote_min, len(candidate_feats))

    if len(candidate_feats) < vote_min_features and fallback_to_ic:
        logger.warning("[FeatureVoting] 降级至纯 IC 排序（fallback_to_ic_only=true）")
        candidate_feats = sorted(normal_feats, key=lambda f: avg_ic_rank.get(f, 0), reverse=True)
        effective_vote_min = 0

    # --- 第五步：Top-K 截断 ---
    # [v3.0] Deterministic tiebreaker: (score, name) tuple breaks ties alphabetically
    candidate_feats = sorted(candidate_feats, key=lambda f: (scores.get(f, 0), f), reverse=True)
    final_features = candidate_feats[:top_k] if top_k and top_k > 0 else candidate_feats

    # v10_pred_rank 走正常筛选流程，不再无条件追加
    # （若 IC/MDA 通过，自然会在 final_features 中；若未通过，说明当期无效）

    # --- 日志（修正：精确分布 vs 累积分布分离，消除 >= 混淆） ---
    vote_counts = [mda_votes.get(f, 0) for f in normal_feats]
    total_vote_events = sum(vote_counts)
    # 精确分布（==n折）
    n_exact = {i: sum(1 for v in vote_counts if v == i) for i in range(n_folds + 1)}
    # 累积分布（≥n折）
    n_cumul = {i: sum(1 for v in vote_counts if v >= i) for i in range(1, n_folds + 1)}

    logger.info("[FeatureVoting] MDA投票分布 (模式=%s, 共%d特征, 投票事件=%d):",
                "t检验(α=%.2f,t=%.3f)" % (mda_t_alpha, t_crit) if t_crit else "阈值(%.4f)" % mda_threshold,
                len(normal_feats), total_vote_events)
    logger.info("  精确分布: 5折=%d, 4折=%d, 3折=%d, 2折=%d, 1折=%d, 0折=%d",
                n_exact.get(5, 0), n_exact.get(4, 0), n_exact.get(3, 0),
                n_exact.get(2, 0), n_exact.get(1, 0), n_exact.get(0, 0))
    logger.info("  累积分布: ≥5折=%d, ≥4折=%d, ≥3折(准入)=%d, ≥2折=%d, ≥1折=%d",
                n_cumul.get(5, 0), n_cumul.get(4, 0), n_cumul.get(3, 0),
                n_cumul.get(2, 0), n_cumul.get(1, 0))

    # 诊断: 各折 MDA>0 vs 投票通过 对比
    for k in range(n_folds):
        mda_dict = fold_mda_results[k] if k < len(fold_mda_results) else {}
        std_dict = fold_mda_stds[k] if (fold_mda_stds and k < len(fold_mda_stds)) else {}
        if mda_dict:
            fold_mda_vals = np.array(list(mda_dict.values()))
            n_gt0 = int(np.sum(fold_mda_vals > 0))
            n_gt_thresh = int(np.sum(fold_mda_vals > mda_threshold))

            # t检验通过数
            if mda_use_std and t_crit is not None and std_dict:
                fold_t_pass = 0
                for f in mda_dict:
                    mda_std = std_dict.get(f, 0.0)
                    se = mda_std / np.sqrt(mda_n_repeats) if mda_n_repeats > 0 else 0.0
                    if se > 0 and mda_dict[f] > t_crit * se:
                        fold_t_pass += 1
                    elif se == 0 and mda_dict[f] > 0:
                        fold_t_pass += 1
                consensus_pass = sum(1 for f in normal_feats
                                     if f in mda_dict and (
                                         (std_dict.get(f, 0.0) / np.sqrt(mda_n_repeats) > 0 and
                                          mda_dict[f] > t_crit * std_dict.get(f, 0.0) / np.sqrt(mda_n_repeats))
                                         or (std_dict.get(f, 0.0) / np.sqrt(mda_n_repeats) == 0 and mda_dict[f] > 0)
                                     ))
                logger.info("  Fold %d: MDA>0=%d, MDA>%.4f=%d, t检验通过=%d | 共识池t检验通过=%d/%d",
                            k + 1, n_gt0, mda_threshold, n_gt_thresh, fold_t_pass,
                            consensus_pass, len(normal_feats))
            else:
                consensus_pass = sum(1 for f in normal_feats if mda_dict.get(f, -999) > mda_threshold)
                logger.info("  Fold %d: MDA>0=%d, MDA>%.4f=%d (其中(0,%.4f]=%d) | 共识池通过=%d/%d",
                            k + 1, n_gt0, mda_threshold, n_gt_thresh, mda_threshold,
                            n_gt0 - n_gt_thresh, consensus_pass, len(normal_feats))

    # 诊断: 0票特征的IC/MDI排名分布（判断是否为低IC+低MDI特征）
    zero_vote_feats = [f for f in normal_feats if mda_votes.get(f, 0) == 0]
    if zero_vote_feats:
        zero_ic_ranks = [avg_ic_rank.get(f, 0) for f in zero_vote_feats]
        zero_mdi_ranks = [avg_mdi_rank.get(f, 0) for f in zero_vote_feats]
        logger.info("  0票特征(%d个)诊断: avg_ic_rank均值=%.3f, avg_mdi_rank均值=%.3f",
                    len(zero_vote_feats), float(np.mean(zero_ic_ranks)), float(np.mean(zero_mdi_ranks)))

    logger.info("  Top-K 截断前: %d, 截断后: %d", len(candidate_feats), len(final_features))

    # 构建报告
    report = {
        'features': final_features,
        'scores': {f: round(scores.get(f, 0), 4) for f in final_features},
        'votes': {f: int(mda_votes.get(f, 0)) for f in final_features},
        'ic_scores': {f: round(avg_ic_rank.get(f, 0), 4) for f in final_features},
        'avg_ic_rank': {f: round(avg_ic_rank.get(f, 0), 4) for f in final_features},
        'avg_mda_rank': {f: round(avg_mda_rank.get(f, 0), 4) for f in final_features},
        'vote_min_folds': effective_vote_min,
        # [v3.5] Vote distribution for V10-style logging
        'vote_distribution': {k: v for k, v in n_cumul.items() if v > 0},
    }
    return final_features, report


def aggregate_cluster_votes(fold_cluster_mda, fold_cluster_mda_std, clusters, config,
                            fold_ic_results=None, fold_mdi_results=None,
                            consensus_features=None):
    """[Test-CFI] 簇级投票聚合（Prado MLAM Ch.6 Clustered MDA 投票）: 单元=特征簇。

    与 aggregate_fold_votes 同构的三层递进, 只是评估单元从"特征"换成"簇":
      1. 折内簇级排名归一化（百分位）→ 跨折均值排名
      2. 簇级综合评分 = w_ic*avg_ic_rank(簇内成员IC均值) + w_mdi*avg_mdi_rank(簇内MDI均值)
                        + w_mda*avg_mda_rank(簇级置换MDA)
      3. 簇级 t 检验投票: 每折该簇 mean > t_crit*SE → 1票; 需 ≥ vote_min_folds 折
      4. 降级保护: 入选簇数 < vote_min_features → 放宽 vote_min_folds → fallback 簇级IC
      5. Top-K 截断（簇维度, 每簇最终出 1 代表 → 特征数 ≤ top_k）

    Args:
        fold_cluster_mda: List[Dict[int, float]] 每折簇级 MDA 均值 (簇索引→均值)
        fold_cluster_mda_std: List[Dict[int, float]] 每折簇级 MDA 标准差
        clusters: List[List[str]] 全局簇结构（索引与 fold_cluster_mda 的 key 对齐）
        config: 配置对象
        fold_ic_results: List[Dict[str, float]] 折内特征 |IC|（用于簇级 IC 均值）
        fold_mdi_results: List[Dict[str, float]] 折内特征 MDI（用于簇级 MDI 均值）
        consensus_features: List[str] IC 共识预筛池 (仅与池有交集的簇参与投票)

    Returns:
        Tuple[List[int], dict]: 入选簇索引列表 + cluster_report（含簇级 score/votes/排名）
    """
    import numpy as np
    import logging
    from scipy.stats import rankdata
    logger = logging.getLogger(__name__)

    use_voting = getattr(config, 'use_voting', True)
    mda_threshold = getattr(config, 'mda_threshold', 0.001)
    mda_use_std = getattr(config, 'mda_use_std_threshold', False)
    mda_t_alpha = getattr(config, 'mda_t_alpha', 0.05)
    mda_n_repeats = getattr(config, 'mda_n_repeats', 5)
    vote_min_folds = getattr(config, 'vote_min_folds', 3)
    vote_min_features = getattr(config, 'vote_min_features', 10)
    top_k = getattr(config, 'top_k_features', 40)
    w_ic = getattr(config, 'weight_ic', 1.0)
    w_mdi = getattr(config, 'weight_mdi', 0.3)
    w_mda = getattr(config, 'weight_mda', 1.5)
    fallback_to_ic = getattr(config, 'fallback_to_ic_only', True)

    if mda_use_std and mda_n_repeats > 1:
        from scipy import stats as sp_stats
        t_crit = float(sp_stats.t.ppf(1 - mda_t_alpha, mda_n_repeats - 1))
        logger.info("[CFI-Vote] 簇级MDA投票: t检验 (alpha=%.2f, t_crit=%.3f, df=%d, n_repeats=%d)",
                    mda_t_alpha, t_crit, mda_n_repeats - 1, mda_n_repeats)
    else:
        t_crit = None
        logger.info("[CFI-Vote] 簇级MDA投票: 固定阈值 (%.6f)", mda_threshold)

    n_folds = len(fold_cluster_mda)
    if n_folds == 0:
        logger.warning("[CFI-Vote] 无有效折，返回空簇集")
        return [], {}

    # ---- 投票池: 与 IC 共识池有交集的簇 (保留原流程 IC 共识预筛层) ----
    if consensus_features is not None and len(consensus_features) > 0:
        consensus_set = set(consensus_features)
        pool_ids = [cid for cid in range(len(clusters)) if set(clusters[cid]) & consensus_set]
        logger.info("[CFI-Vote] IC共识预筛: %d 簇 → %d 簇参与投票 (共识特征=%d)",
                    len(clusters), len(pool_ids), len(consensus_set))
    else:
        pool_ids = list(range(len(clusters)))

    # ---- 折内聚合: 簇级 IC/MDI (成员均值) + 簇级 MDA 排名 + t 检验投票 ----
    cluster_ic_ranks = []
    cluster_mdi_ranks = []
    cluster_mda_ranks = []
    cluster_votes = {cid: 0 for cid in pool_ids}

    for k in range(n_folds):
        mda_dict = fold_cluster_mda[k] if k < len(fold_cluster_mda) else {}
        std_dict = fold_cluster_mda_std[k] if (fold_cluster_mda_std and k < len(fold_cluster_mda_std)) else {}
        ic_dict = fold_ic_results[k] if (fold_ic_results and k < len(fold_ic_results)) else {}
        mdi_dict = fold_mdi_results[k] if (fold_mdi_results and k < len(fold_mdi_results)) else {}

        def _cluster_mean(feature_dict, cid, default=0.0):
            vals = [feature_dict[f] for f in clusters[cid] if f in feature_dict]
            return float(np.mean(vals)) if vals else default

        c_ic = {cid: _cluster_mean(ic_dict, cid) for cid in pool_ids}
        c_mdi = {cid: _cluster_mean(mdi_dict, cid) for cid in pool_ids}
        c_mda = {cid: mda_dict.get(cid, 0.0) for cid in pool_ids}

        ic_vals = np.array([c_ic[cid] for cid in pool_ids])
        mdi_vals = np.array([c_mdi[cid] for cid in pool_ids])
        mda_vals = np.array([c_mda[cid] for cid in pool_ids])
        cluster_ic_ranks.append(dict(zip(pool_ids, rankdata(ic_vals) / max(len(pool_ids), 1))))
        cluster_mdi_ranks.append(dict(zip(pool_ids, rankdata(mdi_vals) / max(len(pool_ids), 1))))
        cluster_mda_ranks.append(dict(zip(pool_ids, rankdata(mda_vals) / max(len(pool_ids), 1))))

        # t 检验投票: 仅对该折有 MDA 记录的簇计票 (无记录=缺失=不投票, 与原 mda_dict.get(f,-999) 对称)
        for cid in pool_ids:
            if cid not in mda_dict:
                continue
            mean = mda_dict[cid]
            if mda_use_std and t_crit is not None:
                std = std_dict.get(cid, 0.0)
                se = std / np.sqrt(mda_n_repeats) if mda_n_repeats > 0 else 0.0
                if se > 0 and mean > t_crit * se:
                    cluster_votes[cid] += 1
                elif se == 0 and mean > 0:
                    cluster_votes[cid] += 1
            else:
                if mean > mda_threshold:
                    cluster_votes[cid] += 1

    # ---- 跨折均值排名 ----
    avg_ic_rank = {cid: float(np.mean([r.get(cid, 0.0) for r in cluster_ic_ranks])) if cluster_ic_ranks else 0.0
                   for cid in pool_ids}
    avg_mdi_rank = {cid: float(np.mean([r.get(cid, 0.0) for r in cluster_mdi_ranks])) if cluster_mdi_ranks else 0.0
                    for cid in pool_ids}
    avg_mda_rank = {cid: float(np.mean([r.get(cid, 0.0) for r in cluster_mda_ranks])) if cluster_mda_ranks else 0.0
                    for cid in pool_ids}

    # ---- 簇级综合评分 ----
    total_w = w_ic + w_mdi + w_mda
    if total_w == 0:
        total_w = 1.0
    scores = {cid: (w_ic * avg_ic_rank[cid] + w_mdi * avg_mdi_rank[cid] + w_mda * avg_mda_rank[cid]) / total_w
              for cid in pool_ids}

    # ---- 簇级投票准入 ----
    has_mda = len(fold_cluster_mda) > 0 and any(d for d in fold_cluster_mda if d)
    if has_mda and use_voting:
        candidate_ids = [cid for cid in pool_ids if cluster_votes[cid] >= vote_min_folds]
    else:
        candidate_ids = list(pool_ids)

    # ---- 降级保护 ----
    effective_vote_min = vote_min_folds
    while len(candidate_ids) < vote_min_features and effective_vote_min > 1:
        prev_min = effective_vote_min
        effective_vote_min -= 1
        candidate_ids = [cid for cid in pool_ids if cluster_votes[cid] >= effective_vote_min]
        logger.warning("[CFI-Vote] 入选簇不足(%d<@%d个簇阈值), 放宽投票至 ≥%d 折 → %d 簇",
                       len(candidate_ids), vote_min_features, effective_vote_min, len(candidate_ids))

    if len(candidate_ids) < vote_min_features and fallback_to_ic:
        logger.warning("[CFI-Vote] 降级至纯 IC 排序（fallback_to_ic_only=true）")
        candidate_ids = sorted(pool_ids, key=lambda cid: avg_ic_rank.get(cid, 0), reverse=True)
        effective_vote_min = 0

    # ---- Top-K 截断（簇维度; tiebreaker: score→簇大小→索引, 大簇优先）----
    candidate_ids = sorted(candidate_ids,
                           key=lambda cid: (scores.get(cid, 0), len(clusters[cid]), -cid),
                           reverse=True)
    final_ids = candidate_ids[:top_k] if top_k and top_k > 0 else candidate_ids

    # ---- 日志 ----
    n_exact = {i: sum(1 for cid in pool_ids if cluster_votes[cid] == i) for i in range(n_folds + 1)}
    n_cumul = {i: sum(1 for cid in pool_ids if cluster_votes[cid] >= i) for i in range(1, n_folds + 1)}
    logger.info("[CFI-Vote] 簇级MDA投票分布 (共%d簇): 精确 %s | 累积 ≥3折=%d ≥2折=%d",
                len(pool_ids), {k: v for k, v in n_exact.items() if v}, n_cumul.get(3, 0), n_cumul.get(2, 0))
    for cid in final_ids:
        logger.info("  #簇[%d](%d特征): votes=%d/%d, score=%.4f, mda_rank=%.3f, members=%s",
                    cid, len(clusters[cid]), cluster_votes.get(cid, 0), n_folds,
                    scores.get(cid, 0), avg_mda_rank.get(cid, 0),
                    ','.join(clusters[cid]))
    logger.info("[CFI-Vote] 簇级Top-K: %d → %d 簇入选", len(candidate_ids), len(final_ids))

    report = {
        'selected_cluster_ids': final_ids,
        'cluster_scores': {cid: round(scores.get(cid, 0), 4) for cid in final_ids},
        'cluster_votes': {cid: int(cluster_votes.get(cid, 0)) for cid in final_ids},
        'cluster_mda_ranks': {cid: round(avg_mda_rank.get(cid, 0), 4) for cid in final_ids},
        'cluster_ic_ranks': {cid: round(avg_ic_rank.get(cid, 0), 4) for cid in final_ids},
        'cluster_members': {cid: list(clusters[cid]) for cid in final_ids},
        'vote_min_folds': effective_vote_min,
        'pool_size': len(pool_ids),
    }
    return final_ids, report


def _sfi_best_member(df_train, members, target_col, config, metric_fn, sample_weight=None,
                     objective_mode='binary'):
    """[Test-CFI] AFML §8.4.1 单特征 OOS 重要性 (SFI): 簇内每个成员单独训练时序模型,
    验证集 OOS 度量最高者为代表。单特征模型不存在相关特征 → 唯一不受替代效应影响的估计。

    Returns:
        Tuple[str, Dict[str, float]]: (代表特征, 各成员 SFI 得分)
    """
    import numpy as np
    import logging
    logger = logging.getLogger(__name__)

    if len(members) == 1:
        return members[0], {members[0]: 0.0}

    unique_dates = sorted(df_train['trade_date'].unique())
    n_dates = len(unique_dates)
    split = max(int(n_dates * 0.8), min(20, max(n_dates // 2, 5)))
    if n_dates - split < 5:
        split = max(n_dates - 5, 1)
    train_dates = unique_dates[:split]
    val_dates = unique_dates[split:]
    tr = df_train[df_train['trade_date'].isin(train_dates)]
    va = df_train[df_train['trade_date'].isin(val_dates)]

    params = {
        'max_depth': getattr(config, 'XGB_MAX_DEPTH', 4),
        'eta': getattr(config, 'XGB_LEARNING_RATE', 0.05),
        'subsample': getattr(config, 'XGB_SUBSAMPLE', 0.8),
        'colsample_bytree': 1.0,  # 单特征, 无需列采样
        'alpha': getattr(config, 'XGB_REG_ALPHA', 0.0),
        'lambda': getattr(config, 'XGB_REG_LAMBDA', 1.0),
        'min_child_weight': getattr(config, 'XGB_MIN_CHILD_WEIGHT', 1),
        'gamma': getattr(config, 'XGB_GAMMA', 0.0),
        'objective': 'reg:squarederror' if objective_mode == 'regression' else 'binary:logistic',
        'eval_metric': 'rmse' if objective_mode == 'regression' else 'auc',
        'seed': getattr(config, 'XGB_RANDOM_STATE', 42),
        'tree_method': getattr(config, 'XGB_TREE_METHOD', 'hist'),
        'device': getattr(config, 'XGB_DEVICE', 'cpu'),
        'nthread': 1,
    }
    n_rounds = getattr(config, 'XGB_N_ESTIMATORS', 300)
    esr = getattr(config, 'XGB_EARLY_STOPPING_ROUNDS', 50)

    scores = {}
    for f in members:
        try:
            wt = None
            if sample_weight is not None:
                wt = sample_weight[tr.index.values]
            dtrain = prepare_dmatrix(tr, [f], tr[target_col].values, weights=wt)
            dval = prepare_dmatrix(va, [f], va[target_col].values)
            model = xgb.train(params, dtrain, num_boost_round=n_rounds,
                              evals=[(dtrain, 'train'), (dval, 'val')],
                              early_stopping_rounds=esr, verbose_eval=False)
            pred = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
            scores[f] = float(metric_fn(va[target_col].values, pred))
        except Exception as e:
            logger.warning("[CFI-SFI] 特征 %s 单特征模型失败: %s", f, e)
            scores[f] = 0.0
    best = max(scores, key=scores.get)
    logger.info("[CFI-SFI] 簇内SFI代表: %s (scores=%s)", best,
                {k: round(v, 4) for k, v in sorted(scores.items(), key=lambda x: -x[1])})
    return best, scores


def select_cluster_reprs(df_train, selected_cluster_ids, clusters, config,
                         target_col='label', sample_weight=None, cluster_report=None,
                         objective_mode='binary'):
    """[Test-CFI] 簇内选代表（CFI 第二层: 簇级入选后的细分选择）。

    repr_method (config.cluster_repr_method):
      'sfi'       — 单特征 OOS 重要性 (AFML §8.4.1, 默认): 簇内每个成员单独训练时序模型,
                    验证集 OOS 度量最高者为代表。无相关特征参与 → 不受替代效应影响。
      'ic_max'    — 簇内成员跨折平均 |IC| 排名最高者（cluster_report.cluster_ic_ranks 为簇级,
                    此模式下用簇内成员折内 IC 的跨折均值; 需 fold_ic_results, 未提供则回退 sfi）
      'score_max' / 'mda_max' — CFI 模式下无特征级 score/MDA（评估单元是簇）,
                    自动回退 sfi 并告警。

    Returns:
        List[str]: 最终特征集（每簇 1 代表, 按入选簇顺序）
    """
    import logging
    logger = logging.getLogger(__name__)

    repr_method = getattr(config, 'cluster_repr_method', 'sfi')
    metric_fn = _objective_metric_fn(objective_mode)

    final = []
    repr_map = {}
    for cid in selected_cluster_ids:
        members = clusters[cid]
        if len(members) == 1:
            repr_map[cid] = members[0]
            final.append(members[0])
            continue
        if repr_method == 'ic_max':
            # 簇内成员折内 IC 跨折均值 (需 fold_ic_results 已在 cluster_report 中预聚合)
            ic_by_member = {}
            for f in members:
                ic_by_member[f] = 0.0
            # cluster_report.cluster_ic_ranks 为簇级, 无成员级信息 → 回退 sfi
            logger.warning("[CFI-Repr] ic_max 需要特征级折内IC数据, 回退 sfi")
            best, _ = _sfi_best_member(df_train, members, target_col, config, metric_fn,
                                       sample_weight, objective_mode)
        elif repr_method not in ('sfi',):
            logger.warning("[CFI-Repr] repr_method=%s 在CFI模式下不可用(评估单元为簇), 回退 sfi", repr_method)
            best, _ = _sfi_best_member(df_train, members, target_col, config, metric_fn,
                                       sample_weight, objective_mode)
        else:
            best, _ = _sfi_best_member(df_train, members, target_col, config, metric_fn,
                                       sample_weight, objective_mode)
        repr_map[cid] = best
        final.append(best)

    logger.info("[CFI-Repr] 簇内代表选择完成 (repr_method=%s): %d 簇 → %d 特征", repr_method,
                len(selected_cluster_ids), len(final))
    return final



def cluster_features(df_train, feature_set, vote_report, config):
    """特征聚类去冗余：层次聚类 on Spearman 相关矩阵，每簇选代表。

    Args:
        df_train: 训练集 DataFrame
        feature_set: 候选特征列表
        vote_report: Dict 来自 aggregate_fold_votes 的报告，含:
            'scores'        (综合评分: w_ic*rank_ic + w_mdi*rank_mdi + w_mda*rank_mda)
            'avg_mda_rank'  (各特征跨折平均 MDA 排名)
            'avg_ic_rank'   (各特征跨折平均 IC 排名)
            用于按 repr_method 选择簇内代表:
            - 'score_max' (默认): 用综合评分
            - 'mda_max': 用 OOS MDA 排名 (Prado AFML Ch.6 建议: 簇代表应选预测能力最强)
            - 'ic_max': 用 IC 排名
        config: 配置对象

    Returns:
        List[str]: 去冗余后的特征列表
    """
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from scipy.stats import spearmanr
    import logging
    logger = logging.getLogger(__name__)

    enabled = getattr(config, 'cluster_enabled', False)
    if not enabled:
        return feature_set

    if len(feature_set) < 3:
        logger.info("[FeatureClustering] 特征数<3，跳过聚类")
        return feature_set

    corr_threshold = getattr(config, 'cluster_corr_threshold', 0.70)
    corr_method = getattr(config, 'cluster_corr_method', 'spearman')
    repr_method = getattr(config, 'cluster_repr_method', 'score_max')
    min_cluster_size = getattr(config, 'cluster_min_size', 2)
    log_clusters = getattr(config, 'cluster_log', True)
    mda_threshold = getattr(config, 'mda_threshold', 0.001)

    logger.info("[FeatureClustering] 计算 %s 相关矩阵 (%d 特征)...", corr_method, len(feature_set))

    normal_feats = [f for f in feature_set if f != 'v10_pred_rank']
    X = df_train[normal_feats].values
    X = np.nan_to_num(X, nan=0.0)

    # 相关矩阵
    try:
        if corr_method == 'spearman':
            from scipy.stats import spearmanr
            corr = spearmanr(X)[0]
        else:
            corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
    except Exception as e:
        logger.warning("[FeatureClustering] 相关矩阵计算失败: %s，跳过聚类", e)
        return feature_set

    # 距离矩阵: 1 - abs(corr)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)

    # 层次聚类 (ward)
    try:
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method='ward')
    except Exception as e:
        logger.warning("[FeatureClustering] 层次聚类失败: %s，跳过聚类", e)
        return feature_set

    # 切割
    labels = fcluster(Z, t=corr_threshold, criterion='distance')

    # 构建簇
    clusters = {}
    for i, feat in enumerate(normal_feats):
        cid = int(labels[i])
        clusters.setdefault(cid, []).append(feat)

    # 每簇选代表
    representatives = []
    details = []
    for cid, members in sorted(clusters.items()):
        if len(members) < min_cluster_size:
            # 孤立特征，直接保留
            representatives.extend(members)
            continue

        if repr_method == 'mda_max':
            # OOS MDA 排名 (Prado Ch.6: 簇代表应选对预测贡献最大的特征)
            best = max(members, key=lambda f: vote_report.get('avg_mda_rank', {}).get(f, 0.0))
        elif repr_method == 'ic_max':
            # IC 排名
            best = max(members, key=lambda f: vote_report.get('avg_ic_rank', {}).get(f, 0.0))
        else:  # score_max (默认): 综合评分
            best = max(members, key=lambda f: vote_report.get('scores', {}).get(f, 0.0))
        representatives.append(best)
        if log_clusters:
            others = [m for m in members if m != best]
            details.append((best, others, len(members)))

    final = representatives + (['v10_pred_rank'] if 'v10_pred_rank' in feature_set else [])

    n_before = len(feature_set)
    n_after = len(final)
    n_clusters = sum(1 for cid, members in clusters.items() if len(members) >= min_cluster_size)

    logger.info("[FeatureClustering] 聚类结果 (corr_threshold=%.2f):", corr_threshold)
    logger.info("  - 聚类前: %d 个特征", n_before)
    logger.info("  - 发现 %d 个簇 (大小≥%d)", n_clusters, min_cluster_size)
    logger.info("  - 合并后: %d 个特征 (保留代表 + 孤立特征)", n_after)

    if log_clusters and details:
        logger.info("[FeatureClustering] 各簇代表:")
        for best, others, size in details:
            logger.info("  - %s (代替: %s, 簇大小=%d)", best, ', '.join(others[:3]), size)

    return final


def cluster_features_global(df_train, feature_set, config):
    """[Test-CFI] 全局特征聚类（CFI 前置步骤, Prado MLAM Ch.6）: 只分组不选代表。

    与 cluster_features 共享同一套距离/链接/切割逻辑:
        Spearman 相关矩阵 → 距离 1-|corr| → ward 层次聚类 → fcluster(criterion='distance')
    在 CV 之前对预处理后的全特征集聚类 → 簇结构跨折稳定 → 每折按"折内特征∩簇成员"
    做簇级置换 MDA → 簇级投票对齐。

    Args:
        df_train: 训练数据 (含特征列)
        feature_set: 候选特征列表（通常为 CV 预处理后的 pre_feats）
        config: 配置 (cluster_enabled / cluster_corr_threshold / cluster_corr_method /
                     cluster_min_size / cluster_log)

    Returns:
        List[List[str]]: 簇结构（每簇为特征成员列表, 含单例簇; v10_pred_rank 恒独立成簇）
        聚类失败/未启用 → 全部单例簇（等价逐特征流程, 安全降级）
    """
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from scipy.stats import spearmanr
    import logging
    logger = logging.getLogger(__name__)

    enabled = getattr(config, 'cluster_enabled', False)
    if not enabled or len(feature_set) < 3:
        return [[f] for f in feature_set]

    corr_threshold = getattr(config, 'cluster_corr_threshold', 0.70)
    corr_method = getattr(config, 'cluster_corr_method', 'spearman')
    min_cluster_size = getattr(config, 'cluster_min_size', 2)
    log_clusters = getattr(config, 'cluster_log', True)

    normal_feats = [f for f in feature_set if f != 'v10_pred_rank']
    X = df_train[normal_feats].values
    X = np.nan_to_num(X, nan=0.0)

    try:
        if corr_method == 'spearman':
            corr = spearmanr(X)[0]
        else:
            corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
    except Exception as e:
        logger.warning("[CFI-Cluster] 相关矩阵计算失败: %s, 全部按单例簇", e)
        return [[f] for f in feature_set]

    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    try:
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method='ward')
        # 距离阈值语义 (config.cluster_legacy_threshold 控制):
        #   标准(默认): t = 1 - corr_threshold → |corr| ≥ corr_threshold(0.70) 归簇
        #                (与 Prado MLAM 相关阈值语义一致)
        #   legacy  : t = corr_threshold → 等效 |corr| ≥ 1-corr_threshold(0.30) 归簇
        #                (与原版 cluster_features 行为一致, 激进压缩;
        #                 供 C 组对照实验隔离"阈值语义"变量, 见因子管线报告 13.5)
        legacy = bool(getattr(config, 'cluster_legacy_threshold', False))
        t_dist = corr_threshold if legacy else 1.0 - corr_threshold
        labels = fcluster(Z, t=t_dist, criterion='distance')
    except Exception as e:
        logger.warning("[CFI-Cluster] 层次聚类失败: %s, 全部按单例簇", e)
        return [[f] for f in feature_set]

    clusters_dict = {}
    for i, feat in enumerate(normal_feats):
        clusters_dict.setdefault(int(labels[i]), []).append(feat)

    # 稳定排序: 按簇内首特征在 feature_set 中的原始索引
    idx_map = {f: i for i, f in enumerate(feature_set)}
    clusters = sorted(clusters_dict.values(), key=lambda c: min(idx_map[f] for f in c))

    if 'v10_pred_rank' in feature_set:
        clusters.append(['v10_pred_rank'])

    n_multi = sum(1 for c in clusters if len(c) >= min_cluster_size)
    n_single = sum(1 for c in clusters if len(c) == 1)
    logger.info("[CFI-Cluster] 全局聚类完成 (corr_threshold=%.2f, %s): %d 特征 → %d 簇 "
                "(≥%d成员簇=%d, 单例簇=%d)",
                corr_threshold, corr_method, len(feature_set), len(clusters),
                min_cluster_size, n_multi, n_single)
    if log_clusters:
        for c in sorted(clusters, key=lambda x: -len(x)):
            if len(c) > 1:
                logger.info("  - 簇(%d): %s", len(c), ', '.join(c))
    return clusters


def cfi_diagnose(models, X_val_df, y_val, selected_features, config,
                 output_path=None, metric_fn=None, n_repeats=20, random_state=42):
    """[CFI-诊断] 生产模型训练后: 簇级置换重要性分布 (Prado MLAM Ch.6 clustered MDA)。

    用途（诊断工具, 不参与特征选择/预测决策）:
      - 模型劣化归因: 模型 AUC 崩塌/回撤异常时, 定位模型依赖的特征簇是否切换/失效
      - 市场结构切换监控: 跨周期对比簇级重要性占比分布, 预警模型换了一套逻辑

    机制:
      1) 聚类: cluster_features_global (标准语义 |corr|≥0.70 归簇)
      2) 置换: 簇整体置换 (簇内全部特征同时打乱 → 相关替代特征无法填补 →
                簇级重要性不受簇内替代效应污染)
      3) 预测: 模型集成平均 (与生产预测口径一致)
      4) 占比: share_i = max(mda_i, 0) / Σ max(mda_j, 0) → "模型 X% 依赖该簇"

    输出: CSV (cluster_id, n_members, members, cluster_mda, mda_share) + 日志。
    纯增量输出: 不改变任何训练/特征选择/预测行为 (调用方可配置开关)。

    Returns:
        Optional[pd.DataFrame]: 簇级重要性分布 (未聚类/异常时 None)
    """
    import numpy as np
    import threading
    import logging
    from collections import defaultdict
    logger = logging.getLogger(__name__)

    if metric_fn is None:
        metric_fn = _objective_metric_fn('binary')
    if len(selected_features) < 2:
        logger.info("[CFI-Diag] 特征数<2, 跳过诊断")
        return None
    if not models:
        logger.warning("[CFI-Diag] 无模型, 跳过诊断")
        return None

    # ---- 1) 聚类 (特征顺序 = 生产模型特征顺序, 保证 DMatrix 与模型一致) ----
    X_df = X_val_df[selected_features]
    clusters = cluster_features_global(X_df, selected_features, config)
    members = list(selected_features)
    feat_to_idx = {f: i for i, f in enumerate(members)}
    X_base = X_df.replace([np.inf, -np.inf], np.nan).values.astype(np.float32)

    # ---- 2) 基线: 集成预测 ----
    def _build_dmat(X_ws):
        return xgb.DMatrix(X_ws, label=y_val, missing=np.nan, feature_names=members)

    def _ensemble_predict(X_ws):
        d = _build_dmat(X_ws)
        return np.mean([m.predict(d) for m in models], axis=0)

    try:
        baseline = float(metric_fn(y_val, _ensemble_predict(X_base)))
    except ValueError:
        logger.warning("[CFI-Diag] 基线度量失败, 跳过诊断")
        return None

    # ---- 3) 簇级置换任务 (簇内全部列同时逐列 shuffle) ----
    rng = np.random.RandomState(random_state)
    tasks = []
    for cid, cluster in enumerate(clusters):
        idxs = [feat_to_idx[f] for f in cluster if f in feat_to_idx]
        if not idxs:
            continue
        for _ in range(n_repeats):
            cols = []
            for idx in idxs:
                col = X_base[:, idx].copy()
                rng.shuffle(col)
                cols.append((idx, col))
            tasks.append((cid, cols))

    _thread_local = threading.local()

    def _eval_one(task):
        cid, cols = task
        if not hasattr(_thread_local, 'X_ws'):
            _thread_local.X_ws = X_base.copy()
        _thread_local.X_ws[:] = X_base
        for idx, col in cols:
            _thread_local.X_ws[:, idx] = col
        try:
            perm_metric = float(metric_fn(y_val, _ensemble_predict(_thread_local.X_ws)))
            return cid, baseline - perm_metric
        except ValueError:
            return cid, 0.0

    if len(tasks) <= 20:
        results = [_eval_one(t) for t in tasks]
    else:
        from concurrent.futures import ThreadPoolExecutor
        max_workers = min(8, os.cpu_count() or 8)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_eval_one, tasks))

    # ---- 4) 聚合 + 占比归一化 ----
    drops = defaultdict(list)
    for cid, v in results:
        drops[cid].append(v)
    mda = {}
    for cid in range(len(clusters)):
        d = drops.get(cid, [])
        mda[cid] = float(np.mean(d)) if d else 0.0
    pos_sum = sum(v for v in mda.values() if v > 0)
    share = {cid: (mda[cid] / pos_sum if pos_sum > 0 else 0.0) for cid in mda}

    rows = []
    for cid in sorted(mda, key=lambda c: -mda[c]):
        rows.append({
            'cluster_id': cid,
            'n_members': len(clusters[cid]),
            'members': ';'.join(clusters[cid]),
            'cluster_mda': round(mda[cid], 6),
            'mda_share': round(share[cid], 6),
        })
    df_out = pd.DataFrame(rows)

    # ---- 5) 输出 ----
    if output_path:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_out.to_csv(output_path, index=False, encoding='utf-8-sig')
        except Exception as e:
            logger.warning("[CFI-Diag] 写文件失败: %s", e)

    logger.info("[CFI-Diag] 簇级重要性分布 (基线=%.4f, %d簇, n_repeats=%d):",
                baseline, len(clusters), n_repeats)
    for r in rows:
        logger.info("  #簇%d(%d特征, share=%.1f%%): %s",
                    r['cluster_id'], r['n_members'], r['mda_share'] * 100, r['members'])
    return df_out


class EnsembleModelManager:
    def __init__(self, model_dir='./models', config=None):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        if config is None:
            raise ValueError("必须提供配置对象 config")
        self.config = config
        self.selected_features = []
        self.feature_scores = {}
        self.feature_max_ic = 0.0
        self.xgb_model = None
        self.xgb_models = []          # [Step2a] 多种子模型列表
        self.ensemble_seeds = []       # [Step2a] 使用的种子列表
        self.xgb_threshold = None
        self.xgb_train_prec = None
        self.xgb_val_prec = None
        self.xgb_mean_return = None
        self.xgb_p_value = None
        self.xgb_train_p_value = None
        self.xgb_val_auc = None
        self.best_model = None
        self.best_threshold = None
        self.best_model_type = 'xgb'
        self.best_p_value = None
        self.best_train_p_value = None
        self.best_val_auc = None
        self.validation_mean_return = None
        self.validation_auc = None
        self.calibrator = None
        self.prob_calibrator = None  # P1-B: Isotonic概率校准器实例

        # [v3.5] LGBM 异质集成 (对齐V10)
        lgb_cfg = getattr(config, 'lightgbm', None) if hasattr(config, 'lightgbm') else config.get('lightgbm', {}) if isinstance(config, dict) else {}
        self._lgb_enabled = lgb_cfg.get('enabled', True) if lgb_cfg else False
        self.lgb_models = []
        self._lgb_w_xgb = lgb_cfg.get('ensemble_weight_xgb', 0.6) if lgb_cfg else 0.6
        self._lgb_w_lgb = lgb_cfg.get('ensemble_weight_lgb', 0.4) if lgb_cfg else 0.4
        self._lgb_params = lgb_cfg.get('params', {}) if lgb_cfg else {}

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(data)



    # ========== [P0-0] 超参候选选择 (轻量版) ==========
    def _select_hyperparams(self, df_train, feature_cols, target_col, sample_weight=None):
        """[超参选择] 用时序末 val 段评估候选超参, 按验证度量选最优并写回 self.config。

        轻量版实现 (候选有限集 + 内层多窗 walk-forward 评估):
          - 候选来自 config.hyperparam_search.candidates (4组: 保守→强表达)
          - [A] 选择-评估隔离: 剔除与生产模型相同的末 prod_val_size 天 (max_hold 规则),
            仅在其前训练段内做选择 → 生产 val 段零接触, 消除选择偏倚 (Cawley-Talbot 嵌套精神)
          - [B] 多窗评估: 训练段最后 eval_windows×inner_val_days 天切 W 个连续窗,
            每窗 train = 该窗之前全部数据 (expanding walk-forward, 训练严格早于评估)
          - [C] 每候选每窗独立 early stopping (轮数上限 = 候选 n_estimators),
            得分 = W 窗验证度量平均 (AUC/IC), 消除固定轮数对 lr 不同候选的偏置
          - 评估特征: 训练段内 IC/PSI 筛选 (与 PKF 折内一致, 只吃训练数据)
          - 选最优 → 写回 self.config 的 XGB_* 属性, PKF 折内/生产模型自动使用
          - 仅由 train() 在 pre_feats 之后调用 (重训周期天然触发, 非重训零开销)
          - enabled=false / 候选为空 / 评估全失败 / 全部候选 AUC<0.5 → (None, None) 不改配置

        Returns:
            (best_name, best_score): 选中候选名与验证度量; 未启用/失败返回 (None, None)
        """
        hp_cfg = getattr(self.config, 'hyperparam_search', None)
        if not hp_cfg or not hp_cfg.get('enabled', False):
            return None, None
        candidates = hp_cfg.get('candidates', [])
        if not candidates:
            logger.warning("[HP-Search] hyperparam_search.candidates 为空, 跳过超参选择")
            return None, None

        obj_mode = getattr(self, 'objective_mode',
                           _detect_objective_mode(df_train[target_col].values, self.config))
        metric_fn = _objective_metric_fn(obj_mode)
        _metric_name = 'IC' if obj_mode == 'regression' else 'AUC'
        es_rounds = int(hp_cfg.get('early_stopping_rounds', 20))
        use_es = bool(hp_cfg.get('use_early_stop', True))
        eval_rounds_cap = int(hp_cfg.get('eval_rounds', 600))
        eval_windows = max(1, int(hp_cfg.get('eval_windows', 3)))
        inner_val_days = max(5, int(hp_cfg.get('inner_val_days', 20)))

        # ============================================================
        # [A] 选择-评估隔离: 剔除生产 val 段 (与生产 prod_val_size 同规则)
        # ============================================================
        all_dates = sorted(df_train['trade_date'].unique())
        n_dates = len(all_dates)
        max_hold = getattr(self.config, 'MAX_HOLD_DAYS', 20)
        prod_val_size = min(max_hold, max(n_dates // 5, 1))
        sel_dates = all_dates[:-prod_val_size] if prod_val_size > 0 else all_dates
        n_sel = len(sel_dates)
        if n_sel < 60:
            logger.warning("[HP-Search] 剔除生产val后训练数据不足 (%d 天), 跳过超参选择", n_sel)
            return None, None
        logger.info("[HP-Search] [A] 选择数据隔离: 剔除生产val %d 天, 仅用 %d 天 (生产val零接触)",
                    prod_val_size, n_sel)

        # ---- 评估特征: 训练段(sel_dates)内 IC/PSI 筛选 (与 PKF 折内一致) ----
        eval_feats = list(feature_cols)
        try:
            mid_raw = max(int(n_sel * getattr(self.config, 'cv_ic_psi_split_ratio', 0.5)), 10)
            mid = min(mid_raw, n_sel - 5)
            ic_set = df_train[df_train['trade_date'].isin(sel_dates[:mid])]
            psi_set = df_train[df_train['trade_date'].isin(sel_dates[mid:])]
            eval_feats, _, _, _ = select_stable_features(
                ic_set, psi_set, feature_cols, self.config,
                target_col=target_col, skip_preprocess=True,
            )
        except Exception as e:
            logger.warning("[HP-Search] 评估特征筛选失败(%s), 降级用全部 %d 特征", e, len(eval_feats))
        if not eval_feats:
            logger.warning("[HP-Search] 筛选后无特征, 降级用全部 %d 特征", len(feature_cols))
            eval_feats = list(feature_cols)
        logger.info("[HP-Search] 候选评估特征: %d 个 (训练段内 IC/PSI 筛选, 与 PKF 一致)",
                    len(eval_feats))

        # ============================================================
        # [B] 多窗 expanding walk-forward: 最后 W×inner_val_days 天切 W 个连续窗
        # ============================================================
        n_win_total = eval_windows * inner_val_days
        if n_sel < n_win_total:
            eval_windows = max(1, n_sel // (inner_val_days * 2))
            n_win_total = eval_windows * inner_val_days
            logger.warning("[HP-Search] 训练天数不足, 窗数降为 %d", eval_windows)
        win_dates = sel_dates[-n_win_total:]
        windows = []  # [(train_dates_set, val_dates_set)]
        # [O7-20260904] 窗间 embargo: 训练段末尾剔除 max_hold 天 (标签窗口重叠隔离,
        #   与 PKF purge/embargo 同精神)。原实现 train 紧贴 val 起始 → 末段训练样本
        #   的标签窗口 (特征日+20天) 与验证窗重叠 → HP 选择偏乐观。
        _hp_embargo = max(int(getattr(self.config, 'MAX_HOLD_DAYS', 20)), 1)
        for w in range(eval_windows):
            v_start = w * inner_val_days
            v_dates = set(win_dates[v_start:v_start + inner_val_days])
            # expanding: train = 该窗开始之前的所有 sel 数据 (训练严格早于评估)
            #            再向前推 _hp_embargo 天 → 验证窗与训练标签窗口零重叠
            _t_end_excl = n_sel - n_win_total + v_start - _hp_embargo
            if _t_end_excl <= 0:
                logger.warning("[HP-Search] embargo 后训练集为空 (窗%d), 跳过该窗", w)
                continue
            t_dates = set(sel_dates[:_t_end_excl])
            windows.append((t_dates, v_dates))
        logger.info("[HP-Search] [B] 多窗评估: %d 窗 × %d 天 (expanding, 每窗train=%d~%d天)",
                    eval_windows, inner_val_days,
                    min(len(w[0]) for w in windows), max(len(w[0]) for w in windows))

        # XGB 参数 → config 属性 映射 (写回后 PKF/生产模型自动生效)
        hp_to_cfg = {
            'xgb_n_estimators': 'XGB_N_ESTIMATORS',
            'xgb_max_depth': 'XGB_MAX_DEPTH',
            'xgb_learning_rate': 'XGB_LEARNING_RATE',
            'xgb_subsample': 'XGB_SUBSAMPLE',
            'xgb_col_sample_by_tree': 'XGB_COL_SAMPLE_BY_TREE',
            'xgb_reg_alpha': 'XGB_REG_ALPHA',
            'xgb_reg_lambda': 'XGB_REG_LAMBDA',
            'xgb_min_child_weight': 'XGB_MIN_CHILD_WEIGHT',
            'xgb_gamma': 'XGB_GAMMA',
        }
        int_keys = {'xgb_n_estimators', 'xgb_max_depth', 'xgb_min_child_weight'}
        # [2026-08-18 修复] 候选配置键(xgb_*前缀) → xgboost 参数名 映射
        # BUG: 原实现直接用 'xgb_max_depth' 等键名传给 xgb.train → xgboost 静默忽略全部候选超参
        #      (日志 WARNING: Parameters are not used) → 4 组候选全用默认参数 → AUC/best_iter 全同
        xgb_param_key = {
            'xgb_max_depth': 'max_depth',
            'xgb_learning_rate': 'eta',
            'xgb_subsample': 'subsample',
            'xgb_col_sample_by_tree': 'colsample_bytree',
            'xgb_reg_alpha': 'alpha',
            'xgb_reg_lambda': 'lambda',
            'xgb_min_child_weight': 'min_child_weight',
            'xgb_gamma': 'gamma',
        }
        base_params = {
            'objective': 'reg:squarederror' if obj_mode == 'regression' else 'binary:logistic',
            'eval_metric': 'rmse' if obj_mode == 'regression' else 'auc',
            'seed': getattr(self.config, 'XGB_RANDOM_STATE', 42),
            'tree_method': getattr(self.config, 'XGB_TREE_METHOD', 'hist'),
            'device': getattr(self.config, 'XGB_DEVICE', 'cpu'),
            'nthread': 1,
        }

        _t0 = time.time()
        results = []        # (name, mean_score, mean_rounds)
        detail = {}         # name -> {'scores': [...], 'rounds': [...]}
        for cand in candidates:
            name = str(cand.get('name', 'c?'))
            missing_keys = [k for k in hp_to_cfg if k not in cand]
            if missing_keys:
                logger.warning("[HP-Search] 候选 %s 缺参数 %s, 跳过", name, missing_keys)
                continue
            params = dict(base_params)
            for ck, xk in xgb_param_key.items():
                params[xk] = int(cand[ck]) if ck in int_keys else float(cand[ck])
            cap = max(int(cand['xgb_n_estimators']), 50)
            cap = min(cap, eval_rounds_cap)
            # 每窗样本权重对齐: 训练段日期 → 索引掩码
            win_scores = []
            win_rounds = []
            win_ok = True
            for (t_dates, v_dates) in windows:
                t_mask = df_train['trade_date'].isin(t_dates)
                v_mask = df_train['trade_date'].isin(v_dates)
                n_t, n_v = int(t_mask.sum()), int(v_mask.sum())
                if n_t < 100 or n_v < 50:
                    logger.warning("[HP-Search] 候选 %s 窗样本不足(train=%d,val=%d), 跳过",
                                   name, n_t, n_v)
                    win_ok = False
                    break
                X_t = df_train.loc[t_mask, eval_feats].replace([np.inf, -np.inf], np.nan).astype(np.float32)
                y_t = df_train.loc[t_mask, target_col].values
                X_v = df_train.loc[v_mask, eval_feats].replace([np.inf, -np.inf], np.nan).astype(np.float32)
                y_v = df_train.loc[v_mask, target_col].values
                wt_t = None
                if sample_weight is not None:
                    wt_t = np.asarray(sample_weight)[t_mask.values]
                dtrain = xgb.DMatrix(X_t, label=y_t, weight=wt_t, missing=np.nan,
                                     feature_names=eval_feats)
                dval = xgb.DMatrix(X_v, label=y_v, missing=np.nan, feature_names=eval_feats)
                try:
                    if use_es:
                        # [C 修复 2026-08-18] 评估协议与生产一致: 加 LRWarmup (生产训练带 warmup)
                        # 原评估无 warmup → 选中超参在"纯eta"协议下评估, 生产却在 warmup 协议下用 → 错配
                        model = xgb.train(
                            params, dtrain, num_boost_round=cap,
                            evals=[(dtrain, 'train'), (dval, 'val')],
                            early_stopping_rounds=es_rounds, verbose_eval=False,
                            callbacks=[LRWarmupCallback(
                                target_lr=float(cand['xgb_learning_rate']))],
                        )
                        rounds = int(getattr(model, 'best_iteration', cap))
                    else:
                        # legacy fallback: 固定轮数
                        n_rounds = min(cap, 80)
                        model = xgb.train(params, dtrain, num_boost_round=n_rounds,
                                          verbose_eval=False,
                                          callbacks=[LRWarmupCallback(
                                              target_lr=float(cand['xgb_learning_rate']))])
                        rounds = model.num_boosted_rounds()
                    win_scores.append(float(metric_fn(y_v, model.predict(dval))))
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
            logger.info("[HP-Search] 候选 %s: val %s=%.4f (W%d窗: %s, rounds=%s) "
                        "[depth=%d lr=%.4f ss=%.2f cs=%.2f l2=%.2f mcw=%d γ=%.3f]",
                        name, _metric_name, mean_score, len(win_scores),
                        ', '.join('%.4f' % s for s in win_scores),
                        win_rounds,
                        int(cand['xgb_max_depth']), float(cand['xgb_learning_rate']),
                        float(cand['xgb_subsample']), float(cand['xgb_col_sample_by_tree']),
                        float(cand['xgb_reg_lambda']), int(cand['xgb_min_child_weight']),
                        float(cand['xgb_gamma']))

        if not results:
            logger.warning("[HP-Search] 全部候选评估失败, 保持原超参")
            return None, None

        # [2026-08-18 防护] binary 模式下全部候选平均 AUC<0.5 → 评估段信号异常/特征反向
        # (如 2026-08-18 日志: 未筛选161特征+弱信号 → 全部 0.462-0.470 反向)
        # → 不写回超参, 保持原配置, 避免把"反向模型"参数带入生产
        if obj_mode == 'regression':
            _sane = True
        else:
            _sane = any(s > 0.5 for _, s, _ in results)
        if not _sane:
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
        for k, attr in hp_to_cfg.items():
            if k in best_cand:
                setattr(self.config, attr, best_cand[k])
        logger.info("[HP-Search] 选中候选 %s: %s=%.4f (共%d组: %s), 耗时 %.1fs → 已写回 XGB_* config",
                    best_name, _metric_name, best_score, len(results),
                    ', '.join('%s=%.4f' % (n, s) for n, s, _ in results),
                    time.time() - _t0)
        return best_name, best_score

    def _purged_kfold_cv(self, df_train, feature_cols, target_col, sample_weight,
                         current_date=None, cfi_clusters=None):
        """
        嵌套 Purged K-Fold CV + 折内 IC+PSI 特征选择。

        Prado AFML §7.4 标准 Purged K-Fold: K折, 每折一个验证块, 其余K-1块合为训练集
        （验证块两侧各embargo天被剔除, 消除标签重叠导致的信息泄露）。
        与expanding window不同：训练集包含验证块之后的数�? 利用完整样本。
        每折:
          1. 训练数据按时间分两半: 前半计算 RankIC, 后半做 PSI 对比
          2. select_stable_features(前半, 后半) → 折内特征集
          3. 单种子 XGBoost 训练 + 验证块评估

        跨折共识: 特征出现在 ≥ cv_feature_consensus 折中 → 生产特征集

        Returns:
            meta_auc: 各折 Val AUC 均值
            consensus_features: 跨折共识特征列表
        """
        K = getattr(self.config, 'cv_folds', 5)
        embargo = getattr(self.config, 'cv_embargo_days', 6)
        consensus_n = getattr(self.config, 'cv_feature_consensus', 3)

        # [L2] 目标模式: binary(二分类) / regression(分位回归), 度量随之切换
        obj_mode = _detect_objective_mode(df_train[target_col].values, self.config)
        metric_fn = _objective_metric_fn(obj_mode)
        _metric_name = 'IC' if obj_mode == 'regression' else 'AUC'

        logger.info("=" * 60)
        logger.info("【嵌套 Purged K-Fold CV】K=%d, embargo=%d天, 目标模式=%s", K, embargo, obj_mode)

        unique_dates = sorted(df_train['trade_date'].unique())
        n_dates = len(unique_dates)
        block_size = n_dates // K
        logger.info("总交易日=%d, 每块≈%d天, 共%d折 (Prado Purged K-Fold)",
                     n_dates, block_size, K)

        fold_aucs = []
        fold_features = []  # list of set

        # v2.0: 收集每折的 IC/MDI/MDA 用于多折投票
        use_voting = getattr(self.config, 'use_voting', True)
        mda_enabled = getattr(self.config, 'mda_enabled', True) and use_voting
        fold_ic_results = []
        fold_mdi_results = []
        fold_mda_results = []
        fold_mda_stds = []
        fold_cluster_mda_results = []   # [Test-CFI] 簇级 MDA (cfi_clusters 非 None 时收集)
        fold_cluster_mda_stds = []
        fold_defs = []  # [Part C] save fold (train_dates, val_dates) for CV ensemble

        # 单种子参数用于CV评估
        base_params = {
            'max_depth': self.config.XGB_MAX_DEPTH,
            'eta': self.config.XGB_LEARNING_RATE,
            'subsample': self.config.XGB_SUBSAMPLE,
            'colsample_bytree': self.config.XGB_COL_SAMPLE_BY_TREE,
            'alpha': self.config.XGB_REG_ALPHA,
            'lambda': self.config.XGB_REG_LAMBDA,
            'min_child_weight': self.config.XGB_MIN_CHILD_WEIGHT,
            'gamma': self.config.XGB_GAMMA,
            # [L2] 回归模式: squarederror + rmse; 二分类: logistic + auc
            'objective': 'reg:squarederror' if obj_mode == 'regression' else 'binary:logistic',
            'eval_metric': 'rmse' if obj_mode == 'regression' else 'auc',
            'seed': self.config.XGB_RANDOM_STATE,
            'tree_method': getattr(self.config, 'XGB_TREE_METHOD', 'hist'),
            'device': getattr(self.config, 'XGB_DEVICE', 'cpu'),
            'nthread': 1,
        }

        # Prado AFML §7.4 Purged K-Fold: 每块轮流做验证集
        # 训练集 = 所有非验证块 (验证块前后各embargo天被purge)
        for i in range(K):
            # 验证块区间
            val_start_idx = i * block_size
            val_end_idx = min(val_start_idx + block_size, n_dates)
            val_dates = unique_dates[val_start_idx:val_end_idx]

            # 训练块: 所有块 − [i−1, i, i+1]（前后各embargo天）
            purge_start_idx = max(0, val_start_idx - embargo)
            purge_end_idx = min(val_end_idx + embargo, n_dates)

            train_dates = [d for j, d in enumerate(unique_dates)
                           if j < purge_start_idx or j >= purge_end_idx]

            if len(train_dates) < 50 or len(val_dates) < 20:
                logger.warning(
                    "Fold %d/%d: 数据不足 (train=%dd, val=%dd), 跳过",
                    i + 1, K, len(train_dates), len(val_dates))
                continue

            # [Part C] 保存折定义用于 CV 折集成
            fold_defs.append((train_dates, val_dates))

            train_set = df_train[df_train['trade_date'].isin(train_dates)].copy()
            val_set = df_train[df_train['trade_date'].isin(val_dates)].copy()

            if len(train_set) < 100 or len(val_set) < 50:
                logger.warning(
                    "Fold %d/%d: 样本不足 (train=%d, val=%d), 跳过",
                    i + 1, K, len(train_set), len(val_set))
                continue

            logger.info("Fold %d/%d: train=%dd(%d样本) val=%dd(%d样本) [前后purge各%dd]",
                        i + 1, K, len(train_dates), len(train_set),
                        len(val_dates), len(val_set), embargo)

            # ---- [P0-1] 折内再切 inner-OOS (MDA 评估段; 验证块只评 AUC) ----
            # 嵌套原则 (Cawley-Talbot): 特征选择与重要性评估只接触训练部分, 验证块零接触
            # 原实现: MDA 在验证块上算 → 验证集参与特征选择 → 选择偏倚 → meta_auc 乐观
            train_unique = sorted(train_set['trade_date'].unique())
            if len(train_unique) < 10:
                logger.warning("Fold %d/%d: 训练日期过少, 跳过", i + 1, K)
                continue
            _frac = float(getattr(self.config, 'mda_inner_oos_frac', 0.15))
            _min_d = int(getattr(self.config, 'mda_inner_oos_min_days', 20))
            inner_oos_size = min(len(train_unique) // 2,
                                 max(_min_d, int(len(train_unique) * _frac)))
            inner_oos_dates = set(train_unique[-inner_oos_size:])
            inner_train_dates = set(train_unique[:-inner_oos_size])
            inner_train_set = train_set[train_set['trade_date'].isin(inner_train_dates)]
            inner_oos_set = train_set[train_set['trade_date'].isin(inner_oos_dates)]
            if len(inner_train_set) < 100 or len(inner_oos_set) < 30:
                logger.warning("Fold %d/%d: inner-OOS 切分样本不足(train=%d,oos=%d), 跳过",
                               i + 1, K, len(inner_train_set), len(inner_oos_set))
                continue
            logger.info("  [P0-1] inner-train=%dd(%d样本) inner-OOS=%dd(%d样本) (MDA评估段)",
                        len(inner_train_dates), len(inner_train_set),
                        len(inner_oos_dates), len(inner_oos_set))

            # ---- 折内 IC+PSI 特征选择（仅用 inner-train, 验证块/内层OOS 零接触）----
            ic_unique = sorted(inner_train_set['trade_date'].unique())
            mid_raw = max(int(len(ic_unique) * getattr(self.config, 'cv_ic_psi_split_ratio', 0.5)), 10)
            mid = min(mid_raw, len(ic_unique) - 5)  # 保证PSI子集至少5天
            ic_dates = ic_unique[:mid]
            psi_dates = ic_unique[mid:]

            ic_set = inner_train_set[inner_train_set['trade_date'].isin(ic_dates)]
            psi_set = inner_train_set[inner_train_set['trade_date'].isin(psi_dates)]

            logger.info("  IC子集=%dd, PSI子集=%dd", len(ic_dates), len(psi_dates))

            fold_feats, _, _, ic_scores_fold = select_stable_features(
                ic_set, psi_set, feature_cols, self.config,
                target_col=target_col, current_date=current_date,
                skip_preprocess=True
            )
            if use_voting:
                fold_ic_results.append(ic_scores_fold)

            if len(fold_feats) == 0:
                logger.warning("Fold %d/%d: 特征选择后无特征, 跳过", i + 1, K)
                continue
            fold_feats = sorted(fold_feats)
            fold_features.append(set(fold_feats))
            logger.info("Fold %d/%d: 选入 %d 个特征", i + 1, K, len(fold_feats))

            # ---- 折内模型训练 (inner-train) + 验证块评估 ----
            y_tr = inner_train_set[target_col].values
            y_va = val_set[target_col].values

            n_pos = int(np.sum(y_tr == 1))
            n_neg = int(np.sum(y_tr == 0))
            spw = float(n_neg / max(n_pos, 1))

            fold_params = dict(base_params)
            # [L2] scale_pos_weight 仅二分类适用
            if obj_mode != 'regression' and getattr(self, 'class_balance_mode', 'auto') != 'none':
                fold_params['scale_pos_weight'] = spw

            # [2026-08-05 修复] 折内模型带样本权重 (与生产模型/CV折集成同口径):
            # 平均唯一性(重叠惩罚) + PT/SL标签质量加权应作用于特征选择与折内评估
            wt_fold = None
            if sample_weight is not None:
                wt_fold = sample_weight[inner_train_set.index.values]
            dtrain = prepare_dmatrix(inner_train_set, fold_feats, y_tr, weights=wt_fold)
            dval = prepare_dmatrix(val_set, fold_feats, y_va)

            # [O3-20260904] Early stopping 用折内 inner-val (inner_oos 时间尾段),
            # 外层验证块 dval 只做最终 OOS 评估 → 上报 meta_auc 不再被
            # "早停轮数选择"污染 (原: 早停+评估共用同一验证块 → AUC 乐观)。
            # inner_oos 无 max_hold 隔离的残余重叠由其自身为时间尾段近似吸收
            # (inner-train 严格早于 inner-oos; 特征日-标签窗重叠在 CV 内属常规)。
            y_es = inner_oos_set[target_col].values
            des = prepare_dmatrix(inner_oos_set, fold_feats, y_es)

            model = xgb.train(
                fold_params, dtrain,
                num_boost_round=self.config.XGB_N_ESTIMATORS,
                evals=[(dtrain, 'train'), (des, 'es_val')],
                early_stopping_rounds=self.config.XGB_EARLY_STOPPING_ROUNDS,
                verbose_eval=False
            )

            val_proba = model.predict(dval)
            # [L2] 统一度量 (binary→AUC, regression→Spearman IC)
            auc_val = float(metric_fn(y_va, val_proba))

            fold_aucs.append(auc_val)
            logger.info("Fold %d/%d: Val %s=%.4f (spw=%.2f)",
                        i + 1, K, _metric_name, auc_val, spw)

            # v2.0: 收集 MDI（inner-train）和 OOS MDA（inner-OOS）
            if use_voting:
                # MDI: XGBoost gain 特征重要性
                try:
                    imp = model.get_score(importance_type='gain')
                    fold_mdi = {f: imp.get(f, 0.0) for f in fold_feats}
                except Exception:
                    fold_mdi = {f: 0.0 for f in fold_feats}
                fold_mdi_results.append(fold_mdi)

                # MDA OOS: [P0-1] 在折内 inner-OOS 段上逐特征置换测试 (验证块只评 AUC)
                if mda_enabled:
                    y_inner_oos = inner_oos_set[target_col].values
                    try:
                        if cfi_clusters is not None:
                            # [Test-CFI] 簇级置换: 每折按"折内特征集 ∩ 全局簇"做簇级 MDA
                            fold_clusters = [[f for f in c if f in fold_feats]
                                             for c in cfi_clusters]
                            fold_c_mda, fold_c_mda_std = compute_mda_oos_clustered(
                                model, inner_oos_set[fold_feats], y_inner_oos, fold_clusters,
                                n_repeats=getattr(self.config, 'mda_n_repeats', 5),
                                random_state=42 + i,  # 每折不同种子
                                metric_fn=metric_fn,  # [L2] 统一度量
                            )
                            fold_cluster_mda_results.append(fold_c_mda)
                            fold_cluster_mda_stds.append(fold_c_mda_std)
                        else:
                            fold_mda, fold_mda_std = compute_mda_oos(
                                model, inner_oos_set[fold_feats], y_inner_oos,
                                fold_feats, n_repeats=getattr(self.config, 'mda_n_repeats', 5),
                                random_state=42 + i,  # 每折不同种子
                                metric_fn=metric_fn,  # [L2] 统一度量
                            )
                            fold_mda_results.append(fold_mda)
                            fold_mda_stds.append(fold_mda_std)
                    except Exception as e:
                        logger.warning("Fold %d/%d: OOS MDA 计算失败: %s, 跳过", i + 1, K, e)
                        if cfi_clusters is not None:
                            fold_cluster_mda_results.append({cid: 0.0 for cid in range(len(cfi_clusters))})
                            fold_cluster_mda_stds.append({cid: 0.0 for cid in range(len(cfi_clusters))})
                        else:
                            fold_mda_results.append({f: 0.0 for f in fold_feats})
                            fold_mda_stds.append({f: 0.0 for f in fold_feats})

        if not fold_aucs:
            logger.error("Purged K-Fold: 无有效折，返回全部特征作为降级")
            fallback = feature_cols[:min(len(feature_cols), self.config.MAX_FEATURES)]
            return 0.5, sorted(fallback)

        # ──── 跨折共识特征 ────
        feature_counts = {}
        for fset in fold_features:
            for f in fset:
                feature_counts[f] = feature_counts.get(f, 0) + 1

        consensus_features = sorted([f for f, cnt in feature_counts.items()
                                     if cnt >= consensus_n])

        # 共识特征不足时取并集并按 IC 排名截断
        if len(consensus_features) < self.config.MIN_FEATURES:
            all_feats = sorted(set().union(*fold_features)) if fold_features else feature_cols
            consensus_features = all_feats[:min(len(all_feats), self.config.MAX_FEATURES)]
            logger.warning(f"共识特征({len(consensus_features)})不足MIN_FEATURES({self.config.MIN_FEATURES})，"
                           f"退至并集截断")

        meta_auc = float(np.mean(fold_aucs))
        logger.info(f"Purged K-Fold 完成: meta_auc={meta_auc:.4f}, "
                    f"共识特征={len(consensus_features)} (来自{len(fold_features)}折)")
        logger.info("=" * 60)

        # v2.0: 附加 IC/MDI/MDA 结果 + [Part C] 保存折定义
        cv_results = (meta_auc, consensus_features, fold_ic_results, fold_mdi_results, fold_mda_results, fold_mda_stds, fold_defs)
        if cfi_clusters is not None:
            # [Test-CFI] 簇级 MDA 结果追加在第 7/8 位（原解包 [0..6] 不受影响）
            cv_results = cv_results + (fold_cluster_mda_results, fold_cluster_mda_stds)
        return cv_results

    # ========== Phase 2: Purged K-Fold CV + 折内特征选择 ==========
    def train(self, train_data=None, df_train=None, df_daily=None, feature_cols=None,
              target_col='label', sample_weight=None, **kwargs):
        """
        训练元标注模型。

        Phase2: 嵌套 Purged K-Fold CV + 折内 IC 特征选择。
        用跨折共识特征在全量 240 天训练数据上训练生产模型。
        meta_auc 来自各折 CV 均值，供 AUC 门控使用。
        """
        # 兼容两种调用方式
        if train_data is not None:
            df_train = train_data
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("【开始训练 XGBoost 模型 (Purged K-Fold CV)】")
        logger.info("【加速设置】tree_method=%s, device=%s",
                    getattr(self.config, 'XGB_TREE_METHOD', 'hist'),
                    getattr(self.config, 'XGB_DEVICE', 'cpu'))
        logger.info("=" * 60)

        # -------- 数据校验 ----------
        if 'trade_date' not in df_train.columns:
            raise ValueError("df_train必须包含trade_date列")
        if target_col not in df_train.columns:
            raise ValueError(f"df_train必须包含{target_col}列")
        # [2026-08-18 修复] 排序并同步重排 sample_weight (消除权重-样本错位)
        df_train, sample_weight = align_sample_weight_to_sort(df_train, sample_weight)
        # [v3.0] Deterministic row index for stable tiebreaking
        df_train['_row_idx'] = range(len(df_train))

        # [L2] 目标模式检测 (binary/regression), 全链路统一使用
        self.objective_mode = _detect_objective_mode(df_train[target_col].values, self.config)
        self.config.OBJECTIVE_MODE = self.objective_mode
        if self.objective_mode == 'regression':
            logger.info("[L2] 目标模式=regression: objective=squarederror, 度量=Spearman IC, "
                        "跳过 scale_pos_weight/Platt校准/温度缩放")

        # -------- 特征列 ----------
        if feature_cols is None:
            feature_cols = [c for c in df_train.columns if c.startswith('alpha_')]
        feature_cols = sorted(feature_cols)
        logger.info("原始特征数: %d", len(feature_cols))

        # ====== 预处理 (once before CV) ======
        current_date = kwargs.get('current_date', None)
        total_n = len(df_train)
        half = total_n // 2
        df_ic = df_train.iloc[:half].copy()
        df_psi = df_train.iloc[half:].copy()
        pre_feats, _, _, _ = select_stable_features(
            df_ic, df_psi, feature_cols, self.config,
            target_col=target_col, current_date=current_date,
            skip_preprocess=False,
            skip_ic=True  # Phase 1: 仅清理(variance+sanitize)，IC筛选留给CV折内
        )
        logger.info("预处理完成: %d -> %d 特征 (入%d折CV)",
                     len(feature_cols), len(pre_feats),
                     getattr(self.config, 'cv_folds', 5))

        # ====== [P0-0] 超参候选选择 (仅重训周期触发; 时序 holdout 评估候选, 选优写回 config) ======
        _hp_t0 = time.time()
        try:
            self._select_hyperparams(df_train, pre_feats, target_col, sample_weight)
        except Exception as e:
            logger.warning(f"[HP-Search] 超参选择异常(跳过, 用原超参): {e}")
            import traceback
            logger.debug(traceback.format_exc())
        logger.info("[HP-Search] 阶段耗时 %.1fs", time.time() - _hp_t0)

        # ====== 嵌套 Purged K-Fold CV ======
        # [Test-CFI] cfi_mode: CV 前全局聚类 → 每折簇级置换 MDA（MLAM Ch.6 Clustered MDA）
        cfi_mode = bool(getattr(self.config, 'cluster_cfi_mode', False)) and \
                   bool(getattr(self.config, 'cluster_enabled', False))
        if cfi_mode:
            clusters = cluster_features_global(df_train, pre_feats, self.config)
            self._cfi_clusters = clusters
            cv_result = self._purged_kfold_cv(
                df_train, pre_feats, target_col, sample_weight,
                current_date=current_date, cfi_clusters=clusters
            )
        else:
            cv_result = self._purged_kfold_cv(
                df_train, pre_feats, target_col, sample_weight,
                current_date=current_date
            )
        meta_auc = cv_result[0]
        cv_consensus_features = cv_result[1]
        fold_ic_results = cv_result[2] if len(cv_result) > 2 else []
        fold_mdi_results = cv_result[3] if len(cv_result) > 3 else []
        fold_mda_results = cv_result[4] if len(cv_result) > 4 else []
        fold_mda_stds = cv_result[5] if len(cv_result) > 5 else []
        self._cv_fold_defs = cv_result[6] if len(cv_result) > 6 else []  # [Part C]
        self.meta_auc = meta_auc

        # v2.0: 多折投票 + 聚类去冗余（或 [Test-CFI] 簇级投票 + 簇内选代表）
        use_voting = getattr(self.config, 'use_voting', True)
        # [O7-20260904] SFI/簇代表选择剔除生产 val 段 (与生产模型 prod_val 同规则):
        #   代表/相关性选择若吃到 prod_val 日期 → 特征集选择与后续 prod_val 校准/门控
        #   重叠 → 选择偏倚 (Cawley-Talbot)。裁剪后进 select_cluster_reprs/cluster_features。
        _repr_unique = sorted(df_train['trade_date'].unique())
        _repr_n = len(_repr_unique)
        _repr_val_size = min(int(getattr(self.config, 'MAX_HOLD_DAYS', 20)),
                             max(_repr_n // 5, 1))
        # [O7-fix-20260904] _repr_unique 元素为标量 Timestamp (df['trade_date'].unique()),
        #   原 [0] 下标是二维数组行的遗留写法 → Timestamp 不可下标 → TypeError。
        #   修复: 直接取分界日期 (裁剪末 _repr_val_size 个唯一交易日作为生产 val 段)。
        _repr_cut = _repr_unique[-_repr_val_size] if _repr_val_size > 0 else None
        df_train_repr = (df_train[df_train['trade_date'] < _repr_cut]
                         if _repr_cut is not None else df_train)
        logger.info("[O7] 簇代表选择数据: 剔除生产 val %d 天 (共%d天, 用%d天)",
                    _repr_val_size, _repr_n, len(df_train_repr['trade_date'].unique()))
        if cfi_mode:
            # [Test-CFI] 簇级投票（use_voting=false 时内部退化为全簇入选）+ 簇内代表
            fold_cluster_mda = cv_result[7] if len(cv_result) > 7 else []
            fold_cluster_mda_std = cv_result[8] if len(cv_result) > 8 else []
            selected_cluster_ids, cluster_report = aggregate_cluster_votes(
                fold_cluster_mda, fold_cluster_mda_std, clusters, self.config,
                fold_ic_results=fold_ic_results,
                fold_mdi_results=fold_mdi_results,
                consensus_features=None,  # [P1-1] 移除 IC 准入: 投票池=基础清理全集, MDA裁决一切
            )
            selected_features = select_cluster_reprs(
                df_train_repr, selected_cluster_ids, clusters, self.config,
                target_col=target_col, sample_weight=sample_weight,
                cluster_report=cluster_report,
                objective_mode=self.objective_mode,
            )
            logger.info(f"Feature selection complete (CFI cluster MDA voting): "
                        f"{len(clusters)}簇 → {len(selected_cluster_ids)}簇入选 → {len(selected_features)}特征")
            for rank, feat in enumerate(selected_features, 1):
                logger.info(f"  #{rank:>2d} {feat:<45s}")
            logger.info(f"Final selected features: {len(selected_features)}")
        elif use_voting:
            voted_features, vote_report = aggregate_fold_votes(
                fold_ic_results, fold_mdi_results, fold_mda_results, self.config,
                # [P1-1] 移除 IC 准入层 (用户决策 2026-08-18):
                # 原 consensus_features(cv_feature_consensus=1≈并集) 限制 MDA 投票池成员,
                # 线性 IC 不应决定非线性模型的投票池 → None = 基础清理全集入池, MDA 投票(vote_min_folds=3)裁决
                consensus_features=None,
                fold_mda_stds=fold_mda_stds
            )

            # 聚类去冗余 (传 vote_report 全文: 供 repr_method 选择 scores/mda/ic 排名)
            selected_features = cluster_features(
                df_train_repr, voted_features, vote_report, self.config
            )

            # [v3.5] V10-style output: vote distribution + ranked list
            consensus_cnt = len(cv_consensus_features) if cv_consensus_features else 0
            vote_dist = vote_report.get('vote_distribution', {})
            if vote_dist:
                parts = [f"{v}fold={c}" for v, c in sorted(vote_dist.items(), reverse=True)]
                logger.info(f"  Vote distribution: {', '.join(parts)}")
            logger.info(f"Feature selection complete (MDA voting): {consensus_cnt if consensus_cnt else '?'} → {len(selected_features)}")

            # Ranked feature list
            scores = vote_report.get('scores', {})
            voted_dict = vote_report.get('votes', {})
            ic_dict = vote_report.get('ic_scores', {})
            for rank, feat in enumerate(sorted(selected_features, key=lambda f: scores.get(f, 0), reverse=True), 1):
                v = int(voted_dict.get(feat, 0))
                ic = float(ic_dict.get(feat, 0))
                sc = float(scores.get(feat, 0))
                logger.info(f"  #{rank:>2d} {feat:<45s} votes={v}/5  IC={ic:.4f}  score={sc:.4f}")
            logger.info(f"Final selected features: {len(selected_features)}")
        else:
            selected_features = cv_consensus_features
            logger.info(f"Feature selection complete (IC consensus only): {len(selected_features)} features")

        # [B3-20260905] P2-3: 簇代表跨期稳定性监控 (观察项, 不改选择逻辑):
        #   Jaccard(本期代表, 上期代表) + 代表数低于下限 max(8, 上期50%) 时告警。
        #   冒烟实证代表数 18→6→17 震荡, 先观察一期再决定是否锚定上期簇。
        _prev_reprs = getattr(self, '_prev_repr_features', None)
        if _prev_reprs:
            _cur, _prev = set(selected_features), set(_prev_reprs)
            _jac = len(_cur & _prev) / max(len(_cur | _prev), 1)
            _floor = max(8, int(len(_prev) * 0.5))
            logger.info(f"[B3] 代表集跨期稳定性: Jaccard={_jac:.3f} "
                        f"(上期{len(_prev)} → 本期{len(_cur)}, 留存{len(_cur & _prev)})")
            if len(selected_features) < _floor:
                logger.warning(f"[B3] 代表数 {len(selected_features)} 低于观察下限 "
                               f"max(8, 上期50%)={_floor}, 结构震荡需关注 (暂不改选择逻辑)")
            if _jac < 0.5:
                logger.warning(f"[B3] Jaccard={_jac:.3f} < 0.5, 代表集跨期更替过半 (暂不改选择逻辑)")
        self._prev_repr_features = list(selected_features)

        self.selected_features = selected_features
        if len(selected_features) == 0:
            logger.error("特征筛选后无特征可用，训练失败")
            return False

        # ====== 生产模型训练（时序 train/val 分割 + early stopping + OOS 校准）======        
        # Fix#4+7: 生产力模型验证集分割（时序末尾 val_size 天）
        # Fix#2: 校准器用验证集 OOS 概率
        # 标签封闭性：df_train 已通过 eligible_train_dates 过滤，val 最后一天标签完全封闭
        
        unique_dates = sorted(df_train['trade_date'].unique())
        n_dates = len(unique_dates)
        max_hold = getattr(self.config, 'MAX_HOLD_DAYS', 20)
        prod_val_size = min(max_hold, max(n_dates // 5, 1))
        
        if n_dates - prod_val_size < 60:
            logger.warning("训练数据不足（%d天, val=%d天），降级为全量无验证集训练", 
                          n_dates, prod_val_size)
            prod_val_dates = set()
            prod_train_dates = set(unique_dates)
        else:
            prod_val_dates = set(unique_dates[-prod_val_size:])
            prod_train_dates = set(unique_dates[:-prod_val_size])
        
        df_prod_train = df_train[df_train['trade_date'].isin(prod_train_dates)]
        has_val = len(prod_val_dates) > 0
        df_prod_val = df_train[df_train['trade_date'].isin(prod_val_dates)] if has_val else None
        
        logger.info("生产模型分割: train=%d天(%d样本), val=%d天(%d样本)",
                     len(prod_train_dates), len(df_prod_train),
                     len(prod_val_dates), len(df_prod_val) if has_val else 0)
        
        # 训练集标签、权重
        y_prod_train = df_prod_train[target_col].values
        wt = None
        if sample_weight is not None:
            wt = sample_weight[df_prod_train.index]
        
        # 正负样本统计 + 类别平衡
        n_pos = int(np.sum(y_prod_train == 1))
        n_neg = int(np.sum(y_prod_train == 0))
        scale_pos_weight = float(n_neg / max(n_pos, 1))
        balance_mode = getattr(self, 'class_balance_mode', 'auto')
        logger.info("训练集正样本数: %d, 负样本数: %d, scale_pos_weight=%.3f (mode=%s)",
                    n_pos, n_neg, scale_pos_weight, balance_mode)
        
        # 构建 DMatrix
        dtrain_prod = prepare_dmatrix(df_prod_train, self.selected_features, y_prod_train, weights=wt)
        
        y_val_prod = None
        dval_prod = None
        if has_val and len(df_prod_val) > 0:
            y_val_prod = df_prod_val[target_col].values
            dval_prod = prepare_dmatrix(df_prod_val, self.selected_features, y_val_prod)
            has_val = True
        else:
            has_val = False
        
        # XGBoost 参数
        # [L2] 目标模式 (train() 入口已检测并写入 self.objective_mode / config.OBJECTIVE_MODE)
        obj_mode = getattr(self, 'objective_mode',
                           _detect_objective_mode(y_prod_train, self.config))
        metric_fn = _objective_metric_fn(obj_mode)
        _metric_name = 'IC' if obj_mode == 'regression' else 'AUC'
        params = {
            'max_depth': self.config.XGB_MAX_DEPTH,
            'eta': self.config.XGB_LEARNING_RATE,
            'subsample': self.config.XGB_SUBSAMPLE,
            'colsample_bytree': self.config.XGB_COL_SAMPLE_BY_TREE,
            'alpha': self.config.XGB_REG_ALPHA,
            'lambda': self.config.XGB_REG_LAMBDA,
            'min_child_weight': self.config.XGB_MIN_CHILD_WEIGHT,
            'gamma': self.config.XGB_GAMMA,
            'objective': 'reg:squarederror' if obj_mode == 'regression' else 'binary:logistic',
            'eval_metric': 'rmse' if obj_mode == 'regression' else 'auc',
            'seed': self.config.XGB_RANDOM_STATE,
            'tree_method': getattr(self.config, 'XGB_TREE_METHOD', 'hist'),
            'device': getattr(self.config, 'XGB_DEVICE', 'cpu'),
            'nthread': 1
        }
        if obj_mode != 'regression' and balance_mode != 'none':
            params['scale_pos_weight'] = scale_pos_weight
        
        # ====== [v3.2] CV 折集成训练（移除多种子冗余，直接使用 CV 折集成） ======
        self.xgb_models = []

        # 保留种子集成的 DMatrix 和验证集用于 fallback
        es_rounds = self.config.XGB_EARLY_STOPPING_ROUNDS if has_val else None
        evals = [(dtrain_prod, 'train')]
        if has_val:
            evals.append((dval_prod, 'validation'))

        # 验证集概率与 AUC 默认值
        # [D 修复 2026-08-18] prod_val_auc 初始为 PKF 折均值, 仅作 fallback 前占位;
        # 主路径门控以折集成 meta_AUC 为准 (见 _train_cv_fold_ensemble → self.meta_auc)
        prod_val_auc = meta_auc
        val_proba_for_calib = None
        auc_train = 0.5

        # 训练集过拟合监测（用 CV 折集成后统一计算）

        self.meta_auc = prod_val_auc
        self.xgb_val_auc = prod_val_auc
        logger.info("[D] PKF 折均值 meta_auc=%.4f (仅初始化占位; 门控以折集成 meta_AUC 为准)",
                    meta_auc)
        # ====== [v3.2] CV 折集成训练（替代多种子集成） ======
        self._train_cv_fold_ensemble(df_train, target_col, sample_weight)

        # ====== [v3.2] Fallback: CV 折为空时，训练单模型确保 self.xgb_models 非空 ======
        if not self.xgb_models:
            logger.warning("CV 折集成无有效模型，回退到单模型训练")
            fallback_params = dict(params)
            if has_val:
                evals_list = [(dtrain_prod, 'train'), (dval_prod, 'validation')]
            else:
                evals_list = [(dtrain_prod, 'train')]
            model = xgb.train(
                fallback_params, dtrain_prod,
                num_boost_round=self.config.XGB_N_ESTIMATORS,
                evals=evals_list,
                early_stopping_rounds=es_rounds,
                verbose_eval=False,
                callbacks=[LRWarmupCallback(target_lr=self.config.XGB_LEARNING_RATE)]
            )
            self.xgb_models = [model]
            if has_val and dval_prod is not None:
                val_proba_for_calib = model.predict(dval_prod)
                # [L2] 统一度量
                prod_val_auc = float(metric_fn(y_val_prod, val_proba_for_calib))
                # Fallback calibration (仅二分类; 回归模式跳过)
                cal_method = getattr(self.config, 'CALIBRATION_METHOD', 'none')
                if obj_mode != 'regression' and cal_method in ('isotonic', 'platt') and len(val_proba_for_calib) > 20:
                    self.prob_calibrator = ProbabilityCalibrator(method=cal_method)
                    self.prob_calibrator.fit(val_proba_for_calib, y_val_prod)
            else:
                self.prob_calibrator = None
            train_proba = model.predict(dtrain_prod)
            auc_train = float(metric_fn(y_prod_train, train_proba))
            self.meta_auc = prod_val_auc
            self.xgb_val_auc = prod_val_auc
        else:
            # [D 修复 2026-08-18] 折集成主路径: 门控/过拟合口径统一为折集成 meta_AUC
            # (原实现 prod_val_auc 保持 PKF 折均值 → 过拟合检测/CV一致性检查口径错乱)
            prod_val_auc = float(self.meta_auc)  # = 折集成 AUC 加权 OOS 平均
            self.xgb_val_auc = prod_val_auc
            self.best_val_auc = prod_val_auc
            # [2026-08-11 修复] CV 折集成路径: 计算真实 train AUC
            # (原实现 auc_train 保持默认 0.5 → 日志恒显示 0.5000, 且过拟合检测失效)
            # [2026-08-18 修复] 必须用 df_prod_train (仅生产训练段) 预测:
            #   原实现 predict_proba(df_train) 输出 72000 样本(含val段) vs
            #   y_prod_train 66000 样本 → roc_auc_score 长度不匹配抛错被吞 → auc_train 恒 0.5
            try:
                train_proba = self.predict_proba(df_prod_train)
                auc_train = float(metric_fn(y_prod_train, train_proba))
            except Exception as e:
                logger.warning(f"train AUC 计算失败(不影响训练): {e}")

        # 向后兼容
        self.xgb_model = self.xgb_models[0]
        self.best_model = self.xgb_model
        self.best_val_auc = prod_val_auc

        # 过拟合检测 (折集成口径: train 度量 = 折集成在训练段, val 度量 = 折集成 OOS)
        # [2026-09-09 审计修复] 阈值与文案按目标模式区分:
        #   回归模式度量为 Spearman IC (实测全期 0.013~0.17, 量级远小于 AUC),
        #   原实现统一用 AUC 尺度的 0.1 阈值 → 回归模式下差值>0.1 几乎不可能出现,
        #   过拟合检测形同虚设; 文案统一写 "AUC" 亦误导诊断 (实为 IC)。
        #   修复: 回归模式阈值 0.05 / 二分类 0.10; 文案使用模式对应的度量名。
        #   注: 本处仅影响日志告警, 不改变任何训练或交易行为。
        _m_name = 'IC' if obj_mode == 'regression' else 'AUC'
        _overfit_thr = 0.05 if obj_mode == 'regression' else 0.10
        logger.info("[D] 门控/过拟合口径: train %s=%.4f, 折集成 OOS %s=%.4f (过拟合阈值差>%.2f)",
                    _m_name, auc_train, _metric_name, prod_val_auc, _overfit_thr)
        if auc_train - prod_val_auc > _overfit_thr:
            logger.warning("过拟合警告: train %s=%.4f, val %s=%.4f (差值=%.4f > 阈值%.2f)",
                         _m_name, auc_train, _m_name, prod_val_auc,
                         auc_train - prod_val_auc, _overfit_thr)
        else:
            logger.info("折集成 train %s=%.4f, val %s=%.4f",
                        _m_name, auc_train, _m_name, prod_val_auc)
        
        # 折集成 vs PKF 折均值一致性检查 (诊断用: 集成是否改善单折平均)
        _consist_thr = 0.03 if obj_mode == 'regression' else 0.05
        if abs(prod_val_auc - meta_auc) > _consist_thr:
            logger.info("[D] 折集成 meta_%s=%.4f vs PKF 折均值=%.4f (差值=%.4f)",
                        _metric_name, prod_val_auc, meta_auc, prod_val_auc - meta_auc)
        
        if meta_auc < self.config.AUC_THRESHOLD and obj_mode != 'regression':
            logger.warning("CV集成 meta AUC %.4f < %.2f，模型预测能力低于门控阈值（AUC门控将触发降仓/空仓）",
                         meta_auc, self.config.AUC_THRESHOLD)
        elif obj_mode == 'regression':
            logger.info("[L2] 回归模式: meta_%s=%.4f (门控由策略端 IC 阈值裁决)",
                        _metric_name, meta_auc)
        elif prod_val_auc < self.config.AUC_THRESHOLD and meta_auc >= self.config.AUC_THRESHOLD:
            logger.info("单折val AUC %.4f < %.2f 但 CV集成 meta AUC %.4f >= %.2f（模型可用，AUC门控将放行）",
                       prod_val_auc, self.config.AUC_THRESHOLD, meta_auc, self.config.AUC_THRESHOLD)
        
        self.best_threshold = 0.5
        self.best_p_value = None
        self.best_train_p_value = None
        self.feature_scores = {}
        self.feature_max_ic = 0.0

        # ====== [CFI-诊断] 簇级重要性分布输出 (纯增量, 不改变任何行为) ======
        if (bool(getattr(self.config, 'cluster_diagnose', False))
                and has_val and dval_prod is not None and y_val_prod is not None
                and len(self.xgb_models) > 0 and len(self.selected_features) > 0):
            try:
                # 输出到回测结果目录下的 cfi_importance/ 单独文件夹
                # (model_dir = {output_dir}/meta_models → dirname = 回测结果目录)
                _diag_dir = os.path.join(os.path.dirname(os.path.abspath(self.model_dir)),
                                         'cfi_importance')
                _ts = current_date if current_date is not None else datetime.now()
                _fname = f'cfi_importance_{_ts.strftime("%Y%m%d")}.csv'
                cfi_diagnose(
                    self.xgb_models, df_prod_val, y_val_prod, self.selected_features,
                    self.config,
                    output_path=os.path.join(_diag_dir, _fname),
                    metric_fn=metric_fn,
                    n_repeats=getattr(self.config, 'mda_n_repeats', 20),
                )
            except Exception as e:
                logger.warning("[CFI-Diag] 诊断输出失败: %s", e)

        elapsed = time.time() - start_time
        logger.info(f"Ensemble training complete ({'CV fold' if self.xgb_models and hasattr(self, '_cv_fold_defs') and self._cv_fold_defs else 'seed'}): "
                    f"{len(self.xgb_models)} XGBoost + {len(self.lgb_models)} LightGBM models, "
                    f"time: {elapsed:.1f}s")

        return True

    # ================================================================
    # [Part C] CV 折集成训练（对齐 V10，替代多种子集成）
    # ================================================================
    def _train_cv_fold_ensemble(self, df_train, target_col, sample_weight):
        """CV 折集成：每折独立 train/val split + 超参数多样性 + AUC 加权投票"""
        if not getattr(self, '_cv_fold_defs', []):
            logger.info("[CV Ensemble] 无 CV 折定义，跳过（保留多种子集成结果）")
            return

        import time
        cv_t0 = time.time()
        fold_defs = self._cv_fold_defs
        n_folds = len(fold_defs)
        features = self.selected_features
        logger.info("[CV折集成] 使用 %d 折 CV 分割训练 %d 个模型", n_folds, min(n_folds, 5))

        # 超参数多样性 [A 修复 2026-08-18]: 围绕"超参选择选中值"小扰动, 保留集成降方差
        # 原硬编码 [6,6,7,7,8]/[0.6..]/[0.6..] 覆盖选中超参 → 超参选择对 depth/ss/cs 传导失效
        # (用户 2026-08-18 日志实证: HP-Search 选 c4(depth=6,ss=0.7,cs=0.8), 折集成却用 6-8/0.6-0.8)
        _sel_d = int(getattr(self.config, 'XGB_MAX_DEPTH', 4))
        _sel_ss = float(getattr(self.config, 'XGB_SUBSAMPLE', 0.7))
        _sel_cs = float(getattr(self.config, 'XGB_COL_SAMPLE_BY_TREE', 0.7))
        depth_list = [_sel_d, _sel_d, _sel_d + 1, _sel_d + 1, _sel_d + 2]
        depth_list = [max(2, d) for d in depth_list]
        ss_list = [max(0.3, min(1.0, _sel_ss - 0.05)), _sel_ss, _sel_ss,
                   max(0.3, min(1.0, _sel_ss + 0.05)), max(0.3, min(1.0, _sel_ss + 0.05))]
        cs_list = [max(0.3, min(1.0, _sel_cs - 0.05)), _sel_cs, max(0.3, min(1.0, _sel_cs + 0.05)),
                   _sel_cs, max(0.3, min(1.0, _sel_cs - 0.05))]
        metric_list = ['auc'] * 5
        n_actual = min(n_folds, 5)
        logger.info(f"  [A] 折集成多样性围绕选中超参: depth={depth_list[:n_actual]}, "
                    f"subsample={ss_list[:n_actual]}, colsample={cs_list[:n_actual]}, "
                    f"metric={metric_list[:n_actual]} (选中: depth={_sel_d}, ss={_sel_ss}, cs={_sel_cs})")

        cv_models = []
        cv_aucs = []
        cv_oos_probs = []  # for merged calibration
        cv_oos_labels = []
        cv_oos_val_feats = []  # [2026-08-05] 每折 val 特征矩阵 (供 LGBM OOS 融合校准)

        for i in range(min(n_folds, 5)):
            train_dates_i, val_dates_i = fold_defs[i]
            fold_train = df_train[df_train['trade_date'].isin(train_dates_i)]
            fold_val = df_train[df_train['trade_date'].isin(val_dates_i)]
            if len(fold_train) < 100 or len(fold_val) < 50:
                logger.warning(f"[CV Ensemble] Fold {i+1}: 样本不足，跳过")
                continue

            # [O7-20260904] 折集成多样种: 每折 seed 取自 ensemble_seeds(缺省 42+i),
            #   替代原所有折共用单一 XGB_RANDOM_STATE (零额外算力, 跨折 seed 多样性)。
            _seed_list = getattr(self, 'ensemble_seeds', None) or [42, 43, 44, 45, 46]
            fold_seed = int(_seed_list[i % len(_seed_list)])

            # [O3-20260904] Early stopping 用折内时间尾段 (inner-val, 从 fold_train 末
            #   段切出), 外层验证块 val_dates_i 只做最终 OOS AUC/加权 → 上报 meta_auc
            #   不再被早停轮数选择污染 (原: 早停与评估共用折验证块 → 乐观)。
            _ft_dates = sorted(fold_train['trade_date'].unique())
            _es_frac = float(getattr(self.config, 'fold_es_val_frac', 0.15))
            _es_size = min(max(10, int(len(_ft_dates) * _es_frac)), max(len(_ft_dates) // 3, 10))
            if len(_ft_dates) - _es_size < 30:
                _es_size = 0
            if _es_size > 0:
                _es_dates = set(_ft_dates[-_es_size:])
                fold_estrain = fold_train[~fold_train['trade_date'].isin(_es_dates)]
                fold_esval = fold_train[fold_train['trade_date'].isin(_es_dates)]
            else:
                fold_estrain = fold_train
                fold_esval = fold_train.tail(1)
            if len(fold_estrain) < 100 or len(fold_esval) < 20:
                logger.warning(f"[CV Ensemble] Fold {i+1}: inner-val 切分样本不足, 跳过")
                continue

            # [2026-08-05] 数值卫生: inf→nan (与 prepare_dmatrix 一致, 防 XGBoost 崩)
            X_tr = fold_estrain[features].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
            y_tr = fold_estrain[target_col].values
            X_vl = fold_val[features].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
            y_vl = fold_val[target_col].values
            X_es = fold_esval[features].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
            y_es = fold_esval[target_col].values

            # Sample weights for this fold
            wt_i = None
            if sample_weight is not None:
                wt_i = sample_weight[fold_estrain.index].values if hasattr(sample_weight, 'values') else sample_weight[fold_estrain.index]

            dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=wt_i)
            dval = xgb.DMatrix(X_vl, label=y_vl)
            des = xgb.DMatrix(X_es, label=y_es)

            # Hyperparameter diversity
            # [L2] 目标模式切换 (regression: squarederror+rmse, 跳过spw)
            _cv_obj_mode = getattr(self, 'objective_mode',
                                   _detect_objective_mode(y_tr, self.config))
            _cv_metric_fn = _objective_metric_fn(_cv_obj_mode)
            _cv_metric_name = 'IC' if _cv_obj_mode == 'regression' else 'AUC'
            params = {
                'max_depth': depth_list[i] if i < len(depth_list) else 6,
                'eta': self.config.XGB_LEARNING_RATE,
                'subsample': ss_list[i] if i < len(ss_list) else 0.7,
                'colsample_bytree': cs_list[i] if i < len(cs_list) else 0.7,
                'alpha': self.config.XGB_REG_ALPHA,
                'lambda': self.config.XGB_REG_LAMBDA,
                'min_child_weight': self.config.XGB_MIN_CHILD_WEIGHT,
                'gamma': self.config.XGB_GAMMA,
                'objective': 'reg:squarederror' if _cv_obj_mode == 'regression' else 'binary:logistic',
                'eval_metric': ('rmse' if _cv_obj_mode == 'regression'
                                else (metric_list[i] if i < len(metric_list) else 'auc')),
                'seed': fold_seed,
                'tree_method': 'hist',
                'device': 'cpu',
                'nthread': 1,
            }
            # Class balance (仅二分类)
            n_pos = int(np.sum(y_tr == 1))
            n_neg = int(np.sum(y_tr == 0))
            if _cv_obj_mode != 'regression' and n_pos > 0:
                params['scale_pos_weight'] = float(n_neg / n_pos)

            model = xgb.train(
                params, dtrain,
                num_boost_round=self.config.XGB_N_ESTIMATORS,
                evals=[(dtrain, 'train'), (des, 'es_val')],
                early_stopping_rounds=self.config.XGB_EARLY_STOPPING_ROUNDS,
                verbose_eval=False,
                callbacks=[LRWarmupCallback(target_lr=self.config.XGB_LEARNING_RATE)]  # v3.2: read config lr
            )

            # Evaluate
            # [L2] 统一度量 (binary→AUC, regression→Spearman IC)
            val_probs = model.predict(dval)
            auc_i = float(_cv_metric_fn(y_vl, val_probs))

            # [Part C] Weak model exclusion (binary: AUC<0.5; regression: IC<=0)
            _weak_thr = 0.0 if _cv_obj_mode == 'regression' else 0.50
            if auc_i <= _weak_thr:
                logger.warning(f"[CV Ensemble] Fold {i+1}: {_cv_metric_name}={auc_i:.4f} <= {_weak_thr}, 排除弱模型")
                continue

            cv_models.append(model)
            cv_aucs.append(auc_i)
            cv_oos_probs.append(val_probs)
            cv_oos_labels.append(y_vl)
            # [2026-08-05] 保存该折 val 特征 (fillna(0) 与 predict 侧 LGBM 输入一致)
            cv_oos_val_feats.append(
                fold_val[features].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
            )
            best_iter = getattr(model, 'best_iteration', self.config.XGB_N_ESTIMATORS)
            logger.info(f"  Fold {i+1}/{n_folds} (seed={params['seed']}, depth={params['max_depth']}, "
                       f"ss={params['subsample']}, cs={params['colsample_bytree']}, "
                       f"metric={params['eval_metric']}): train={len(fold_train)}, val={len(fold_val)}, "
                       f"best_iter={best_iter}, val_{_cv_metric_name}={auc_i:.4f}")

        if not cv_models:
            logger.warning("[CV Ensemble] 无有效折模型，保留多种子集成")
            return

        # AUC/IC-weighted weights (binary: max(0, AUC-0.5); regression: max(0, IC))
        _base = 0.0 if _cv_obj_mode == 'regression' else 0.50
        excess_aucs = [max(0, a - _base) for a in cv_aucs]
        sum_excess = sum(excess_aucs) or 1.0
        cv_weights = [e / sum_excess for e in excess_aucs]

        # V10-style summary
        cv_time = time.time() - cv_t0
        w_str = ', '.join([f'fold{i+1} {_cv_metric_name}={a:.4f} w={w:.3f}' for i, (a, w) in enumerate(zip(cv_aucs, cv_weights))])
        logger.info(f"CV折集成完成: {len(cv_models)} 模型 ({len(cv_models)} active), 耗时 {cv_time:.1f}s, meta_{_cv_metric_name}={float(np.average(cv_aucs, weights=cv_weights)):.4f}")
        logger.info(f"  集成权重: [{w_str}]")

        # Replace multi-seed models with CV fold models
        self.xgb_models = cv_models
        self.xgb_model = cv_models[0]
        self.best_model = cv_models[0]
        self._cv_model_weights = cv_weights
        self._cv_model_aucs = cv_aucs

        # [v3.5] LGBM 异质集成训练 (提前到校准之前: 校准器须在最终融合分布上拟合)
        self._train_lgb_ensemble_if_enabled(df_train, target_col, sample_weight)

        # Merged OOS calibration (校准器输入 = 与 predict_proba 完全一致的融合分布)
        if len(cv_oos_probs) > 0:
            merged_probs = np.concatenate(cv_oos_probs)
            merged_labels = np.concatenate(cv_oos_labels)
            self.meta_auc = float(np.average(cv_aucs, weights=cv_weights))
            # [L2] 统一度量
            self.xgb_val_auc = float(_cv_metric_fn(merged_labels, merged_probs))
            self.best_val_auc = self.xgb_val_auc

            # Platt/Iso calibration on merged OOS (仅二分类; 回归模式跳过)
            cal_method = getattr(self.config, 'CALIBRATION_METHOD', 'none')
            if _cv_obj_mode != 'regression' and cal_method in ('isotonic', 'platt') and len(merged_probs) > 20:
                # [2026-08-05 修复] 校准器拟合"融合 OOS" = XGB折集成×w_xgb + LGBM×w_lgb
                # (业界: calibrate the final ensemble output, 校准器输入分布=预测分布)
                calib_input = merged_probs
                if (self._lgb_enabled and self.lgb_models and LGB_AVAILABLE
                        and len(cv_oos_val_feats) == len(cv_oos_probs)):
                    try:
                        lgb_oos = []
                        for feats_i in cv_oos_val_feats:
                            preds_i = [m.predict(feats_i) for m in self.lgb_models]
                            lgb_oos.append(np.column_stack(preds_i).mean(axis=1))
                        lgb_merged = np.concatenate(lgb_oos)
                        calib_input = self._lgb_w_xgb * merged_probs + self._lgb_w_lgb * lgb_merged
                        logger.info(f"[CV Calibrate/{cal_method}] 校准器拟合融合OOS "
                                    f"(XGB {self._lgb_w_xgb} + LGB {self._lgb_w_lgb}, {len(calib_input)} samples)")
                    except Exception as e:
                        logger.warning(f"[CV Calibrate] LGBM OOS 融合失败 ({e}), 校准器使用 XGB-only OOS")
                        calib_input = merged_probs
                self.prob_calibrator = ProbabilityCalibrator(method=cal_method)
                self.prob_calibrator.fit(calib_input, merged_labels)
                logger.info(f"[CV Calibrate/{cal_method}] cal_AUC={self.xgb_val_auc:.4f} | meta_AUC={self.meta_auc:.4f}")

            logger.info(f"Val: {_cv_metric_name}={self.meta_auc:.4f} (CV折集成meta {_cv_metric_name})")

    def _train_lgb_ensemble_if_enabled(self, df_train, target_col, sample_weight=None):
        """[v3.5] 训练 LightGBM 异质集成 (对齐V10)

        [2026-08-05 修复] 按业界最佳实践补齐三项:
        1. 样本权重: lgb.Dataset(weight=...) 与 XGBoost 同口径 (平均唯一性×标签质量)
        2. 早停: 时序 val 分割 (末 min(max_hold, n/5) 天) + early_stopping,
           替代此前固定 200 轮 (防过拟合)
        3. 校准: 由 _train_cv_fold_ensemble 在"融合 OOS"上统一拟合校准器
           (calibrate final ensemble output, 校准器输入分布 = 预测分布)
        """
        if not self._lgb_enabled or not LGB_AVAILABLE:
            return
        # [L2] 回归模式暂不支持 LGBM (lgb 目标/标签需另行适配), 直接跳过
        if getattr(self, 'objective_mode', 'binary') == 'regression':
            logger.info("[L2] 回归模式: 跳过 LightGBM 集成 (XGBoost-only)")
            return
        try:
            import time
            lgb_t0 = time.time()
            X_all = df_train[self.selected_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
            y_all = df_train[target_col].values
            if len(X_all) < 200:
                logger.info("[LGB] Training data insufficient, skipping")
                return

            # 时序 val 分割 (与生产模型一致: 末 min(max_hold, n/5) 天)
            unique_dates = sorted(df_train['trade_date'].unique())
            n_dates = len(unique_dates)
            max_hold = getattr(self.config, 'MAX_HOLD_DAYS', 20)
            val_size = min(max_hold, max(n_dates // 5, 1))
            if n_dates - val_size < 60:
                val_dates = set()
            else:
                val_dates = set(unique_dates[-val_size:])
            tr_mask = ~df_train['trade_date'].isin(val_dates)
            X_tr = X_all[tr_mask.values]
            y_tr = y_all[tr_mask.values]
            wt_tr = None
            if sample_weight is not None:
                wt_tr = np.asarray(sample_weight)[tr_mask.values]
            X_va = X_all[~tr_mask.values]
            y_va = y_all[~tr_mask.values]
            has_val = len(X_va) >= 50

            seeds = self.ensemble_seeds if self.ensemble_seeds else [42, 43, 44, 45, 46]
            self.lgb_models = []
            for seed in seeds:
                lgb_params = dict(self._lgb_params)
                lgb_params['seed'] = seed
                lgb_params['feature_fraction_seed'] = seed
                lgb_params['bagging_seed'] = seed
                dtrain_lgb = lgb.Dataset(X_tr, label=y_tr.astype(int), weight=wt_tr)
                if has_val:
                    dval_lgb = lgb.Dataset(X_va, label=y_va.astype(int))
                    lgb_model = lgb.train(lgb_params, dtrain_lgb, num_boost_round=200,
                                          valid_sets=[dval_lgb],
                                          callbacks=[lgb.early_stopping(20, verbose=False)])
                else:
                    lgb_model = lgb.train(lgb_params, dtrain_lgb, num_boost_round=200)
                self.lgb_models.append(lgb_model)
            lgb_time = time.time() - lgb_t0
            n_rounds = [int(getattr(m, 'best_iteration', 200)) for m in self.lgb_models]
            logger.info(f"[LGB CV Ensemble] {len(self.lgb_models)} models trained "
                        f"(weighted{' + early_stop' if has_val else ''}), "
                        f"best_iter={n_rounds}, time: {lgb_time:.1f}s")
        except Exception as e:
            logger.warning(f"[LGB Ensemble] Training failed: {type(e).__name__}, falling back to XGBoost-only")

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("模型未训练")
        if data.empty:
            return np.array([])
        feat_cols = self.selected_features
        # [2026-08-05 修复] 业界最佳实践: 训练/预测特征集必须严格一致。
        # 缺失列直接 fail-fast 抛错（显式暴露数据管线问题），
        # 而非静默降级为子集（XGBoost 列数不符会报错/预测分布偏移）。
        missing_cols = set(feat_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(
                f'预测数据缺少训练特征列: {sorted(missing_cols)} — 训练/预测特征集必须一致'
            )
        available_cols = feat_cols
        if not available_cols:
            return np.array([])
        data = data[available_cols]
        dtest = prepare_dmatrix(data, available_cols)
        # [Part C] CV 折集成 AUC 加权投票（回退到简单均值）
        if len(self.xgb_models) > 1:
            probs_list = [m.predict(dtest) for m in self.xgb_models]
            if hasattr(self, '_cv_model_weights') and len(self._cv_model_weights) == len(probs_list):
                probs = np.average(probs_list, axis=0, weights=self._cv_model_weights)
            else:
                probs = np.mean(probs_list, axis=0)
        else:
            probs = self.best_model.predict(dtest)

        # ====== P1-B: 应用Isotonic概率校准 (回归模式跳过) ======
        _pred_mode = getattr(self, 'objective_mode', 'binary')

        # [v3.5] LGBM 异质集成预测融合 (回归模式 LGBM 未训练, 自动跳过)
        if _pred_mode != 'regression' and self._lgb_enabled and self.lgb_models and LGB_AVAILABLE:
            try:
                X_lgb = data[available_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
                lgb_preds = []
                for m in self.lgb_models:
                    try:
                        lgb_preds.append(m.predict(X_lgb))
                    except Exception:
                        continue
                if lgb_preds:
                    lgb_pred = np.column_stack(lgb_preds).mean(axis=1)
                    w_xgb = self._lgb_w_xgb
                    w_lgb = self._lgb_w_lgb
                    probs = w_xgb * probs + w_lgb * lgb_pred
            except Exception as e:
                logger.debug(f"[LGB Blend] prediction failed: {e}, using XGBoost-only")

        # ====== P1-B: 应用Isotonic概率校准 (仅二分类; 回归模式跳过) ======
        if (_pred_mode != 'regression'
                and self.prob_calibrator is not None
                and self.prob_calibrator.is_fitted
                and len(probs) > 0):
            try:
                probs = self.prob_calibrator.predict(probs)
                # [P1-B4] 裁剪已上移至 calibrator.predict()，此处保留作双重保险
                probs = np.clip(probs, 0.05, 0.95)
            except Exception as e:
                logger.warning(f"[P1-B] 校准应用失败: {e}, 使用原始概率")

        # ====== [v3.2] 温度缩放: 仅在无 Platt/Iso 校准时启用（避免双重校准） ======
        # [L2] 回归模式跳过温度缩放 (概率语义不存在), 输出裁剪到 [0,1] 分位域
        if _pred_mode == 'regression':
            probs = np.clip(probs, 0.0, 1.0)
        elif (self.prob_calibrator is None or not self.prob_calibrator.is_fitted):
            temperature = getattr(self.config, 'PROB_TEMPERATURE', 1.0)
            if temperature != 1.0 and len(probs) > 0:
                probs = np.power(np.clip(probs, 1e-6, 1.0 - 1e-6), 1.0 / temperature)
                probs = np.clip(probs, 0.05, 0.95)

        # ====== [Part C] 概率坍缩检测 ======
        prob_std = np.std(probs)
        if prob_std < 0.005:
            logger.warning(
                f"[ProbCollapse] 概率标准差={prob_std:.4f} < 0.005, "
                f"范围=[{probs.min():.4f}, {probs.max():.4f}], "
                f"模型可能无区分力"
            )

        return probs

    def save(self, suffix=''):
        # [Step2a] 保存所有种子模型
        if len(self.xgb_models) > 1:
            for i, model in enumerate(self.xgb_models):
                seed = self.ensemble_seeds[i] if i < len(self.ensemble_seeds) else i
                model_path = os.path.join(self.model_dir, f'xgb_model{suffix}_seed{seed}.json')
                model.save_model(model_path)
            model_path_main = os.path.join(self.model_dir, f'xgb_model{suffix}.json')
            self.xgb_models[0].save_model(model_path_main)  # 保兼容
            logger.info(f"保存 {len(self.xgb_models)} 个种子模型到 {self.model_dir}")
        elif self.best_model:
            model_path = os.path.join(self.model_dir, f'xgb_model{suffix}.json')
            self.best_model.save_model(model_path)
            logger.info(f"模型已保存到 {model_path}")
        cfg_path = os.path.join(self.model_dir, f'config{suffix}.pkl')
        with open(cfg_path, 'wb') as f:
            pickle.dump({
                'selected_features': self.selected_features,
                'feature_scores': self.feature_scores,
                'feature_max_ic': self.feature_max_ic,
                'xgb_threshold': self.xgb_threshold,
                'best_threshold': self.best_threshold,
                'best_val_auc': self.best_val_auc,
                'xgb_val_auc': self.xgb_val_auc,
                'ensemble_seeds': self.ensemble_seeds,  # [Step2a]
                # [P1-B] 概率校准器状态
                'prob_calibrator_fitted': (
                    self.prob_calibrator.is_fitted if self.prob_calibrator else False
                ),
            }, f)
        # [P1-B] 保存概率校准器
        if self.prob_calibrator is not None and self.prob_calibrator.is_fitted:
            cal_path = os.path.join(self.model_dir, f'prob_calibrator{suffix}.pkl')
            self.prob_calibrator.save(cal_path)
            logger.info(f"[P1-B] 概率校准器已保存到 {cal_path}")


    def load(self, suffix=''):
        # [Step2a] 尝试加载多种子模型，兼容旧版单模型
        cfg_path = os.path.join(self.model_dir, f'config{suffix}.pkl')
        if os.path.exists(cfg_path):
            with open(cfg_path, 'rb') as f:
                cfg = pickle.load(f)
            self.selected_features = cfg.get('selected_features', [])
            self.feature_scores = cfg.get('feature_scores', {})
            self.feature_max_ic = cfg.get('feature_max_ic', 0.0)
            self.xgb_threshold = cfg.get('xgb_threshold', 0.5)
            self.best_threshold = cfg.get('best_threshold', 0.5)
            self.best_val_auc = cfg.get('best_val_auc', None)
            self.xgb_val_auc = cfg.get('xgb_val_auc', None)
            self.ensemble_seeds = cfg.get('ensemble_seeds', [])

        # [Step2a] 尝试加载每个种子模型
        self.xgb_models = []
        if self.ensemble_seeds and len(self.ensemble_seeds) > 1:
            for seed in self.ensemble_seeds:
                mp = os.path.join(self.model_dir, f'xgb_model{suffix}_seed{seed}.json')
                if os.path.exists(mp):
                    m = xgb.Booster()
                    m.load_model(mp)
                    self.xgb_models.append(m)
            if self.xgb_models:
                logger.info(f"加载 {len(self.xgb_models)} 个种子模型")
                self.best_model = self.xgb_models[0]
                self.xgb_model = self.best_model
                return

        # 向后兼容：加载单模型
        model_path = os.path.join(self.model_dir, f'xgb_model{suffix}.json')
        if os.path.exists(model_path):
            self.best_model = xgb.Booster()
            self.best_model.load_model(model_path)
            self.xgb_model = self.best_model
            self.xgb_models = [self.best_model]
            self.ensemble_seeds = self.ensemble_seeds or [42]
            logger.info(f"模型已从 {model_path} 加载")

        # [P1-B] 加载概率校准器
        if cfg.get('prob_calibrator_fitted', False):
            cal_path = os.path.join(self.model_dir, f'prob_calibrator{suffix}.pkl')
            if os.path.exists(cal_path):
                self.prob_calibrator = ProbabilityCalibrator()
                self.prob_calibrator.load(cal_path)
                logger.info(f"[P1-B] 概率校准器已加载")
            else:
                self.prob_calibrator = None
        else:
            self.prob_calibrator = None

