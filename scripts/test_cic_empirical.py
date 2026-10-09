# -*- coding: utf-8 -*-
"""实证: CIC(条件IC) 剔除的因子在树模型(XGB)中是否真的无效?

方法:
1. 用 samples_neutral 全量数据重现 cycle=1 (2022-08-01 调仓, 训练窗口前240交易日)
   的 CIC 判定 (与 ConditionalICAnalyzer 同逻辑: 按季度分桶×3桶, RankIC(factor, fwd_ret),
   任一桶 IC>+0.03 且任一桶 IC<−0.03 → 符号翻转剔除; 或 max-min>0.10 → 范围过大剔除)
2. 训练简化 XGBoost, 用置换重要性 (MDA) 度量每个因子在树模型中的真实贡献
3. 对比: CIC剔除组 vs 保留组的 MDA 分布; 找出 "被CIC剔除但MDA显著为正" 的因子 (=错杀)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = r'E:\quant\gpsef_project'
sys.path.insert(0, PROJECT_ROOT)

NEU = os.path.join(PROJECT_ROOT, 'results', '20260808_131106', 'samples_neutral.parquet')
TRAIN_END = pd.Timestamp('2022-08-01')   # cycle=1 调仓日
TRAIN_DAYS = 240                          # 训练窗口交易日数
IC_POS, IC_NEG = 0.03, -0.03              # CIC 符号翻转阈值
IC_RANGE = 0.10                           # CIC 范围阈值
N_BINS = 3                                # 每条件分桶数
SAMPLE_N = 120000                         # 抽样规模 (全量太重)
XGB_ROUNDS = 60
N_REPEATS = 5                            # 置换重要性重复次数


def load_train_window():
    cols = ['snapshot_date', 'fwd_ret', 'label']
    neu = pd.read_parquet(NEU, columns=None)
    feat_cols = [c for c in neu.columns if c.startswith('alpha_')]
    sub = neu[neu['snapshot_date'] <= TRAIN_END].copy()
    days = sorted(sub['snapshot_date'].unique())
    win_days = days[-TRAIN_DAYS:]
    sub = sub[sub['snapshot_date'].isin(win_days)]
    sub = sub[sub['fwd_ret'].notna()]
    rng = np.random.RandomState(42)
    if len(sub) > SAMPLE_N:
        sub = sub.sample(SAMPLE_N, random_state=42)
    return sub, feat_cols


def cic_judge(df, feat_cols):
    """重现 CIC: 按季度分桶×3桶, RankIC; 符号翻转/范围过大 → 剔除。"""
    dates = pd.to_datetime(df['snapshot_date'])
    quarters = np.array([f"Q{(d.month - 1) // 3 + 1}-{d.year}" for d in dates])
    y = df['fwd_ret'].values
    flagged = set()
    records = {}
    for f in feat_cols:
        records[f] = []
    for q in sorted(set(quarters)):
        m = quarters == q
        if m.sum() < 50:
            continue
        yq = y[m]
        rq = pd.Series(yq).rank().values
        for f in feat_cols:
            fv = df[f].values[m]
            ok = np.isfinite(fv) & np.isfinite(yq)
            if ok.sum() < 30:
                continue
            fr = pd.Series(fv[ok]).rank().values
            if np.std(fr) < 1e-9:
                continue
            ic = np.corrcoef(fr, rq[ok])[0, 1]
            if np.isnan(ic):
                ic = 0.0
            records[f].append(ic)
    for f in feat_cols:
        ics = records[f]
        if len(ics) < 2:
            continue
        rng_ = max(ics) - min(ics)
        has_pos = any(ic > IC_POS for ic in ics)
        has_neg = any(ic < IC_NEG for ic in ics)
        if rng_ > IC_RANGE or (has_pos and has_neg):
            flagged.add(f)
    return flagged


def xgb_mda(df, feat_cols, flagged):
    """训练 XGB + 置换重要性。返回 (MDA Series, 特征gain重要性)。"""
    import xgboost as xgb
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = df['label'].values
    model = xgb.XGBClassifier(
        n_estimators=XGB_ROUNDS, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, n_jobs=4,
        eval_metric='auc', verbosity=0, random_state=42)
    model.fit(X, y)
    base = model.predict_proba(X)[:, 1]
    from sklearn.metrics import roc_auc_score
    base_auc = roc_auc_score(y, base)
    # 置换重要性
    mda = {}
    rng = np.random.RandomState(7)
    X_perm = X.copy()
    for j, f in enumerate(feat_cols):
        aucs = []
        for _ in range(N_REPEATS):
            X_perm[:, j] = rng.permutation(X[:, j])
            p = model.predict_proba(X_perm)[:, 1]
            aucs.append(roc_auc_score(y, p))
            X_perm[:, j] = X[:, j]
        mda[f] = base_auc - np.mean(aucs)
    mda_s = pd.Series(mda)
    gain_s = pd.Series(model.feature_importances_, index=feat_cols)
    return base_auc, mda_s, gain_s


def main():
    t0 = time.time()
    print("加载训练窗口数据 (cycle=1, 240交易日, 抽样)...")
    df, feat_cols = load_train_window()
    print(f"  样本 {len(df):,}, 特征 {len(feat_cols)}, 耗时 {time.time()-t0:.0f}s")

    print("重现 CIC 判定...")
    flagged = cic_judge(df, feat_cols)
    kept = [f for f in feat_cols if f not in flagged]
    print(f"  CIC 剔除 {len(flagged)}/{len(feat_cols)} 个因子 ({len(flagged)/len(feat_cols)*100:.0f}%)")

    print("训练 XGB + 置换重要性 (这步较慢)...")
    base_auc, mda_s, gain_s = xgb_mda(df, feat_cols, flagged)
    print(f"  XGB 基线 AUC = {base_auc:.4f}, 耗时 {time.time()-t0:.0f}s")

    mda_flag = mda_s[list(flagged)]
    mda_keep = mda_s[list(kept)]
    print()
    print("=" * 60)
    print("实证结果: 被CIC剔除 vs 保留 因子的树模型MDA分布")
    print("=" * 60)
    print(f"  CIC剔除组 ({len(mda_flag)}): MDA 均值 {mda_flag.mean():.5f} | 中位 {mda_flag.median():.5f} | MDA>0: {(mda_flag>0).sum()} ({(mda_flag>0).mean()*100:.0f}%)")
    print(f"  保留组   ({len(mda_keep)}): MDA 均值 {mda_keep.mean():.5f} | 中位 {mda_keep.median():.5f} | MDA>0: {(mda_keep>0).sum()} ({(mda_keep>0).mean()*100:.0f}%)")
    # 被剔除但 MDA 显著为正 (= 潜在错杀): MDA > 保留组中位数
    thr = mda_keep.median()
    wrongful = mda_flag[mda_flag > thr].sort_values(ascending=False)
    print(f"\n  [错杀候选] 被CIC剔除但MDA > 保留组中位数({thr:.5f}): {len(wrongful)} 个")
    if len(wrongful):
        print("  Top-5 错杀候选 (被剔除但树模型认为重要):")
        for f, v in wrongful.head(5).items():
            print(f"    {f}: MDA={v:.5f} (gain={gain_s[f]:.4f})")
    # 被剔除且 MDA 为负 (正确剔除)
    neg = mda_flag[mda_flag < 0]
    print(f"\n  [正确剔除] 被CIC剔除且MDA<0: {len(neg)} 个 ({len(neg)/len(mda_flag)*100:.0f}%)")
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
