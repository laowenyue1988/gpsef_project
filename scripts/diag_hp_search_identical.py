# -*- coding: utf-8 -*-
"""诊断: HP-Search 4 组候选 val AUC/best_iter 完全一致的根因复现

假设: 弱信号数据 + early_stopping(20) 在 val AUC 高方差下过早触发
      → 4 组候选都只训练 ~30 轮(欠拟合) → 都学到同一弱主导分裂 → 预测几乎相同
验证:
  1. 构造弱信号数据 (大部分特征纯噪声 + 1 个弱信号)
  2. 用与 _select_hyperparams 完全相同的逻辑 (时序切分 + 早停20) 评估 c1-c4
  3. 检查: 4 组 AUC 是否相同? best_iter 是否相同? 预测相关性?
  4. 对照组: 固定轮数训练 (100轮, 无早停) → 4 组 AUC 应出现差异
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import strategies.small_alpha191_test.ensemble as ens_mod
from sklearn.metrics import roc_auc_score


def make_weak_signal_data(n_days=240, n_stocks=300, n_feats=158, signal=0.10, seed=42):
    """模拟真实小盘策略数据: 158 特征, 大部分噪声, 1 个弱信号, 正样本率~0.60"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2022-01-04', periods=n_days, freq='B')
    cols = [f'alpha_{i:03d}_neu' for i in range(n_feats)]
    rows = []
    for d in dates:
        # 弱信号: f0 与标签相关 (信号强度 0.10)
        f0 = rng.normal(size=n_stocks)
        for i in range(n_stocks):
            label = 1 if f0[i] + rng.normal(0, 1.0 / signal) > 0.05 else 0
            rec = [d, f's{i:04d}', f0[i]]
            rec += [rng.normal() for _ in range(n_feats - 1)]  # 纯噪声
            rec.append(label)
            rows.append(rec)
    df = pd.DataFrame(rows, columns=['trade_date', 'ts_code'] + cols + ['label_profit'])
    return df


def run_hp_search(df, candidates, val_size=20, es_rounds=20, use_early_stop=True):
    """复刻 _select_hyperparams 的核心评估逻辑"""
    feats = [c for c in df.columns if c.startswith('alpha_')]
    unique_dates = sorted(df['trade_date'].unique())
    val_dates = set(unique_dates[-val_size:])
    tr_mask = ~df['trade_date'].isin(val_dates)
    X_tr = df.loc[tr_mask, feats].astype(np.float32)
    y_tr = df.loc[tr_mask, 'label_profit'].values
    X_va = df.loc[~tr_mask, feats].astype(np.float32)
    y_va = df.loc[~tr_mask, 'label_profit'].values
    dtrain = xgb.DMatrix(X_tr, label=y_tr, missing=np.nan, feature_names=feats)
    dval = xgb.DMatrix(X_va, label=y_va, missing=np.nan, feature_names=feats)

    results = []
    preds_all = []
    for cand in candidates:
        params = {
            'max_depth': cand['xgb_max_depth'], 'eta': cand['xgb_learning_rate'],
            'subsample': cand['xgb_subsample'], 'colsample_bytree': cand['xgb_col_sample_by_tree'],
            'reg_alpha': cand['xgb_reg_alpha'], 'reg_lambda': cand['xgb_reg_lambda'],
            'min_child_weight': cand['xgb_min_child_weight'], 'gamma': cand['xgb_gamma'],
            'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': 42,
            'tree_method': 'hist', 'device': 'cpu', 'nthread': 1,
        }
        cap = int(cand['xgb_n_estimators'])
        if use_early_stop:
            model = xgb.train(params, dtrain, num_boost_round=cap,
                              evals=[(dtrain, 'train'), (dval, 'val')],
                              early_stopping_rounds=es_rounds, verbose_eval=False)
            best_iter = getattr(model, 'best_iteration', cap)
            preds = model.predict(dval)
        else:
            # 对照组: 固定轮数, 无早停
            n_rounds = min(cap, 100)
            model = xgb.train(params, dtrain, num_boost_round=n_rounds, verbose_eval=False)
            best_iter = n_rounds
            preds = model.predict(dval)
        auc = float(roc_auc_score(y_va, preds))
        results.append((cand['name'], auc, best_iter))
        preds_all.append(preds)
    return results, preds_all, y_va


CANDIDATES = [
    {'name': 'c1_conservative', 'xgb_max_depth': 3, 'xgb_n_estimators': 300,
     'xgb_learning_rate': 0.05, 'xgb_subsample': 0.6, 'xgb_col_sample_by_tree': 0.6,
     'xgb_reg_alpha': 0.5, 'xgb_reg_lambda': 3.0, 'xgb_min_child_weight': 30, 'xgb_gamma': 0.1},
    {'name': 'c2_baseline_s', 'xgb_max_depth': 4, 'xgb_n_estimators': 400,
     'xgb_learning_rate': 0.03, 'xgb_subsample': 0.6, 'xgb_col_sample_by_tree': 0.7,
     'xgb_reg_alpha': 0.5, 'xgb_reg_lambda': 2.0, 'xgb_min_child_weight': 25, 'xgb_gamma': 0.0},
    {'name': 'c3_balanced', 'xgb_max_depth': 5, 'xgb_n_estimators': 500,
     'xgb_learning_rate': 0.025, 'xgb_subsample': 0.7, 'xgb_col_sample_by_tree': 0.7,
     'xgb_reg_alpha': 0.3, 'xgb_reg_lambda': 1.5, 'xgb_min_child_weight': 20, 'xgb_gamma': 0.05},
    {'name': 'c4_strong', 'xgb_max_depth': 6, 'xgb_n_estimators': 600,
     'xgb_learning_rate': 0.02, 'xgb_subsample': 0.7, 'xgb_col_sample_by_tree': 0.8,
     'xgb_reg_alpha': 0.2, 'xgb_reg_lambda': 1.0, 'xgb_min_child_weight': 15, 'xgb_gamma': 0.0},
]


def main():
    print('=== 实验1: 弱信号 + early_stopping(20) [复刻生产 HP-Search] ===')
    df = make_weak_signal_data(signal=0.10, seed=7)
    print(f'数据: {len(df)} 样本, {len([c for c in df.columns if c.startswith("alpha_")])} 特征, '
          f'正样本率={df["label_profit"].mean():.3f}')
    res, preds, y_va = run_hp_search(df, CANDIDATES, use_early_stop=True)
    for name, auc, bi in res:
        print(f'  {name}: val AUC={auc:.4f} best_iter={bi}')
    aucs = [r[1] for r in res]
    its = [r[2] for r in res]
    same_auc = len(set(aucs)) == 1
    same_iter = len(set(its)) == 1
    corr = np.corrcoef(preds[0], preds[1])[0, 1]
    print(f'  4组AUC全同: {same_auc} | best_iter全同: {same_iter} | '
          f'c1/c2预测相关: {corr:.6f}')

    print('\n=== 实验2: 弱信号 + 固定100轮(无早停) [对照组] ===')
    res2, preds2, _ = run_hp_search(df, CANDIDATES, use_early_stop=False)
    for name, auc, bi in res2:
        print(f'  {name}: val AUC={auc:.4f} best_iter={bi}')
    aucs2 = [r[1] for r in res2]
    print(f'  4组AUC全同: {len(set(aucs2)) == 1} | AUC范围: {min(aucs2):.4f}~{max(aucs2):.4f}')

    print('\n=== 实验3: 强信号 + early_stopping(20) [对照: 信号强时早停是否还一致] ===')
    df_s = make_weak_signal_data(signal=0.50, seed=7)
    res3, _, _ = run_hp_search(df_s, CANDIDATES, use_early_stop=True)
    for name, auc, bi in res3:
        print(f'  {name}: val AUC={auc:.4f} best_iter={bi}')
    aucs3 = [r[1] for r in res3]
    print(f'  4组AUC全同: {len(set(aucs3)) == 1}')


if __name__ == '__main__':
    main()
