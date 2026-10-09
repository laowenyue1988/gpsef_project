# -*- coding: utf-8 -*-
"""[P0-1 MDA 内层化 + P1-1 移除 IC 准入] 单元 + 端到端测试 (AST 提取生产源码)

覆盖:
  P1-1a: aggregate_fold_votes(consensus_features=None) → 投票池=全部特征 (全集入池)
  P1-1b: aggregate_fold_votes(consensus_features=['f0','f1']) → 投票池被限制 (对照: 原IC准入行为)
  P1-1c: 生产源码断言: train() 两处投票调用均传 consensus_features=None
  P0-1a: 生产源码断言: MDA 在 inner_oos_set 上算, 不再用 val_set[fold_feats]
  P0-1b: 生产源码断言: 折内训练/特征选择用 inner_train_set
  P0-1c: 端到端: 真实 train() 跑通 (inner-OOS 切分 + MDA + 投票全链路), 选中特征非空
"""
import ast
import os
import sys
import time
import types
import logging
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STRAT_DIR = os.environ.get('TEST_STRATEGY_DIR', 'small_alpha191')
import importlib
ens_mod = importlib.import_module(f'strategies.{_STRAT_DIR}.ensemble')
strat_mod = importlib.import_module(f'strategies.{_STRAT_DIR}.strategy')
GlobalConfig = strat_mod.GlobalConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('test_p0_p1')

ENSEMBLE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'strategies', _STRAT_DIR, 'ensemble.py')


def extract_method_source(path, cls_name, method_name):
    with open(path, encoding='utf-8') as f:
        src = f.read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return ast.get_source_segment(src, item)
    raise RuntimeError(f'{cls_name}.{method_name} 未找到')


def check(tag, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'[{status}] {tag} {detail}')
    return cond


def make_vote_cfg():
    return types.SimpleNamespace(
        use_voting=True, mda_threshold=0.001, mda_use_std_threshold=False,
        mda_t_alpha=0.05, mda_n_repeats=3, vote_min_folds=2, vote_min_features=1,
        top_k_features=10, weight_ic=0.0, weight_mdi=0.3, weight_mda=1.5,
        fallback_to_ic_only=True,
    )


def main():
    ok = True

    # ============ P1-1: IC 准入层移除 ============
    fold_ic = [{f'f{i}': 0.1 for i in range(5)} for _ in range(3)]
    fold_mdi = [{f'f{i}': 10 - i for i in range(5)} for _ in range(3)]
    fold_mda = [{f'f{i}': 0.01 * (i + 1) for i in range(5)} for _ in range(3)]
    cfg = make_vote_cfg()

    # P1-1a: None → 全部特征入投票池
    feats_all, report_all = ens_mod.aggregate_fold_votes(
        fold_ic, fold_mdi, fold_mda, cfg, consensus_features=None, fold_mda_stds=None)
    all_in = all(f'f{i}' in feats_all for i in range(5))
    ok &= check('P1-1a 投票池=全集(None)', all_in, f'(特征={feats_all})')

    # P1-1b: 非 None → 限制在子集 (对照: 原IC准入行为, 证明差异存在)
    feats_sub, _ = ens_mod.aggregate_fold_votes(
        fold_ic, fold_mdi, fold_mda, cfg, consensus_features=['f0', 'f1'],
        fold_mda_stds=None)
    sub_ok = all(f in ('f0', 'f1') for f in feats_sub)
    ok &= check('P1-1b 对照: 子集限制仍有效', sub_ok, f'(特征={feats_sub})')

    # P1-1c: 生产源码断言 投票调用均 None
    # [2026-08-20 回退] O1+ 已弃 (统计无显著差异) → _run_feature_selection_round 不再存在,
    #   投票调用回到 train() 内直接调用, 断言位置同步改回 train()
    train_src = extract_method_source(ENSEMBLE_PATH, 'EnsembleModelManager', 'train')
    n_none_call = train_src.count('consensus_features=None')
    ok &= check('P1-1c 投票调用均None(在 train)', n_none_call == 2,
                f'(consensus_features=None 调用数={n_none_call})')

    # ============ P0-1: MDA 内层化 ============
    pkf_src = extract_method_source(ENSEMBLE_PATH, 'EnsembleModelManager', '_purged_kfold_cv')

    # P0-1a: MDA 数据源 = inner_oos_set, 不再用 val_set[fold_feats]
    mda_inner = pkf_src.count('inner_oos_set[fold_feats]')
    mda_val_old = pkf_src.count('val_set[fold_feats]')
    ok &= check('P0-1a MDA在inner-OOS算', mda_inner == 2 and mda_val_old == 0,
                f'(inner_oos引用={mda_inner}, 残留val_set[fold_feats]={mda_val_old})')

    # P0-1b: 折内训练/特征选择用 inner_train_set
    train_inner = pkf_src.count('inner_train_set')
    oos_split = 'inner_oos_size' in pkf_src and 'inner_oos_dates' in pkf_src
    ok &= check('P0-1b 训练/选择用inner-train', train_inner >= 5 and oos_split,
                f'(inner_train_set引用={train_inner}, 切分逻辑存在={oos_split})')

    # P0-1c: 端到端真实 train() 全链路
    from strategies.small_alpha191.ensemble import EnsembleModelManager
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(7)
        n_days, n_stocks = 150, 60
        dates = pd.date_range('2022-01-04', periods=n_days, freq='B')
        cols = [f'f{i}' for i in range(12)]
        rows = []
        for d in dates:
            f0 = rng.normal(size=n_stocks)
            for i in range(n_stocks):
                rows.append([d, f's{i:04d}', f0[i]] + [rng.normal() for _ in range(11)]
                            + [1 if f0[i] + rng.normal(0, 0.6) > 0 else 0])
        df_big = pd.DataFrame(rows, columns=['trade_date', 'ts_code'] + cols + ['label_profit'])

        cfg_e = GlobalConfig()
        cfg_e.XGB_N_ESTIMATORS = 100
        cfg_e.XGB_MAX_DEPTH = 4
        cfg_e.XGB_LEARNING_RATE = 0.03
        cfg_e.XGB_SUBSAMPLE = 0.6
        cfg_e.XGB_COL_SAMPLE_BY_TREE = 0.7
        cfg_e.XGB_REG_ALPHA = 0.5
        cfg_e.XGB_REG_LAMBDA = 2.0
        cfg_e.XGB_MIN_CHILD_WEIGHT = 25
        cfg_e.XGB_GAMMA = 0.0
        cfg_e.XGB_RANDOM_STATE = 42
        cfg_e.XGB_TREE_METHOD = 'hist'
        cfg_e.XGB_DEVICE = 'cpu'
        cfg_e.XGB_EARLY_STOPPING_ROUNDS = 10
        cfg_e.XGB_EVAL_METRIC = 'auc'
        cfg_e.ENSEMBLE_SEEDS = [42]
        cfg_e.THRESHOLD_SEARCH_POINTS = 10
        cfg_e.cv_folds = 2
        cfg_e.cv_embargo_days = 0
        cfg_e.cv_feature_consensus = 1
        cfg_e.cv_ic_psi_split_ratio = 0.5
        cfg_e.PSI_THRESHOLD = 1.0
        cfg_e.MIN_FEAT_IC = 0.0
        cfg_e.NAN_THRESHOLD = 0.9
        cfg_e.MAX_CORR = 0.999
        cfg_e.MIN_FEATURES = 1
        cfg_e.MAX_FEATURES = 12
        cfg_e.use_voting = True
        cfg_e.mda_enabled = True
        cfg_e.mda_n_repeats = 2
        cfg_e.mda_threshold = 0.0
        cfg_e.mda_use_std_threshold = False
        cfg_e.mda_t_alpha = 0.05
        cfg_e.mda_inner_oos_frac = 0.15
        cfg_e.mda_inner_oos_min_days = 20
        cfg_e.vote_min_folds = 1
        cfg_e.vote_min_features = 1
        cfg_e.top_k_features = 12
        cfg_e.weight_ic = 0.0
        cfg_e.weight_mdi = 0.3
        cfg_e.weight_mda = 1.5
        cfg_e.fallback_to_ic_only = True
        cfg_e.cluster_enabled = False
        cfg_e.cluster_diagnose = False
        cfg_e.MAX_HOLD_DAYS = 20
        cfg_e.AUC_THRESHOLD = 0.5
        cfg_e.CALIBRATION_METHOD = 'none'
        cfg_e.USE_CALIBRATION = False
        cfg_e.VAL_SPLIT_RATIO = 0.1
        cfg_e.lightgbm = {'enabled': False}
        cfg_e.PROB_TEMPERATURE = 1.0
        cfg_e.hyperparam_search = {'enabled': False}   # 跳过超参选择, 专注 PKF/MDA/投票链

        mgr = EnsembleModelManager(model_dir=os.path.join(tmp, 'meta_models'), config=cfg_e)
        mgr.class_balance_mode = 'auto'
        t0 = time.time()
        try:
            success = mgr.train(df_train=df_big, feature_cols=cols,
                                target_col='label_profit', current_date='2022-06-30')
            dt = time.time() - t0
            ok &= check('P0-1c 端到端train成功', success is True, f'(耗时{dt:.1f}s)')
            ok &= check('P0-1c 选中特征非空', len(getattr(mgr, 'selected_features', [])) > 0,
                        f'(n={len(getattr(mgr, "selected_features", []))})')
            ok &= check('P0-1c 模型已训练', len(getattr(mgr, 'xgb_models', [])) > 0,
                        f'(n_xgb={len(getattr(mgr, "xgb_models", []))})')
        except Exception as e:
            import traceback
            traceback.print_exc()
            ok &= check('P0-1c 端到端train成功', False, str(e))

    print('\n==== P0-1/P1-1 测试完成: %s ====' % ('ALL PASS' if ok else '有失败!'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
