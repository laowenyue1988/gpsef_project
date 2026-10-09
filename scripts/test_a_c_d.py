# -*- coding: utf-8 -*-
"""[A+C+D] small_alpha191 修复专项测试 (AST 源码断言 + 端到端)

背景 (用户 2026-08-18 回测日志质疑, 三项均已修复):
  A: 折集成多样性硬编码 [6,6,7,7,8] 覆盖超参选择选中值 → 改为围绕选中超参扰动
  C: HP-Search 评估协议无 LRWarmup, 生产带 warmup → 评估加 LRWarmup 与生产一致
  D: 门控"生产模型 val AUC"实为 PKF 折均值别名 (误导文案+双轨) → 门控口径统一为折集成 meta_AUC
  另: config strategy 段误指向 small_alpha191_test → 修复为自身

覆盖:
  A1 源码: 折集成多样性由 config.XGB_* 派生, 无硬编码 [6,6,7,7,8]
  A2 端到端: 设 XGB_MAX_DEPTH=3 → 折集成 depth 从 3 起 (日志)
  C1 源码: _select_hyperparams 训练带 LRWarmupCallback
  D1 源码: ensemble 门控文案/口径 (折集成 meta_AUC)
  D2 源码: strategy 门控文案 (折集成 meta_AUC)
  D3 端到端: train() 后 meta_auc = 折集成 OOS 平均 (非 PKF 均值别名)
  E1 config: strategy 段指向自身, 无 test 残留
"""
import ast
import os
import sys
import time
import logging
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
ens_mod = importlib.import_module('strategies.small_alpha191.ensemble')
strat_mod = importlib.import_module('strategies.small_alpha191.strategy')
GlobalConfig = strat_mod.GlobalConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('test_a_c_d')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENS_PATH = os.path.join(BASE, 'strategies', 'small_alpha191', 'ensemble.py')
STRAT_PATH = os.path.join(BASE, 'strategies', 'small_alpha191', 'strategy.py')
CFG_PATH = os.path.join(BASE, 'strategies', 'small_alpha191', 'config.yaml')


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


def make_data(n_days=150, n_stocks=60, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2022-01-04', periods=n_days, freq='B')
    cols = [f'f{i}' for i in range(12)]
    rows = []
    for d in dates:
        f0 = rng.normal(size=n_stocks)
        for i in range(n_stocks):
            rows.append([d, f's{i:04d}', f0[i]] + [rng.normal() for _ in range(11)]
                        + [1 if f0[i] + rng.normal(0, 0.6) > 0 else 0])
    return pd.DataFrame(rows, columns=['trade_date', 'ts_code'] + cols + ['label_profit'])


def make_cfg(**over):
    cfg = GlobalConfig()
    cfg.XGB_N_ESTIMATORS = 100
    cfg.XGB_MAX_DEPTH = 4
    cfg.XGB_LEARNING_RATE = 0.03
    cfg.XGB_SUBSAMPLE = 0.6
    cfg.XGB_COL_SAMPLE_BY_TREE = 0.7
    cfg.XGB_REG_ALPHA = 0.5
    cfg.XGB_REG_LAMBDA = 2.0
    cfg.XGB_MIN_CHILD_WEIGHT = 25
    cfg.XGB_GAMMA = 0.0
    cfg.XGB_RANDOM_STATE = 42
    cfg.XGB_TREE_METHOD = 'hist'
    cfg.XGB_DEVICE = 'cpu'
    cfg.XGB_EARLY_STOPPING_ROUNDS = 10
    cfg.XGB_EVAL_METRIC = 'auc'
    cfg.ENSEMBLE_SEEDS = [42]
    cfg.THRESHOLD_SEARCH_POINTS = 10
    cfg.cv_folds = 2
    cfg.cv_embargo_days = 0
    cfg.cv_feature_consensus = 1
    cfg.cv_ic_psi_split_ratio = 0.5
    cfg.PSI_THRESHOLD = 1.0
    cfg.MIN_FEAT_IC = 0.0
    cfg.NAN_THRESHOLD = 0.9
    cfg.MAX_CORR = 0.999
    cfg.MIN_FEATURES = 1
    cfg.MAX_FEATURES = 12
    cfg.use_voting = True
    cfg.mda_enabled = True
    cfg.mda_n_repeats = 2
    cfg.mda_threshold = 0.0
    cfg.mda_use_std_threshold = False
    cfg.mda_t_alpha = 0.05
    cfg.mda_inner_oos_frac = 0.15
    cfg.mda_inner_oos_min_days = 20
    cfg.vote_min_folds = 1
    cfg.vote_min_features = 1
    cfg.top_k_features = 12
    cfg.weight_ic = 0.0
    cfg.weight_mdi = 0.3
    cfg.weight_mda = 1.5
    cfg.fallback_to_ic_only = True
    cfg.cluster_enabled = False
    cfg.cluster_diagnose = False
    cfg.MAX_HOLD_DAYS = 20
    cfg.AUC_THRESHOLD = 0.5
    cfg.CALIBRATION_METHOD = 'none'
    cfg.USE_CALIBRATION = False
    cfg.VAL_SPLIT_RATIO = 0.1
    cfg.lightgbm = {'enabled': False}
    cfg.PROB_TEMPERATURE = 1.0
    cfg.hyperparam_search = {'enabled': False}
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def main():
    ok = True
    ens_src = open(ENS_PATH, encoding='utf-8').read()
    strat_src = open(STRAT_PATH, encoding='utf-8').read()
    cfg_src = open(CFG_PATH, encoding='utf-8').read()

    # ============ A: 折集成多样性围绕选中超参 ============
    cv_src = extract_method_source(ENS_PATH, 'EnsembleModelManager', '_train_cv_fold_ensemble')
    a_hardcode_gone = '[6, 6, 7, 7, 8]' not in cv_src
    a_derived = 'depth_list = [_sel_d' in cv_src and '_sel_ss' in cv_src and '_sel_cs' in cv_src
    a_clip = 'max(2, d)' in cv_src and '0.3, min(1.0' in cv_src
    ok &= check('A1 多样性由选中超参派生', a_hardcode_gone and a_derived and a_clip,
                f'(硬编码移除={a_hardcode_gone}, 派生={a_derived}, 越界保护={a_clip})')

    # A2 端到端: depth=3 → 折集成 depth 从 3 起 (捕获日志)
    import io
    import logging as _lg
    from strategies.small_alpha191.ensemble import EnsembleModelManager
    buf = io.StringIO()
    h = _lg.StreamHandler(buf)
    h.setLevel(_lg.INFO)
    ens_logger = _lg.getLogger('strategies.small_alpha191.ensemble')
    ens_logger.addHandler(h)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            df = make_data()
            cols = [c for c in df.columns if c.startswith('f')]
            mgr = EnsembleModelManager(model_dir=os.path.join(tmp, 'mm'),
                                       config=make_cfg(XGB_MAX_DEPTH=3))
            mgr.class_balance_mode = 'auto'
            success = mgr.train(df_train=df, feature_cols=cols,
                                target_col='label_profit', current_date='2022-06-30')
            logtxt = buf.getvalue()
            depth_log = 'depth=[3, 3, 4, 4, 5]' in logtxt or 'depth=[3' in logtxt
            ok &= check('A2 端到端: 折集成depth从选中值起', success is True and depth_log,
                        f'(depth日志={"存在" if depth_log else "缺失"})')
    finally:
        ens_logger.removeHandler(h)

    # ============ C: HP-Search 评估带 LRWarmup ============
    hp_src = extract_method_source(ENS_PATH, 'EnsembleModelManager', '_select_hyperparams')
    c_warmup = 'LRWarmupCallback' in hp_src and 'target_lr=float(cand[\'xgb_learning_rate\'])' in hp_src
    ok &= check('C1 HP-Search 评估带 LRWarmup', c_warmup)

    # ============ D: 门控口径统一 ============
    d_no_mislead = '用于AUC门控' not in ens_src
    d_unified = 'prod_val_auc = float(self.meta_auc)' in ens_src
    d_gate_log = '[D] 折集成 meta_AUC' in strat_src or '折集成 meta_AUC' in strat_src
    d_pkf_placeholder = '仅初始化占位' in ens_src
    ok &= check('D1 ensemble 门控口径统一', d_no_mislead and d_unified and d_pkf_placeholder,
                f'(误导文案移除={d_no_mislead}, 口径统一={d_unified}, 占位标注={d_pkf_placeholder})')
    ok &= check('D2 strategy 门控文案=折集成', d_gate_log)

    # D3 端到端: meta_auc = 折集成 OOS (非 PKF 均值别名)
    # 验证: 折集成 meta_AUC = np.average(cv_aucs) → mgr.meta_auc 应 > 0.5 且与 xgb_val_auc 一致
    with tempfile.TemporaryDirectory() as tmp:
        df = make_data()
        cols = [c for c in df.columns if c.startswith('f')]
        mgr = EnsembleModelManager(model_dir=os.path.join(tmp, 'mm2'), config=make_cfg())
        mgr.class_balance_mode = 'auto'
        success = mgr.train(df_train=df, feature_cols=cols,
                            target_col='label_profit', current_date='2022-06-30')
        ma = float(mgr.meta_auc)
        ok &= check('D3 门控=折集成OOS AUC', success and 0.5 < ma <= 1.0
                    and abs(ma - float(mgr.xgb_val_auc)) < 1e-9,
                    f'(meta_auc={ma:.4f}, xgb_val_auc={mgr.xgb_val_auc:.4f})')

    # ============ E: config 误引用修复 ============
    e_self = 'class: strategies.small_alpha191.strategy.SmallAlpha191Strategy' in cfg_src
    e_no_test = 'small_alpha191_test' not in cfg_src
    ok &= check('E1 config 指向自身', e_self and e_no_test)

    print('\n==== A+C+D 测试完成: %s ====' % ('ALL PASS' if ok else '有失败!'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
