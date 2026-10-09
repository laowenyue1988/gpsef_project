# -*- coding: utf-8 -*-
"""[超参选择] _select_hyperparams 单元测试 (AST 提取生产源码 + SimpleNamespace mock self)
+ 小规模端到端冒烟: EnsembleModelManager.train() 全链路 (含超参选择阶段)

覆盖:
  H1 enabled=false → (None, None), config 不变
  H2 candidates 为空 → (None, None)
  H3 正常选择: 返回选中候选名, config XGB_* 写回与选中候选一致, score 合理
  H4 全部候选缺参数失败 → (None, None), config 保持原值 (降级保护)
  H5 端到端: train() 超参选择阶段正常执行且不改变最终训练成功
"""
import ast
import os
import sys
import types
import time
import logging
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('test_hyperparam_search')

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STRAT_DIR = os.environ.get('TEST_STRATEGY_DIR', 'small_alpha191')
ENSEMBLE_PATH = os.path.join(_PROJ_ROOT, 'strategies', _STRAT_DIR, 'ensemble.py')
CLS_NAME = 'EnsembleModelManager'
METH_NAME = '_select_hyperparams'

import xgboost as xgb  # noqa: E402
import importlib  # noqa: E402
ens_mod = importlib.import_module(f'strategies.{_STRAT_DIR}.ensemble')
strat_mod = importlib.import_module(f'strategies.{_STRAT_DIR}.strategy')
GlobalConfig = strat_mod.GlobalConfig


def extract_method_source(path, cls_name, method_name):
    with open(path, encoding='utf-8') as f:
        src = f.read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return ast.get_source_segment(src, item)
    raise RuntimeError(f'{cls_name}.{method_name} 未找到: {path}')


_SRC = extract_method_source(ENSEMBLE_PATH, CLS_NAME, METH_NAME)
_ns = {
    'np': np, 'xgb': xgb, 'time': time, 'logger': logger,
    '_detect_objective_mode': ens_mod._detect_objective_mode,
    '_objective_metric_fn': ens_mod._objective_metric_fn,
    'LRWarmupCallback': ens_mod.LRWarmupCallback,  # [C 修复 2026-08-18] HP-Search 评估带 warmup
}
exec(compile(_SRC, f'{CLS_NAME}.{METH_NAME}', 'exec'), _ns)
_select_hyperparams = _ns[METH_NAME]


def make_cfg(hp_cfg):
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
    # 评估特征筛选 (select_stable_features) 所需属性
    cfg.PSI_THRESHOLD = 1.0
    cfg.MIN_FEAT_IC = 0.0
    cfg.NAN_THRESHOLD = 0.9
    cfg.MAX_CORR = 0.999
    cfg.MIN_FEATURES = 1
    cfg.MAX_FEATURES = 12
    cfg.cv_ic_psi_split_ratio = 0.5
    cfg.MAX_HOLD_DAYS = 20
    cfg.hyperparam_search = hp_cfg
    return cfg


def make_data(n_days=120, n_stocks=80, seed=42, n_feats=12, reverse_from_day=None):
    """reverse_from_day: 从第 N 天起标签取反 (模拟特征方向漂移 → 该段 val AUC<0.5)"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2022-01-04', periods=n_days, freq='B')
    cols = [f'f{i}' for i in range(n_feats)]
    rows = []
    for d in dates:
        f0 = rng.normal(size=n_stocks)
        day_idx = len(rows) // n_stocks  # 当前第几天
        reversed_ = reverse_from_day is not None and day_idx >= reverse_from_day
        for i in range(n_stocks):
            rec = [d, f's{i:04d}', f0[i]] + [rng.normal() for _ in range(n_feats - 1)]
            if reversed_:
                rec.append(1 if -f0[i] + rng.normal(0, 0.6) > 0 else 0)  # 反向
            else:
                rec.append(1 if f0[i] + rng.normal(0, 0.6) > 0 else 0)
            rows.append(rec)
    df = pd.DataFrame(rows, columns=['trade_date', 'ts_code'] + cols + ['label_profit'])
    return df


def make_feats(n_feats=12):
    return [f'f{i}' for i in range(n_feats)]


CANDIDATES = [
    {'name': 'fast_small', 'xgb_max_depth': 2, 'xgb_n_estimators': 30,
     'xgb_learning_rate': 0.1, 'xgb_subsample': 0.8, 'xgb_col_sample_by_tree': 0.8,
     'xgb_reg_alpha': 0.0, 'xgb_reg_lambda': 1.0, 'xgb_min_child_weight': 5,
     'xgb_gamma': 0.0},
    {'name': 'mid', 'xgb_max_depth': 4, 'xgb_n_estimators': 50,
     'xgb_learning_rate': 0.05, 'xgb_subsample': 0.8, 'xgb_col_sample_by_tree': 0.8,
     'xgb_reg_alpha': 0.0, 'xgb_reg_lambda': 1.0, 'xgb_min_child_weight': 2,
     'xgb_gamma': 0.0},
    {'name': 'deep_big', 'xgb_max_depth': 6, 'xgb_n_estimators': 80,
     'xgb_learning_rate': 0.05, 'xgb_subsample': 0.8, 'xgb_col_sample_by_tree': 0.8,
     'xgb_reg_alpha': 0.0, 'xgb_reg_lambda': 0.5, 'xgb_min_child_weight': 1,
     'xgb_gamma': 0.0},
]


def check(tag, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'[{status}] {tag} {detail}')
    return cond


def main():
    ok = True
    df = make_data()
    feats = make_feats()

    # ---- H1: enabled=false → 跳过 ----
    cfg = make_cfg({'enabled': False, 'candidates': CANDIDATES})
    self_mock = types.SimpleNamespace(config=cfg, objective_mode='binary')
    best, score = _select_hyperparams(self_mock, df, feats, 'label_profit')
    ok &= check('H1 enabled=false 跳过', best is None and score is None
                and cfg.XGB_MAX_DEPTH == 4, f'(best={best})')

    # ---- H2: candidates 为空 ----
    cfg = make_cfg({'enabled': True, 'candidates': []})
    self_mock = types.SimpleNamespace(config=cfg, objective_mode='binary')
    best, score = _select_hyperparams(self_mock, df, feats, 'label_profit')
    ok &= check('H2 候选为空跳过', best is None and score is None)

    # ---- H3: 正常选择 ----
    cfg = make_cfg({'enabled': True, 'val_size_days': 15,
                    'early_stopping_rounds': 10, 'candidates': CANDIDATES})
    self_mock = types.SimpleNamespace(config=cfg, objective_mode='binary')
    import warnings
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter('always')
        t0 = time.time()
        best, score = _select_hyperparams(self_mock, df, feats, 'label_profit')
    dt = time.time() - t0
    ok &= check('H3 返回选中候选', best in [c['name'] for c in CANDIDATES],
                f'(best={best}, score={score:.4f}, 耗时{dt:.1f}s)')
    ok &= check('H3 score 合理', score is not None and 0.0 <= score <= 1.0)
    # [2026-08-18 回归] 候选超参必须真正传给 xgboost:
    # 原 bug: params 用 xgb_* 键名 → xgboost 忽略全部候选超参 (WARNING: are not used)
    #       → 4 组候选全用默认参数 → AUC/best_iter 全同 → 选择失效
    not_used = [str(x.message) for x in wlist if 'are not used' in str(x.message)]
    ok &= check('H3 无忽略参数警告(bug回归)', len(not_used) == 0,
                f'(忽略参数警告数={len(not_used)})')
    # [2026-08-18 回归] 候选间必须有区分度 (修复前 4 组 AUC 完全相同到 4 位小数):
    hp_res = getattr(self_mock, '_hp_search_results', [])
    if hp_res:
        aucs = [r[1] for r in hp_res]
        spread = max(aucs) - min(aucs)
        # 测试数据信号强 → 候选差异在第 4 位小数; 修复前差异=0.0000
        ok &= check('H3 候选AUC有区分度', spread > 0.0001,
                    f'(候选AUC={[round(a, 4) for a in aucs]}, spread={spread:.4f})')
    else:
        ok &= check('H3 候选AUC有区分度', False, '(无 _hp_search_results)')
    # 写回一致性: config 与选中候选完全一致
    best_cand = next(c for c in CANDIDATES if c['name'] == best)
    mapping = {
        'xgb_n_estimators': 'XGB_N_ESTIMATORS', 'xgb_max_depth': 'XGB_MAX_DEPTH',
        'xgb_learning_rate': 'XGB_LEARNING_RATE', 'xgb_subsample': 'XGB_SUBSAMPLE',
        'xgb_col_sample_by_tree': 'XGB_COL_SAMPLE_BY_TREE',
        'xgb_reg_alpha': 'XGB_REG_ALPHA', 'xgb_reg_lambda': 'XGB_REG_LAMBDA',
        'xgb_min_child_weight': 'XGB_MIN_CHILD_WEIGHT', 'xgb_gamma': 'XGB_GAMMA',
    }
    consistent = all(int(getattr(cfg, a)) == int(best_cand[k]) if k in ('xgb_n_estimators', 'xgb_max_depth', 'xgb_min_child_weight')
                     else abs(float(getattr(cfg, a)) - float(best_cand[k])) < 1e-9
                     for k, a in mapping.items())
    ok &= check('H3 config 写回与选中候选一致', consistent,
                f'(XGB_MAX_DEPTH={cfg.XGB_MAX_DEPTH}, '
                f'XGB_N_ESTIMATORS={cfg.XGB_N_ESTIMATORS}, '
                f'XGB_REG_LAMBDA={cfg.XGB_REG_LAMBDA}, '
                f'XGB_MIN_CHILD_WEIGHT={cfg.XGB_MIN_CHILD_WEIGHT})')

    # ---- H4: 全部候选缺参数 → 降级保护 ----
    bad_cands = [{'name': 'bad1', 'xgb_max_depth': 3}]  # 缺 8 个参数
    cfg = make_cfg({'enabled': True, 'candidates': bad_cands})
    self_mock = types.SimpleNamespace(config=cfg, objective_mode='binary')
    best, score = _select_hyperparams(self_mock, df, feats, 'label_profit')
    ok &= check('H4 全失败保持原超参', best is None and score is None
                and cfg.XGB_MAX_DEPTH == 4, f'(best={best})')

    # ---- H6: [2026-08-18 防护] 全部候选平均 val AUC<0.5 → 不写回超参 ----
    # 场景: 选择窗(训练段尾部)标签反向 (特征方向漂移) → 各窗模型学到正信号在窗 val 上反向
    # 注意: 反向必须发生在"选择窗"(sel_dates 尾部), 生产val段已被[A]隔离
    df_rev = make_data(n_days=150, seed=11, reverse_from_day=60)
    cfg = make_cfg({'enabled': True, 'eval_windows': 2,
                    'eval_rounds': 100, 'candidates': CANDIDATES})
    self_mock = types.SimpleNamespace(config=cfg, objective_mode='binary')
    best, score = _select_hyperparams(self_mock, df_rev, feats, 'label_profit')
    hp_res6 = getattr(self_mock, '_hp_search_results', [])
    all_below = all(s < 0.5 for _, s, _ in hp_res6) if hp_res6 else True
    ok &= check('H6 全部AUC<0.5不写回', best is None and score is None
                and cfg.XGB_MAX_DEPTH == 4 and all_below,
                f'(best={best}, hp_aucs={[round(a, 3) for _, a, _ in hp_res6]})')

    # ---- H7: [A 隔离验证] 生产val段(末20天)标签反向 → 选择结果应完全不变 ----
    # A 修复前: 选择用末20天 → 反向标签 → 选择结果改变 (偏倚证据)
    # A 修复后: 末20天零接触 → 两次选择结果完全一致 (seed 固定, 确定性)
    df_a1 = make_data(n_days=150, seed=21)
    df_a2 = make_data(n_days=150, seed=21, reverse_from_day=130)  # 仅末20天反向
    cfg_a = make_cfg({'enabled': True, 'eval_windows': 2, 'candidates': CANDIDATES})
    m1 = types.SimpleNamespace(config=make_cfg({'enabled': True, 'eval_windows': 2,
                                                'candidates': CANDIDATES}), objective_mode='binary')
    m2 = types.SimpleNamespace(config=make_cfg({'enabled': True, 'eval_windows': 2,
                                                'candidates': CANDIDATES}), objective_mode='binary')
    b1, s1 = _select_hyperparams(m1, df_a1, feats, 'label_profit')
    b2, s2 = _select_hyperparams(m2, df_a2, feats, 'label_profit')
    ok &= check('H7 生产val反向不影响选择(A隔离)', b1 is not None and b1 == b2
                and abs(s1 - s2) < 1e-9, f'(best1={b1}, best2={b2}, s1={s1:.4f}, s2={s2:.4f})')

    # ---- H8: [B 多窗验证] 每候选评估 = eval_windows 窗平均, 窗间分数独立 ----
    hp_det = getattr(m1, '_hp_search_detail', {})
    if b1 and b1 in hp_det:
        det = hp_det[b1]
        n_w = len(det.get('scores', []))
        n_r = len(det.get('rounds', []))
        ok &= check('H8 多窗评估生效(W窗)', n_w == 2 and n_r == 2,
                    f'(best={b1}, scores={det.get("scores")}, rounds={det.get("rounds")})')
        # 窗间分数应有差异 (不同时间窗信号不同)
        ok &= check('H8 窗间分数独立', len(set(round(s, 4) for s in det.get('scores', []))) > 1,
                    f'(scores={[round(s, 4) for s in det.get("scores", [])]})')
    else:
        ok &= check('H8 多窗评估生效(W窗)', False, '(无 detail)')

    # ---- H9: [C 早停验证] 每候选每窗独立早停 → 不同候选轮数不全同 ----
    if hp_det:
        rounds_by_cand = {n: d.get('rounds', []) for n, d in hp_det.items() if d.get('rounds')}
        cap_by_cand = {c['name']: min(max(int(c['xgb_n_estimators']), 50), 600)
                       for c in CANDIDATES}  # 与生产 _select_hyperparams 的 cap 逻辑一致
        # 早停: 每窗轮数 ≤ 该候选轮数上限 (候选 n_estimators)
        under_cap = all(r <= cap_by_cand[n] for n, rr in rounds_by_cand.items() for r in rr)
        # 不同候选轮数不全同 (lr 不同 → 早停点不同, 证明 C 生效而非固定轮数)
        rounds_unique = len(set(tuple(rr) for rr in rounds_by_cand.values())) > 1
        ok &= check('H9 每候选独立早停', under_cap and rounds_unique,
                    f'(rounds_by_cand={rounds_by_cand})')
    else:
        ok &= check('H9 每候选独立早停', False, '(无 detail)')

    # ---- H5: 端到端冒烟 (真实 train(), 超参选择融入主链路) ----
    try:
        EnsembleModelManager = ens_mod.EnsembleModelManager
    except Exception as e:
        logger.warning(f'H5 导入失败: {e}')
        ok &= check('H5 ensemble 导入', False, str(e))
        print('\n==== 超参选择测试完成: %s ====' % ('ALL PASS' if ok else '有失败!'))
        return 1 if not ok else 0

    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_cfg({'enabled': True, 'val_size_days': 15,
                        'early_stopping_rounds': 10, 'candidates': CANDIDATES})
        # 端到端所需其它配置
        cfg.cv_folds = 2
        cfg.cv_embargo_days = 0
        cfg.cv_feature_consensus = 1
        cfg.cv_ic_psi_split_ratio = 0.5
        cfg.PSI_THRESHOLD = 1.0          # 小数据分布漂移容忍
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
        cfg.XGB_EARLY_STOPPING_ROUNDS = 10
        cfg.XGB_EVAL_METRIC = 'auc'
        cfg.ENSEMBLE_SEEDS = [42]
        cfg.THRESHOLD_SEARCH_POINTS = 10
        cfg.lightgbm = {'enabled': False}
        cfg.PROB_TEMPERATURE = 1.0

        df_big = make_data(n_days=150, n_stocks=60, seed=7)
        mgr = EnsembleModelManager(model_dir=os.path.join(tmp, 'meta_models'), config=cfg)
        mgr.class_balance_mode = 'auto'
        t0 = time.time()
        try:
            success = mgr.train(df_train=df_big,
                                feature_cols=make_feats(),
                                target_col='label_profit',
                                current_date='2022-06-30')
            dt = time.time() - t0
            ok &= check('H5 train() 端到端成功', success is True, f'(耗时{dt:.1f}s)')
            ok &= check('H5 选中特征非空', len(getattr(mgr, 'selected_features', [])) > 0,
                        f'(features={getattr(mgr, "selected_features", [])})')
            ok &= check('H5 模型已训练', len(getattr(mgr, 'xgb_models', [])) > 0,
                        f'(n_xgb={len(getattr(mgr, "xgb_models", []))})')
            # [2026-08-18 train AUC 修复验证] 修复前 predict_proba(df_train) 与 y_prod_train
            # 长度不匹配 → roc_auc_score 抛错被吞 → train AUC 恒 0.5。
            # 手工重算: 只对生产训练段(末20天为val)预测, 应得到真实非 0.5 AUC。
            try:
                from sklearn.metrics import roc_auc_score
                _dates = sorted(df_big['trade_date'].unique())
                _val_dates = set(_dates[-20:])
                _df_prod_tr = df_big[~df_big['trade_date'].isin(_val_dates)]
                _y_prod = _df_prod_tr['label_profit'].values
                _sel = [f for f in mgr.selected_features if f in _df_prod_tr.columns]
                _proba = mgr.predict_proba(_df_prod_tr[_sel]).flatten()
                _auc_manual = float(roc_auc_score(_y_prod, _proba))
                ok &= check('H5 train AUC 修复生效(非0.5)', 0.5 < _auc_manual <= 1.0,
                            f'(train AUC={_auc_manual:.4f}, n={len(_y_prod)})')
            except Exception as e:
                ok &= check('H5 train AUC 修复生效(非0.5)', False, str(e))
        except Exception as e:
            import traceback
            traceback.print_exc()
            ok &= check('H5 train() 端到端成功', False, str(e))

        # ---- H5b: [2026-08-18 sample_weight 对齐修复] 带权重的端到端 ----
        # 修复前: 乱序 df 排序后权重未重排 → 权重错位 (张冠李戴)
        # 修复后: align_sample_weight_to_sort 重排 → 权重-样本对齐
        # 验证: 乱序数据 + 唯一权重 → train 成功 + 权重进入 DMatrix 无异常
        try:
            df_messy = make_data(n_days=150, n_stocks=60, seed=7)
            # 打乱行序 (模拟 strategy 传入的未排序 train_data)
            df_messy = df_messy.sample(frac=1.0, random_state=3).reset_index(drop=True)
            sw = pd.Series(np.random.default_rng(1).uniform(0.5, 2.0, len(df_messy)),
                           index=df_messy.index)
            cfg_b = make_cfg({'enabled': True, 'eval_windows': 2,
                              'early_stopping_rounds': 10, 'candidates': CANDIDATES})
            cfg_b.cv_folds = 2
            cfg_b.cv_embargo_days = 0
            cfg_b.cv_feature_consensus = 1
            cfg_b.cv_ic_psi_split_ratio = 0.5
            cfg_b.PSI_THRESHOLD = 1.0
            cfg_b.MIN_FEAT_IC = 0.0
            cfg_b.NAN_THRESHOLD = 0.9
            cfg_b.MAX_CORR = 0.999
            cfg_b.MIN_FEATURES = 1
            cfg_b.MAX_FEATURES = 12
            cfg_b.use_voting = True
            cfg_b.mda_enabled = True
            cfg_b.mda_n_repeats = 2
            cfg_b.mda_threshold = 0.0
            cfg_b.mda_use_std_threshold = False
            cfg_b.mda_t_alpha = 0.05
            cfg_b.vote_min_folds = 1
            cfg_b.vote_min_features = 1
            cfg_b.top_k_features = 12
            cfg_b.weight_ic = 0.0
            cfg_b.weight_mdi = 0.3
            cfg_b.weight_mda = 1.5
            cfg_b.fallback_to_ic_only = True
            cfg_b.cluster_enabled = False
            cfg_b.cluster_diagnose = False
            cfg_b.MAX_HOLD_DAYS = 20
            cfg_b.AUC_THRESHOLD = 0.5
            cfg_b.CALIBRATION_METHOD = 'none'
            cfg_b.USE_CALIBRATION = False
            cfg_b.VAL_SPLIT_RATIO = 0.1
            cfg_b.XGB_EARLY_STOPPING_ROUNDS = 10
            cfg_b.XGB_EVAL_METRIC = 'auc'
            cfg_b.ENSEMBLE_SEEDS = [42]
            cfg_b.THRESHOLD_SEARCH_POINTS = 10
            cfg_b.lightgbm = {'enabled': False}
            cfg_b.PROB_TEMPERATURE = 1.0
            mgr_b = EnsembleModelManager(model_dir=os.path.join(tmp, 'meta_models_b'), config=cfg_b)
            mgr_b.class_balance_mode = 'auto'
            success_b = mgr_b.train(df_train=df_messy,
                                    feature_cols=make_feats(),
                                    target_col='label_profit',
                                    sample_weight=sw,
                                    current_date='2022-06-30')
            ok &= check('H5b 带权重端到端成功', success_b is True
                        and len(getattr(mgr_b, 'xgb_models', [])) > 0,
                        f'(n_xgb={len(getattr(mgr_b, "xgb_models", []))})')
        except Exception as e:
            import traceback
            traceback.print_exc()
            ok &= check('H5b 带权重端到端成功', False, str(e))

    print('\n==== 超参选择测试完成: %s ====' % ('ALL PASS' if ok else '有失败!'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
