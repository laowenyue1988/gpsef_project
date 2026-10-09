# -*- coding: utf-8 -*-
"""V10 对齐 small_alpha191 修复专项测试 (2026-08-18)

覆盖:
  A. HP-Search (V1-V8): 禁用跳过/候选空/写回一致/反向不写回/多窗detail/隔离/warmup/区分度
  B. inner-OOS (V9):   源码断言 MDA 迁移 inner_oos_set, 验证块只评 AUC
  C. 折集成扰动 (V10): 源码断言围绕选中超参派生 + metric 全 auc
  D. LGBM 停用 (V11):  config enabled=false → _lgb_enabled=False
  E. retrain_interval (V12-V16): 边界/冷启动/沿用/重置/未就绪
  F. 三件套移除 (V17): 源码零残留 + import 正常

遵循项目测试约定: AST 从生产源码提取方法真实代码 + SimpleNamespace 模拟 self。
"""
import ast
import os
import sys
import time
import logging
import types

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('test_v10_align')

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

V10_PATH = os.path.join(PROJ, 'strategies', 'wyckoff20', 'gpsef_strategy_v10.py')
V10_MOD = 'strategies.wyckoff20.gpsef_strategy_v10'

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from typing import Tuple, Optional, List  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ok = True


def check(name, cond, extra=''):
    global ok
    mark = 'PASS' if cond else 'FAIL'
    if not cond:
        ok = False
    print(f'[{mark}] {name} {extra}')
    return cond


def extract_method(path, cls, meth):
    src = open(path, encoding='utf-8').read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == meth:
                    return ast.get_source_segment(src, item)
    raise RuntimeError(f'{cls}.{meth} not found')


def make_data(n_days=150, n_stocks=60, seed=42, n_feats=12, reverse_val=False, val_size=15):
    """V10 风格数据: snapshot_date + label + 特征列"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2022-01-04', periods=n_days, freq='B')
    cols = [f'f{i}' for i in range(n_feats)]
    rows = []
    n_val = val_size
    for d in dates:
        f0 = rng.normal(size=n_stocks)
        is_val = (len(rows) >= (n_days - n_val) * n_stocks)
        for i in range(n_stocks):
            rec = [d, f's{i:04d}', f0[i]] + [rng.normal() for _ in range(n_feats - 1)]
            if reverse_val and is_val:
                rec.append(1 if -f0[i] + rng.normal(0, 0.6) > 0 else 0)
            else:
                rec.append(1 if f0[i] + rng.normal(0, 0.6) > 0 else 0)
            rows.append(rec)
    return pd.DataFrame(rows, columns=['snapshot_date', 'ts_code'] + cols + ['label'])


def make_v10_config(hp_cfg):
    cfg = {
        'hyperparam_search': hp_cfg,
        'xgb_random_state': 42,
        'xgb_tree_method': 'hist',
        'xgb_device': 'cpu',
        'xgb_n_estimators': 100,
    }
    return cfg


def make_self(cfg, hp_cfg):
    """构造 mock self (V10 策略实例)"""
    self_mock = types.SimpleNamespace(
        config=cfg,
        xgb_params={
            'max_depth': 4, 'eta': 0.03, 'subsample': 0.7,
            'colsample_bytree': 0.7, 'min_child_weight': 25,
            'alpha': 0.5, 'lambda': 2.0, 'gamma': 0.0,
            'seed': 42, 'objective': 'binary:logistic', 'eval_metric': 'auc',
            'tree_method': 'hist', 'device': 'cpu', 'nthread': 1,
        },
        full_calendar=[],
        pred_days=20,
        _apply_class_balance=lambda params, y: params,  # mock: 不注入 spw
        _hp_search_results=None,
        _hp_search_detail=None,
    )
    return self_mock


# 提取 _select_hyperparams 真实代码
_HP_SRC = extract_method(V10_PATH, 'WyckoffV10Strategy', '_select_hyperparams')
_hp_ns = {
    'np': np, 'pd': pd, 'xgb': xgb, 'time': time, 'logger': logger,
    'roc_auc_score': roc_auc_score, 'Tuple': Tuple, 'Optional': Optional,
    'List': List,
}
# 从 V10 模块导入依赖
try:
    import importlib
    _v10_mod = importlib.import_module(V10_MOD)
    _hp_ns['LRWarmupCallback'] = _v10_mod.LRWarmupCallback
    _hp_ns['compute_sample_weights'] = _v10_mod.compute_sample_weights
    _hp_ns['WyckoffV10Strategy'] = _v10_mod.WyckoffV10Strategy
except Exception as e:
    check('V0 模块导入', False, str(e))
exec(compile(_HP_SRC, 'WyckoffV10Strategy._select_hyperparams', 'exec'), _hp_ns)
_select_hyperparams = _hp_ns['_select_hyperparams']

CANDIDATES = [
    {'name': 'c1', 'xgb_max_depth': 3, 'xgb_learning_rate': 0.05,
     'xgb_n_estimators': 50, 'xgb_subsample': 0.6, 'xgb_col_sample_by_tree': 0.6,
     'xgb_reg_alpha': 0.5, 'xgb_reg_lambda': 3.0, 'xgb_min_child_weight': 30, 'xgb_gamma': 0.1},
    {'name': 'c2', 'xgb_max_depth': 4, 'xgb_learning_rate': 0.03,
     'xgb_n_estimators': 80, 'xgb_subsample': 0.7, 'xgb_col_sample_by_tree': 0.7,
     'xgb_reg_alpha': 0.3, 'xgb_reg_lambda': 2.0, 'xgb_min_child_weight': 20, 'xgb_gamma': 0.0},
    {'name': 'c3', 'xgb_max_depth': 5, 'xgb_learning_rate': 0.02,
     'xgb_n_estimators': 120, 'xgb_subsample': 0.8, 'xgb_col_sample_by_tree': 0.8,
     'xgb_reg_alpha': 0.1, 'xgb_reg_lambda': 1.0, 'xgb_min_child_weight': 10, 'xgb_gamma': 0.0},
]


def main():
    global ok
    # ================= V1: enabled=false 跳过 =================
    cfg = make_v10_config({'enabled': False, 'candidates': CANDIDATES})
    s = make_self(cfg, cfg['hyperparam_search'])
    best, score = _select_hyperparams(s, make_data(), ['f%d' % i for i in range(12)])
    ok &= check('V1 enabled=false 跳过', best is None and score is None, f'({best})')

    # ================= V2: 候选空 =================
    cfg = make_v10_config({'enabled': True, 'candidates': []})
    s = make_self(cfg, cfg['hyperparam_search'])
    best, score = _select_hyperparams(s, make_data(), ['f0'])
    ok &= check('V2 候选空跳过', best is None and score is None, f'({best})')

    # ================= V3: 正常选择 + 写回一致 =================
    cfg = make_v10_config({'enabled': True, 'inner_val_days': 10, 'eval_windows': 2,
                           'early_stopping_rounds': 10, 'eval_rounds': 200,
                           'use_early_stop': True, 'candidates': CANDIDATES})
    s = make_self(cfg, cfg['hyperparam_search'])
    df = make_data(n_days=120)
    feats = ['f%d' % i for i in range(12)]
    with warnings_silenced():
        t0 = time.time()
        best, score = _select_hyperparams(s, df, feats)
    dt = time.time() - t0
    ok &= check('V3 返回选中候选', best in [c['name'] for c in CANDIDATES],
                f'(best={best}, score={score:.4f}, 耗时{dt:.1f}s)')
    if best:
        bc = next(c for c in CANDIDATES if c['name'] == best)
        keymap = {'xgb_max_depth': 'max_depth', 'xgb_learning_rate': 'eta',
                  'xgb_subsample': 'subsample', 'xgb_col_sample_by_tree': 'colsample_bytree',
                  'xgb_reg_alpha': 'alpha', 'xgb_reg_lambda': 'lambda',
                  'xgb_min_child_weight': 'min_child_weight', 'xgb_gamma': 'gamma'}
        consistent = all(s.xgb_params[k] == bc[ck] for ck, k in keymap.items())
        consistent &= s.config['xgb_n_estimators'] == bc['xgb_n_estimators']
        ok &= check('V3 写回 self.xgb_params 一致(8参数+n_estimators)', consistent,
                    f'(xgb_params={ {k: s.xgb_params[k] for k in keymap.values()} })')

    # 候选区分度 (修复 xgb_* 键名 bug 的回归)
    hp_res = getattr(s, '_hp_search_results', [])
    if hp_res:
        aucs = [r[1] for r in hp_res]
        spread = max(aucs) - min(aucs)
        ok &= check('V3b 候选AUC有区分度', spread > 0.0001,
                    f'(候选AUC={[round(a, 4) for a in aucs]}, spread={spread:.4f})')
    else:
        ok &= check('V3b 候选AUC有区分度', False, '(无 _hp_search_results)')

    # 无忽略参数警告 (xgb 参数名正确传递)
    # 通过 V3 正常完成训练且区分度 > 0 已经隐含 (若参数被忽略 → 全同)

    # ================= V4: 反向数据 → 不写回 =================
    cfg = make_v10_config({'enabled': True, 'inner_val_days': 10, 'eval_windows': 2,
                           'early_stopping_rounds': 10, 'eval_rounds': 200,
                           'use_early_stop': True, 'candidates': CANDIDATES})
    s = make_self(cfg, cfg['hyperparam_search'])
    df_rev = make_data(n_days=120, seed=11, reverse_val=True, val_size=40)
    with warnings_silenced():
        best, score = _select_hyperparams(s, df_rev, feats)
    hp_res4 = getattr(s, '_hp_search_results', [])
    all_below = all(r[1] < 0.5 for r in hp_res4) if hp_res4 else True
    ok &= check('V4 全部AUC<0.5不写回', best is None and score is None
                and s.xgb_params['max_depth'] == 4 and all_below,
                f'(best={best}, hp_aucs={[round(a, 3) for _, a, _ in hp_res4]})')

    # ================= V5: 多窗 detail =================
    detail = getattr(s, '_hp_search_detail', None) or {}
    has_detail = all('scores' in d and 'rounds' in d for d in detail.values()) if detail else False
    ok &= check('V5 多窗 detail 存在', has_detail, f'(n={len(detail)})')

    # ================= V6: [A] 隔离 (生产val段反向不影响选择) =================
    cfg1 = make_v10_config({'enabled': True, 'inner_val_days': 10, 'eval_windows': 2,
                            'early_stopping_rounds': 10, 'eval_rounds': 200,
                            'use_early_stop': True, 'candidates': CANDIDATES})
    s1 = make_self(cfg1, cfg1['hyperparam_search'])
    df1 = make_data(n_days=120, seed=5)
    with warnings_silenced():
        b1, sc1 = _select_hyperparams(s1, df1, feats)
    cfg2 = make_v10_config({'enabled': True, 'inner_val_days': 10, 'eval_windows': 2,
                            'early_stopping_rounds': 10, 'eval_rounds': 200,
                            'use_early_stop': True, 'candidates': CANDIDATES})
    s2 = make_self(cfg2, cfg2['hyperparam_search'])
    df2 = make_data(n_days=120, seed=5, reverse_val=True, val_size=20)  # 生产val(末20天)反向
    with warnings_silenced():
        b2, sc2 = _select_hyperparams(s2, df2, feats)
    ok &= check('V6 [A] 隔离: 生产val反向不影响选择', b1 == b2,
                f'(b1={b1}, b2={b2}, score {sc1:.4f} vs {sc2:.4f})')

    # ================= V7: warmup 已加入评估 (源码断言) =================
    ok &= check('V7 评估含 LRWarmupCallback', 'LRWarmupCallback' in _HP_SRC
                and 'target_lr=float(cand[' in _HP_SRC)

    # ================= V8: [B] 多窗 + [C] 早停 (源码断言) =================
    ok &= check('V8 多窗 expanding 实现', 'eval_windows' in _HP_SRC
                and 'expanding' in _HP_SRC)
    ok &= check('V8b 每窗独立早停', 'early_stopping_rounds=es_rounds' in _HP_SRC)

    # ================= V9: inner-OOS 源码断言 =================
    # [2026-08-20 D3] MDA/训练作用域从 pre_filtered_feats 升级为 fold_feats (折内 IC/PSI 再筛选),
    #   断言同步: inner-OOS MDA 2 处 (CFI 分支 + 逐特征分支) + 训练在 inner-train 上
    pkf_src = extract_method(V10_PATH, 'WyckoffV10Strategy', '_purged_kfold_cv')
    n_inner_mda = pkf_src.count('inner_oos_set[fold_feats]')
    # fold_val[fold_feats] 应只剩 early-stopping 的 dval 一处 (X_val_fold)
    n_val_feats = pkf_src.count('fold_val[fold_feats]')
    ok &= check('V9 inner-OOS: MDA 迁移 (2处, 折内特征)', n_inner_mda == 2, f'(inner_oos_set[fold_feats]={n_inner_mda})')
    ok &= check('V9b inner-OOS: 验证块只评AUC (fold_val残留≤1)', n_val_feats <= 1, f'(fold_val[fold_feats]={n_val_feats})')
    ok &= check('V9c inner-train 训练', 'inner_train_set' in pkf_src
                and 'inner_train_set[fold_feats]' in pkf_src)

    # ================= V10: 折集成扰动源码断言 =================
    ens_src = extract_method(V10_PATH, 'WyckoffV10Strategy', '_train_cv_ensemble')
    ok &= check('V10 折集成围绕选中超参扰动', '_sel_d = int(self.xgb_params.get' in ens_src
                and '_sel_d + 1' in ens_src and '_sel_ss - 0.05' in ens_src
                and '_sel_cs + 0.05' in ens_src)
    ok &= check('V10b metric 全 auc', "metric_list = ['auc'] * 5" in ens_src)
    ok &= check('V10c 硬编码列表已弃用', 'self._ensemble_depth_list' not in ens_src)

    # ================= V11: LGBM 停用 (config + 初始化逻辑) =================
    import yaml
    v10_cfg = yaml.safe_load(open(os.path.join(PROJ, 'strategies', 'wyckoff20', 'config_v10.yaml'), encoding='utf-8'))
    lgb_cfg = v10_cfg['ensemble']['lightgbm']
    ok &= check('V11 LGBM enabled=false', lgb_cfg['enabled'] is False)
    ok &= check('V11b LGBM 权重 1.0/0.0',
                lgb_cfg['ensemble_weight_xgb'] == 1.0 and lgb_cfg['ensemble_weight_lgb'] == 0.0)

    # ================= E: retrain_interval =================
    ri_src = extract_method(V10_PATH, 'WyckoffV10Strategy', '_should_retrain_model')
    _ri_ns = {'logger': logger, 'Tuple': tuple}
    exec(compile(ri_src, 'WyckoffV10Strategy._should_retrain_model', 'exec'), _ri_ns)
    _should_retrain = _ri_ns['_should_retrain_model']

    def mk_ri(retrain_interval=5, models=True, cv_defs=True, count=0):
        s = types.SimpleNamespace(
            retrain_interval=retrain_interval,
            models=[object()] if models else [],
            _cv_fold_defs=[(1, 2)] if cv_defs else [],
            _rebalances_since_train=count,
        )
        return s

    s = mk_ri(count=0)
    now, _ = _should_retrain(s)
    ok &= check('V12 count=0且模型就绪→沿用', now is False)
    s = mk_ri(count=3)
    now, _ = _should_retrain(s)
    ok &= check('V13 未达间隔沿用', now is False)
    s = mk_ri(count=5)
    now, _ = _should_retrain(s)
    ok &= check('V14 已达间隔重训', now is True)
    s = mk_ri(retrain_interval=1)
    now, _ = _should_retrain(s)
    ok &= check('V15 n=1 每次重训', now is True)
    s = mk_ri(models=False)
    now, _ = _should_retrain(s)
    ok &= check('V16 模型未就绪强制重训', now is True)

    # before_market_open 包裹结构断言
    bm_src = extract_method(V10_PATH, 'WyckoffV10Strategy', 'before_market_open')
    ok &= check('V16b 计数+判断+沿用分支', '_rebalances_since_train += 1' in bm_src
                and 'if not retrain_now:' in bm_src
                and '跳过训练直接选股' in bm_src)
    ok &= check('V16c 训练成功重置', 'self._rebalances_since_train = 0' in bm_src)

    # ================= F: 三件套移除 =================
    full_src = open(V10_PATH, encoding='utf-8').read()
    ok &= check('V17 无 factor_validator 残留', 'factor_validator' not in full_src)
    ok &= check('V17b 无 _validation_pipeline 残留', '_validation_pipeline' not in full_src)
    ok &= check('V17c 无 ConditionalIC 残留', 'ConditionalICAnalyzer' not in full_src)
    ok &= check('V17d config 无 factor_validation 段', 'factor_validation' not in open(
        os.path.join(PROJ, 'strategies', 'wyckoff20', 'config_v10.yaml'), encoding='utf-8').read())

    # import 验证
    try:
        import importlib
        m = importlib.import_module(V10_MOD)
        cls = m.WyckoffV10Strategy
        ok &= check('V18 模块 import 正常', cls is not None, f'({cls.__name__})')
    except Exception as e:
        ok &= check('V18 模块 import 正常', False, str(e))

    print('\n==== V10 对齐测试完成: %s ====' % ('ALL PASS' if ok else '有失败!'))
    return 0 if ok else 1


import warnings
import contextlib


@contextlib.contextmanager
def warnings_silenced():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


if __name__ == '__main__':
    sys.exit(main())
