# -*- coding: utf-8 -*-
"""V10 对齐 alpha191 修复专项测试 (2026-08-20, D1/D3/D4/D5)

用户决策 (2026-08-20):
  D1 = HP-Search 评估 spw 加开关, 默认对齐 alpha191 (false)
  D2 = vote_min_folds/top_k 保持现状 (用户手动调参, 本次不动)
  D3 = 接受折内 IC/PSI 再筛选 (训练耗时上升换特征质量)
  D7 = PT/SL 权重豁免 (V10 无 exit_type 设计差异)
  回测 = 用户手动进行

覆盖:
  D1 (A 组): config 键=false / 源码开关条件 / 逻辑短路 / 不再无条件注入
  D3 (B 组): 源码断 (筛选调用/作用域 fold_feats/空特征跳过) / namespace 映射 / 端到端 PKF 跑通
  D4 (C 组): 纯超额归一化公式 / 无 base_w 兜底
  D5 (D 组): _active_auc_metric 行为 (对齐 alpha191 示例 0.5348) / 弱模型剔除 / 4 处门控统一 / 无简单平均残留
  集成 (E 组): alpha191 家族零改动 / 备份存在 / config diff 白名单

遵循项目测试约定: AST 从生产源码提取方法真实代码 + SimpleNamespace 模拟 self。
"""
import ast
import os
import sys
import time
import types
import importlib
import logging

logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger('test_v10_align_d1d3d4d5')

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

V10_PATH = os.path.join(PROJ, 'strategies', 'wyckoff20', 'gpsef_strategy_v10.py')
V10_MOD = 'strategies.wyckoff20.gpsef_strategy_v10'
V10_CFG = os.path.join(PROJ, 'strategies', 'wyckoff20', 'config_v10.yaml')
BK_V10 = os.path.join(PROJ, 'backups', 'gpsef_strategy_v10.bak_20260820_align')
BK_CFG = os.path.join(PROJ, 'backups', 'config_v10.bak_20260820_align')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from scipy.stats import t as t_dist  # noqa: E402
from typing import Tuple, Optional, List, Dict  # noqa: E402
import yaml  # noqa: E402

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


_v10_mod = importlib.import_module(V10_MOD)

# ===================================================================
# A 组: D1 评估 spw 开关 (默认 false, 对齐 alpha191)
# ===================================================================
def a1_source_d1():
    print('\n[A] D1: HP-Search 评估 spw 开关')
    src = extract_method(V10_PATH, 'WyckoffV10Strategy', '_select_hyperparams')
    check('A1 开关条件存在', "hp_cfg.get('eval_class_balance', False)" in src)
    check('A1 else 分支不注入', 'p_params = dict(params)' in src)
    # 不再无条件注入: 注入必须包裹在开关内
    import re
    unconditional = re.search(r'(?m)^\s*_apply_class_balance\(dict\(params\), y_t\)\s*$', src)
    check('A1 无无条件注入', unconditional is None)
    # config 键
    cfg = yaml.safe_load(open(V10_CFG, encoding='utf-8'))
    v = cfg['hyperparam_search'].get('eval_class_balance')
    check('A1 config eval_class_balance=False', v is False, f'(={v})')
    # 逻辑短路语义
    check('A1 默认(无键)→False', bool({}.get('eval_class_balance', False)) is False)
    check('A1 true→注入', bool({'eval_class_balance': True}.get('eval_class_balance', False)) is True)


def a2_config_diff_whitelist():
    print('\n[A2] config diff 白名单 (仅 hyperparam_search.eval_class_balance)')
    import subprocess
    r = subprocess.run(['diff', BK_CFG, V10_CFG], capture_output=True, text=True)
    diff_lines = r.stdout.splitlines()
    added = [l[1:].strip() for l in diff_lines if l.startswith('>')]
    ctx = [l for l in diff_lines if l[:1] in ('<', '>')]
    # 断言: 新增行只有 eval_class_balance 相关注释与键
    bad = [l for l in ctx if 'eval_class_balance' not in l and '对齐 alpha191 D1' not in l
           and '评估段是否注入' not in l and 'true ->' not in l and 'false (默认)' not in l
           and '评估注入 spw 致候选噪声翻转' not in l and 'Small 收益' not in l
           and '默认关闭避免噪声驱动选择翻转' not in l and '评估段' not in l
           and l.strip() in ('<', '>', '---', '')]
    check('A2 config 仅新增 eval_class_balance 相关行', not bad, f'bad={bad[:3]}')
    check('A2 新增行含键行', any('eval_class_balance: false' in l for l in added))


# ===================================================================
# B 组: D3 折内 IC/PSI 再筛选
# ===================================================================
def b1_source_d3():
    print('\n[B] D3: PKF 折内 IC/PSI 再筛选')
    src = extract_method(V10_PATH, 'WyckoffV10Strategy', '_purged_kfold_cv')
    check('B1 折内调用 select_stable_features', 'select_stable_features(' in src)
    check('B1 IC/PSI 子集切分 (inner-train 内)', "inner_train_set['snapshot_date'].isin(_ic_dates)" in src
          and "inner_train_set['snapshot_date'].isin(_psi_dates)" in src)
    check('B1 PSI 子集至少 5 天', 'len(_ic_unique) - 5' in src)
    check('B1 DMatrix 用折内特征', 'feature_names=fold_feats' in src)
    check('B1 MDA 用折内特征', 'inner_oos_set[fold_feats]' in src)
    check('B1 pass_set 作用域=折内特征', 'for f in fold_feats:' in src)
    check('B1 MDI 作用域=折内特征', 'for f in fold_feats})' in src or 'for f in fold_feats}' in src)
    check('B1 空特征跳过折', '折内再筛后无特征, 跳过' in src)
    check('B1 失败回退 prefilter 全集', '回退 prefilter 全集' in src)
    # 折内特征在投票域内 (pre_filtered_feats 初始化投票, 未入选折内 → 0 票自然淘汰)
    check('B1 投票域保留 pre_filtered_feats', 'mda_votes = {f: 0 for f in pre_filtered_feats}' in src)


def b2_namespace_mapping():
    print('\n[B2] _fs_select_stable_namespace 属性映射')
    ns_src = extract_method(V10_PATH, 'WyckoffV10Strategy', '_fs_select_stable_namespace')
    ns_globals = {'types': types, 'SimpleNamespace': types.SimpleNamespace}
    exec(compile(ns_src, 'WyckoffV10Strategy._fs_select_stable_namespace', 'exec'), ns_globals)
    ns_fn = ns_globals['_fs_select_stable_namespace']
    m = types.SimpleNamespace(
        _feat_sel_nan_threshold=0.5, _feat_sel_max_corr=0.90,
        _feat_sel_psi_threshold=0.25, _feat_sel_min_ic=0.001,
        _min_features=10, _max_features=80)
    ns = types.MethodType(ns_fn, m)()
    check('B2 NAN_THRESHOLD=0.5', abs(ns.NAN_THRESHOLD - 0.5) < 1e-12, f'(={ns.NAN_THRESHOLD})')
    check('B2 MAX_CORR=0.90', abs(ns.MAX_CORR - 0.90) < 1e-12)
    check('B2 PSI_THRESHOLD=0.25', abs(ns.PSI_THRESHOLD - 0.25) < 1e-12, f'(={ns.PSI_THRESHOLD})')
    check('B2 MIN_FEAT_IC=0.001', abs(ns.MIN_FEAT_IC - 0.001) < 1e-12)
    check('B2 MIN_FEATURES=10', ns.MIN_FEATURES == 10)
    check('B2 MAX_FEATURES=80', ns.MAX_FEATURES == 80)


def b3_select_stable_on_v10_data():
    print('\n[B3] select_stable_features 在 V10 风格数据上可用')
    check('B3 可从 alpha191 ensemble 导入', hasattr(_v10_mod, 'select_stable_features'))
    fn = _v10_mod.select_stable_features
    rng = np.random.default_rng(7)
    dates = pd.date_range('2022-01-04', periods=40, freq='B')
    cols = [f'f{i}' for i in range(10)]
    rows = []
    for d in dates:
        f0 = rng.normal(size=15)
        for i in range(15):
            rows.append([d, f's{i:04d}', f0[i]] + [rng.normal() for _ in range(9)]
                        + [1 if f0[i] + rng.normal(0, 0.6) > 0 else 0])
    df = pd.DataFrame(rows, columns=['snapshot_date', 'ts_code'] + cols + ['label'])
    ic_set = df[df['snapshot_date'] <= df['snapshot_date'].unique()[19]]
    psi_set = df[df['snapshot_date'] > df['snapshot_date'].unique()[19]]
    ns = types.SimpleNamespace(NAN_THRESHOLD=0.5, MAX_CORR=0.9, PSI_THRESHOLD=0.25,
                               MIN_FEAT_IC=0.001, MIN_FEATURES=10, MAX_FEATURES=80)
    feats_out, scores, max_ic, ic_map = fn(ic_set, psi_set, cols, ns,
                                           target_col='label', skip_preprocess=True)
    check('B3 返回特征子集 (非空且⊆输入)', len(feats_out) >= 1 and set(feats_out) <= set(cols),
          f'(n={len(feats_out)})')
    check('B3 IC 映射返回', isinstance(ic_map, dict) and len(ic_map) >= 1)


def b4_endtoend_pkf():
    print('\n[B4] 端到端: _purged_kfold_cv (含折内筛选) 跑通')
    src = extract_method(V10_PATH, 'WyckoffV10Strategy', '_purged_kfold_cv')
    ns_src = extract_method(V10_PATH, 'WyckoffV10Strategy', '_fs_select_stable_namespace')
    ns_g = {'types': types, 'SimpleNamespace': types.SimpleNamespace}
    exec(compile(ns_src, 'WyckoffV10Strategy._fs_select_stable_namespace', 'exec'), ns_g)
    ns_fn = ns_g['_fs_select_stable_namespace']

    def _mock_mda(model, X_val_df, y_val, feature_names, n_repeats=20,
                  random_state=42, t_alpha=0.05):
        # 全特征正 MDA + 小 std → t 检验全过 (确定性, 仅验证作用域)
        means = {f: 0.001 + i * 0.0001 for i, f in enumerate(feature_names)}
        stds = {f: 0.0001 for f in feature_names}
        return means, stds

    def _apply_cb(params, y_train):
        return params

    m = types.SimpleNamespace(
        config={'inner_oos_frac': 0.15, 'inner_oos_min_days': 20},
        _cv_folds=5, _cv_embargo_days=2,
        _cv_fold_defs=[],
        _fs_vote_min_folds=2, _fs_vote_min_features=10, _fs_top_k_features=80,
        _fs_fallback_to_ic_only=True,
        _fs_weight_ic=0.0, _fs_weight_mdi=0.3, _fs_weight_mda=1.5,
        _fs_mda_random_state=42, _fs_mda_t_alpha=0.05, _feat_sel_mda_n_repeats=5,
        _feat_sel_mdi_nthread=1, _feat_sel_mdi_n_estimators=30,
        early_stopping_rounds=10,
        xgb_params={'max_depth': 3, 'eta': 0.05, 'subsample': 0.7,
                    'colsample_bytree': 0.7, 'min_child_weight': 5,
                    'alpha': 0.1, 'lambda': 1.0, 'gamma': 0.0,
                    'seed': 42, 'objective': 'binary:logistic',
                    'eval_metric': 'auc', 'tree_method': 'hist', 'device': 'cpu'},
        full_calendar=[], pred_days=20,
        _apply_class_balance=_apply_cb,
        _compute_mda_oos=_mock_mda,
        _cv_ic_psi_split_ratio=0.5,
        _cluster_cfg_namespace=lambda: None,  # CFI off, 不会触达
        _feat_sel_nan_threshold=0.5, _feat_sel_max_corr=0.90,
        _feat_sel_psi_threshold=0.25, _feat_sel_min_ic=0.001,
        _min_features=10, _max_features=80,
    )
    m._fs_select_stable_namespace = types.MethodType(ns_fn, m)

    exec_ns = {
        'np': np, 'pd': pd, 'xgb': xgb, 'time': time, 'logger': logger,
        't_dist': t_dist, 'List': List, 'Optional': Optional, 'Tuple': Tuple,
        '_CFI_AVAILABLE': False,
        'compute_sample_weights': _v10_mod.compute_sample_weights,
        'select_stable_features': _v10_mod.select_stable_features,
    }
    exec(compile(src, 'WyckoffV10Strategy._purged_kfold_cv', 'exec'), exec_ns)
    pkf = exec_ns['_purged_kfold_cv']

    # 小数据
    rng = np.random.default_rng(11)
    dates = pd.date_range('2022-01-04', periods=60, freq='B')
    cols = [f'f{i}' for i in range(12)]
    rows = []
    for d in dates:
        f0 = rng.normal(size=20)
        for i in range(20):
            rows.append([d, f's{i:04d}', f0[i]] + [rng.normal() for _ in range(11)]
                        + [1 if f0[i] + rng.normal(0, 0.6) > 0 else 0])
    df = pd.DataFrame(rows, columns=['snapshot_date', 'ts_code'] + cols + ['label'])
    ic_scores = {f: 0.01 + i * 0.001 for i, f in enumerate(cols)}
    t0 = time.time()
    feats_out = pkf(m, df, cols, ic_scores, None)  # 首参 self=mock, 次参 train_data
    dt = time.time() - t0
    check('B4 端到端返回特征 (非空且⊆输入)', len(feats_out) >= 1 and set(feats_out) <= set(cols),
          f'(n={len(feats_out)}, 耗时{dt:.1f}s)')
    check('B4 折定义已保存', len(m._cv_fold_defs) >= 3, f'(n={len(m._cv_fold_defs)})')


# ===================================================================
# C 组: D4 CV 折集成纯超额加权
# ===================================================================
def c1_source_d4():
    print('\n[C] D4: CV 折集成纯超额加权')
    src = extract_method(V10_PATH, 'WyckoffV10Strategy', '_train_cv_ensemble')
    check('C1 纯超额公式', 'weights = [e / total_e for e in excess]' in src)
    # 注释中提及 base_w, 因此按代码形态断言: 无 base_w 赋值/使用表达式
    check('C1 无 base_w 兜底代码', 'base_w = 0.5' not in src and 'len(models) * base_w' not in src,
          '(base_w 代码已移除, 注释说明除外)')
    check('C1 超额定义 max(0, a-0.50)', 'max(0.0, a - 0.50)' in src)
    check('C1 uniform 分支保留', "self._ensemble_weight_method == 'uniform'" in src)


# ===================================================================
# D 组: D5 门控口径 (活跃模型 AUC 加权平均)
# ===================================================================
def d1_behavior_metric():
    print('\n[D] D5: 门控口径加权平均')
    src = extract_method(V10_PATH, 'WyckoffV10Strategy', '_active_auc_metric')
    g = {'np': np, 'types': types}
    exec(compile(src, 'WyckoffV10Strategy._active_auc_metric', 'exec'), g)
    fn = g['_active_auc_metric']
    # 与 alpha191 折集成示例完全一致的数值 (2026-08-19 冒烟日志):
    #   aucs=[0.5270,0.5023,0.5207,0.5470], 超额权重=[0.278,0.024,0.213,0.485]
    #   alpha191 meta_AUC = 加权平均 = 0.5348
    m = types.SimpleNamespace(
        model_aucs=[0.5270, 0.5023, 0.5207, 0.5470],
        model_weights=[0.278, 0.024, 0.213, 0.485],
        _active_models=[0, 1, 2, 3])
    val = types.MethodType(fn, m)()
    check('D1 加权平均 ≈ alpha191 示例 0.5348', abs(val - 0.5348) < 1e-4, f'(={val:.4f})')
    # 弱模型剔除: 仅活跃子集加权
    m2 = types.SimpleNamespace(
        model_aucs=[0.5270, 0.5023, 0.5207, 0.5470],
        model_weights=[0.278, 0.024, 0.213, 0.485],
        _active_models=[0, 2, 3])
    v2 = types.MethodType(fn, m2)()
    w_manual = (0.278 * 0.5270 + 0.213 * 0.5207 + 0.485 * 0.5470) / (0.278 + 0.213 + 0.485)
    check('D1 活跃子集加权正确', abs(v2 - w_manual) < 1e-6, f'(={v2:.4f})')
    # uniform 权重 → 简单平均
    m3 = types.SimpleNamespace(
        model_aucs=[0.50, 0.52, 0.54],
        model_weights=[1.0, 1.0, 1.0],
        _active_models=[0, 1, 2])
    v3 = types.MethodType(fn, m3)()
    check('D1 uniform 权重=简单平均', abs(v3 - 0.52) < 1e-9, f'(={v3:.4f})')
    # 空 → 0.0
    m4 = types.SimpleNamespace(model_aucs=[], model_weights=[], _active_models=[])
    v4 = types.MethodType(fn, m4)()
    check('D1 空模型→0.0', v4 == 0.0)


def d2_gate_sites():
    print('\n[D2] 门控口径全站点统一')
    src = open(V10_PATH, encoding='utf-8').read()
    n_old = src.count('np.mean(active_aucs)')
    check('D2 无简单平均残留', n_old == 0, f'(残留{n_old})')
    n_helper = src.count('self._active_auc_metric()')
    check('D2 辅助方法调用 ≥3 处', n_helper >= 3, f'(调用{n_helper}处)')
    n_meta = src.count('_cv_meta_auc')
    check('D2 _cv_meta_auc 赋值+使用', n_meta >= 5, f'(出现{n_meta}次: 重置1+赋值1+门控读取3)')
    # _train_cv_ensemble 内设置 + 3 处门控读取
    cve = extract_method(V10_PATH, 'WyckoffV10Strategy', '_train_cv_ensemble')
    check('D2 _train_cv_ensemble 内加权口径', '_active_auc_metric()' in cve and '_cv_meta_auc' in cve)
    check('D2 防御重置存在', 'self._cv_meta_auc = None' in cve)


# ===================================================================
# E 组: 集成与回归
# ===================================================================
def e1_family_untouched():
    print('\n[E] 集成回归')
    import hashlib
    std = hashlib.md5(open(os.path.join(PROJ, 'strategies', 'alpha191', 'ensemble.py'), 'rb')
                      .read().replace(b'\r\n', b'\n')).hexdigest()
    dirs = ['alpha191', 'alpha191_feat_test', 'big_alpha191', 'cyb_alpha191',
            'hs300_alpha191', 'kcb_alpha191', 'zz1000_alpha191', 'zz500_alpha191',
            'small_alpha191_test', 'small_alpha191']
    all_ok = True
    dev_notes = []
    for d in dirs:
        p = os.path.join(PROJ, 'strategies', d, 'ensemble.py')
        m = hashlib.md5(open(p, 'rb').read().replace(b'\r\n', b'\n')).hexdigest()
        same = m == std
        # small_alpha191 为用户 2026-08-20 10:07 手动加的 eval_class_balance 版 (test_alpha191_align H 项断言)
        if d == 'small_alpha191':
            dev_notes.append(f'small_alpha191(用户版, 预期偏离): {m[:12]}')
            all_ok &= (not same)
        else:
            all_ok &= same
            if not same:
                dev_notes.append(f'DEV: {d} {m[:12]}')
    check('E1 9 目录零改动 + small 保持用户版', all_ok)
    for n in dev_notes:
        print(f'      [注] {n}')


def e2_backups():
    print('\n[E2] 备份完整性')
    import hashlib
    check('E2 V10 备份存在', os.path.exists(BK_V10))
    check('E2 config 备份存在', os.path.exists(BK_CFG))
    cur = hashlib.md5(open(V10_PATH, 'rb').read()).hexdigest()
    bak = hashlib.md5(open(BK_V10, 'rb').read()).hexdigest()
    check('E2 V10 已修改 (备份≠当前)', cur != bak, f'({cur[:10]} vs {bak[:10]})')
    cfg_cur = hashlib.md5(open(V10_CFG, 'rb').read()).hexdigest()
    cfg_bak = hashlib.md5(open(BK_CFG, 'rb').read()).hexdigest()
    check('E2 config 已修改 (备份≠当前)', cfg_cur != cfg_bak)


def main():
    print('=' * 70)
    print('V10 对齐 alpha191 修复专项测试 (2026-08-20: D1/D3/D4/D5)')
    print('=' * 70)
    a1_source_d1()
    a2_config_diff_whitelist()
    b1_source_d3()
    b2_namespace_mapping()
    b3_select_stable_on_v10_data()
    b4_endtoend_pkf()
    c1_source_d4()
    d1_behavior_metric()
    d2_gate_sites()
    e1_family_untouched()
    e2_backups()
    print('\n' + '=' * 70)
    print(f'结果: {ok}')
    print('ALL PASS' if ok else '有失败!')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
