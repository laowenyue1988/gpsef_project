# -*- coding: utf-8 -*-
"""alpha191 系全量对齐 small_alpha191 专项测试 (2026-08-19)

断言 (每目录):
  A. ensemble.py md5 == small (整体替换成功)
  B. 无 factor_validator 残留 (strategy 源码 + 文件)
  C. HP-Search / inner-OOS / 折集成扰动 / sample_weight 对齐 存在
  D. _should_retrain_model 在策略类内 (AST)
  E. config: hyperparam_search(4候选) / retrain_interval=1 / lightgbm=false / validation 已删
  F. import 冒烟 (策略类可加载)
  G. test 门控文案对齐
"""
import ast
import hashlib
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMALL = 'strategies/small_alpha191'
DIRS = ['alpha191', 'alpha191_feat_test', 'big_alpha191', 'cyb_alpha191',
        'hs300_alpha191', 'kcb_alpha191', 'zz1000_alpha191', 'zz500_alpha191',
        'small_alpha191_test']

ok = True


def check(name, cond, extra=''):
    global ok
    if not cond:
        ok = False
    print('[%s] %s %s' % ('PASS' if cond else 'FAIL', name, extra))
    return cond


def md5(p):
    return hashlib.md5(open(p, encoding='utf-8').read().encode()).hexdigest()


def main():
    global ok
    # [2026-08-20] small_alpha191 有意领先 (评估-部署协议一致性参数化, 用户指定仅 small):
    #   标准目录 md5 基准从 small 改为 alpha191 (8 标准目录须彼此一致)
    std_ens_md5 = md5(os.path.join(PROJ, 'strategies', 'alpha191', 'ensemble.py'))

    for d in DIRS:
        base = os.path.join(PROJ, 'strategies', d)
        ens = open(os.path.join(base, 'ensemble.py'), encoding='utf-8').read()
        strat = open(os.path.join(base, 'strategy.py'), encoding='utf-8').read()
        cfg = yaml.safe_load(open(os.path.join(base, 'config.yaml'), encoding='utf-8'))

        # A. ensemble md5
        # [2026-08-20 回退] O1+ 方案已弃 (统计无显著差异 + 开销) → small_alpha191_test
        #   回退到标准版; 全部目录均须 md5=alpha191
        ok &= check(f'[{d}] A ensemble md5=alpha191', md5(os.path.join(base, 'ensemble.py')) == std_ens_md5)

        # B. 三件套残留
        ok &= check(f'[{d}] B 无 factor_validator 引用', 'factor_validator' not in strat)
        ok &= check(f'[{d}] B2 无 _val_pipeline 残留', '_val_pipeline' not in strat)
        ok &= check(f'[{d}] B3 factor_validator.py 已删',
                    not os.path.exists(os.path.join(base, 'factor_validator.py')))

        # C. 结构
        ok &= check(f'[{d}] C HP-Search', '_select_hyperparams' in ens)
        ok &= check(f'[{d}] C2 inner-OOS', 'inner_oos_set' in ens)
        ok &= check(f'[{d}] C3 折集成扰动', '_sel_d' in ens)
        ok &= check(f'[{d}] C4 sample_weight 对齐', 'align_sample_weight_to_sort' in ens)
        ok &= check(f'[{d}] C5 P1-1 投票池=全集', 'consensus_features=None' in ens)

        # D. _should_retrain_model 在类内
        tree = ast.parse(strat)
        in_class = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '_should_retrain_model':
                        in_class = True
        ok &= check(f'[{d}] D _should_retrain_model 在类内', in_class)

        # E. config
        ok &= check(f'[{d}] E hyperparam_search 4候选',
                    'hyperparam_search' in cfg and len(cfg['hyperparam_search']['candidates']) == 4)
        ok &= check(f'[{d}] E2 retrain_interval=1', cfg.get('retrain_interval') == 1)
        ok &= check(f'[{d}] E3 lightgbm 停用', cfg['lightgbm']['enabled'] is False
                    and cfg['lightgbm']['ensemble_weight_xgb'] == 1.0
                    and cfg['lightgbm']['ensemble_weight_lgb'] == 0.0)
        ok &= check(f'[{d}] E4 validation 已删', 'validation' not in cfg)

        # F. import 冒烟
        try:
            mod = importlib.import_module('strategies.%s.strategy' % d)
            cls = next((v for k, v in vars(mod).items()
                        if isinstance(v, type) and k.endswith('Strategy') and k != 'StageTracker'), None)
            ok &= check(f'[{d}] F import OK', cls is not None, f'({cls.__name__ if cls else "?"})')
        except Exception as e:
            ok &= check(f'[{d}] F import OK', False, str(e)[:80])

    # G. test 门控文案
    t_src = open(os.path.join(PROJ, 'strategies', 'small_alpha191_test', 'strategy.py'), encoding='utf-8').read()
    ok &= check('[test] G 门控文案对齐', '[AUC门控] 折集成 meta_AUC' in t_src
                and '生产模型 val AUC' not in t_src)

    # H. [2026-08-20] small_alpha191 有意偏离 (评估-部署协议一致性参数化, 用户指定仅 small):
    #    ensemble 含 eval_class_balance 开关 + config 含键; 其余标准目录(alpha191等)不含
    s_ens = open(os.path.join(PROJ, 'strategies', 'small_alpha191', 'ensemble.py'), encoding='utf-8').read()
    s_cfg = yaml.safe_load(open(os.path.join(PROJ, 'strategies', 'small_alpha191', 'config.yaml'), encoding='utf-8'))
    ok &= check('[small] H 协议一致性参数化', 'eval_class_balance' in s_ens
                and s_cfg['hyperparam_search'].get('eval_class_balance') is False)
    a_ens = open(os.path.join(PROJ, 'strategies', 'alpha191', 'ensemble.py'), encoding='utf-8').read()
    ok &= check('[small] H2 标准目录不受影响', 'eval_class_balance' not in a_ens)

    print('\n==== alpha191 系对齐测试完成: %s ====' % ('ALL PASS' if ok else '有失败!'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
