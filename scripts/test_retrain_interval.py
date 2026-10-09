# -*- coding: utf-8 -*-
"""[重训间隔] _should_retrain_model 单元测试 (AST 提取生产源码, SimpleNamespace 模拟 self)

覆盖边界:
  T1 冷启动 force_retrain=True → 重训
  T2 模型未就绪 (ensemble_manager=None / xgb_models 空) → 重训
  T3 retrain_interval<=1 → 每次调仓都重训
  T4 计数未达间隔 → 沿用 (不重训)
  T5 计数达到间隔 (>=n) → 重训
  T6 调仓循环模拟: n=5, 首次重训后计数重置, 第 2~5 次沿用, 第 6 次重训
  T7 训练失败不重置计数 → 后续沿用旧模型; 模型缺失时每次重试
"""
import ast
import types
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('test_retrain_interval')

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STRAT_DIR = os.environ.get('TEST_STRATEGY_DIR', 'small_alpha191')
STRATEGY_PATH = os.path.join(_PROJ_ROOT, 'strategies', _STRAT_DIR, 'strategy.py')
CLS_NAME = 'SmallAlpha191Strategy'
METH_NAME = '_should_retrain_model'


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


# AST 提取生产源码 (保证测试的是真实代码, 非复制品)
_SRC = extract_method_source(STRATEGY_PATH, CLS_NAME, METH_NAME)
_ns = {}
exec(compile(_SRC, f'{CLS_NAME}.{METH_NAME}', 'exec'), _ns)
_should_retrain_model = _ns[METH_NAME]


class FakeEnsemble:
    def __init__(self, ready=True):
        self.xgb_models = ['m1', 'm2'] if ready else []


def make_strategy(**kw):
    defaults = dict(force_retrain=False,
                    ensemble_manager=FakeEnsemble(ready=True),
                    RETRAIN_INTERVAL=5,
                    _rebalances_since_train=1)
    defaults.update(kw)
    return types.SimpleNamespace(**defaults)


def check(tag, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'[{status}] {tag} {detail}')
    return cond


def main():
    ok = True
    # T1 冷启动
    s = make_strategy(force_retrain=True, ensemble_manager=None)
    r, reason = _should_retrain_model(s)
    ok &= check('T1 force_retrain', r is True, f'({reason})')

    # T2 模型未就绪: manager 为空
    s = make_strategy(ensemble_manager=None)
    r, reason = _should_retrain_model(s)
    ok &= check('T2a manager为空', r is True, f'({reason})')
    # T2 模型未就绪: xgb_models 空 (上次训练失败)
    s = make_strategy(ensemble_manager=FakeEnsemble(ready=False))
    r, reason = _should_retrain_model(s)
    ok &= check('T2b xgb_models空', r is True, f'({reason})')

    # T3 interval<=1 → 每次重训
    for iv in (0, 1):
        s = make_strategy(RETRAIN_INTERVAL=iv)
        r, reason = _should_retrain_model(s)
        ok &= check(f'T3 interval={iv}', r is True, f'({reason})')

    # T4 未达间隔 → 沿用
    s = make_strategy(RETRAIN_INTERVAL=5, _rebalances_since_train=2)
    r, reason = _should_retrain_model(s)
    ok &= check('T4 未达间隔', r is False, f'({reason})')

    # T5 达到间隔 → 重训
    s = make_strategy(RETRAIN_INTERVAL=5, _rebalances_since_train=5)
    r, reason = _should_retrain_model(s)
    ok &= check('T5 达到间隔', r is True, f'({reason})')
    s = make_strategy(RETRAIN_INTERVAL=5, _rebalances_since_train=6)
    r, _ = _should_retrain_model(s)
    ok &= check('T5b 超过间隔', r is True)

    # T6 调仓循环模拟 (n=5): 模拟调仓日主流程片段
    #   首次(计数1): 模型未就绪 → 重训, 成功 → 计数归 0
    #   第2~5次(计数1~4): 沿用
    #   第6次(计数5): 重训
    s = make_strategy(force_retrain=False, RETRAIN_INTERVAL=5,
                      ensemble_manager=FakeEnsemble(ready=False))
    retrain_events = []
    for rebal_idx in range(1, 7):  # 第1~6次调仓
        s._rebalances_since_train += 1
        now, reason = _should_retrain_model(s)
        if now:
            retrain_events.append(rebal_idx)
            s._rebalances_since_train = 0  # 训练成功重置
            s.ensemble_manager = FakeEnsemble(ready=True)
    ok &= check('T6 n=5 重训事件', retrain_events == [1, 6],
                f'(got {retrain_events}, expect [1, 6])')
    ok &= check('T6 计数复位', s._rebalances_since_train == 0)

    # T6b n=1: 每次调仓都重训
    s = make_strategy(RETRAIN_INTERVAL=1, ensemble_manager=FakeEnsemble(ready=True))
    events = []
    for _ in range(4):
        s._rebalances_since_train += 1
        now, _ = _should_retrain_model(s)
        if now:
            events.append(1)
            s._rebalances_since_train = 0
    ok &= check('T6b n=1 每次重训', len(events) == 4, f'(got {len(events)})')

    # T7 训练失败(旧模型仍在)不重置计数: 计数持续累积, 下次未达间隔时沿用旧模型
    s = make_strategy(RETRAIN_INTERVAL=5, _rebalances_since_train=4)  # 模型就绪
    s._rebalances_since_train += 1  # 第1次调仓 → 计数5, 达到间隔 → 重训
    r1, reason1 = _should_retrain_model(s)
    # 训练失败: 主流程 return, 不重置计数 (计数仍=5, 旧模型仍就绪)
    s._rebalances_since_train = 5
    # 第2次调仓 → 计数6 ≥ 5 → 再次重训 (失败后不会拖延到第10次)
    s._rebalances_since_train += 1
    r2, reason2 = _should_retrain_model(s)
    ok &= check('T7 失败不重置且达间隔即重训', r1 is True and r2 is True
                and s._rebalances_since_train == 6, f'(r1={r1},{reason1} r2={r2},{reason2})')
    # 失败后计数未达间隔 → 沿用旧模型
    s2 = make_strategy(RETRAIN_INTERVAL=5, _rebalances_since_train=2)  # 模型就绪
    r3, reason3 = _should_retrain_model(s2)
    ok &= check('T7c 失败后未达间隔沿用旧模型', r3 is False, f'({reason3})')
    # 但模型未就绪时无论计数多少都应重试:
    s3 = make_strategy(RETRAIN_INTERVAL=5, ensemble_manager=FakeEnsemble(ready=False))
    s3._rebalances_since_train = 9
    r4, reason4 = _should_retrain_model(s3)
    ok &= check('T7b 模型未就绪强制重试', r4 is True, f'({reason4})')

    print('\n==== 重训间隔测试完成: %s ====' % ('ALL PASS' if ok else '有失败!'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
