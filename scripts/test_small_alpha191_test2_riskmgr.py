# -*- coding: utf-8 -*-
"""
方案A 组合级风控 测试 (small_alpha191_test2)
=============================================
覆盖: 回撤熔断状态机 (触发/恢复/死锁回归/边界epsilon/disabled)
      调仓日趋势过滤 | build_scale 组合 | AST 源码断言 | config 断言 | 指数数据可用性
方法: AST 从生产源码提取真实方法代码 + SimpleNamespace mock self (项目测试约定)
运行: .venv/Scripts/python.exe scripts/test_small_alpha191_test2_riskmgr.py
"""
import os
import sys
import types
import logging
import inspect
import datetime as _dt
from datetime import timedelta

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRAT_DIR = os.path.join(PROJ, 'strategies', 'small_alpha191_test2')
STRAT_PY = os.path.join(STRAT_DIR, 'strategy.py')
CFG_YAML = os.path.join(STRAT_DIR, 'config.yaml')

PASS = 0
FAIL = 0


def check(name, cond, extra=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  [PASS] {name}')
    else:
        FAIL += 1
        print(f'  [FAIL] {name} {extra}')


def load_method(path, cls, method):
    """AST 提取类方法真实源码 → exec 编译为可调用函数 (不 import 策略模块, 避免重依赖)。"""
    import ast
    src = open(path, encoding='utf-8').read()
    tree = ast.parse(src)
    # 方法签名/注解引用的符号 (不 import 重模块, 提供 dummy)
    _DummyCtx = type('StrategyContext', (), {})
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method:
                    seg = ast.get_source_segment(src, item)
                    g = dict(logger=logging.getLogger('t'), pd=pd, np=np, os=os,
                             datetime=_dt.datetime, timedelta=timedelta,
                             StrategyContext=_DummyCtx)
                    exec(compile(seg, f'<{cls}.{method}>', 'exec'), g)
                    return g[method]
    return None


def extract_src(path, cls, method):
    """AST 提取方法源码字符串 (供文本断言用, 不编译执行)。"""
    import ast
    src = open(path, encoding='utf-8').read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method:
                    return ast.get_source_segment(src, item)
    return None


DEF_DD = {'enabled': True, 'dd_half': 0.10, 'dd_flat': 0.15,
          'dd_recover_full': 0.05, 'trend_confirm_days': 3,
          'index_code': 'sh000001', 'index_ma': 20}
DEF_TG = {'enabled': True, 'index_code': 'sh000001', 'ma': 20, 'half_position': True}


def make_self(dd_cfg=None, state=0, prev=1e6, peak=1e6, above=0, scale=1.0,
              tg_cfg=None, idx_down=False):
    return types.SimpleNamespace(
        _dd_cfg=dd_cfg or dict(DEF_DD),
        _dd_state=state, _dd_scale=scale, _dd_above_days=above,
        _prev_nav=prev, _dd_peak=peak,
        _trend_gate_cfg=tg_cfg or dict(DEF_TG),
        _trend_scale=1.0,
        _breaker_index_down=lambda date, ma: idx_down,
    )


# ===================================================================
def t_breaker():
    print('\n[T-回撤熔断状态机 _update_dd_breaker]')
    up = load_method(STRAT_PY, 'SmallAlpha191Strategy', '_update_dd_breaker')
    check('方法提取成功', up is not None)
    if up is None:
        return

    def run(self, date=_dt.datetime(2024, 1, 2)):
        up.__get__(self, types.SimpleNamespace)(date)
        return self._dd_state, self._dd_scale

    # ---------- 触发 (保守优先) ----------
    s = make_self(prev=1e6 * 0.84, peak=1e6)          # dd=-16% ≤ -15%
    st, sc = run(s)
    check('触发: 回撤-16% → FLAT(scale 0)', st == 2 and sc == 0.0, f'({st},{sc})')
    s = make_self(prev=1e6 * 0.88, peak=1e6)          # dd=-12% ≤ -10%
    st, sc = run(s)
    check('触发: 回撤-12% → HALF(scale 0.5)', st == 1 and sc == 0.5, f'({st},{sc})')
    s = make_self(prev=1e6 * 0.95, peak=1e6)          # dd=-5% 无触发
    st, sc = run(s)
    check('无触发: 回撤-5% → NORMAL(scale 1)', st == 0 and sc == 1.0, f'({st},{sc})')

    # ---------- 边界 epsilon (浮点 85/100-1=-0.1499.. 类误判) ----------
    s = make_self(prev=1e6 * 0.849999, peak=1e6)      # dd≈-15.0001% 明确超
    st, sc = run(s)
    check('边界: dd=-15.0001% → FLAT', st == 2, f'({st})')
    s = make_self(prev=1e6 * 0.850001, peak=1e6)      # dd≈-14.9999% 未超 flat
    st, sc = run(s)
    check('边界: dd=-14.9999% → HALF 非 FLAT (epsilon 不误判)', st == 1, f'({st})')

    # ---------- 死锁回归 (grid_lowvol 教训: 空仓净值恒定 → 恢复必须市场信号) ----------
    s = make_self(state=2, above=1, prev=1e6 * 0.84, peak=1e6)   # FLAT 中 dd 恒-16%
    st, sc = run(s)
    check('死锁回归: FLAT dd恒定-16% 确认未满 → 保持 FLAT', st == 2, f'({st})')
    s2 = make_self(state=2, above=2, prev=1e6 * 0.84, peak=1e6)  # 指数确认第3日 (dd 仍-16%)
    st, sc = run(s2)
    check('死锁回归: FLAT dd恒定-16% 指数确认3日 → 恢复 HALF', st == 1 and sc == 0.5, f'({st},{sc})')
    check('死锁回归: FLAT→HALF 峰值重置(基线归零)', abs(s2._dd_peak - s2._prev_nav) < 1e-9)
    check('死锁回归: FLAT→HALF 确认计数清零', s2._dd_above_days == 0)
    # 振荡回归: max 实现峰值不重置 → FLAT→HALF 后 dd 残留-16% 次日必回 FLAT (死锁振荡)。
    # 直接赋值修复后: 峰值=恢复点, 次日 dd 重新起算; 净值小跌-7% (在-10%~-5%间) → 保持 HALF 不回 FLAT
    s3 = make_self(state=1, above=0, prev=s2._dd_peak * 0.93, peak=s2._dd_peak, scale=0.5)
    st, sc = run(s3)
    check('振荡回归: 峰值重置后净值-7% → 保持 HALF 不回 FLAT', st == 1 and sc == 0.5, f'({st},{sc})')

    # ---------- 分层恢复 (HALF→NORMAL 双条件) ----------
    s = make_self(state=1, above=2, prev=1e6 * 0.965, peak=1e6)  # dd=-3.5% 收窄
    st, sc = run(s)
    check('恢复: HALF 回撤收窄-3.5% → NORMAL', st == 0 and sc == 1.0, f'({st},{sc})')
    s = make_self(state=1, above=2, prev=1e6 * 0.94, peak=1e6)   # dd=-6% 未收窄
    st, sc = run(s)
    check('滞后带: HALF 回撤-6% 未收窄 → 保持 HALF', st == 1, f'({st})')

    # ---------- 升级: HALF 中回撤加深 → FLAT ----------
    s = make_self(state=1, prev=1e6 * 0.80, peak=1e6)   # dd=-20%
    st, sc = run(s)
    check('升级: HALF 回撤加深-20% → FLAT + 计数清零', st == 2 and s._dd_above_days == 0, f'({st})')

    # ---------- disabled ----------
    cfg_off = dict(DEF_DD); cfg_off['enabled'] = False
    s = make_self(dd_cfg=cfg_off, state=2, prev=1e6 * 0.7, peak=1e6)
    st, sc = run(s)
    check('disabled → 强制 NORMAL scale 1', st == 0 and sc == 1.0, f'({st},{sc})')


def t_trend_gate():
    print('\n[T-调仓日趋势过滤 _update_rebalance_trend_gate]')
    tg = load_method(STRAT_PY, 'SmallAlpha191Strategy', '_update_rebalance_trend_gate')
    check('方法提取成功', tg is not None)
    if tg is None:
        return

    def run(self, date=_dt.datetime(2024, 1, 2)):
        tg.__get__(self, types.SimpleNamespace)(date)
        return self._trend_scale

    s = make_self(idx_down=True)
    check('指数<MA → 建仓 x0.5', run(s) == 0.5, f'({run(s)})')
    s = make_self(idx_down=False)
    check('指数≥MA → 建仓 x1.0', run(s) == 1.0)
    cfg_off = dict(DEF_TG); cfg_off['enabled'] = False
    s = make_self(tg_cfg=cfg_off, idx_down=True)
    check('disabled → x1.0', run(s) == 1.0)
    cfg_nohalf = dict(DEF_TG); cfg_nohalf['half_position'] = False
    s = make_self(tg_cfg=cfg_nohalf, idx_down=True)
    check('half_position=false → x1.0', run(s) == 1.0)
    # [2026-08-31 实验2] 深度条件: require_ma_down 配置读取 + MA 下行逻辑存在
    init = extract_src(STRAT_PY, 'SmallAlpha191Strategy', 'initialize')
    tg_src = open(STRAT_PY, encoding='utf-8').read()
    check('require_ma_down 配置读取', init is not None and 'require_ma_down' in init)
    check('深度条件实现(MA_t<MA_{t-1})', 'ma_now < ma_prev' in tg_src)


def t_build_scale():
    print('\n[T-build_scale 组合 _current_build_scale]')
    bs = load_method(STRAT_PY, 'SmallAlpha191Strategy', '_current_build_scale')
    check('方法提取成功', bs is not None)
    if bs is None:
        return

    def run(dd_scale, trend_scale):
        s = types.SimpleNamespace(_dd_scale=dd_scale, _trend_scale=trend_scale)
        return bs.__get__(s, types.SimpleNamespace)()

    check('0.5 × 0.5 = 0.25 (HALF+趋势双降)', run(0.5, 0.5) == 0.25)
    check('1.0 × 1.0 = 1.0', run(1.0, 1.0) == 1.0)
    check('0.0 × 0.5 = 0.0 (FLAT 一票否决)', run(0.0, 0.5) == 0.0)
    check('0.5 × 1.0 = 0.5', run(0.5, 1.0) == 0.5)


def t_ast():
    print('\n[T-AST 源码断言]')
    src = open(STRAT_PY, encoding='utf-8').read()
    hb = extract_src(STRAT_PY, 'SmallAlpha191Strategy', 'handle_bar')
    bmo = extract_src(STRAT_PY, 'SmallAlpha191Strategy', 'before_market_open')
    init = extract_src(STRAT_PY, 'SmallAlpha191Strategy', 'initialize')

    check('handle_bar 建仓应用 build_scale',
          hb is not None and 'target_w * build_scale' in hb
          and 'build_scale = self._current_build_scale()' in hb)
    check('handle_bar 调仓日先更新趋势门',
          hb is not None and 'self._update_rebalance_trend_gate(date)' in hb)
    check('handle_bar 日终回撤跟踪',
          hb is not None and 'self._prev_nav = portfolio.total_value' in hb
          and 'self._dd_peak = max(self._dd_peak, portfolio.total_value)' in hb)
    check('before_market_open 每日更新回撤状态机',
          bmo is not None and 'self._update_dd_breaker(current_date)' in bmo)
    check('before_market_open 熔断更新在早退 return 之前(每日执行)',
          bmo is not None and 'need_rebalance_check' in bmo
          and bmo.index('self._update_dd_breaker(current_date)') < bmo.index('if not need_rebalance_check'))
    check('before_market_open FLAT 清空目标',
          bmo is not None and '_dd_state == 2' in bmo and 'self.target_stocks = []' in bmo)
    check('initialize 加载双风控配置',
          init is not None and "self.config.get('drawdown_breaker', {})" in init
          and "self.config.get('rebalance_trend_gate', {})" in init)
    check('FLAT 触发时计数清零', '_dd_above_days = 0' in src)
    check('恢复用市场信号非净值', '_dd_above_days >= c[\'trend_confirm_days\']' in src)
    check('峰值重置(直接赋值)', '_dd_peak = self._prev_nav' in src
          and '_dd_peak = max(self._dd_peak, self._prev_nav)' not in src)
    # 无未来: 指数严格 < date
    bd = extract_src(STRAT_PY, 'SmallAlpha191Strategy', '_breaker_index_down')
    check('指数判断 ≤T-1 无未来', bd is not None and 'index < pd.Timestamp(date)' in bd)
    lc = extract_src(STRAT_PY, 'SmallAlpha191Strategy', '_load_breaker_index_close')
    check('指数加载 data/index_daily', lc is not None and "'data', 'index_daily'" in lc)


def t_config():
    print('\n[T-config.yaml 断言]')
    import yaml
    cfg = yaml.safe_load(open(CFG_YAML, encoding='utf-8'))
    check('策略名/class 指向 test2',
          cfg.get('strategy', {}).get('name') == 'SmallAlpha191Test2'
          and 'small_alpha191_test2' in cfg.get('strategy', {}).get('class', ''))
    check('auc_threshold 0.52→0.54', cfg.get('auc_threshold') == 0.54)
    se = cfg.get('staged_exit', {})
    s1 = se.get('stage1', {})
    check('S1 pt_sigma 3.0 (最终裁决 R12)', s1.get('pt_sigma') == 3.0, f'({s1.get("pt_sigma")})')
    check('S1 sl_sigma -3.0 (最终裁决 R12)', s1.get('sl_sigma') == -3.0)
    dd = cfg.get('drawdown_breaker', {})
    check('drawdown_breaker 段', dd.get('enabled') is True
          and dd.get('dd_half') == 0.10 and dd.get('dd_flat') == 0.15
          and dd.get('dd_recover_full') == 0.05
          and dd.get('trend_confirm_days') == 3
          and dd.get('index_code') == 'sh000001' and dd.get('index_ma') == 20)
    tg = cfg.get('rebalance_trend_gate', {})
    check('rebalance_trend_gate 段 (MA10 最终裁决)', tg.get('enabled') is True
          and tg.get('index_code') == 'sh000001' and tg.get('ma') == 10
          and tg.get('half_position') is True
          and tg.get('require_ma_down') is False)


def t_data():
    print('\n[T-指数数据可用性]')
    p = os.path.join(PROJ, 'data', 'index_daily', 'sh000001.parquet')
    check('sh000001.parquet 存在', os.path.exists(p))
    try:
        df = pd.read_parquet(p)
        check('含 date/close 列', 'date' in df.columns and 'close' in df.columns)
        check('数据量充足', len(df) > 500, f'({len(df)})')
    except Exception as e:
        check('可读取', False, f'({e})')


# ===================================================================
if __name__ == '__main__':
    t_breaker()
    t_trend_gate()
    t_build_scale()
    t_ast()
    t_config()
    t_data()
    print(f'\n结果: {PASS} PASS, {FAIL} FAIL')
    sys.exit(1 if FAIL else 0)
