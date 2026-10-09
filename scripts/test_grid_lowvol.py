# -*- coding: utf-8 -*-
"""grid_lowvol 策略专项测试: EGARCH / 网格触发 / T+1 / RV 初筛 / 分钟线 / 性能。

遵循项目测试约定: 直接 import 生产模块 + 合成数据 + mock broker 验证行为。
"""
import os
import sys
import time
import types
import datetime

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from strategies.grid_lowvol.egarch import EGARCHModel  # noqa: E402
from strategies.grid_lowvol.grid_engine import GridEngine  # noqa: E402
from strategies.grid_lowvol.data_utils import (  # noqa: E402
    top_low_rv, load_minute_5m, minute_of_day, compute_rv20)

ok = True


def check(name, cond, extra=''):
    global ok
    mark = 'PASS' if cond else 'FAIL'
    if not cond:
        ok = False
    print(f'[{mark}] {name} {extra}')
    return cond


class MockBroker:
    """记录 order_value 调用的 mock broker。"""

    def __init__(self, cash=1e8):
        self.cash = cash
        self.orders = []
        self.positions = {}
        self.current_date = datetime.datetime(2023, 4, 19)

    def order_value(self, code, value, trade_time=None, exec_price_override=None,
                    apply_slippage=True, trade_reason=None):
        is_buy = value > 0
        price = exec_price_override if exec_price_override else 10.0
        qty = abs(value) / price
        self.orders.append({'code': code, 'value': value, 'price': price,
                            'time': trade_time, 'reason': trade_reason})
        # 模拟持仓变化 (简化: 不取整)
        if is_buy:
            self.cash -= abs(value)
            self.positions[code] = self.positions.get(code, 0.0) + qty
        else:
            self.cash += abs(value)
            self.positions[code] = self.positions.get(code, 0.0) - qty
            if self.positions[code] <= 0:
                del self.positions[code]
        return True


def make_minute(prices_seq, date='2023-04-19'):
    """构造 5 分钟 K 线: 每根 (open, high, low, close)。"""
    rows = []
    for i, (o, h, l, c) in enumerate(prices_seq):
        t = pd.Timestamp(f'{date} 09:35:00') + pd.Timedelta(minutes=5 * i)
        rows.append({'trade_time': t, 'trade_date': date, 'open': o,
                     'high': h, 'low': l, 'close': c})
    return pd.DataFrame(rows)


# ===================================================================
# T1: EGARCH 合成数据恢复
# ===================================================================
def t1_egarch():
    print('\n[T1] EGARCH')
    rng = np.random.default_rng(42)
    # 合成 GARCH 型序列: 波动集聚 (σ_t 由过去 |r| 驱动)
    n = 600
    r = np.zeros(n)
    sigma = np.zeros(n)
    s2 = 0.0004
    for t in range(n):
        sigma[t] = np.sqrt(s2)
        r[t] = rng.normal(0, sigma[t])
        s2 = 0.000005 + 0.1 * r[t] ** 2 + 0.88 * s2
    m = EGARCHModel(window=250)
    t0 = time.time()
    okf = m.fit(r[300:])
    dt = time.time() - t0
    check('T1 fit 成功', okf)
    check('T1 参数收敛 (β>0.8)', m.params is not None and m.params[3] > 0.8,
          f'(β={m.params[3]:.3f})' if m.params is not None else '')
    check('T1 预测输出有限且为正', np.isfinite(m.predict_cumvol(20)) and m.predict_cumvol(20) > 0)
    # 波动集聚检测: 高波动段后预测波动应高于低波动段
    m_hi = EGARCHModel(window=250)
    m_lo = EGARCHModel(window=250)
    r_hi = r[100:200] * 3.0   # 高波动段
    r_lo = r[400:500] * 0.5   # 低波动段
    m_hi.fit(r_hi); m_lo.fit(r_lo)
    check('T1 高波动段预测 > 低波动段', m_hi.predict_cumvol(20) > m_lo.predict_cumvol(20),
          f'({m_hi.predict_cumvol(20):.4f} vs {m_lo.predict_cumvol(20):.4f})')
    # 性能: 10 只 (含 numba 首次编译)
    t0 = time.time()
    for _ in range(10):
        rr = rng.normal(0, 0.02, 250)
        m2 = EGARCHModel(); m2.fit(rr); m2.predict_cumvol(20)
    dt10 = time.time() - t0
    check('T1 10只fit+预测 <8s (含numba首次编译)', dt10 < 8, f'({dt10:.1f}s)')
    print(f'      稳态单只 ≈ {dt*1000:.0f}ms (200只/日 ≈ {dt*200:.0f}s)')


# ===================================================================
# T2: 网格线构建
# ===================================================================
def t2_grid_lines():
    print('\n[T2] 网格线')
    ge = GridEngine(levels=5, width_sigma=0.5, band_sigma=2.0)
    sell, buy = ge.build_grid(10.0, 0.02)   # σ=2% → 步长 0.1
    check('T2 上2格卖出线 (10.1, 10.2)', len(sell) == 2 and abs(sell[0] - 10.1) < 1e-9
          and abs(sell[1] - 10.2) < 1e-9, f'({sell})')
    check('T2 下2格买入线 {9.8, 9.9}', len(buy) == 2 and abs(buy[0] - 9.9) < 1e-9
          and abs(buy[1] - 9.8) < 1e-9, f'({buy})')
    # σ 自适应: 高 σ → 更宽网格
    sell2, buy2 = ge.build_grid(10.0, 0.04)
    check('T2 σ自适应 (高σ网格宽)', sell2[0] > sell[0] and buy2[0] < buy[0])


# ===================================================================
# T3: 网格触发逻辑 (2026-08-21 P0 重构: 跨日 GridState 穿越触发)
# ===================================================================
def t3_grid_trigger():
    print('\n[T3] 网格触发 (GridState 跨日)')
    ge = GridEngine(levels=5, width_sigma=0.5, band_sigma=2.0)
    broker = MockBroker()
    state = ge.make_state(10.0, 0.02)   # base=10, step=0.5×0.02×10=0.1
    # 行情: 先涨破上沿线 10.1 (high=10.15) → 卖 1 格; 再跌破下沿线 9.9 (low=9.85) → 买 1 格
    mdf = make_minute([
        (10.00, 10.05, 9.98, 10.02),   # 无触发
        (10.02, 10.15, 10.00, 10.12),  # high≥10.1 → 卖1格
        (10.12, 10.18, 9.85, 9.90),    # low≤9.9 → 买1格
    ])
    stats = ge.execute_day(mdf, broker, 'sh.600000', state, grid_value=100000,
                           max_sell_value=100000)
    check('T3 卖出1格', stats['sells'] == 1, f'({stats})')
    check('T3 买入1格', stats['buys'] == 1, f'({stats})')
    sell_order = [o for o in broker.orders if o['reason'] == 'grid_sell']
    buy_order = [o for o in broker.orders if o['reason'] == 'grid_buy']
    check('T3 卖价=上沿线10.1', sell_order and abs(sell_order[0]['price'] - 10.1) < 1e-9)
    check('T3 买价=下沿线9.9', buy_order and abs(buy_order[0]['price'] - 9.9) < 1e-9)
    if sell_order and buy_order:
        check('T3 先卖后买', sell_order[0]['time'] <= buy_order[0]['time'])
    # 单根 K 线穿越两条线 → 卖 2 买 2 (经典网格: 每穿越一条线交易一次)
    mdf2 = make_minute([
        (10.00, 10.30, 9.80, 10.10),   # high 破 10.1/10.2, low 破 9.9/9.8
    ])
    broker2 = MockBroker()
    state2 = ge.make_state(10.0, 0.02)
    stats2 = ge.execute_day(mdf2, broker2, 'sh.600000', state2, grid_value=100000,
                            max_sell_value=200000)
    check('T3 同K线穿两线卖2格', stats2['sells'] == 2, f'({stats2["sells"]})')
    check('T3 同K线穿两线买2格', stats2['buys'] == 2, f'({stats2["buys"]})')
    # 多根 K 线逐格触发: 连续 2 根上涨 K 线 → 卖 2 格
    mdf3 = make_minute([
        (10.00, 10.15, 9.95, 10.12),
        (10.12, 10.25, 10.10, 10.22),
    ])
    broker3 = MockBroker()
    state3 = ge.make_state(10.0, 0.02)
    stats3 = ge.execute_day(mdf3, broker3, 'sh.600000', state3, grid_value=100000,
                            max_sell_value=200000)
    check('T3 两K线两格卖出', stats3['sells'] == 2, f'({stats3["sells"]})')


# ===================================================================
# T4: T+1 卖出上限 (2026-08-21 P0 重构: GridState)
# ===================================================================
def t4_tplus():
    print('\n[T4] T+1 卖出上限')
    ge = GridEngine(levels=5, width_sigma=0.5, band_sigma=2.0)
    broker = MockBroker()
    state = ge.make_state(10.0, 0.02)
    # 可卖上限 5 万股 → 卖出金额上限 = 50000×10×0.99 = 495000; 但 2 格触发要求 2×100000=200000 < 上限
    mdf = make_minute([
        (10.00, 10.15, 9.95, 10.12),
        (10.12, 10.25, 10.10, 10.22),
    ])
    stats = ge.execute_day(mdf, broker, 'sh.600000', state, grid_value=100000,
                           max_sell_value=495000)
    check('T4 可卖充足时正常卖出', stats['sells'] == 2)
    # 可卖不足: 上限 50000 → 每格卖出金额被限制为 50000
    broker2 = MockBroker()
    state2 = ge.make_state(10.0, 0.02)
    stats2 = ge.execute_day(mdf, broker2, 'sh.600000', state2, grid_value=100000,
                            max_sell_value=50000)
    sold = [o for o in broker2.orders if o['reason'] == 'grid_sell']
    check('T4 卖出金额受 max_sell_value 限制', stats2['sells'] <= 2
          and (not sold or all(abs(o['value']) <= 50000 + 1 for o in sold)),
          f'(sells={stats2["sells"]}, values={[o["value"] for o in sold]})')


# ===================================================================
# T5: RV 初筛
# ===================================================================
def t5_rv():
    print('\n[T5] RV 初筛')
    rng = np.random.default_rng(7)
    dates = pd.date_range('2023-01-03', periods=60, freq='B')
    rmap = {}
    for i in range(30):
        vol = 0.005 + 0.03 * (i % 3)   # 3 档波动
        r = pd.Series(rng.normal(0, vol, 60), index=dates)
        rmap[f's{i:04d}'] = r
    d = pd.Timestamp('2023-04-19')
    top = top_low_rv(rmap, d, k=5, min_days=20)
    check('T5 返回 k 只', len(top) <= 5 and len(top) > 0, f'({len(top)})')
    # 最低波动档 (i%3==0, vol=0.005) 应优先
    low_codes = {f's{i:04d}' for i in range(30) if i % 3 == 0}
    check('T5 低波动优先', top[0] in low_codes, f'(top={top[:3]})')
    # min_days 过滤
    short = pd.Series(rng.normal(0, 0.01, 5), index=dates[:5])
    rmap['short_one'] = short
    top2 = top_low_rv(rmap, d, k=30, min_days=20)
    check('T5 样本不足剔除', 'short_one' not in top2)


# ===================================================================
# T6: 分钟线加载
# ===================================================================
def t6_minute():
    print('\n[T6] 分钟线')
    df = load_minute_5m('sh.600000')
    check('T6 5分钟线加载', df is not None and len(df) > 10000, f'({len(df) if df is not None else 0}根)')
    if df is not None:
        m = minute_of_day(df, datetime.datetime(2023, 4, 19))
        check('T6 单日48根', m is not None and len(m) == 48, f'({len(m) if m is not None else 0})')
        check('T6 按时间排序', m['trade_time'].is_monotonic_increasing)
        no = minute_of_day(df, datetime.datetime(2020, 1, 1))
        check('T6 无数据→None', no is None)


# ===================================================================
# T7: 无未来 (静态断言)
# ===================================================================
def t7_no_future():
    print('\n[T7] 无未来')
    src = open(os.path.join(PROJ, 'strategies', 'grid_lowvol', 'strategy.py'),
               encoding='utf-8').read()
    # [2026-08-21 exdiv-fix1] 严格 ≤T-1: 排除当日(盘前不可知)收益率
    check('T7 RV/EGARCH 用 <date (严格≤T-1)', 'r.index[r.index < date]' in src
          and 'r.index[r.index <= date]' not in src)
    check('T7 网格用当日K线时序', "minute_of_day(load_minute_5m(code), date)" in src)
    check('T7 换仓用开盘', "trade_reason='rebalance_base'" in src
          and "trade_reason='vol_alert_liquidate'" in src)


# ===================================================================
# T8: 2026-08-21 除权审查三项修正专项
# ===================================================================
def _extract_method(path, cls, method):
    import ast
    src = open(path, encoding='utf-8').read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method:
                    return ast.get_source_segment(src, item)
    return None


def t8_exdiv_fixes():
    print('\n[T8] 除权审查三项修正 (2026-08-21)')
    from strategies.grid_lowvol.data_utils import has_dividend_data

    # ---------- F1: 严格 ≤T-1 (无未来) ----------
    for f, label in [('strategy.py', 'strategy'), ('data_utils.py', 'data_utils')]:
        s = open(os.path.join(PROJ, 'strategies', 'grid_lowvol', f),
                 encoding='utf-8').read()
        check(f'F1 {label} 含 < date', 'r.index[r.index < date]' in s)
        check(f'F1 {label} 无 <= date 残留', 'r.index[r.index <= date]' not in s)

    # ---------- F3: has_dividend_data 行为 ----------
    check('F3 有数据股 True', has_dividend_data('sh.600000'))
    check('F3 虚构股 False', not has_dividend_data('sh.999999'))
    strat_src = open(os.path.join(PROJ, 'strategies', 'grid_lowvol', 'strategy.py'),
                     encoding='utf-8').read()
    check('F3 策略含缺口预警', '无除权数据' in strat_src
          and 'has_dividend_data(code)' in strat_src)

    # ---------- F2: 增量锁定行为 (AST 提取真实 handle_bar 执行) ----------
    hb = _extract_method(os.path.join(PROJ, 'strategies', 'grid_lowvol', 'strategy.py'),
                         'GridLowVolStrategy', 'handle_bar')
    check('F2 handle_bar 含增量逻辑', hb is not None
          and 'qty_before' in hb and 'bought_inc' in hb and '_est_qty' not in hb)
    if hb is not None:
        ns = types.SimpleNamespace(
            _plan={'to_liquidate': [], 'to_add': ['sh.600000'],
                   'targets': [], 'vol': {}},
            _today_bought={},
            _verbose=False,
            top_n=10,
            base_fraction=0.4,
            # [2026-08-22 combo] handle_bar 状态机引用
            _combo_state=0,
            _half_done=False,
            _from_flat_recover=False,   # [2026-08-22 recover] HALF 恢复模式标记
            _grid_states={},
            _alert_hist={},
            _peak_value=1e8,
            _prev_nav=1e8,
        )

        class PortfolioMock:
            def __init__(self):
                self.positions = {'sh.600000': types.SimpleNamespace(quantity=1000.0)}
                self.total_value = 1_000_000.0

        class BrokerMock:
            def __init__(self, p):
                self.p = p
                self.calls = []

            def order_value(self, code, value, trade_time=None,
                            exec_price_override=None, apply_slippage=True,
                            trade_reason=None):
                self.calls.append((code, value, trade_reason))
                # 模拟成交: 补仓 200 股 (1000 → 1200)
                self.p.positions[code].quantity += 200.0
                return True

            def order_target_percent(self, code, percent, trade_reason=None):
                self.calls.append((code, percent, trade_reason))
                return True

        p = PortfolioMock()
        b = BrokerMock(p)
        ctx = types.SimpleNamespace(broker=b, portfolio=p,
                                    current_dt=datetime.datetime(2023, 4, 19))
        captured = {}
        # 网格阶段捕获当日锁定 (handle_bar 末尾⑦会清空, 锁定只在网格执行期间生效)
        def fake_run_grid(broker, portfolio, code, date):
            captured['locked'] = ns._today_bought.get(code, 0.0)
            captured['held'] = portfolio.positions[code].quantity
        ns._run_grid = fake_run_grid
        from gpsef.core import StrategyContext  # noqa: E402
        globals()['StrategyContext'] = StrategyContext   # exec 方法签名注解需要
        exec(hb, globals())
        handle_bar(ns, ctx)
        locked = captured.get('locked')
        held = captured.get('held')
        check('F2 网格阶段锁定=200 (非总量1200)', locked is not None
              and abs(locked - 200.0) < 1e-9, f'(locked={locked})')
        check('F2 持仓=1200 且可卖=1000', held is not None
              and abs(held - 1200.0) < 1e-9 and abs(held - locked - 1000.0) < 1e-9,
              f'(held={held}, sellable={held - locked if held else None})')


def t9_pool_filter():
    print('\n[T9] 基础股票池对齐 alpha191 (2026-08-21 pool-fix)')
    # ---------- F1: config 含 stock_pool 段 (与 alpha191 一致过滤) ----------
    import yaml
    cfg = yaml.safe_load(open(os.path.join(PROJ, 'strategies', 'grid_lowvol', 'config.yaml'),
                              encoding='utf-8'))
    sp = cfg.get('data', {}).get('stock_pool', {})
    filters = sp.get('filters', [])
    check('F1 config 含 stock_pool', sp.get('type') == 'dynamic' and len(filters) >= 4,
          f'(filters={filters})')
    check('F1 四过滤与 alpha191 一致',
          'st' in filters and 'new_stock' in filters
          and any(f.startswith('price_min:') for f in filters)
          and any(f.startswith('amount_pct:') for f in filters),
          f'(filters={filters})')

    # ---------- F2: strategy 接入过滤池 ----------
    src = open(os.path.join(PROJ, 'strategies', 'grid_lowvol', 'strategy.py'),
               encoding='utf-8').read()
    check('F2 strategy 用 context.stock_pool', 'context.stock_pool' in src)
    check('F2 strategy 有 _base_pool', 'def _base_pool' in src)
    du = open(os.path.join(PROJ, 'strategies', 'grid_lowvol', 'data_utils.py'),
              encoding='utf-8').read()
    check('F2 RV 初筛限定过滤池', 'only_codes=base_pool' in src and 'only_codes' in du)

    # ---------- F3: top_low_rv only_codes 行为 (合成数据) ----------
    rng = np.random.default_rng(11)
    dates = pd.date_range('2023-01-03', periods=40, freq='B')
    rmap = {}
    for i in range(10):
        vol = 0.005 + 0.02 * (i % 2)   # 偶数低波动, 奇数高波动
        rmap[f's{i:02d}'] = pd.Series(rng.normal(0, vol, 40), index=dates)
    d = pd.Timestamp('2023-03-01')
    only = {'s00', 's02', 's08', 's09'}   # 限子集 (s00/s02 低波动)
    top = top_low_rv(rmap, d, k=3, min_days=20, only_codes=only)
    check('F3 only_codes 限定子集', all(c in only for c in top), f'(top={top})')
    check('F3 子集外股票不入选', not any(c in {'s01', 's03', 's05', 's07'} for c in top))
    top_all = top_low_rv(rmap, d, k=3, min_days=20)
    check('F3 缺省全市场行为不变', len(top_all) == 3 and top_all[0] in {'s00', 's02'})


# ===================================================================
# T10: GridState 跨日网格专项 (2026-08-21 P0 重构)
# ===================================================================
def t10_gridstate():
    print('\n[T10] GridState 跨日网格 (P0 重构)')
    from strategies.grid_lowvol.grid_engine import GridState

    # ① 配对震荡: 买1卖1 → completed
    st = GridState(10.0, 0.1, n_up=2, n_dn=2)
    ns, nb = st.scan(10.0, 9.85, 9.90)    # 下穿 9.9 → 买1
    check('T10 下穿买线买1', nb == 1 and ns == 0, f'({ns},{nb})')
    st.commit(ns, nb)
    check('T10 未配对 → 不完成', not st.completed)
    ns, nb = st.scan(10.15, 10.0, 10.12)  # 上穿 10.1 → 卖1
    check('T10 上穿卖线卖1', ns == 1 and nb == 0, f'({ns},{nb})')
    st.commit(ns, nb)
    check('T10 配对完成', st.completed)

    # ② 阴跌只买: 连续下跌 → bought=2 sold=0, 永不完成 (不重置 → 基准固定)
    st2 = GridState(10.0, 0.1, 2, 2)
    for hi, lo, cl in [(9.95, 9.80, 9.85), (9.85, 9.60, 9.70)]:
        ns, nb = st2.scan(hi, lo, cl)
        st2.commit(ns, nb)
    check('T10 阴跌买2卖0', st2.bought == 2 and st2.sold == 0,
          f'(bought={st2.bought}, sold={st2.sold})')
    check('T10 阴跌不完成 → 基准不重置', not st2.completed)

    # ③ 满格停机: bought 到上限后不再买入
    ns, nb = st2.scan(9.70, 9.30, 9.40)   # 再跌穿 9.6/9.5 → bought 已满
    check('T10 满格停机 (scan不再买)', nb == 0, f'(scan_buy={nb})')
    st2.commit(ns, nb)
    check('T10 bought 卡在上限2', st2.bought == 2)

    # ③b [bug-fix] 单根 K 线大阴线不超买: bought=1 时一根 K 线同时跌破 2 条买线 → 最多再买 1 格
    st5 = GridState(10.0, 0.1, 2, 2)
    st5.prev_price = 10.0
    ns, nb = st5.scan(9.95, 9.80, 9.85)   # 跌破 9.9/9.8 两线 → n_buy=2, bought 0→2
    check('T10 大阴线穿两线买2', nb == 2, f'({nb})')
    st5.commit(ns, nb)                     # bought=2 (满)
    st5.prev_price = 9.85
    ns, nb = st5.scan(9.80, 9.50, 9.60)   # 再跌: 9.7 线也跌破, 但 bought 已满 → 0
    check('T10 满格后大阴线不再买 (不超限)', nb == 0, f'({nb})')
    # 半满时单根 K 线穿两线 → 只补到上限
    st6 = GridState(10.0, 0.1, 2, 2)
    st6.prev_price = 10.0
    ns, nb = st6.scan(9.95, 9.80, 9.85)
    st6.commit(0, 1)                       # bought=1 (模拟只成交 1 格)
    st6.prev_price = 9.85
    ns, nb = st6.scan(9.80, 9.50, 9.60)   # 再跌穿 9.7/9.6/9.5 → 剩余额度 1 → 最多 1
    check('T10 半满大阴线只补到上限 (买1)', nb == 1, f'({nb})')

    # ④ 隔夜跳空: prev 在卖线下, 今日跳空上穿 → 卖触发
    st3 = GridState(10.0, 0.1, 2, 2)
    st3.prev_price = 10.05                # 前日 close (卖线 10.1 下方)
    ns, nb = st3.scan(10.16, 10.14, 10.15)
    check('T10 隔夜跳空上穿卖1', ns == 1, f'({ns})')
    st3.commit(ns, nb)                    # sold=1
    # 线上徘徊不重复: prev=10.15 已在 10.1 上方, high 未触及 10.2 → 无新穿越
    ns, nb = st3.scan(10.16, 10.12, 10.13)
    check('T10 线上徘徊不重复卖', ns == 0, f'({ns})')
    # 离开线再穿越: 回落至线下方再上穿 10.1 → 卖第 2 格 (穿越一次交易一次)
    st3.prev_price = 10.05
    ns, nb = st3.scan(10.15, 10.0, 10.10)
    check('T10 离开线再穿越 → 再卖1', ns == 1, f'({ns})')
    st3.commit(ns, nb)                    # sold=2 → 满格

    # ⑤ reset: 新基准 = 指定价, 格数清零
    st3.reset(10.5)
    check('T10 reset 新基准', abs(st3.base_price - 10.5) < 1e-9 and st3.bought == 0
          and st3.sold == 0)


# ===================================================================
# T11: 趋势闸门 (P0) + 预警 N 日 (P1)
# ===================================================================
def t11_trend_gate():
    print('\n[T11] 趋势闸门 + 预警 N 日 (P0/P1)')
    import types as _t
    td = _extract_method(os.path.join(PROJ, 'strategies', 'grid_lowvol', 'strategy.py'),
                         'GridLowVolStrategy', '_trend_down')
    check('T11 _trend_down 存在', td is not None)
    if td is not None:
        dates = pd.date_range('2023-01-03', periods=40, freq='B')
        up = pd.Series(np.linspace(0.001, 0.01, 40), index=dates)      # 上涨收益率
        dn = pd.Series(np.linspace(-0.001, -0.01, 40), index=dates)    # 下跌收益率
        exec(td, globals())
        d = pd.Timestamp('2023-03-01')
        ns_up = _t.SimpleNamespace(_ret_map={'sh.x': up}, trend_gate_ma=20)
        check('T11 上涨序列非趋势向下', _trend_down(ns_up, 'sh.x', d) is False)
        ns_dn = _t.SimpleNamespace(_ret_map={'sh.x': dn}, trend_gate_ma=20)
        check('T11 下跌序列趋势向下', _trend_down(ns_dn, 'sh.x', d) is True)
        ns_off = _t.SimpleNamespace(_ret_map={'sh.x': dn}, trend_gate_ma=0)
        check('T11 关闭闸门 → False', _trend_down(ns_off, 'sh.x', d) is False)
    src = open(os.path.join(PROJ, 'strategies', 'grid_lowvol', 'strategy.py'),
               encoding='utf-8').read()
    check('T11 预警连续N日逻辑', '_alert_hist' in src and 'vol_alert_days' in src
          and '>= self.vol_alert_days' in src)
    # [bug-fix] EGARCH 塌缩防护: 网格步长下限 1%
    check('T11 daily_sigma 下限1%', 'max(daily_sigma, 0.01)' in src)
    # config 参数
    import yaml
    cfg = yaml.safe_load(open(os.path.join(PROJ, 'strategies', 'grid_lowvol', 'config.yaml'),
                              encoding='utf-8'))
    g = cfg['grid_lowvol']
    check('T11 config 新参数', g.get('grid_width_sigma') == 1.0
          and g.get('vol_alert_mult') == 2.5 and g.get('vol_alert_days') == 5
          and g.get('trend_gate_ma') == 20)


# ===================================================================
# T12: 2026-08-22 收益优化专项 (P0 底仓保护 / P1 底仓提升 / P2 行业分散)
# ===================================================================
def t12_opt():
    print('\n[T12] 收益优化专项 (2026-08-22)')
    from strategies.grid_lowvol.grid_engine import GridState
    from strategies.grid_lowvol.data_utils import load_industry_map

    # ---------- P0: base_locked 底仓保护 ----------
    st = GridState(10.0, 0.1, 2, 2)
    st.base_locked = True                       # 趋势向下 → 锁底仓
    ns, nb = st.scan(10.2, 10.0, 10.1)          # 未买入 → 完全不能卖 (底仓锁)
    check('P0 锁底仓+未买入 → 不卖', ns == 0, f'({ns})')
    st.prev_price = 10.05
    st.bought = 1
    ns, nb = st.scan(10.2, 10.0, 10.1)          # 已买 1 格 → 配对卖 1
    check('P0 锁底仓+买1格 → 配对卖1', ns == 1, f'({ns})')
    st.commit(ns, nb)                           # sold=1, bought=1
    st.prev_price = 10.05
    ns, nb = st.scan(10.3, 10.0, 10.2)          # sold 已达 bought → 底仓仍锁
    check('P0 锁底仓+配对卖完 → 不再卖底仓', ns == 0, f'({ns})')
    # 解锁恢复
    st.base_locked = False
    st.prev_price = 10.05
    ns, nb = st.scan(10.3, 10.0, 10.2)          # 解锁 → 可卖底仓 (sold<2)
    check('P0 解锁 → 恢复卖底仓', ns == 1, f'({ns})')

    # ---------- P2: 行业分散行为 (语义复现自 before_market_open) ----------
    ind_map = load_industry_map()
    check('P2 行业映射加载 ≥5000 只', len(ind_map) >= 5000, f'({len(ind_map)})')
    # 行业分散选择逻辑: 每行业 ≤2 (opt2: 3→2 收敛价值暴露)
    ranked = [('sh.600000', 0.01), ('sh.601988', 0.011), ('sh.601398', 0.012),
              ('sh.601939', 0.013), ('sh.600900', 0.014), ('sz.000002', 0.015),
              ('sz.000001', 0.016), ('sz.002594', 0.017), ('sz.300750', 0.018)]
    max_psi = 2
    ind_count, targets = {}, []
    for c, _ in ranked:
        ind = ind_map.get(c.replace('sh.', '').replace('sz.', ''), '未知')
        if ind_count.get(ind, 0) >= max_psi:
            continue
        ind_count[ind] = ind_count.get(ind, 0) + 1
        targets.append(c)
    check('P2 每行业 ≤2 只', all(v <= 2 for v in ind_count.values()), f'({ind_count})')
    check('P2 分散后仍选出目标', len(targets) >= 5, f'({len(targets)})')
    # 真实数据: 银行类(600000/601988/601398/601939)应被限制
    bank = [c for c in targets if c in ('sh.600000', 'sh.601988', 'sh.601398', 'sh.601939')]
    check('P2 同行业(银行)被截断', len(bank) <= 2, f'(bank={len(bank)})')

    # ---------- config 参数 ----------
    src = open(os.path.join(PROJ, 'strategies', 'grid_lowvol', 'strategy.py'),
               encoding='utf-8').read()
    check('P1/P2 代码落点', 'base_locked' in src and 'max_per_industry' in src
          and 'load_industry_map' in src)
    import yaml
    cfg = yaml.safe_load(open(os.path.join(PROJ, 'strategies', 'grid_lowvol', 'config.yaml'),
                              encoding='utf-8'))
    g = cfg['grid_lowvol']
    check('P1 config base_fraction=0.6', g.get('base_fraction') == 0.6)
    check('P2 config max_per_industry=2', g.get('max_per_industry') == 2)


# ===================================================================
# T13: smallcap 基础池子集 (2026-08-22 用户提议: 过滤池∩市值最低N只)
# ===================================================================
def t13_smallcap_pool():
    print('\n[T13] smallcap 基础池子集')
    from strategies.grid_lowvol.data_utils import smallcap_pool

    # ---------- 真实缓存行为 (市值最低, 严格 ≤date, 都在池内) ----------
    import datetime
    codes = {'sh.600000', 'sh.601988', 'sh.601398', 'sz.300844', 'sz.301192',
             'sh.688251', 'sz.301300', 'sz.301359', 'sz.301209', 'sz.301049'}
    d = datetime.datetime(2023, 4, 19)
    s = smallcap_pool(codes, d, 5)
    check('T13 返回 ≤5 只', len(s) <= 5 and len(s) > 0, f'({len(s)})')
    check('T13 全部在输入池内', set(s).issubset(codes), f'({s})')
    check('T13 微盘优先 (300844/301192 等入选)', 'sz.300844' in s or 'sz.301192' in s, f'({s})')
    # 市值升序验证 (用缓存)
    import pandas as pd
    mv = pd.read_parquet(os.path.join(PROJ, 'data', 'total_mv_cache.parquet'))
    sub = mv[mv['trade_date'] <= pd.Timestamp('2023-04-19')]
    latest = sub.sort_values('trade_date').groupby('ts_code').last()
    vals = [latest.loc[x, 'total_mv'] for x in s]
    check('T13 市值升序', vals == sorted(vals), f'({[round(v/1e8,1) for v in vals]})亿')
    # 无市值数据 → 回退原池
    s2 = smallcap_pool({'sh.600000', 'sz.300844'}, datetime.datetime(2015, 1, 1), 5)
    check('T13 早期日期回退原池', len(s2) == 2, f'({len(s2)})')
    # 空池
    check('T13 空池', smallcap_pool(set(), d, 5) == [])

    # ---------- config 与代码落点 ----------
    src = open(os.path.join(PROJ, 'strategies', 'grid_lowvol', 'strategy.py'),
               encoding='utf-8').read()
    check('T13 strategy 接入', "pool_subset == 'smallcap'" in src
          and 'smallcap_pool(set(pool), date, self.smallcap_size)' in src)
    import yaml
    cfg = yaml.safe_load(open(os.path.join(PROJ, 'strategies', 'grid_lowvol', 'config.yaml'),
                              encoding='utf-8'))
    g = cfg['grid_lowvol']
    check('T13 config 默认 all', g.get('pool_subset') == 'all'
          and g.get('smallcap_size', 300) >= 300)


# ===================================================================
# T14: 组合级风控状态机 (2026-08-22 P0 组合闸门 + P1 回撤熔断 + 分层恢复)
# ===================================================================
def t14_combo_gate():
    print('\n[T14] 组合级风控状态机 (P0+P1+分层恢复)')
    import types as _t
    import datetime as _dt

    # ---------- AST 提取真实 _update_combo_state ----------
    path = os.path.join(PROJ, 'strategies', 'grid_lowvol', 'strategy.py')
    uc = _extract_method(path, 'GridLowVolStrategy', '_update_combo_state')
    check('T14 _update_combo_state 存在', uc is not None)
    if uc is None:
        return
    exec(uc, globals())

    def make_self(cfg=None, state=0, prev=1e8, peak=1e8, above=0,
                  recover=False, half_done=False):
        return _t.SimpleNamespace(
            _combo_cfg=cfg or {'enabled': True, 'index_ma': 40, 'dd_half': 0.05,
                               'dd_flat': 0.08,
                               'dd_recover_full': 0.03, 'trend_confirm_days': 2},
            _combo_state=state, _idx_above_days=above,
            _half_done=half_done, _from_flat_recover=recover,
            _prev_nav=prev, _peak_value=peak,
            _index_close=None,
        )

    def run(self, dd, idx_down=False):
        import types
        self._prev_nav = self._peak_value * (1 + dd)   # 用 prev/peak 表达回撤
        self._idx_flag = idx_down
        self._index_ma60_down = lambda date: idx_down  # mock 指数趋势
        bound = _update_combo_state.__get__(self, types.SimpleNamespace)
        bound(_dt.datetime(2024, 1, 2))
        return self._combo_state

    # ---------- 触发 ----------
    check('T14 回撤-9% → FLAT', run(make_self(), -0.09) == 3)
    check('T14 回撤-6% → HALF', run(make_self(), -0.06) == 2)
    check('T14 回撤-3% + 指数下 → TREND_DOWN', run(make_self(), -0.03, idx_down=True) == 1)
    check('T14 正常 → NORMAL', run(make_self(), 0.0) == 0)
    check('T14 HALF 中回撤加深-9% → 升级 FLAT', run(make_self(state=2), -0.09) == 3)

    # ---------- 死锁回归 (2026-08-22 recover) ----------
    # 原缺陷: 空仓净值恒定 → dd 恒为 -8% → "回撤收窄<3%"恢复永不可达 → 永久空仓。
    # 修复: ①触发守卫(已在 FLAT 不重复锁死) ②恢复改市场信号(指数≥MA 连续确认)+峰值重置。
    # opt2: trend_confirm_days=2 → 确认 = 指数≥MA 连续 2 日 (above 传入 +1 为当日累计)
    s_flat1 = make_self(state=3, above=0)     # FLAT 中 dd 恒定 -8%, 指数确认第 1 日
    check('T14 死锁回归: FLAT dd恒定-8% 确认未满 → 保持 FLAT', run(s_flat1, -0.08) == 3)
    s_flat2 = make_self(state=3, above=1)     # 指数确认第 2 日 (dd 仍恒 -8%!)
    check('T14 死锁回归: FLAT dd恒定-8% 指数确认2日 → 恢复 HALF', run(s_flat2, -0.08) == 2)
    check('T14 死锁回归: FLAT→HALF 峰值重置(基线归零)',
          abs(s_flat2._peak_value - s_flat2._prev_nav) < 1e-9)
    check('T14 死锁回归: FLAT→HALF 进入恢复模式', s_flat2._from_flat_recover is True)
    check('T14 死锁回归: FLAT→HALF 重置确认计数', s_flat2._idx_above_days == 0)
    # opt2: FLAT 升级时清零确认计数 (防跨状态累计导致清仓次日即恢复)
    s_up0 = make_self(state=2, above=5)       # HALF 中回撤加深 → 升级 FLAT
    run(s_up0, -0.09)
    check('T14 opt2: FLAT 升级清零确认计数', s_up0._idx_above_days == 0
          and s_up0._combo_state == 3)

    # ---------- 分层恢复 (市场信号确认) ----------
    s3r = make_self(state=3, above=1, prev=1e8 * 0.975, peak=1e8)  # FLAT, 指数确认满
    check('T14 FLAT 指数确认 → HALF', run(s3r, -0.025) == 2)
    s2r = make_self(state=2, above=1, prev=1e8 * 0.988, peak=1e8)  # HALF, 指数确认满+回撤-1.2%
    check('T14 HALF 指数确认+回撤收窄 → NORMAL', run(s2r, -0.012) == 0)
    s2x = make_self(state=2, above=1, prev=1e8 * 0.96, peak=1e8)   # 指数确认满但回撤-4%
    check('T14 HALF 指数确认但回撤未收窄 → 保持', run(s2x, -0.04) == 2)
    s2y = make_self(state=2, above=0, prev=1e8 * 0.988, peak=1e8)  # 回撤-1.2% 但指数未确认
    check('T14 HALF 回撤收窄但指数未确认 → 保持', run(s2y, -0.012) == 2)
    s1a = make_self(state=1, above=0, prev=1e8, peak=1e8)  # TREND_DOWN 第1日确认
    check('T14 TREND_DOWN 第1日仍保持', run(s1a, 0.0) == 1)
    s1b = make_self(state=1, above=1, prev=1e8, peak=1e8)  # 第2日确认
    check('T14 TREND_DOWN 第2日确认 → NORMAL', run(s1b, 0.0) == 0)

    # ---------- 防振荡滞后带 ----------
    check('T14 回撤-4.9% 不触发 HALF', run(make_self(), -0.049) == 0)
    check('T14 回撤-5.1% 触发 HALF', run(make_self(), -0.051) == 2)
    s_c = make_self(state=2, prev=1e8 * 0.96, peak=1e8)   # HALF 中回撤-4%
    check('T14 HALF 回撤-4% 不恢复 (滞后带)', run(s_c, -0.04) == 2)

    # ---------- 关闭 ----------
    cfg_off = {'enabled': False, 'index_ma': 40, 'dd_half': 0.05, 'dd_flat': 0.08,
               'dd_recover_full': 0.03, 'trend_confirm_days': 2}
    check('T14 disabled → 强制 NORMAL', run(make_self(cfg=cfg_off, state=2,
                                                      prev=1e8 * 0.9, peak=1e8), -0.10) == 0)

    # ---------- handle_bar 状态执行 (AST 断言) ----------
    hb = _extract_method(path, 'GridLowVolStrategy', 'handle_bar')
    check('T14 handle_bar FLAT 清仓', hb is not None and 'combo == 3' in hb
          and "trade_reason='combo_flat'" in hb)
    check('T14 handle_bar HALF 减半', hb is not None and 'combo == 2 and not self._half_done' in hb
          and "trade_reason='combo_half'" in hb)
    # [opt2] 减半语义: order_target_value(当前市值×0.5), 禁止 order_target_percent(0.5) 反向加仓
    check('T14 opt2: HALF 减半=当前市值×0.5', hb is not None
          and 'order_target_value(code, float(pos_.market_value) * 0.5' in hb
          and 'order_target_percent(code, 0.5' not in hb)
    check('T14 建仓限制 NORMAL/HALF恢复', hb is not None
          and 'allow_build = (combo == 0) or (combo == 2 and self._from_flat_recover)' in hb)
    check('T14 HALF 恢复半仓预算', hb is not None and 'per_stock *= 0.5' in hb)
    check('T14 峰值/前值跟踪', '_peak_value = max(self._peak_value' in hb
          and 'self._prev_nav = portfolio.total_value' in hb)
    src = open(path, encoding='utf-8').read()
    check('T14 组合禁买', 'self._combo_state == 0' in src
          and 'state.base_locked = trend_down or self._combo_state != 0' in src)

    # ---------- config ----------
    import yaml
    cfg = yaml.safe_load(open(os.path.join(PROJ, 'strategies', 'grid_lowvol', 'config.yaml'),
                              encoding='utf-8'))
    cg = cfg['grid_lowvol']['combo_gate']
    check('T14 config combo_gate(opt2缩短空仓)', cg.get('enabled') is True
          and cg.get('index_ma') == 40
          and cg.get('dd_half') == 0.05 and cg.get('dd_flat') == 0.08
          and cg.get('dd_recover_full') == 0.03
          and cg.get('trend_confirm_days') == 2
          and 'dd_recover_half' not in cg)


def main():
    print('=' * 70)
    print('grid_lowvol 策略专项测试')
    print('=' * 70)
    t1_egarch()
    t2_grid_lines()
    t3_grid_trigger()
    t4_tplus()
    t5_rv()
    t6_minute()
    t7_no_future()
    t8_exdiv_fixes()
    t9_pool_filter()
    t10_gridstate()
    t11_trend_gate()
    t12_opt()
    t13_smallcap_pool()
    t14_combo_gate()
    print('\n' + '=' * 70)
    print(f'结果: {ok}')
    print('ALL PASS' if ok else '有失败!')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
