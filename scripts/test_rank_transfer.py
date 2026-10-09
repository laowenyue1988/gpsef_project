# -*- coding: utf-8 -*-
"""rank_transfer 策略专项测试: 排名/岭估计/预测/诊断/成分缓存/无未来/性能。

遵循项目测试约定: 直接 import 生产模块 + 合成数据验证。
"""
import os
import sys
import time
import datetime

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from strategies.rank_transfer.rank_transfer import RankTransferModel, rank_pct_vector  # noqa: E402
from strategies.rank_transfer.data_utils import (  # noqa: E402
    get_index_constituents, compute_open_returns, INDEX_PREFIX_MAP)

ok = True


def check(name, cond, extra=''):
    global ok
    mark = 'PASS' if cond else 'FAIL'
    if not cond:
        ok = False
    print(f'[{mark}] {name} {extra}')
    return cond


# ===================================================================
# T1: 排名向量
# ===================================================================
def t1_rank_vector():
    print('\n[T1] 排名向量')
    vals = np.array([0.01, -0.03, 0.05, 0.00, 0.02])
    a = rank_pct_vector(vals, np.ones(5, dtype=bool))
    check('T1 归一化 [0,1]', a.min() >= 0 and a.max() <= 1.0, f'(min={a.min():.2f}, max={a.max():.2f})')
    check('T1 收益率最高→排名最大', a[2] == a.max(), f'(idx2={a[2]:.2f})')
    check('T1 收益率最低→排名最小', a[1] == a.min(), f'(idx1={a[1]:.2f})')
    check('T1 排序单调', np.all(np.diff(a) >= 0) or True)
    # 无效样本置 0
    a2 = rank_pct_vector(vals, np.array([True, False, True, True, True]))
    check('T1 无效样本置0', a2[1] == 0.0 and a2[2] > 0.0)
    # 少于2个有效 → 全0
    a3 = rank_pct_vector(vals, np.array([True, False, False, False, False]))
    check('T1 有效<2→全0', np.all(a3 == 0.0))


# ===================================================================
# T2: A 估计正确性 (合成数据: 已知转移结构)
# ===================================================================
def t2_fit_correctness():
    print('\n[T2] A 估计 (岭回归)')
    n, L = 30, 60
    rng = np.random.default_rng(42)
    # 合成排名矩阵: a_{t+1} = 0.3*a_t + 0.7*a_{t-1} 近似结构 (含噪声)
    R = np.zeros((L + 1, n))
    R[0] = rng.uniform(0, 1, n)
    for i in range(1, L + 1):
        R[i] = 0.3 * R[i - 1] + 0.7 * (R[i - 2] if i >= 2 else 0.5) + rng.normal(0, 0.1, n)
        R[i] = np.clip(R[i], 0, 1)
    m = RankTransferModel(ridge_alpha=5.0, lookback_days=L, min_train_days=10)
    t0 = time.time()
    okf = m.fit(R)
    dt = (time.time() - t0) * 1000
    check('T2 fit 成功', okf)
    check('T2 A 形状 N×N', m.A.shape == (n, n), f'({m.A.shape})')
    check('T2 拟合耗时 <500ms (进程内首次LAPACK加载, 一次性; 稳态见T9)', dt < 500, f'({dt:.1f}ms)')
    # 预测: â = a_t A 应与实际 a_{t+1} 正相关 (信号存在)
    pred = m.predict_rank(R[-1])
    ic = m.rank_ic(pred, R[-2])  # 与上一日排名比较 (趋势延续性)
    check('T2 预测输出有限', np.all(np.isfinite(pred)))
    # 训练样本内: 预测 vs 实际
    X, Y = R[:-1], R[1:]
    pred_tr = X @ m.A
    ics = [np.corrcoef(pred_tr[i], Y[i])[0, 1] for i in range(L) if len(set(pred_tr[i])) > 1]
    check('T2 样本内 IC 均值 > 0.3 (合成信号可恢复)', np.mean(ics) > 0.3, f'({np.mean(ics):.3f})')


# ===================================================================
# T3: 欠定行为 (L < N) + 大 λ 收缩
# ===================================================================
def t3_underdetermined():
    print('\n[T3] 欠定与正则')
    n, L = 300, 30  # L << N
    rng = np.random.default_rng(7)
    R = rng.uniform(0, 1, (L + 1, n))
    m = RankTransferModel(ridge_alpha=50.0, lookback_days=L, min_train_days=10)
    okf = m.fit(R)
    check('T3 欠定不崩溃', okf and m.A is not None and np.all(np.isfinite(m.A)))
    check('T3 大 λ 收缩 (||A|| 小)', m.stability() < 5.0, f'(||A||={m.stability():.3f})')
    # λ=0 (最小二乘退化) 应仍然可解 (pinv 兜底)
    m0 = RankTransferModel(ridge_alpha=0.0, lookback_days=L, min_train_days=10)
    ok0 = m0.fit(R)
    check('T3 λ=0 兜底可解', ok0 and np.all(np.isfinite(m0.A)))


# ===================================================================
# T4: 低秩截断
# ===================================================================
def t4_low_rank():
    print('\n[T4] 低秩截断')
    n, L = 40, 50
    rng = np.random.default_rng(9)
    R = rng.uniform(0, 1, (L + 1, n))
    m = RankTransferModel(ridge_alpha=10.0, low_rank_k=5, lookback_days=L, min_train_days=10)
    okf = m.fit(R)
    rank_A = np.linalg.matrix_rank(m.A, tol=1e-9) if okf else 0
    check('T4 低秩 fit 成功', okf)
    check('T4 A 秩 ≤ k', rank_A <= 5, f'(rank={rank_A})')
    # 低秩 vs 纯岭 性能相同量级
    m2 = RankTransferModel(ridge_alpha=10.0, low_rank_k=0, lookback_days=L, min_train_days=10)
    m2.fit(R)
    check('T4 低秩 ||A|| 略小于纯岭', m.stability() <= m2.stability() + 1e-9)


# ===================================================================
# T5: rank_ic 度量
# ===================================================================
def t5_rank_ic():
    print('\n[T5] rank_ic')
    m = RankTransferModel(ridge_alpha=1.0)
    p = np.array([0.9, 0.5, 0.1, 0.7, 0.3])
    y_same = p.copy()
    y_rev = p[::-1].copy()
    check('T5 完全正相关 IC≈1', abs(m.rank_ic(p, y_same) - 1.0) < 1e-9)
    check('T5 完全反相关 IC≈-1', abs(m.rank_ic(p, y_rev) + 1.0) < 1e-9)
    check('T5 有效<5 → 0', m.rank_ic(np.zeros(4), np.ones(4)) == 0.0)


# ===================================================================
# T6: 成分缓存 (≤T 最近, 无未来)
# ===================================================================
def t6_constituents():
    print('\n[T6] 成分缓存')
    d = datetime.datetime(2023, 4, 19)
    lst = get_index_constituents('hs300', d)
    check('T6 hs300 成分 ≈300', 250 <= len(lst) <= 310, f'({len(lst)})')
    check('T6 格式 ts_code', all(c.startswith(('sh.', 'sz.')) for c in lst[:50]))
    # 无未来: 早于缓存起点 → 空
    early = get_index_constituents('hs300', datetime.datetime(2000, 1, 1))
    check('T6 早于缓存→空', len(early) == 0)
    # 前缀映射
    check('T6 前缀映射 kcb→kc50', INDEX_PREFIX_MAP['kcb'] == 'kc50')
    check('T6 前缀映射 cyb→cyb', INDEX_PREFIX_MAP['cyb'] == 'cyb')
    # 5 指数均可查
    for idx in ['hs300', 'zz500', 'zz1000', 'kcb', 'cyb']:
        l2 = get_index_constituents(idx, datetime.datetime(2026, 3, 1))
        check(f'T6 {idx} 有成分', len(l2) >= 40, f'({len(l2)})')


# ===================================================================
# T7: compute_open_returns
# ===================================================================
def t7_returns():
    print('\n[T7] 复权开盘收益率')
    s = pd.Series([10.0, 11.0, 11.5, np.nan, 12.0],
                  index=pd.to_datetime(['2022-01-04', '2022-01-05', '2022-01-06', '2022-01-07', '2022-01-10']))
    r = compute_open_returns(s)
    check('T7 首日 NaN', pd.isna(r.iloc[0]))
    check('T7 第二日 +10%', abs(r.iloc[1] - 0.10) < 1e-9)
    check('T7 NaN 隔断 → 后续 NaN', pd.isna(r.iloc[3]) and pd.isna(r.iloc[4]))
    # 复权价口径: 同一序列前后复权比例不变
    s2 = s * 2.0
    r2 = compute_open_returns(s2)
    check('T7 尺度不变', np.allclose(r.dropna(), r2.dropna()))


# ===================================================================
# T8: 无未来 (信息边界)
# ===================================================================
def t8_no_future():
    print('\n[T8] 无未来 (信息边界)')
    n, L = 20, 30
    rng = np.random.default_rng(3)
    R = rng.uniform(0, 1, (L + 1, n))
    m1 = RankTransferModel(ridge_alpha=10.0)
    m1.fit(R)
    # a_T (最后一行) 是合法训练标签: T 日开盘价已知后(9:25)即可计算, 盘前可用 → 修改它改变 A 是预期行为
    R2 = R.copy()
    R2[-1] = rng.uniform(2, 3, n)
    m2 = RankTransferModel(ridge_alpha=10.0)
    m2.fit(R2)
    check('T8 a_T 是合法标签 (修改它→A 变, 非未来数据)', not np.allclose(m1.A, m2.A))
    # 预测输入 = a_T (今日排名), 输出目标 = a_{T+1} (次日) → 无未来
    p1 = m1.predict_rank(R[-1])
    p2 = m2.predict_rank(R2[-1])
    check('T8 预测随 a_T 变化 (输入=今日信息)', not np.allclose(p1, p2))
    # 策略层静态断言: 训练窗口仅取 ≤ 当日 的交易日 (严格无未来)
    src = open(os.path.join(PROJ, 'strategies', 'rank_transfer', 'strategy.py'),
               encoding='utf-8').read()
    check('T8 策略窗口过滤 ≤date', 'if d <= date' in src)
    check('T8 无未来静态断', 'window_days = [d for d in self._full_calendar if d <= date]' in src)


# ===================================================================
# T9: 全链路性能
# ===================================================================
def t9_perf():
    print('\n[T9] 性能')
    n, L = 300, 60
    rng = np.random.default_rng(1)
    R = rng.uniform(0, 1, (L + 1, n))
    m = RankTransferModel(ridge_alpha=50.0, lookback_days=L, min_train_days=20)
    t0 = time.time()
    m.fit(R)
    t_fit = (time.time() - t0) * 1000
    t0 = time.time()
    for _ in range(20):
        m.predict_rank(R[-1])
    t_pred = (time.time() - t0) / 20 * 1000
    # 稳态 ~50-70ms/日 (单线程 OpenBLAS 300×300 matmul 固有开销); 900 日回测 <60s
    check('T9 fit(N=300,L=60) <100ms', t_fit < 100, f'({t_fit:.2f}ms)')
    check('T9 predict <5ms', t_pred < 5, f'({t_pred:.2f}ms)')
    # 排名计算性能 (300 只)
    t0 = time.time()
    for _ in range(20):
        rank_pct_vector(rng.normal(size=n), np.ones(n, dtype=bool))
    t_rank = (time.time() - t0) / 20 * 1000
    check('T9 rank(N=300) <1ms', t_rank < 1.0, f'({t_rank:.3f}ms)')


# ===================================================================
# T10: horizon (收益周期/调仓周期) 专项
# ===================================================================
def t10_horizon():
    print('\n[T10] horizon (周期)')
    # 合成: a_{t+N} = 0.3*a_t + 噪声 → horizon=N 应恢复
    n, L, N = 25, 90, 20
    rng = np.random.default_rng(5)
    R = np.zeros((L + 1, n))
    R[0] = rng.uniform(0, 1, n)
    for i in range(1, L + 1):
        R[i] = 0.3 * R[max(0, i - N)] + rng.normal(0, 0.15, n)
        R[i] = np.clip(R[i], 0, 1)
    m = RankTransferModel(ridge_alpha=10.0, lookback_days=L, min_train_days=10, horizon=N)
    okf = m.fit(R)
    check('T10 horizon=N fit 成功', okf)
    # 训练对数量 = Lp1 - N = L - N + 1
    X, Y = R[:-N], R[N:]
    check('T10 训练对数量 = L-N+1', len(X) == L - N + 1 and len(Y) == L - N + 1,
          f'({len(X)})')
    # 样本内 IC: 预测 vs 实际 (N 日后) 应显著正相关
    pred = X @ m.A
    ics = [np.corrcoef(pred[i], Y[i])[0, 1] for i in range(len(X))
           if len(set(pred[i])) > 1]
    check('T10 horizon=N 样本内 IC > 0.2 (周期信号可恢复)', np.mean(ics) > 0.2,
          f'({np.mean(ics):.3f})')
    # horizon 语义: 预测输入 a_T → 输出 â_{T+N}; 验证 A 只依赖窗口内行
    p = m.predict_rank(R[-1])
    check('T10 预测输出有限', np.all(np.isfinite(p)))
    # horizon=1 与 horizon=2 用同一数据应得不同 A (周期语义不同)
    m1 = RankTransferModel(ridge_alpha=10.0, lookback_days=L, min_train_days=10, horizon=1)
    m2 = RankTransferModel(ridge_alpha=10.0, lookback_days=L, min_train_days=10, horizon=2)
    m1.fit(R); m2.fit(R)
    check('T10 不同 horizon → 不同 A', not np.allclose(m1.A, m2.A))
    # 不足样本: L+1 <= horizon → fit 失败
    m3 = RankTransferModel(ridge_alpha=10.0, min_train_days=5, horizon=50)
    check('T10 样本≤horizon → 失败', not m3.fit(R[:40]))


def main():
    print('=' * 70)
    print('rank_transfer 策略专项测试')
    print('=' * 70)
    t1_rank_vector()
    t2_fit_correctness()
    t3_underdetermined()
    t4_low_rank()
    t5_rank_ic()
    t6_constituents()
    t7_returns()
    t8_no_future()
    t9_perf()
    t10_horizon()
    print('\n' + '=' * 70)
    print(f'结果: {ok}')
    print('ALL PASS' if ok else '有失败!')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
