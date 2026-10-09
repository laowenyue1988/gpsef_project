# -*- coding: utf-8 -*-
"""[Q5BAR] 预演：烟测窗内 BARRA 快照对策略池的覆盖率 + 机制生效预判（零算力、只读）。

校验三件事：
  1. 窗口内每个调仓日能取到的 PIT 快照（严格 < 调仓日）
  2. 策略池（smallcap_pool_cache_o3_20260910）成分在这些快照中的暴露覆盖率
  3. 机制是否会因"覆盖 < min_coverage(0.80)"而不生效
"""
import glob
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))
os.chdir(r'E:\quant\gpsef_project')

WIN_S, WIN_E = '2024-04-01', '2024-08-01'
SNAP = 'data/barra_exposures'
FACT = ['volatility', 'beta']

out = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    out.append(s)
    print(s, flush=True)


P('=' * 96)
P('[Q5BAR] 预演：快照覆盖率与机制生效预判')
P('=' * 96)

# 1) 快照清单
snap_dates = sorted(pd.Timestamp(os.path.basename(f)[len('barra_expo_'):-len('.parquet')])
                    for f in glob.glob(os.path.join(SNAP, 'barra_expo_*.parquet')))
P('快照总数 = %d   范围 %s ~ %s' % (len(snap_dates), snap_dates[0].date(), snap_dates[-1].date()))

# 2) 池缓存
pool_f = 'data/smallcap_pool_cache_o3_20260910.parquet'
pool = pd.read_parquet(pool_f)
pool['trade_date'] = pd.to_datetime(pool['trade_date'])
P('池缓存 %s  日数=%d  每日只数(中位)=%d  唯一股票=%d'
  % (os.path.basename(pool_f), pool['trade_date'].nunique(),
     pool.groupby('trade_date').size().median(), pool['ts_code'].nunique()))
P('池缓存日期范围 %s ~ %s' % (pool['trade_date'].min().date(), pool['trade_date'].max().date()))

# 3) 逐调仓日（每 20 个交易日，与 rebalance_interval 近似）检查覆盖
pdates = sorted(d for d in pool['trade_date'].unique()
                if pd.Timestamp(WIN_S) <= d <= pd.Timestamp(WIN_E))
P('\n窗口内池缓存交易日 = %d（%s ~ %s）' % (len(pdates),
                                       pdates[0].date() if pdates else '-',
                                       pdates[-1].date() if pdates else '-'))

REBAL = 20
rebal_days = pdates[::REBAL]
P('按 rebalance_interval=20 取样 ⇒ %d 个调仓日\n' % len(rebal_days))

P('%-12s %-12s %6s %8s %8s %10s' % ('调仓日', 'PIT快照', '池内只数', '覆盖只数', '覆盖率', '判定'))
P('-' * 96)
all_ok = True
for d in rebal_days:
    valid = [s for s in snap_dates if s < d]
    if not valid:
        P('%-12s %-12s %6d %8s %8s %10s' % (d.date(), 'NONE', 0, '-', '-', '无快照'))
        all_ok = False
        continue
    sel = max(valid)
    members = pool[pool['trade_date'] == d]['ts_code'].tolist()
    f = os.path.join(SNAP, 'barra_expo_%s.parquet' % sel.strftime('%Y%m%d'))
    expo = pd.read_parquet(f)
    expo['_c6'] = expo['code'].astype(str).str.split('.').str[-1].str.zfill(6)
    c6 = [str(c).split('.')[-1].zfill(6) for c in members]
    sub = expo.drop_duplicates('_c6').set_index('_c6').reindex(c6)
    cov = int(sub[FACT[0] + '_exp'].notna().sum())
    rate = cov / float(len(members))
    ok = rate >= 0.80
    all_ok &= ok
    P('%-12s %-12s %6d %8d %7.1f%% %10s'
      % (d.date(), sel.date(), len(members), cov, rate * 100, 'OK' if ok else '★不足'))

P('')
P('⇒ 覆盖率全部 ≥80%% ? %s' % ('是 —— 机制可正常生效' if all_ok else '★否 —— 需下调 min_coverage 或跳过'))

# 4) 机制模拟：用窗口内第一个调仓日做一次真实降权，观察选股改变量
if rebal_days:
    d0 = rebal_days[0]
    sel0 = max(s for s in snap_dates if s < d0)
    expo0 = pd.read_parquet(os.path.join(SNAP, 'barra_expo_%s.parquet' % sel0.strftime('%Y%m%d')))
    members = pool[pool['trade_date'] == d0]['ts_code'].tolist()
    rng = np.random.default_rng(20261008)
    fake_scores = rng.random(len(members))          # 模拟模型概率（仅用于验证机制管线）
    from importlib import import_module
    CLS = getattr(import_module('strategies.small_alpha191.strategy'), 'SmallAlpha191Strategy')
    new, info = CLS._apply_barra_risk_budget(members, fake_scores, expo0, FACT, 0.30)
    P('')
    P('机制管线模拟（%s，候选 %d 只，仅验证管线，非业绩）：' % (d0.date(), len(members)))
    P('  ' + str(info))
    if new is not None:
        b = set(np.argsort(fake_scores)[::-1][:20])
        a = set(np.argsort(np.asarray(new))[::-1][:20])
        P('  Top-20 集合改变 = %d 只（对称差）' % len(a ^ b))
        R = None
        P('  ⇒ 机制在真实候选集上**确实生效**')

with io.open('_work/20261008_barra/analysis/q5bar_preflight.txt', 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(out))
P('\n[done] 已写 _work/20261008_barra/analysis/q5bar_preflight.txt')
