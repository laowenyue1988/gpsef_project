# -*- coding: utf-8 -*-
"""[Q5BAR] 烟测分析器：三臂（Z1/B1/S1）＋ 与 D-342 的 A 臂逐字节对照。

判据（跑完后运行）：
  A. ★零影响闭合：Z1 的实质产物 md5 == D-342 A 臂（`results/EQ_final_small_alpha191_20261008`）
  B. R-126 四硬：ERROR=0 / 缺列=0 / 训练失败=0 / 见 meta_IC / `调仓完成，持仓数`>0
  C. 机制自证：B1/S1 的 `[Q5BAR] 风险预算降权生效:` 行数（应 ≈ 调仓次数）
  D. R-153 选股集合改变量：`orders.csv` 中 `direction=buy` 且 `trade_reason=new_open`
     的 per-date 集合，B1/S1 vs Z1 对称差
  E. 烟测窗业绩（仅参考，**不作业绩判据** —— MDE 过大）
"""
import hashlib
import io
import os
import re
import sys

import numpy as np
import pandas as pd

os.chdir(r'E:\quant\gpsef_project')
WD = '_work/20261008_barra'
ARMS = {'Z1': 'results/Q5BAR_Z1_20261008',
        'B1': 'results/Q5BAR_B1_20261008',
        'S1': 'results/Q5BAR_S1_20261008'}
A_ARM = 'results/EQ_final_small_alpha191_20261008'
FILES = ['daily_values.csv', 'orders.csv', 'positions.csv', 'dividends.csv',
         'report.json', 'report.txt', 'performance.png']
OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)


def md5(p):
    if not os.path.exists(p):
        return 'MISSING'
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def metrics(tv):
    tv = pd.Series(tv).astype(float).reset_index(drop=True)
    r = tv.pct_change().fillna(0.0)
    r.iloc[0] = 0.0
    n = len(r)
    cum = (1 + r).cumprod()
    mdd = abs((cum / cum.cummax() - 1).min())
    vol = r.std(ddof=1) * np.sqrt(252)
    ann = (tv.iloc[-1] / tv.iloc[0]) ** (252.0 / n) - 1
    return ann, mdd, (ann / vol if vol > 0 else float('nan')), n


def new_open_sets(p):
    d = pd.read_csv(p, encoding='utf-8-sig')
    m = (d['direction'] == 'buy') & (d['trade_reason'] == 'new_open')
    return d[m].groupby('date')['code'].apply(set)


P('=' * 100)
P('[Q5BAR] 烟测分析  %s' % pd.Timestamp.now().strftime('%F %T'))
P('=' * 100)

# ── A. 零影响闭合 ────────────────────────────────────────────
P('')
P('【A】★零影响闭合：Z1（新代码 mode=off）vs D-342 A 臂（改造前代码）')
P('     A 臂目录 = %s' % A_ARM)
P('')
P('%-20s %-14s %-14s %s' % ('产物', 'A 臂', 'Z1', '判定'))
P('-' * 78)
zero_ok = True
z1_missing = []
for f in FILES:
    ha, hz = md5(os.path.join(A_ARM, f)), md5(os.path.join(ARMS['Z1'], f))
    if ha == 'MISSING' or hz == 'MISSING':
        st = '⚠ 缺文件'
        z1_missing.append(f)
    else:
        st = '✅ 同' if ha == hz else '❌ 异'
        zero_ok &= (ha == hz)
    P('%-20s %-14s %-14s %s' % (f, ha[:12], hz[:12], st))
P('')
P('    ⇒ 零影响闭合 = %s' % ('✅ PASS（逐字节相同）' if zero_ok else '❌ FAIL（见下方差异定位）'))
if z1_missing:
    P('    ⚠ 缺文件: %s' % z1_missing)

# 若不同 ⇒ 逐行定位
if not zero_ok:
    P('')
    P('    --- 差异定位（仅 Z1 侧）---')
    for f in ('daily_values.csv', 'orders.csv', 'positions.csv'):
        pa, pz = os.path.join(A_ARM, f), os.path.join(ARMS['Z1'], f)
        if not (os.path.exists(pa) and os.path.exists(pz)):
            continue
        la = io.open(pa, encoding='utf-8-sig', errors='ignore').read().splitlines()
        lz = io.open(pz, encoding='utf-8-sig', errors='ignore').read().splitlines()
        P('    %-20s A=%d 行  Z1=%d 行' % (f, len(la), len(lz)))
        n = min(len(la), len(lz))
        dif = [i for i in range(n) if la[i] != lz[i]]
        P('        首个不同行号 = %s（共 %d 行不同）'
          % (dif[0] + 1 if dif else '无', len(dif)))
        for i in dif[:3]:
            P('          L%d A : %s' % (i + 1, la[i][:130]))
            P('          L%d Z1: %s' % (i + 1, lz[i][:130]))

# ── 日志归一化比对（Z1 vs A） ───────────────────────────────
P('')
P('【A2】backtest.log 归一化比对（Z1 vs A 臂）')


def norm_log(p):
    if not os.path.exists(p):
        return []
    t = io.open(p, encoding='utf-8', errors='ignore').read()
    t = re.sub(r'^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}:', '', t, flags=re.M)
    t = t.replace('EQ_final_small_alpha191_20261008', '<OUT>')
    t = t.replace('Q5BAR_Z1_20261008', '<OUT>')
    t = re.sub(r'\d+\.\d+s', '<T>', t)
    t = re.sub(r'耗时 [\d.]+', '耗时 <T>', t)
    return sorted(t.splitlines())


na, nz = norm_log(os.path.join(A_ARM, 'backtest.log')), norm_log(os.path.join(ARMS['Z1'], 'backtest.log'))
onlya = set(na) - set(nz)
onlyz = set(nz) - set(na)
P('    A 行数=%d  Z1 行数=%d  仅A有=%d  仅Z1有=%d' % (len(na), len(nz), len(onlya), len(onlyz)))
for l in sorted(onlya)[:4]:
    P('      A only : %s' % l[:130])
for l in sorted(onlyz)[:4]:
    P('      Z1 only: %s' % l[:130])

# ── B/C. 四硬 + 机制自证 ─────────────────────────────────────
P('')
P('【B/C】R-126 四硬 + Q5BAR 机制自证（逐臂）')
P('')
P('%-4s %8s %8s %10s %10s %12s %14s' %
  ('臂', 'ERROR', '缺列', '训练失败', 'meta_IC', '调仓完成(最大)', 'Q5BAR生效行'))
P('-' * 84)
for k, d in ARMS.items():
    lg = os.path.join(WD, 'logs', 'arm_%s.log' % k)
    if not os.path.exists(lg):
        P('%-4s  (日志缺失)' % k)
        continue
    t = io.open(lg, encoding='utf-8', errors='ignore').read()
    err = len(re.findall(r'\bERROR\b', t))
    miss = t.count('缺列')
    tfail = t.count('训练失败')
    ic = len(re.findall(r'meta_IC', t))
    holds = [int(x) for x in re.findall(r'调仓完成，持仓数[:：]\s*(\d+)', t)]
    rb = len(re.findall(r'\[Q5BAR\] 风险预算降权生效', t))
    rbskip = len(re.findall(r'\[Q5BAR\] 风险预算降权未生效', t))
    P('%-4s %8d %8d %10d %10d %12s %14d' %
      (k, err, miss, tfail, ic, max(holds) if holds else '-', rb))
    if rbskip:
        P('      ⚠ 未生效行 = %d' % rbskip)

# ── D. 选股集合改变量（R-153） ─────────────────────────────
P('')
P('【D】R-153 选股集合改变量（orders.csv: direction=buy ∧ trade_reason=new_open）')
try:
    sZ = new_open_sets(os.path.join(ARMS['Z1'], 'orders.csv'))
    P('    Z1 调仓日数 = %d  总 new_open = %d' % (len(sZ), sum(len(v) for v in sZ)))
    for k in ('B1', 'S1'):
        pk = os.path.join(ARMS[k], 'orders.csv')
        if not os.path.exists(pk):
            P('    %s: 产物缺失' % k)
            continue
        sk = new_open_sets(pk)
        days = sorted(set(sZ.index) & set(sk.index))   # ★修：Series 的 set() 取的是值（set of set ⇒ unhashable），须用 .index
        tot_sym = 0
        tot_n = 0
        P('')
        P('    %s vs Z1（共同调仓日 %d）:' % (k, len(days)))
        for dd in days:
            a, b = sZ[dd], sk[dd]
            sym = len(a ^ b)
            tot_sym += sym
            tot_n += len(a | b)
            P('      %s  对称差=%2d  Z1独有=%2d  %s独有=%2d' %
              (dd, sym, len(a - b), k, len(b - a)))
        P('      ⇒ 累计对称差 = %d / %d（%.1f%%）' % (tot_sym, tot_n, 100.0 * tot_sym / max(tot_n, 1)))
except Exception as e:
    P('    选股集合比对失败: %r' % (e,))

# ── E. 烟测窗业绩（仅参考） ─────────────────────────────────
P('')
P('【E】烟测窗业绩（**仅参考，不作业绩判据**；窗口 2024-04-01~2024-08-01）')
P('')
P('%-4s %10s %10s %10s %8s' % ('臂', '年化%', '|MaxDD|%', '★Sharpe', 'n'))
P('-' * 50)
for k in ('Z1', 'B1', 'S1'):
    p = os.path.join(ARMS[k], 'daily_values.csv')
    if not os.path.exists(p):
        P('%-4s  (缺 daily_values.csv)' % k)
        continue
    d = pd.read_csv(p, encoding='utf-8-sig')
    col = 'total_value' if 'total_value' in d.columns else d.columns[-1]
    ann, mdd, shp, n = metrics(d[col])
    P('%-4s %10.4f %10.2f %10.4f %8d' % (k, ann * 100, mdd * 100, shp, n))
# 同时打印 A 臂（对照）
p = os.path.join(A_ARM, 'daily_values.csv')
if os.path.exists(p):
    d = pd.read_csv(p, encoding='utf-8-sig')
    col = 'total_value' if 'total_value' in d.columns else d.columns[-1]
    ann, mdd, shp, n = metrics(d[col])
    P('%-4s %10.4f %10.2f %10.4f %8d  ← D-342 A 臂' % ('A', ann * 100, mdd * 100, shp, n))

with io.open('%s/analysis/smoke_verdict.txt' % WD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(OUT))
P('')
P('[done] %s/analysis/smoke_verdict.txt' % WD)
