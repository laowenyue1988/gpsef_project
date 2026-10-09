# -*- coding: utf-8 -*-
"""[Q5BAR-B] 预演：直接调用 `solve_barra_weights` 验证求解器可用（零算力、只读）。

验证项：
  1. 求解器返回 ok=True（不是静默降级）
  2. Σw ≈ total_exposure、max_w ≤ max_weight、HHI
  3. **权重转移**：BARRA 解 vs「按概率配权」的 L1 距离 + 与波动率暴露的秩相关
     （★这是针对 D-243 「HHI 盲区」的结构证据 —— 若秩相关显著为负，
       说明权重确实从高波动搬向低波动，而 HHI 可能看不见）
  4. 三档（V1/V2/V3 参数）均能出解
"""
import os
import sys
import time

import numpy as np
import pandas as pd

os.chdir(r'E:\quant\gpsef_project')
sys.path.insert(0, os.path.abspath('.'))

from strategies.wyckoff20.barra_constraint import solve_barra_weights  # noqa: E402

POOL = 'data/smallcap_pool_cache_o3_20260910.parquet'
SNAP = 'data/barra_exposures'
OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)


P('=' * 100)
P('[Q5BAR-B] 求解器预演  %s' % time.strftime('%F %T'))
P('=' * 100)

# ── 取一个调仓日 ─────────────────────────────────────────────
pool = pd.read_parquet(POOL)
pool['trade_date'] = pd.to_datetime(pool['trade_date'])
D0 = pd.Timestamp('2024-04-01')
codes = pool[pool['trade_date'] == D0]['ts_code'].tolist()
snap_f = os.path.join(SNAP, 'barra_expo_20240327.parquet')
expo = pd.read_parquet(snap_f)
P('[1] 调仓日 %s｜候选 %d 只｜快照 %s（%d 行）'
  % (D0.date(), len(codes), os.path.basename(snap_f), len(expo)))

rng = np.random.default_rng(20261008)
probs = rng.random(len(codes))
# ★关键修正：真实调用点 `codes = list(self.target_weights.keys())` ⇒ 只有**选中的 20 只**
#   （设计档 §2 原文：「优化器只在「已选中的那 20 只」内部配权」）
_top = np.argsort(probs)[::-1][:20]
codes = [codes[i] for i in _top]
probs = probs[_top]
w_prob = probs / probs.sum()                      # 「按概率配权」= 不做 BARRA 的对照

ARMS = {
    'V1 当年档':   dict(risk_aversion=0.5,  lambda_dev=0.2, lambda_l2=0.3, lambda_ind=0.3),
    'V2 风控主导': dict(risk_aversion=2.0,  lambda_dev=0.2, lambda_l2=1.0, lambda_ind=0.3),
    'V3 纯alpha端': dict(risk_aversion=0.05, lambda_dev=0.02, lambda_l2=0.05, lambda_ind=0.05),
}

# 波动率暴露（用于结构证据）
expo['_c6'] = expo['code'].astype(str).str.split('.').str[-1].str.zfill(6)
ex = expo.drop_duplicates('_c6').set_index('_c6')
c6 = [str(c).split('.')[-1].zfill(6) for c in codes]
vol_exp = ex.reindex(c6)['volatility_exp'].to_numpy(dtype=float)

P('')
P('[2] 逐档求解')
P('')
P('%-14s %6s %10s %10s %10s %12s %14s %14s'
  % ('档位', 'ok', 'Σw', 'max_w', 'HHI', 'L1(vs概率)', 'ρ(Δw,波动率)', '耗时s'))
P('-' * 104)

sols = {}
for name, kw in ARMS.items():
    t0 = time.time()
    try:
        w, ok = solve_barra_weights(
            preds=probs, selected_codes=codes, barra_expo=expo,
            max_weight=0.10, constraint_date=D0, total_exposure=1.0,
            baseline='sh932000', expected_ic=0.05, alpha_scale=10.0, **kw)
        dt = time.time() - t0
        if w is None or not ok:
            P('%-14s %6s  ← 求解失败/降级' % (name, str(ok)))
            continue
        w = np.asarray(w, dtype=float)
        dw = w - w_prob
        l1 = float(np.abs(dw).sum())
        m = np.isfinite(vol_exp)
        if m.sum() > 5:
            rho = float(pd.Series(dw[m]).corr(pd.Series(vol_exp[m]), method='spearman'))
        else:
            rho = float('nan')
        sols[name] = w
        P('%-14s %6s %10.6f %10.6f %10.5f %12.6f %14.4f %14.2f'
          % (name, str(ok), w.sum(), w.max(), float((w ** 2).sum()), l1, rho, dt))
    except Exception as e:                                  # noqa: BLE001
        P('%-14s ★异常: %s: %s' % (name, type(e).__name__, str(e)[:70]))

P('')
P('[3] 结构证据解读（针对 D-243 「HHI 盲区」）')
P('    · L1(vs概率) > 0 且 ρ(Δw, 波动率) < 0 ⇒ 权重确实从高波动搬向低波动')
P('      ⇒ 这是 HHI 看不见、但风险特征会变的情形 ⇒ 支持"判据错配"判断')
P('    · 若两两 HHI 接近但 L1 距离大 ⇒ **直接证明"HHI 相同 ≠ 解相同"**')

if len(sols) >= 2:
    P('')
    P('    档位两两对比：')
    ks = list(sols)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = sols[ks[i]], sols[ks[j]]
            l1 = float(np.abs(a - b).sum())
            dH = abs(float((a ** 2).sum() - (b ** 2).sum()))
            P('      %-14s vs %-14s  L1=%.4f  |ΔHHI|=%.6f  (HHI: %.5f / %.5f)'
              % (ks[i], ks[j], l1, dH, float((a ** 2).sum()), float((b ** 2).sum())))

with open('_work/20261008_barra/analysis/solver_preflight.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(OUT))
P('')
P('[done] _work/20261008_barra/analysis/solver_preflight.txt')
