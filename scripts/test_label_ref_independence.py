# -*- coding: utf-8 -*-
"""验证: 标签计算与全局复权基准日解耦 (2026-08-30 修复)
同一事件集下 reference_date=None(修复后) 的输出应与任意全局 ref 完全一致。
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from datetime import datetime

from strategies.small_alpha191_test2.labels import TripleBarrierLabeler
from gpsef.data.local_provider import LocalDataProvider

# ── 日历: 2021-07 ~ 2023-12 交易日 (训练窗口 + 持有期) ──
cal = pd.bdate_range('2021-07-01', '2023-12-31')

# ── staged_cfg (与 config_R12 的 staged_exit 一致: S1 3.0σ) ──
staged_cfg = {
    1: {'pt': 3.0, 'sl': -3.0, 'pct': 0.25, 'max_hold': 5},
    2: {'pt': 3.0, 'sl': -3.0, 'pct': 0.35, 'max_hold': 10},
    3: {'pt': 4.0, 'sl': -4.0, 'pct': 0.20, 'max_hold': 15},
    4: {'pt': 9.0, 'sl': -4.0, 'pct': 0.20, 'max_hold': 20},
}

dp = LocalDataProvider('data')
labeler = TripleBarrierLabeler(
    dp, pt_mult=2.0, sl_mult=1.5, max_hold_days=20, minute_freq='5',
    daily_vol_window=20, vol_method='ewma', vol_ewma_lambda=0.94,
    vol_ewma_seed_days=60, staged_cfg=staged_cfg, hard_sl=-6.0)

# ── 事件集: 8 只股票 × 30 个事件日 (2022-08 前, 含除权股) ──
codes = ['sh.600000', 'sz.002494', 'sh.600004', 'sz.300640',
         'sz.002629', 'sh.605488', 'sz.002719', 'sh.688182']
event_dates = list(pd.bdate_range('2022-01-04', '2022-04-29'))[:30]
events = []
for code in codes:
    for d in event_dates:
        events.append({'trade_date': pd.Timestamp(d), 'ts_code': code, 'entry_price': 1.0})
events_df = pd.DataFrame(events).set_index('trade_date')

results = {}
for tag, ref in [('FIX(None)', None), ('REF2024', datetime(2024, 3, 1)), ('REF2026', datetime(2026, 8, 4))]:
    lab = labeler.compute_labels_staged(events_df, cal, reference_date=ref)
    binv = np.sign(lab['realized_ret'].fillna(0).values)
    dist = {0: int((binv <= 0).sum()), 1: int((binv > 0).sum())}
    nan_n = int(lab['realized_ret'].isna().sum())
    results[tag] = dist
    print(f'{tag}: 标签分布 {dist}  NaN={nan_n}  (共 {len(lab)})', flush=True)

base = results['FIX(None)']
for tag, dist in results.items():
    same = dist == base
    print(f'{"✅" if same else "❌"} {tag} vs FIX(None): {"一致" if same else f"不一致 (Δ{dist[0]-base[0]})"}', flush=True)
