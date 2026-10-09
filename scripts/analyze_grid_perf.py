# -*- coding: utf-8 -*-
"""分析 grid_lowvol 两次失败回测: 从 backtest.log 重建净值序列(AUDIT total 按交易日归类)+行为统计。"""
import os
import re
from collections import Counter, defaultdict

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUNS = [
    ('candidate_pool=200', 'results/GridLowVol_20220801_20260522_20260821194255'),
    ('candidate_pool=2000', 'results/GridLowVol_20220801_20260522_20260821200421'),
]


def analyze(run_name, rel):
    path = os.path.join(PROJ, rel, 'backtest.log')
    lines = open(path, encoding='utf-8', errors='ignore').read().splitlines()

    cur_date = None
    net_by_date = {}          # date -> last AUDIT total
    grid_buy = Counter()      # date -> 买格数
    grid_sell = Counter()
    alert_dates = Counter()   # date -> 预警清仓数
    pool_hist = []
    n_orders = 0
    n_grid_days = 0

    for ln in lines:
        # 1) AUDIT 行 (归入当前交易日)
        m = re.search(r'\[BROKER AUDIT\] (BUY|SELL) .*? total=([\d,]+)', ln)
        if m and cur_date:
            net_by_date[cur_date] = float(m.group(2).replace(',', ''))
            n_orders += 1
        # 2) 网格行 (自带日期)
        m = re.search(r'\[grid_lowvol\] (\d{4}-\d{2}-\d{2}) \S+ 网格: 卖(\d+)格/买(\d+)格', ln)
        if m:
            d = m.group(1)
            grid_sell[d] += int(m.group(2))
            grid_buy[d] += int(m.group(3))
            n_grid_days += 1
        # 3) 日期标记行 (目标池/预警)
        m = re.search(r'\[grid_lowvol\] (\d{4}-\d{2}-\d{2}) 目标池top(\d+): (.*?)\.\.\. 预警清仓(\d+)', ln)
        if m:
            cur_date = m.group(1)
            alert_dates[cur_date] += int(m.group(4))
            pool_hist.append(ln)
            continue
        m = re.search(r'\[grid_lowvol\] (\d{4}-\d{2}-\d{2})', ln)
        if m:
            cur_date = m.group(1)

    print(f'\n===== {run_name} =====')
    dates = sorted(net_by_date)
    if dates:
        start_v, end_v = net_by_date[dates[0]], net_by_date[dates[-1]]
        print(f'净值(重建): {dates[0]} {start_v:,.0f} → {dates[-1]} {end_v:,.0f} '
              f'= {(end_v/start_v-1)*100:+.2f}%  ({len(dates)} 个有交易的交易日)')
        # 每 3 个月一个观察点
        by_q = {}
        for d in dates:
            by_q[d[:7]] = net_by_date[d]
        prev = None
        for k, v in sorted(by_q.items()):
            chg = f'{(v/prev-1)*100:+.2f}%' if prev else '—'
            print(f'  {k}: {v:,.0f} ({chg})')
            prev = v
        peak = start_v
        mdd = 0.0
        for d in dates:
            peak = max(peak, net_by_date[d])
            mdd = min(mdd, net_by_date[d] / peak - 1)
        print(f'最大回撤(重建): {mdd*100:.2f}%')
    else:
        print('无 AUDIT 行')

    total_buy = sum(grid_buy.values())
    total_sell = sum(grid_sell.values())
    total_alert = sum(alert_dates.values())
    print(f'订单数={n_orders} | 网格触发日={n_grid_days} | 网格买格={total_buy} vs 卖格={total_sell} '
          f'(买:卖={total_buy/max(total_sell,1):.2f}:1) | 预警清仓总股数={total_alert}')
    # 预警分布
    if alert_dates:
        print(f'预警清仓触发日数={len(alert_dates)} (日均股数={total_alert/max(len(alert_dates),1):.1f})')
    # 目标池构成
    codes = Counter()
    for ln in pool_hist:
        m = re.search(r'目标池top\d+: (.*?)\.\.\.', ln)
        if m:
            for c in m.group(1).split(','):
                c = c.strip().strip("[]'")
                if c:
                    codes[c] += 1
    print(f'目标池出现最多 12 只 (共 {len(codes)} 只):')
    for c, n in codes.most_common(12):
        print(f'  {c}: {n} 次')


for name, rel in RUNS:
    analyze(name, rel)
