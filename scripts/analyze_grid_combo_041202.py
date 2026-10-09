# -*- coding: utf-8 -*-
"""GridLowVol 041202 (P0+P1+空仓恢复, all 模式) 深度分析。

分析维度:
  1. 净值/回撤/仓位曲线 + 关键事件标注
  2. combo 风控事件时间线 (FLAT/HALF 触发、空仓时段、躲跌/踏空)
  3. 2024-07-10 最大回撤归因
  4. 网格买卖统计 / 收益分解 / 持仓集中度
  5. combo_half 语义缺陷影响量化
"""
import os
import pandas as pd
import numpy as np

BASE = r'E:\quant\gpsef_project\results\GridLowVol_20220801_20260522_20260822041202'
INIT = 1e8

dv = pd.read_csv(os.path.join(BASE, 'daily_values.csv'))
orders = pd.read_csv(os.path.join(BASE, 'orders.csv'))
pos = pd.read_csv(os.path.join(BASE, 'positions.csv'))

# ---------------- 1. 净值/回撤/仓位 ----------------
dv['nav'] = dv['total_value'] / INIT
dv['peak'] = dv['nav'].cummax()
dv['dd'] = dv['nav'] / dv['peak'] - 1.0
dv['position'] = dv['position_value'] / dv['total_value']
dv['date'] = pd.to_datetime(dv['date'])

print('=' * 78)
print('1. 总体绩效')
print('=' * 78)
final = dv['nav'].iloc[-1]
print(f"期末净值 {final:.4f}  总收益 {(final-1)*100:.2f}%")
print(f"平均仓位 {dv['position'].mean()*100:.1f}%  仓位>50% 天数 {(dv['position']>0.5).sum()}/{len(dv)}")
print(f"空仓(仓位<1%)天数 {(dv['position']<0.01).sum()}, 占比 {(dv['position']<0.01).mean()*100:.1f}%")
print(f"回撤>5% 天数 {(dv['dd']<-0.05).sum()}, >8% 天数 {(dv['dd']<-0.08).sum()}")

# 最大回撤时段
dd_min = dv.loc[dv['dd'].idxmin()]
print(f"最大回撤 {dd_min['dd']*100:.2f}% @ {dd_min['date'].date()}")
# 找回撤区间起点
peak_before = dv.loc[(dv['date'] <= dd_min['date']) & (dv['nav'] == dv['peak'])].iloc[-1]
print(f"  回撤起点 {peak_before['date'].date()} nav={peak_before['nav']:.4f}")

# 分年度
print('\n--- 分年度 ---')
dv['year'] = dv['date'].dt.year
for y, g in dv.groupby('year'):
    if len(g) < 2:
        continue
    nav0, nav1 = g['nav'].iloc[0], g['nav'].iloc[-1]
    mdd = (g['nav'] / g['nav'].cummax() - 1).min()
    print(f"  {y}: 收益 {(nav1/nav0-1)*100:+7.2f}%  最大回撤 {mdd*100:6.2f}%  平均仓位 {g['position'].mean()*100:5.1f}%")

# ---------------- 2. combo 事件时间线 ----------------
print('\n' + '=' * 78)
print('2. combo 风控事件时间线')
print('=' * 78)
combo = orders[orders['trade_reason'].str.startswith('combo', na=False)].copy()
combo['date'] = pd.to_datetime(combo['date'])
combo['amt'] = combo['value'].abs()

# 按日期聚合: 每日 combo 动作
daily_c = combo.groupby('date').agg(
    flat_sell=('amt', lambda s: s[combo.loc[s.index, 'trade_reason'] == 'combo_flat'].sum()),
    half_buy=('amt', lambda s: s[(combo.loc[s.index, 'trade_reason'] == 'combo_half') & (combo.loc[s.index, 'direction'] == 'buy')].sum()),
    half_sell=('amt', lambda s: s[(combo.loc[s.index, 'trade_reason'] == 'combo_half') & (combo.loc[s.index, 'direction'] == 'sell')].sum()),
    n_flat=('amt', lambda s: (combo.loc[s.index, 'trade_reason'] == 'combo_flat').sum()),
    n_half=('amt', lambda s: (combo.loc[s.index, 'trade_reason'] == 'combo_half').sum()),
).reset_index()

dv_i = dv.set_index('date')
events = []
for _, r in daily_c.iterrows():
    d = r['date']
    nav_t = dv_i.loc[d, 'nav'] if d in dv_i.index else np.nan
    events.append((d, r['n_half'], r['half_buy'], r['half_sell'], r['n_flat'], r['flat_sell'], nav_t))

print(f"{'日期':<12}{'HALF单数':>6}{'HALF买额(万)':>12}{'HALF卖额(万)':>12}{'FLAT单数':>8}{'FLAT卖额(万)':>12}{'当日净值':>10}")
for d, nh, hb, hs, nf, fs, nav in events:
    print(f"{d.date()!s:<12}{nh:>6}{hb/1e4:>12.0f}{hs/1e4:>12.0f}{nf:>8}{fs/1e4:>12.0f}{nav:>10.4f}")

# FLAT 空仓时段识别 (FLAT 清仓日 → 下一批建仓日)
flat_dates = sorted(daily_c[daily_c['n_flat'] > 0]['date'])
print(f'\nFLAT 清仓事件共 {len(flat_dates)} 次: {[d.date().isoformat() for d in flat_dates]}')

# 每次 FLAT 后的空仓期与躲跌/踏空
print('\n--- FLAT 空仓期分析 (清仓日 → 重新建仓日) ---')
all_orders = orders.copy()
all_orders['date'] = pd.to_datetime(all_orders['date'])
rebal = all_orders[all_orders['trade_reason'] == 'rebalance_base']
for i, fd in enumerate(flat_dates):
    # 下一次非 combo 买入 (rebalance_base buy = 恢复建仓)
    nxt = rebal[(rebal['date'] > fd) & (rebal['direction'] == 'buy')]
    if nxt.empty:
        continue
    rb = nxt['date'].iloc[0]
    # 空仓期 = fd 次日 → rb 前日 (若 fd 当日已清仓, 空仓从 fd 收盘后开始)
    hold_out = (dv_i.index > fd) & (dv_i.index < rb)
    if hold_out.sum() == 0:
        continue
    nav_in = dv_i.loc[dv_i.index <= fd, 'nav'].iloc[-1]
    nav_out = dv_i.loc[dv_i.index <= rb, 'nav'].iloc[-1]
    # 期间指数与策略净值变化 (策略空仓期间净值≈现金, 变化来自分红/成本)
    chg_strat = (nav_out / nav_in - 1) * 100
    bench = dv_i.loc[hold_out, 'benchmark_cumulative']
    chg_bench = (bench.iloc[-1] - bench.iloc[0]) if len(bench) else 0
    days = int(hold_out.sum())
    print(f"  {fd.date()} → {rb.date()} 空仓{days}日: 策略净值{chg_strat:+.2f}% vs 基准{chg_bench:+.1f}pp "
          f"(躲跌/踏空 {chg_bench - 0 if chg_bench < 0 else -chg_bench:+.1f}pp 净值口径)")

# ---------------- 3. 最大回撤归因 ----------------
print('\n' + '=' * 78)
print('3. 2024-07-10 最大回撤区间剖析 (-15.13%)')
print('=' * 78)
mdd_win = dv_i.loc[(dv_i.index >= pd.Timestamp('2024-05-01')) & (dv_i.index <= pd.Timestamp('2024-08-01'))]
if len(mdd_win):
    m = mdd_win
    print(f"  05-01→08-01: 净值 {m['nav'].iloc[0]:.4f} → {m['nav'].iloc[-1]:.4f} "
          f"({(m['nav'].iloc[-1]/m['nav'].iloc[0]-1)*100:+.1f}%)  区间最低 {m['nav'].min():.4f}")
    # 6-26 HALF buy 的持仓表现
    buy26 = orders[(orders['date'] == '2024-06-26') & (orders['trade_reason'] == 'combo_half')]
    print(f"  6-26 HALF 买入: {list(zip(buy26['code'], (buy26['value']/1e4).round(0)))} 万")
    for _, b in buy26.iterrows():
        c = b['code']
        ppos = pos[(pos['code'] == c) & (pos['date'] >= '2024-06-26') & (pos['date'] <= '2024-07-15')]
        if len(ppos):
            mv0, mv1 = ppos['market_value'].iloc[0], ppos['market_value'].iloc[-1]
            print(f"    {c}: 持仓市值 {mv0/1e4:.0f}万 → {mv1/1e4:.0f}万 ({(mv1/mv0-1)*100:+.1f}%)")

# ---------------- 4. 网格/收益分解 ----------------
print('\n' + '=' * 78)
print('4. 交易统计与收益分解')
print('=' * 78)
g = orders.copy()
g['amt'] = g['value'].abs()
buy = g[g['direction'] == 'buy']['amt'].sum()
sell = g[g['direction'] == 'sell']['amt'].sum()
print(f"总买入 {buy/1e8:.2f}亿  总卖出 {sell/1e8:.2f}亿  买卖比 {sell/buy:.2f}")
print(f"网格类(rebalance/grid)买卖比: ", end='')
gg = g[g['trade_reason'].isin(['rebalance_base', 'grid_buy', 'grid_sell', 'vol_alert_liquidate']) | g['trade_reason'].str.startswith('grid', na=False)]
gb = gg[gg['direction'] == 'buy']['amt'].sum()
gs = gg[gg['direction'] == 'sell']['amt'].sum()
print(f"{gs/gb:.2f} (买{gb/1e8:.2f}亿/卖{gs/1e8:.2f}亿)")
print(f"\n订单 trade_reason 分布:")
print(g.groupby('trade_reason').agg(n=('amt', 'size'), amt=('amt', 'sum')).sort_values('amt', ascending=False).head(10).to_string())

# 分红
div = dv['dividend_income'].sum()
print(f"\n累计分红 {div/1e4:.0f}万 ({div/INIT*100:.2f}%)")
cost = g['commission'].sum() + g['tax'].sum() + g['transfer_fee'].sum()
print(f"交易成本 佣金{g['commission'].sum()/1e4:.0f}万+税{g['tax'].sum()/1e4:.0f}万+过户 {cost/1e4:.0f}万合计 ({cost/INIT*100:.2f}%)")

# ---------------- 5. 持仓集中度 ----------------
print('\n' + '=' * 78)
print('5. 持仓集中度')
print('=' * 78)
pos['mv'] = pos['market_value']
top = pos.groupby('code')['mv'].agg(['sum', 'max']).sort_values('sum', ascending=False).head(12)
print('全程累计持仓市值 Top12 (近似持有时间/金额):')
print(top.to_string())
hold_days = pos.groupby('code').size().sort_values(ascending=False).head(10)
print('\n持有天数 Top10:')
print(hold_days.to_string())

# ---------------- 6. 价值暴露与行业 ----------------
print('\n' + '=' * 78)
print('6. 因子暴露轨迹 (2024 年后)')
print('=' * 78)
fe = pd.read_csv(os.path.join(BASE, 'attribution', 'factor_exposures.csv'))
fe['date'] = pd.to_datetime(fe['date'])
fe24 = fe[fe['date'] >= '2024-01-01']
if len(fe24):
    print(fe24[['date', 'size', 'value', 'volatility']].tail(6).to_string(index=False))
