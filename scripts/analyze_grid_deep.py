# -*- coding: utf-8 -*-
"""深度分析 GridLowVol 修复后完整回测 (20260822000509): 净值/分红贡献/订单/持仓/月度分解。"""
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

R = 'results/GridLowVol_20220801_20260522_20260822000509'
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(PROJ, R)

# ============ 1. 净值与分红贡献 ============
dv = pd.read_csv(f'{RD}/daily_values.csv', parse_dates=['date'])
dv = dv.sort_values('date').reset_index(drop=True)
print('===== 1. 净值/分红分解 =====')
print(f'区间: {dv.date.iloc[0].date()} ~ {dv.date.iloc[-1].date()}  ({len(dv)} 交易日)')
print(f'期末 total_value: {dv.total_value_eod.iloc[-1]:,.0f}  | 初始 100,000,000')
print(f'累计分红净收入: {dv.accumulated_dividend_net.iloc[-1]:,.0f} '
      f'({dv.accumulated_dividend_net.iloc[-1]/1e8*100:.2f}% 初始资金)')
# 无分红情形下的净值
no_div = dv.total_value_eod.iloc[0] + (dv.total_value_eod.iloc[-1] - dv.accumulated_dividend_net.iloc[-1])
print(f'扣除分红后的期末净值: {no_div:,.0f} → 收益 {(no_div/1e8-1)*100:+.2f}%')
print(f'分红贡献占总收益: {dv.accumulated_dividend_net.iloc[-1]/(dv.total_value_eod.iloc[-1]-1e8)*100:.0f}%')
# 持仓市值占比 (仓位)
dv['pos_ratio'] = dv['position_value'] / dv['total_value_eod']
print(f'平均仓位: {dv.pos_ratio.mean()*100:.1f}%  (中位 {dv.pos_ratio.median()*100:.1f}%, '
      f'空仓日 {int((dv.pos_ratio < 0.05).sum())}/{len(dv)})')
# 年度分解
dv['year'] = dv.date.dt.year
print('\n--- 年度收益分解 ---')
for y, g in dv.groupby('year'):
    r = (g.total_value_eod.iloc[-1] / g.total_value_eod.iloc[0] - 1) * 100
    div = g.dividend_income.sum()
    rng = g.total_value_eod.max() / g.total_value_eod.min() - 1
    print(f'{y}: 收益 {r:+.2f}% | 分红 {div:,.0f} | 年内高低差 {rng*100:.1f}% | 平均仓位 {g.pos_ratio.mean()*100:.0f}%')

# ============ 2. 订单结构 ============
ord = pd.read_csv(f'{RD}/orders.csv')
print(f'\n===== 2. 订单结构 (总 {len(ord)}) =====')
rc = ord['trade_reason'].value_counts()
for k, v in rc.items():
    print(f'  {k}: {v}')
# 成本
print(f'佣金 {ord.commission.sum():,.0f} + 印花税 {ord.tax.sum():,.0f} '
      f'= {ord.commission.sum()+ord.tax.sum():,.0f} ({ord.commission.sum()+ord.tax.sum():,} 元)')
# 网格买卖金额
grid = ord[ord['trade_reason'].isin(['grid_buy', 'grid_sell'])]
gb = grid[grid['trade_reason'] == 'grid_buy']
gs = grid[grid['trade_reason'] == 'grid_sell']
print(f'网格买 {len(gb)} 笔 {gb.amount.sum()/1e6:.1f}M | 网格卖 {len(gs)} 笔 {gs.amount.sum()/1e6:.1f}M')

# ============ 3. 持仓集中度 (期末 + 全程) ============
pos = pd.read_csv(f'{RD}/positions.csv')
print(f'\n===== 3. 持仓集中度 ({len(pos)} 行) =====')
print('持仓行业构成(按 code 前缀/名称近似):')
codes = pos['code']
print(f'  sh.6xx(沪主板): {(codes.str.startswith("sh.6")).sum()}, '
      f'sh.688(科创): {(codes.str.startswith("sh.688")).sum()}, '
      f'sz.000(深主板): {(codes.str.startswith("sz.000")).sum()}, '
      f'sz.002(中小): {(codes.str.startswith("sz.002")).sum()}, '
      f'sz.300(创业): {(codes.str.startswith("sz.300")).sum()}')

# ============ 4. backtest.log 关键统计 ============
log = open(f'{RD}/backtest.log', encoding='utf-8', errors='ignore').read()
print(f'\n===== 4. 行为统计 (backtest.log) =====')
grid_re = re.findall(r'网格: 卖(\d+)格/买(\d+)格', log)
sell_g = sum(int(a) for a, b in grid_re)
buy_g = sum(int(b) for a, b in grid_re)
print(f'网格触发行数: {len(grid_re)} | 卖格 {sell_g} vs 买格 {buy_g} '
      f'({buy_g/max(sell_g,1):.2f}:1)')
print(f'趋势闸门只卖不买(sellable=N): {log.count("sellable=N")}')
print(f'预警清仓(vol_alert_liquidate): {log.count("vol_alert_liquidate")}')
print(f'除权事件日志(含分红): {log.count("现金分红")+log.count("送股")}')
