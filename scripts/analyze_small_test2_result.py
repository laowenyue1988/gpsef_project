# -*- coding: utf-8 -*-
"""
SmallAlpha191Test2 格栅结果分析脚本
=====================================
输入: 回测结果目录 (含 report.txt / daily_values.csv / orders.csv / attribution/ / backtest.log)
输出: 指标汇总表 + 方案A验证点检查 (与手册 §4.2A 对应)
用法: .venv/Scripts/python.exe scripts/analyze_small_test2_result.py --dir results/xxx --label R0
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd


def _read_text(p):
    """自动检测编码 (框架 report.txt 为 GBK, 其他文件多为 UTF-8)。"""
    for enc in ('utf-8', 'gb18030', 'gbk'):
        try:
            with open(p, encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(p, encoding='utf-8', errors='ignore') as f:
        return f.read()


def load_report(d):
    p = os.path.join(d, 'report.txt')
    out = {}
    if os.path.exists(p):
        for line in _read_text(p).splitlines():
            m = re.match(r'^([^:：]+)[:：]\s*(-?[\d.]+)%?', line.strip())
            if m:
                try:
                    out[m.group(1).strip()] = float(m.group(2))
                except ValueError:
                    pass
    return out


def analyze(d, label):
    print(f'\n{"="*70}\n[{label}] 结果目录: {d}\n{"="*70}')
    rep = load_report(d)
    if rep:
        keys = ['总收益率', '总收益', '年化收益率', '年化收益', '年化波动率',
                '最大回撤', '夏普比率', 'Sharpe', '日胜率', '胜率', '盈亏比', '换手率']
        for k in keys:
            if k in rep:
                print(f'  {k}: {rep[k]:.2f}')
    else:
        print('  ⚠️ 无 report.txt')

    # ---- daily_values ----
    dv_p = os.path.join(d, 'daily_values.csv')
    if os.path.exists(dv_p):
        dv = pd.read_csv(dv_p)
        dv['date'] = pd.to_datetime(dv['date'])
        nav_col = 'nav' if 'nav' in dv.columns else 'total_value'
        pos_col = 'position' if 'position' in dv.columns else 'position_value'
        if 'daily_return' in dv.columns:
            dv['ret'] = dv['daily_return']
        else:
            dv['ret'] = dv[nav_col].pct_change().fillna(0)
        dv['nav'] = dv[nav_col] / dv[nav_col].iloc[0]
        dv['dd'] = dv['nav'] / dv['nav'].cummax() - 1
        if pos_col in dv.columns and nav_col in dv.columns:
            dv['pos'] = dv[pos_col] / dv[nav_col]
        else:
            dv['pos'] = 0.0
        print(f'  区间: {dv["date"].iloc[0].date()} ~ {dv["date"].iloc[-1].date()} ({len(dv)} 交易日)')
        print(f'  总收益: {(dv["nav"].iloc[-1]-1)*100:+.2f}%  最大回撤: {dv["dd"].min()*100:.2f}% @ {dv.loc[dv["dd"].idxmin(),"date"].date()}')
        print(f'  平均仓位: {dv["pos"].mean()*100:.1f}%  空仓天数(pos<5%): {(dv["pos"]<0.05).sum()}')
        # 分年度
        dv['year'] = dv['date'].dt.year
        for y, g in dv.groupby('year'):
            if len(g) < 2:
                continue
            print(f'    {y}: 收益 {(g["nav"].iloc[-1]/g["nav"].iloc[0]-1)*100:+7.2f}% 回撤 {g["dd"].min()*100:6.2f}% 仓位 {g["pos"].mean()*100:5.1f}%')
        # 2024-02 危机窗口 (红绿灯)
        crisis = dv[(dv['date'] >= '2024-01-15') & (dv['date'] <= '2024-02-29')]
        if len(crisis):
            base = dv[dv['date'] < '2024-01-15']['nav'].iloc[-1] if len(dv[dv['date'] < '2024-01-15']) else 1
            crisis_dd = (crisis['nav'] / crisis['nav'].cummax() - 1).min() if len(crisis) else 0
            print(f'  [红绿灯] 2024-01-15~02-29 危机窗: 组合回撤 {crisis_dd*100:.2f}% (安全线 >-8%, 143844同期约-0.4%)')
        # 2024-01-22 调仓日趋势过滤检查点
        jan22 = dv[(dv['date'] >= '2024-01-19') & (dv['date'] <= '2024-01-24')]
        if len(jan22):
            print(f'  2024-01-22 前后净值: {jan22["nav"].iloc[0]:.4f} → {jan22["nav"].iloc[-1]:.4f} '
                  f'({(jan22["nav"].iloc[-1]/jan22["nav"].iloc[0]-1)*100:+.2f}%)')
    else:
        print('  ⚠️ 无 daily_values.csv')

    # ---- orders ----
    o_p = os.path.join(d, 'orders.csv')
    if os.path.exists(o_p):
        o = pd.read_csv(o_p)
        reason_col = 'trade_reason' if 'trade_reason' in o.columns else 'reason'
        print(f'  订单数: {len(o)}')
        if reason_col in o.columns:
            s1 = o[o[reason_col].astype(str).str.contains('stage1|Stage1|S1', na=False)]
            print(f'    S1 阶段触发订单: {len(s1)} (143844 约 1746 次)')
            flat = o[o[reason_col].astype(str).str.contains('combo_flat|dd_flat', na=False)]
            half = o[o[reason_col].astype(str).str.contains('combo_half|dd_half', na=False)]
            alert = o[o[reason_col].astype(str).str.contains('vol_alert', na=False)]
            print(f'    熔断订单: FLAT {len(flat)} 笔, HALF {len(half)} 笔; 预警清仓 {len(alert)} 笔')
    else:
        print('  ⚠️ 无 orders.csv')

    # ---- backtest.log 熔断触发统计 ----
    log_p = os.path.join(d, 'backtest.log')
    if os.path.exists(log_p):
        txt = open(log_p, encoding='utf-8', errors='ignore').read()
        flat_n = len(re.findall(r'回撤熔断.*?→ FLAT', txt))
        half_n = len(re.findall(r'回撤熔断.*?→ HALF', txt))
        rec_n = len(re.findall(r'回撤熔断.*?→ NORMAL', txt))
        trend_n = len(re.findall(r'趋势过滤.*?建仓', txt))
        print(f'  [日志] 熔断状态迁移: FLAT {flat_n} 次, HALF {half_n} 次, 恢复NORMAL {rec_n} 次; 趋势过滤触发 {trend_n} 次')
    return rep


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='回测结果目录')
    ap.add_argument('--label', default='R?', help='格栅编号')
    args = ap.parse_args()
    analyze(args.dir, args.label)
