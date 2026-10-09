# -*- coding: utf-8 -*-
"""
SmallAlpha191Test2 格栅实验最终分析器
=====================================
- 扫描 results/ 下所有 SmallAlpha191Test2_20220801_20240301_* 目录
- 按 config_full.json 参数指纹识别组 (R0-R12 / T1-T2), 同指纹取时间戳最新(重跑覆盖首跑)
- 生成 results/grid_summary.csv (全组指标总表) + results/FINAL_CONCLUSION.md (结论初稿)
用法: .venv/Scripts/python.exe scripts/final_analyze.py
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJ)


def read_cfg(d):
    try:
        return json.load(open(os.path.join(d, 'config_full.json'), encoding='utf-8'))
    except Exception:
        return {}


def fingerprint(cfg):
    """参数指纹 → 组名。"""
    dd = cfg.get('drawdown_breaker', {})
    tg = cfg.get('rebalance_trend_gate', {})
    se = cfg.get('staged_exit', {})
    s1 = se.get('stage1', {})
    auc = cfg.get('auc_threshold')
    dd_half, dd_flat = dd.get('dd_half'), dd.get('dd_flat')
    confirm = dd.get('trend_confirm_days')
    dd_ma = dd.get('index_ma')
    tg_ma = tg.get('ma')
    pt = s1.get('pt_sigma')
    if tg_ma == 10 and tg_ma != dd_ma:
        return 'T1'
    if tg_ma == 30 and tg_ma != dd_ma:
        return 'T2'
    if confirm == 2 and dd_ma == 10 and pt == 3.0:
        return 'R11'
    if confirm == 2 and dd_ma == 10 and pt == 2.2:
        return 'R12'
    if auc == 0.52:
        return 'R9'
    if auc == 0.56:
        return 'R10'
    if pt == 2.2:
        return 'R7'
    if pt == 3.0:
        return 'R8'
    if dd_half == 0.12:
        return 'R2'
    if dd_half == 0.08:
        return 'R1'
    if dd_half == 0.10:
        return 'R0'
    return 'UNKNOWN'


def calc(d):
    """从 daily_values 计算指标 (首日归一, 与 analyze_small_test2_result 同口径)。"""
    try:
        dv = pd.read_csv(os.path.join(d, 'daily_values.csv'))
    except Exception:
        return None
    dv['date'] = pd.to_datetime(dv['date'])
    dv = dv.sort_values('date')
    tv = dv['total_value']
    ret = tv.pct_change().fillna(0)
    nav = tv / tv.iloc[0]
    n = len(dv)
    ann = nav.iloc[-1] ** (252 / n) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else 0
    mdd = (nav / nav.cummax() - 1).min()
    pos = (dv['position_value'] / tv).mean()
    empty = int((dv['position_value'] / tv < 0.05).sum())
    # report.txt (GBK)
    rep = {}
    rp = os.path.join(d, 'report.txt')
    if os.path.exists(rp):
        for enc in ('utf-8', 'gb18030'):
            try:
                txt = open(rp, encoding=enc).read()
                break
            except UnicodeDecodeError:
                continue
        else:
            txt = open(rp, encoding='utf-8', errors='ignore').read()
        for line in txt.splitlines():
            m = re.match(r'^([^:：]+)[:：]\s*(-?[\d.]+)%?', line.strip())
            if m:
                try:
                    rep[m.group(1).strip()] = float(m.group(2))
                except ValueError:
                    pass
    return dict(
        ret=(nav.iloc[-1] - 1) * 100, ann=ann * 100, vol=vol * 100,
        sharpe=sharpe, mdd=mdd * 100, pos=pos * 100, empty_days=empty,
        rep_ret=rep.get('总收益率'), rep_sharpe=rep.get('夏普比率'),
        rep_mdd=rep.get('最大回撤'), orders=rep.get('总订单数'))


def main():
    dirs = sorted(glob.glob('results/SmallAlpha191Test2_20220801_20240301_*'),
                  key=os.path.getmtime)
    rows = {}
    for d in dirs:
        cfg = read_cfg(d)
        g = fingerprint(cfg)
        if g == 'UNKNOWN':
            continue
        m = calc(d)
        if m is None:
            continue
        # 同组取最新 (重跑覆盖首跑)
        if g not in rows or os.path.getmtime(d) > rows[g]['mtime']:
            m['group'] = g
            m['dir'] = os.path.basename(d)
            m['mtime'] = os.path.getmtime(d)
            rows[g] = m
    if not rows:
        print('未找到有效结果目录')
        return
    df = pd.DataFrame(list(rows.values())).sort_values('group')
    df = df.drop(columns=['mtime'])
    df.to_csv('results/grid_summary.csv', index=False, encoding='utf-8-sig')
    print(df.to_string(index=False))
    # 结论初稿
    best = df.loc[df['sharpe'].idxmax()]
    lines = [
        '# SmallAlpha191Test2 格栅实验结论（初稿，待人工复核）',
        '',
        f'生成时间: {pd.Timestamp.now()}',
        '',
        '## 全组指标总表（同期 2022-08-01 ~ 2024-03-01）',
        '',
        df.to_markdown(index=False),
        '',
        '## 初步结论（机械生成）',
        '',
        f'- 最优 Sharpe 组: **{best["group"]}** (Sharpe {best["sharpe"]:.2f}, 收益 {best["ret"]:+.2f}%, 回撤 {best["mdd"]:.2f}%)',
        f'- 最低回撤组: **{df.loc[df["mdd"].idxmax(), "group"]}** ({df["mdd"].max():.2f}%)',
        f'- 最高收益组: **{df.loc[df["ret"].idxmax(), "group"]}** ({df["ret"].max():+.2f}%)',
        f'- 最高仓位组: **{df.loc[df["pos"].idxmax(), "group"]}** ({df["pos"].max():.1f}%)',
        '',
        '> 注: 本文件为自动生成初稿; 统计检验(ab_sig_test/Holm)与最终裁决由主代理完成。',
        '',
    ]
    with open('results/FINAL_CONCLUSION.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\nFINAL_CONCLUSION.md 已生成')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        with open(os.path.join('results', 'final_analyze_error.log'), 'a', encoding='utf-8') as f:
            f.write(f'[{pd.Timestamp.now()}] {e}\n{traceback.format_exc()}\n')
        print(f'FATAL: {e}', flush=True)
        raise
