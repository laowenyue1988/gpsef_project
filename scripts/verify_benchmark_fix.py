# -*- coding: utf-8 -*-
"""验证基准数据修正: ①日志确认走 index_daily 路径 ②benchmark_cumulative 与 index_daily/sh000300 独立核算一致"""
import glob
import os
import re
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
LOG = 'results/smoke2.log'


def wait_done(timeout=1500):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(LOG):
            txt = open(LOG, encoding='utf-8', errors='ignore').read()
            if '绩效报告已保存' in txt:
                return True
        time.sleep(30)
    return False


def main():
    ok = wait_done()
    print(f'冒烟2 完成: {ok}')
    txt = open(LOG, encoding='utf-8', errors='ignore').read()
    # ① 基准加载路径
    m = re.findall(r'基准数据从本地加载: (\S+ \(\d+条\))', txt)
    print('基准加载日志:', m if m else '未找到(可能下载或缺失)')
    loaded_from_index_daily = any('index_daily' in s for s in m)

    # ② 结果目录
    dirs = sorted(glob.glob('results/SmallAlpha191Test2_20220801_20220930_*'),
                  key=os.path.getmtime)
    d = dirs[-1]
    dv = pd.read_csv(os.path.join(d, 'daily_values.csv'))
    dv['date'] = pd.to_datetime(dv['date'])
    print(f'结果目录: {d}, 行数 {len(dv)}, 基准列存在: {"benchmark_cumulative" in dv.columns}')

    # ③ 独立核算: benchmark_cumulative = index_daily close 首日归一
    idx = pd.read_parquet('data/index_daily/sh000300.parquet')
    idx['date'] = pd.to_datetime(idx['date'])
    merged = dv[['date', 'benchmark_cumulative']].merge(
        idx[['date', 'close']], on='date', how='inner')
    if len(merged):
        # 回测窗口内基准首日归一 (回测内部基准从窗口首日算)
        ref = merged['close'].iloc[0]
        exp = merged['close'] / ref
        # 回测 benchmark_cumulative 可能含前日持仓收益口径, 对比相对变化率
        dv_ret = merged['benchmark_cumulative'].pct_change().fillna(0)
        exp_ret = exp.pct_change().fillna(0)
        max_diff = (dv_ret - exp_ret).abs().max()
        corr = dv_ret.corr(exp_ret)
        print(f'核算: 对齐 {len(merged)} 日, 基准日收益最大差 {max_diff:.2e}, 相关 {corr:.6f}')
        print(f'基准日收益完全一致: {max_diff < 1e-9}')
        return 0 if (loaded_from_index_daily and max_diff < 1e-9) else 1
    print('无法对齐')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
