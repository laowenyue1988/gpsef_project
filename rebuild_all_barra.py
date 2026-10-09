# -*- coding: utf-8 -*-
"""后台重建全部 BARRA 快照为最新版 (6因子正交化, 本地离线 PIT)。

用法: python rebuild_all_barra.py [日期文件]
日期文件每行一个 YYYYMMDD。已存在的快照跳过。
"""
import sys, os, time

sys.path.insert(0, r'E:\quant\gpsef_project')
os.chdir(r'E:\quant\gpsef_project')

from datetime import datetime
from strategies.wyckoff20.barra_builder import build_barra_snapshot

dates_file = sys.argv[1] if len(sys.argv) > 1 else r'/tmp/barra_dates_all.txt'
dates = [l.strip() for l in open(dates_file, encoding='utf-8') if l.strip()]

ok = skip = fail = 0
t0 = time.time()
for i, d in enumerate(dates):
    out = f'data/barra_exposures/barra_expo_{d}.parquet'
    if os.path.exists(out):
        skip += 1
        continue
    try:
        expo = build_barra_snapshot(datetime(int(d[:4]), int(d[4:6]), int(d[6:8])),
                                    force_rebuild=True, baseline='csi800')
        if expo is not None and len(expo) > 100:
            ok += 1
            print(f'[{i+1}/{len(dates)}] {d} OK ({len(expo)})', flush=True)
        else:
            fail += 1
            print(f'[{i+1}/{len(dates)}] {d} FAIL', flush=True)
    except Exception as e:
        fail += 1
        print(f'[{i+1}/{len(dates)}] {d} ERR {e}', flush=True)

elapsed = time.time() - t0
print(f'完成: {ok} 成功, {skip} 跳过, {fail} 失败, 耗时 {elapsed/60:.1f} 分钟', flush=True)
