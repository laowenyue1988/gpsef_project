# -*- coding: utf-8 -*-
"""对比两种 end_date 下 DynamicStockPool 在 2022-08-01 的候选池差异 (带错误输出)"""
import sys
import traceback
sys.path.insert(0, '.')
import pandas as pd
from datetime import datetime

out = open('results/pool_diff.txt', 'w', encoding='utf-8')

try:
    from gpsef.pool.dynamic_pool import DynamicStockPool
    from gpsef.data.local_provider import LocalDataProvider

    dp = LocalDataProvider('data')
    results = {}
    for tag, end in [('SHORT', datetime(2024, 3, 1)), ('FULL', datetime(2026, 8, 4))]:
        pool = DynamicStockPool({'filters': ['st', 'new_stock']})
        pool.initialize(dp, min_date=datetime(2020, 12, 9), end_date=end)
        cands = pool.get_candidates(pd.Timestamp('2022-08-01'), None)
        results[tag] = set(cands)
        out.write(f'{tag}(end={end.date()}): 候选 {len(cands)} 只\n')
        out.flush()

    ss, fs = results['SHORT'], results['FULL']
    out.write(f'\n仅SHORT {len(ss - fs)} 只: {sorted(ss - fs)[:12]}\n')
    out.write(f'仅FULL  {len(fs - ss)} 只: {sorted(fs - ss)[:12]}\n')
    out.write(f'交集 {len(ss & fs)} 只\n')
    out.write('POOL_DIFF_DONE\n')
except Exception:
    out.write('ERROR:\n' + traceback.format_exc())
finally:
    out.close()
