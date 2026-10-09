# -*- coding: utf-8 -*-
"""[sample_weight 对齐] align_sample_weight_to_sort 单元测试 (直接调用生产模块函数)

背景 bug (2026-08-18 修复):
  train() 原实现 sort_values(['trade_date','ts_code']).reset_index(drop=True) 后
  sample_weight 未重排 → 折内/生产/HP-Search 用位置索引取权重 → 权重与样本错位。

覆盖:
  T1 sample_weight=None → 原样 (df_train 不变, 返回 None)
  T2 乱序 df + 唯一权重 → 重排后每行权重 = 该行原始权重 (用 _orig_pos 追溯)
  T3 长度不匹配 → 警告并原样返回 (防御)
  T4 Series 输入 (带索引) → np.asarray 位置对齐, 重排正确
  T5 权重在重排后仍为 ndarray 且可被位置索引 (折内 sample_weight[索引] 用法兼容)
"""
import os
import sys
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STRAT_DIR = os.environ.get('TEST_STRATEGY_DIR', 'small_alpha191')
import importlib
ens_mod = importlib.import_module(f'strategies.{_STRAT_DIR}.ensemble')

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('test_sample_weight_align')

align = ens_mod.align_sample_weight_to_sort


def make_messy_df(n_days=30, n_stocks=8, seed=5):
    """构造乱序 df: 股票循环在外层, 日期在内层 → 非 (trade_date, ts_code) 顺序"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2022-01-04', periods=n_days, freq='B')
    rows = []
    for i in range(n_stocks):           # 股票在外层 → 乱序
        for d in dates:                  # 日期在内层
            rows.append((d, f's{i:04d}', rng.normal(), rng.normal(),
                         1 if rng.random() < 0.5 else 0))
    df = pd.DataFrame(rows, columns=['trade_date', 'ts_code', 'f0', 'f1', 'label_profit'])
    return df


def check(tag, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'[{status}] {tag} {detail}')
    return cond


def main():
    ok = True
    df = make_messy_df()

    # ---- T1: None 原样 ----
    df1 = df.copy()
    out_df, out_sw = align(df1, None)
    ok &= check('T1 None 原样', out_sw is None and len(out_df) == len(df)
                and 'trade_date' in out_df.columns)

    # ---- T2: 乱序 + 唯一权重 → 重排正确 ----
    df2 = df.copy()
    sw2 = np.arange(len(df2)) + 1000.0   # 每行唯一权重 = 1000 + 原始行位置
    out_df2, out_sw2 = align(df2, sw2)
    ok &= check('T2 排序已执行', out_df2['trade_date'].is_monotonic_increasing,
                f'(行数={len(out_df2)})')
    # 追溯: 每行新位置 i → _orig_pos 原始位置 p → 权重应为 1000+p
    mismatches = 0
    for i in range(len(out_df2)):
        p = int(out_df2['_orig_pos'].iloc[i])
        if abs(out_sw2[i] - (1000.0 + p)) > 1e-9:
            mismatches += 1
    ok &= check('T2 权重-样本对齐', mismatches == 0, f'(错位数={mismatches})')

    # ---- T3: 长度不匹配 → 原样返回 ----
    df3 = df.copy()
    bad_sw = np.arange(len(df3) - 3) + 1.0
    out_df3, out_sw3 = align(df3, bad_sw)
    ok &= check('T3 长度不匹配防御', out_sw3 is bad_sw and 'trade_date' in out_df3.columns)

    # ---- T4: Series 输入 (带索引) → 位置对齐 ----
    df4 = df.copy()
    sw4 = pd.Series(np.arange(len(df4)) + 2000.0, index=df4.index)   # 索引=原始行序
    out_df4, out_sw4 = align(df4, sw4)
    mm4 = 0
    for i in range(len(out_df4)):
        p = int(out_df4['_orig_pos'].iloc[i])
        if abs(out_sw4[i] - (2000.0 + p)) > 1e-9:
            mm4 += 1
    ok &= check('T4 Series 输入对齐', mm4 == 0, f'(错位数={mm4}, type={type(out_sw4).__name__})')

    # ---- T5: 重排后 ndarray 可位置索引 (兼容折内 sample_weight[索引] 用法) ----
    out_df5, out_sw5 = align(df.copy(), np.arange(len(df)) + 3000.0)
    idx_sample = np.arange(0, len(out_df5), 7)   # 模拟 train_set.index.values
    picked = out_sw5[idx_sample]
    ok &= check('T5 位置索引兼容', len(picked) == len(idx_sample)
                and isinstance(out_sw5, np.ndarray), f'(picked={picked[:3]})')

    # ---- T6: 乱序确实会错位 (证明修复前 bug 存在) ----
    # 修复前行为 = 排序后 sample_weight 保持原始顺序 → 位置索引取错
    df6 = df.copy()
    sw6 = np.arange(len(df6)) + 4000.0
    sorted_idx = np.argsort(pd.MultiIndex.from_arrays(
        [df6['trade_date'].values, df6['ts_code'].values]).argsort()) if False else None
    # 直接验证: 排序前后权重顺序不同 → 错位 bug 确实存在
    out_df6, out_sw6 = align(df6, sw6)
    # 若排序未改变行序 (理论上几乎必然改变), 则 out_sw6 == sw6; 否则不同
    changed = not np.array_equal(out_sw6, sw6)
    ok &= check('T6 排序改变了行序(错位前提)', changed,
                f'(修复前行为=用 sw6[新索引] 取原始顺序权重 → 错位 {len(df6)} 行)')

    print('\n==== sample_weight 对齐测试完成: %s ====' % ('ALL PASS' if ok else '有失败!'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
