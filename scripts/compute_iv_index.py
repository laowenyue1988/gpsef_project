"""
预计算 CSI 300 ETF 期权 QVIX 指数并保存为 Parquet

数据源: AKShare index_option_300etf_qvix() — 免费，无需 API Key
输出: data/iv_index.parquet

用法:
    python scripts/compute_iv_index.py

防未来数据设计:
    - QVIX[d] 在日期 d 收盘后发布（~15:30）
    - 回测在日期 T 的 before_market_open 中需要 IV 数据
    - 可用数据截止 T-1
    - scale 列提前 shift 1 日 → 回测直接查 scale[T] 即为 T-1 的 QVIX 计算值
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from collections import deque
import akshare as ak
from pathlib import Path

# ============================================================
#  配置
# ============================================================
OUTPUT_DIR = Path(os.path.join(os.path.dirname(__file__), '..', 'data'))
OUTPUT_FILE = OUTPUT_DIR / 'iv_index.parquet'

# IV 仓位控制参数（与 config 默认值保持一致）
SMOOTH_DAYS = 5        # EMA 平滑窗口
LOOKBACK = 252         # 滚动百分位窗口
PCT_LOW = 0.25         # 25 分位以下满仓
PCT_HIGH = 0.75        # 75 分位以上最低仓
S_MIN = 0.25           # 最低仓位比例
SCALING = 'linear'     # 线性缩放
MIN_HISTORY = 20       # 历史不足 20 天时返回中立

# ============================================================
#  数据获取
# ============================================================
def fetch_qvix() -> pd.DataFrame:
    """从 AKShare 获取 CSI 300 ETF QVIX 日频数据"""
    print('[1/5] Fetching QVIX data from AKShare...')
    df = ak.index_option_300etf_qvix()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    print(f'      Fetched {len(df)} rows, {df["close"].isna().sum()} NaN, '
          f'range [{df.index[0].date()}, {df.index[-1].date()}]')
    return df


# ============================================================
#  数据清洗
# ============================================================
def clean_qvix(df: pd.DataFrame) -> pd.Series:
    """清洗 QVIX 数据：前向填充 NaN，检查交易日历"""
    print('[2/5] Cleaning data...')
    qvix = df['close'].copy()

    # 1. 前向填充缺失值
    nan_before = qvix.isna().sum()
    qvix = qvix.ffill()
    nan_after = qvix.isna().sum()
    print(f'      NaN: {nan_before} → {nan_after} (ffill)')

    # 2. 找到第一个有效日期
    first_valid = qvix.first_valid_index()
    qvix = qvix.loc[first_valid:]
    print(f'      First valid date: {first_valid.date()}')

    # 3. 检查日内异常值（QVIX 通常 10-60，超出此范围的标记）
    outliers = (qvix < 8) | (qvix > 80)
    if outliers.any():
        print(f'      WARNING: {outliers.sum()} outliers detected (QVXI < 8 or > 80)')
        print(f'      Outlier dates: {list(qvix[outliers].index)}')

    # 4. 检查大幅跳空（日间变化 > 10 个百分点）
    jumps = qvix.diff().abs() > 15
    if jumps.any():
        print(f'      NOTE: {jumps.sum()} large jumps (>15pp day-over-day)')
        for d in qvix[jumps].index[:5]:
            print(f'            {d.date()}: from {qvix.shift(1).loc[d]:.1f} to {qvix.loc[d]:.1f}')

    return qvix


# ============================================================
#  EMA 平滑
# ============================================================
def compute_ema(qvix: pd.Series, n: int = 5) -> pd.Series:
    """计算 N 日 EMA 平滑"""
    alpha = 2.0 / (n + 1)
    return qvix.ewm(alpha=alpha, adjust=False).mean()


# ============================================================
#  滚动百分位（增量更新模拟）
# ============================================================
def compute_percentile(series: pd.Series, window: int = 252, min_history: int = 20) -> pd.Series:
    """计算每个日期的滚动百分位（相对于过去 window 天）

    结果是一个 Series，每个值表示当日 IV 在过去 window 天中的分位。
    分位范围 (0, 1)，0=历史最低，1=历史最高。

    使用 deque 增量更新，确保每天的计算只用到当天及之前的数据。
    """
    print('[3/5] Computing rolling percentile (252-day window)...')
    history = deque(maxlen=window)
    results = []

    for date, iv in series.items():
        history.append(iv)
        if len(history) < min_history:
            results.append(0.5)  # 历史不足 → 中立
        else:
            count_less = sum(1 for h in history if h <= iv)
            pct = count_less / len(history)
            results.append(pct)

    pct_series = pd.Series(results, index=series.index, dtype=float)
    print(f'      Computed {len(pct_series)} percentiles, range [{pct_series.min():.3f}, {pct_series.max():.3f}]')
    return pct_series


# ============================================================
#  缩放因子
# ============================================================
def compute_scale(pct_series: pd.Series, pct_low=0.25, pct_high=0.75, s_min=0.25,
                  scaling='linear') -> pd.Series:
    """根据百分位计算仓位缩放因子"""
    print('[4/5] Computing scale factors...')
    results = []

    for pct in pct_series:
        if pct <= pct_low:
            results.append(1.0)
        elif pct >= pct_high:
            results.append(s_min)
        else:
            if scaling == 'linear':
                x = (pct - pct_low) / (pct_high - pct_low)
                results.append(1.0 - (1.0 - s_min) * x)
            else:  # sigmoid
                midpoint = (pct_low + pct_high) / 2
                x = 1.0 / (1.0 + np.exp(-10.0 * (midpoint - pct)))
                results.append(s_min + (1.0 - s_min) * x)

    scale = pd.Series(results, index=pct_series.index, dtype=float)
    print(f'      Scale range [{scale.min():.2f}, {scale.max():.2f}], mean={scale.mean():.2f}')
    return scale


# ============================================================
#  防未来数据：Shift 1 日
# ============================================================
def shift_for_backtest(series: pd.Series) -> pd.Series:
    """将 scale 向前 shift 1 个交易日 → 回测中查 scale[T] 用的是 T-1 的 QVIX"""
    shifted = series.shift(1)
    # 第一天会变成 NaN → 填充为 1.0（满仓，因为不知道历史 IV）
    shifted.iloc[0] = 1.0
    return shifted


# ============================================================
#  组装并保存
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 获取数据
    df = fetch_qvix()

    # 2. 清洗
    qvix_raw = clean_qvix(df)

    # 3. EMA 平滑
    qvix_smooth = compute_ema(qvix_raw, SMOOTH_DAYS)

    # 4. 滚动百分位
    pct = compute_percentile(qvix_smooth, LOOKBACK, MIN_HISTORY)

    # 5. 缩放因子
    scale_raw = compute_scale(pct, PCT_LOW, PCT_HIGH, S_MIN, SCALING)

    # 6. 防未来数据：shift 1 日
    scale_bt = shift_for_backtest(scale_raw)

    # 7. 组装输出
    out = pd.DataFrame({
        'qvix_raw': qvix_raw,
        'qvix_smooth': qvix_smooth,
        'percentile': pct,
        'scale': scale_raw,            # 原始缩放（含当日 QVIX）
        'scale_bt': scale_bt,          # 回测用缩放（shift 1 日，无未来数据）
    })

    # 8. 保存
    print(f'[5/5] Saving to {OUTPUT_FILE}...')
    out.to_parquet(OUTPUT_FILE)
    print(f'      Saved {len(out)} rows')

    # 9. 验证
    print('\n=== Verification ===')
    # 检查 scale vs scale_bt 的差异
    diff = (out['scale'] != out['scale_bt']).sum()
    print(f'      Days where scale != scale_bt: {diff} (shift effect)')

    # 在关键日期验证
    for date_str in ['2020-02-03', '2020-03-23', '2022-04-27', '2024-01-15']:
        try:
            row = out.loc[date_str]
            print(f'      {date_str}: QVIX={row["qvix_smooth"]:.1f} '
                  f'pct={row["percentile"]:.2f} scale_bt={row["scale_bt"]:.2f}')
        except:
            pass

    print('\nDone. Use scale_bt column for backtest.')


if __name__ == '__main__':
    main()
