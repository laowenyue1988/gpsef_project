# estimate_ffd_params.py
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller
from gpsef.data.local_provider import LocalDataProvider
from gpsef.utils.fracdiff import fracDiff_FFD
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_BASE_PATH = './data'
OUTPUT_FILE = './data/ffd_params/stock_d_star.parquet'
LOOKBACK_DAYS = 500
DEFAULT_D = 0.35


def find_min_d(series, max_d=1.0, step=0.05, pvalue_threshold=0.05):
    print(f"\n[DEBUG find_min_d] series length={len(series)}")
    for d in np.arange(0, max_d + step, step):
        if d == 0:
            test_series = series.dropna()
        else:
            diff_result = fracDiff_FFD(series, d)
            print(f"[DEBUG find_min_d] d={d:.2f}, diff_result type={type(diff_result)}, len(diff_result)={len(diff_result)}")
            # 根据类型包装
            if isinstance(diff_result, pd.Series):
                test_series = diff_result.dropna()
            else:
                # 如果长度不匹配，额外警告
                if len(diff_result) != len(series):
                    print(f"[ERROR] Length mismatch! diff_result={len(diff_result)}, series={len(series)}. Fixing by truncating/padding.")
                    # 紧急修复：截断或补NaN，同时记录这种情况
                    if len(diff_result) > len(series):
                        diff_result = diff_result[:len(series)]
                    else:
                        diff_result = np.pad(diff_result, (0, len(series)-len(diff_result)), constant_values=np.nan)
                test_series = pd.Series(diff_result, index=series.index).dropna()

        if len(test_series) < 10:
            continue
        try:
            result = adfuller(test_series, maxlag=min(12, len(test_series)//5))
            if result[1] < pvalue_threshold:
                print(f"[DEBUG find_min_d] Found min d={d:.2f}, p-value={result[1]:.4f}")
                return d
        except:
            continue
    print(f"[DEBUG find_min_d] No stationary d found, using default {DEFAULT_D}")
    return DEFAULT_D


def main():
    dp = LocalDataProvider(DATA_BASE_PATH)
    all_codes = dp.get_stock_list()
    logger.info(f"Total stocks: {len(all_codes)}")

    end_dt = pd.Timestamp.today()
    start_dt = end_dt - pd.Timedelta(days=LOOKBACK_DAYS)
    d_stars = {}

    for i, code in enumerate(all_codes):
        df = dp.get_daily_data(code, start_dt, end_dt)
        if df.empty or len(df) < 100:
            d_stars[code] = DEFAULT_D
            continue

        log_price = np.log(df['close'])
        print(f"\n===== Processing {code} ({i+1}/{len(all_codes)}) =====")
        d = find_min_d(log_price)
        d_stars[code] = d

        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(all_codes)}")
        # 为了快速看到问题，第一只股票后就暂停（可选）
        # if i == 0:
        #     break

    result_df = pd.DataFrame(list(d_stars.items()), columns=['ts_code', 'd_star'])
    median_d = result_df['d_star'].median()

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    result_df.to_parquet(OUTPUT_FILE)

    median_file = os.path.join(os.path.dirname(OUTPUT_FILE), 'd_star_median.txt')
    with open(median_file, 'w') as f:
        f.write(str(median_d))

    logger.info(f"Median d*: {median_d:.3f}, saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()