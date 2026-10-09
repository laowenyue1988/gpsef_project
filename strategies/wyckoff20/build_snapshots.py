# strategies/wyckoff20/build_snapshots.py (v5.1)
"""
Pre-build Wyckoff factor snapshots as parquet files (v5.1).

【v5.1 升级说明】
- 40 factors x 3 (raw/score/conf) = 120 feature columns
- Added industry classification (from local performance_reports, same as fundamental_ml)
- Added total_mv (calculated from amount/turn, same as fundamental_ml)
- Output includes code, industry, total_mv, {40 factors x 3}, total_score

CRITICAL FIXES (v3.2+):
1. Snapshot for date T uses data up to T-1 close ONLY
2. reference_date = T-1 (NOT T, NOT END_DATE)

v5.1 FIXES:
3. Trade calendar: reuse strategies.alpha191.data_utils.get_trade_calendar (same as fundamental_ml)
4. Industry data: from local performance_reports parquet files (same as fundamental_ml_v2/data_utils.py)
5. Market cap: calculated from amount/(turn/100) (same as fundamental_ml)

Usage:
  cd /path/to/gpsef_project
  python -m strategies.wyckoff20.build_snapshots

Output:
  data/wyckoff20_v5_snapshots/
    factors_20200102.parquet
    factors_20200103.parquet
    ...
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List

# P0-2 FIX: Point-in-Time industry classification
from strategies.wyckoff20.industry_pit import IndustryMapPiT

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ Configuration ============
SNAPSHOT_DIR = os.path.join(PROJECT_ROOT, 'data', 'wyckoff20_v5_snapshots')
INDUSTRY_MAP_FILE = os.path.join(PROJECT_ROOT, 'data', 'industry_map.parquet')
START_DATE = '20240322'
END_DATE = '20240922'
VALID_PREFIXES = ('sh.600', 'sh.601', 'sh.603', 'sh.605', 'sh.688', 'sh.689',
                  'sz.000', 'sz.001', 'sz.002', 'sz.003', 'sz.300', 'sz.301', 'sz.302')
MIN_DAILY_BARS = 60

os.makedirs(SNAPSHOT_DIR, exist_ok=True)


# ============ Industry Classification (aligned with fundamental_ml_v2/data_utils.py) ============

def load_or_build_industry_map() -> Dict[str, str]:
    """
    Load or build industry mapping {code_6digit: industry_name}.

    Implementation aligned with fundamental_ml_v2/data_utils.py:
    1. Check local parquet cache (data/industry_map.parquet)
    2. Build from data/performance_reports/*.parquet (same source as fundamental_ml)
    3. Save to parquet cache for reuse
    """
    # 1. Try local parquet cache
    if os.path.exists(INDUSTRY_MAP_FILE):
        try:
            df = pd.read_parquet(INDUSTRY_MAP_FILE)
            if 'code' in df.columns and 'industry' in df.columns:
                cache = dict(zip(df['code'].astype(str).str.zfill(6), df['industry']))
                if len(cache) > 100:
                    logger.info(f"Loaded industry map from parquet cache: {len(cache)} stocks")
                    return cache
        except Exception as e:
            logger.warning(f"Failed to read industry_map.parquet: {e}")

    # 2. Build from local performance_reports (same as fundamental_ml_v2/data_utils.py)
    perf_dir = os.path.join(PROJECT_ROOT, 'data', 'performance_reports')
    if not os.path.exists(perf_dir):
        logger.warning(f"Performance reports directory not found: {perf_dir}")
        return {}

    # Find the latest report file (most comprehensive)
    perf_files = sorted([f for f in os.listdir(perf_dir) if f.endswith('.parquet')])
    if not perf_files:
        logger.warning("No performance report parquet files found")
        return {}

    # Try the latest file first (should have the most stocks)
    for fname in reversed(perf_files):
        fpath = os.path.join(perf_dir, fname)
        try:
            df = pd.read_parquet(fpath)
            if 'code' not in df.columns:
                continue

            # Try Chinese column name first (same as fundamental_ml_v2)
            ind_col = None
            for col in ['所处行业', 'industry', '行业']:
                if col in df.columns:
                    ind_col = col
                    break

            if ind_col is None:
                continue

            df['code'] = df['code'].astype(str).str.zfill(6)
            df_map = df[['code', ind_col]].drop_duplicates().rename(columns={ind_col: 'industry'})

            # Filter to valid stock codes
            df_map = df_map[df_map['code'].str.match(r'^[036]\d{5}$')]

            if len(df_map) > 100:
                # Save to parquet cache (same format as fundamental_ml_v2)
                df_map.to_parquet(INDUSTRY_MAP_FILE, index=False)
                cache = dict(zip(df_map['code'], df_map['industry']))
                logger.info(f"Built industry map from {fname}: {len(cache)} stocks")
                return cache

        except Exception as e:
            logger.warning(f"Failed to read {fname}: {e}")
            continue

    logger.warning("Failed to build industry map from performance reports")
    return {}


# ============ Trade Calendar (aligned with fundamental_ml_v2) ============

def get_trade_calendar(start_date: str, end_date: str) -> List[datetime]:
    """
    Get trade calendar. Reuses strategies.alpha191.data_utils.get_trade_calendar
    which reads from local data/trade_datas.csv (same as fundamental_ml strategy).

    Args:
        start_date: YYYYMMDD format
        end_date: YYYYMMDD format

    Returns:
        Sorted list of trading day datetimes
    """
    try:
        from strategies.alpha191.data_utils import get_trade_calendar as _get_cal
        cal_df = _get_cal(start_date, end_date)
        if cal_df is not None and not cal_df.empty:
            dates = sorted([pd.Timestamp(d).to_pydatetime() for d in cal_df['calendar_date'].tolist()])
            logger.info(f"Loaded trade calendar from alpha191.data_utils: {len(dates)} trading days")
            return dates
    except ImportError:
        logger.warning("strategies.alpha191.data_utils not available, trying direct file read")
    except Exception as e:
        logger.warning(f"strategies.alpha191.data_utils.get_trade_calendar failed: {e}")

    # Fallback: directly read trade_datas.csv (same file that alpha191.data_utils reads)
    trade_datas_path = os.path.join(PROJECT_ROOT, 'data', 'trade_datas.csv')
    if os.path.exists(trade_datas_path):
        try:
            df = pd.read_csv(trade_datas_path)
            df['calendar_date'] = pd.to_datetime(df['calendar_date'])
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')
            trading_days = df[df['is_trading_day'] == 1]['calendar_date'].tolist()
            result = [d for d in trading_days if start_dt <= d <= end_dt]
            if len(result) > 10:
                logger.info(f"Loaded trade calendar from trade_datas.csv: {len(result)} trading days")
                return sorted(result)
        except Exception as e:
            logger.warning(f"Failed to read trade_datas.csv: {e}")

    # Fallback: infer from raw_daily parquet files (same as LocalDataProvider)
    raw_daily_dir = os.path.join(PROJECT_ROOT, 'data', 'raw_daily')
    if os.path.exists(raw_daily_dir):
        try:
            import glob
            all_dates = set()
            parquet_files = sorted(glob.glob(os.path.join(raw_daily_dir, '*.parquet')))[:30]
            for f in parquet_files:
                try:
                    df = pd.read_parquet(f, columns=['trade_date'])
                    all_dates.update(pd.to_datetime(df['trade_date']).tolist())
                except Exception:
                    continue
            if all_dates:
                start_dt = datetime.strptime(start_date, '%Y%m%d')
                end_dt = datetime.strptime(end_date, '%Y%m%d')
                result = sorted([d for d in all_dates if start_dt <= d <= end_dt])
                if len(result) > 10:
                    logger.info(f"Inferred trade calendar from raw_daily: {len(result)} trading days")
                    return result
        except Exception as e:
            logger.warning(f"Failed to infer calendar from raw_daily: {e}")

    logger.error("No trade calendar available! Check data/trade_datas.csv")
    return []


# ============ Stock List ============

def get_stock_list() -> List[str]:
    """
    Get list of valid stock codes.
    Reuses LocalDataProvider.get_stock_list() (same as fundamental_ml strategy).

    Returns:
        List of stock codes like 'sh.600000', 'sz.000001'
    """
    # Method 1: Use LocalDataProvider (same as fundamental_ml strategy)
    try:
        from gpsef.data import LocalDataProvider
        dp = LocalDataProvider(base_path=os.path.join(PROJECT_ROOT, 'data'))
        codes = dp.get_stock_list()
        if codes and len(codes) > 100:
            logger.info(f"Got stock list from LocalDataProvider: {len(codes)} stocks")
            return codes
    except Exception:
        pass

    # Method 2: Scan raw_daily directory
    raw_daily_dir = os.path.join(PROJECT_ROOT, 'data', 'raw_daily')
    if os.path.exists(raw_daily_dir):
        code_list = []
        for fname in sorted(os.listdir(raw_daily_dir)):
            if not fname.endswith('.parquet'):
                continue
            code = fname.replace('.parquet', '').replace('_', '.')
            if code.startswith(VALID_PREFIXES):
                code_list.append(code)
        if code_list:
            logger.info(f"Got stock list from raw_daily scan: {len(code_list)} stocks")
            return code_list

    logger.error("Failed to get stock list from any source!")
    return []


def find_prev_trading_day(trade_dates: List[datetime],
                          target_date: datetime) -> Optional[datetime]:
    """Find the previous trading day before target_date from trade calendar."""
    prev = None
    for d in trade_dates:
        if d < target_date:
            prev = d
        else:
            break
    return prev


def build_single_snapshot(trade_date: datetime,
                          prev_trade_date: datetime,
                          data_provider,
                          factor_engine,
                          code_list: List[str],
                          industry_map: Dict[str, str]) -> Optional[str]:
    """
    Build factor snapshot for a single trading day (v5.1).

    CRITICAL:
    - Uses data up to prev_trade_date (T-1) close ONLY
    - Uses reference_date = prev_trade_date

    Saves as parquet: data/wyckoff20_v5_snapshots/factors_YYYYMMDD.parquet
    Columns: code, industry, total_mv, {factor_name}_raw, {factor_name}_score,
             {factor_name}_conf, total_score

    Returns:
        output path if successful, None otherwise
    """
    date_str = trade_date.strftime('%Y%m%d')
    out_path = os.path.join(SNAPSHOT_DIR, f'factors_{date_str}.parquet')

    if os.path.exists(out_path):
        logger.info(f"[{trade_date.date()}] Already exists, skipping")
        return out_path

    lookback_start = trade_date - timedelta(days=250)

    # Compute factors for each stock
    results = []
    for code in code_list:
        try:
            df = data_provider.get_daily_data(
                code,
                lookback_start,
                prev_trade_date,
                reference_date=prev_trade_date
            )
            if df is None or len(df) < MIN_DAILY_BARS:
                continue

            # Process dataframe
            if 'trade_date' in df.columns:
                df = df.copy()
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.set_index('trade_date')
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            if 'volume' in df.columns:
                df = df[df['volume'] > 0]

            if len(df) < MIN_DAILY_BARS:
                continue

            # Price filter
            # [FIX 2026-09-15 · D3] 原实现用 df['close']（= 以 prev_trade_date 为基准日的**前复权价**）
            #   配合固定阈值 2.0/1000.0 ⇒ 价格水平随基准日逐日漂移，同一历史日的判定会在不同
            #   构建批次间翻转（实测 122 只股票曾翻转、600162 达 1112 天）⇒ 回测不可复现 +
            #   小市值偏好被扭曲。
            #   修法：过滤器改用**未复权价 raw_close**（对基准日免疫，且「2~1000 元」本质上就是
            #   绝对价格区间的语义）；缺失时回退 close 并告警。
            if 'raw_close' in df.columns:
                last_close = float(df['raw_close'].iloc[-1])
            else:
                last_close = float(df['close'].iloc[-1])
                logger.warning("[D3] %s 缺 raw_close ⇒ 过滤器回退 close（判定可能随基准日漂移）", code)
            last_volume = float(df['volume'].iloc[-1])
            if last_close < 2.0 or last_close > 1000.0:
                continue
            if last_volume < 500000:
                continue

            # Compute all 40 factors (v5.1)
            factor_results = factor_engine.calculate_all_factors(
                df_daily=df,
                df_5min=None,
                current_date=date_str
            )
            composite = factor_engine.compute_composite_score(factor_results)

            # Build row with 3 columns per factor (v5.0)
            code_digits = code.replace('sh.', '').replace('sz.', '')
            row = {
                'code': code_digits,
                'industry': industry_map.get(code_digits.zfill(6), 'unknown'),
                'total_score': composite['total_score']
            }

            # Calculate total_mv from latest day's data
            # Same formula as fundamental_ml: total_mv = amount / (turn / 100)
            last_amount = float(df['amount'].iloc[-1]) if 'amount' in df.columns else np.nan
            last_turn = float(df['turn'].iloc[-1]) if 'turn' in df.columns else np.nan
            if pd.notna(last_amount) and pd.notna(last_turn) and last_turn > 0:
                row['total_mv'] = last_amount / (last_turn / 100.0)
            else:
                row['total_mv'] = np.nan

            for fname in factor_engine.ALL_FACTORS:
                result = factor_results.get(fname)
                if result is not None:
                    row[f'{fname}_raw'] = result.raw_value
                    row[f'{fname}_score'] = result.score
                    row[f'{fname}_conf'] = result.confidence

            results.append(row)

        except Exception as e:
            continue

    if not results:
        logger.warning(f"[{trade_date.date()}] No valid factor results")
        return None

    df_out = pd.DataFrame(results)

    # Cross-section normalization (only for score columns)
    for fname in factor_engine.ALL_FACTORS:
        score_col = f'{fname}_score'
        if score_col not in df_out.columns:
            continue
        vals = df_out[score_col].values
        if len(vals) < 10:
            continue
        # Winsorize
        lower, upper = np.percentile(vals, 1), np.percentile(vals, 99)
        vals = np.clip(vals, lower, upper)
        # Z-score normalize then sigmoid
        mean, std = np.mean(vals), np.std(vals)
        if std > 1e-8:
            vals = (vals - mean) / std
            vals = 1.0 / (1.0 + np.exp(-vals))
        df_out[score_col] = vals

    # Ensure code is string (6-digit)
    df_out['code'] = df_out['code'].astype(str).str.zfill(6)

    # Column ordering: code, industry, total_mv, total_score, then factor columns
    base_cols = ['code', 'industry', 'total_mv', 'total_score']
    factor_cols = [c for c in df_out.columns if c not in base_cols]
    factor_cols.sort()
    ordered_cols = base_cols + factor_cols
    ordered_cols = [c for c in ordered_cols if c in df_out.columns]
    df_out = df_out[ordered_cols]

    df_out.to_parquet(out_path, index=False)
    logger.info(f"[{trade_date.date()}] Saved {len(df_out)} stocks "
                f"(ref_date={prev_trade_date.strftime('%Y%m%d')})")
    return out_path


def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("Wyckoff Factor Snapshot Builder (v5.1)")
    logger.info(f"  Date range: {START_DATE} ~ {END_DATE}")
    logger.info(f"  Output: {SNAPSHOT_DIR}")
    logger.info(f"  CRITICAL: Snapshot for T uses data up to T-1 close ONLY")
    logger.info(f"  CRITICAL: reference_date = T-1")
    logger.info(f"  Industry: from data/performance_reports (same as fundamental_ml)")
    logger.info(f"  Trade calendar: from alpha191.data_utils (same as fundamental_ml)")
    logger.info(f"  Features: 40 factors x 3 (raw/score/conf) = 120 columns + industry + total_mv")
    logger.info("=" * 60)

    # Load industry map from local performance reports (same as fundamental_ml_v2/data_utils.py)
    industry_pit = IndustryMapPiT()  # P0-2 FIX
    logger.info("Industry: Point-in-Time classification (P0-2 FIX)")

    # Get trade calendar (reuse alpha191.data_utils, same as fundamental_ml)
    # Extend start date by 30 days to ensure the first snapshot date has a prev trading day
    extended_start = (datetime.strptime(START_DATE, '%Y%m%d') - timedelta(days=30)).strftime('%Y%m%d')
    trade_dates_full = get_trade_calendar(extended_start, END_DATE)
    logger.info(f"Total trading days (with extension): {len(trade_dates_full)}")

    # Filter to the actual desired range
    start_dt = datetime.strptime(START_DATE, '%Y%m%d')
    trade_dates = [d for d in trade_dates_full if d >= start_dt]
    logger.info(f"Total trading days [{START_DATE} ~ {END_DATE}]: {len(trade_dates)}")

    if not trade_dates:
        logger.error("No trading days found! Check data/trade_datas.csv")
        return

    # Get stock list from LocalDataProvider (same as fundamental_ml)
    code_list = get_stock_list()
    logger.info(f"Total stocks: {len(code_list)}")

    if not code_list:
        logger.error("No stocks found! Check data/raw_daily/ directory")
        return

    # Init data provider
    from gpsef.data import LocalDataProvider
    dp = LocalDataProvider(base_path=os.path.join(PROJECT_ROOT, 'data'))

    # Init factor engine (v5.1)
    from strategies.wyckoff20.factors_v5 import WyckoffFactorEngineV5
    factor_engine = WyckoffFactorEngineV5()

    # Build snapshots
    start_time = time.time()
    success_count = 0
    fail_count = 0

    for i, td in enumerate(trade_dates):
        # Find previous trading day from FULL calendar (includes pre-START_DATE days)
        prev_td = find_prev_trading_day(trade_dates_full, td)
        if prev_td is None:
            logger.warning(f"[{td.date()}] No previous trading day, skipping")
            fail_count += 1
            continue

        industry_map = industry_pit.get_map(td)  # P0-2 FIX
        result = build_single_snapshot(td, prev_td, dp, factor_engine, code_list, industry_map)
        if result is not None and os.path.exists(result):
            success_count += 1
            if (i + 1) % 20 == 0 or i == 0:
                elapsed = time.time() - start_time
                logger.info(f"  Progress: {i+1}/{len(trade_dates)}, "
                            f"elapsed: {elapsed:.0f}s")
        else:
            fail_count += 1

    total_elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Snapshot build complete: {success_count} success, "
                f"{fail_count} fail, elapsed: {total_elapsed:.0f}s")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
