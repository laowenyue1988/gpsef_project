# strategies/fundamental_ml/build_snapshots.py (最终稳定高速版)
import os, sys, time, logging
from datetime import datetime, timedelta
from typing import Optional, Dict
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from strategies.alpha191.data_utils import get_trade_calendar

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_BASE_PATH = os.path.join(PROJECT_ROOT, 'data')
RAW_DAILY_DIR = os.path.join(DATA_BASE_PATH, 'raw_daily')
SNAPSHOT_DIR = os.path.join(DATA_BASE_PATH, 'fundamental_ml_snapshots')
PERF_CACHE_DIR = os.path.join(DATA_BASE_PATH, 'performance_reports')
START_DATE = '20260501'
END_DATE = '20260510'
QUANTILE_WINDOW = 252
VALID_PREFIXES = ('sh.600', 'sh.601', 'sh.603', 'sh.605', 'sh.688',
                  'sz.000', 'sz.002', 'sz.300')

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(PERF_CACHE_DIR, exist_ok=True)


# ---------- 报告期与财报缓存 ----------
def get_report_date_for_date(d: datetime) -> Optional[str]:
    candidates = []
    y = d.year
    if d >= datetime(y, 5, 1):
        candidates.append(f"{y - 1}1231")
    else:
        candidates.append(f"{y - 2}1231")
    if d >= datetime(y, 5, 1):
        candidates.append(f"{y}0331")
    if d >= datetime(y, 9, 1):
        candidates.append(f"{y}0630")
    if d >= datetime(y, 11, 1):
        candidates.append(f"{y}0930")
    candidates.sort(reverse=True)
    for rd in candidates:
        if os.path.exists(os.path.join(PERF_CACHE_DIR, f"{rd}.parquet")):
            return rd
    return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        '代码': 'code', '股票代码': 'code',
        '名称': 'name', '股票简称': 'name',
        '每股收益': 'eps',
        '每股净资产': 'bvps',
        '净资产收益率': 'roe',
        '营业总收入-营业总收入': 'revenue',
        '营业总收入同比增长': 'rev_growth',
        '净利润-净利润': 'net_profit',
        '净利润同比增长': 'profit_growth',
        '营业总收入-季度环比增长': 'rev_qoq',
        '净利润-季度环比增长': 'profit_qoq',
        '销售毛利率': 'gross_margin',
        '每股经营现金流量': 'ocf_per_share',
        '每股经营现金流': 'ocf_per_share',
        '所处行业': 'industry',
    }
    for src, tgt in rename.items():
        if src in df.columns and src != tgt:
            df = df.rename(columns={src: tgt})
    needed = ['code', 'eps', 'bvps', 'roe', 'revenue', 'net_profit',
              'rev_growth', 'profit_growth', 'rev_qoq', 'profit_qoq',
              'gross_margin', 'ocf_per_share', 'industry']
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan
    df['code'] = df['code'].astype(str).str.zfill(6)
    for c in needed:
        if c in df.columns and c not in ('code', 'industry'):
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def download_performance(report_date: str) -> Optional[pd.DataFrame]:
    cache_file = os.path.join(PERF_CACHE_DIR, f"{report_date}.parquet")
    if os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            if 'eps' in df.columns and 'bvps' in df.columns:
                return df
        except Exception:
            pass
        logger.warning(f"缓存 {report_date} 无效，重新下载")
        os.remove(cache_file)
    import akshare as ak
    try:
        logger.info(f"下载财报: {report_date}")
        df = ak.stock_yjbb_em(date=report_date)
        if df is None or df.empty:
            logger.info(f"财报 {report_date} 返回空")
            return None
        df = standardize_columns(df)
        df.to_parquet(cache_file, index=False)
        logger.info(f"财报 {report_date} 缓存成功，{len(df)} 条")
        return df
    except Exception as e:
        if 'NoneType' not in str(e) and 'not subscriptable' not in str(e):
            logger.warning(f"下载财报 {report_date} 失败: {e}")
        return None


# ---------- 预加载日线 ----------
def preload_all_daily() -> pd.DataFrame:
    logger.info("预加载全市场日线...")
    frames = []
    for fname in os.listdir(RAW_DAILY_DIR):
        if not fname.endswith('.parquet'):
            continue
        code = fname.replace('.parquet', '').replace('_', '.')
        if not code.startswith(VALID_PREFIXES):
            continue
        digits = code.replace('sh.', '').replace('sz.', '')
        try:
            df = pd.read_parquet(os.path.join(RAW_DAILY_DIR, fname),
                                 columns=['trade_date', 'close', 'amount', 'turn'])
        except Exception:
            continue
        df['code'] = digits
        frames.append(df)
    if not frames:
        raise RuntimeError("无日线文件")
    all_data = pd.concat(frames, ignore_index=True)
    all_data['trade_date'] = pd.to_datetime(all_data['trade_date'])
    all_data['code'] = all_data['code'].astype(str).str.zfill(6)  # 确保字符串补零
    all_data.sort_values(['code', 'trade_date'], inplace=True)
    all_data.reset_index(drop=True, inplace=True)
    logger.info(f"日线加载完成，{len(all_data)} 条，code 示例: {all_data['code'].iloc[-100:]}")
    return all_data


# ---------- 附加历史财务数据 ----------
def attach_fundamental_to_daily(all_data, report_cache):
    logger.info("附加历史财务数据...")
    all_dates = sorted(all_data['trade_date'].unique())
    date_to_rd, rd_cache = {}, {}
    for d in all_dates:
        rd = get_report_date_for_date(d)
        if rd:
            date_to_rd[d] = rd
            if rd not in rd_cache and rd in report_cache:
                rd_cache[rd] = report_cache[rd].set_index('code')

    all_data['eps'] = np.nan
    all_data['bvps'] = np.nan
    for d, rd in date_to_rd.items():
        if rd not in rd_cache:
            continue
        mask = all_data['trade_date'] == d
        idx = rd_cache[rd]
        all_data.loc[mask, 'eps'] = all_data.loc[mask, 'code'].map(idx['eps']).values
        all_data.loc[mask, 'bvps'] = all_data.loc[mask, 'code'].map(idx['bvps']).values

    all_data['pe'] = np.where((all_data['eps'] > 0) & all_data['eps'].notna(),
                              all_data['close'] / all_data['eps'], np.nan)
    all_data['pb'] = np.where((all_data['bvps'] > 0) & all_data['bvps'].notna(),
                              all_data['close'] / all_data['bvps'], np.nan)
    logger.info("财务数据附加完毕")


# ---------- 向量化计算滚动窗口分位数（一次性处理） ----------
def add_rolling_percentiles(all_data):
    logger.info("正在计算滚动窗口分位数（pe_pct, pb_pct, close_pct）...")
    # 确保按 code 和日期排序
    all_data.sort_values(['code', 'trade_date'], inplace=True)

    def rolling_pct(series):
        # 使用 numpy 滑动窗口计算分位数（比 pandas rolling apply 快得多）
        arr = series.values
        n = len(arr)
        out = np.full(n, np.nan)
        if n < QUANTILE_WINDOW * 0.8:
            return out
        # 为每个位置计算过去窗口中的分位数
        for i in range(QUANTILE_WINDOW, n):
            window = arr[i - QUANTILE_WINDOW:i]
            if len(window) >= QUANTILE_WINDOW * 0.8:
                out[i] = (window < arr[i]).mean()
        return out

    # 按 code 分组，应用上述函数
    all_data['close_pct'] = all_data.groupby('code')['close'].transform(rolling_pct)
    all_data['pe_pct'] = all_data.groupby('code')['pe'].transform(rolling_pct)
    all_data['pb_pct'] = all_data.groupby('code')['pb'].transform(rolling_pct)
    logger.info("滚动分位数计算完成")


# ---------- 单日快照生成 ----------
def build_single_snapshot(trade_date: datetime, all_data: pd.DataFrame,
                          report_cache: Dict[str, pd.DataFrame]):
    date_str = trade_date.strftime('%Y%m%d')
    out_path = os.path.join(SNAPSHOT_DIR, f'factors_{date_str}.parquet')
    if os.path.exists(out_path):
        logger.info(f"[{trade_date.date()}] 已跳过")
        return

    day_mask = all_data['trade_date'] == trade_date
    if not day_mask.any():
        logger.warning(f"[{trade_date.date()}] 无行情")
        return
    today = all_data[day_mask].copy()
    today['total_mv'] = np.where((today['amount'] > 0) & (today['turn'] > 0),
                                 today['amount'] / (today['turn'] / 100.0), np.nan)

    rd = get_report_date_for_date(trade_date)
    if rd is None or rd not in report_cache:
        logger.warning(f"[{trade_date.date()}] 无财报")
        return
    perf = report_cache[rd].copy()
    perf['code'] = perf['code'].astype(str).str.zfill(6)
    today['code'] = today['code'].astype(str).str.zfill(6)
    merged = today.merge(perf, on='code', how='inner')
    if merged.empty:
        logger.warning(f"[{trade_date.date()}] 合并空")
        return

    # 截面排名
    for col in ['pe', 'pb', 'roe', 'gross_margin', 'ocf_per_share',
                'rev_growth', 'rev_qoq', 'profit_growth', 'profit_qoq']:
        if col in merged.columns:
            merged[f'{col}_rank'] = merged[col].rank(pct=True)

    # 输出，code 列保证为字符串
    merged['code'] = merged['code'].astype(str).str.zfill(6)
    out_cols = ['code', 'close', 'pe', 'pb', 'roe', 'gross_margin',
                'ocf_per_share', 'rev_growth', 'profit_growth', 'rev_qoq',
                'profit_qoq', 'pe_pct', 'pb_pct', 'close_pct', 'total_mv', 'industry']
    rank_cols = [f'{c}_rank' for c in ['pe', 'pb', 'roe', 'gross_margin',
                                       'ocf_per_share', 'rev_growth', 'rev_qoq', 'profit_growth', 'profit_qoq']
                 if f'{c}_rank' in merged.columns]
    out_cols.extend(rank_cols)
    merged['date'] = trade_date
    existing = [c for c in out_cols if c in merged.columns]
    merged[existing].to_parquet(out_path, index=False)
    logger.info(f"[{trade_date.date()}] 保存 {len(merged)} 条 (报告期={rd})")


def main():
    start_dt = datetime.strptime(START_DATE, '%Y%m%d')
    end_dt = datetime.strptime(END_DATE, '%Y%m%d')
    all_data = preload_all_daily()

    cal_df = get_trade_calendar(START_DATE, END_DATE)
    trade_dates = sorted([pd.to_datetime(d) for d in cal_df['calendar_date'].tolist()])
    logger.info(f"共 {len(trade_dates)} 个交易日")

    all_reports = set()
    for td in trade_dates:
        rd = get_report_date_for_date(td)
        if rd:
            all_reports.add(rd)
    for y in range(2010, end_dt.year + 1):
        all_reports.update([f"{y}0331", f"{y}0630", f"{y}0930", f"{y}1231"])
    all_reports = {r for r in all_reports if int(r[:4]) <= end_dt.year + 1}

    logger.info(f"需获取 {len(all_reports)} 个报告期财报")
    report_cache = {}
    for rd in sorted(all_reports):
        df = download_performance(rd)
        if df is not None and not df.empty:
            report_cache[rd] = df
    logger.info(f"实际加载 {len(report_cache)} 个报告期")

    attach_fundamental_to_daily(all_data, report_cache)
    add_rolling_percentiles(all_data)  # 一次性计算所有历史分位数

    for i, td in enumerate(trade_dates):
        logger.info(f"[{i + 1}/{len(trade_dates)}] {td.date()}")
        t0 = time.time()
        build_single_snapshot(td, all_data, report_cache)
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            logger.info(f"  进度: {i + 1}/{len(trade_dates)}，本日耗时 {elapsed:.1f}s")


if __name__ == '__main__':
    main()