# -*- coding: utf-8 -*-
"""
[test3] 独立因子快照预计算脚本（GPSEF 框架适配版）
功能：输入日期区间，批量生成区间内每个交易日的全市场股票因子快照。

[A1-20260905] rank() 横截面语义恢复: calculate_alpha_factors(rank_mode='cross_section') 两遍法
[A2-20260905] 快照宇宙重构: raw_daily 全集 ∪ historical_stocks, list_date=首 bar (补退市股, 抗幸存者偏差)
[A3-20260905] 停牌过滤: 快照当日 tradestatus=='1' 才进入快照 (硬性要求: 全市场非停牌股票全覆盖)
[A4-20260905] 加载窗 365→400 自然日 (alpha_025 的 250 交易日不再截断)
[A5-20260905] 快照净化: inf→NaN, |x|>1e30→NaN
[A8-20260905] import 全部指向 small_alpha191_test3 模块 + 一致性断言
[A9-20260905] FFD d* 文件启动硬校验
[A10-20260905] orig/FFD 行合并: 因子引擎对 orig/FFD 任务分行返回, 须按 ts_code 合并后再过滤保存
[A11-20260905] 新股/短历史补齐: worker 的 <10bar 守卫会丢弃新股, 按当日 bar 补齐全 NaN 因子行
支持 FFD 因子增量生成：若快照已存在但缺少 _FFD 列，则只补充计算 FFD 部分并合并。
"""

import os
import sys
import time
import logging
import gc
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpsef.data import LocalDataProvider
# [A8-20260905] 指向 test3 模块 (原指向旧目录 small_alpha191, 存在未来漂移风险)
from strategies.small_alpha191_test3.data_utils import get_trade_calendar, VALID_STOCK_PREFIXES
from strategies.small_alpha191_test3.factors import calculate_alpha_factors

# [A8] 一致性断言: 确保 import 的确实是 test3 模块
import strategies.small_alpha191_test3.factors as _factors_mod
assert 'small_alpha191_test3' in _factors_mod.__file__, \
    f"[A8] factors 模块路径异常: {_factors_mod.__file__}"

# ==================== 全局变量配置 ====================
START_DATE = '20220801'
END_DATE = '20220803'
DATA_BASE_PATH = r'E:\quant\gpsef_project\data'
SNAPSHOT_DIR_NAME = 'factor_snapshots_test'      # [隔离] 新快照目录, 不覆盖原 factor_snapshots
BENCHMARK_CODE = 'sh.000300'
FACTOR_BATCH_SIZE = 800
USE_PARALLEL = True
MAX_WORKERS = 10
SKIP_EXISTING = True
RANK_MODE = 'cross_section'                      # [A1] 'cross_section'=截面rank两遍法 | 'timeseries'=旧行为
LOAD_WINDOW_DAYS = 400                           # [A4] 365→400 自然日
# ================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def build_stock_universe(data_dir: str) -> pd.DataFrame:
    """[A2-20260905] 构建快照宇宙: raw_daily 全部有效前缀股票 ∪ historical_stocks。
    list_date = 各股票数据文件首 bar 日期 (个股自身历史, PIT 安全)。
    结果缓存到 stock_list/universe_raw_cache_test3.parquet (带文件 mtime 增量更新)。"""
    cache_file = os.path.join(data_dir, 'stock_list', 'universe_raw_cache_test3.parquet')
    raw_dir = os.path.join(data_dir, 'raw_daily')

    # 1) 候选 code 全集
    codes = set()
    hs_file = os.path.join(data_dir, 'stock_list', 'historical_stocks.parquet')
    if os.path.exists(hs_file):
        hs = pd.read_parquet(hs_file)
        codes |= {c for c in hs['code'] if c.startswith(VALID_STOCK_PREFIXES)}
        logger.info(f"historical_stocks 提供 {len(codes)} 只有效前缀股票")
    n_hs = len(codes)
    for f in os.listdir(raw_dir):
        if f.endswith('.parquet'):
            code = f[:-len('.parquet')].replace('_', '.')
            if code.startswith(VALID_STOCK_PREFIXES):
                codes.add(code)
    logger.info(f"并入 raw_daily 后宇宙 {len(codes)} 只 (新增 {len(codes) - n_hs} 只有数据但不在 historical_stocks 的股票)")

    # 2) 读缓存, 增量推断 list_date
    cache = {}
    if os.path.exists(cache_file):
        df_cache = pd.read_parquet(cache_file)
        for _, row in df_cache.iterrows():
            fpath = os.path.join(raw_dir, row['code'].replace('.', '_') + '.parquet')
            if os.path.exists(fpath) and os.path.getmtime(fpath) == row['mtime']:
                cache[row['code']] = row['list_date']

    records = []
    n_scan = 0
    for code in sorted(codes):
        fpath = os.path.join(raw_dir, code.replace('.', '_') + '.parquet')
        if not os.path.exists(fpath):
            continue
        if code in cache:
            records.append({'code': code, 'list_date': cache[code],
                            'mtime': os.path.getmtime(fpath)})
            continue
        try:
            df_raw = pd.read_parquet(fpath, columns=['trade_date'])
            if df_raw.empty:
                continue
            records.append({'code': code, 'list_date': df_raw['trade_date'].min(),
                            'mtime': os.path.getmtime(fpath)})
            n_scan += 1
        except Exception as e:
            logger.debug(f"  读取 {code} 失败: {e}")

    df = pd.DataFrame(records)
    df.to_parquet(cache_file, index=False)
    logger.info(f"宇宙构建完成: {len(df)} 只 (新扫描 {n_scan} 只, 缓存命中 {len(cache)} 只)")
    return df


def get_benchmark_data(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """获取基准指数数据"""
    try:
        # [A8-20260905] 指向 test3 模块
        from strategies.small_alpha191_test3.benchmark import fetch_benchmark_data
        return fetch_benchmark_data(start_date, end_date, BENCHMARK_CODE)
    except Exception as e:
        logger.warning(f"获取基准数据失败: {e}")
        return None


def _sanitize_snapshot(day_data: pd.DataFrame) -> pd.DataFrame:
    """[A5-20260905] 快照净化: inf→NaN, |x|>1e30→NaN (有限大值保留给下游 winsorize)"""
    alpha_cols = [c for c in day_data.columns if c.startswith('alpha_')]
    if not alpha_cols:
        return day_data
    sub = day_data[alpha_cols].astype(np.float64)
    n_inf = int(np.isinf(sub.values).sum())
    n_big = int((np.abs(sub.values) > 1e30).sum())
    sub = sub.replace([np.inf, -np.inf], np.nan)
    sub = sub.where(sub.abs() <= 1e30, np.nan)
    day_data[alpha_cols] = sub.astype(np.float32)
    if n_inf or n_big:
        logger.info(f"  [A5] 快照净化: inf→NaN {n_inf} 个, |x|>1e30→NaN {n_big} 个")
    return day_data


def _filter_suspended(day_data: pd.DataFrame) -> pd.DataFrame:
    """[A3-20260905] 停牌过滤: 仅保留 tradestatus=='1' 的股票 (硬性要求)"""
    if 'tradestatus' not in day_data.columns:
        logger.warning("  [A3] day_data 无 tradestatus 列, 跳过停牌过滤 (不应发生)")
        return day_data
    before = len(day_data)
    out = day_data[day_data['tradestatus'].astype(str) == '1'].copy()
    logger.info(f"  [A3] 停牌过滤: {before} → {len(out)} (剔除停牌 {before - len(out)} 只)")
    return out


def generate_snapshot_for_date(trade_date: datetime,
                               stock_universe: pd.DataFrame,
                               data_provider: LocalDataProvider) -> bool:
    logger.info(f"开始处理交易日: {trade_date.date()}")

    # 1. 获取已有快照（可能为 None）
    existing_snapshot = None
    if SKIP_EXISTING:
        existing_snapshot = data_provider.get_factor_snapshot(trade_date)

    # 2. 如果已有快照且包含 FFD 列，直接跳过
    if existing_snapshot is not None:
        if any(col.endswith('_FFD') for col in existing_snapshot.columns):
            logger.info(f"  快照已存在且包含 FFD 因子，跳过")
            return True
        else:
            logger.info(f"  快照已存在但缺少 FFD 因子，将仅补充计算 FFD 部分")

    # 3. 获取已上市股票 [A2: 宇宙来自 raw_daily ∪ historical_stocks]
    stock_universe = stock_universe.copy()
    stock_universe['list_date'] = pd.to_datetime(stock_universe['list_date'])
    listed_stocks = stock_universe[
        stock_universe['list_date'] <= trade_date
    ]['code'].tolist()
    if not listed_stocks:
        logger.warning(f"  无已上市股票，跳过")
        return False

    logger.info(f"  已上市股票数量: {len(listed_stocks)}")

    # 4. 构造日线数据 [A4: 400 自然日窗口]
    extended_start = trade_date - timedelta(days=LOAD_WINDOW_DAYS)
    extended_end = trade_date
    stock_data_list = []
    for code in listed_stocks:
        try:
            stock_data = data_provider.get_daily_data(
                code, extended_start, extended_end, reference_date=trade_date
            )
            if stock_data is not None and not stock_data.empty:
                stock_data['ts_code'] = code
                stock_data_list.append(stock_data)
        except Exception as e:
            logger.debug(f"  加载 {code} 失败: {e}")

    if not stock_data_list:
        logger.error(f"  无有效日线数据，跳过")
        return False

    df_daily_snapshot = pd.concat(stock_data_list, ignore_index=True)
    df_daily_snapshot = df_daily_snapshot.drop_duplicates(subset=['ts_code', 'trade_date'])
    df_daily_snapshot = df_daily_snapshot[
        df_daily_snapshot['ts_code'].str.startswith(VALID_STOCK_PREFIXES)
    ]

    # 5. 计算因子（根据是否已有快照，决定是全量还是只算 FFD）
    if existing_snapshot is not None:
        # ---- 只算 FFD ----
        logger.info("  开始增量计算 FFD 因子...")
        df_factors_ffd = calculate_alpha_factors(
            df_daily_snapshot,
            local_manager=None,
            use_parallel=USE_PARALLEL,
            max_workers=MAX_WORKERS,
            reference_date=trade_date,
            save_to_local=False,
            enable_ffd=True,
            ffd_only=True,
            data_base_path=DATA_BASE_PATH,
            rank_mode=RANK_MODE                     # [A1]
        )
        day_data = df_factors_ffd[df_factors_ffd['trade_date'].dt.date == trade_date.date()].copy()
        if day_data.empty:
            logger.warning(f"  未提取到当日 FFD 因子数据")
            return False

        ffd_cols = [c for c in day_data.columns if c.endswith('_FFD')]
        if not ffd_cols:
            logger.warning("  未生成任何 _FFD 列，跳过")
            return False

        # [A5] FFD 列净化
        day_data = _sanitize_snapshot(day_data)

        # 左连接：保留原快照所有行，补充 FFD 列
        merged = existing_snapshot.merge(
            day_data[['ts_code'] + ffd_cols],
            on='ts_code',
            how='left'
        )
        data_provider.save_factor_snapshot(merged, trade_date)
        logger.info(f"  交易日 {trade_date.date()} FFD 因子合并成功，总列数 {len(merged.columns)}")
    else:
        # ---- 全量生成 ----
        df_factors_full = calculate_alpha_factors(
            df_daily_snapshot,
            local_manager=None,
            use_parallel=USE_PARALLEL,
            max_workers=MAX_WORKERS,
            reference_date=trade_date,
            save_to_local=False,
            enable_ffd=True,
            data_base_path=DATA_BASE_PATH,
            rank_mode=RANK_MODE                     # [A1]
        )
        day_data = df_factors_full[df_factors_full['trade_date'].dt.date == trade_date.date()].copy()
        if day_data.empty:
            logger.warning(f"  未提取到当日因子数据")
            return False

        # [A10-20260905] orig/FFD 行合并: calculate_alpha_factors 对 orig 与 FFD 任务分别返回
        # 行集 (orig 行只有 alpha_* 列有值, FFD 行只有 alpha_*_FFD 列有值, 二者按行 concat)。
        # 必须先按 ts_code 合并为一行/股, 否则停牌过滤会把无 tradestatus 的 FFD 行丢弃,
        # 导致快照 _FFD 列全 NaN (已在 2022-08-02 验证快照中实证)。
        ffd_cols = [c for c in day_data.columns if c.endswith('_FFD')]
        if ffd_cols and day_data['ts_code'].duplicated().any():
            orig_alpha = [c for c in day_data.columns
                          if c.startswith('alpha_') and not c.endswith('_FFD')]
            orig_rows = day_data[day_data[orig_alpha].notna().any(axis=1)].drop(columns=ffd_cols)
            ffd_rows = day_data[day_data[ffd_cols].notna().any(axis=1)][['ts_code'] + ffd_cols]
            n_ffd = len(ffd_rows)
            day_data = orig_rows.merge(ffd_rows, on='ts_code', how='left')
            logger.info(f"  [A10] orig/FFD 行合并: orig {len(orig_rows)} 行 + FFD {n_ffd} 行 → {len(day_data)} 行/股")

        # [A11-20260905] 新股/短历史补齐: 因子 worker 有 len(df)<10 → return None 的计算守卫,
        # 上市初期 (窗口内 <10 根 bar) 的股票被整体丢弃, 违反"快照当日全市场非停牌股票全覆盖"
        # 硬性要求 (T7 审计实证: 2022-08-01 遗漏 12 只 7 月末新股, 如 sh.603201/sh.688130)。
        # 此类股票最长因子窗口 250 交易日, 因子值本就全 NaN —— 直接补齐行:
        # 基础行情列 (open/close/volume/tradestatus/isST 等) 取当日 bar, 全部因子列 NaN,
        # 再交由停牌过滤统一裁决 (当日停牌的新股仍被剔除, 口径一致)。
        day_bars = df_daily_snapshot[df_daily_snapshot['trade_date'] == trade_date]
        missing_codes = sorted(set(day_bars['ts_code']) - set(day_data['ts_code']))
        if missing_codes:
            supp = day_bars[day_bars['ts_code'].isin(missing_codes)].copy()
            for c in day_data.columns:
                if c not in supp.columns:
                    supp[c] = np.nan
            day_data = pd.concat([day_data, supp[day_data.columns]], ignore_index=True)
            logger.info(f"  [A11] 新股/短历史补齐: {len(missing_codes)} 只 "
                        f"(因子全NaN, 行情列取当日bar): {missing_codes[:5]}{'...' if len(missing_codes) > 5 else ''}")

        # [A3] 停牌过滤 → [A5] 净化 → 保存
        day_data = _filter_suspended(day_data)
        day_data = _sanitize_snapshot(day_data)

        data_provider.save_factor_snapshot(day_data, trade_date)
        logger.info(f"  交易日 {trade_date.date()} 处理成功，保存 {len(day_data)} 只股票因子")

    return True


def main():
    start_dt = datetime.strptime(START_DATE, '%Y%m%d')
    end_dt = datetime.strptime(END_DATE, '%Y%m%d')

    logger.info("=" * 70)
    logger.info("【[test3] 独立因子快照预计算脚本 - GPSEF适配版】")
    logger.info(f"日期区间: {START_DATE} 至 {END_DATE}")
    logger.info(f"数据目录: {DATA_BASE_PATH}")
    logger.info(f"快照输出: {SNAPSHOT_DIR_NAME} (隔离目录)")
    logger.info(f"rank 模式: {RANK_MODE} | 加载窗口: {LOAD_WINDOW_DAYS} 自然日")
    logger.info(f"跳过已存在快照: {SKIP_EXISTING}")
    logger.info("=" * 70)

    # [A9-20260905] FFD d* 文件启动硬校验
    d_star_file = os.path.join(DATA_BASE_PATH, 'ffd_params', 'd_star_median.txt')
    if not os.path.exists(d_star_file):
        raise FileNotFoundError(
            f"[A9] FFD 参数文件缺失: {d_star_file}\n"
            f"缺失将静默回退 d=0.35 造成跨期 FFD 口径不一致, 请先恢复该文件")
    with open(d_star_file, 'r') as f:
        logger.info(f"[A9] FFD d* = {f.read().strip()} (文件校验通过)")

    cal_df = get_trade_calendar(START_DATE, END_DATE)
    if cal_df.empty:
        logger.error("指定区间内无交易日")
        return
    trade_dates = sorted(cal_df['calendar_date'].tolist())
    trade_dates = [pd.to_datetime(d) for d in trade_dates]
    logger.info(f"共有 {len(trade_dates)} 个交易日需要处理")

    # [A2] 宇宙: raw_daily ∪ historical_stocks
    stock_universe = build_stock_universe(DATA_BASE_PATH)

    extended_start = (start_dt - timedelta(days=LOAD_WINDOW_DAYS)).strftime('%Y%m%d')
    extended_end = end_dt.strftime('%Y%m%d')
    benchmark_df = get_benchmark_data(extended_start, extended_end)
    if benchmark_df is None:
        logger.warning("未能获取基准指数数据，部分依赖指数的因子将无法计算（将置为NaN）")

    data_provider = LocalDataProvider(DATA_BASE_PATH)
    # [隔离] 实例级覆盖快照目录 → factor_snapshots_test (零框架改动)
    data_provider.snapshot_path = os.path.join(DATA_BASE_PATH, SNAPSHOT_DIR_NAME)
    os.makedirs(data_provider.snapshot_path, exist_ok=True)
    logger.info(f"快照写入目录: {data_provider.snapshot_path}")

    success_count = 0
    skip_count = 0
    fail_count = 0
    total_start_time = time.time()

    for idx, trade_date in enumerate(trade_dates):
        logger.info("-" * 50)
        logger.info(f"[{idx+1}/{len(trade_dates)}] 交易日: {trade_date.date()}")

        success = generate_snapshot_for_date(trade_date, stock_universe, data_provider)
        if success:
            success_count += 1
        else:
            fail_count += 1

        gc.collect()

        elapsed = time.time() - total_start_time
        if idx > 0:
            avg_time = elapsed / (idx + 1)
            remaining = avg_time * (len(trade_dates) - idx - 1)
            logger.info(f"进度: {idx+1}/{len(trade_dates)} | 成功: {success_count} | 跳过: {skip_count} | 失败: {fail_count} | 预计剩余: {remaining//60:.0f}分{remaining%60:.0f}秒")
        else:
            logger.info(f"进度: {idx+1}/{len(trade_dates)} | 成功: {success_count} | 跳过: {skip_count} | 失败: {fail_count}")

    total_elapsed = time.time() - total_start_time
    logger.info("=" * 70)
    logger.info("【批量生成完成】")
    logger.info(f"总计交易日: {len(trade_dates)}")
    logger.info(f"成功生成: {success_count}")
    logger.info(f"跳过（已存在）: {skip_count}")
    logger.info(f"失败: {fail_count}")
    logger.info(f"总耗时: {total_elapsed//60:.0f}分{total_elapsed%60:.0f}秒")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
