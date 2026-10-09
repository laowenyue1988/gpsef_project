# -*- coding: utf-8 -*-
"""
独立因子快照预计算脚本（GPSEF 框架适配版）
功能：输入日期区间，批量生成区间内每个交易日的全市场股票因子快照。
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
from strategies.alpha191.data_utils import get_trade_calendar, VALID_STOCK_PREFIXES
from strategies.alpha191.factors import calculate_alpha_factors

# ==================== 全局变量配置 ====================
START_DATE = '20200101'
END_DATE = '20260901'
DATA_BASE_PATH = r'E:\quant\gpsef_project\data'
BENCHMARK_CODE = 'sh.000300'
FACTOR_BATCH_SIZE = 800
USE_PARALLEL = True
SKIP_EXISTING = True
# ================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_stock_basic_info(data_dir: str) -> pd.DataFrame:
    """加载股票基本信息（上市日期等）"""
    stock_list_file = os.path.join(data_dir, 'stock_list', 'historical_stocks.parquet')
    if not os.path.exists(stock_list_file):
        raise FileNotFoundError(f"股票列表文件不存在: {stock_list_file}\n请先运行 download_data.py 下载股票列表")
    df = pd.read_parquet(stock_list_file)
    df = df[df['code'].str.startswith(VALID_STOCK_PREFIXES)]
    if 'list_date' not in df.columns:
        raw_dir = os.path.join(data_dir, 'raw_daily')
        list_dates = {}
        for f in os.listdir(raw_dir):
            if f.endswith('.parquet'):
                code = f.replace('.parquet', '').replace('_', '.')
                try:
                    df_raw = pd.read_parquet(os.path.join(raw_dir, f), columns=['trade_date'])
                    if not df_raw.empty:
                        list_dates[code] = df_raw['trade_date'].min()
                except:
                    continue
        df['list_date'] = df['code'].map(list_dates)
        df = df.dropna(subset=['list_date'])
    logger.info(f"加载股票列表: {len(df)} 只")
    return df


def get_benchmark_data(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """获取基准指数数据"""
    try:
        from strategies.alpha191.benchmark import fetch_benchmark_data
        return fetch_benchmark_data(start_date, end_date, BENCHMARK_CODE)
    except Exception as e:
        logger.warning(f"获取基准数据失败: {e}")
        return None


def _merge_factor_rows(df: pd.DataFrame) -> pd.DataFrame:
    """[2026-08-06 修复] 每股合并为 1 行：alpha 因子 + FFD 因子同存。

    背景：calculate_alpha_factors 对每只股票生成 2 个任务（orig 原始因子 + ffd FFD 因子），
    两任务输出 concat 后同日期出现 2 行/股：
      - orig 行：191 个 alpha 列有值，_FFD 列全 NaN；
      - ffd 行 ：191 个 alpha 列全 NaN（concat 列对齐副产物），_FFD 列有值。
    修复前快照把 2 行都保存，导致训练/预测每股票 2 行（一行 alpha 全 NULL），
    且 orig 行的 _FFD 列为 NaN → alpha_xxx_FFD_neu 特征在 orig 行缺失（FFD 因子实际被
    模型使用，见 Top-5 MDA alpha_038_FFD_neu，不能丢弃）。

    本函数：按 (ts_code, trade_date) 分组合并，每列取组内**第一个非 NaN 值**（原值不重算）：
      - alpha 列取 orig 行值（非 NaN）；
      - _FFD 列取 ffd 行值（orig 行 NaN，ffd 行有值）；
      - close/volume 等基础列两行同值，取任意行。
    结果：每股 1 行，alpha + FFD 因子值**与修复前完全一致**（仅行重组）。

    实现：组内 transform('bfill')（C 级向量化，~0.5s/全市场，比 groupby.agg(lambda)
    快 ~300 倍）+ 取每组首行；与逐组 lambda 语义逐单元格一致（实测 0 差异）。
    """
    if df is None or df.empty:
        return df
    if 'ts_code' not in df.columns:
        return df
    key_cols = ['ts_code']
    if 'trade_date' in df.columns:
        key_cols.append('trade_date')
    df = df.sort_values(key_cols).reset_index(drop=True)
    num_cols = [c for c in df.columns if c not in key_cols]
    if not num_cols:
        return df.groupby(key_cols, as_index=False).first()
    # 组内 bfill: orig 行(前)的 _FFD NaN 被 ffd 行(后)的值向上填充; alpha 列 orig 行本有值不受影响
    filled = df.groupby(key_cols)[num_cols].transform('bfill')
    merged = pd.concat([df[key_cols].reset_index(drop=True), filled.reset_index(drop=True)],
                       axis=1)
    merged = merged.groupby(key_cols, as_index=False).first()
    # 恢复原始列序
    merged = merged[df.columns.tolist()]
    return merged


def generate_snapshot_for_date(trade_date: datetime,
                               stock_basic_info: pd.DataFrame,
                               data_provider: LocalDataProvider) -> bool:
    date_str = trade_date.strftime('%Y%m%d')
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

    # 3. 获取已上市股票
    stock_basic_info['list_date'] = pd.to_datetime(stock_basic_info['list_date'])
    listed_stocks = stock_basic_info[
        stock_basic_info['list_date'] <= trade_date
    ]['code'].tolist()
    if not listed_stocks:
        logger.warning(f"  无已上市股票，跳过")
        return False

    logger.info(f"  已上市股票数量: {len(listed_stocks)}")

    # 4. 构造日线数据
    extended_start = trade_date - timedelta(days=365)
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
            reference_date=trade_date,
            save_to_local=False,
            enable_ffd=True,
            ffd_only=True,
            data_base_path=DATA_BASE_PATH
        )
        day_data = df_factors_ffd[df_factors_ffd['trade_date'].dt.date == trade_date.date()].copy()
        if day_data.empty:
            logger.warning(f"  未提取到当日 FFD 因子数据")
            return False

        ffd_cols = [c for c in day_data.columns if c.endswith('_FFD')]
        if not ffd_cols:
            logger.warning("  未生成任何 _FFD 列，跳过")
            return False

        # 左连接：保留原快照所有行，补充 FFD 列
        merged = existing_snapshot.merge(
            day_data[['ts_code'] + ffd_cols],
            on='ts_code',
            how='left'
        )
        # [2026-08-06 修复] 每股合并为 1 行 (alpha + FFD 同存, 兼容旧 2 行快照)
        merged = _merge_factor_rows(merged)
        data_provider.save_factor_snapshot(merged, trade_date)
        logger.info(f"  交易日 {trade_date.date()} FFD 因子合并成功，总列数 {len(merged.columns)}，"
                    f"每股 1 行 ({len(merged)} 行, alpha+FFD 同存)")
    else:
        # ---- 全量生成 ----
        df_factors_full = calculate_alpha_factors(
            df_daily_snapshot,
            local_manager=None,
            use_parallel=USE_PARALLEL,
            reference_date=trade_date,
            save_to_local=False,
            enable_ffd=True,
            data_base_path=DATA_BASE_PATH
        )
        day_data = df_factors_full[df_factors_full['trade_date'].dt.date == trade_date.date()].copy()
        if day_data.empty:
            logger.warning(f"  未提取到当日因子数据")
            return False
        # [2026-08-06 修复] 每股合并为 1 行 (alpha + FFD 同存)
        # 修复前 calculate_alpha_factors 双任务(orig+ffd)输出 concat 后每股 2 行:
        #   orig 行有 191 alpha 但 _FFD 全 NaN; ffd 行 alpha 全 NaN 但有 _FFD 值
        #  → 快照 2 行/股 (一行 alpha 全 NULL), 训练/预测样本翻倍且 alpha_xxx_FFD_neu 特征缺失
        day_data = _merge_factor_rows(day_data)
        data_provider.save_factor_snapshot(day_data, trade_date)
        logger.info(f"  交易日 {trade_date.date()} 处理成功，保存 {len(day_data)} 只股票因子 "
                    f"(每股 1 行, alpha+FFD 同存)")

    return True


def main():
    start_dt = datetime.strptime(START_DATE, '%Y%m%d')
    end_dt = datetime.strptime(END_DATE, '%Y%m%d')

    logger.info("=" * 70)
    logger.info("【独立因子快照预计算脚本 - GPSEF适配版】")
    logger.info(f"日期区间: {START_DATE} 至 {END_DATE}")
    logger.info(f"数据目录: {DATA_BASE_PATH}")
    logger.info(f"跳过已存在快照: {SKIP_EXISTING}")
    logger.info("=" * 70)

    cal_df = get_trade_calendar(START_DATE, END_DATE)
    if cal_df.empty:
        logger.error("指定区间内无交易日")
        return
    trade_dates = sorted(cal_df['calendar_date'].tolist())
    trade_dates = [pd.to_datetime(d) for d in trade_dates]
    logger.info(f"共有 {len(trade_dates)} 个交易日需要处理")

    stock_basic_info = load_stock_basic_info(DATA_BASE_PATH)

    extended_start = (start_dt - timedelta(days=400)).strftime('%Y%m%d')
    extended_end = end_dt.strftime('%Y%m%d')
    benchmark_df = get_benchmark_data(extended_start, extended_end)
    if benchmark_df is None:
        logger.warning("未能获取基准指数数据，部分依赖指数的因子将无法计算（将置为NaN）")

    data_provider = LocalDataProvider(DATA_BASE_PATH)

    success_count = 0
    skip_count = 0
    fail_count = 0
    total_start_time = time.time()

    for idx, trade_date in enumerate(trade_dates):
        logger.info("-" * 50)
        logger.info(f"[{idx+1}/{len(trade_dates)}] 交易日: {trade_date.date()}")

        success = generate_snapshot_for_date(trade_date, stock_basic_info, data_provider)
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