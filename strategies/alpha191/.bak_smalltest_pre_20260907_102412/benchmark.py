# -*- coding: utf-8 -*-
"""基准数据获取工具（仅本地缓存，禁用网络请求）"""
import os
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DIR = './cache'
os.makedirs(CACHE_DIR, exist_ok=True)


def fetch_benchmark_data(start_date: str, end_date: str, benchmark_code: str = 'sh.000300') -> Optional[pd.DataFrame]:
    """
    获取基准指数数据，仅从本地缓存读取。如果缓存不存在，返回 None。
    网络下载由独立的预热脚本完成，优化进程中不进行任何网络请求。
    """
    cache_file = os.path.join(CACHE_DIR, f'benchmark_{benchmark_code}_{start_date}_{end_date}.parquet')
    if os.path.exists(cache_file):
        logger.debug(f"从缓存加载基准数据: {cache_file}")
        return pd.read_parquet(cache_file)
    else:
        logger.warning(f"基准数据缓存不存在: {cache_file}，返回 None")
        return None