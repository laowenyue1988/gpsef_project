# strategies/fundamental_ml/data_utils.py
import os
import logging
import pandas as pd
from typing import Dict

logger = logging.getLogger(__name__)
INDUSTRY_MAP_FILE = os.path.join('data', 'industry_map.parquet')


def load_or_build_industry_map() -> Dict[str, str]:
    """加载或创建行业映射 {股票数字代码: 行业名称}"""
    if os.path.exists(INDUSTRY_MAP_FILE):
        df = pd.read_parquet(INDUSTRY_MAP_FILE)
        return dict(zip(df['code'], df['industry']))

    perf_path = os.path.join('data', 'performance_reports', '20251231.parquet')
    if not os.path.exists(perf_path):
        logger.error("行业映射构建失败：缺少业绩报表文件")
        return {}
    df = pd.read_parquet(perf_path)
    if 'code' not in df.columns:
        return {}
    ind_col = '所处行业'
    if ind_col not in df.columns:
        return {}
    df_map = df[['code', ind_col]].drop_duplicates().rename(columns={ind_col: 'industry'})
    df_map.to_parquet(INDUSTRY_MAP_FILE, index=False)
    return dict(zip(df_map['code'], df_map['industry']))