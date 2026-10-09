#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基准数据缓存预热脚本，在运行优化前执行一次"""
import os
import pandas as pd
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_benchmark(start_date: str, end_date: str, benchmark_code: str = 'sh.000300'):
    try:
        import baostock as bs
    except ImportError:
        logger.error("baostock未安装")
        return

    lg = bs.login()
    if lg.error_code != '0':
        logger.error(f"登录失败: {lg.error_msg}")
        return

    if len(start_date) == 8:
        start_fmt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    else:
        start_fmt = start_date
    if len(end_date) == 8:
        end_fmt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
    else:
        end_fmt = end_date

    rs = bs.query_history_k_data_plus(
        benchmark_code,
        "date,code,open,high,low,close,preclose,volume,amount,pctChg",
        start_date=start_fmt, end_date=end_fmt, frequency="d"
    )
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())

    bs.logout()

    if data_list:
        df = pd.DataFrame(data_list, columns=rs.fields)
        df['date'] = pd.to_datetime(df['date'])
        numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'pctChg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.rename(columns={'date': 'trade_date'})

        os.makedirs('./cache', exist_ok=True)
        cache_file = f"./cache/benchmark_{benchmark_code}_{start_date}_{end_date}.parquet"
        df.to_parquet(cache_file, index=False)
        logger.info(f"基准数据已缓存至: {cache_file}")
    else:
        logger.warning("未获取到基准数据")


if __name__ == '__main__':
    with open('strategies/alpha191/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    backtest_cfg = config['backtest']
    download_benchmark(backtest_cfg['start_date'], backtest_cfg['end_date'])