# -*- coding: utf-8 -*-
"""更新沪深300基准指数数据 (data/raw_daily/sh_000300.parquet + data/index_daily/sh000300.parquet)
机制: 本地文件存在则引擎直接使用(从不自动增量), 本脚本用 baostock 拉取本地末日之后的数据增量合并。
用法: .venv/Scripts/python.exe scripts/update_benchmark_index.py [index_code]
"""
import os
import sys

import pandas as pd

sys.path.insert(0, '.')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

INDEX = sys.argv[1] if len(sys.argv) > 1 else 'sh.000300'
RAW_P = os.path.join(ROOT, 'data', 'raw_daily', INDEX.replace('.', '_') + '.parquet')
# index_daily 目录文件名无下划线 (sh000300.parquet)
IDX_P = os.path.join(ROOT, 'data', 'index_daily', INDEX.replace('.', '') + '.parquet')


def fetch_baostock(code, start, end):
    """baostock 拉取日线, 返回 DataFrame(date, code, open, high, low, close, preclose, volume, amount, pctChg)。"""
    import baostock as bs
    from strategies.alpha191.data_utils import get_baostock_session
    session = get_baostock_session()
    session.ensure_logged_in()
    rs = bs.query_history_k_data_plus(
        code, "date,code,open,high,low,close,preclose,volume,amount,pctChg",
        start_date=start, end_date=end, frequency="d")
    rows = []
    while rs.error_code == '0' and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=rs.fields)
    df['date'] = pd.to_datetime(df['date'])
    for c in ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'pctChg']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def main():
    if not os.path.exists(RAW_P):
        print(f'raw_daily 文件不存在: {RAW_P} → 仅更新 index_daily')
    last_date = None
    if os.path.exists(RAW_P):
        old = pd.read_parquet(RAW_P)
        old['trade_date'] = pd.to_datetime(old['trade_date'])
        last_date = old['trade_date'].max().date()
    elif os.path.exists(IDX_P):
        d2 = pd.read_parquet(IDX_P)
        dc = 'trade_date' if 'trade_date' in d2.columns else 'date'
        last_date = pd.to_datetime(d2[dc]).max().date()
    else:
        print('本地无任何基准数据文件, 请先确认数据')
        sys.exit(1)
    today = pd.Timestamp.now().date()
    print(f'本地截止: {last_date}, 目标: {today}')
    if last_date >= today:
        print('已是最新, 无需更新')
        return

    start = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    print(f'拉取 {start} ~ {today} ...')
    new = fetch_baostock(INDEX, start, today.strftime('%Y-%m-%d'))
    if new.empty:
        print('baostock 无新数据(可能非交易日或无数据)')
        return
    print(f'baostock 返回 {len(new)} 行, {new["date"].min().date()} ~ {new["date"].max().date()}')

    # ── 转 raw_daily 格式 (若文件存在) ──
    if os.path.exists(RAW_P):
        old = pd.read_parquet(RAW_P)
        old['trade_date'] = pd.to_datetime(old['trade_date'])
        new = new.rename(columns={'date': 'trade_date'})
        new['trade_date'] = pd.to_datetime(new['trade_date'])
        new['ts_code'] = INDEX
        new['turn'] = float('nan')
        new['tradestatus'] = 1
        new['isST'] = 0
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
        merged.to_parquet(RAW_P, index=False)
        print(f'raw_daily 更新: {len(merged)} 行, 截止 {merged["trade_date"].max().date()}')

    # ── index_daily 简单格式 ──
    if os.path.exists(IDX_P):
        d2 = pd.read_parquet(IDX_P)
        dc = 'trade_date' if 'trade_date' in d2.columns else 'date'
        d2[dc] = pd.to_datetime(d2[dc])
        for c in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            if c in d2.columns:
                d2[c] = pd.to_numeric(d2[c], errors='coerce')
        new2 = new[['date', 'open', 'high', 'low', 'close', 'volume']].rename(columns={'date': dc})
        merged2 = pd.concat([d2, new2], ignore_index=True)
        merged2 = merged2.drop_duplicates(subset=[dc]).sort_values(dc).reset_index(drop=True)
        merged2.to_parquet(IDX_P, index=False)
        print(f'index_daily 更新: {len(merged2)} 行, 截止 {pd.to_datetime(merged2[dc]).max().date()}')
    print('DONE')


if __name__ == '__main__':
    main()
