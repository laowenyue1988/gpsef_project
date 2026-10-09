# -*- coding: utf-8 -*-
"""
中证2000 (932000) 指数日线下载器 (独立脚本, 中证官网数据源)

用途: 中证2000 不在 baostock/掘金/腾讯/新浪, 唯一免费源 = 中证指数官网。
      官网提供 2018-05-26 起的回算历史 (含 2023-08-11 发布前的点位)。
      下载/增量更新 data/index_daily/sh932000.parquet (列: date/open/close/high/low/volume)

用法 (PyCharm Terminal, venv 已激活):
    全量/补最新:   python -u scripts/download_zz2000_index.py
    指定区间:      python -u scripts/download_zz2000_index.py --start 2026-08-01 --end 2026-09-01
    起始截止(完整): python -u scripts/download_zz2000_index.py --start 2018-05-26 --end 2026-09-01

说明: 已存在文件时按 date 增量合并 (同源值一致, keep='last' 幂等), 不会覆盖丢失历史。
"""
import argparse
import os
import io

import pandas as pd

IDX_DIR = 'data/index_daily'
CODE = '932000'


def save_merged(path: str, new_df: pd.DataFrame, out: io.StringIO):
    """与已存在文件按 date 合并去重后写回 (增量安全)。"""
    cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    new_df = new_df[cols].copy()
    if os.path.exists(path):
        old = pd.read_parquet(path)
        old['date'] = pd.to_datetime(old['date'])
        both = pd.concat([old[cols], new_df], ignore_index=True)
        both = both.drop_duplicates(subset='date', keep='last').sort_values('date').reset_index(drop=True)
        out.write(f'[合并] 原 {len(old)} 行 + 新 {len(new_df)} 行 → {len(both)} 行\n')
    else:
        both = new_df.sort_values('date').reset_index(drop=True)
        out.write(f'[新建] {len(both)} 行\n')
    both.to_parquet(path, index=False)
    out.write(f'[OK] {path}: {both["date"].min().date()} ~ {both["date"].max().date()} ({len(both)} 行)\n')


def main():
    ap = argparse.ArgumentParser(description='中证2000 指数日线下载 (中证官网, 存 data/index_daily/sh932000.parquet)')
    ap.add_argument('--start', default='20180526', help='起始日 YYYYMMDD 或 YYYY-MM-DD (默认 20180526=官网回算起点)')
    ap.add_argument('--end', default='20990101', help='结束日 YYYYMMDD 或 YYYY-MM-DD (默认 20990101=拉至最新)')
    args = ap.parse_args()
    start = args.start.replace('-', '')
    end = args.end.replace('-', '')

    out = io.StringIO()
    os.makedirs(IDX_DIR, exist_ok=True)
    try:
        import akshare as ak
    except ImportError:
        out.write('[FAIL] akshare 未安装: pip install akshare\n')
        print(out.getvalue(), flush=True)
        return

    df = ak.stock_zh_index_hist_csindex(symbol=CODE, start_date=start, end_date=end)
    out.write(f'中证官网拉取: {len(df)} 行, {df.iloc[0]["日期"]} ~ {df.iloc[-1]["日期"]}\n')
    if df.empty:
        out.write('[WARN] 区间无数据\n')
        print(out.getvalue(), flush=True)
        return

    rename = {'日期': 'date', '开盘': 'open', '最高': 'high',
              '最低': 'low', '收盘': 'close', '成交量': 'volume'}
    d = df.rename(columns=rename)[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
    d['date'] = pd.to_datetime(d['date'])
    for c in ['open', 'high', 'low', 'close']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.sort_values('date').reset_index(drop=True)

    # 交易日完整性校验 (与上证重叠区间不应缺交易日)
    try:
        sz = pd.read_parquet(os.path.join(IDX_DIR, 'sh000001.parquet'))
        sz['date'] = pd.to_datetime(sz['date'])
        miss = sorted(set(sz['date']) - set(d['date']))
        mid_miss = [x for x in miss if d['date'].min() < x < pd.Timestamp('2026-08-28')]
        out.write(f'与上证重叠区间缺失日: {len(mid_miss)} {str(mid_miss[:5])}\n')
    except Exception as e:
        out.write(f'交易日校验跳过: {str(e)[:60]}\n')

    save_merged(os.path.join(IDX_DIR, 'sh932000.parquet'), d, out)
    res = out.getvalue()
    print(res, flush=True)
    open('results/_download_zz2000_index.log', 'a', encoding='utf-8').write(res + '\n')


if __name__ == '__main__':
    main()
