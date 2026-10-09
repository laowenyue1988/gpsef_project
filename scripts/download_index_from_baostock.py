# -*- coding: utf-8 -*-
"""
baostock 指数日线下载器 (独立脚本)

用途: 下载 baostock 支持的指数历史日线, 存储到 data/index_daily/ 统一位置
      文件名: {code 去掉点}.parquet, 如 sh.000852 -> sh000852.parquet
      列: date/open/close/high/low/volume (与本地 index_daily 现有文件一致)

用法 (PyCharm Terminal, venv 已激活):
    python -u scripts/download_index_from_baostock.py --codes sh.000852,sz.399303 --start 2018-01-01 --end 2026-09-01
    # 只下一个: python -u scripts/download_index_from_baostock.py --codes sh.000905

说明:
    - baostock 支持的指数有限 (上证/深证/沪深300/中证500/中证1000/国证2000/创业板等),
      中证2000(sh.932000) 不在 baostock, 需用 scripts/download_zz2000_index.py (中证官网源)。
    - 默认区间 2018-01-01 ~ 最新, 覆盖 small_alpha191 系列回测窗口。
"""
import argparse
import os
import time
import io

import baostock as bs
import pandas as pd

IDX_DIR = 'data/index_daily'


def download_one(code: str, start: str, end: str) -> pd.DataFrame:
    """下载单指数日线 (不复权=指数点位本身)。"""
    rs = bs.query_history_k_data_plus(
        code, 'date,code,open,high,low,close,preclose,volume,amount',
        start_date=start, end_date=end, frequency='d', adjustflag='3')
    if rs.error_code != '0':
        raise RuntimeError(f'{code} 查询失败: {rs.error_code} {rs.error_msg}')
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'])
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df[['date', 'open', 'high', 'low', 'close', 'volume']].sort_values('date').reset_index(drop=True)


def save_merged(path: str, new_df: pd.DataFrame, out: io.StringIO) -> None:
    """与已存在文件按 date 合并去重后写回 (增量安全, 不覆盖丢失历史)。"""
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
    ap = argparse.ArgumentParser(description='baostock 指数日线下载 (存 data/index_daily/)')
    ap.add_argument('--codes', required=True, help='指数代码, 逗号分隔: sh.000852,sz.399303')
    ap.add_argument('--start', default='2018-01-01', help='起始日 (默认 2018-01-01)')
    ap.add_argument('--end', default='2026-09-01', help='结束日 (默认 2026-09-01)')
    args = ap.parse_args()

    os.makedirs(IDX_DIR, exist_ok=True)
    out = io.StringIO()
    lg = bs.login()
    out.write(f'[baostock] login: {lg.error_code} {lg.error_msg}\n')
    ok = 0
    for code in args.codes.split(','):
        code = code.strip()
        try:
            df = download_one(code, args.start, args.end)
            if df.empty:
                out.write(f'[SKIP] {code}: 区间内无数据\n')
                continue
            fname = code.replace('.', '') + '.parquet'
            path = os.path.join(IDX_DIR, fname)
            save_merged(path, df, out)
            ok += 1
        except Exception as e:
            out.write(f'[FAIL] {code}: {str(e)[:120]}\n')
        time.sleep(1.5)
    try:
        bs.logout()
    except Exception:
        pass
    out.write(f'完成: 成功 {ok}/{len(args.codes.split(","))}\n')
    res = out.getvalue()
    print(res, flush=True)
    open('results/_download_bs_index.log', 'a', encoding='utf-8').write(res + '\n')


if __name__ == '__main__':
    main()
