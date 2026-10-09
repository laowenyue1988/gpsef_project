# -*- coding: utf-8 -*-
"""
backfill_minute_data.py — 补齐 5min 分钟线数据（对照日线缺口）

背景（2026-08-31）:
  raw_daily 日线主体从 2018 年起, 但 raw_minute/5min 部分股票从 2024/2025/2026 才开始
  (如 sz.301116 5min 仅 2026-05-22 起) → 回测区间内分钟撮合/止盈止损缺数据。
  已审计: 回测区间 [2022-08-01 ~ 2026-08-04] 内 984 只股票分钟数据缺早期段。

用法（PyCharm 终端, 需先启动掘金终端 127.0.0.1:7001）:
  python -u scripts/backfill_minute_data.py                    # 全量补回测区间
  python -u scripts/backfill_minute_data.py --start 2022-08-01 --end 2026-08-04
  python -u scripts/backfill_minute_data.py --limit 5          # 只处理前 5 只(测试)

说明:
  - 数据源: 掘金 SDK gm.api.history(frequency='5m'), 按月循环下载(规避单次条数限制)
  - 合并: 与现有 parquet 去重合并(drop_duplicates), 列格式一致
  - 断点续跑: 每只处理完写 results/backfill_minute_progress.txt, 重启跳过已完成
"""
import argparse
import glob
import json
import os
import sys
import time
import traceback

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY_DIR = os.path.join(ROOT, "data", "raw_daily")
MIN5_DIR = os.path.join(ROOT, "data", "raw_minute", "5min")
PROGRESS = os.path.join(ROOT, "results", "backfill_minute_progress.txt")
GAP_JSON = os.path.join(ROOT, "results", "minute_gap_analysis.json")

# 掘金 token（与 run_select.py 同源）
GM_TOKEN = "177c16ceb2ea308dca384cda74a9386b1afb8376"


def load_progress():
    done = set()
    if os.path.exists(PROGRESS):
        for line in open(PROGRESS, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#"):
                done.add(line.split()[0])
    return done


def save_progress(code):
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(f"{code} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def find_gap_stocks(start="2022-08-01", end="2026-08-04", limit=0):
    """扫描日线 vs 分钟线, 返回回测区间内分钟缺失的股票 (code, 下载起点, 下载终点, 已有末日)"""
    daily_range = {}
    for f in glob.glob(os.path.join(DAILY_DIR, "*.parquet")):
        code = os.path.basename(f).replace(".parquet", "")
        try:
            df = pd.read_parquet(f, columns=["trade_date"])
            daily_range[code] = (df["trade_date"].min(), df["trade_date"].max())
        except Exception:
            pass

    min5_range = {}
    for f in glob.glob(os.path.join(MIN5_DIR, "*.parquet")):
        fname = os.path.basename(f).replace(".parquet", "")
        code = fname.replace("_", ".", 1)  # sh_600000 -> sh.600000
        try:
            df = pd.read_parquet(f, columns=["trade_date"])
            min5_range[code] = (df["trade_date"].min(), df["trade_date"].max())
        except Exception:
            pass

    START = pd.Timestamp(start)
    END = pd.Timestamp(end)
    gaps = []
    for fname, (dmin, dmax) in sorted(daily_range.items()):
        code = fname.replace("_", ".", 1)  # sh_601279 -> sh.601279 (与 ts_code 列一致)
        dmin, dmax = pd.Timestamp(dmin), pd.Timestamp(dmax)
        # 回测区间与日线区间的交集
        dl = max(START, dmin)
        dr = min(END, dmax)
        if dl > dr:
            continue  # 日线不覆盖回测区间
        if code in min5_range:
            mmin, mmax = pd.Timestamp(min5_range[code][0]), pd.Timestamp(min5_range[code][1])
            if mmin <= dl and mmax >= dr:
                continue  # 分钟已完整覆盖回测区间
            fetch_l = max(dl, mmin + pd.Timedelta(days=1))  # 已有末日次日开始
            if fetch_l > dr:
                continue
            gaps.append((code, fetch_l, dr, mmin))
        else:
            gaps.append((code, dl, dr, None))  # 完全无分钟
    if limit:
        gaps = gaps[:limit]
    return gaps


def fetch_month(symbol, y, m, fields):
    """下载单月 5min (返回 DataFrame, 可能为空)"""
    from gm.api import history
    month_end = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
    s = pd.Timestamp(y, m, 1).strftime("%Y-%m-%d")
    e = month_end.strftime("%Y-%m-%d")
    df = history(symbol=symbol, frequency="5m", start_time=s, end_time=e,
                 fields=fields, df=True)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    return df


def gm_symbol(code):
    """sh.601279 -> SHSE.601279; sz.301116 -> SZSE.301116"""
    pre, num = code.split(".")
    return ("SHSE" if pre == "sh" else "SZSE") + "." + num


def to_std(df, code):
    """掘金 history 返回 -> 现有 parquet 格式"""
    out = pd.DataFrame({
        "trade_time": pd.to_datetime(df["eob"]),
        "trade_date": pd.to_datetime(df["eob"]).dt.normalize(),
        "ts_code": code,
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume": df["volume"].astype(float).astype("int64"),
        "amount": df["amount"].astype(float),
    })
    return out.sort_values("trade_time").reset_index(drop=True)


def backfill(code, fetch_l, fetch_r, fields):
    """下载 [fetch_l, fetch_r] 缺失段并合并写入文件"""
    fpath = os.path.join(MIN5_DIR, code.replace(".", "_") + ".parquet")
    old = pd.read_parquet(fpath) if os.path.exists(fpath) else None
    symbol = gm_symbol(code)

    pieces = []
    cur = fetch_l
    month = 0
    while cur <= fetch_r:
        y, m = cur.year, cur.month
        try:
            df = fetch_month(symbol, y, m, fields)
            if len(df):
                pieces.append(to_std(df, code))
            month += 1
        except Exception as e:
            print(f"    [{code}] {y}-{m:02d} 下载失败: {e}")
            time.sleep(2)
            return False  # 留待下次续跑
        # 月末推进
        cur = (pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1))
        time.sleep(0.3)  # 限速, 避免触发风控

    if not pieces:
        print(f"    [{code}] 无新数据")
        return True
    new = pd.concat(pieces, ignore_index=True)
    merged = pd.concat([old, new], ignore_index=True) if old is not None else new
    merged = merged.drop_duplicates(subset=["trade_time"]).sort_values("trade_time").reset_index(drop=True)
    merged.to_parquet(fpath, index=False)
    print(f"    [{code}] 补齐 {month} 个月, {len(new)} 行 → 共 {len(merged)} 行 "
          f"({merged['trade_date'].min().date()} ~ {merged['trade_date'].max().date()})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-08-01")
    ap.add_argument("--end", default="2026-08-04")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 只(测试)")
    args = ap.parse_args()

    os.environ["GM_TOKEN"] = GM_TOKEN
    from gm.api import set_token
    set_token(GM_TOKEN)

    gaps = find_gap_stocks(args.start, args.end, args.limit)
    print(f"需补分钟数据: {len(gaps)} 只 (区间 {args.start} ~ {args.end})")
    if not gaps:
        print("无缺口, 完成")
        return

    done = load_progress()
    fields = "open,high,low,close,volume,amount"
    ok = fail = skip = 0
    for code, fetch_l, fetch_r, mmin in gaps:
        if code in done:
            skip += 1
            continue
        print(f"[{skip+ok+fail+1}/{len(gaps)}] {code}: 下载 {fetch_l.date()} ~ {fetch_r.date()} "
              f"(分钟现从 {mmin.date() if mmin else '无'})")
        try:
            if backfill(code, fetch_l, fetch_r, fields):
                save_progress(code)
                ok += 1
            else:
                fail += 1
        except Exception as e:
            traceback.print_exc()
            fail += 1
            break  # 单只失败先停, 避免空转

    print(f"\n完成: 成功 {ok} / 跳过(已补) {skip} / 失败 {fail}")
    print(f"进度记录: {PROGRESS} (断点续跑: 直接重跑本命令)")


if __name__ == "__main__":
    main()
