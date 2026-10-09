# -*- coding: utf-8 -*-
"""
export_signals.py — 从 gpsef 回测结果导出"调仓信号文件"（离线 → 实盘 的桥）

设计意图（用户 2026-08-31 三问确认后的协议 v3）:
  gpsef(离线) 负责训练/选股/风控 → 把每个调仓日的 target(股票+权重+σ+scale)
  写成 JSON 信号文件; Eureka(实盘) 读取信号 → 按开盘价执行建仓/调仓 +
  用信号 σ 做实时止盈止损(规则, 不训练)。

信号协议 v3 要点（与 v2 的差异）:
  1. 【σ 一致性】信号加入 sigma（离线 _compute_entry_vol 同口径: rolling/ewma
     daily vol）——实盘直接用信号 σ 建 tracker, 保证止盈止损触发点与离线一致。
  2. 【不发价格】删除 ref_price——离线盘前只生成目标与资金分配, 买入价是
     开盘后实盘读取的实际开盘价, 发参考价会误导股数估算。
  3. 【资金分配】新增 scale（调仓日 build_scale = 回撤熔断×趋势过滤）——
     实盘按 target_value = 总资产 × weight × scale 分配资金。
  4. 【除权信号】新增 corporate_action 信号类型（离线检测到持仓股票除权后,
     实盘收到后仅同步股数 qty, σ/阈值不动——前复权价连续性保证）。

用法:
  .venv/Scripts/python.exe scripts/export_signals.py \
      --orders results/<回测>/orders.csv \
      --log results/<回测>/backtest.log \
      --config strategies/small_alpha191_test2/config.yaml \
      --out E:/quant/Eureka/signals
"""
import argparse
import json
import math
import os
import re

import pandas as pd


# ─────────────── 波动率计算（与策略 _compute_entry_vol 同口径） ───────────────

def compute_entry_vol(dp, code, date, config):
    """复现策略 _compute_entry_vol: rolling std / EWMA (λ=0.94, seed=60d)。

    参数取自 config['meta_labeling'] 段; 数据用 gpsef LocalDataProvider。
    """
    meta = config.get('meta_labeling', {})
    vol_method = meta.get('vol_method', 'rolling')
    daily_vol_window = int(meta.get('daily_vol_window', 20))
    vol_ewma_lambda = float(meta.get('vol_ewma_lambda', 0.94))
    vol_ewma_seed_days = int(meta.get('vol_ewma_seed_days', 60))
    lookback = max(daily_vol_window, vol_ewma_seed_days)
    ref = config.get('data', {}).get('reference_date')
    if ref:
        try:
            ref = pd.Timestamp(str(ref))
        except Exception:
            ref = None
    try:
        hist = dp.get_daily_data(code, pd.Timestamp(date) - pd.Timedelta(days=lookback * 2),
                                 pd.Timestamp(date), reference_date=ref)
    except Exception:
        return 0.02
    if hist is None or hist.empty:
        return 0.02
    if 'raw_close' in hist.columns and 'backAdjustFactor' in hist.columns:
        adj = hist['raw_close'] * hist['backAdjustFactor']
        rets = adj.pct_change().dropna()
    else:
        rets = hist['close'].pct_change().dropna()
    if len(rets) < 2:
        return 0.02
    if vol_method == 'ewma':
        ev = rets.ewm(alpha=1 - vol_ewma_lambda, adjust=False).var().iloc[-1]
        return float(math.sqrt(ev)) if ev > 0 and not math.isnan(ev) else 0.02
    dv = rets.std()
    return float(dv) if not (math.isnan(dv) or dv == 0) else 0.02


# ─────────────── 调仓日 scale 提取（从回测日志） ───────────────

def parse_scale_by_date(log_path):
    """解析 backtest.log 的"建仓规模 xS (回撤熔断=X, 趋势=Y)" → {date: scale}"""
    result = {}
    if not log_path or not os.path.exists(log_path):
        return result
    pat = re.compile(
        r'\[(\d{4}-\d{2}-\d{2})\] \[方案A\] 建仓规模 x(\d+\.?\d*) '
        r'\(回撤熔断=(\d+\.?\d*), 趋势=(\d+\.?\d*)\)')
    with open(log_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pat.search(line)
            if m:
                result[m.group(1)] = {
                    'scale': float(m.group(2)),
                    'dd_scale': float(m.group(3)),
                    'trend_scale': float(m.group(4)),
                }
    return result


# ─────────────── 除权信号提取（从回测日志） ───────────────

CA_TYPES = {"现金分红": "cash", "送股": "stock", "转增": "stock",
            "现金分红+送股": "both", "现金分红+转增": "both",
            "送股+转增": "stock", "现金分红+送股+转增": "both"}


def parse_corporate_actions(log_path):
    """解析 backtest.log 的除权事件块 → 除权信号列表。

    日志格式（引擎 on_ex_dividend 输出）:
      ========== [YYYY-MM-DD] CODE 除权事件 ==========
        类型: 现金分红
        公告: 10派2.2元（含税...）
        除权参考价: 5.58
        股数: 4100 → 4100 (不变)  |  4100 → 5330 (送转)
        现金分红: 4100股 × 0.2200元 = 902.00(税前)

    返回: [{type, code, ex_date, stock_ratio, cash_per_share, ex_factor}]
    """
    result = []
    if not log_path or not os.path.exists(log_path):
        return result
    lines = open(log_path, encoding="utf-8", errors="ignore").read().splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = re.search(r'========== \[(\d{4}-\d{2}-\d{2})\] (\S+) 除权事件', line)
        if not m:
            i += 1
            continue
        ex_date, code = m.group(1), m.group(2)
        # 读取后续块内容
        block = []
        j = i + 1
        while j < n and not lines[j].startswith("=" * 5):
            block.append(lines[j])
            j += 1
        i = j
        blk = "\n".join(block)
        # 类型
        mt = re.search(r"类型:\s*(\S+)", blk)
        if not mt:
            continue
        # 公告解析: 10派2.2元 / 10送3转7
        note = ""
        mn = re.search(r"公告:\s*(.+)", blk)
        if mn:
            note = mn.group(1).strip()
        # stock_ratio: 送转比例 (送X转Y → (X+Y)/10)
        stock_ratio = 0.0
        ms = re.search(r"送(\d+)", note)
        mz = re.search(r"转(\d+)", note)
        if ms or mz:
            stock_ratio = ((int(ms.group(1)) if ms else 0)
                           + (int(mz.group(1)) if mz else 0)) / 10.0
        # cash_per_share: 派X元 → X/10
        cash_ps = 0.0
        mc = re.search(r"派([\d.]+)元", note)
        if mc:
            cash_ps = float(mc.group(1)) / 10.0
        # ex_factor: 除权参考价/原始收盘价
        ex_factor = 1.0
        mp = re.search(r"原始收盘价:\s*([\d.]+) → ([\d.]+)", blk)
        me = re.search(r"除权参考价:\s*([\d.]+)", blk)
        if me and mp:
            try:
                ex_factor = float(me.group(1)) / float(mp.group(1))
            except (ValueError, ZeroDivisionError):
                ex_factor = 1.0
        if stock_ratio <= 0 and cash_ps <= 0:
            continue
        result.append({
            "type": "corporate_action",
            "code": code,
            "ex_date": ex_date,
            "stock_ratio": round(stock_ratio, 6),
            "cash_per_share": round(cash_ps, 6),
            "bonus_ps": 0.0,
            "ex_factor": round(ex_factor, 6),
            "note": note,
        })
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", required=True, help="回测 orders.csv 路径")
    ap.add_argument("--log", default="", help="回测 backtest.log 路径(提取 scale)")
    ap.add_argument("--config", default=r"strategies/small_alpha191_test2/config.yaml",
                    help="gpsef 策略 config.yaml (波动率参数/数据路径)")
    ap.add_argument("--out", required=True, help="信号输出目录")
    ap.add_argument("--cash", type=float, default=1_000_000.0, help="初始资金(仅参考)")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import yaml
    from gpsef.data.local_provider import LocalDataProvider

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    dp = LocalDataProvider(config.get('data', {}).get('base_path', 'data'))

    o = pd.read_csv(args.orders)
    o["date"] = pd.to_datetime(o["date"])
    buys = o[(o["direction"] == "buy") & (o["trade_reason"] == "new_open")]

    scales = parse_scale_by_date(args.log)

    os.makedirs(args.out, exist_ok=True)
    dates = sorted(buys["date"].unique())
    n = 0
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        day = buys[buys["date"] == d]
        total = day["value"].sum()
        targets = []
        for _, r in day.iterrows():
            code = r["code"]
            sigma = compute_entry_vol(dp, code, d.date(), config)
            targets.append({
                "code": code,
                "weight": round(r["value"] / total, 6),   # 目标权重(等权 ≈ 1/20)
                "sigma": round(sigma, 6),                 # 入场 daily vol (离线同口径)
            })
        scale = scales.get(ds, {}).get("scale", 1.0)
        payload = {
            "signal_date": ds,
            "strategy": "small_alpha191_test2",
            "target_count": len(targets),
            "scale": scale,                     # 调仓日 build_scale (回撤熔断×趋势过滤)
            "targets": targets,
            "note": "由 gpsef R12 回测导出 (协议v3: weight+sigma+scale, 无参考价)",
        }
        fn = os.path.join(args.out, d.strftime("%Y%m%d") + ".json")
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        n += 1
    print(f"已导出 {n} 个调仓日信号 → {args.out}")
    print(f"区间: {dates[0].date()} ~ {dates[-1].date()}")
    print(f"每期股票数: {buys.groupby('date').size().unique().tolist()}")
    if scales:
        s0 = scales.get(dates[0].strftime("%Y-%m-%d"))
        if s0:
            print(f"首期 scale: {s0['scale']} (dd={s0['dd_scale']}, trend={s0['trend_scale']})")
    # σ 抽样展示
    ex = json.load(open(os.path.join(args.out, dates[0].strftime("%Y%m%d") + ".json"),
                        encoding="utf-8"))
    print(f"首期 σ 示例: {[(t['code'], t['sigma']) for t in ex['targets'][:3]]}")

    # ── 除权信号导出（从回测日志提取真实除权事件） ──
    if args.log and os.path.exists(args.log):
        cas = parse_corporate_actions(args.log)
        n_ca = 0
        for ca in cas:
            # 文件名 = 除权日; 多个同日除权用 _2/_3 后缀避免覆盖
            fn = os.path.join(args.out, ca["ex_date"].replace("-", "") + ".json")
            k = 2
            while os.path.exists(fn):
                fn = os.path.join(args.out,
                                  ca["ex_date"].replace("-", "") + f"_{k}.json")
                k += 1
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(ca, f, ensure_ascii=False, indent=2)
            n_ca += 1
        print(f"已导出 {n_ca} 个除权信号 (送转/分红事件)")
        if cas:
            s = cas[0]
            print(f"除权示例: {s['code']} {s['ex_date']} "
                  f"stock_ratio={s['stock_ratio']} cash_ps={s['cash_per_share']} "
                  f"ex_factor={s['ex_factor']}")


if __name__ == "__main__":
    main()
