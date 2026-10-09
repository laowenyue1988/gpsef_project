# -*- coding: utf-8 -*-
"""
指数成分股缓存管理工具 (index_stocks_cache.parquet)
====================================================

背景
----
`data/index_stocks_cache.parquet` 由旧框架 `E:\\quant\\quant\\alpha191\\modules\\data_manager.py`
的 `get_multi_index_stocks_by_akshare()` 通过掘金量化 SDK 生成:
    stk_get_index_constituents(index='SHSE.000300'/'SHSE.000905'/'SHSE.000852',
                               trade_date=query_date_str)
获取指定交易日收盘后指数成分股, 过滤 VALID_STOCK_PREFIXES 后转 'sh.xxx'/'sz.xxx' 格式,
由 `_save_index_stocks_cache()` 落盘。生成时间 2026-05-08 (旧框架 01:14 / gpsef 15:05)。

已知问题
--------
gpsef 版缓存 (900 条) 相比旧框架版 (1317 条) 缺少 2021-04-06 ~ 2023-04-28 段(含 2022 全年),
即生成时该段掘金拉取失败未重试。旧框架版 2022 完整且共有日期名单与 gpsef 版 100% 一致,
因此可用旧框架版合并补齐。

用法
----
  python scripts/manage_index_cache.py merge    # 合并旧框架副本, 补齐缺口 (自动备份)
  python scripts/manage_index_cache.py verify   # 校验: 缺口/数量/相邻变化率/中证官网交叉验证
  python scripts/manage_index_cache.py rebuild  # 掘金拉取补齐 (需掘金终端运行 + GM_TOKEN)

校验规则
--------
1. 缺口: 任意相邻缓存日期 (datetime) 间隔 > 45 天 视为缺口 (定期调仓期 6/12 月外不应出现)
2. 数量: hs300 ∈ [280, 305], zz500 ∈ [470, 505], zz1000 ∈ [900, 1010]
3. 相邻变化率: 平均 < 5%, 单期 > 30% 仅允许出现在 6 月/12 月 (中证定期调仓)
4. 中证官网交叉验证 (akshare, 需网络): 当前样本 vs 最近一期缓存, 重合率应 > 85%
"""
import argparse
import datetime as dt
import os
import shutil
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_FILE = os.path.join(PROJECT_ROOT, 'data', 'index_stocks_cache.parquet')
# 旧框架副本 (同源, 2022 完整)
LEGACY_CACHE_FILE = r'E:\quant\quant\alpha191\data\index_stocks_cache.parquet'

INDICES = ('hs300', 'zz500', 'zz1000')
INDEX_SYMBOLS = {'hs300': 'SHSE.000300', 'zz500': 'SHSE.000905', 'zz1000': 'SHSE.000852',
                 'cyb': 'SZSE.399006', 'kc50': 'SHSE.000688'}
# 成分股数量校验范围 (cyb=创业板指100只/kc50=科创50只; 掘金可能返回略多含备选样本)
COUNT_RANGE = {'hs300': (280, 305), 'zz500': (470, 505), 'zz1000': (900, 1010),
               'cyb': (90, 115), 'kc50': (45, 60)}


def load_cache(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=['cache_key', 'timestamp', 'stock_list'])
    return pd.read_parquet(path)


def parse_key(key):
    """cache_key: 'hs300_20200409' -> (index, 'YYYYMMDD')"""
    idx, d = key.split('_')[0], key.split('_')[1]
    return idx, d


def to_records(df):
    out = []
    for _, r in df.iterrows():
        out.append({'cache_key': r['cache_key'], 'timestamp': r['timestamp'],
                    'stock_list': list(r['stock_list'])})
    return pd.DataFrame(out)


def merge_legacy(backup=True):
    """合并旧框架副本到主缓存 (主缓存优先, 旧框架补齐主缓存缺失的 key)。"""
    main = load_cache(CACHE_FILE)
    legacy = load_cache(LEGACY_CACHE_FILE)
    if main.empty or legacy.empty:
        raise SystemExit(f'缓存文件缺失: main={len(main)} legacy={len(legacy)}')

    if backup:
        bak = CACHE_FILE + '.bak_20260804'
        shutil.copy2(CACHE_FILE, bak)
        print(f'[merge] 已备份主缓存 -> {bak}')

    main_keys = set(main['cache_key'])
    leg_keys = set(legacy['cache_key'])
    add = legacy[~legacy['cache_key'].isin(main_keys)]
    print(f'[merge] 主缓存 {len(main)} 条, 旧框架 {len(legacy)} 条, '
          f'将新增 {len(add)} 条 (交集 {len(main_keys & leg_keys)})')

    merged = pd.concat([main, add], ignore_index=True)
    # wpg 以主缓存为准 (已含), 其它指数旧框架补齐
    merged = merged.drop_duplicates(subset='cache_key', keep='first')
    merged = merged.sort_values('cache_key').reset_index(drop=True)
    merged = to_records(merged)
    merged.to_parquet(CACHE_FILE, index=False)
    print(f'[merge] 合并完成: {len(main)} -> {len(merged)} 条 -> {CACHE_FILE}')

    # 合并后缺口速报
    report_gaps(merged)
    return merged


def report_gaps(df, threshold_days=45):
    """按指数报告相邻日期缺口 (> threshold_days 天)。"""
    all_gaps = []
    for idx in INDEX_SYMBOLS:
        sub = df[df['cache_key'].str.startswith(idx + '_')].copy()
        sub['d'] = pd.to_datetime(sub['cache_key'].str.split('_').str[1], format='%Y%m%d')
        sub = sub.sort_values('d')
        prev = None
        for _, r in sub.iterrows():
            if prev is not None:
                gap = (r['d'] - prev).days
                if gap > threshold_days:
                    all_gaps.append((idx, prev.strftime('%Y-%m-%d'), r['d'].strftime('%Y-%m-%d'), gap))
            prev = r['d']
    if all_gaps:
        print(f'[gap] 发现 {len(all_gaps)} 个缺口 (> {threshold_days} 天):')
        for g in all_gaps[:20]:
            print(f'  {g[0]}: {g[1]} -> {g[2]} ({g[3]} 天)')
        return False
    print(f'[gap] 无缺口 (相邻间隔均 <= {threshold_days} 天) ✓')
    return True


def verify(net_check=True):
    """完整校验: 缺口 / 数量 / 相邻变化率 / 中证交叉验证。返回 True 表示全部通过。"""
    df = load_cache(CACHE_FILE)
    print(f'[verify] 缓存 {len(df)} 条, {df["cache_key"].min()} ~ {df["cache_key"].max()}')
    ok = True

    # 1. 缺口
    ok &= report_gaps(df)

    # 2. 数量范围
    for idx in INDEX_SYMBOLS:
        sub = df[df['cache_key'].str.startswith(idx + '_')]
        if sub.empty:
            print(f'[count] {idx}: 无数据, 跳过')
            continue
        n = sub['stock_list'].apply(len)
        lo, hi = COUNT_RANGE[idx]
        rng = f'[{n.min()}, {n.max()}]'
        within = (n >= lo).all() and (n <= hi).all()
        print(f'[count] {idx}: n={len(sub)} 条, 成分股数 {rng} (要求 {lo}~{hi}) '
              f'{"✓" if within else "✗ 越界"}')
        ok &= within

    # 3. 相邻变化率 (仅允许 6/12 月大幅变化)
    for idx in INDEX_SYMBOLS:
        sub = df[df['cache_key'].str.startswith(idx + '_')].copy()
        if sub.empty:
            print(f'[churn] {idx}: 无数据, 跳过')
            continue
        sub['d'] = pd.to_datetime(sub['cache_key'].str.split('_').str[1], format='%Y%m%d')
        sub = sub.sort_values('d')
        prev, prevd = None, None
        bad = []
        rates = []
        for _, r in sub.iterrows():
            cur = set(r['stock_list'])
            if prev is not None:
                rate = len(cur ^ prev) / max(len(cur), 1)
                rates.append(rate)
                if rate > 0.30 and prevd.month not in (6, 12):
                    bad.append((prevd.strftime('%Y%m%d'), r['d'].strftime('%Y%m%d'), round(rate, 3)))
            prev, prevd = cur, r['d']
        mean_rate = float(np.mean(rates)) if rates else 0.0
        print(f'[churn] {idx}: 平均相邻变化率 {mean_rate*100:.1f}% '
              f'{"✓" if mean_rate < 0.05 else "✗ 偏高"} '
              f'{"非调仓月异常" + str(bad[:5]) if bad else ""}')
        ok &= mean_rate < 0.05 and not bad

    # 4. 中证官网交叉验证 (可选, 需网络)
    if net_check:
        try:
            import akshare as ak
            for idx, name in [('000300', 'hs300'), ('000905', 'zz500'), ('000852', 'zz1000')]:
                cons = ak.index_stock_cons_csindex(symbol=idx)
                cur = set(cons['成分券代码'].map(
                    lambda c: ('sh.' if str(c).zfill(6).startswith(('6', '9')) else 'sz.') + str(c).zfill(6)))
                sub = df[df['cache_key'].str.startswith(name + '_')]
                last_key = sorted(sub['cache_key'])[-1]
                cache = set(sub[sub['cache_key'] == last_key].iloc[0]['stock_list'])
                inter = len(cur & cache)
                ratio = inter / max(len(cache), 1)
                print(f'[csindex] {name}: 中证当前 {len(cur)} 只 vs 缓存最近 {last_key} {len(cache)} 只, '
                      f'重合 {inter} ({ratio*100:.1f}%) {"✓" if ratio > 0.85 else "✗ <85%"}')
                ok &= ratio > 0.85
        except Exception as e:
            print(f'[csindex] 交叉验证跳过: {type(e).__name__} {str(e)[:80]}')

    print(f'[verify] 结果: {"全部通过 ✓" if ok else "存在异常 ✗"}')
    return ok


def repro(n_samples=6, seed=20260804):
    """复现性验证: 从缓存中均匀抽取 n 个日期, 用掘金重新拉取同日期成分股并与缓存对比。

    若全部一致 -> 缓存确由掘金 stk_get_index_constituents 直出, 无手工篡改,
    生成方式彻底坐实。需掘金终端运行 + GM_TOKEN。
    """
    try:
        from gm.api import set_token, stk_get_index_constituents
    except ImportError:
        raise SystemExit('[repro] 掘金 SDK 未安装: pip install gm')
    token = os.environ.get('GM_TOKEN')
    if not token:
        raise SystemExit('[repro] 未设置 GM_TOKEN 环境变量')
    set_token(token)

    df = load_cache(CACHE_FILE)
    print(f'[repro] 缓存 {len(df)} 条, 抽样 {n_samples} 个日期 × 3 指数与掘金逐条对比')
    rng = np.random.default_rng(seed)
    all_ok = True
    for idx in INDEX_SYMBOLS:
        sub = df[df['cache_key'].str.startswith(idx + '_')].copy()
        sub['d'] = sub['cache_key'].str.split('_').str[1]
        sub = sub.sort_values('d')
        picks = sorted(rng.choice(sub['d'].values, size=min(n_samples, len(sub)), replace=False))
        for dstr in picks:
            cache_codes = set(sub[sub['d'] == dstr].iloc[0]['stock_list'])
            try:
                r = stk_get_index_constituents(
                    index=INDEX_SYMBOLS[idx],
                    trade_date=pd.to_datetime(dstr, format='%Y%m%d').strftime('%Y-%m-%d'))
                gm_codes = set()
                if r is not None and not r.empty:
                    col = 'symbol' if 'symbol' in r.columns else r.columns[0]
                    for full in r[col].tolist():
                        if str(full).startswith('SHSE.'):
                            c = 'sh.' + str(full)[5:]
                        elif str(full).startswith('SZSE.'):
                            c = 'sz.' + str(full)[5:]
                        else:
                            continue
                        if c.startswith(('sh.600', 'sh.601', 'sh.603', 'sh.605', 'sh.688', 'sh.689',
                                         'sz.000', 'sz.001', 'sz.002', 'sz.003', 'sz.300', 'sz.301', 'sz.302')):
                            gm_codes.add(c)
                inter = len(cache_codes & gm_codes)
                ratio = inter / max(len(cache_codes), 1)
                ok = ratio >= 0.99 and len(cache_codes ^ gm_codes) <= max(2, 0.01 * len(cache_codes))
                all_ok &= ok
                print(f'  {idx} {dstr}: 缓存 {len(cache_codes)} vs 掘金 {len(gm_codes)} '
                      f'重合 {inter} ({ratio*100:.1f}%) 对称差 {len(cache_codes ^ gm_codes)} '
                      f'{"✓" if ok else "✗"}')
            except Exception as e:
                all_ok = False
                print(f'  {idx} {dstr}: 掘金调用失败 {e}')
    print(f'[repro] 结果: {"全部一致, 缓存=掘金直出 ✓" if all_ok else "存在差异 ✗"}')
    return all_ok


def rebuild(start='2020-01-01', end=None, interval_days=7, force=False):
    """掘金拉取补齐 (需掘金终端运行 + GM_TOKEN)。断点续传: 已有 cache_key 跳过。

    语义: 从 start 起每 interval_days 天取一个交��日, 调用 stk_get_index_constituents
    获取当日收盘后成分股 (与缓存原生成方式一致, 严格无未来: cache_key 当日盘后已知,
    回测中 D 日盘前取 cache_key <= D 最近一条)。
    """
    try:
        from gm.api import set_token, stk_get_index_constituents
    except ImportError:
        raise SystemExit('[rebuild] 掘金 SDK 未安装: pip install gm')

    token = os.environ.get('GM_TOKEN')
    if not token:
        raise SystemExit('[rebuild] 未设置 GM_TOKEN 环境变量 (可参考 run_select.py 中的值)')
    set_token(token)

    end = end or dt.date.today().strftime('%Y-%m-%d')
    df = load_cache(CACHE_FILE)
    existing = set(df['cache_key'])

    # 交易日历 (从本地 trade_datas.csv 或掘金)
    cal_file = os.path.join(PROJECT_ROOT, 'data', 'trade_datas.csv')
    if os.path.exists(cal_file):
        cal = pd.read_csv(cal_file)
        cal['calendar_date'] = pd.to_datetime(cal['calendar_date'])
        trade_days = sorted(cal.loc[cal['is_trading_day'] == 1, 'calendar_date'])
    else:
        raise SystemExit(f'[rebuild] 缺少交易日历 {cal_file}')

    # 生成查询日期序列: 每 interval_days 天取一个交易日
    query_dates = []
    d0 = pd.to_datetime(start)
    d1 = pd.to_datetime(end)
    cur = d0
    while cur <= d1:
        td = [d for d in trade_days if d >= cur]
        if td and td[0] <= cur + dt.timedelta(days=interval_days):
            query_dates.append(td[0])
        cur += dt.timedelta(days=interval_days)
    print(f'[rebuild] 计划查询 {len(query_dates)} 个日期 ({start} ~ {end}, 每 {interval_days} 天)')

    new_records = []
    for qd in query_dates:
        dstr = qd.strftime('%Y%m%d')
        for name, sym in INDEX_SYMBOLS.items():
            key = f'{name}_{dstr}'
            if key in existing and not force:
                continue
            try:
                r = stk_get_index_constituents(index=sym, trade_date=qd.strftime('%Y-%m-%d'))
                codes = []
                if r is not None and not r.empty:
                    col = 'symbol' if 'symbol' in r.columns else r.columns[0]
                    for full in r[col].tolist():
                        if str(full).startswith('SHSE.'):
                            c = 'sh.' + str(full)[5:]
                        elif str(full).startswith('SZSE.'):
                            c = 'sz.' + str(full)[5:]
                        else:
                            continue
                        if c.startswith(('sh.600', 'sh.601', 'sh.603', 'sh.605', 'sh.688', 'sh.689',
                                         'sz.000', 'sz.001', 'sz.002', 'sz.003', 'sz.300', 'sz.301', 'sz.302')):
                            codes.append(c)
                codes = sorted(set(codes))
                if not codes:
                    # 指数该日无数据(如科创50在2020-07-23发布前): 跳过空条目, 不污染缓存
                    print(f'  {name} {dstr}: 空 (跳过)')
                    continue
                new_records.append({'cache_key': key, 'timestamp': dt.datetime.now().timestamp(),
                                    'stock_list': codes})
                existing.add(key)
                print(f'  {name} {dstr}: {len(codes)} 只')
            except Exception as e:
                print(f'  {name} {dstr}: 失败 {e}')
    if new_records:
        merged = pd.concat([df, pd.DataFrame(new_records)], ignore_index=True)
        # force 重拉时旧口径条目仍存在: 按 cache_key 去重保留新拉取 (keep='last')
        merged = merged.drop_duplicates(subset='cache_key', keep='last')
        merged = merged.sort_values('cache_key').reset_index(drop=True)
        merged = to_records(merged)
        merged.to_parquet(CACHE_FILE, index=False)
        print(f'[rebuild] 新增 {len(new_records)} 条, 缓存 {len(df)} -> {len(merged)} 条')
    else:
        print('[rebuild] 无新增 (全部命中缓存或全部失败)')
    return verify(net_check=False)



def rekey():
    """按缓存已有全部日期重拉 5 指数 (新过滤口径), 消除新旧口径混杂 (2026-08-05 前缀修复)。"""
    try:
        from gm.api import set_token, stk_get_index_constituents
    except ImportError:
        raise SystemExit('[rekey] 掘金 SDK 未安装')
    token = os.environ.get('GM_TOKEN')
    if not token:
        raise SystemExit('[rekey] 未设置 GM_TOKEN 环境变量')
    set_token(token)
    df = load_cache(CACHE_FILE)
    # 收集 5 指数全部日期 (wpg 保留不动)
    dates = set()
    for idx in INDEX_SYMBOLS:
        dates |= set(k.split('_')[1] for k in df['cache_key'] if k.startswith(idx + '_'))
    dates = sorted(dates)
    print(f'[rekey] 重拉 {len(dates)} 个日期 × {len(INDEX_SYMBOLS)} 指数 (wpg 保留)')
    new_rows = []
    for dstr in dates:
        td = pd.to_datetime(dstr, format='%Y%m%d').strftime('%Y-%m-%d')
        for name, sym in INDEX_SYMBOLS.items():
            try:
                r = stk_get_index_constituents(index=sym, trade_date=td)
                codes = []
                if r is not None and not r.empty:
                    col = 'symbol' if 'symbol' in r.columns else r.columns[0]
                    for full in r[col].tolist():
                        if str(full).startswith('SHSE.'):
                            c = 'sh.' + str(full)[5:]
                        elif str(full).startswith('SZSE.'):
                            c = 'sz.' + str(full)[5:]
                        else:
                            continue
                        if c.startswith(('sh.600', 'sh.601', 'sh.603', 'sh.605', 'sh.688', 'sh.689',
                                         'sz.000', 'sz.001', 'sz.002', 'sz.003', 'sz.300',
                                         'sz.301', 'sz.302')):
                            codes.append(c)
                codes = sorted(set(codes))
                if codes:
                    new_rows.append({'cache_key': f'{name}_{dstr}',
                                     'timestamp': dt.datetime.now().timestamp(),
                                     'stock_list': codes})
            except Exception as e:
                print(f'  {name} {dstr}: 失败 {e}')
    if new_rows:
        keep = df[~df['cache_key'].str.split('_').str[0].isin(INDEX_SYMBOLS)]  # 保留 wpg
        merged = pd.concat([keep, pd.DataFrame(new_rows)], ignore_index=True)
        merged = merged.drop_duplicates(subset='cache_key', keep='last')
        merged = merged.sort_values('cache_key').reset_index(drop=True)
        merged = to_records(merged)
        merged.to_parquet(CACHE_FILE, index=False)
        print(f'[rekey] 完成: 指数条目 {len(new_rows)} 条, 总缓存 {len(df)} -> {len(merged)} 条')
    else:
        print('[rekey] 无新记录 (全部失败)')
    return verify(net_check=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='指数成分股缓存管理')
    ap.add_argument('cmd', choices=['merge', 'verify', 'rebuild', 'repro', 'rekey'])
    ap.add_argument('--start', default='2020-01-01')
    ap.add_argument('--end', default=None)
    ap.add_argument('--interval', type=int, default=7)
    ap.add_argument('--force', action='store_true', help='强制重拉已有日期 (修正过滤口径后使用)')
    ap.add_argument('--samples', type=int, default=6, help='repro 抽样日期数')
    ap.add_argument('--no-net', action='store_true', help='verify 跳过中证官网交叉验证')
    args = ap.parse_args()
    if args.cmd == 'merge':
        merge_legacy()
    elif args.cmd == 'verify':
        verify(net_check=not args.no_net)
    elif args.cmd == 'repro':
        repro(n_samples=args.samples)
    elif args.cmd == 'rekey':
        rekey()
    elif args.cmd == 'rebuild':
        rebuild(args.start, args.end, args.interval, force=args.force)
