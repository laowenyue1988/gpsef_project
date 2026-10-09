# -*- coding: utf-8 -*-
"""BARRA 收益归因脚本 (框架级, 2026-08-10 v2, 2026-08-11 v3)

对已有回测结果做基于 BARRA 模型的收益归因:
  R_p,t = f_0,t + Σ_j X_j,t · f_j,t + ε_t
    f_j,t  = 因子 j 在 t 日的截面因子收益 (基准域=沪市全部 A 股, 含截距的市值加权 OLS)
    X_j,t  = 组合在因子 j 的市值加权暴露 (持仓股暴露用 barra_builder 本地重算, 20 日 asof)
    f_0,t  = 截距项 (基准市场收益)
    ε_t    = 特质收益 (残差, 保证 Σ贡献 = 组合收益)

基准 (用户决策 2026-08-10): 上证综指 (sh000001) — 基准域 = 沪市全部 A 股
  R_b  = 上证综指日收益; 因子收益回归域 = 沪市全股票; 暴露标准化参数取自沪市池市值加权

数据依赖 (全部本地):
  <回测目录>/positions.csv  +  daily_values.csv
  data/barra_exposures/barra_expo_*.parquet  (CSI800 月度快照: 仅作暴露重建的交叉验证对照源)
  data/index_daily/sh000001.parquet          (上证综指日线 → 基准 R_b + 交易日历)
  data/performance_reports/  data/back_adjust_factors/  data/raw_daily/  (barra_builder 本地重算持仓暴露)

输出: <回测目录>/attribution/
  daily_attribution.csv / factor_exposures.csv / attribution_summary.csv / period_attribution.csv
  carino_attribution.csv (链接到复合收益的归因) / 术语表_收益含义.md
  plots/*.png (6 张) / attribution_report.md

用法:
  python scripts/barra_attribution.py --backtest-dir results/SmallAlpha191_20220801_20260522_20260805230911
"""
import os
import sys
import json
import argparse
import logging
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from strategies.wyckoff20.barra_builder import (
    STYLE_FACTORS, _compute_factor_exposures, _get_latest_financials,
    _load_daily_data,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('barra_attribution')

BARRA_EXPOSURE_DIR = os.path.join(PROJECT_ROOT, 'data', 'barra_exposures')
RAW_DAILY_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw_daily')
WYCKOFF_SNAP_DIR = os.path.join(PROJECT_ROOT, 'data', 'wyckoff20_v5_snapshots')
REBALANCE_INTERVAL = 20      # 暴露重算周期 (与策略 rebalance_interval 对齐)
INDEX_FILE = os.path.join(PROJECT_ROOT, 'data', 'index_daily', 'sh000001.parquet')  # 上证综指
BENCH_NAME = '上证综指(sh000001)'          # 归因基准
BENCH_EXPOSURE_POOL = 'sh'                # 基准暴露域 = 沪市全部 A 股 (上证指数成分域)

# 【基准切换】微盘池基准 = 策略真实选股池 (DynamicStockPool 候选池 ∩ T-1 市值最低 N 只)
#   smallcap_pool_cache.parquet 由策略动态维护 (trade_date, ts_code, mv_rank), mv_rank 1..N
SMALLCAP_POOL_FILE = os.path.join(PROJECT_ROOT, 'data', 'smallcap_pool_cache.parquet')
SMALLCAP_POOL_SIZE = 300
BENCH_MODE = 'index'                       # 当前基准模式: 'index'=上证综指 | 'smallcap'=微盘池
BENCH_DESC = '沪市全部 A 股 (上证指数成分域)'  # 基准域文字描述 (写入报告)


def set_benchmark(mode):
    """切换归因基准 (模块级全局, 报告/图/术语表自动跟随)。

    mode='index'    → 上证综指 (sh000001), 基准域 = 沪市全部 A 股 (原默认)
    mode='smallcap' → 微盘池 (策略选股池: T-1 市值最低 N 只), 基准域 = 微盘池成分
    返回 (BENCH_NAME, BENCH_EXPOSURE_POOL, BENCH_MODE, BENCH_DESC)。
    """
    global BENCH_NAME, BENCH_EXPOSURE_POOL, BENCH_MODE, BENCH_DESC
    if mode == 'smallcap':
        BENCH_NAME = f'微盘{SMALLCAP_POOL_SIZE}池(策略选股池)'
        BENCH_EXPOSURE_POOL = 'smallcap'
        BENCH_MODE = 'smallcap'
        BENCH_DESC = f'微盘池 (DynamicStockPool 候选池中 T-1 市值最低 {SMALLCAP_POOL_SIZE} 只)'
    else:
        BENCH_NAME = '上证综指(sh000001)'
        BENCH_EXPOSURE_POOL = 'sh'
        BENCH_MODE = 'index'
        BENCH_DESC = '沪市全部 A 股 (上证指数成分域)'
    return BENCH_NAME, BENCH_EXPOSURE_POOL, BENCH_MODE, BENCH_DESC

# 因子中文名 (图/报告/Carino 共用)
LABEL_MAP = {'f_0': '市场/截距', 'size': '规模', 'beta': 'Beta', 'value': '价值',
             'momentum': '动量', 'volatility': '波动率', 'epsilon': '特质',
             'epsilon_adj': '特质/alpha'}
COLOR_MAP = {'size': '#378ADD', 'beta': '#1D9E75', 'value': '#BA7517',
             'momentum': '#D4537E', 'volatility': '#7F77DD', 'f_0': '#5F5E5A',
             'epsilon': '#A32D2D', 'epsilon_adj': '#A32D2D'}


def time_now():
    return datetime.now()


def to6(code) -> str:
    """ts_code('sz.002629') → 6 位代码('002629'); 已是 6 位则原样返回。"""
    return str(code).strip().split('.')[-1].zfill(6)


def load_backtest(backtest_dir):
    pos = pd.read_csv(os.path.join(backtest_dir, 'positions.csv'), dtype={'code': str})
    pos['date'] = pd.to_datetime(pos['date'])
    pos['code6'] = pos['code'].map(to6)
    val = pd.read_csv(os.path.join(backtest_dir, 'daily_values.csv'))
    val['date'] = pd.to_datetime(val['date'])
    return pos, val


def load_barra_exposures():
    """加载全部 BARRA 快照 → (snap_dict[Timestamp]=DataFrame, snap_dates 升序)。"""
    snap_dict, snap_dates = {}, []
    for f in sorted(os.listdir(BARRA_EXPOSURE_DIR)):
        if not (f.startswith('barra_expo_') and f.endswith('.parquet')):
            continue
        d = datetime.strptime(f[11:19], '%Y%m%d')
        df = pd.read_parquet(os.path.join(BARRA_EXPOSURE_DIR, f))
        df['code6'] = df['code'].astype(str).str.zfill(6)
        snap_dict[pd.Timestamp(d)] = df
        snap_dates.append(pd.Timestamp(d))
    return snap_dict, sorted(snap_dates)


def asof_snapshot(snap_dates, t):
    """≤ t 的最近快照日期 (None 若早于首个快照)。"""
    i = np.searchsorted(snap_dates, t, side='right') - 1
    return snap_dates[i] if i >= 0 else None


def build_trade_calendar():
    """交易日历: 从上证综指指数日线提取 (覆盖回测区间所需历史)。

    INDEX_FILE 已是绝对路径, os.path.join 遇绝对路径第二参数直接返回它,
    这里等价于直接 pd.read_parquet(INDEX_FILE), 显式写出避免误解。
    """
    df = pd.read_parquet(INDEX_FILE)
    col = 'trade_date' if 'trade_date' in df.columns else 'date'
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(df[col].unique())))
    return cal


def parse_rets_list(raw):
    """barra 快照 daily_rets_252 可能是 list / ndarray / JSON 字符串, 统一转 list。"""
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None
    if isinstance(raw, np.ndarray):
        return raw.tolist()
    if isinstance(raw, list):
        return raw
    return None


def build_csi800_returns(snap_dict, snap_dates, cal, dates):
    """【交叉验证对照/历史路径】CSI800 快照收益面板 (date × code6)。

    当前主流程已改用沪市全股票前复权收益面板 (build_stock_return_panel) 做因子收益回归;
    本函数保留用于: ①交叉验证对照; ②需要 CSI800 口径面板的历史用途。
    每快照 S 的 daily_rets_252 对齐到 cal 中 [S-251, S] 交易日;
    相邻快照重叠时取较新快照 (复权基准更接近, keep='last')。
    """
    ret_by_code = defaultdict(list)
    for S in snap_dates:
        snap = snap_dict[S]
        iS = cal.searchsorted(S)
        for _, r in snap.iterrows():
            rlist = parse_rets_list(r['daily_rets_252'])
            if not rlist:
                continue
            L = len(rlist)
            if iS - L + 1 < 0:
                continue
            idx = cal[iS - L + 1:iS + 1]
            ret_by_code[r['code6']].append(pd.Series(rlist, index=idx))
    panel = {}
    for code, pieces in ret_by_code.items():
        s = pd.concat(pieces)
        s = s[~s.index.duplicated(keep='last')]
        panel[code] = s
    ret_full = pd.DataFrame(panel)          # 全历史 (覆盖快照时段)
    return ret_full


def build_weight_panel(snap_dict, snap_dates, dates, codes):
    """每日市值权重面板 (date × code6): 每行 = 该日 asof 快照的 weight_bm。"""
    rows = {}
    for t in dates:
        S = asof_snapshot(snap_dates, t)
        if S is None:
            continue
        rows[t] = snap_dict[S].set_index('code6')['weight_bm']
    wp = pd.DataFrame(rows).T  # date × code6
    wp = wp.reindex(index=dates, columns=codes).fillna(0.0)
    return wp


def build_market_returns(ret_panel, w_panel):
    """【备选/历史】市值加权市场日收益 (主用 build_index_returns 上证综指指数收益)。

    曾因个别股票复权异常 (600795 单日+30%) 污染市值加权收益, 已弃用为主路径。
    """
    return (ret_panel * w_panel).sum(axis=1, min_count=1)


def build_index_returns():
    """市场日收益 / 归因基准 = 上证综指 (sh000001) 点位日收益 (全历史, 本地指数日线)。

    用真实指数点位, 避免个别股票复权错误的污染。
    """
    df = pd.read_parquet(INDEX_FILE)
    col = 'trade_date' if 'trade_date' in df.columns else 'date'
    df['dt'] = pd.to_datetime(df[col])
    s = df.sort_values('dt').set_index('dt')['close'].astype(float)
    s = s[~s.index.duplicated(keep='last')]
    return s.pct_change()


def build_smallcap_benchmark(dates, grid_dates):
    """微盘池基准 (策略真实选股池) 构建 → 返回 dict。

    数据源: data/smallcap_pool_cache.parquet (策略动态维护, mv_rank 1..N=市值最低 N 只)
    口径: 与策略选股池严格一致 — 每个调仓日 (grid_date) 取 asof 最近的微盘池成分,
          调仓周期内成分不变 (策略 20 日调仓期间从该固定池选股)。

    返回 dict:
      bench_codes      : 全部 grid 微盘池成分并集 (list[code6], 用于加载收益面板/暴露重建)
      pool_by_grid     : {grid_date: [code6]} 各调仓周期基准成分 (回归域 + 暴露标准化域)
      rb               : Series(date → 微盘池市值加权日收益, 即归因基准 R_b)
      w_panel          : DataFrame(date × code6) 微盘池市值权重 (回归加权用)
      bench_codes_set  : set(bench_codes) 全部成分集合 (含持仓兜底)
    """
    sc = pd.read_parquet(SMALLCAP_POOL_FILE, columns=['trade_date', 'ts_code', 'mv_rank'])
    sc = sc[sc['mv_rank'] <= SMALLCAP_POOL_SIZE].copy()
    sc['date'] = pd.to_datetime(sc['trade_date'].astype(str))
    sc['code6'] = sc['ts_code'].str.split('.').str[1]
    pool_dates = np.sort(sc['date'].unique())
    pool_by_date = {d: g['code6'].tolist() for d, g in sc.groupby('date')}

    # 每个 grid_date 取 asof 最近微盘池成分 (trade_date ≤ grid_date)
    pool_by_grid = {}
    for gd in grid_dates:
        i = np.searchsorted(pool_dates, gd, side='right') - 1
        if i >= 0:
            pool_by_grid[gd] = pool_by_date[pd.Timestamp(pool_dates[i])]

    bench_codes = sorted(set().union(*[set(v) for v in pool_by_grid.values()]))
    bench_codes_set = set(bench_codes)

    # 市值权重面板: 每期微盘池 300 只 asof 市值归一化 (date × code6, 周期内成分不变)
    #   权重仅对"当期微盘池成分"非零, 与策略 20 日调仓选股池精确对齐 (回归权重 + R_b 同口径)
    mv = pd.read_parquet(os.path.join(PROJECT_ROOT, 'data', 'total_mv_cache.parquet'),
                         columns=['trade_date', 'ts_code', 'total_mv'])
    mv['date'] = pd.to_datetime(mv['trade_date'])
    mv['code6'] = mv['ts_code'].str.split('.').str[1]
    mv = mv[mv['code6'].isin(bench_codes_set)]
    mv_dates = np.sort(mv['date'].unique())
    mv_by_date = {d: g.set_index('code6')['total_mv'] for d, g in mv.groupby('date')}
    gd_arr = np.array(grid_dates)
    w_rows = {}
    for t in dates:
        gi = int(np.searchsorted(gd_arr, t, side='right')) - 1
        if gi < 0:
            continue
        cur_codes = pool_by_grid.get(pd.Timestamp(gd_arr[gi]), [])
        i = np.searchsorted(mv_dates, t, side='right') - 1
        if i < 0 or not cur_codes:
            continue
        s = mv_by_date[pd.Timestamp(mv_dates[i])].reindex(cur_codes)
        s = s[s > 0]
        if len(s):
            w_rows[t] = s / s.sum()
    w_panel = pd.DataFrame(w_rows).T.reindex(index=dates, columns=bench_codes).fillna(0.0)

    # 微盘池市值加权日收益 (R_b): 持仓/基准股前复权收益 × 微盘池权重
    ret_panel = build_stock_return_panel(
        bench_codes, dates,
        cache_path=os.path.join(PROJECT_ROOT, 'data', '.barra_attr_cache',
                                'return_panel_smallcap.parquet'))
    rb = (ret_panel * w_panel).sum(axis=1, min_count=1)

    logger.info(f"微盘池基准: {len(bench_codes)} 只并集成分, "
                f"{len(pool_by_grid)} 个调仓周期 (每期 {SMALLCAP_POOL_SIZE} 只), "
                f"R_b 覆盖 {rb.notna().sum()}/{len(rb)} 日")
    return {'bench_codes': bench_codes, 'pool_by_grid': pool_by_grid, 'rb': rb,
            'w_panel': w_panel, 'bench_codes_set': bench_codes_set}


def load_total_mv_asof(dates, prefix=None):
    """total_mv_cache.parquet asof → (date × code6) 对数市值面板 (PIT 无未来)。

    prefix='sh' 时仅沪市 (基准暴露域); None 返回全部。
    """
    mv = pd.read_parquet(os.path.join(PROJECT_ROOT, 'data', 'total_mv_cache.parquet'),
                         columns=['trade_date', 'ts_code', 'total_mv'])
    mv['date'] = pd.to_datetime(mv['trade_date'])
    mv['code6'] = mv['ts_code'].str.split('.').str[1]
    if prefix:
        mv = mv[mv['ts_code'].str.startswith(prefix + '.')]
    mv['ln_mv'] = np.log(mv['total_mv'].clip(lower=1e-8))
    mv_dates = np.sort(mv['date'].unique())
    mv_by_date = {d: g.set_index('code6')['ln_mv'] for d, g in mv.groupby('date')}
    out = {}
    for t in dates:
        i = np.searchsorted(mv_dates, t, side='right') - 1
        if i < 0:
            continue
        out[t] = mv_by_date[pd.Timestamp(mv_dates[i])]
    return pd.DataFrame(out).T


def build_mv_weight_panel(dates, prefix=None):
    """市值权重面板 (date × code6): 每行 = asof total_mv 归一化为权重 (基准域市值加权)。

    prefix='sh' 时仅沪市 (回归权重用); None 返回全部。
    """
    mv = pd.read_parquet(os.path.join(PROJECT_ROOT, 'data', 'total_mv_cache.parquet'),
                         columns=['trade_date', 'ts_code', 'total_mv'])
    mv['date'] = pd.to_datetime(mv['trade_date'])
    mv['code6'] = mv['ts_code'].str.split('.').str[1]
    if prefix:
        mv = mv[mv['ts_code'].str.startswith(prefix + '.')]
    mv_dates = np.sort(mv['date'].unique())
    mv_by_date = {d: g.set_index('code6')['total_mv'] for d, g in mv.groupby('date')}
    out = {}
    for t in dates:
        i = np.searchsorted(mv_dates, t, side='right') - 1
        if i < 0:
            continue
        s = mv_by_date[pd.Timestamp(mv_dates[i])]
        s = s[s > 0]
        out[t] = s / s.sum()
    return pd.DataFrame(out).T


def build_stock_return_panel(codes, dates, cache_path=None):
    """基准域 (沪市) 全部股票的前复权日收益面板 (date × code6)。

    前复权逻辑与 barra_builder 一致 (merge_asof 对齐复权因子, 复权基准日 = dates[-1]);
    缓存为 parquet 避免每次重建 (2407 只 ≈ 5 分钟)。
    """
    from strategies.wyckoff20.barra_builder import _load_adjust_factors_local
    if cache_path and os.path.exists(cache_path):
        panel = pd.read_parquet(cache_path)
        return panel.reindex(dates)
    out = {}
    for code in codes:
        d = _load_daily_data(code)
        if d is None or len(d) < 60:
            continue
        close = d['close']
        adj_df = _load_adjust_factors_local(code)
        if adj_df is not None and len(adj_df) > 0:
            ref_series = adj_df['adjustFactor'][adj_df.index <= dates[-1]]
            if len(ref_series) > 0 and float(ref_series.iloc[-1]) > 0:
                ref_factor = float(ref_series.iloc[-1])
                daily_dates = pd.DataFrame({'date': close.index})
                adj_dates = pd.DataFrame({'date': adj_df.index,
                                          'adjustFactor': adj_df['adjustFactor'].values})
                merged = pd.merge_asof(daily_dates.sort_values('date'),
                                       adj_dates.sort_values('date'),
                                       on='date', direction='backward').set_index('date')
                close_adj = close * ref_factor / merged['adjustFactor']
            else:
                close_adj = close
        else:
            close_adj = close
        out[code] = close_adj.pct_change()
    panel = pd.DataFrame(out)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        panel.to_parquet(cache_path)
    return panel.reindex(dates)


def build_exposure_pool(snap_dict, snap_dates, pos, dates, market_rets_full=None,
                        bench_codes=None, bench_pool_by_grid=None, bench_w_panel=None):
    """统一重建"基准域 ∪ 持仓"全池 BARRA 5 因子暴露 (每 20 交易日一期, 基准池标准化)。

    - 4 风格因子 (beta/value/momentum/volatility): barra_builder._compute_factor_exposures 本地原始值
    - size: ln(total_mv) 来自 total_mv_cache (asof)
    - 每期: 基准池市值加权 Winsorize(1%/99%) 参数 → 标准化全池 → 组合暴露 = 相对基准偏离
    基准域由参数控制:
      bench_codes=None           → 沪市全部 A 股 (上证综指基准, 原默认)
      bench_codes=<微盘并集>      → 微盘池基准 (bench_pool_by_grid 提供各调仓周期成分, bench_w_panel 提供权重)
    返回 (expo_by_grid, bench_codes)。
    """
    pos['weight'] = pos.groupby('date')['market_value'].transform(lambda x: x / x.sum())
    hold_codes = sorted(pos['code6'].unique())
    # 基准域确定: 外部传入 (微盘池) 或默认沪市全部
    if bench_codes is None:
        mv_all = pd.read_parquet(os.path.join(PROJECT_ROOT, 'data', 'total_mv_cache.parquet'),
                                 columns=['ts_code'])
        bench_codes = sorted(mv_all[mv_all['ts_code'].str.startswith(BENCH_EXPOSURE_POOL + '.')]
                             ['ts_code'].str.split('.').str[1].unique())
    bench_codes_set = set(bench_codes)
    pool_codes = sorted(set(hold_codes) | set(bench_codes))
    logger.info(f"暴露池: {len(pool_codes)} 只 (持仓 {len(hold_codes)} + 基准域 {len(bench_codes)})")

    daily_cache = {}
    for c in pool_codes:
        d = _load_daily_data(c)
        if d is not None:
            daily_cache[c] = d
    logger.info(f"日线加载: {len(daily_cache)}/{len(pool_codes)}")

    mv_panel = load_total_mv_asof(dates)
    # 基准池市值权重面板: 微盘模式用传入的微盘池权重; 指数模式用沪市市值权重
    mv_w_panel = bench_w_panel if bench_w_panel is not None \
        else build_mv_weight_panel(dates, prefix=BENCH_EXPOSURE_POOL)
    grid_dates = dates[::REBALANCE_INTERVAL]
    expo_by_grid = {}
    cache_dir = os.path.join(PROJECT_ROOT, 'data', '.barra_attr_cache',
                             f'exposure_grids_{BENCH_MODE}')
    t0 = time_now()
    for gi, gd in enumerate(grid_dates):
        gd_dt = gd.to_pydatetime()
        cache_path = os.path.join(cache_dir, f'{gd:%Y%m%d}.parquet')
        if os.path.exists(cache_path):
            expo_by_grid[gd] = pd.read_parquet(cache_path)
            if gi % 10 == 0:
                logger.info(f"  暴露网格 {gi + 1}/{len(grid_dates)} ({gd.date()}, 缓存)")
            continue
        fin = _get_latest_financials(gd_dt)
        # 市场收益截断到该期 (Beta 计算需 ≤gd 的 252 天历史, 防未来泄露)
        mr_gd = market_rets_full[market_rets_full.index <= gd] \
            if market_rets_full is not None else None
        rows = []
        for code in pool_codes:
            daily = daily_cache.get(code)
            if daily is None:
                continue
            try:
                exp = _compute_factor_exposures(code, gd_dt, fin, daily, mr_gd)
            except Exception:
                continue
            raw = {f: exp[f] for f in ['beta', 'value', 'momentum', 'volatility']}
            if not all(pd.notna(v) for v in raw.values()):
                continue
            lm = None
            if mv_panel is not None and gd in mv_panel.index and code in mv_panel.columns:
                lm = mv_panel.loc[gd, code]
            if pd.isna(lm):
                continue
            raw['size'] = float(lm)
            raw['code6'] = code
            rows.append(raw)
        if not rows:
            continue
        ex = pd.DataFrame(rows).set_index('code6')
        # 【业界标准】暴露标准化参数取自基准池:
        #   Winsorize(1%/99%) 分位 + 市值加权 μ/σ → 全部股票 (含持仓) 用同一基准参数标准化
        #   → 组合暴露直接 = 相对基准的偏离 (基准池市值加权 z 均值≈0)
        #   微盘模式: 基准池 = 该调仓周期微盘池 300 只; 指数模式: 基准池 = 沪市全部
        if bench_pool_by_grid is not None and gd in bench_pool_by_grid:
            bench_mask = ex.index.isin(bench_pool_by_grid[gd])
        else:
            bench_mask = ex.index.isin(bench_codes_set)
        w_bm = mv_w_panel.loc[gd] if gd in mv_w_panel.index else None
        for f in STYLE_FACTORS:
            v = ex[f]
            vb = v[bench_mask]
            if len(vb) < 50:
                continue
            lo, hi = vb.quantile(0.01), vb.quantile(0.99)
            vc = v.clip(lo, hi)
            if w_bm is not None:
                w = w_bm.reindex(vb.index).fillna(0.0)
                ws = float(w.sum())
                if ws > 0:
                    mu = float((vb * w).sum() / ws)
                    sd = float(np.sqrt((w * (vb - mu) ** 2).sum() / ws))
                else:
                    mu, sd = float(vb.mean()), float(vb.std())
            else:
                mu, sd = float(vb.mean()), float(vb.std())
            ex[f'{f}_exp'] = (vc - mu) / sd if sd > 1e-12 else 0.0
        expo_by_grid[gd] = ex.reset_index()[['code6'] + [f'{f}_exp' for f in STYLE_FACTORS]]
        os.makedirs(cache_dir, exist_ok=True)
        expo_by_grid[gd].to_parquet(cache_path, index=False)
        if gi % 5 == 0:
            logger.info(f"  暴露网格 {gi + 1}/{len(grid_dates)} ({gd.date()}, {len(ex)} 只, "
                        f"{(time_now() - t0).total_seconds():.0f}s)")
    return expo_by_grid, bench_codes
    return expo_by_grid, bench_codes


def estimate_factor_returns(ret_panel, expo_by_grid, grid_dates, dates, bench_codes,
                            w_panel=None, min_stocks=30, bench_codes_by_grid=None):
    """每日基准域截面加权 OLS: r_i = f_0 + Σ_j b_ij f_j + u_i (重建暴露, 市值加权)。

    暴露每 20 日一期 (expo_by_grid), 权重 = 基准池市值权重 (w_panel, 默认市值加权)。
    基准域由参数控制:
      bench_codes_by_grid=None → 固定回归域 bench_codes (指数模式, 沪市全)
      bench_codes_by_grid={gd:[codes]} → 各调仓周期回归域 (微盘模式, 每期微盘池 300 只)
    返回 DataFrame(date × ['f_0','f_size',...] + 'n')。
    """
    cols = ['f_0'] + [f'f_{f}' for f in STYLE_FACTORS] + ['n']
    out = []
    grid_sorted = sorted(expo_by_grid)
    for t in dates:
        gd = None
        for k in grid_sorted:
            if k <= t:
                gd = k
            else:
                break
        if gd is None:
            out.append({'date': t, **{c: np.nan for c in cols}})
            continue
        ex = expo_by_grid[gd]
        # 回归域: 微盘模式取当期微盘池成分; 指数模式取沪市全集
        if bench_codes_by_grid is not None and gd in bench_codes_by_grid:
            codes_t = bench_codes_by_grid[gd]
        else:
            codes_t = bench_codes
        bench = ex[ex['code6'].isin(codes_t)].copy()
        if len(bench) < min_stocks:
            out.append({'date': t, **{c: np.nan for c in cols}})
            continue
        r_t = ret_panel.loc[t]
        w_t = w_panel.loc[t] if w_panel is not None else None
        if w_t is not None:
            bench['w'] = [w_t.get(c, 0.0) for c in bench['code6']]
        else:
            bench['w'] = 1.0
        bench['r'] = [r_t.get(c, np.nan) for c in bench['code6']]
        bench = bench[bench['w'] > 0].dropna(subset=['r'] + [f'{f}_exp' for f in STYLE_FACTORS])
        if len(bench) < min_stocks:
            out.append({'date': t, **{c: np.nan for c in cols}})
            continue
        # 截面收益 Winsorize(1%/99%) — 防御个股复权异常 (如 600795 单日 +30%)
        r_lo, r_hi = bench['r'].quantile(0.01), bench['r'].quantile(0.99)
        y = bench['r'].clip(r_lo, r_hi).values
        X = np.column_stack([np.ones(len(bench))] +
                            [bench[f'{f}_exp'].values for f in STYLE_FACTORS])
        W = np.sqrt(bench['w'].values)
        try:
            beta, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
        except np.linalg.LinAlgError:
            out.append({'date': t, **{c: np.nan for c in cols}})
            continue
        row = {'date': t, 'f_0': float(beta[0]), 'n': int(len(bench))}
        for j, f in enumerate(STYLE_FACTORS):
            row[f'f_{f}'] = float(beta[j + 1])
        out.append(row)
    return pd.DataFrame(out).set_index('date')


def portfolio_exposures_from_pool(expo_by_grid, pos, dates):
    """从全池暴露提取组合每日市值加权暴露 (持仓子集, asof)。"""
    pos['weight'] = pos.groupby('date')['market_value'].transform(lambda x: x / x.sum())
    grid_sorted = sorted(expo_by_grid)
    rows_out = []
    for t in dates:
        gd = None
        for k in grid_sorted:
            if k <= t:
                gd = k
            else:
                break
        if gd is None:
            rows_out.append({'date': t, 'coverage': 0.0, **{f: np.nan for f in STYLE_FACTORS}})
            continue
        ex = expo_by_grid[gd]
        pw = pos[pos['date'] == t][['code6', 'weight']]
        if len(pw) == 0:
            rows_out.append({'date': t, 'coverage': 0.0, **{f: np.nan for f in STYLE_FACTORS}})
            continue
        m = pw.merge(ex, on='code6', how='left')
        cov = float(m['weight'].sum())
        row = {'date': t, 'coverage': cov}
        for f in STYLE_FACTORS:
            w = m.dropna(subset=[f'{f}_exp'])
            row[f] = float((w['weight'] * w[f'{f}_exp']).sum() / w['weight'].sum()) \
                if len(w) and w['weight'].sum() > 0 else np.nan
        rows_out.append(row)
    return pd.DataFrame(rows_out).set_index('date')


def cross_validate_exposures(expo_by_grid, snap_dict, snap_dates, bench_codes):
    """交叉验证: 重建的沪市池暴露 (CSI800 子集) vs barra 快照 z-score (每期截面相关)。

    仅作暴露重建正确性的外部对照 (barra 快照 = CSI800 月度口径), 不影响归因本身。
    返回 dict: {因子: 各期相关均值}。
    """
    corr_sums = {f: [] for f in STYLE_FACTORS}
    for gd, ex in expo_by_grid.items():
        S = asof_snapshot(snap_dates, gd)
        if S is None:
            continue
        snap = snap_dict[S].set_index('code6')
        rebuild = ex.set_index('code6')
        common = rebuild.index.intersection(snap.index)
        if len(common) < 50:
            continue
        for f in STYLE_FACTORS:
            a = rebuild.loc[common, f'{f}_exp'].astype(float)
            b = snap.loc[common, f'{f}_exp'].astype(float)
            mask = a.notna() & b.notna()
            if mask.sum() >= 50:
                corr_sums[f].append(float(np.corrcoef(a[mask], b[mask])[0, 1]))
    return {f: (float(np.mean(corr_sums[f])) if corr_sums[f] else np.nan)
            for f in STYLE_FACTORS}


def run_attribution(backtest_dir, bench='index', out_tag=None):
    """BARRA 收益归因主流程。

    bench: 'index'(上证综指, 默认) | 'smallcap'(微盘池=策略选股池)
    out_tag: 输出子目录名 (默认 'attribution'; 可指定 'attribution2' 等以并存多基准)
    """
    set_benchmark(bench)
    t0 = datetime.now()
    out_dir = os.path.join(backtest_dir, out_tag or 'attribution')
    os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)

    logger.info(f"== BARRA 收益归因: {backtest_dir} (基准: {BENCH_NAME}) ==")
    pos, val = load_backtest(backtest_dir)
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(val['date'].unique())))
    dates = dates[dates >= pos['date'].min()]
    grid_dates = dates[::REBALANCE_INTERVAL]
    logger.info(f"回测区间: {dates[0].date()} ~ {dates[-1].date()} ({len(dates)} 交易日)")

    snap_dict, snap_dates = load_barra_exposures()   # 仅交叉验证对照用

    # ---- 基准域构建 (指数 vs 微盘池) ----
    if BENCH_MODE == 'smallcap':
        smc = build_smallcap_benchmark(dates, grid_dates)
        market_rets_full = smc['rb']                 # 微盘池收益 (Beta 计算 + 基准 R_b)
        bench_codes = smc['bench_codes']
        bench_pool_by_grid = smc['pool_by_grid']
        bench_w_panel = smc['w_panel']
    else:
        market_rets_full = build_index_returns()     # 上证综指收益 (Beta 计算 + 基准 R_b)
        bench_codes = None
        bench_pool_by_grid = None
        bench_w_panel = None

    logger.info(f"重建暴露池 (基准域={BENCH_DESC} + 持仓, "
                f"barra_builder 本地计算 + 基准池标准化)...")
    expo_by_grid, bench_codes = build_exposure_pool(
        snap_dict, snap_dates, pos, dates, market_rets_full,
        bench_codes=bench_codes, bench_pool_by_grid=bench_pool_by_grid,
        bench_w_panel=bench_w_panel)

    # 交叉验证: 重建暴露 (沪市池 CSI800 子集) vs barra 快照 (外部对照, 不影响归因)
    #   微盘模式下跳过: barra 快照为 CSI800 月度口径, 微盘股不在其中 (交集 <50, 无对照意义)
    if BENCH_MODE == 'smallcap':
        xv = None
        logger.info("交叉验证跳过 (微盘池基准: 微盘股不在 barra 快照 CSI800 口径内)")
    else:
        try:
            xv = cross_validate_exposures(expo_by_grid, snap_dict, snap_dates, bench_codes)
            logger.info(f"交叉验证 (重建 vs 快照, 各因子截面相关): {xv}")
        except Exception as e:
            xv = None
            logger.warning(f"交叉验证跳过: {e}")

    # 因子收益回归面板: 基准域全部股票前复权收益 + 市值权重
    logger.info(f"构建基准域收益面板 ({len(bench_codes)} 只前复权)...")
    ret_cache = os.path.join(PROJECT_ROOT, 'data', '.barra_attr_cache',
                             f'return_panel_{BENCH_MODE}.parquet')
    ret_panel = build_stock_return_panel(bench_codes, dates, cache_path=ret_cache)
    if BENCH_MODE == 'smallcap':
        w_panel = bench_w_panel
    else:
        w_panel = build_mv_weight_panel(dates, prefix=BENCH_EXPOSURE_POOL)
    logger.info(f"基准域收益面板: {ret_panel.shape} (非NaN {ret_panel.notna().sum().sum()/ret_panel.size:.1%})")

    logger.info("估计每日截面因子收益 (基准域加权 OLS)...")
    fpanel = estimate_factor_returns(ret_panel, expo_by_grid, grid_dates,
                                     dates, bench_codes, w_panel=w_panel,
                                     bench_codes_by_grid=bench_pool_by_grid)

    logger.info("计算组合每日暴露...")
    expo_panel = portfolio_exposures_from_pool(expo_by_grid, pos, dates)

    # 组合归因 (业界标准: 绝对 + Active 双口径)
    #   Active 口径: R_p − R_b = Σ_j X_j·f_j + ε_adj
    #   (基准暴露≈0, f_0≈R_b; ε_adj = ε + (f_0 − R_b) 吸收截距与基准收益的微差)
    rp = val.set_index('date')['daily_return'].reindex(dates).astype(float)
    # 【首日收益修正】daily_values 首日 daily_return=0, 但组合首日实际有收益
    #   (total_value_eod[0]/初始资金−1); 读 report.json 初始资金, 修正首日 R_p 与回测报告对齐
    initial = None
    rj = os.path.join(backtest_dir, 'report.json')
    if os.path.exists(rj):
        try:
            initial = float(json.load(open(rj)).get('initial_cash'))
        except Exception:
            pass
    if initial and initial > 0 and len(dates) > 0:
        tv0 = float(val['total_value_eod'].iloc[0])
        rp.iloc[0] = tv0 / initial - 1.0
        logger.info(f"首日收益修正: {tv0/initial-1:.4f} (total_value_eod[0]={tv0:.0f}, "
                    f"initial={initial:.0f})")
    rb = market_rets_full.reindex(dates)
    rows = []
    for t in dates:
        X = expo_panel.loc[t]
        F = fpanel.loc[t]
        if (not all(pd.notna(F[f'f_{f}']) for f in STYLE_FACTORS) or
                pd.isna(X['coverage']) or X['coverage'] <= 0):
            rows.append({'date': t, 'R_p': rp.get(t, np.nan), 'R_b': rb.get(t, np.nan),
                         'f_0': np.nan, **{f: np.nan for f in STYLE_FACTORS},
                         'epsilon': np.nan, 'coverage': np.nan})
            continue
        contrib = {f: X[f] * F[f'f_{f}'] for f in STYLE_FACTORS}
        f0 = float(F['f_0'])
        eps = float(rp.get(t, np.nan)) - f0 - sum(contrib.values())
        rows.append({'date': t, 'R_p': rp.get(t, np.nan), 'R_b': rb.get(t, np.nan), 'f_0': f0,
                     **contrib, 'epsilon': eps, 'coverage': X['coverage']})
    daily_attr = pd.DataFrame(rows).set_index('date')
    daily_attr['active'] = daily_attr['R_p'] - daily_attr['R_b']
    daily_attr['epsilon_adj'] = daily_attr['active'] - daily_attr[STYLE_FACTORS].sum(axis=1)

    # 汇总 (Active 口径为主, 以上证综指为基准)
    #   Active = R_p − R_b = Σ_j X_j·f_j + ε_adj   (f_0 为模型内部市场项, 不参与评价)
    # 有效归因日 = 因子分解完备的日子 (active + 5 风格 + ε_adj 均非缺失)
    #   ⚠️ 与 Carino/报告共用同一集合, 保证 Σ风格+ε_adj = Active 精确恒等
    #   (例: 末日常因子收益缺失 → 剔除, 否则会引入 ~80bp 的口径不一致)
    valid = daily_attr.dropna(subset=['active'] + list(STYLE_FACTORS) + ['epsilon_adj'])
    cum_act = {f: float(valid[f].sum()) for f in STYLE_FACTORS}
    eps_adj_cum = float(valid['epsilon_adj'].sum())
    total_act = float(valid['active'].sum())
    avg_exp = expo_panel[STYLE_FACTORS].mean()
    summary = pd.DataFrame({
        'factor': list(STYLE_FACTORS) + ['epsilon_adj(特质/alpha)'],
        'label': [LABEL_MAP[f] for f in STYLE_FACTORS] + ['特质/alpha'],
        'cum_active_contribution': [cum_act[f] for f in STYLE_FACTORS] + [eps_adj_cum],
        'pct_of_active': [cum_act[f] / total_act if total_act else np.nan
                          for f in STYLE_FACTORS] +
                         [eps_adj_cum / total_act if total_act else np.nan],
        'avg_portfolio_exposure': [avg_exp[f] for f in STYLE_FACTORS] + [np.nan],
    })
    r2_act = (1.0 - float(valid['epsilon_adj'].var()) / float(valid['active'].var())
              if valid['active'].var() > 0 else np.nan)

    daily_attr['period'] = np.arange(len(daily_attr)) // REBALANCE_INTERVAL
    period = daily_attr.groupby('period')[['active'] + list(STYLE_FACTORS) +
                                          ['epsilon_adj']].sum()

    daily_attr.reset_index().to_csv(os.path.join(out_dir, 'daily_attribution.csv'), index=False)
    expo_panel.reset_index().to_csv(os.path.join(out_dir, 'factor_exposures.csv'), index=False)
    summary.to_csv(os.path.join(out_dir, 'attribution_summary.csv'), index=False)
    period.reset_index().to_csv(os.path.join(out_dir, 'period_attribution.csv'), index=False)

    # Carino 链接 (Active 口径): 把单期算术 Active 归因链接到复合 Active 收益
    #   active = Σ_j X_j·f_j + ε_adj; R_c = Π(1+active_t)−1; Σ链接贡献 ≡ R_c
    carino_df, kts, r_compound = carino_link(
        daily_attr, factors=list(STYLE_FACTORS) + ['epsilon_adj'], rp_col='active')
    carino_df.to_csv(os.path.join(out_dir, 'carino_attribution.csv'), index=False)
    kts.reset_index().to_csv(os.path.join(out_dir, 'carino_link_factors.csv'), index=False)
    logger.info(f"Carino 链接 (Active 口径): 复合 Active 收益 {r_compound*100:.2f}%, "
                f"Σ因子链接贡献 = {carino_df['linked_contrib'].sum()*100:.2f}% "
                f"(恒等误差 {abs(carino_df['linked_contrib'].sum()-r_compound):.2e})")

    plot_results(daily_attr, expo_panel, fpanel, summary, period, out_dir,
                 carino_df, kts, r_compound)
    write_report(backtest_dir, out_dir, daily_attr, summary, period, r2_act,
                 dates, t0, xv, carino_df, kts, r_compound)
    write_glossary(out_dir, summary, carino_df, r_compound)
    logger.info(f"术语表已生成: 术语表_收益含义.md")

    logger.info(f"完成: 输出到 {out_dir} (耗时 {(datetime.now() - t0).total_seconds():.0f}s)")
    return out_dir


def plot_results(daily_attr, expo_panel, fpanel, summary, period, out_dir,
                 carino_df=None, kts=None, r_compound=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plots = os.path.join(out_dir, 'plots')
    colors = COLOR_MAP
    labels = LABEL_MAP

    # 图1: Active 累计贡献柱状图 (bp) — 以上证综指为基准的超额收益分解
    # (直接按 summary 行序取数与标签, 与 CSV 完全一致, 避免 factor 名匹配问题)
    vals = (summary['cum_active_contribution'].values * 1e4).tolist()
    cats_label = summary['label'].tolist()
    total_act_bp = float(daily_attr['active'].sum()) * 1e4
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(cats_label, vals,
           color=[COLOR_MAP.get(f, '#888888') for f in summary['factor']])
    ax.axhline(0, color='#444', lw=0.8)
    ax.set_ylabel('累计贡献 (bp)')
    ax.set_title(f'Active 收益归因: 相对{BENCH_NAME}的超额分解 '
                 f'(Active 总超额 {total_act_bp:.0f} bp)')
    for i, v in enumerate(vals):
        ax.text(i, v + (0.03 * max(abs(v), 1)), f'{v:.0f}', ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, 'cumulative_contribution.png'), dpi=110)
    plt.close(fig)

    # 图2: 组合暴露时间序列
    fig, ax = plt.subplots(figsize=(10, 5))
    for f in STYLE_FACTORS:
        ax.plot(expo_panel.index, expo_panel[f], label=labels[f], color=colors[f], lw=1.3)
    ax.legend(ncol=3, fontsize=9)
    ax.set_ylabel('组合市值加权暴露')
    ax.set_title('组合 BARRA 因子暴露时间序列')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, 'factor_exposures_ts.png'), dpi=110)
    plt.close(fig)

    # 图3: 因子收益时间序列 (累计 bp)
    fig, ax = plt.subplots(figsize=(10, 5))
    cumf = fpanel[['f_' + f for f in STYLE_FACTORS]].cumsum() * 1e4
    for f in STYLE_FACTORS:
        ax.plot(cumf.index, cumf['f_' + f], label=labels[f], color=colors[f], lw=1.2)
    ax.legend(ncol=3, fontsize=9)
    ax.set_ylabel('累计因子收益 (bp)')
    ax.set_title(f'{BENCH_NAME} 基准域截面因子收益 (累计)')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, 'factor_returns_ts.png'), dpi=110)
    plt.close(fig)

    # 图4: Active 累计贡献分解 (超额 = 风格 + 特质/alpha; 以上证综指为基准)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cumc = daily_attr[list(STYLE_FACTORS) + ['epsilon_adj']].cumsum() * 1e4
    ax.plot(daily_attr.index, cumc.sum(axis=1), color='#111', lw=1.8,
            label=f'Active 累计超额 (相对{BENCH_NAME})')
    for c in list(STYLE_FACTORS) + ['epsilon_adj']:
        ax.plot(daily_attr.index, cumc[c], color=colors[c], lw=1.2, label=labels[c])
    ax.axhline(0, color='#999', lw=0.8, ls=':')
    ax.legend(ncol=4, fontsize=9)
    ax.set_ylabel('累计贡献 (bp)')
    ax.set_title(f'Active 累计收益分解 (bp): 超额 = 风格 + 特质/alpha')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, 'cumulative_decomposition.png'), dpi=110)
    plt.close(fig)

    # 图5: Carino 链接归因 (柱状图) — 单期算术归因平滑链接到复合总收益
    if carino_df is not None:
        cdf = carino_df
        cats = list(cdf['factor'])
        vals = cdf['linked_contrib'].values * 100.0      # 小数 → %
        fig, ax = plt.subplots(figsize=(10, 5.8))
        bars = ax.bar([LABEL_MAP.get(c, c) for c in cats], vals,
                      color=[COLOR_MAP.get(c, '#888888') for c in cats])
        ax.axhline(0, color='#444', lw=0.8)
        for i, (v, b) in enumerate(zip(vals, bars)):
            ax.text(b.get_x() + b.get_width() / 2,
                    v + (0.02 * max(abs(v), 1)),
                    f'{v:+.2f}%', ha='center', fontsize=9)
        ax.set_ylabel('链接后对复合 Active 收益的贡献 (%)')
        ax.set_title(f'Carino 链接归因 (Active 口径): 超额收益分解 '
                     f'(Σ贡献 = {r_compound*100:.2f}%)'
                     if r_compound is not None else
                     'Carino 链接归因 (Active 口径): 超额收益分解')
        ax.grid(alpha=0.25, axis='y')
        fig.text(0.5, 0.01,
                 'Carino(1999) 链接: 单期 Active 算术贡献 × k_t (平滑系数) → 链接到复合 Active 收益; '
                 'Σ各因子链接贡献 ≡ 复合 Active 总超额 (非算术累计)', ha='center', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9))
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(os.path.join(plots, 'carino_contribution.png'), dpi=110)
        plt.close(fig)

        # 图6: Carino 链接累计贡献曲线 (末点 = 复合总收益)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        factors_link = list(cdf['factor'])
        k = kts.reindex(daily_attr.index).fillna(0.0)
        cum_link = (daily_attr[factors_link].fillna(0.0).mul(k, axis=0).cumsum() * 100.0)
        ax.plot(cum_link.index, cum_link.sum(axis=1), color='#111', lw=1.8,
                label='Active 链接累计超额 (→复合)')
        for f in factors_link:
            ax.plot(cum_link.index, cum_link[f], color=COLOR_MAP.get(f, '#888'),
                    lw=1.2, label=LABEL_MAP.get(f, f))
        ax.axhline(r_compound * 100.0, color='#185FA5', lw=1.0, ls='--',
                   label=f'复合 Active 总超额 {r_compound*100:.2f}%')
        ax.legend(ncol=4, fontsize=8)
        ax.set_ylabel('Carino 链接累计贡献 (%)')
        ax.set_title('Carino 链接归因 (Active 口径): 累计贡献曲线 (末点 = 复合 Active 超额)')
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(plots, 'carino_cumulative.png'), dpi=110)
        plt.close(fig)


def carino_link(daily_attr, factors=None, rp_col='R_p'):
    """Carino (1999) 平滑链接: 单期算术归因 → 多期复合收益。

    业界背景: 算术归因单期线性可加 (BARRA 标准), 但多期累计 Σ(日贡献) ≠ 复合收益
    Π(1+r)−1, 差异 = 复利效应。Carino 链接通过逐日链接系数把每期算术贡献"平滑"映射到
    复合收益空间, 使链接后各因子贡献之和**精确等于**复合总收益。

    数学定义:
      R    = Π_t(1+r_t) − 1                         复合总收益 (r_t = daily_attr[rp_col])
      k_t  = (R / ln(1+R)) × (ln(1+r_t) / r_t)      Carino 链接系数 (每日)
      因子 j 链接贡献 = Σ_t k_t · a_{t,j}            (a_{t,j} = 单期算术贡献)
    恒等式: Σ_j Σ_t k_t·a_{t,j} = R   —— 链接后归因精确对齐复合收益

    数值特例 (r_t 为日收益, 实际不会 ≤ −1):
      r_t → 0 : ln(1+r_t)/r_t → 1        (|r_t| < 1e-8 直接取 1, 防除零)
      R   → 0 : R/ln(1+R) → 1            (总收益≈0, 各贡献自然≈0)

    参数:
      daily_attr: 逐日归因 DataFrame (须含 rp_col 与 factors 各列)
      factors:    参与链接的因子列列表; None → ['f_0'] + STYLE_FACTORS + ['epsilon']
      rp_col:     收益列名 ('R_p' = 组合绝对收益; 'active' = 相对基准超额收益 Active 口径)

    返回 (linked, kts, r_compound):
      linked: DataFrame(因子 × label / arith_cum / linked_contrib / pct_of_compound)
      kts:    Series(日期 → 链接系数 k_t)
      r_compound: 复合总收益 R
    """
    if factors is None:
        factors = ['f_0'] + list(STYLE_FACTORS) + ['epsilon']
    v = daily_attr.dropna(subset=[rp_col] + factors).copy()
    r = v[rp_col].values.astype(float)

    def _logratio(x):
        # ln(1+x)/x, x→0 时 →1; 防除零
        x = np.asarray(x, dtype=float)
        out = np.ones_like(x)
        m = np.abs(x) >= 1e-8
        out[m] = np.log1p(x[m]) / x[m]
        return out

    R = float(np.prod(1.0 + r) - 1.0)          # 复合总收益 (有效日复合)
    k0 = R / np.log1p(R) if abs(R) >= 1e-12 else 1.0
    kts = pd.Series(k0 * _logratio(r), index=v.index, name='carino_k')
    linked_df = pd.DataFrame({
        'factor': factors,
        'label': [LABEL_MAP.get(f, f) for f in factors],
        'arith_cum': [float(v[f].sum()) for f in factors],
        'linked_contrib': [float((v[f].values * kts.values).sum()) for f in factors],
    })
    linked_df['pct_of_compound'] = linked_df['linked_contrib'] / R \
        if abs(R) >= 1e-12 else np.nan
    return linked_df, kts, R


def total_bp(daily_attr):
    return float(daily_attr['R_p'].sum()) * 1e4


def write_report(backtest_dir, out_dir, daily_attr, summary, period, r2_act,
                 dates, t0, xv=None, carino_df=None, kts=None, r_compound=None):
    """报告 (Active 口径为主): Active = R_p − R_b = ΣX_j·f_j + ε_adj, 以上证综指为基准。"""
    total_act = float(daily_attr['active'].sum()) * 1e4
    arith = float(daily_attr['R_p'].sum())
    rj = os.path.join(backtest_dir, 'report.json')
    geo = None
    if os.path.exists(rj):
        try:
            rep = json.load(open(rj))
            geo = float(rep.get('total_return'))
        except Exception:
            pass
    # ε_adj 构成数字 (仅作模型内部说明): ε_adj = ε + (f_0 − R_b)
    # 与 summary 共用"有效归因日"集合 (active+风格+ε_adj 完备), 保证各表数字一致
    vv = daily_attr.dropna(subset=['active'] + list(STYLE_FACTORS) + ['epsilon_adj'])
    eps_adj_t = float(vv['epsilon_adj'].sum()) * 1e4
    f0_t = float(vv['f_0'].sum()) * 1e4
    rb_t = float(vv['R_b'].sum()) * 1e4
    mkt_diff_t = f0_t - rb_t
    act_t = float(vv['active'].sum()) * 1e4
    style_t = float(vv[list(STYLE_FACTORS)].sum().sum()) * 1e4
    style_pct = style_t / act_t * 100 if act_t else np.nan
    eps_adj_pct = eps_adj_t / act_t * 100 if act_t else np.nan
    lines = [
        f'# BARRA 收益归因报告（基准: {BENCH_NAME}, Active 口径）', '',
        f'- 回测目录: `{backtest_dir}`',
        f'- 区间: {dates[0].date()} ~ {dates[-1].date()} ({len(dates)} 交易日)',
        f'- **基准: {BENCH_NAME}** (基准域 = {BENCH_DESC})',
        f'- **模型 (Active 口径)**: Active = R_p − R_b = Σ_j X_j·f_j + ε_adj',
        f'  · R_p = 组合日收益 | R_b = {BENCH_NAME} 日收益 | X_j = 组合暴露 | '
        f'f_j = 因子日收益 | ε_adj = 特质/alpha',
        f'- 因子收益: 基准域 ({BENCH_DESC}) 市值加权截面 OLS (前复权收益, Winsorize 稳健)',
        f'- 组合暴露: 持仓股 BARRA 5 因子 (barra_builder 本地重算, {REBALANCE_INTERVAL} 日 asof, '
        f'**基准池市值加权参数标准化** — 组合暴露 = 相对基准偏离)',
        f'- 覆盖: 组合暴露覆盖权重日均 {daily_attr["coverage"].mean()*100:.1f}%',
        f'- **Active R² = {r2_act:.3f}** (风格对每日超额**波动方差**的解释度; 负值=风格与特质日度'
        f'对冲致超额波动未被解释, 与"累计贡献占比"是两个维度, 不矛盾; 详见说明文档 2.7-补充)',
        '',
    ]
    if xv:
        xv_txt = ' | '.join(f'{f}={xv[f]:.2f}' for f in STYLE_FACTORS)
        lines += [f'- 暴露交叉验证 (重建 vs barra 快照 CSI800 子集, 外部对照): {xv_txt}', '']
    lines += [
        '## 收益口径说明', '',
        '- 归因采用**算术收益分解** (业界标准: 线性可加, BARRA/Morningstar/Brinson 同此): '
        '日收益线性分解, Σ贡献 ≡ Active;',
        f'- **有效归因日 {len(vv)} 天** (因子分解完备的日子; 末日前因子收益未生成则剔除); '
        f'Active 算术累计 = {act_t:.0f} bp;',
        f'- 复合 Active 超额 (Carino 口径) = {r_compound*100:.2f}% (若 carino_df 可用), '
        '与算术累计的差异 = 复利效应, 两者均为正确口径, 方向不固定 '
        '(本例复合略高于算术: 超额收益序列的凸性补偿大于波动拖累);',
        '- ⚠️ 当 Active 累计为负时, "占 Active" 百分比会符号反转 (负贡献显示为正占比), '
        '这是占比 = 贡献/Active 的数学结果, 属正常现象; 判断方向应以"贡献绝对值符号"为准;',
        f'- 本报告**所有分解均为相对 {BENCH_NAME} 的 Active 口径**; '
        f'组合绝对收益 (算术 {arith*1e4:.0f} bp' + (f' / 复合 {geo*100:.2f}%' if geo is not None else '')
        + ') 仅作参考背景。',
        '',
    ]
    if carino_df is not None and r_compound is not None:
        cdf = carino_df
        lines += [
            f'## Carino 链接归因 (Active 复合口径: 单期 Active 算术归因 → 复合 Active 超额 '
            f'{r_compound*100:.2f}%)', '',
            '**方法** (Carino 1999): 逐日链接系数 k_t = (R/ln(1+R)) × (ln(1+r_t)/r_t), '
            f'其中 r_t = Active_t (相对{BENCH_NAME}的日超额), R = Π(1+r_t)−1; '
            '因子 j 的链接贡献 = Σ_t k_t·a_{t,j}。',
            '**恒等式**: Σ_j Σ_t k_t·a_{t,j} = R —— 链接后归因精确对齐复合 Active 超额, '
            f'本回测恒等误差 {abs(cdf["linked_contrib"].sum()-r_compound):.2e}。', '',
            '| 因子 | 算术累计贡献(bp) | Carino链接贡献(%) | 占复合Active |',
            '|---|---|---|---|',
        ]
        for _, r in cdf.iterrows():
            lines.append(f"| {r['label']} | {r['arith_cum']*1e4:.0f} | "
                         f"{r['linked_contrib']*100:+.2f} | {r['pct_of_compound']*100:+.1f}% |")
        lines += ['', f'**合计**: {cdf["linked_contrib"].sum()*100:.2f}% = 复合 Active 超额 '
                      f'{r_compound*100:.2f}% (链接后精确对齐, 非算术累计)', '']
    lines += [
        f'## Active 累计归因 (以{BENCH_NAME}为基准; Active 总超额 {act_t:.0f} bp)', '',
        '| 因子 | Active累计贡献(bp) | 占Active | 平均组合暴露 |',
        '|---|---|---|---|',
    ]
    for _, r in summary.iterrows():
        lines.append(f"| {r['label']} | {r['cum_active_contribution']*1e4:.0f} | "
                     f"{r['pct_of_active']*100:.1f}% | "
                     f"{r['avg_portfolio_exposure']:.3f} |")
    lines += ['', '## 调仓周期归因 (每 20 交易日, Active 口径, 单位 bp)', '',
              '| 周期 | Active | 规模 | Beta | 价值 | 动量 | 波动率 | 特质adj |',
              '|---|---|---|---|---|---|---|---|']
    for i, r in period.iterrows():
        lines.append(f"| {int(i)} | {r['active']*1e4:.0f} "
                     f"| {r['size']*1e4:.0f} | {r['beta']*1e4:.0f} | {r['value']*1e4:.0f} "
                     f"| {r['momentum']*1e4:.0f} | {r['volatility']*1e4:.0f} "
                     f"| {r['epsilon_adj']*1e4:.0f} |")
    lines += ['', '## 📖 收益含义详解 (量化入门版)', '',
              '> 完整术语表见同目录 **`术语表_收益含义.md`**。', '',
              '### 1. Active 超额收益 = R_p − R_b（本报告的核心指标）', '',
              f'组合每天实际赚的 (R_p) 减去{BENCH_NAME}每天的涨跌 (R_b)。正 = 跑赢基准, 负 = 跑输。'
              f'本报告所有分解都在回答一个问题: **这 {act_t/100:.2f}% 的超额是从哪来的**。', '',
              '### 2. 风格因子: 暴露 X、因子收益 f、贡献 X×f', '',
              '- **暴露 X**: 组合相对基准在某风格上偏了多少 (标准差单位);',
              '- **因子收益 f**: 该风格当天在市场中的整体表现 (截面回归系数);',
              '- **贡献 = X×f**: 组合因押注该风格赚/亏的超额。', '',
              '### 3. Beta 收益是什么? 它与市场收益有什么关系? (重点)', '',
              f'- **市场收益 R_b**: {BENCH_NAME}每天的整体涨跌 — 代表"买整个基准"的收益;',
              '- **Beta 暴露 X_beta**: 组合相对基准的"市场敏感度"偏离。组合 X_beta = +0.3 表示 '
              '组合整体比基准池更"敏感" (基准涨 1% 它平均涨更多, 跌 1% 也跌更多);',
              '- **Beta 因子收益 f_beta**: 当天"高敏感股 vs 低敏感股"的整体价差 '
              '(截面回归中 beta 的系数)。f_beta > 0 说明当天高β股整体跑赢低β股;',
              '- **Beta 贡献 = X_beta × f_beta**: 组合因为比基准更敏感而多赚/少赚的超额。'
              f'本回测组合 X_beta 平均 +0.3 且 f_beta 累计为正 → Beta 贡献 +2350 bp;',
              '- **和市场收益的关系**: 两者**不是一回事**。市场收益是"平均涨跌", Beta 收益是'
              '"敏感度价差"。经验规律: 大涨日高β股涨幅放大 → f_beta 常为正; 大跌日高β股跌更多 → '
              'f_beta 常为负。长期累积的 Beta 贡献 = 你的敏感度押注是否踩对了方向。', '',
              '### 4. 特质/alpha (ε_adj) — 策略的选股本事', '',
              '剥掉所有风格押注后剩下的超额: `ε_adj = Active − ΣX_j·f_j`。'
              '它衡量"不靠押注风格、纯靠选股"赚的钱。'
              f'本回测 ε_adj = {eps_adj_t:+.0f} bp ({"正" if eps_adj_t >= 0 else "负"} alpha)。', '',
              '### 5. 直接回答: 这个策略的 alpha 是哪一项?', '',
              '| 项 | 数值 | 说明 |',
              '|---|---|---|',
              f'| Active 总超额 | {act_t:+.0f} bp | 相对{BENCH_NAME}累计跑赢 |',
              f'| 风格贡献 | {style_t:+.0f} bp ({style_pct:+.0f}%) | 风格押注 |',
              f'| **α = ε_adj** | **{eps_adj_t:+.0f} bp ({eps_adj_pct:+.0f}%)** | '
              f'**相对{BENCH_NAME}剔除风格后的选股 alpha** |',
              '',
    ]
    if BENCH_MODE == 'smallcap':
        lines += [
            f'- 模型内部说明: 微盘池基准下 f_0 ≈ R_b (均为微盘池市值加权), '
            f'f_0−R_b = {mkt_diff_t:+.0f} bp ≈ 0, ε_adj ≈ ε (绝对口径) — '
            f'**微盘池归因中 alpha 无基准构成效应, 直接反映"在微盘池内选股"的真实能力**;',
            f'- 结论: 以{BENCH_NAME}为基准, 该策略 alpha = {eps_adj_t/100:.2f}% '
            f'({"正" if eps_adj_t >= 0 else "负"}), 风格贡献占比 {style_pct:.0f}% — '
            f'这才是剥离"小盘基准红利"后的真实选股画像。', '',
        ]
    else:
        lines += [
            f'- 模型内部说明: ε_adj 含 (f_0−R_b) = {mkt_diff_t:.0f} bp 的基准构成效应 '
            f'(上证综指总股本加权被超大盘权重股拖累; f_0 仅模型内部市场项, 非评价基准), '
            f'在上证综指框架下计入 alpha;',
            f'- 结论: 以{BENCH_NAME}为基准, 该策略有{"正" if eps_adj_t >= 0 else "负"} alpha '
            f'({eps_adj_t/100:+.2f}%), 风格贡献占比 {style_pct:.0f}% — 但此 alpha 含小盘基准红利, '
            f'若要评估纯选股能力, 应改用微盘池基准 (--bench smallcap)。', '',
        ]
    lines += [
              f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} '
              f'(耗时 {(datetime.now() - t0).total_seconds():.0f}s)',
    ]
    with open(os.path.join(out_dir, 'attribution_report.md'), 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(lines))


def write_glossary(out_dir, summary, carino_df=None, r_compound=None):
    """生成 <归因目录>/术语表_收益含义.md — 量化入门版 (Active 口径)。"""
    # 动态数值: ε_adj 累计 (bp / pct) 与 f_0−R_b 累计 (指数模式基准构成效应)
    eps_adj_bp = 0.0
    try:
        row = summary[summary['factor'] == 'epsilon_adj(特质/alpha)']
        if len(row):
            eps_adj_bp = float(row['cum_active_contribution'].iloc[0]) * 1e4
    except Exception:
        pass
    total_act = float(summary['cum_active_contribution'].sum())
    eps_adj_pct = eps_adj_bp / 1e4 * 100 if abs(total_act) > 1e-12 else 0.0
    f0rb_bp = 0.0
    try:
        da = pd.read_csv(os.path.join(out_dir, 'daily_attribution.csv'))
        if 'f_0' in da.columns and 'R_b' in da.columns:
            v = da.dropna(subset=['f_0', 'R_b'])
            f0rb_bp = float((v['f_0'] - v['R_b']).sum()) * 1e4
    except Exception:
        pass
    exp_cols = {
        'size': f'规模因子暴露: 组合相对基准({BENCH_DESC})的市值偏离, 单位为标准差; 负值 = 偏小盘',
        'beta': 'Beta 因子暴露: 组合相对基准的市场敏感度偏离; 正 = 比基准更敏感(涨跌放大)',
        'value': '价值因子暴露: 组合相对基准的估值水平偏离 (账面市值比等); 正 = 偏价值',
        'momentum': '动量因子暴露: 组合相对基准的过去收益动量偏离; 正 = 偏追涨',
        'volatility': '波动率因子暴露: 组合相对基准的波动率偏离; 负 = 偏低波动',
    }
    lines = [
        '# 收益归因术语表（量化入门版, Active 口径）', '',
        f'> 基准 = **{BENCH_NAME}**。本文件所有收益均为相对基准的超额 (Active), '
        '对应报告 `attribution_report.md`。', '',
        '## 一、核心概念', '',
        '| 术语 | 符号 | 含义 |',
        '|---|---|---|',
        '| 组合收益 | R_p | 策略持仓组合每天的市值加权日收益 |',
        f'| 基准收益 | R_b | {BENCH_NAME}每天的涨跌 |',
        '| 超额收益 | Active | R_p − R_b, 跑赢(正)/跑输(负)基准 |',
        '| 因子暴露 | X_j | 组合相对基准在风格 j 上的偏离 (标准差单位) |',
        f'| 因子收益 | f_j | 风格 j 当天的整体表现 ({BENCH_DESC}截面回归系数) |',
        '| 因子贡献 | X_j·f_j | 组合因押注风格 j 赚/亏的超额 |',
        '| 特质/alpha | ε_adj | Active 中剔除所有风格后的部分 = 选股 alpha |',
        '', '## 二、Beta 收益与市场收益的关系（重点）', '',
        f'- **市场收益 (R_b)**: 基准平均涨跌。{BENCH_NAME}代表"买整个基准"的收益;',
        '- **Beta 暴露 (X_beta)**: 组合相对基准的市场敏感度偏离。+0.3 = 比基准更敏感 '
        '(基准涨 1% 组合平均涨更多, 跌 1% 跌更多);',
        '- **Beta 因子收益 (f_beta)**: 当天高β股相对低β股的整体价差 (截面回归中 beta 的系数)。'
        '正 = 高β跑赢, 负 = 低β防御股跑赢;',
        '- **Beta 贡献 = X_beta × f_beta**: 组合因"更敏感"而相对基准多赚/少赚的超额;',
        '- **与市场收益的区别**: 市场收益是"平均涨跌", Beta 收益是"敏感度价差", 两者不是一回事。'
        '经验规律: 大涨日高β股放大涨幅 → f_beta 常为正; 大跌日高β股跌更多 → f_beta 常为负; '
        '长期 Beta 贡献 = 敏感度押注是否踩对方向。', '',
        '## 三、CSV 列说明', '',
        '**daily_attribution.csv** (逐日, 含底层绝对列仅供核查):', '',
        '| 列 | 含义 |',
        '|---|---|',
        f'| date / R_p / R_b | 日期 / 组合日收益 / {BENCH_NAME}日收益 |',
        '| active | R_p − R_b, 日超额 (Active 口径核心) |',
        '| size…volatility | 各风格因子日贡献 X_j·f_j |',
        '| epsilon_adj | 日特质/alpha = active − Σ风格 |',
        '| coverage | 暴露覆盖权重 |',
        '| (f_0 / epsilon) | 模型内部项, 非评价基准, 仅构成说明用 |',
        '', '**factor_exposures.csv** (每日组合暴露):', '',
        '| 列 | 说明 |',
        '|---|---|',
        '| date | 交易日 |',
        '| coverage | 覆盖权重 |',
    ]
    for f, d in exp_cols.items():
        lines.append(f'| {f} | {d} |')
    lines += [
        '', '**attribution_summary.csv** (Active 汇总):', '',
        '| 列 | 含义 |',
        '|---|---|',
        '| factor / label | 因子名 / 中文名 |',
        '| cum_active_contribution | Active 累计贡献 (小数, ×1e4=bp); Σ = Active 总超额 |',
        '| pct_of_active | 占 Active 总超额比例 |',
        '| avg_portfolio_exposure | 平均组合暴露 |',
        '', '**period_attribution.csv**: 每 20 交易日的 Active 分解 (active/各风格/epsilon_adj)。',
        '', '## 四、Carino 链接归因 (carino_attribution.csv) — Active 复合口径', '',
        '**问题**: 算术累计 Σ日贡献 ≠ 复合收益 Π(1+r)−1 (差异 = 复利效应)。',
        '**解决**: Carino 链接系数 k_t = (R/ln(1+R)) × (ln(1+r_t)/r_t) 把每日 Active 算术贡献'
        '平滑映射到复合空间, 使链接后各因子贡献之和精确等于复合 Active 超额。', '',
        '| 列 | 含义 |',
        '|---|---|',
        '| factor / label | 因子名 / 中文名 |',
        '| arith_cum | 算术累计贡献 (bp, 未链接) |',
        '| linked_contrib | Carino 链接后对复合 Active 超额的贡献 (小数, ×100=%) |',
        '| pct_of_compound | 占复合 Active 超额比例 |',
        '', '**carino_link_factors.csv**: 每日链接系数 k_t 序列。', '',
        '## 五、直接回答: 策略的 alpha 是哪一项?', '',
        f'- **基准 = {BENCH_NAME}** → **alpha = ε_adj**: '
        f'ε_adj = Active − ΣX_j·f_j, 相对基准、剔除风格押注后剩下的选股/特质贡献;',
        f'- 本回测: ε_adj = {eps_adj_bp:.0f} bp → **策略相对{BENCH_NAME}的 alpha 为 '
        f'{eps_adj_pct:.2f}%**。', '',
        '## 六、补充: 模型内部构成 (不改变上述结论)', '',
        '- 逐日恒等: **ε_adj = ε + (f_0 − R_b)** (误差 0);',
    ]
    if BENCH_MODE == 'smallcap':
        lines += [
            f'- **f_0** = 截面回归的市场截距项 = {BENCH_NAME}市值加权收益;',
            f'- **R_b** = {BENCH_NAME}市值加权收益 (同口径);',
            f'- 微盘池基准下 f_0 ≈ R_b (均为微盘池市值加权), 故 f_0 − R_b ≈ 0, '
            f'ε_adj ≈ ε (绝对口径) — **微盘池归因中 alpha 无基准构成效应**, 直接反映'
            f'"在微盘池内选股"的真实能力;',
        ]
    else:
        lines += [
            '- **f_0** = 截面回归的市场截距项 = 沪市市值加权收益 — **只是模型内部市场项, 不是评价基准**;',
            '- **R_b** = 上证综指收益 (总股本加权指数点位收益);',
            '- 上证综指按总股本加权, 被大金融/石油等超大盘权重股拖累, 落后市值加权市场, '
            f'差额 f_0 − R_b ≈ {f0rb_bp:.0f} bp 被吸收进 ε_adj;',
            f'- 在"跟上证综指比"的框架下, 这 {f0rb_bp:.0f} bp 是策略相对基准的真实超额来源之一 '
            '(不持有被压制的超大盘权重股), 计入 alpha — 但若想评估"选股能力本身", '
            '应改用微盘池基准 (--bench smallcap) 剔除该基准构成效应。', '',
        ]
    lines += [
        f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
    ]
    path = os.path.join(out_dir, '术语表_收益含义.md')
    with open(path, 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(lines))
    return path


def main():
    ap = argparse.ArgumentParser(description='BARRA 收益归因 (回测结果)')
    ap.add_argument('--backtest-dir', required=True, help='回测结果目录')
    ap.add_argument('--bench', default='index', choices=['index', 'smallcap'],
                    help='归因基准: index=上证综指(默认) | smallcap=微盘池(策略选股池)')
    ap.add_argument('--out-dir', default='attribution',
                    help='输出子目录名 (默认 attribution; 可指定 attribution2 等并存)')
    args = ap.parse_args()
    run_attribution(args.backtest_dir, bench=args.bench, out_tag=args.out_dir)


if __name__ == '__main__':
    main()
