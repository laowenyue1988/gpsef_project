# -*- coding: utf-8 -*-
"""
Barra 因子暴露离线构建器 — 机构级实现

功能：
  1. 通过 baostock 获取 CSI 800 历史成分股（HS300 ∪ ZZ500）
  2. 从 data/performance_reports 获取已发布最新财报（防未来数据泄露）
  3. 前复权到 constraint_date，逐股票计算五因子暴露：
     Size / Beta / Value(BP) / Momentum / Volatility
  4. 保存为本地快照：data/barra_exposures/barra_expo_{YYYYMMDD}.parquet

设计原则：
  - 防泄露：财报按 最新公告日期 ≤ constraint_date 过滤
  - 前复权：基准日 = constraint_date，动态计算
  - 快照复用：同一约束日只计算一次，后续直接加载

依赖：
  - baostock (指数成分股查询)
  - data/raw_daily/{code}.parquet (不复权日线)
  - data/performance_reports/YYYYMMDD.parquet (季度财报)
  - data/wyckoff20_v5_snapshots/factors_YYYYMMDD.parquet (市值)

Author: BULLION GPSEF Framework
Date: 2026-07-10
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# ─── 路径配置 ───────────────────────────────────────────────
_REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'performance_reports')
_SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'wyckoff20_v5_snapshots')
_BARRA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'barra_exposures')
_RAW_DAILY_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw_daily')
_ADJ_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'back_adjust_factors')

# ─── 风格因子名称 ─────────────────────────────────────────────
# [2026-09-08 BARRA 修复] 5→6 因子, 新增 liquidity(流动性, 微盘策略最关键风险因子)。
# 业界对齐: CNE5 风格因子含 Liquidity(STOM/STOQ/STOA), 此处用 ln(252日均换手率) 近似 STOA。
STYLE_FACTORS = ['size', 'beta', 'value', 'momentum', 'volatility', 'liquidity']

# ─── 基线指数注册表 (config barra.baseline_index_code 可切换) ──
# csi800 = 沪深300∪中证500 (默认, 兼容旧行为); hs300 = 沪深300; zz500 = 中证500;
# sh000001 = 上证指数成分; sh932000 = 中证2000(微盘, 与策略池更匹配)
BASELINE_INDEX = {
    'csi800': ['hs300', 'zz500'],
    'hs300': ['hs300'],
    'zz500': ['zz500'],
    'sh000001': ['sh000001'],
    'sh932000': ['sh932000'],
}

# ─── 缓存 ──────────────────────────────────────────────────
_industry_map_cache: Dict[str, str] = {}  # {code: industry}
_financial_cache: Dict[str, pd.DataFrame] = {}  # {period: DataFrame}


def _load_report(report_file: str) -> Optional[pd.DataFrame]:
    """加载单个财报文件。"""
    if report_file in _financial_cache:
        return _financial_cache[report_file]
    path = os.path.join(_REPORTS_DIR, report_file)
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    # [P1 Fix] 公告日期列名: 优先中文，回退英文
    pub_col = None
    for c in df.columns:
        if not c.isascii() and '公告' in c:
            pub_col = c
            break
    if pub_col is None:
        for candidate in ['pub_date', 'publish_date', 'ann_date', 'announce_date']:
            if candidate in df.columns:
                pub_col = candidate
                break
    if pub_col is None:
        logger.warning(f"Cannot find publish date column in {report_file}")
        return None
    df['_pub_date'] = pd.to_datetime(df[pub_col], errors='coerce').dt.date
    _financial_cache[report_file] = df
    return df


def _get_latest_financials(constraint_date: datetime) -> pd.DataFrame:
    """
    获取 constraint_date 前已发布的最新财报（防未来数据泄露）。

    遍历 data/performance_reports/ 下所有文件，
    对每只股票，取 _pub_date ≤ constraint_date 且报告期最新的一条。
    """
    if not os.path.exists(_REPORTS_DIR):
        logger.warning(f"Reports dir not found: {_REPORTS_DIR}")
        return pd.DataFrame()

    all_reports = []
    for fname in sorted(os.listdir(_REPORTS_DIR)):
        if not fname.endswith('.parquet'):
            continue
        df = _load_report(fname)
        if df is None or len(df) == 0:
            continue

        # 防泄露：只保留 constraint_date 前已发布的记录
        mask = df['_pub_date'] <= constraint_date.date()
        valid = df[mask].copy()
        if len(valid) == 0:
            continue

        # 标记报告期
        report_period = fname.replace('.parquet', '')
        valid['_report_period'] = report_period
        all_reports.append(valid)

    if not all_reports:
        return pd.DataFrame()

    combined = pd.concat(all_reports, ignore_index=True)
    # 每股票取报告期最新
    combined = combined.sort_values('_report_period')
    latest = combined.groupby('code').last().reset_index()

    # 标准化 code 为 6 位数字字符串
    latest['code'] = latest['code'].astype(str).str.zfill(6)

    logger.info(f"[Barra Builder] constraint={constraint_date.date()}: "
                f"{len(latest)} stocks with valid financials")

    return latest


def _get_csi800_constituents(constraint_date: datetime, bs=None) -> pd.DataFrame:
    """兼容旧接口: 等价于 _get_constituents(constraint_date, 'csi800', bs)。"""
    return _get_constituents(constraint_date, 'csi800', bs)


# [2026-09-08] 本地成分缓存 (日频 PIT, index_stocks_cache.parquet)
_INDEX_STOCKS_CACHE = None


def _load_index_stocks_cache() -> pd.DataFrame:
    """加载本地指数成分缓存 (PIT: cache_key 日期 = 该交易日成分)。"""
    global _INDEX_STOCKS_CACHE
    if _INDEX_STOCKS_CACHE is None:
        p = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'index_stocks_cache.parquet')
        if os.path.exists(p):
            df = pd.read_parquet(p)
            df['prefix'] = df['cache_key'].str.split('_').str[0]
            df['date'] = df['cache_key'].str.split('_').str[1]
            _INDEX_STOCKS_CACHE = df
        else:
            _INDEX_STOCKS_CACHE = pd.DataFrame()
    return _INDEX_STOCKS_CACHE


def _get_constituents(constraint_date: datetime, baseline: str = 'csi800',
                      bs=None) -> pd.DataFrame:
    """[2026-09-08] 本地 PIT 成分获取 (离线, 无 baostock 依赖)。

    从 data/index_stocks_cache.parquet 取"最近 ≤ constraint_date"的成分股:
      - csi800 = hs300 ∪ zz500 (默认)
      - hs300  = hs300
      - zz500  = zz500
    PIT 保证: cache_key 日期为成分发布交易日, 取 ≤ constraint_date 的最新一条,
    绝不使用未来成分调整。
    (sh000001/sh932000 无本地成分缓存, 由 build_barra_snapshot 用快照市值构建。)
    """
    df = _load_index_stocks_cache()
    if df.empty:
        return pd.DataFrame(columns=['code', 'code_name'])
    date_str = constraint_date.strftime('%Y%m%d')
    targets = BASELINE_INDEX.get(baseline, BASELINE_INDEX['csi800'])
    codes = {}
    for t in targets:
        sub = df[df['prefix'] == t]
        sub = sub[sub['date'] <= date_str]
        if sub.empty:
            continue
        latest = sub.sort_values('date').iloc[-1]
        for s in latest['stock_list']:
            c = s.replace('sh.', '').replace('sz.', '')
            codes[c] = ''
    out = pd.DataFrame({'code': list(codes.keys()), 'code_name': ''})
    out['code'] = out['code'].astype(str).str.zfill(6)
    logger.info(f"[Barra Builder] {baseline} constituents (本地PIT) at {date_str}: {len(out)} stocks")
    return out


def _load_daily_data(code: str) -> Optional[pd.DataFrame]:
    """加载本地不复权日线数据。支持 sh_XXXXXX.parquet 和 XXXXXX.parquet 两种命名。"""
    # 尝试多种文件名格式
    candidates = [
        f'{code}.parquet',
        f'sh_{code}.parquet',
        f'sz_{code}.parquet',
    ]
    path = None
    for c in candidates:
        p = os.path.join(_RAW_DAILY_DIR, c)
        if os.path.exists(p):
            path = p
            break
    
    if path is None:
        return None
    
    df = pd.read_parquet(path)
    if 'close' not in df.columns:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        elif 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.set_index('trade_date')
    df = df.sort_index()
    return df


def _load_adjust_factors_local(code: str) -> Optional[pd.DataFrame]:
    """从本地缓存加载复权因子。data/back_adjust_factors/{prefix}_{code}_back_adjust.parquet"""
    candidates = [
        f'sh_{code}_back_adjust.parquet',
        f'sz_{code}_back_adjust.parquet',
    ]
    for c in candidates:
        path = os.path.join(_ADJ_DIR, c)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if 'dividOperateDate' in df.columns and 'adjustFactor' in df.columns:
                df['date'] = pd.to_datetime(df['dividOperateDate'])
                df['adjustFactor'] = pd.to_numeric(df['adjustFactor'], errors='coerce')
                df = df.set_index('date').sort_index()[['adjustFactor']]
                return df
    return None


def _compute_factor_exposures(code: str, constraint_date: datetime,
                               fin_data: pd.DataFrame,
                               daily_data: Optional[pd.DataFrame],
                               market_rets: Optional[pd.Series] = None,
                               window: int = 252
                               ) -> Dict[str, Optional[float]]:
    """
    计算单股票五因子暴露（本地复权因子精确前复权）。

    日线数据从 data/raw_daily/ 读取，复权因子从 data/adjust_factors/ 读取。
    前复权基准日 = constraint_date。
    """
    result = {f: np.nan for f in STYLE_FACTORS}
    result['daily_rets_252'] = None

    if daily_data is None or len(daily_data) < 60:
        return result

    # 加载本地复权因子
    adj_df = _load_adjust_factors_local(code)

    close = daily_data['close']
    close = close[close.index <= constraint_date].tail(window + 1)
    if len(close) < 60:
        return result

    # 前复权：用 asof 方法将稀疏的复权因子对齐到每日日期
    if adj_df is not None and len(adj_df) > 0:
        # ref_factor = constraint_date 前最后一条 adjustFactor
        ref_series = adj_df['adjustFactor']
        ref_series = ref_series[ref_series.index <= constraint_date]
        if len(ref_series) > 0:
            ref_factor = float(ref_series.iloc[-1])
            if ref_factor > 0:
                # 用 merge_asof 将每日日期对齐到最近的前一个除权日
                daily_dates = pd.DataFrame({'date': close.index})
                adj_dates = pd.DataFrame({
                    'date': adj_df.index,
                    'adjustFactor': adj_df['adjustFactor'].values
                })
                merged = pd.merge_asof(
                    daily_dates.sort_values('date'),
                    adj_dates.sort_values('date'),
                    on='date', direction='backward'
                )
                merged = merged.set_index('date')
                close_adj = close * ref_factor / merged['adjustFactor']
            else:
                close_adj = close
        else:
            close_adj = close
    else:
        close_adj = close

    close_adj = close_adj.tail(window + 1)
    if len(close_adj) < 60:
        return result

    # 日收益率
    rets = close_adj.pct_change().dropna().tail(window)
    n_ret = len(rets)
    if n_ret < 50:
        return result

    # 2. Volatility
    result['volatility'] = float(rets.std()) * np.sqrt(252)

    # 3. Momentum: 12-1 月
    if len(close_adj) >= 252:
        p_1m = float(close_adj.iloc[-22]) if len(close_adj) >= 22 else float(close_adj.iloc[0])
        p_12m = float(close_adj.iloc[-252])
        result['momentum'] = p_1m / p_12m - 1.0
    else:
        result['momentum'] = 0.0

    # [2026-09-08] 6. Liquidity: ln(252 日均换手率) ≈ CNE5 STOA
    #   raw_daily 含 turn(换手率) 字段; 取窗口内均值, 0/NaN 剔除, min 保护
    if 'turn' in daily_data.columns:
        turn_win = daily_data['turn']
        turn_win = turn_win[turn_win.index <= constraint_date].tail(window)
        turn_win = pd.to_numeric(turn_win, errors='coerce').dropna()
        turn_win = turn_win[turn_win > 0]
        if len(turn_win) >= 20:
            result['liquidity'] = float(np.log(np.mean(turn_win)))


    # 4. Beta
    if market_rets is not None and len(market_rets) >= 50:
        cl = min(n_ret, len(market_rets))
        ri = rets.values[-cl:]
        rm = market_rets.values[-cl:]
        mask = ~np.isnan(ri) & ~np.isnan(rm)
        if mask.sum() >= 30:
            cov_im = np.cov(ri[mask], rm[mask])[0, 1]
            var_m = np.var(rm[mask])
            result['beta'] = cov_im / var_m if var_m > 1e-10 else 1.0
        else:
            result['beta'] = 1.0
    else:
        result['beta'] = 1.0

    # 5. Value(BP): bvps / price_adj
    stock_fin = fin_data[fin_data['code'] == code]
    if len(stock_fin) > 0:
        bvps = stock_fin['bvps'].iloc[0]
        price = float(close_adj.iloc[-1])
        if pd.notna(bvps) and bvps > 0 and price > 0:
            result['value'] = bvps / price

    result['daily_rets_252'] = rets.values.tolist()
    return result


def _orthogonalize_style_factors(df: pd.DataFrame) -> pd.DataFrame:
    """[2026-09-08 BARRA 修复 P0-1] 风格因子正交化 (BARRA/CNE5 标准)。

    对每个风格因子: 行业去均值 → 去 Size 线性回归(残差) → Winsorize(1%/99%) → z-score。
      - size 自身: ln(mv) 行业去均值后 z-score (不对 size 回归自身)
      - value: ln(BP) 同上
      - beta/momentum/volatility/liquidity: 去行业 → 去 size 残差 → z-score

    目的: 消除因子间相关性(Value/Momentum/Liquidity 天然与 Size 相关),
    使因子协方差矩阵对角化、约束方向正交、风险可归因。残差后的 volatility
    即为近似的 residual volatility (剔除 size/行业系统性波动后的特质波动)。
    """
    ind = df['industry'].fillna('other')

    def _ind_demean(x: pd.Series) -> pd.Series:
        return x - x.groupby(ind).transform('mean')

    def _winsor_z(x: pd.Series) -> pd.Series:
        x = x.dropna()
        if len(x) < 50:
            return pd.Series(np.nan, index=x.index)
        lo, hi = x.quantile(0.01), x.quantile(0.99)
        xc = x.clip(lo, hi)
        sd = xc.std()
        return (xc - xc.mean()) / sd if sd > 1e-12 else pd.Series(0.0, index=xc.index)

    valid_mv = df['size_raw'].notna() & (df['size_raw'] > 0)
    log_mv = pd.Series(np.nan, index=df.index)
    log_mv[valid_mv] = np.log(df.loc[valid_mv, 'size_raw'].values.astype(float))

    valid_bp = df['value_raw'].notna() & (df['value_raw'] > 0)
    log_bp = pd.Series(np.nan, index=df.index)
    log_bp[valid_bp] = np.log(df.loc[valid_bp, 'value_raw'].values.astype(float))

    def _size_resid(x: pd.Series) -> pd.Series:
        """对 log_mv 做截面线性回归, 返回去均值残差。"""
        valid = x.notna() & log_mv.notna()
        if valid.sum() < 50:
            return pd.Series(np.nan, index=x.index)
        xv = x[valid].values.astype(float)
        mv = log_mv[valid].values.astype(float)
        mv_c = mv - mv.mean()
        denom = float(np.dot(mv_c, mv_c))
        if denom < 1e-12:
            return pd.Series(np.nan, index=x.index)
        beta = float(np.dot(xv - xv.mean(), mv_c) / denom)
        resid = xv - beta * mv_c
        out = pd.Series(np.nan, index=x.index)
        out[valid] = resid
        return out

    # size: ln(mv) 行业去均值 → zscore
    df['size_exp'] = _winsor_z(_ind_demean(log_mv)).reindex(df.index)
    # value: ln(bp) 去行业 → 去 size → zscore
    df['value_exp'] = _winsor_z(_size_resid(_ind_demean(log_bp))).reindex(df.index)
    # 其余: 去行业 → 去 size → zscore
    for f in ['beta', 'momentum', 'volatility', 'liquidity']:
        df[f'{f}_exp'] = _winsor_z(_size_resid(_ind_demean(df[f'{f}_raw']))).reindex(df.index)

    return df


def _constituents_from_snapshot_mv(snap: pd.DataFrame, prefix: str = None,
                                   mv_rank_range: tuple = None) -> pd.DataFrame:
    """[2026-09-08] 从本地快照按市值构建基线成分 (用于无 baostock 接口的指数)。

    - prefix: 代码前缀过滤 ('sh'=上证)
    - mv_rank_range: (lo, hi) 市值排名区间 (1-based), 如中证2000=(1801, 3800)
    """
    if snap is None or len(snap) == 0 or 'total_mv' not in snap.columns:
        return pd.DataFrame(columns=['code'])
    s = snap.dropna(subset=['total_mv']).copy()
    s = s[s['total_mv'] > 0]
    if 'code' not in s.columns:
        return pd.DataFrame(columns=['code'])
    if prefix is not None:
        s = s[s['code'].astype(str).str.startswith(prefix)]
    s = s.sort_values('total_mv', ascending=False)
    if mv_rank_range is not None:
        lo, hi = mv_rank_range
        s = s.iloc[int(lo) - 1:int(hi)]
    out = pd.DataFrame({'code': s['code'].astype(str).str.zfill(6)})
    out['code_name'] = ''
    return out


# [2026-09-08] 本地市场收益率缓存 (index_daily/sh000001 上证综指)
_MARKET_RETS_CACHE = {}


def _get_market_returns(constraint_date: datetime, window: int = 252) -> Optional[pd.Series]:
    """[2026-09-08] 本地市场收益率 (上证综指 sh000001, 离线 PIT)。

    作为 Beta 的市场代理, 用 data/index_daily/sh000001.parquet 的 close 日收益率,
    截断 ≤ constraint_date, PIT 合规 (不用未来数据)。
    """
    cache_key = 'sh000001'
    if cache_key in _MARKET_RETS_CACHE:
        s = _MARKET_RETS_CACHE[cache_key]
    else:
        p = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'index_daily', 'sh000001.parquet')
        if not os.path.exists(p):
            return None
        df = pd.read_parquet(p)
        dc = [c for c in df.columns if 'date' in c.lower()][0]
        cc = [c for c in df.columns if c.lower() == 'close'][0]
        df[dc] = pd.to_datetime(df[dc])
        df = df.sort_values(dc).set_index(dc)
        s = df[cc].pct_change().dropna()
        _MARKET_RETS_CACHE[cache_key] = s
    rets = s[s.index <= constraint_date].tail(window)
    return rets if len(rets) >= 50 else None


def build_barra_snapshot(constraint_date: datetime, force_rebuild: bool = False,
                         baseline: str = 'csi800') -> Optional[pd.DataFrame]:
    """
    构建单个 constraint_date 的 Barra 暴露快照。

    Args:
        baseline: 基线指数 (csi800/hs300/zz500/sh000001/sh932000), 见 BASELINE_INDEX
                  决定成分股范围与基准权重; 默认 csi800 兼容旧行为。

    Returns:
        DataFrame with columns:
          code, weight_bm, industry, report_period,
          size_exp, beta_exp, value_exp, momentum_exp, volatility_exp, liquidity_exp,
          roe (Quality备用), daily_rets_252 (日收益率JSON)

        或 None 如果数据不足。
    """
    os.makedirs(_BARRA_DIR, exist_ok=True)

    date_str = constraint_date.strftime('%Y%m%d')
    # 快照命名: csi800 兼容旧命名(barra_expo_{date}.parquet), 其它带 baseline 后缀
    suffix = '' if baseline == 'csi800' else f'_{baseline}'
    out_path = os.path.join(_BARRA_DIR, f'barra_expo_{date_str}{suffix}.parquet')

    if not force_rebuild and os.path.exists(out_path):
        logger.info(f"[Barra Snapshot] {date_str} ({baseline}) already exists, loading")
        return pd.read_parquet(out_path)

    logger.info(f"[Barra Snapshot] Building for {constraint_date.date()} (baseline={baseline}) ...")
    t0 = time.time()

    # 1. 基线成分 (本地 PIT, 离线)
    snap_path = os.path.join(_SNAPSHOT_DIR, f'factors_{date_str}.parquet')
    if os.path.exists(snap_path):
        snap = pd.read_parquet(snap_path)
    else:
        snap = pd.DataFrame()

    if baseline == 'sh932000':
        # 中证2000: 本地快照市值排名区间(1801~3800)近似构建
        constituents = _constituents_from_snapshot_mv(snap, mv_rank_range=(1801, 3800))
    elif baseline == 'sh000001':
        # 上证指数: 快照 sh 前缀 (近似全上交所)
        constituents = _constituents_from_snapshot_mv(snap, prefix='sh')
    else:
        constituents = _get_constituents(constraint_date, baseline, bs=None)

    if len(constituents) < 100:
        logger.warning("Too few constituents, aborting")
        return None

    # 2. 已发布财报
    fin_data = _get_latest_financials(constraint_date)
    if len(fin_data) == 0:
        logger.warning("No financial data available, aborting")
        return None

    # 3. 市值数据已在第 1 步从快照读取 (snap), 此处无需重复

    # 4. [2026-09-08] 市场收益率 (Beta 用) —— 本地 index_daily (离线 PIT)
    market_rets = _get_market_returns(constraint_date)

    # 5. 逐股票计算因子
    records = []
    n_total = len(constituents)
    last_log_pct = 0

    for idx, (_, row) in enumerate(constituents.iterrows()):
        code = row['code']
        daily_data = _load_daily_data(code)

        # 五因子（不复权，本地数据直接算）
        expo = _compute_factor_exposures(
            code, constraint_date, fin_data, daily_data, market_rets
        )

        # 行业
        stock_fin = fin_data[fin_data['code'] == code]
        industry = stock_fin['industry'].iloc[0] if len(stock_fin) > 0 and pd.notna(stock_fin['industry'].iloc[0]) else 'other'
        report_period = stock_fin['_report_period'].iloc[0] if len(stock_fin) > 0 else ''

        # 市值
        total_mv = np.nan
        if len(snap) > 0 and code in snap['code'].values:
            sv = snap[snap['code'] == code]
            if len(sv) > 0 and 'total_mv' in sv.columns:
                total_mv = float(sv['total_mv'].iloc[0])

        records.append({
            'code': code,
            'weight_raw': total_mv,
            'industry': industry,
            'report_period': report_period,
            'size_raw': total_mv,
            'beta_raw': expo['beta'],
            'value_raw': expo['value'],
            'momentum_raw': expo['momentum'],
            'volatility_raw': expo['volatility'],
            'liquidity_raw': expo['liquidity'],
            'daily_rets_252': json.dumps(expo['daily_rets_252']) if expo['daily_rets_252'] else None,
        })

    df = pd.DataFrame(records)

    # 6. [2026-09-08 P0-1] 因子正交化 (Size+行业回归取残差) + Winsorize + z-score
    for f in STYLE_FACTORS:
        df[f'{f}_exp'] = np.nan
    df = _orthogonalize_style_factors(df)

    # 权重归一化
    df['weight_bm'] = np.nan
    valid_w = df['weight_raw'].notna() & (df['weight_raw'] > 0)
    if valid_w.sum() > 0:
        total = df.loc[valid_w, 'weight_raw'].sum()
        df.loc[valid_w, 'weight_bm'] = (df.loc[valid_w, 'weight_raw'] / total).values

    # 清理列
    exp_cols = ['code', 'weight_bm', 'industry', 'report_period']
    exp_cols += [f'{f}_exp' for f in STYLE_FACTORS]
    exp_cols += ['daily_rets_252']
    out_df = df[exp_cols].copy()

    # 存盘
    out_df.to_parquet(out_path, index=False)
    elapsed = time.time() - t0
    logger.info(f"[Barra Snapshot] Saved {len(out_df)} stocks → {out_path} ({elapsed:.1f}s)")

    return out_df


def get_or_build_snapshot(constraint_date: datetime, baseline: str = 'csi800'
                          ) -> Optional[pd.DataFrame]:
    """[2026-09-08 P1-3] 快照接口: 优先加载"最近 ≤ constraint_date 的回溯快照"。

    修复前为精确日期匹配, 调仓日(20 天一次)与周频快照日期错开 → 每次都联网构建
    (慢 + 口径不一致 + 潜在用当日收盘数据的前视)。改为 merge_asof 语义:
    找 constraint_date 之前最近的快照; 若无则构建。
    """
    suffix = '' if baseline == 'csi800' else f'_{baseline}'
    # 列举所有匹配 baseline 的快照日期
    if not os.path.isdir(_BARRA_DIR):
        return build_barra_snapshot(constraint_date, baseline=baseline)
    prefix = 'barra_expo_'
    dates = []
    for fn in os.listdir(_BARRA_DIR):
        if not fn.startswith(prefix) or not fn.endswith('.parquet'):
            continue
        stem = fn[len(prefix):-len('.parquet')]
        # csi800 文件无后缀; 其它带 _{baseline}
        if suffix:
            if not stem.endswith(suffix):
                continue
            stem = stem[:-len(suffix)]
        try:
            dates.append(datetime.strptime(stem, '%Y%m%d'))
        except ValueError:
            continue
    # [PIT] 最近且严格 < constraint_date 的快照 (确保不用调仓日当天收盘数据 = 无未来泄露)
    #   快照文件名日期 = 该快照的数据截止日(含该日收盘); 调仓决策在 constraint_date 开盘前,
    #   故只能用 < constraint_date 的快照(等价 ≤ 前一交易日收盘)
    valid = [d for d in dates if d < constraint_date]
    if valid:
        nearest = max(valid)
        out_path = os.path.join(_BARRA_DIR, f'barra_expo_{nearest.strftime("%Y%m%d")}{suffix}.parquet')
        logger.debug(f"[Barra Snapshot] 回溯加载 {nearest.strftime('%Y%m%d')} < {constraint_date.strftime('%Y%m%d')} (PIT)")
        return pd.read_parquet(out_path)
    return build_barra_snapshot(constraint_date, baseline=baseline)


def compute_benchmark_exposures(expo: pd.DataFrame) -> np.ndarray:
    """
    计算基线指数市值加权的风格因子暴露。
    Returns: shape (len(STYLE_FACTORS),)
    """
    w = expo['weight_bm'].fillna(0).values
    w = w / w.sum() if w.sum() > 0 else np.ones(len(expo)) / len(expo)

    benchmark = np.zeros(len(STYLE_FACTORS))
    for k, f in enumerate(STYLE_FACTORS):
        col = f'{f}_exp'
        if col in expo.columns:
            v = expo[col].fillna(0).values
            benchmark[k] = np.dot(w, v)
    return benchmark


def compute_on_the_fly_exposures(codes: List[str], constraint_date: datetime,
                                  prebuilt_expo: pd.DataFrame) -> pd.DataFrame:
    """
    为 prebuilt_expo 中缺失的股票即时计算 Barra 暴露。

    使用与 build_barra_snapshot 完全相同的计算逻辑：
    - 同一天的数据（日线、复权因子、财报）
    - 同一套 _compute_factor_exposures
    - 截面 Z-score 以自身均值为基准（与原快照逻辑一致）

    Args:
        codes: 需要计算的股票代码列表
        constraint_date: 约束日（T-1）
        prebuilt_expo: 预构建的 CSI 800 暴露快照

    Returns:
        DataFrame，列与快照一致
    """
    fin_data = _get_latest_financials(constraint_date)
    if len(fin_data) == 0:
        return pd.DataFrame()

    # [2026-09-08] 市场收益率 (Beta 用) —— 本地 index_daily (离线 PIT)
    market_rets = _get_market_returns(constraint_date)

    records = []
    for code in codes:
        daily_data = _load_daily_data(code)
        expo = _compute_factor_exposures(code, constraint_date, fin_data,
                                          daily_data, market_rets)

        stock_fin = fin_data[fin_data['code'] == code]
        industry = 'other'
        report_period = ''
        if len(stock_fin) > 0:
            industry = stock_fin['industry'].iloc[0] if pd.notna(stock_fin['industry'].iloc[0]) else 'other'
            report_period = stock_fin['_report_period'].iloc[0]

        records.append({
            'code': code,
            'weight_bm': np.nan,
            'industry': industry,
            'report_period': report_period,
            'beta_raw': expo.get('beta', np.nan),
            'value_raw': expo.get('value', np.nan),
            'momentum_raw': expo.get('momentum', np.nan),
            'volatility_raw': expo.get('volatility', np.nan),
            'liquidity_raw': expo.get('liquidity', np.nan),
            'daily_rets_252': json.dumps(expo.get('daily_rets_252'))
            if expo.get('daily_rets_252') else None,
        })

    df = pd.DataFrame(records)
    if len(df) == 0:
        return df

    # 初始化各列
    if 'size_raw' not in df.columns:
        df['size_raw'] = np.nan
    for f in STYLE_FACTORS:
        df[f'{f}_exp'] = np.nan

    # [P1 Fix] Z-score using prebuilt snapshot statistics (not local small sample)
    prebuilt_stats = {}
    for factor_name in ['beta_raw', 'momentum_raw', 'volatility_raw', 'liquidity_raw']:
        exp_col = factor_name.replace('_raw', '_exp')
        if exp_col in prebuilt_expo.columns:
            vals_pb = prebuilt_expo[exp_col].dropna()
            if len(vals_pb) >= 50:
                prebuilt_stats[factor_name] = {
                    'mean': float(vals_pb.mean()), 'std': float(vals_pb.std()),
                    'lo': float(vals_pb.quantile(0.01)), 'hi': float(vals_pb.quantile(0.99))
                }

    for factor_name in ['beta_raw', 'momentum_raw', 'volatility_raw', 'liquidity_raw']:
        valid = df[factor_name].notna()
        if valid.sum() < 1:
            continue
        vals = df.loc[valid, factor_name]
        stats = prebuilt_stats.get(factor_name)
        if stats and stats['std'] > 0:
            lo, hi = stats['lo'], stats['hi']
            mean, std = stats['mean'], stats['std']
        else:
            lo, hi = vals.quantile(0.01), vals.quantile(0.99)
            mean, std = vals.mean(), vals.std() if len(vals) > 1 else (0, 1)
        vals_c = vals.clip(lo, hi)
        z = (vals_c - mean) / std
        df.loc[valid, factor_name.replace('_raw', '_exp')] = z.values

    # Size: ln(mv) → Z-score（从 raw_daily 获取市值信息）
    # 使用 close * 隐含股本的方式：从快照或日线中获取
    for code in codes:
        row = df[df['code'] == code]
        if len(row) == 0:
            continue
        daily = _load_daily_data(code)
        if daily is not None and 'amount' in daily.columns and 'turn' in daily.columns:
            last = daily[daily.index <= constraint_date].iloc[-1]
            amt = float(last['amount']) if 'amount' in last.index else np.nan
            trn = float(last['turn']) if 'turn' in last.index else np.nan
            if pd.notna(amt) and pd.notna(trn) and trn > 0:
                df.loc[df['code'] == code, 'size_raw'] = amt / (trn / 100.0)

    valid_sz = df['size_raw'].notna() & (df['size_raw'] > 0)
    if valid_sz.sum() >= 2:
        log_mv = np.log(df.loc[valid_sz, 'size_raw'])
        df.loc[valid_sz, 'size_exp'] = ((log_mv - log_mv.mean()) / log_mv.std()).values

    # Value: BP → Z-score
    valid_bp = df['value_raw'].notna() & (df['value_raw'] > 0)
    if valid_bp.sum() >= 2:
        bp = np.log(df.loc[valid_bp, 'value_raw'])
        df.loc[valid_bp, 'value_exp'] = ((bp - bp.mean()) / bp.std()).values

    exp_cols = ['code', 'weight_bm', 'industry', 'report_period']
    exp_cols += [f'{f}_exp' for f in STYLE_FACTORS]
    exp_cols += ['daily_rets_252']
    return df[exp_cols].copy()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        dt = datetime.strptime(sys.argv[1], '%Y%m%d')
    else:
        dt = datetime(2025, 6, 30)
    df = build_barra_snapshot(dt, force_rebuild=True)
    if df is not None:
        print(f"Built snapshot: {len(df)} stocks")
        print(f"Style exposures summary:")
        for f in STYLE_FACTORS:
            col = f'{f}_exp'
            if col in df.columns:
                vals = df[col].dropna()
                print(f"  {f}: mean={vals.mean():.4f}, std={vals.std():.4f}, n={len(vals)}")
        bench = compute_benchmark_exposures(df)
        print(f"Benchmark exposures: {dict(zip(STYLE_FACTORS, bench))}")
