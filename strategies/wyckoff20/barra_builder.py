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
_BARRA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'barra_exposures')
_RAW_DAILY_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw_daily')
_ADJ_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'back_adjust_factors')

# ═══ [FIX 2026-09-15 · D4] 统一复权实现的辅助件 ═══════════════════════════════
#   目的：让 barra_builder 复用 LocalDataProvider 的正典前复权（含 smooth_bf），
#   消除「第二套复权实现」——它是 D1（公式方向反）长期未被发现的根因。
_PROVIDER_SINGLETON = None


def _get_provider():
    """惰性构造 LocalDataProvider 单例（失败返回 None，调用方回退本地实现）。"""
    global _PROVIDER_SINGLETON
    if _PROVIDER_SINGLETON is None:
        try:
            from gpsef.data.local_provider import LocalDataProvider
            base = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
            _PROVIDER_SINGLETON = LocalDataProvider(os.path.abspath(base))
        except Exception as e:  # pragma: no cover
            logger.warning('[D4] LocalDataProvider 构造失败，将回退本地复权实现: %s', e)
            _PROVIDER_SINGLETON = False
    return _PROVIDER_SINGLETON or None


def _resolve_provider_code(code: str) -> str:
    """把 6 位代码规范化为 provider 期望的 `sh.XXXXXX` / `sz.XXXXXX`（按 raw_daily 实际文件名判定）。"""
    c = str(code).strip()
    if '.' in c:
        return c
    c6 = c.zfill(6)
    for pref in ('sh', 'sz'):
        if os.path.exists(os.path.join(_RAW_DAILY_DIR, f'{pref}_{c6}.parquet')):
            return f'{pref}.{c6}'
    if c6.startswith(('6', '9')):
        return f'sh.{c6}'
    return f'sz.{c6}'
_MV_CACHE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'total_mv_cache.parquet')

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


def _norm_code6(code: str) -> str:
    """[Fix-20260908] code 归一为 6 位数字: 'sz.300588'/'sh.600000' → '300588'/'600000'。

    快照内 code 统一为 6 位; strategy 层传入选股 codes 为 ts_code 格式(带交易所前缀)。
    此前两格式混用导致持仓股暴露匹配失败、on-the-fly 取数失败(B_port 全 0)。
    """
    s = str(code).strip()
    if '.' in s:
        s = s.split('.')[-1]
    return s.zfill(6)


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
    # [FIX 2026-09-15 · D4] 统一复权实现：**优先复用 LocalDataProvider 正典路径**
    #   （含 smooth_bf 因子平滑 + 已验证的 `raw × bf / bf_T` 公式），
    #   以消除「barra_builder 自建第二套复权实现」这一 D1 长期潜伏的根因，
    #   并同时消除「本地用原始 bf / provider 用 smooth_bf」的分歧。
    #   本地路径（已修 D1）保留为**回退**，仅在 provider 不可用时使用。
    close_adj = None
    try:
        from gpsef.data.local_provider import LocalDataProvider
        _p = _get_provider()
        _w = close[close.index <= constraint_date].tail(window + 1)
        if _p is not None and len(_w) >= 60:
            _full = _resolve_provider_code(code)
            _dfp = _p.get_daily_data(_full, _w.index[0], constraint_date,
                                     reference_date=constraint_date)
            if _dfp is not None and not _dfp.empty and 'close' in _dfp.columns:
                _dc = 'trade_date' if 'trade_date' in _dfp.columns else 'date'
                _s = _dfp.set_index(pd.to_datetime(_dfp[_dc]))['close'].astype(float)
                _s = _s[~_s.index.duplicated(keep='last')].sort_index()
                _al = _s.reindex(_w.index).ffill().bfill()
                if _al.notna().sum() >= 60:
                    close_adj = _al
    except Exception as _e:
        logger.debug('[D4] provider 正典复权不可用(%s)，回退本地实现: %s', code, _e)
    if close_adj is None and adj_df is not None and len(adj_df) > 0:
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
                # [FIX 2026-09-15 · D1] 原实现为 `close * ref_factor / adjustFactor`（方向反）:
                #   adjustFactor 是**后复权因子**（除权日递增；实测 398 只样本 7277 递增步 vs 101 递减步,
                #   600000 首值 1.0 → 末值 13.367）⇒ 后复权价 = raw × bf；
                #   以前复权基准日 T 归一 ⇒ 正确式 = raw × bf / bf_T = close * adjustFactor / ref_factor。
                #   实测（600000 @2018-07-13 除权日, bf 9.385732→9.484842）:
                #     raw 单日 −0.836% ；修复式 +0.211%（除权被抵消, 平滑）；原式 −1.872%（虚假放大）。
                #   独立验证脚本：`_verify_d1_adj_direction.py`。
                close_adj = close * merged['adjustFactor'] / ref_factor
            else:
                close_adj = close
        else:
            close_adj = close
    elif close_adj is None:
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
        # [Fix-20260908 审核] 数据不足置 NaN (原 0.0 会污染截面分布, 造假中值)
        result['momentum'] = np.nan

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
    # [2026-09-08 市场代理] 市场收益代理带 DatetimeIndex 时做日期对齐;
    # 无市场收益时置 NaN (两阶段流程 Phase 1 / 数据缺失), 不再置 1.0 污染截面。
    if market_rets is not None:
        if isinstance(market_rets, pd.Series) and isinstance(market_rets.index, pd.DatetimeIndex):
            # 日期对齐: 个股停牌日与代理序列取交集
            common = rets.index.intersection(market_rets.index)
            if len(common) >= 30:
                ri = rets.loc[common].values.astype(float)
                rm = market_rets.loc[common].values.astype(float)
                cov_im = np.cov(ri, rm)[0, 1]
                var_m = np.var(rm)
                result['beta'] = cov_im / var_m if var_m > 1e-10 else np.nan
            else:
                result['beta'] = np.nan
        else:
            # 兼容旧式 ndarray (右对齐)
            cl = min(n_ret, len(market_rets))
            ri = rets.values[-cl:].astype(float)
            rm = np.asarray(market_rets, dtype=float)[-cl:]
            mask = ~np.isnan(ri) & ~np.isnan(rm)
            if mask.sum() >= 30:
                cov_im = np.cov(ri[mask], rm[mask])[0, 1]
                var_m = np.var(rm[mask])
                result['beta'] = cov_im / var_m if var_m > 1e-10 else np.nan
            else:
                result['beta'] = np.nan
    else:
        result['beta'] = np.nan

    # 5. Value(BP): bvps / price_adj
    stock_fin = fin_data[fin_data['code'] == code]
    if len(stock_fin) > 0:
        bvps = stock_fin['bvps'].iloc[0]
        price = float(close_adj.iloc[-1])
        if pd.notna(bvps) and bvps > 0 and price > 0:
            result['value'] = bvps / price

    result['daily_rets_252'] = rets.values.tolist()
    # [2026-09-08 市场代理] 带日期索引的收益序列, 供宇宙市值加权市场收益聚合 (两阶段 Phase 1)
    result['rets_series'] = rets
    return result


def _orthogonalize_style_factors(df: pd.DataFrame):
    """[2026-09-08 BARRA 修复 P0-1] 风格因子正交化 (BARRA/CNE5 标准)。

    对每个风格因子: 行业去均值 → 去 Size 线性回归(残差) → Winsorize(1%/99%) → z-score。
      - size 自身: ln(mv) 行业去均值后 z-score (不对 size 回归自身)
      - value: ln(BP) 同上
      - beta/momentum/volatility/liquidity: 去行业 → 去 size 残差 → z-score

    目的: 消除因子间相关性(Value/Momentum/Liquidity 天然与 Size 相关),
    使因子协方差矩阵对角化、约束方向正交、风险可归因。

    [Fix-20260908 P0-1 完整化] 返回 (df, params):
    params 记录每个因子的截面变换参数(行业均值/size回归beta/winsor分位/z统计量),
    供 on-the-fly 补算的持仓股应用同一正交化变换(与快照内股票严格同口径)。
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

    # ── 截面参数收集 (供 on-the-fly 复用同一正交化变换) ──────────────
    params = {}

    def _p_ind_means(x: pd.Series):
        g = x.groupby(ind).mean()
        d = {str(k): float(v) for k, v in g.items() if pd.notna(v)}
        return d, float(x.mean()) if pd.notna(x.mean()) else 0.0

    def _p_winsor_stats(x: pd.Series):
        x = x.dropna()
        if len(x) < 50:
            return None
        lo, hi = float(x.quantile(0.01)), float(x.quantile(0.99))
        xc = x.clip(lo, hi)
        sd = float(xc.std())
        if sd <= 1e-12:
            return None
        return {'lo': lo, 'hi': hi, 'zmean': float(xc.mean()), 'zstd': sd}

    def _p_size_beta(x: pd.Series):
        valid = x.notna() & log_mv.notna()
        if valid.sum() < 50:
            return None
        xv = x[valid].values.astype(float)
        mv = log_mv[valid].values.astype(float)
        mv_c = mv - mv.mean()
        denom = float(np.dot(mv_c, mv_c))
        if denom < 1e-12:
            return None
        return {'beta': float(np.dot(xv - xv.mean(), mv_c) / denom),
                'logmv_mean': float(mv.mean())}

    # size: ln(mv) 行业去均值 → zscore
    im, fb = _p_ind_means(log_mv)
    ws = _p_winsor_stats(_ind_demean(log_mv))
    params['size'] = {'ind_mean': im, 'ind_fallback': fb, 'winsor': ws,
                      'log_input': True, 'size_resid': False}
    df['size_exp'] = _winsor_z(_ind_demean(log_mv)).reindex(df.index)
    # value: ln(bp) 去行业 → 去 size → zscore
    im, fb = _p_ind_means(log_bp)
    sb = _p_size_beta(_ind_demean(log_bp))
    ws = _p_winsor_stats(_size_resid(_ind_demean(log_bp)))
    params['value'] = {'ind_mean': im, 'ind_fallback': fb, 'size_beta': sb,
                       'winsor': ws, 'log_input': True, 'size_resid': True}
    df['value_exp'] = _winsor_z(_size_resid(_ind_demean(log_bp))).reindex(df.index)
    # 其余: 去行业 → 去 size → zscore
    for f in ['beta', 'momentum', 'volatility', 'liquidity']:
        raw = df[f'{f}_raw']
        im, fb = _p_ind_means(raw)
        sb = _p_size_beta(_ind_demean(raw))
        ws = _p_winsor_stats(_size_resid(_ind_demean(raw)))
        params[f] = {'ind_mean': im, 'ind_fallback': fb, 'size_beta': sb,
                     'winsor': ws, 'log_input': False, 'size_resid': True}
        df[f'{f}_exp'] = _winsor_z(_size_resid(_ind_demean(raw))).reindex(df.index)

    return df, params


def _apply_ortho_params(params: dict, factor: str, raw_value: float,
                        logmv_value: float, industry: str) -> float:
    """[Fix-20260908 P0-1] 对单只股票应用快照截面正交化变换(与快照内股票同口径)。

    factor: size/value/beta/momentum/volatility/liquidity
    raw_value: 对应 raw 因子值 (size/value 传原始市值/BP, 函数内部取对数)
    logmv_value: ln(市值) (size 因子自身传 raw_value)
    """
    p = params.get(factor)
    if p is None:
        return np.nan
    # 输入变换: size/value 取对数
    if p.get('log_input'):
        if raw_value is None or not np.isfinite(raw_value) or raw_value <= 0:
            return np.nan
        x = float(np.log(raw_value))
        if factor == 'size':
            logmv_value = x
    else:
        x = float(raw_value) if raw_value is not None and np.isfinite(raw_value) else np.nan
        if not np.isfinite(x):
            return np.nan
    # 行业去均值 (未知行业 → 全局均值 fallback)
    ind_key = str(industry) if industry is not None and str(industry) != 'nan' else 'other'
    im = p.get('ind_mean', {})
    if ind_key in im:
        t = x - im[ind_key]
    else:
        t = x - p.get('ind_fallback', 0.0)
    # 去 size 残差
    if p.get('size_resid') and p.get('size_beta') is not None and np.isfinite(logmv_value):
        sb = p['size_beta']
        t = t - sb['beta'] * (logmv_value - sb['logmv_mean'])
    # winsorize + z-score
    ws = p.get('winsor')
    if ws is None:
        return np.nan
    t = min(max(t, ws['lo']), ws['hi'])
    return (t - ws['zmean']) / ws['zstd']


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


def _load_mv_cache_day(constraint_date: datetime, nearest: bool = False
                       ) -> Optional[pd.DataFrame]:
    """[2026-09-08 彻底解耦] total_mv_cache 截面加载 (BARRA 宇宙唯一主源)。

    数据: data/total_mv_cache.parquet (trade_date, ts_code, total_mv)
      total_mv = amount * 100 / turn (raw_daily 全股票, 修复 sz_000 误过滤后 =
      当日全部成交股票; 与 Wyckoff/V10 管线零依赖)。
    - nearest=False: 精确 constraint_date 当日截面
    - nearest=True:  < constraint_date 的最近截面 (极端兜底)
    返回 code(6位)+total_mv 或 None。
    """
    if not os.path.exists(_MV_CACHE):
        return None
    try:
        import pyarrow.parquet as pq
        if not nearest:
            t = pq.read_table(_MV_CACHE, columns=['trade_date', 'ts_code', 'total_mv'],
                              filters=[('trade_date', '=', constraint_date.date())])
        else:
            t = pq.read_table(_MV_CACHE, columns=['trade_date', 'ts_code', 'total_mv'])
        mv_df = t.to_pandas()
        if len(mv_df) == 0:
            return None
        mv_df['trade_date'] = pd.to_datetime(mv_df['trade_date']).dt.normalize()
        if nearest:
            dmax = mv_df.loc[mv_df['trade_date'] < pd.Timestamp(constraint_date.normalize()),
                             'trade_date'].max()
            if pd.isna(dmax):
                return None
            mv_df = mv_df[mv_df['trade_date'] == dmax]
        mv_df['code'] = mv_df['ts_code'].astype(str).str.split('.').str[-1].str.zfill(6)
        mv_df['total_mv'] = pd.to_numeric(mv_df['total_mv'], errors='coerce')
        mv_df = mv_df[mv_df['total_mv'] > 0]
        out = mv_df[['code', 'total_mv']]
        logger.info(f"[Barra Snapshot] 宇宙 = total_mv_cache "
                    f"{'最近日' if nearest else '当日'}截面 {len(out)} 只")
        return out
    except Exception as e:
        logger.warning(f"[Barra Snapshot] total_mv_cache 加载失败: {e}")
        return None


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


def _aggregate_universe_market_returns(rets_dict: Dict[str, pd.Series],
                                       mv_map: Dict[str, float],
                                       min_stocks: int = 300) -> Optional[pd.Series]:
    """[2026-09-08 业界标准] 宇宙市值加权市场收益序列 (CNE5 市场代理)。

    聚合 r_m(t) = Σ_i w_i·r_i(t) / Σ_{i: r_i(t) 有效} w_i,
    w_i = 快照日截面 total_mv (固定权重, 业界常见"当期权重"近似,
    类似 Fama-French 市场组合的年度更新权重)。

    - 日期对齐: 每交易日仅当日有收益的股票参与, 停牌/缺数据自动剔除
    - 返回 pd.Series(DatetimeIndex); 有效股票 < min_stocks 时返回 None
      (调用方降级为上证指数代理)
    """
    if len(rets_dict) < min_stocks:
        return None
    R = pd.concat(rets_dict, axis=1)   # dates × codes, 自动日期对齐
    w = pd.Series({c: float(mv_map.get(c, np.nan)) for c in R.columns}, dtype=float)
    w = w.where(w > 0)
    wsum = w.sum()
    if not np.isfinite(wsum) or wsum <= 0:
        return None
    w = w / wsum
    num = R.mul(w, axis=1).sum(axis=1)
    den = R.notna().mul(w, axis=1).sum(axis=1)
    rm = num / den.replace(0, np.nan)
    return rm.dropna()


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
    # [Fix-20260908 业界标准] 暴露快照与基线解耦 → 单一命名 barra_expo_{date}.parquet。
    # 基线仅运行时用于聚合 B_bench (见 compute_benchmark_exposures_from_df)。
    suffix = ''
    out_path = os.path.join(_BARRA_DIR, f'barra_expo_{date_str}.parquet')

    if not force_rebuild and os.path.exists(out_path):
        logger.info(f"[Barra Snapshot] {date_str} already exists, loading")
        return pd.read_parquet(out_path)

    logger.info(f"[Barra Snapshot] Building for {constraint_date.date()} (universe=全市场) ...")
    t0 = time.time()

    # 1. 暴露宇宙 = total_mv_cache 当日全 A 截面 (CNE5 全市场口径)
    #    [2026-09-08 彻底解耦] BARRA 宇宙与 Wyckoff/V10 因子快照零依赖:
    #    原实现借用 wyckoff20_v5_snapshots 当宇宙源 (历史惰性耦合, 只读 code+total_mv
    #    两列却受制于 v5 管线的生成与过滤); 现统一改由 data/total_mv_cache.parquet
    #    (raw_daily 全股票 × amount*100/turn, 2018-01~2026-09 全覆盖, 修复深市 000 段
    #    误过滤后 = 当日全部成交股票, 与策略选股快照同日行数一致)。
    #    兜底链: 当日截面 → 最近日前截面 (极端) → 本地成分缓存。
    snap = _load_mv_cache_day(constraint_date)
    if snap is None:
        logger.warning(f"[Barra Snapshot] total_mv_cache 无 {date_str} 当日记录, "
                       f"回溯最近日截面 (极端兜底)")
        snap = _load_mv_cache_day(constraint_date, nearest=True)

    if snap is not None and len(snap) > 0 and 'code' in snap.columns:
        constituents = snap[['code']].copy()
        constituents['code'] = constituents['code'].astype(str).str.zfill(6)
        if 'total_mv' in snap.columns:
            # 保留市值供权重/基准聚合
            constituents = constituents.merge(
                snap[['code', 'total_mv']], on='code', how='left')
            constituents['code'] = constituents['code'].astype(str).str.zfill(6)
        constituents = constituents.drop_duplicates(subset=['code'])
    else:
        # 极端兜底: 本地成分缓存 (仅当 total_mv_cache 完全不可用)
        constituents = _get_constituents(constraint_date, baseline, bs=None)

    if len(constituents) < 500:
        logger.warning(f"Too few universe stocks ({len(constituents)}), aborting")
        return None

    # 2. 已发布财报
    fin_data = _get_latest_financials(constraint_date)
    if len(fin_data) == 0:
        logger.warning("No financial data available, aborting")
        return None

    # 3. 市值数据已随宇宙快照 (snap) 获得
    mv_map = dict(zip(constituents['code'], constituents.get('total_mv', np.nan))) \
        if 'total_mv' in constituents.columns else {}

    # 4. [2026-09-08 业界标准] 两阶段:
    #    Phase 1: 逐股算因子暴露 (beta 延后) + 收集带日期的收益序列
    #    Phase 2: 聚合宇宙市值加权市场收益序列 (CNE5 市场代理) → 逐股重算 beta
    records = []
    rets_dict: Dict[str, pd.Series] = {}
    n_total = len(constituents)
    last_log_pct = 0

    for idx, (_, row) in enumerate(constituents.iterrows()):
        code = row['code']
        daily_data = _load_daily_data(code)

        # 五因子（不复权，本地数据直接算）; market_rets=None → beta 暂置 NaN, Phase 2 填充
        expo = _compute_factor_exposures(
            code, constraint_date, fin_data, daily_data, None
        )
        if expo.get('rets_series') is not None:
            rets_dict[code] = expo['rets_series']

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
            'beta_raw': np.nan,
            'value_raw': expo['value'],
            'momentum_raw': expo['momentum'],
            'volatility_raw': expo['volatility'],
            'liquidity_raw': expo['liquidity'],
            'daily_rets_252': json.dumps(expo['daily_rets_252']) if expo['daily_rets_252'] else None,
        })

    # 4b. Phase 2: 宇宙市值加权市场收益 → 逐股 beta
    market_meta = None
    market_series = _aggregate_universe_market_returns(rets_dict, mv_map)
    if market_series is None or len(market_series) < 30:
        # 降级: 上证指数代理 (宇宙不足时, 保留旧行为)
        logger.warning(f"[Barra Snapshot] 宇宙市值加权市场收益不可用, 降级上证指数代理")
        market_series = _get_market_returns(constraint_date)
    if market_series is not None and len(market_series) >= 30:
        market_meta = {
            'dates': [d.strftime('%Y-%m-%d') for d in market_series.index],
            'values': [float(v) for v in market_series.values],
        }
        n_beta_filled = 0
        for rec in records:
            r = rets_dict.get(rec['code'])
            if r is None:
                continue
            common = r.index.intersection(market_series.index)
            if len(common) < 30:
                continue
            a = r.loc[common].values.astype(float)
            b = market_series.loc[common].values.astype(float)
            var_b = float(np.var(b))
            if var_b > 1e-10:
                rec['beta_raw'] = float(np.cov(a, b)[0, 1] / var_b)
                n_beta_filled += 1
        logger.info(f"[Barra Snapshot] 市场收益序列 {len(market_series)} 天 (宇宙市值加权), "
                    f"beta 填充 {n_beta_filled}/{n_total} 只")
    else:
        logger.warning("[Barra Snapshot] 无可用市场收益序列, beta 全 NaN")

    # [Fix-20260908] df 必须在 Phase 2 之后构建, 否则 beta 填充不进 df
    df = pd.DataFrame(records)

    # 6. [2026-09-08 P0-1] 因子正交化 (Size+行业回归取残差) + Winsorize + z-score
    #    [Fix-20260908] 同时产出截面正交化参数并落盘, 供 on-the-fly 持仓股复用同口径
    for f in STYLE_FACTORS:
        df[f'{f}_exp'] = np.nan
    df, _ortho_params = _orthogonalize_style_factors(df)

    # 清理列 (业界标准: 存 total_mv 供运行时按基线聚合 B_bench; 弃构建期 weight_bm)
    exp_cols = ['code', 'industry', 'report_period']
    if 'total_mv' not in df.columns and 'size_raw' in df.columns:
        df['total_mv'] = df['size_raw']
    if 'total_mv' in df.columns:
        exp_cols.append('total_mv')
    exp_cols += [f'{f}_exp' for f in STYLE_FACTORS]
    exp_cols += ['daily_rets_252']
    out_df = df[exp_cols].copy()

    # 存盘 (附正交化参数 json + 市场收益序列 json, 供 on-the-fly 应用同一变换/口径)
    out_df.attrs['ortho_params'] = _ortho_params
    out_df.attrs['market_rets'] = market_meta
    try:
        with open(out_path.replace('.parquet', '_ortho.json'), 'w', encoding='utf-8') as f:
            json.dump(_ortho_params, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"[Barra Snapshot] 保存正交化参数失败: {e}")
    if market_meta is not None:
        try:
            with open(out_path.replace('.parquet', '_market_rets.json'), 'w', encoding='utf-8') as f:
                json.dump(market_meta, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"[Barra Snapshot] 保存市场收益序列失败: {e}")
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
    # 列举全部快照日期 (业界标准: 单一命名, 无 baseline 后缀)
    if not os.path.isdir(_BARRA_DIR):
        return build_barra_snapshot(constraint_date)
    prefix = 'barra_expo_'
    dates = []
    for fn in os.listdir(_BARRA_DIR):
        if not fn.startswith(prefix) or not fn.endswith('.parquet'):
            continue
        stem = fn[len(prefix):-len('.parquet')]
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
        out_path = os.path.join(_BARRA_DIR, f'barra_expo_{nearest.strftime("%Y%m%d")}.parquet')
        # [Fix-20260908 版本自愈] 旧版快照缺 _ortho.json (正交化参数) 或 _market_rets.json
        # (市场收益序列) → 自动重建该日, 杜绝跨版本口径混跑。
        ortho_path = out_path.replace('.parquet', '_ortho.json')
        mkt_path = out_path.replace('.parquet', '_market_rets.json')
        if not os.path.exists(ortho_path) or not os.path.exists(mkt_path):
            logger.warning(f"[Barra Snapshot] {os.path.basename(out_path)} 缺正交化参数或市场收益序列(旧版), "
                           f"自动重建该日快照...")
            return build_barra_snapshot(nearest, force_rebuild=True)
        logger.debug(f"[Barra Snapshot] 回溯加载 {nearest.strftime('%Y%m%d')} < {constraint_date.strftime('%Y%m%d')} (PIT)")
        df = pd.read_parquet(out_path)
        # [Fix-20260908] 附带正交化参数 (on-the-fly 复用同一变换)
        try:
            with open(ortho_path, encoding='utf-8') as f:
                df.attrs['ortho_params'] = json.load(f)
        except Exception:
            pass
        # [2026-09-08 市场代理] 附带市场收益序列 dict (on-the-fly beta 同口径)
        # 注意: pandas 3.x 中 DataFrame.attrs 不可存放 pd.Series —— concat/nsmallest 等
        # 操作触发 attrs 比较 (Series 无歧义真值) 会崩溃; 只存可安全比较的 dict,
        # Series 由消费方 (compute_on_the_fly_exposures) 现场构造。
        try:
            with open(mkt_path, encoding='utf-8') as f:
                df.attrs['market_rets'] = json.load(f)
        except Exception:
            pass
        return df
    return build_barra_snapshot(constraint_date)


def compute_benchmark_exposures(expo: pd.DataFrame, baseline: str = 'csi800',
                                constraint_date: Optional[datetime] = None) -> np.ndarray:
    """[Fix-20260908 业界标准] 计算基线指数市值加权的风格因子暴露。

    宇宙快照(全市场)与基线解耦后, B_bench = Σ_{i∈基线成分∩快照} mv_i·exp_i / Σ mv_i。
    - baseline + constraint_date 提供时: 本地成分缓存查基线成分, total_mv 加权
    - 兼容旧快照(含 weight_bm): 直接用它
    - 兜底: 全宇宙 total_mv 加权
    """
    if 'weight_bm' in expo.columns:
        w = expo['weight_bm'].fillna(0).values
        w = w / w.sum() if w.sum() > 0 else np.ones(len(expo)) / len(expo)
    elif constraint_date is not None and 'total_mv' in expo.columns:
        # [2026-09-08 微盘/沪市基准] sh932000/sh000001 无本地成分缓存,
        # 运行时从宇宙快照派生成分 (PIT: 快照日截面, 无未来成分调整):
        #   sh932000 ≈ 市值排名 1801-3800 (中证2000 官方口径近似)
        #   sh000001 ≈ 沪市全部 (code 以 6 开头)
        sub = None
        if baseline == 'sh932000':
            s = expo.dropna(subset=['total_mv'])
            s = s[s['total_mv'] > 0].sort_values('total_mv', ascending=False)
            sub = s.iloc[1800:3800].copy()
        elif baseline == 'sh000001':
            sub = expo[expo['code'].astype(str).str.startswith('6')].copy()
        if sub is not None and len(sub) >= 100:
            mv = sub['total_mv'].fillna(0).values
            wsum = mv.sum()
            if wsum > 0:
                w = mv / wsum
                sub['_w'] = w
            else:
                sub['_w'] = 1.0 / len(sub)
            benchmark = np.zeros(len(STYLE_FACTORS))
            for k, f in enumerate(STYLE_FACTORS):
                col = f'{f}_exp'
                if col in sub.columns:
                    v = sub[col].fillna(0).values
                    benchmark[k] = float(np.dot(sub['_w'].values, v))
            logger.info(f"[Barra] 基线 {baseline} 由宇宙快照派生成分: {len(sub)} 只")
            return benchmark
        const = _get_constituents(constraint_date, baseline, bs=None)
        if len(const) > 0 and 'code' in const.columns:
            codes = set(const['code'].astype(str).str.zfill(6))
            sub = expo[expo['code'].isin(codes)].copy()
            if len(sub) >= 100:
                mv = sub['total_mv'].fillna(0).values
                wsum = mv.sum()
                if wsum > 0:
                    w = mv / wsum
                    sub['_w'] = w
                else:
                    sub['_w'] = 1.0 / len(sub)
                benchmark = np.zeros(len(STYLE_FACTORS))
                for k, f in enumerate(STYLE_FACTORS):
                    col = f'{f}_exp'
                    if col in sub.columns:
                        v = sub[col].fillna(0).values
                        benchmark[k] = float(np.dot(sub['_w'].values, v))
                return benchmark
    # 兜底: 全宇宙等权/total_mv 加权
    if 'total_mv' in expo.columns:
        w = expo['total_mv'].fillna(0).values
        w = w / w.sum() if w.sum() > 0 else np.ones(len(expo)) / len(expo)
    else:
        w = np.ones(len(expo)) / max(len(expo), 1)

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
    - 截面 Z-score 以 prebuilt 快照截面统计量为基准（与快照内股票同口径）

    [Fix-20260908] code 统一归一为 6 位（与快照一致）后再取数；修复此前
    ts_code 格式('sz.300588')导致的数据加载失败 → 暴露全 NaN → B_port 全 0 的缺陷。

    Args:
        codes: 需要计算的股票代码列表 (6 位或 ts_code 均可)
        constraint_date: 约束日（T-1）
        prebuilt_expo: 预构建的暴露快照（截面统计量来源）

    Returns:
        DataFrame，列与快照一致 (code 为 6 位格式)
    """
    fin_data = _get_latest_financials(constraint_date)
    if len(fin_data) == 0:
        return pd.DataFrame()

    # [Fix-20260908] 统一 6 位 code 做取数/匹配
    codes6 = [_norm_code6(c) for c in codes]

    # [2026-09-08 市场代理] 优先复用快照的宇宙市值加权市场收益序列 (与快照内 beta 同口径);
    # 旧快照无此参数时降级上证指数代理。
    # 注意: attrs 只存 dict (pandas 3.x 下 Series 会引发 concat attrs 比较崩溃),
    # Series 在此处现场构造。
    market_rets = None
    m = getattr(prebuilt_expo, 'attrs', {}).get('market_rets')
    if isinstance(m, dict) and len(m.get('values', [])) >= 30:
        market_rets = pd.Series(m['values'], index=pd.to_datetime(m['dates']))
    if market_rets is None:
        market_rets = _get_market_returns(constraint_date)

    # prebuilt 截面统计量（降级路径用: 旧快照无 ortho 参数时）
    prebuilt_stats = {}
    for factor_name in ['beta_raw', 'momentum_raw', 'volatility_raw', 'liquidity_raw',
                        'size_raw', 'value_raw']:
        exp_col = factor_name.replace('_raw', '_exp')
        if exp_col in prebuilt_expo.columns:
            vals_pb = prebuilt_expo[exp_col].dropna()
            if len(vals_pb) >= 50:
                prebuilt_stats[factor_name] = {
                    'mean': float(vals_pb.mean()), 'std': float(vals_pb.std()),
                    'lo': float(vals_pb.quantile(0.01)), 'hi': float(vals_pb.quantile(0.99)),
                }

    records = []
    for code, code6 in zip(codes, codes6):
        daily_data = _load_daily_data(code6)
        expo = _compute_factor_exposures(code6, constraint_date, fin_data,
                                         daily_data, market_rets)

        stock_fin = fin_data[fin_data['code'] == code6]
        industry = 'other'
        report_period = ''
        if len(stock_fin) > 0:
            industry = stock_fin['industry'].iloc[0] if pd.notna(stock_fin['industry'].iloc[0]) else 'other'
            report_period = stock_fin['_report_period'].iloc[0]

        # size: _compute_factor_exposures 不产出 size; 用 amount/turn 估算市值
        # (与快照 total_mv 定义一致: 市值 ≈ 成交额/换手率, log 尺度近似同源)
        size_val = np.nan
        if daily_data is not None and len(daily_data) > 0:
            last_row = daily_data[daily_data.index <= constraint_date]
            if len(last_row) > 0 and 'amount' in last_row.columns and 'turn' in last_row.columns:
                amt = last_row['amount'].iloc[-1]
                trn = last_row['turn'].iloc[-1]
                if pd.notna(amt) and pd.notna(trn) and trn > 0:
                    size_val = float(amt) / (trn / 100.0)

        records.append({
            'code': code6,
            'weight_bm': np.nan,
            'industry': industry,
            'report_period': report_period,
            'size_raw': size_val,
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

    # [Fix-20260908 P0-1 完整化] 优先: 应用快照落盘的正交化参数
    # (industry demean + size 回归残差 + winsor + z, 与快照内股票严格同口径)。
    ortho = getattr(prebuilt_expo, 'attrs', {}).get('ortho_params') \
        if hasattr(prebuilt_expo, 'attrs') else None
    if ortho:
        for idx, row in df.iterrows():
            logmv = np.nan
            sr = row.get('size_raw')
            if sr is not None and np.isfinite(sr) and sr > 0:
                logmv = float(np.log(sr))
            ind_v = row.get('industry', 'other')
            for f in STYLE_FACTORS:
                raw_col = 'size_raw' if f == 'size' else f'{f}_raw'
                rv = row.get(raw_col, np.nan)
                df.at[idx, f'{f}_exp'] = _apply_ortho_params(
                    ortho, f, rv, logmv, ind_v)
    else:
        # 降级(旧快照无参数): 用快照截面统计量做 winsorize+z-score 近似
        prebuilt_stats = {}
        for factor_name in ['size_raw', 'beta_raw', 'momentum_raw', 'volatility_raw',
                            'liquidity_raw', 'value_raw']:
            exp_col = factor_name.replace('_raw', '_exp')
            if exp_col in prebuilt_expo.columns:
                vals_pb = prebuilt_expo[exp_col].dropna()
                if len(vals_pb) >= 50:
                    prebuilt_stats[factor_name] = {
                        'mean': float(vals_pb.mean()), 'std': float(vals_pb.std()),
                        'lo': float(vals_pb.quantile(0.01)), 'hi': float(vals_pb.quantile(0.99)),
                    }
        for factor_name in ['size_raw', 'beta_raw', 'momentum_raw', 'volatility_raw',
                            'liquidity_raw', 'value_raw']:
            exp_col = factor_name.replace('_raw', '_exp')
            raw_series = df[factor_name]
            valid = raw_series.notna()
            if valid.sum() < 1:
                continue
            vals = raw_series.loc[valid]
            stats = prebuilt_stats.get(factor_name)
            if stats and stats['std'] > 0:
                lo, hi, mean, std = stats['lo'], stats['hi'], stats['mean'], stats['std']
            else:
                lo, hi = vals.quantile(0.01), vals.quantile(0.99)
                mean, std = (vals.mean(), vals.std()) if len(vals) > 1 else (0.0, 1.0)
                if std == 0:
                    std = 1.0
            vals_c = vals.clip(lo, hi)
            df.loc[valid, exp_col] = ((vals_c - mean) / std).values

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
