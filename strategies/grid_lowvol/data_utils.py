# -*- coding: utf-8 -*-
"""grid_lowvol 数据工具: 全市场日线预加载/RV 初筛 + 5 分钟线预加载(缓存) + 交易日历。

- 日线: data/raw_daily/*.parquet (5441 只, close 复权口径经 data_provider);
- 5 分钟线: data/raw_minute/5min/*.parquet (5415 只, 2020~2026, 48 根/日, 已排序);
- RV20 初筛: 全市场向量化 <1s → 低波动候选池。
"""
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DAILY_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw_daily')
MINUTE_5M_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw_minute', '5min')
DIVIDEND_DIR = os.path.join(PROJECT_ROOT, 'data', 'dividend_data')

_minute_cache: Dict[str, pd.DataFrame] = {}
_daily_ret_cache: Dict[str, pd.Series] = {}
_factor_cache: Dict[str, pd.Series] = {}


def _load_back_factor(code: str) -> Optional[pd.Series]:
    """后复权因子 (按 trade_date, 前向填充)。进程内缓存。"""
    if code in _factor_cache:
        return _factor_cache[code]
    safe = code.replace('sh.', 'sh_').replace('sz.', 'sz_')
    path = os.path.join(PROJECT_ROOT, 'data', 'back_adjust_factors', f'{safe}_back_adjust.parquet')
    if not os.path.exists(path):
        return None
    try:
        b = pd.read_parquet(path, columns=['dividOperateDate', 'backAdjustFactor'])
        b['dividOperateDate'] = pd.to_datetime(b['dividOperateDate'])
        s = b.set_index('dividOperateDate')['backAdjustFactor'].sort_index()
        _factor_cache[code] = s
        return s
    except Exception:
        return None


def preload_daily_returns_fast(codes: List[str], start: datetime, end: datetime
                               ) -> Dict[str, pd.Series]:
    """轻量直读前复权日收益率 (不经过 data_provider, 避免全市场大缓存 OOM)。

    直读 data/raw_daily (close 2列) + back_adjust_factors (后复权因子) →
    前复权 close = raw × factor(t)/factor(ref) → pct_change。
    性能: ~10ms/只 (两个 parquet 列切片) → 全市场 5400 只 ≈ 1 分钟, 内存 <100MB。
    """
    out: Dict[str, pd.Series] = {}
    for code in codes:
        safe = code.replace('sh.', 'sh_').replace('sz.', 'sz_')
        path = os.path.join(RAW_DAILY_DIR, f'{safe}.parquet')
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_parquet(path, columns=['trade_date', 'close'])
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.set_index('trade_date').sort_index()['close']
            df = df[(df.index >= start) & (df.index <= end)]
            if len(df) < 30:
                continue
            f = _load_back_factor(code)
            if f is not None and len(f) > 0:
                # 前复权: raw × factor(t)/factor(窗口末端)
                ref = float(f.iloc[-1])
                fac = f.reindex(df.index.union(f.index)).ffill().reindex(df.index)
                df = df * fac / ref
            r = df.pct_change().replace([np.inf, -np.inf], np.nan)
            out[code] = r
        except Exception:
            continue
    return out


def preload_daily_returns(data_provider, codes: List[str],
                          start: datetime, end: datetime,
                          reference_date: Optional[datetime] = None) -> Dict[str, pd.Series]:
    """预加载复权日线收盘价 → 日收益率序列 (全市场, 供 RV 初筛与 EGARCH)。"""
    out: Dict[str, pd.Series] = {}
    for code in codes:
        try:
            df = data_provider.get_daily_data(code, start, end,
                                               reference_date=reference_date)
            if df is None or df.empty or 'close' not in df.columns:
                continue
            c = df[['trade_date', 'close']].copy()
            c['trade_date'] = pd.to_datetime(c['trade_date'])
            c = c.set_index('trade_date').sort_index()['close']
            r = c.pct_change().replace([np.inf, -np.inf], np.nan)
            out[code] = r
        except Exception:
            continue
    return out


def has_dividend_data(code: str) -> bool:
    """检查股票是否有除权除息数据 (data/dividend_data)。

    [2026-08-21 exdiv-fix3] 框架 engine 对无 dividend_data 的股票走旧 bf fallback
    (对含送转事件不精确) — 策略在选池时对目标股缺口预警。
    """
    safe = code.replace('sh.', 'sh_').replace('sz.', 'sz_')
    return os.path.exists(os.path.join(DIVIDEND_DIR, f'{safe}_dividend.parquet'))


_industry_map: Optional[Dict[str, str]] = None


def load_industry_map() -> Dict[str, str]:
    """加载行业映射 {6位code: 行业名} (data/industry_map.parquet, 进程内缓存)。

    [2026-08-22 opt-P2] 行业分散约束用: 候选池选 top_n 时按行业上限去重,
    控制 size/value 风格过度集中 (BARRA 归因: size=-0.86/value=+0.80 拖累 -13pp)。
    """
    global _industry_map
    if _industry_map is None:
        _industry_map = {}
        try:
            m = pd.read_parquet(os.path.join(PROJECT_ROOT, 'data', 'industry_map.parquet'))
            for _, r in m.iterrows():
                code = str(r['code']).zfill(6)
                ind = str(r['industry'])
                if code and ind and ind != 'nan':
                    _industry_map[code] = ind
        except Exception as e:
            logger.warning(f"[grid_lowvol] 行业映射加载失败: {e}")
    return _industry_map


_mv_df: Optional[pd.DataFrame] = None


def _load_mv_cache() -> Optional[pd.DataFrame]:
    """加载全市场每日市值缓存 (data/total_mv_cache.parquet, 进程内缓存)。

    由 small_alpha191 的 _ensure_total_mv_cache 构建 (total_mv = amount×100/turn),
    列: trade_date(datetime), ts_code('sh.600000'), total_mv。
    """
    global _mv_df
    if _mv_df is None:
        try:
            _mv_df = pd.read_parquet(os.path.join(PROJECT_ROOT, 'data', 'total_mv_cache.parquet'))
            _mv_df['trade_date'] = pd.to_datetime(_mv_df['trade_date'])
        except Exception as e:
            logger.warning(f"[grid_lowvol] 市值缓存加载失败: {e}")
            _mv_df = None
    return _mv_df


def smallcap_pool(only_codes: Set[str], date: datetime, n: int = 300) -> List[str]:
    """过滤池 ∩ 市值最低 n 只 (≤date 最新市值, 严格无未来)。

    [2026-08-22 opt] 用户提议: 基础池 = 框架过滤池后市值最低 n 只 (微盘/小盘)。
    市值缓存缺失/不足时回退原池。注意: 微盘股波动高/流动性弱, 与网格弱适配,
    建议作为实验参数 (pool_subset=smallcap) 经完整回测裁决。
    """
    df = _load_mv_cache()
    if df is None or not only_codes:
        return sorted(only_codes)
    sub = df[df['trade_date'] <= pd.Timestamp(date)]
    if sub.empty:
        return sorted(only_codes)
    latest = (sub.sort_values('trade_date')
              .groupby('ts_code').last()[['total_mv']])
    cand = latest[latest.index.isin(only_codes)].sort_values('total_mv')
    if len(cand) < 2:
        return sorted(only_codes)
    return cand.index.tolist()[:n]


def compute_rv20(returns: pd.Series) -> float:
    """20 日已实现波动率 (年化): std×√252。不足 10 个样本 → NaN。"""
    r = returns.dropna()
    if len(r) < 10:
        return np.nan
    return float(r.tail(20).std() * np.sqrt(252))


def top_low_rv(returns_map: Dict[str, pd.Series], date: datetime,
               k: int = 200, min_days: int = 20,
               only_codes: Optional[Set[str]] = None) -> List[str]:
    """RV20 初筛: 返回波动率最低的 k 只候选 (≤T-1 数据, 严格无未来)。

    [2026-08-21 pool-fix] 新增 only_codes: 限定基础股票池 (DynamicStockPool 过滤池);
    全市场选低波动会选入退市股/ST股 (如 sz.000606/000005 已退市但历史有低波动期)。

    Returns: 按 RV 升序的 code 列表 (长度 ≤ k)。
    """
    scores = []
    for code, r in returns_map.items():
        if only_codes is not None and code not in only_codes:
            continue
        idx = r.index[r.index < date]   # [2026-08-21 exdiv-fix1] 严格≤T-1: 排除当日(盘前不可知)收益率
        if len(idx) < min_days:
            continue
        rv = compute_rv20(r.loc[idx])
        if np.isfinite(rv):
            scores.append((rv, code))
    scores.sort()
    return [c for _, c in scores[:k]]


def load_minute_5m(code: str) -> Optional[pd.DataFrame]:
    """加载 5 分钟线 (进程内缓存)。返回 DataFrame(trade_time/trade_date/open/high/low/close)。"""
    if code in _minute_cache:
        return _minute_cache[code]
    safe = code.replace('sh.', 'sh_').replace('sz.', 'sz_')
    path = os.path.join(MINUTE_5M_DIR, f'{safe}.parquet')
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path, columns=['trade_time', 'trade_date', 'open',
                                            'high', 'low', 'close'])
        df = df.sort_values('trade_time').reset_index(drop=True)
        _minute_cache[code] = df
        return df
    except Exception as e:
        logger.warning(f"[grid_lowvol] 加载 5 分钟线失败 {code}: {e}")
        return None


def minute_of_day(df: pd.DataFrame, date: datetime) -> Optional[pd.DataFrame]:
    """取指定交易日的 5 分钟线 (按时间排序)。无数据 → None。"""
    if df is None or df.empty:
        return None
    dstr = pd.Timestamp(date).strftime('%Y-%m-%d')
    sub = df[df['trade_date'] == dstr]
    return sub if not sub.empty else None
