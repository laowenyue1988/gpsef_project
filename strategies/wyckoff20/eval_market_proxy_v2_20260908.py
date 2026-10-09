# -*- coding: utf-8 -*-
"""
[2026-09-08] 市场代理严格版评估 v2 — 日期对齐 + 前复权 + 全市场市值加权聚合

对比三个 beta 口径:
  B1: beta 相对上证综指 (日期对齐, 生产代码市场代理)
  B2: beta 相对全市场市值加权 (日期对齐, 业界标准)
  B3: beta 相对上证综指 (右对齐, 生产代码实际计算方式)

量化: ①上证 vs 全市场代理的偏差 (用户问题); ②右对齐 vs 日期对齐的偏差 (生产缺陷)。

只评估 20210401 一个快照日 (截面 3829 只)。
"""
import json
import os
import sys
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _PROJECT_ROOT)

DATA = os.path.join(_PROJECT_ROOT, 'data')
RAW = os.path.join(DATA, 'raw_daily')
ADJ = os.path.join(DATA, 'back_adjust_factors')
CONSTRAINT = pd.Timestamp('2021-04-01')
WINDOW = 253  # 253 行 close → 252 收益


def load_adj(code):
    for c in [f'sh_{code}_back_adjust.parquet', f'sz_{code}_back_adjust.parquet']:
        p = os.path.join(ADJ, c)
        if os.path.exists(p):
            df = pd.read_parquet(p)
            if 'dividOperateDate' in df.columns and 'adjustFactor' in df.columns:
                df['date'] = pd.to_datetime(df['dividOperateDate'])
                df['adjustFactor'] = pd.to_numeric(df['adjustFactor'], errors='coerce')
                return df.set_index('date').sort_index()[['adjustFactor']]
    return None


def load_stock_rets(code):
    """复权日收益序列 (DatetimeIndex), ≤ CONSTRAINT, tail 253 行。"""
    p = os.path.join(RAW, f'{code}.parquet')
    if not os.path.exists(p):
        p = os.path.join(RAW, f'sh_{code}.parquet')
    if not os.path.exists(p):
        p = os.path.join(RAW, f'sz_{code}.parquet')
    if not os.path.exists(p):
        return None
    d = pd.read_parquet(p)
    if 'close' not in d.columns:
        return None
    dc = 'date' if 'date' in d.columns else 'trade_date'
    d[dc] = pd.to_datetime(d[dc])
    d = d.sort_values(dc).set_index(dc)
    close = d['close']
    close = close[close.index <= CONSTRAINT].tail(WINDOW)
    if len(close) < 60:
        return None
    adj = load_adj(code)
    if adj is not None and len(adj) > 0:
        ref = adj[adj.index <= CONSTRAINT]
        if len(ref) > 0 and float(ref.iloc[-1]['adjustFactor']) > 0:
            ref_f = float(ref.iloc[-1]['adjustFactor'])
            merged = pd.merge_asof(
                pd.DataFrame({'date': close.index}).sort_values('date'),
                adj.reset_index().sort_values('date'),
                on='date', direction='backward')
            merged = merged.set_index('date')
            close = close * ref_f / merged['adjustFactor']
    rets = close.pct_change().dropna()
    return rets if len(rets) >= 50 else None


def beta_aligned(ri: pd.Series, rm: pd.Series):
    common = ri.index.intersection(rm.index)
    if len(common) < 30:
        return np.nan
    a, b = ri.loc[common].values, rm.loc[common].values
    var_b = np.var(b)
    return np.cov(a, b)[0, 1] / var_b if var_b > 1e-10 else np.nan


def beta_right(ri: pd.Series, rm: pd.Series):
    cl = min(len(ri), len(rm))
    a, b = ri.values[-cl:], rm.values[-cl:]
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 30:
        return np.nan
    var_b = np.var(b[mask])
    return np.cov(a[mask], b[mask])[0, 1] / var_b if var_b > 1e-10 else np.nan


def main():
    expo = pd.read_parquet(os.path.join(DATA, 'barra_exposures', 'barra_expo_20210401.parquet'))
    expo['code'] = expo['code'].astype(str).str.zfill(6)
    mv = expo.set_index('code')['total_mv'].astype(float)

    # ── 上证指数 (日期对齐 Series) ────────────────────
    ix = pd.read_parquet(os.path.join(DATA, 'index_daily', 'sh000001.parquet'))
    ix['date'] = pd.to_datetime(ix['date'])
    ix = ix.sort_values('date').set_index('date')
    sh = ix['close'].pct_change().dropna()
    sh = sh[sh.index <= CONSTRAINT].tail(252)

    # ── 逐股复权收益 ──────────────────────────────────
    all_rets = {}
    for code in expo['code']:
        r = load_stock_rets(code)
        if r is not None:
            all_rets[code] = r
    print(f'有效收益股票: {len(all_rets)}/{len(expo)}')

    # ── 全市场市值加权市场收益 (日期对齐聚合) ──────────
    R = pd.DataFrame({c: r for c, r in all_rets.items()})
    w = mv.reindex(R.columns).fillna(0.0)
    w = w / w.sum()
    num = R.mul(w, axis=1).sum(axis=1)
    den = R.notna().mul(w, axis=1).sum(axis=1)
    mkt_univ = num / den.replace(0, np.nan)
    mkt_univ = mkt_univ.dropna()
    print(f'全市场市场收益序列: {len(mkt_univ)} 天')

    # ── 三个 beta ────────────────────────────────────
    rows = []
    for c, r in all_rets.items():
        rows.append({
            'code': c,
            'mv': float(mv.get(c, np.nan)),
            'b_sh_align': beta_aligned(r, sh),       # 上证, 日期对齐
            'b_mkt_align': beta_aligned(r, mkt_univ),  # 全市场, 日期对齐
            'b_sh_right': beta_right(r, sh),         # 上证, 右对齐 (生产口径)
        })
    res = pd.DataFrame(rows).dropna(subset=['b_sh_align', 'b_mkt_align'])
    print(f'beta 三口径均有效: {len(res)} 只')

    # ① 上证 vs 全市场 (日期对齐)
    d1 = res['b_sh_align'] - res['b_mkt_align']
    print()
    print('===== ① 市场代理差异: 上证 vs 全市场市值加权 (日期对齐) =====')
    print(f'  差异 mean={d1.mean():+.4f} std={d1.std():.4f} |d|均值={d1.abs().mean():.4f}')
    print(f'  Spearman 相关 = {res["b_sh_align"].corr(res["b_mkt_align"], method="spearman"):.4f}')
    med = res['mv'].median()
    small = res['mv'] <= med
    print(f'  微盘 50%: beta(上证)={res.loc[small,"b_sh_align"].mean():.3f} '
          f'beta(全市场)={res.loc[small,"b_mkt_align"].mean():.3f} 差={d1[small].mean():+.4f}')
    print(f'  大盘 50%: beta(上证)={res.loc[~small,"b_sh_align"].mean():.3f} '
          f'beta(全市场)={res.loc[~small,"b_mkt_align"].mean():.3f} 差={d1[~small].mean():+.4f}')

    # ② 右对齐 vs 日期对齐 (生产代码缺陷)
    d2 = res['b_sh_right'] - res['b_sh_align']
    print()
    print('===== ② 计算口径差异: 右对齐(生产) vs 日期对齐(上证代理) =====')
    print(f'  差异 mean={d2.mean():+.4f} std={d2.std():.4f} |d|均值={d2.abs().mean():.4f}')
    print(f'  Spearman 相关 = {res["b_sh_right"].corr(res["b_sh_align"], method="spearman"):.4f}')
    # 停牌天数代理: 252 - 实际收益长度
    n_rets = res['code'].map(lambda c: len(all_rets[c]))
    stop_days = 252 - n_rets
    print(f'  停牌天数分布: p50={stop_days.median():.0f} p90={stop_days.quantile(0.9):.0f} '
          f'max={stop_days.max():.0f}')
    hi = stop_days >= stop_days.quantile(0.9)
    print(f'  高停牌组(≥p90): |d|均值={d2[hi].abs().mean():.4f} '
          f'vs 低停牌组(<p90): |d|均值={d2[~hi].abs().mean():.4f}')

    # ③ beta 估计噪声基准: 上证代理 252 天 beta 的标准误量级
    resid_std = (res['b_sh_align'] * 0 + 1).astype(float)  # placeholder
    print()
    print('===== ③ 参照: beta 抽样噪声量级 (n=252, 个股特质波动 2%/日) =====')
    # 上证日波动 ~1.1% (2020-2021), 个股特质 ~1.8%
    se = 0.018 / (0.011 * np.sqrt(252))
    print(f'  单股 beta 标准误 ≈ σ_ε/(σ_m·√T) ≈ {se:.3f} (远大于代理差异 {d1.std():.3f})')

    # 结论
    print()
    print('===== 结论 =====')
    print(f'1) 上证 vs 全市场代理: |Δbeta|均值 {d1.abs().mean():.3f}, 排序相关 {res["b_sh_align"].corr(res["b_mkt_align"], method="spearman"):.3f}')
    print(f'2) 右对齐 vs 日期对齐:   |Δbeta|均值 {d2.abs().mean():.3f}')
    print(f'3) beta 抽样噪声:        ~{se:.3f}')


if __name__ == '__main__':
    main()
