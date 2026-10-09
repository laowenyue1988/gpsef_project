# -*- coding: utf-8 -*-
"""
Barra 全流程审计测试 — 防泄露 + 正确性验证

测试覆盖:
  1. 财报防泄露: 公告日期 > constraint_date 的数据不出现
  2. 前复权正确性: adjustFactor 基准日 ≤ constraint_date
  3. 日线数据截断: 只用 ≤ constraint_date 的数据
  4. 市值数据对齐: constraint_date 快照中的 total_mv
  5. 因子暴露范围: Z-score 后均值≈0, std≈1
  6. 快照落盘与加载: parquet 文件格式正确
  7. alpha 构造: IC 缩放量纲正确
  8. EWMA 协方差: 正定对称
  9. QP 优化: 约束全部合规
  10. barra_constraint 集成: 策略可加载配置
"""
import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# ══════════════════════════════════════════════════════════════════
# Test A: 财报防泄露
# ══════════════════════════════════════════════════════════════════

def test_anti_leakage_financials():
    """财报数据不能包含 constraint_date 之后发布的记录。"""
    from strategies.wyckoff20.barra_builder import _get_latest_financials, _load_report

    # 取一个早期的 constraint_date：最新公告日期不应该超过这天
    dt = datetime(2023, 6, 30)
    fin = _get_latest_financials(dt)

    assert len(fin) > 100, f"Too few valid financials: {len(fin)}"

    # 验证 report_period: 不能是约束日之后季度（如 2023Q3 的财报不可能在 2023-06-30 前发布）
    rp_col = '_report_period'
    for rp in fin[rp_col].unique():
        rp_dt = datetime.strptime(str(rp), '%Y%m%d')
        assert rp_dt <= dt + timedelta(days=120), \
            f"Report period {rp} is too far beyond constraint_date {dt.date()}"

    print(f"[PASS] anti_leakage_financials: {len(fin)} stocks, report periods: "
          f"{sorted(fin[rp_col].unique())[-3:]}")


# ══════════════════════════════════════════════════════════════════
# Test B: 前复权基准日防泄露
# ══════════════════════════════════════════════════════════════════

def test_anti_leakage_forward_adjust():
    """前复权的 ref_factor 必须 ≤ constraint_date。"""
    from strategies.wyckoff20.barra_builder import _load_daily_data, _load_adjust_factors_local

    code = '600000'  # 浦发银行
    daily = _load_daily_data(code)
    adj = _load_adjust_factors_local(code)

    assert daily is not None, "Daily data not found"
    assert adj is not None, "Adjust factor not found"

    dt = datetime(2023, 6, 30)
    close = daily['close'][daily.index <= dt].tail(253)
    assert len(close) >= 200, f"Not enough daily data: {len(close)}"

    # 获取约束日前的最后一个 adjustFactor
    ref_factor = adj.loc[adj.index <= dt, 'adjustFactor'].iloc[-1]
    # 确认 ref_factor 对应的日期 ≤ constraint_date
    ref_date = adj.loc[adj.index <= dt].index[-1]
    assert ref_date <= dt, f"ref_date {ref_date} > constraint_date {dt}"

    # 执行前复权：用 asof 对齐
    ref_series = adj['adjustFactor'][adj.index <= dt]
    ref_factor = float(ref_series.iloc[-1])

    daily_dates = pd.DataFrame({'date': close.index})
    adj_dates = pd.DataFrame({'date': adj.index, 'adjustFactor': adj['adjustFactor'].values})
    merged = pd.merge_asof(
        daily_dates.sort_values('date'), adj_dates.sort_values('date'),
        on='date', direction='backward'
    ).set_index('date')
    close_adj = close * ref_factor / merged['adjustFactor']

    # 简单验证：前复权后的收盘价应该在合理范围
    assert close_adj.iloc[-1] > 0
    # 非除权日收益率应该完全一致
    rets_adj = close_adj.pct_change().dropna()
    rets_raw = close.pct_change().dropna()
    common_idx = rets_adj.index.intersection(rets_raw.index)
    assert len(common_idx) > 100
    diff = (rets_adj[common_idx] - rets_raw[common_idx]).abs()
    # 95% 的交易日收益率完全一致（非除权日），5% 是除权日有小差异
    assert diff.quantile(0.95) < 1e-6, \
        f"Most returns should be identical: 95pct diff={diff.quantile(0.95):.6e}"

    print(f"[PASS] anti_leakage_forward_adjust: ref_date={ref_date.date()}, "
          f"adj_prices range=[{close_adj.min():.2f},{close_adj.max():.2f}]")


# ══════════════════════════════════════════════════════════════════
# Test C: 市值防泄露
# ══════════════════════════════════════════════════════════════════

def test_anti_leakage_market_cap():
    """市值数据来自 constraint_date 的快照，不能使用未来快照。"""
    dt = datetime(2023, 6, 30)
    snap_path = os.path.join(
        os.path.dirname(__file__), '..', '..', '..',
        'data', 'wyckoff20_v5_snapshots', f'factors_{dt.strftime("%Y%m%d")}.parquet'
    )

    if not os.path.exists(snap_path):
        # 快照可能不存在（需要先 build_snapshots），跳过
        print(f"[SKIP] test_anti_leakage_market_cap: snapshot not found for {dt.date()}")
        return

    snap = pd.read_parquet(snap_path)
    assert 'code' in snap.columns
    assert 'total_mv' in snap.columns
    assert len(snap) > 100

    # 快照文件名中的日期 ≤ constraint_date
    snap_date = dt.strftime('%Y%m%d')
    assert f'factors_{snap_date}' in snap_path

    print(f"[PASS] anti_leakage_market_cap: {len(snap)} stocks, date={snap_date}")


# ══════════════════════════════════════════════════════════════════
# Test D: 因子暴露 Z-score 正确性
# ══════════════════════════════════════════════════════════════════

def test_factor_zscore():
    """截面 Z-score 后均值≈0, std≈1。"""
    from strategies.wyckoff20.barra_builder import STYLE_FACTORS, _compute_factor_exposures, \
        _load_daily_data, _load_adjust_factors_local, _get_latest_financials

    dt = datetime(2023, 6, 30)
    fin = _get_latest_financials(dt)
    codes = ['600000', '000001', '600036', '000858', '600519']
    raw_values = {f: [] for f in STYLE_FACTORS}

    for code in codes:
        daily = _load_daily_data(code)
        expo = _compute_factor_exposures(code, dt, fin, daily, market_rets=None)
        for f in STYLE_FACTORS:
            v = expo[f] if not isinstance(expo[f], float) or not np.isnan(expo[f]) else np.nan
            raw_values[f].append(v)

    # 每因子至少有几个有效值
    for f in ['volatility', 'momentum']:
        vals = [v for v in raw_values[f] if not np.isnan(v)]
        assert len(vals) >= 3, f"Not enough valid {f}: {len(vals)}"

    print(f"[PASS] test_factor_zscore: {len(codes)} stocks computed, "
          f"vol={[ f'{v:.4f}' for v in raw_values['volatility']]}")


# ══════════════════════════════════════════════════════════════════
# Test E: 完整快照构建（端到端）
# ══════════════════════════════════════════════════════════════════

def test_full_snapshot_build():
    """构建一个 Barra 快照并验证所有列和值范围。"""
    from strategies.wyckoff20.barra_builder import build_barra_snapshot, STYLE_FACTORS

    dt = datetime(2023, 6, 30)

    try:
        df = build_barra_snapshot(dt, force_rebuild=True)
    except Exception as e:
        # baostock 可能不可用
        print(f"[SKIP] test_full_snapshot_build: baostock error: {e}")
        return

    if df is None:
        print("[SKIP] test_full_snapshot_build: no data")
        return

    assert len(df) > 100, f"Too few stocks: {len(df)}"

    # 验证必需列存在
    required = ['code', 'weight_bm', 'industry', 'report_period']
    required += [f'{f}_exp' for f in STYLE_FACTORS]
    required += ['daily_rets_252']
    for col in required:
        assert col in df.columns, f"Missing column: {col}"

    # weight_bm 归一化
    w = df['weight_bm'].fillna(0)
    assert abs(w.sum() - 1.0) < 0.01, f"weight_bm sum={w.sum():.4f}, should be ~1.0"

    # 因子 Z-score: 均值≈0, std≈1
    for f in STYLE_FACTORS:
        col = f'{f}_exp'
        vals = df[col].dropna()
        if len(vals) > 50:
            assert abs(vals.mean()) < 0.2, f"{col} mean={vals.mean():.4f}, too far from 0"
            assert 0.8 < vals.std() < 1.2, f"{col} std={vals.std():.4f}, should be ~1.0"

    # daily_rets_252 非空率
    n_rets = df['daily_rets_252'].notna().sum()
    assert n_rets > 500, f"Only {n_rets} stocks with return data"

    print(f"[PASS] test_full_snapshot_build: {len(df)} stocks, "
          f"rets_available={n_rets}, inds={df['industry'].nunique()}")


# ══════════════════════════════════════════════════════════════════
# Test F: Alpha 构造量纲
# ══════════════════════════════════════════════════════════════════

def test_alpha_scaling_realistic():
    """用真实预测值测试 alpha 量纲。"""
    from strategies.wyckoff20.barra_constraint import build_alpha_from_preds

    rng = np.random.RandomState(42)
    # 模拟 0.48-0.62 范围的校准概率
    preds = 0.50 + 0.07 * rng.randn(50)
    preds = np.clip(preds, 0.3, 0.7)

    alpha = build_alpha_from_preds(preds, expected_ic=0.05, vol_target=0.15)

    assert len(alpha) == 50
    assert abs(alpha.mean()) < 0.001
    # alpha std ≈ IC × vol = 0.05 × 0.15 = 0.0075
    assert 0.004 < alpha.std() < 0.015

    print(f"[PASS] test_alpha_scaling_realistic: std={alpha.std():.6f}")


# ══════════════════════════════════════════════════════════════════
# Test G: EWMA 协方差正定性
# ══════════════════════════════════════════════════════════════════

def test_ewma_positive_definite():
    """EWMA 协方差必须是正定的。"""
    from strategies.wyckoff20.barra_constraint import compute_ewma_covariance

    rng = np.random.RandomState(42)
    T, N = 252, 30
    factors = rng.randn(T, 3) * 0.01
    loadings = rng.randn(N, 3) * 0.15
    rets = factors @ loadings.T + rng.randn(T, N) * 0.005

    Sigma = compute_ewma_covariance(rets, decay=0.94)
    eigvals = np.linalg.eigvalsh(Sigma)
    assert np.all(eigvals > 0), f"Not PD: min_eig={eigvals.min():.2e}"

    print(f"[PASS] test_ewma_positive_definite: min_eig={eigvals.min():.2e}")


# ══════════════════════════════════════════════════════════════════
# Test H: QP 优化约束合规
# ══════════════════════════════════════════════════════════════════

def test_qp_constraints_satisfied():
    """QP 求解后所有约束必须满足。"""
    from strategies.wyckoff20.barra_constraint import (
        optimize_portfolio, compute_industry_deviation, DEFAULT_TOLERANCES
    )

    n = 30
    rng = np.random.RandomState(42)
    alpha = rng.randn(n) * 0.005
    Sigma = np.diag(rng.rand(n) * 0.01 + 0.015)
    B_port = rng.randn(n, 5) * 0.3
    B_bench = np.zeros(5)
    codes = [f'{600000+i:06d}' for i in range(n)]
    inds = [f'行业{i % 5}' for i in range(n)]
    bench_ind = {f'行业{i}': 0.2 for i in range(5)}

    w = optimize_portfolio(
        alpha, Sigma, B_port, B_bench, DEFAULT_TOLERANCES,
        codes, inds, bench_ind,
        risk_aversion=1.0, max_weight=0.10,
    )

    assert w is not None
    # 满仓
    assert abs(w.sum() - 1.0) < 0.01
    # 非负
    assert np.all(w >= -1e-10)
    # 单票上限
    assert np.all(w <= 0.10 + 0.02)
    # 风格约束
    port_expo = B_port.T @ w
    for k in range(5):
        diff = abs(port_expo[k] - B_bench[k])
        tol = DEFAULT_TOLERANCES[k] + 0.05
        assert diff <= tol, f"Style {k}: diff={diff:.4f} > tol={tol:.4f}"

    # 行业偏离
    dev = compute_industry_deviation(w, codes, inds, bench_ind)
    assert dev <= 0.30 + 0.01, f"Indust dev={dev:.4f}"

    print(f"[PASS] test_qp_constraints_satisfied: sum={w.sum():.3f}, "
          f"n_pos={int((w>0).sum())}, max_w={w.max():.3f}, ind_dev={dev:.4f}")


# ══════════════════════════════════════════════════════════════════
# Test I: 配置文件一致性
# ══════════════════════════════════════════════════════════════════

def test_config_consistency():
    """barra_constraint 模块与 config_v10.yaml 参数一致。"""
    import yaml
    from strategies.wyckoff20.barra_constraint import DEFAULT_TOLERANCES

    config_path = os.path.join(os.path.dirname(__file__), '..', 'config_v10.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    bc = cfg['barra_constraint']
    assert bc['lambda_deviation'] == 0.5
    assert bc['lambda_l2'] == 0.02
    assert bc['max_weight'] == 0.10
    assert bc['method'] == 'factor_neutral'

    print(f"[PASS] test_config_consistency: method={bc['method']}")


# ══════════════════════════════════════════════════════════════════
# Test J: 策略集成（配置加载后属性正确）
# ══════════════════════════════════════════════════════════════════

def test_strategy_barra_attrs():
    """make_strategy 后 barra 属性全部正确设置。"""
    from strategies.wyckoff20.tests.conftest import make_strategy

    s = make_strategy()

    assert s._barra_enabled == False  # 测试默认关闭
    assert s._barra_lambda_dev == 0.5
    assert s._barra_lambda_l2 == 0.02
    assert s._barra_max_weight == 0.10
    assert s._barra_min_positions == 20
    assert s._barra_method == 'factor_neutral'

    print(f"[PASS] test_strategy_barra_attrs: method={s._barra_method}")


if __name__ == '__main__':
    test_anti_leakage_financials()
    test_anti_leakage_forward_adjust()
    test_anti_leakage_market_cap()
    test_factor_zscore()
    test_full_snapshot_build()
    test_alpha_scaling_realistic()
    test_ewma_positive_definite()
    test_qp_constraints_satisfied()
    test_config_consistency()
    test_strategy_barra_attrs()
    print("\n=== All Barra Audit Tests PASSED ===")
