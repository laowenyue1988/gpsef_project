# -*- coding: utf-8 -*-
"""
测试模块: Barra 约束优化器

验证:
  1. alpha_builder: IC缩放 + Z-score → 量纲正确
  2. ewma_covariance: EWMA 协方差正定、对称
  3. industry_deviation: 行业偏离计算正确
  4. softmax_fallback: 数据不足时降级 softmax
  5. barra_optimization: 约束全部生效
  6. benchmark_exposures: 基准暴露计算
  7. config_parse: barra_constraint 配置段解析
  8. integration: 集成到策略不报错
"""
import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from strategies.wyckoff20.barra_constraint import (
    build_alpha_from_preds,
    compute_ewma_covariance,
    compute_covariance_from_snapshot,
    compute_industry_deviation,
    optimize_portfolio,
    softmax_weights,
    solve_barra_weights,
    compute_benchmark_exposures_from_df,
    DEFAULT_TOLERANCES,
    STYLE_FACTORS,
)
from strategies.wyckoff20.tests.conftest import make_strategy


# ══════════════════════════════════════════════════════════════
# Test 1: alpha 构造
# ══════════════════════════════════════════════════════════════

def test_alpha_builder():
    """IC缩放 + Z-score 后 alpha 量纲正确。"""
    rng = np.random.RandomState(42)
    preds = 0.5 + 0.1 * rng.randn(50)  # 模拟校准概率

    alpha = build_alpha_from_preds(preds, expected_ic=0.05, vol_target=0.15)

    assert len(alpha) == 50
    assert abs(alpha.mean()) < 1e-10, f"alpha should be zero-mean, got {alpha.mean():.6f}"

    # alpha 标准差 ≈ expected_ic × vol_target = 0.0075
    assert 0.005 < alpha.std() < 0.015, f"alpha std should be ~0.0075, got {alpha.std():.6f}"

    # 正 pred → 正 alpha, 负 pred → 负 alpha
    assert alpha[preds > preds.mean()].mean() > 0
    assert alpha[preds < preds.mean()].mean() < 0

    print(f"[PASS] test_alpha_builder: mean={alpha.mean():.6f}, std={alpha.std():.6f}")


# ══════════════════════════════════════════════════════════════
# Test 2: EWMA 协方差
# ══════════════════════════════════════════════════════════════

def test_ewma_covariance():
    """EWMA 协方差正定、对称、量纲正确。"""
    rng = np.random.RandomState(42)
    T, N = 252, 30
    # 生成有相关结构的收益率
    factors = rng.randn(T, 3) * 0.01
    loadings = rng.randn(N, 3) * 0.2
    rets = factors @ loadings.T + rng.randn(T, N) * 0.005

    Sigma = compute_ewma_covariance(rets, decay=0.94)

    assert Sigma.shape == (N, N)
    # 对称
    assert np.allclose(Sigma, Sigma.T, atol=1e-8), "Cov should be symmetric"
    # 正定（特征值 > 0）
    eigvals = np.linalg.eigvalsh(Sigma)
    assert np.all(eigvals > 0), f"All eigenvalues should be > 0, min={eigvals.min():.2e}"
    # 年化量纲 (~15% vol → var ~0.0225)
    diag_mean = np.mean(np.diag(Sigma))
    assert 0.005 < diag_mean < 0.10, f"Annualized var should be >0, got {diag_mean:.4f}"

    print(f"[PASS] test_ewma_covariance: diag_mean={diag_mean:.4f}, min_eig={eigvals.min():.2e}")


def test_ewma_covariance_nan_handling():
    """EWMA 协方差处理 NaN 不崩溃。"""
    rng = np.random.RandomState(42)
    rets = rng.randn(100, 10) * 0.01
    rets[10:20, 3] = np.nan  # 部分 NaN
    rets[50:60, 7] = np.nan

    Sigma = compute_ewma_covariance(rets, decay=0.94)
    assert Sigma.shape == (10, 10)
    assert not np.any(np.isnan(Sigma)), "No NaN in output"
    # 正定
    eigvals = np.linalg.eigvalsh(Sigma)
    assert np.all(eigvals > 0)

    print(f"[PASS] test_ewma_covariance_nan_handling: OK")


# ══════════════════════════════════════════════════════════════
# Test 3: 行业偏离
# ══════════════════════════════════════════════════════════════

def test_industry_deviation():
    """行业偏离 = 0.5 × Σ|w_i^port - w_i^bench|."""
    bench_ind = {'金融': 0.3, '科技': 0.2, '消费': 0.2, '工业': 0.3}
    codes = ['A', 'B', 'C', 'D']
    inds = ['金融', '科技', '消费', '工业']
    weights = np.array([0.25, 0.25, 0.25, 0.25])

    dev = compute_industry_deviation(weights, codes, inds, bench_ind)

    # 组合: 金融=0.25, 科技=0.25, 消费=0.25, 工业=0.25
    # 偏离: |0.25-0.3| + |0.25-0.2| + |0.25-0.2| + |0.25-0.3| = 0.05+0.05+0.05+0.05 = 0.20
    # 0.5 × 0.20 = 0.10
    assert abs(dev - 0.10) < 0.001, f"Expected 0.10, got {dev:.4f}"

    # 完全等于基准 → 偏离 = 0
    dev2 = compute_industry_deviation(
        np.array([0.3, 0.2, 0.2, 0.3]), codes, inds, bench_ind
    )
    assert dev2 < 0.001, f"Should be ~0, got {dev2:.4f}"

    print(f"[PASS] test_industry_deviation: equal_dev=0.10, perfect_dev={dev2:.4f}")


# ══════════════════════════════════════════════════════════════
# Test 4: Softmax 降级
# ══════════════════════════════════════════════════════════════

def test_softmax_fallback():
    """Softmax 降级路径正确。"""
    preds = np.array([0.6, 0.58, 0.55, 0.52, 0.50])
    
    # 构建最小 barra_expo（只有 3 只股票有数据 < min_positions=20）
    expo = pd.DataFrame({
        'code': ['000001', '000002', '000003'],
        'weight_bm': [0.01, 0.01, 0.01],
        'industry': ['金融', '科技', '消费'],
        'report_period': ['20251231'] * 3,
        'daily_rets_252': [json.dumps(list(np.random.randn(200) * 0.01)) for _ in range(3)],
    })
    for f in STYLE_FACTORS:
        expo[f'{f}_exp'] = np.random.randn(3) * 0.5

    w, is_barra = solve_barra_weights(preds, ['000001', '000002', '000003', '000004', '000005'],
                                        expo)

    assert not is_barra, "Should fallback because only 3 stocks have Barra data"
    assert len(w) == 5
    assert abs(w.sum() - 1.0) < 0.01

    print(f"[PASS] test_softmax_fallback: is_barra={is_barra}")


# ══════════════════════════════════════════════════════════════
# Test 5: Barra 优化
# ══════════════════════════════════════════════════════════════

def test_barra_optimization():
    """Barra QP 优化：约束全部生效。"""
    n = 30
    rng = np.random.RandomState(42)
    codes = [f'{600000 + i:06d}' for i in range(n)]

    # 模拟 alpha: 前 10 只高分
    alpha = np.zeros(n)
    alpha[:10] = rng.rand(10) * 0.01 + 0.005
    alpha[10:] = rng.rand(20) * 0.005 - 0.002

    # 模拟协方差: 对角主导
    Sigma = np.diag(rng.rand(n) * 0.01 + 0.015)  # ~15% vol

    # 因子暴露: 前 10 只因子完全匹配基准，后 20 只有偏离
    B_port = np.zeros((n, 5))
    B_bench = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    for i in range(n):
        B_port[i] = rng.randn(5) * 0.2

    # 行业
    inds_list = [f'行业{j%5}' for j in range(n)]
    bench_ind = {f'行业{k}': 0.2 for k in range(5)}

    w = optimize_portfolio(
        alpha, Sigma, B_port, B_bench, DEFAULT_TOLERANCES,
        codes, inds_list, bench_ind,
        risk_aversion=1.0, max_weight=0.10
    )

    assert w is not None, "Optimizer should return weights"
    assert abs(w.sum() - 1.0) < 0.01, f"Sum should be 1.0, got {w.sum():.4f}"
    assert np.all(w >= 0), "All weights should be >= 0"
    assert np.all(w <= 0.10 + 0.02), f"Max weight should be <= 0.10, got {w.max():.3f}"

    # 验证因子偏离约束
    port_expo = B_port.T @ w
    for k in range(5):
        diff = abs(port_expo[k] - B_bench[k])
        tol = DEFAULT_TOLERANCES[k] + 0.05  # 允许 0.05 松弛（SLSQP 近似解）
        assert diff <= tol, f"Factor {k}: diff={diff:.4f} > tol={tol:.4f}"

    print(f"[PASS] test_barra_optimization: sum={w.sum():.3f}, "
          f"max_w={w.max():.3f}, n_pos={int((w>0).sum())}")


# ══════════════════════════════════════════════════════════════
# Test 6: 基准暴露计算
# ══════════════════════════════════════════════════════════════

def test_benchmark_exposures():
    """CSI 800 基准暴露 = 市值加权均值。"""
    expo = pd.DataFrame({
        'code': ['A', 'B', 'C'],
        'weight_bm': [0.5, 0.3, 0.2],
        'size_exp': [1.0, 0.0, -1.0],
        'beta_exp': [0.5, 1.0, 1.5],
        'value_exp': [0.0, 0.0, 0.0],
        'momentum_exp': [0.0, 0.0, 0.0],
        'volatility_exp': [0.0, 0.0, 0.0],
    })
    bench = compute_benchmark_exposures_from_df(expo)
    
    # Size: 0.5×1 + 0.3×0 + 0.2×(-1) = 0.5 - 0.2 = 0.3
    assert abs(bench[0] - 0.3) < 0.01, f"Size bench: {bench[0]:.4f}"
    # Beta: 0.5×0.5 + 0.3×1.0 + 0.2×1.5 = 0.25 + 0.30 + 0.30 = 0.85
    assert abs(bench[1] - 0.85) < 0.01, f"Beta bench: {bench[1]:.4f}"

    print(f"[PASS] test_benchmark_exposures: bench={bench}")


# ══════════════════════════════════════════════════════════════
# Test 7: Config 解析
# ══════════════════════════════════════════════════════════════

def test_barra_config_parse():
    """barra_constraint 配置段正确解析。"""
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config_v10.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    assert 'barra_constraint' in config, "barra_constraint section should exist"
    bc = config['barra_constraint']

    assert bc['enabled'] in (True, False), f"enabled should be bool, got {bc['enabled']}"
    assert bc['lambda_deviation'] == 0.5
    assert bc['lambda_l2'] == 0.02
    assert bc['max_weight'] == 0.10
    assert len(bc['style_factors']) == 5

    names = [sf['name'] for sf in bc['style_factors']]
    assert names == ['size', 'beta', 'value', 'momentum', 'volatility']

    print(f"[PASS] test_barra_config_parse: OK")


# ══════════════════════════════════════════════════════════════
# Test 8: 策略集成（不报错 + 默认降级 softmax）
# ══════════════════════════════════════════════════════════════

def test_strategy_integration():
    """策略实例化后 barra 属性正确初始化（默认关闭）。"""
    s = make_strategy()

    assert hasattr(s, '_barra_enabled')
    assert s._barra_enabled == False, "Barra should be disabled by default"
    assert hasattr(s, '_barra_lambda_dev')
    assert s._barra_lambda_dev == 0.5
    assert hasattr(s, '_barra_lambda_l2')
    assert s._barra_lambda_l2 == 0.02
    assert hasattr(s, '_barra_max_weight')
    assert hasattr(s, '_barra_min_positions')
    assert s._barra_method == 'factor_neutral'

    print(f"[PASS] test_strategy_integration: method={s._barra_method}")


# ══════════════════════════════════════════════════════════════
# Test 9: 协方差从快照构建
# ══════════════════════════════════════════════════════════════

def test_covariance_from_snapshot():
    """从 daily_rets_252 JSON 列构建协方差。"""
    rng = np.random.RandomState(42)
    n = 10
    expo = pd.DataFrame({
        'code': [f'{600000 + i:06d}' for i in range(n)],
        'weight_bm': [0.1] * n,
        'industry': ['金融'] * n,
        'daily_rets_252': [json.dumps((rng.randn(252) * 0.01).tolist()) for _ in range(n)],
    })

    Sigma = compute_covariance_from_snapshot(expo, expo['code'].tolist())
    assert Sigma.shape == (n, n)
    assert np.allclose(Sigma, Sigma.T)
    eigvals = np.linalg.eigvalsh(Sigma)
    assert np.all(eigvals > 0), "Should be positive definite"

    # 一只股票无数据
    expo2 = expo.copy()
    expo2.loc[0, 'daily_rets_252'] = None
    Sigma2 = compute_covariance_from_snapshot(expo2, expo2['code'].tolist())
    assert not np.any(np.isnan(Sigma2))
    # 无数据的对角线应填默认值
    assert Sigma2[0, 0] > 0

    print(f"[PASS] test_covariance_from_snapshot: OK")


if __name__ == '__main__':
    test_alpha_builder()
    test_ewma_covariance()
    test_ewma_covariance_nan_handling()
    test_industry_deviation()
    test_softmax_fallback()
    test_barra_optimization()
    test_benchmark_exposures()
    test_barra_config_parse()
    test_strategy_integration()
    test_covariance_from_snapshot()
    print("\n=== All Barra tests PASSED ===")
