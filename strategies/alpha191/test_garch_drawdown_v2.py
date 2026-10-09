"""
GARCH 仓位控制 & 回撤叠加保护 — 功能测试套件 v2
================================================
修正版: 修正了分散化效应、冷却触发阈值、状态依赖等问题
"""
import sys
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from garch_position import GARCHPositionController

# ============================================================
# 辅助函数
# ============================================================

def make_config(**overrides):
    cfg = {
        'vol_target': 0.20,
        'lookback': 250,
        'vol_method': 'garch',
        'ewma_lambda': 0.94,
        'min_scale': 0.25,
        'max_scale': 1.0,
        'drawdown_limit': 0.20,
        'use_drawdown_control': True,
        'cooldown_periods': 3,
    }
    cfg.update(overrides)
    return cfg


def gen_correlated_returns(n=250, n_stocks=20, daily_vol=0.02, corr=0.6, seed=42):
    """
    生成相关的多股票收益率 — 更真实
    corr=0.6 时，20只股票的组合波动率约 = daily_vol * sqrt((1+(n-1)*corr)/n)
    = 0.02 * sqrt((1+19*0.6)/20) = 0.02 * sqrt(0.62) = 0.02 * 0.787 = 0.01574
    年化 = 0.01574 * sqrt(252) = 0.2499 ≈ 25%
    """
    rng = np.random.RandomState(seed)
    # 生成相关矩阵
    cov = np.full((n_stocks, n_stocks), corr * daily_vol**2)
    np.fill_diagonal(cov, daily_vol**2)
    # Cholesky分解
    L = np.linalg.cholesky(cov)
    returns = rng.randn(n, n_stocks) @ L.T
    return returns


def gen_garch_returns(n=250, omega=0.00002, alpha=0.08, beta=0.91, seed=42):
    """生成真实的 GARCH(1,1) 收益率序列"""
    rng = np.random.RandomState(seed)
    r = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        z = rng.randn()
        r[t] = np.sqrt(sigma2[t-1]) * z
        sigma2[t] = omega + alpha * r[t-1]**2 + beta * sigma2[t-1]
    return r


def gen_volatility_clustering(n=250, seed=42):
    """波动率聚集: 前半低波动, 后半高波动"""
    rng = np.random.RandomState(seed)
    r = np.zeros(n)
    for t in range(n):
        if t < n // 2:
            vol = 0.008
        else:
            vol = 0.025
        r[t] = rng.randn() * vol
    return r


def gen_crash_returns(n=250, crash_day=200, crash_mag=-0.06, seed=42):
    """含崩盘事件的收益率序列"""
    rng = np.random.RandomState(seed)
    r = rng.randn(n) * 0.012
    r[crash_day] = crash_mag
    for t in range(crash_day + 1, min(crash_day + 30, n)):
        r[t] = rng.randn() * 0.03
    return r


def print_result(test_name, scale, annual_vol, info, expected_range=None):
    status = "PASS" if expected_range and expected_range[0] <= scale <= expected_range[1] else "INFO"
    print(f"\n{'='*70}")
    print(f"  TEST: {test_name}")
    print(f"{'='*70}")
    print(f"  final_scale     = {scale:.4f}  ({scale*100:.1f}%)")
    print(f"  annual_vol      = {annual_vol:.4f}  ({annual_vol*100:.2f}%)")
    print(f"  method          = {info.get('method', 'N/A')}")
    print(f"  daily_vol       = {info.get('daily_vol', 0):.6f}")
    print(f"  vol_scale       = {info.get('vol_scale', 0):.4f}")
    print(f"  current_dd      = {info.get('current_drawdown', 0):.4f}  ({info.get('current_drawdown', 0)*100:.2f}%)")
    print(f"  dd_scale        = {info.get('dd_scale', 0):.4f}")
    print(f"  converged       = {info.get('converged', False)}")
    print(f"  cooldown_active = {info.get('cooldown_active', False)}")
    if expected_range:
        in_range = expected_range[0] <= scale <= expected_range[1]
        print(f"  expected_range  = [{expected_range[0]:.2f}, {expected_range[1]:.2f}]")
        print(f"  STATUS          = {'PASS' if in_range else 'FAIL'} {'[OK]' if in_range else '[FAIL]'}")


# ============================================================
# 测试套件
# ============================================================

def test_1_normal_low_volatility():
    """
    测试1: 正常低波动市场 — 无回撤
    经济学场景: 2024年初A股平稳运行
    使用相关股票(corr=0.6, vol=1.2%/天)，组合年化波动率约15%
    """
    print("\n" + "#"*70)
    print("  TEST SUITE 1: 正常低波动市场（无回撤）")
    print("  场景: 2024年初A股平稳运行，20只相关股票，corr=0.6, vol=1.2%/天")
    print("#"*70)

    cfg = make_config()
    ctrl = GARCHPositionController(cfg)

    N, T = 20, 250
    returns = gen_correlated_returns(T, N, daily_vol=0.012, corr=0.6, seed=42)
    weights = np.ones(N) / N
    nav = 1_000_000

    scale, annual_vol, info = ctrl.compute(weights, returns, nav)
    print_result("正常低波动（无回撤）", scale, annual_vol, info, expected_range=(0.8, 1.0))

    port_ret = returns @ weights
    simple_vol = np.std(port_ret) * np.sqrt(252)
    print(f"\n  [验证] 简单历史年化波动率 = {simple_vol:.4f} ({simple_vol*100:.2f}%)")
    print(f"  [验证] GARCH年化波动率    = {annual_vol:.4f} ({annual_vol*100:.2f}%)")
    print(f"  [验证] vol_scale = 0.20/{annual_vol:.4f} = {0.20/annual_vol:.4f} (clip前)")
    print(f"  [验证] vol_scale clip后  = {info['vol_scale']:.4f}")
    print(f"  [验证] 无回撤→dd_scale=1.0→final=min(vol_scale,1.0)={'有效' if scale >= 0.8 else '异常'}")
    return scale, info


def test_2_high_volatility():
    """
    测试2: 高波动市场 — 无回撤
    经济学场景: 2015年6月股灾，股票相关性飙升至0.8，日均波动3%
    组合年化波动率约 = 0.03 * sqrt((1+19*0.8)/20) * sqrt(252) ≈ 47%
    预期: vol_scale = 0.20/0.47 ≈ 0.42
    """
    print("\n" + "#"*70)
    print("  TEST SUITE 2: 高波动市场（无回撤）")
    print("  场景: 2015年6月股灾，corr=0.8, vol=3.0%/天")
    print("#"*70)

    cfg = make_config()
    ctrl = GARCHPositionController(cfg)

    N, T = 20, 250
    returns = gen_correlated_returns(T, N, daily_vol=0.03, corr=0.8, seed=42)
    weights = np.ones(N) / N
    nav = 1_000_000

    scale, annual_vol, info = ctrl.compute(weights, returns, nav)
    print_result("高波动（无回撤）", scale, annual_vol, info, expected_range=(0.25, 0.6))

    port_ret = returns @ weights
    simple_vol = np.std(port_ret) * np.sqrt(252)
    expected_vol_scale = 0.20 / annual_vol
    print(f"\n  [验证] 简单历史年化波动率 = {simple_vol:.4f} ({simple_vol*100:.2f}%)")
    print(f"  [验证] GARCH年化波动率    = {annual_vol:.4f} ({annual_vol*100:.2f}%)")
    print(f"  [验证] 理论vol_scale     = {expected_vol_scale:.4f}")
    print(f"  [验证] 实际vol_scale     = {info['vol_scale']:.4f}")
    print(f"  [验证] 高波动→降仓 = {'有效' if scale < 0.6 else '不足'}")
    print(f"  [验证] 波动率目标化 = target_vol/actual_vol = 0.20/{annual_vol:.4f} = {0.20/annual_vol:.4f}")
    return scale, info


def test_3_moderate_drawdown():
    """
    测试3: 中等回撤（12%）— 波动率正常
    经济学场景: 组合从峰值120万回撤至105.6万（12%回撤）
    预期: dd_scale = max(0.25, 1-0.12/0.20) = 0.40, 冷却触发(12%>10%)
    """
    print("\n" + "#"*70)
    print("  TEST SUITE 3: 中等回撤（12%）— 波动率正常")
    print("  场景: 组合从峰值120万回撤至105.6万（12%回撤）")
    print("#"*70)

    cfg = make_config()
    ctrl = GARCHPositionController(cfg)

    N, T = 20, 250
    returns = gen_correlated_returns(T, N, daily_vol=0.012, corr=0.6, seed=42)
    weights = np.ones(N) / N
    nav = 1_056_000  # 12%回撤
    ctrl._nav_peak = 1_200_000  # 预设峰值

    scale, annual_vol, info = ctrl.compute(weights, returns, nav)
    print_result("中等回撤12%", scale, annual_vol, info, expected_range=(0.35, 0.50))

    expected_dd = 0.12
    expected_dd_scale = max(0.25, 1 - expected_dd / 0.20)
    print(f"\n  [验证] 理论回撤     = {expected_dd:.4f} ({expected_dd*100:.1f}%)")
    print(f"  [验证] 理论dd_scale = max(0.25, 1-{expected_dd}/0.20) = {expected_dd_scale:.4f}")
    print(f"  [验证] 实际dd_scale = {info['dd_scale']:.4f}")
    dd_diff = abs(info['dd_scale'] - expected_dd_scale)
    print(f"  [验证] dd_scale偏差 = {dd_diff:.6f} {'OK' if dd_diff < 0.001 else 'MISMATCH'}")
    print(f"  [验证] 冷却触发条件 = dd(12%) > drawdown_limit*0.5(10%) = True")
    print(f"  [验证] 冷却触发     = {'是' if info['cooldown_active'] else '否'} {'OK' if info['cooldown_active'] else 'FAIL'}")
    return scale, info


def test_4_severe_drawdown():
    """
    测试4: 严重回撤（25%）— 触及地板
    经济学场景: 组合从峰值120万回撤至90万（25%回撤）
    预期: dd_scale = max(0.25, 1-0.25/0.20) = max(0.25, -0.25) = 0.25
    """
    print("\n" + "#"*70)
    print("  TEST SUITE 4: 严重回撤（25%）— 触及地板")
    print("  场景: 组合从峰值120万回撤至90万（25%回撤）")
    print("#"*70)

    cfg = make_config()
    ctrl = GARCHPositionController(cfg)

    N, T = 20, 250
    returns = gen_correlated_returns(T, N, daily_vol=0.015, corr=0.6, seed=42)
    weights = np.ones(N) / N
    nav = 900_000
    ctrl._nav_peak = 1_200_000

    scale, annual_vol, info = ctrl.compute(weights, returns, nav)
    print_result("严重回撤25%", scale, annual_vol, info, expected_range=(0.25, 0.26))

    print(f"\n  [验证] 理论dd_scale = max(0.25, 1-0.25/0.20) = max(0.25, -0.25) = 0.25 (地板)")
    print(f"  [验证] 实际dd_scale = {info['dd_scale']:.4f}")
    print(f"  [验证] 触及地板     = {'是' if abs(info['dd_scale'] - 0.25) < 0.001 else '否'}")
    print(f"  [验证] 冷却触发     = {'是' if info['cooldown_active'] else '否'}")
    return scale, info


def test_5_cooldown_protection():
    """
    测试5: 冷却期防"V型反弹"
    使用12%回撤（>10%阈值）确保冷却触发
    """
    print("\n" + "#"*70)
    print("  TEST SUITE 5: 冷却期防V型反弹")
    print("  场景: 12%回撤触发冷却后，净值快速恢复至峰值")
    print("#"*70)

    cfg = make_config(cooldown_periods=3)
    ctrl = GARCHPositionController(cfg)

    N, T = 20, 250
    returns = gen_correlated_returns(T, N, daily_vol=0.012, corr=0.6, seed=42)
    weights = np.ones(N) / N

    # Phase 1: 12%回撤触发冷却
    print("\n  --- Phase 1: 触发回撤（NAV=105.6万，峰值120万，回撤12%）---")
    nav = 1_056_000
    ctrl._nav_peak = 1_200_000
    scale1, _, info1 = ctrl.compute(weights, returns, nav)
    print(f"  scale={scale1:.4f}, dd_scale={info1['dd_scale']:.4f}, cooldown_active={info1['cooldown_active']}")
    print(f"  cooldown_remaining={ctrl._cooldown_remaining}, last_scale={ctrl._last_scale:.4f}")

    # Phase 2: 净值恢复到峰值，冷却第1轮
    print("\n  --- Phase 2: 净值恢复至120万（冷却第1轮，应阻止全仓）---")
    nav = 1_200_000
    scale2, _, info2 = ctrl.compute(weights, returns, nav)
    print(f"  scale={scale2:.4f}, dd_scale={info2['dd_scale']:.4f}, cooldown_active={info2['cooldown_active']}")
    print(f"  cooldown_remaining={ctrl._cooldown_remaining}")
    print(f"  [验证] 冷却阻止全仓 = {'是 OK' if scale2 < 1.0 else '否 FAIL — V型反弹未被阻止!'}")
    print(f"  [验证] dd_scale锁定 = min(1.0, last_scale={info1.get('final_scale',0):.4f}) = {info2['dd_scale']:.4f}")

    # Phase 3: 冷却第2轮
    print("\n  --- Phase 3: 净值保持120万（冷却第2轮）---")
    scale3, _, info3 = ctrl.compute(weights, returns, nav)
    print(f"  scale={scale3:.4f}, cooldown_active={info3['cooldown_active']}, remaining={ctrl._cooldown_remaining}")

    # Phase 4: 冷却第3轮
    print("\n  --- Phase 4: 冷却第3轮（最后一轮）---")
    scale4, _, info4 = ctrl.compute(weights, returns, nav)
    print(f"  scale={scale4:.4f}, cooldown_active={info4['cooldown_active']}, remaining={ctrl._cooldown_remaining}")

    # Phase 5: 冷却结束
    print("\n  --- Phase 5: 冷却结束，应恢复全仓 ---")
    scale5, _, info5 = ctrl.compute(weights, returns, nav)
    print(f"  scale={scale5:.4f}, cooldown_active={info5['cooldown_active']}")
    print(f"  [验证] 冷却后恢复全仓 = {'是 OK' if scale5 > scale2 else '否 FAIL'}")

    # 关键结论
    print(f"\n  === 冷却机制验证总结 ===")
    print(f"  Phase1(回撤12%): scale={scale1:.4f} — 回撤降仓")
    print(f"  Phase2(恢复,冷却1): scale={scale2:.4f} — 冷却锁定,防V型反弹")
    print(f"  Phase3(冷却2):     scale={scale3:.4f}")
    print(f"  Phase4(冷却3):     scale={scale4:.4f}")
    print(f"  Phase5(冷却结束):  scale={scale5:.4f} — 恢复全仓")
    print(f"  冷却期内scale是否<=Phase1: {all(s <= scale1 + 0.001 for s in [scale2, scale3, scale4])}")

    return [scale1, scale2, scale3, scale4, scale5]


def test_6_volatility_clustering():
    """
    测试6: 波动率聚集效应 — 单只股票GARCH
    使用单只股票避免分散化效应
    """
    print("\n" + "#"*70)
    print("  TEST SUITE 6: 波动率聚集效应（单资产GARCH）")
    print("  场景: 市场从低波动(~0.8%)突然转入高波动(~2.5%)")
    print("#"*70)

    cfg = make_config()
    ctrl = GARCHPositionController(cfg)

    T = 250
    rets = gen_volatility_clustering(T, seed=42)
    returns = rets.reshape(-1, 1)  # 单只股票
    weights = np.array([1.0])
    nav = 1_000_000

    scale, annual_vol, info = ctrl.compute(weights, returns, nav)
    print_result("波动率聚集", scale, annual_vol, info, expected_range=(0.25, 0.8))

    simple_vol = np.std(rets) * np.sqrt(252)
    pre_vol = np.std(rets[:T//2]) * np.sqrt(252)
    post_vol = np.std(rets[T//2:]) * np.sqrt(252)
    print(f"\n  [验证] 前半段年化波动率 = {pre_vol:.4f} ({pre_vol*100:.2f}%)")
    print(f"  [验证] 后半段年化波动率 = {post_vol:.4f} ({post_vol*100:.2f}%)")
    print(f"  [验证] 简单历史年化波动率 = {simple_vol:.4f} ({simple_vol*100:.2f}%)")
    print(f"  [验证] GARCH年化波动率 = {annual_vol:.4f} ({annual_vol*100:.2f}%)")
    print(f"  [验证] GARCH > 简单波动率 = {'是 — GARCH预测了更高的条件波动' if annual_vol > simple_vol else '否'}")
    print(f"  [验证] GARCH接近后半段波动率 = {'是' if abs(annual_vol - post_vol)/post_vol < 0.3 else '否'}")
    print(f"         (GARCH应更关注近期高波动，因α+β=0.99持续性高)")
    return scale, info


def test_7_crash_event():
    """
    测试7: 单日崩盘事件（单资产）
    """
    print("\n" + "#"*70)
    print("  TEST SUITE 7: 单日崩盘事件（2015黑色星期一风格）")
    print("  场景: 250天中第200天暴跌6%，之后波动率飙升")
    print("#"*70)

    cfg = make_config()
    ctrl = GARCHPositionController(cfg)

    T = 250
    rets = gen_crash_returns(T, crash_day=200, crash_mag=-0.06, seed=42)
    returns = rets.reshape(-1, 1)
    weights = np.array([1.0])
    nav = 1_000_000

    scale, annual_vol, info = ctrl.compute(weights, returns, nav)
    print_result("崩盘事件", scale, annual_vol, info, expected_range=(0.25, 0.7))

    pre_vol = np.std(rets[:200]) * np.sqrt(252)
    post_vol = np.std(rets[200:]) * np.sqrt(252)
    simple_vol = np.std(rets) * np.sqrt(252)
    print(f"\n  [验证] 崩盘前年化波动率 = {pre_vol:.4f} ({pre_vol*100:.2f}%)")
    print(f"  [验证] 崩盘后年化波动率 = {post_vol:.4f} ({post_vol*100:.2f}%)")
    print(f"  [验证] 简单历史年化波动率 = {simple_vol:.4f} ({simple_vol*100:.2f}%)")
    print(f"  [验证] GARCH年化波动率  = {annual_vol:.4f} ({annual_vol*100:.2f}%)")
    print(f"  [验证] GARCH > 简单波动率 = {'是 — GARCH捕获了崩盘后波动率飙升' if annual_vol > simple_vol else '否'}")
    print(f"  [验证] GARCH接近崩盘后波动率 = {'是' if abs(annual_vol - post_vol)/post_vol < 0.4 else '否'}")
    print(f"  [验证] 降仓效果 = {'有效' if scale < 0.7 else '不足'}")
    return scale, info


def test_8_ewma_fallback():
    """
    测试8: GARCH vs EWMA对比
    """
    print("\n" + "#"*70)
    print("  TEST SUITE 8: GARCH vs EWMA 波动率对比")
    print("#"*70)

    N, T = 20, 250
    returns = gen_correlated_returns(T, N, daily_vol=0.02, corr=0.6, seed=42)
    weights = np.ones(N) / N
    nav = 1_000_000

    # GARCH
    ctrl_g = GARCHPositionController(make_config(vol_method='garch'))
    scale_g, vol_g, info_g = ctrl_g.compute(weights, returns, nav)

    # EWMA
    ctrl_e = GARCHPositionController(make_config(vol_method='ewma'))
    scale_e, vol_e, info_e = ctrl_e.compute(weights, returns, nav)

    print(f"\n  GARCH: scale={scale_g:.4f}, vol={vol_g:.4f} ({vol_g*100:.2f}%), method={info_g['method']}, converged={info_g['converged']}")
    print(f"  EWMA:  scale={scale_e:.4f}, vol={vol_e:.4f} ({vol_e*100:.2f}%), method={info_e['method']}")
    vol_diff = abs(vol_g - vol_e) / max(vol_g, vol_e) * 100
    print(f"  波动率差异: {vol_diff:.1f}%")
    print(f"  [验证] GARCH收敛 = {'是' if info_g['converged'] else '否'}")
    print(f"  [验证] 两种方法降仓一致 = {'是' if abs(scale_g - scale_e) < 0.1 else '否'}")
    return scale_g, info_g, scale_e, info_e


def test_9_insufficient_data():
    """
    测试9: 数据不足
    """
    print("\n" + "#"*70)
    print("  TEST SUITE 9: 数据不足（<60天）")
    print("#"*70)

    cfg = make_config()
    ctrl = GARCHPositionController(cfg)

    N, T = 20, 30
    returns = gen_correlated_returns(T, N, daily_vol=0.012, corr=0.6, seed=42)
    weights = np.ones(N) / N
    nav = 1_000_000

    scale, annual_vol, info = ctrl.compute(weights, returns, nav)
    print_result("数据不足30天", scale, annual_vol, info, expected_range=(1.0, 1.0))

    print(f"\n  [验证] method = {info['method']} {'OK' if info['method'] == 'insufficient_data' else 'FAIL'}")
    print(f"  [验证] scale = 1.0 (全仓, 不控制) {'OK' if scale == 1.0 else 'FAIL'}")
    print(f"  [验证] 风险提示: 数据不足时默认全仓，存在风控盲区")
    return scale, info


def test_10_full_simulation():
    """
    测试10: 完整模拟 — 20个调仓周期的仓位变化轨迹
    """
    print("\n" + "#"*70)
    print("  TEST SUITE 10: 完整模拟 — 20个调仓周期仓位轨迹")
    print("  场景: 组合从峰值→逐步回撤15%→触底→恢复")
    print("#"*70)

    cfg = make_config()
    ctrl = GARCHPositionController(cfg)

    N, T = 20, 250
    returns = gen_correlated_returns(T, N, daily_vol=0.012, corr=0.6, seed=42)
    weights = np.ones(N) / N

    navs = np.concatenate([
        np.linspace(1_000_000, 850_000, 10),
        np.linspace(850_000, 1_000_000, 10),
    ])

    print(f"\n  {'周期':>4} | {'NAV':>10} | {'回撤%':>7} | {'vol_sc':>7} | {'dd_sc':>7} | {'final':>7} | {'冷却':>4} | {'方法':>8}")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*4}-+-{'-'*8}")

    results = []
    for i, nav in enumerate(navs):
        scale, vol, info = ctrl.compute(weights, returns, nav)
        dd = info['current_drawdown']
        results.append({
            'period': i, 'nav': nav, 'dd': dd,
            'vol_scale': info['vol_scale'], 'dd_scale': info['dd_scale'],
            'final': scale, 'cooldown': info['cooldown_active'],
        })
        print(f"  {i:>4} | {nav:>10,.0f} | {dd*100:>6.2f}% | {info['vol_scale']:>7.4f} | {info['dd_scale']:>7.4f} | {scale:>7.4f} | {'Y' if info['cooldown_active'] else 'N':>4} | {info['method']:>8}")

    print(f"\n  === 轨迹分析 ===")
    peak_scale = results[0]['final']
    trough_scale = min(r['final'] for r in results)
    trough_period = [r for r in results if r['final'] == trough_scale][0]['period']
    recovery_final = results[-1]['final']
    cooldown_periods = sum(1 for r in results if r['cooldown'])

    print(f"  峰值仓位      = {peak_scale:.4f}")
    print(f"  谷底仓位      = {trough_scale:.4f} (周期 {trough_period})")
    print(f"  恢复后最终仓位= {recovery_final:.4f}")
    print(f"  最大降仓幅度  = {(1 - trough_scale/peak_scale)*100:.1f}%")
    print(f"  冷却期总轮数   = {cooldown_periods}")
    print(f"  [验证] 回撤期降仓 = {'有效' if trough_scale < 0.5 else '不足'}")
    print(f"  [验证] 恢复期渐进 = {'有效' if recovery_final > trough_scale else '无效'}")
    print(f"  [验证] 冷却防反弹 = {'有效' if cooldown_periods >= 3 else '无效'}")

    # 关键观察: 冷却期内的仓位锁定行为
    print(f"\n  === 冷却期行为详细分析 ===")
    for r in results:
        if r['period'] >= 6 and r['period'] <= 18:
            phase = "回撤" if r['nav'] < 1_000_000 else "恢复"
            cd = "冷却中" if r['cooldown'] else "正常"
            print(f"  周期{r['period']:>2}: NAV={r['nav']:>10,.0f} dd={r['dd']*100:>5.1f}% "
                  f"dd_scale={r['dd_scale']:.4f} final={r['final']:.4f} [{phase}/{cd}]")

    return results


def test_11_drawdown_sensitivity():
    """
    测试11: 回撤-仓位映射灵敏度分析
    每次预设_nav_peak = 1_000_000
    """
    print("\n" + "#"*70)
    print("  TEST SUITE 11: 回撤-仓位映射灵敏度分析")
    print("  逐点验证 dd_scale = max(0.25, 1 - DD/0.20)")
    print("#"*70)

    cfg = make_config()
    N, T = 20, 250
    returns = gen_correlated_returns(T, N, daily_vol=0.008, corr=0.6, seed=42)
    weights = np.ones(N) / N
    nav_peak = 1_000_000

    drawdowns = [0.00, 0.02, 0.05, 0.08, 0.10, 0.11, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]

    print(f"\n  {'回撤%':>6} | {'NAV':>10} | {'理论dd_sc':>10} | {'实际dd_sc':>10} | {'final':>8} | {'冷却':>4} | {'匹配':>4}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*4}-+-{'-'*4}")

    all_match = True
    for dd in drawdowns:
        ctrl = GARCHPositionController(cfg)
        ctrl._nav_peak = nav_peak  # 预设峰值
        nav = nav_peak * (1 - dd)
        scale, vol, info = ctrl.compute(weights, returns, nav)
        theoretical = max(0.25, 1 - dd / 0.20)
        match = abs(info['dd_scale'] - theoretical) < 0.001
        if not match:
            all_match = False
        cooldown_expected = dd > 0.10  # 严格大于
        cd_match = info['cooldown_active'] == cooldown_expected
        print(f"  {dd*100:>5.1f}% | {nav:>10,.0f} | {theoretical:>10.4f} | {info['dd_scale']:>10.4f} | {scale:>8.4f} | {'Y' if info['cooldown_active'] else 'N':>4} | {'OK' if match and cd_match else 'XX'}")

    print(f"\n  [验证] 所有点匹配 = {'是' if all_match else '否'}")
    print(f"  [验证] 冷却触发阈值 = dd > 10% (严格大于, =10%不触发)")
    return all_match


def test_12_garch_vs_ewma_accuracy():
    """
    测试12: GARCH vs EWMA 波动率预测精度对比
    """
    print("\n" + "#"*70)
    print("  TEST SUITE 12: GARCH vs EWMA 波动率预测精度对比")
    print("#"*70)

    true_omega, true_alpha, true_beta = 0.00002, 0.08, 0.91
    true_uncond_var = true_omega / (1 - true_alpha - true_beta)
    true_uncond_vol = np.sqrt(true_uncond_var)

    rets = gen_garch_returns(n=250, omega=true_omega, alpha=true_alpha,
                              beta=true_beta, seed=42)

    # 计算真实最后一步条件波动率
    rng = np.random.RandomState(42)
    sigma2 = np.zeros(250)
    sigma2[0] = true_uncond_var
    r = np.zeros(250)
    for t in range(1, 250):
        z = rng.randn()
        r[t] = np.sqrt(sigma2[t-1]) * z
        sigma2[t] = true_omega + true_alpha * r[t-1]**2 + true_beta * sigma2[t-1]
    true_cond_vol = np.sqrt(sigma2[-1])
    true_annual_vol = true_cond_vol * np.sqrt(252)

    # GARCH拟合
    ctrl_g = GARCHPositionController(make_config(vol_method='garch'))
    weights = np.array([1.0])
    returns_2d = rets.reshape(-1, 1)
    scale_g, vol_g, info_g = ctrl_g.compute(weights, returns_2d, 1_000_000)

    # EWMA
    ctrl_e = GARCHPositionController(make_config(vol_method='ewma'))
    scale_e, vol_e, info_e = ctrl_e.compute(weights, returns_2d, 1_000_000)

    print(f"\n  真实 GARCH 参数: w={true_omega}, a={true_alpha}, b={true_beta}")
    print(f"  无条件日波动率 = {true_uncond_vol:.6f} ({true_uncond_vol*100:.4f}%)")
    print(f"  最后一步真实条件波动率 = {true_cond_vol:.6f} ({true_cond_vol*100:.4f}%)")
    print(f"  真实年化条件波动率     = {true_annual_vol:.4f} ({true_annual_vol*100:.2f}%)")
    print(f"\n  GARCH 估计年化波动率   = {vol_g:.4f} ({vol_g*100:.2f}%) [method={info_g['method']}, converged={info_g['converged']}]")
    print(f"  EWMA  估计年化波动率   = {vol_e:.4f} ({vol_e*100:.2f}%) [method={info_e['method']}]")
    garch_err = abs(vol_g - true_annual_vol)/true_annual_vol*100
    ewma_err = abs(vol_e - true_annual_vol)/true_annual_vol*100
    print(f"\n  GARCH 误差 = {garch_err:.2f}%")
    print(f"  EWMA  误差 = {ewma_err:.2f}%")
    print(f"  GARCH 更准确 = {'是' if garch_err < ewma_err else '否 — EWMA在此场景更准确'}")
    print(f"  说明: EWMA对近期高波动更敏感(lambda=0.94), 而GARCH的MLE可能平滑了尾部")

    return vol_g, vol_e, true_annual_vol


def test_13_combined_vol_and_drawdown():
    """
    测试13: 波动率+回撤双重约束
    经济学场景: 2015年股灾 — 高波动(47%年化) + 严重回撤(18%)
    预期: vol_scale和dd_scale都降低，取min
    """
    print("\n" + "#"*70)
    print("  TEST SUITE 13: 波动率+回撤双重约束")
    print("  场景: 2015股灾 — 高波动(47%年化) + 回撤18%")
    print("#"*70)

    cfg = make_config()
    ctrl = GARCHPositionController(cfg)

    N, T = 20, 250
    returns = gen_correlated_returns(T, N, daily_vol=0.03, corr=0.8, seed=42)
    weights = np.ones(N) / N
    nav = 984_000  # 18%回撤
    ctrl._nav_peak = 1_200_000

    scale, annual_vol, info = ctrl.compute(weights, returns, nav)
    print_result("波动率+回撤双重约束", scale, annual_vol, info, expected_range=(0.10, 0.25))

    expected_vol_scale = min(1.0, max(0.25, 0.20 / annual_vol))
    expected_dd_scale = max(0.25, 1 - 0.18 / 0.20)
    print(f"\n  [验证] 年化波动率 = {annual_vol:.4f} ({annual_vol*100:.2f}%)")
    print(f"  [验证] vol_scale = clip(0.20/{annual_vol:.4f}, 0.25, 1.0) = {expected_vol_scale:.4f}")
    print(f"  [验证] 实际vol_scale = {info['vol_scale']:.4f}")
    print(f"  [验证] dd = 18%")
    print(f"  [验证] dd_scale = max(0.25, 1-0.18/0.20) = {expected_dd_scale:.4f}")
    print(f"  [验证] 实际dd_scale = {info['dd_scale']:.4f}")
    print(f"  [验证] final = min(vol_scale, dd_scale) = min({expected_vol_scale:.4f}, {expected_dd_scale:.4f}) = {min(expected_vol_scale, expected_dd_scale):.4f}")
    print(f"  [验证] 实际final = {scale:.4f}")
    print(f"  [验证] 双重约束更严格 = {'是' if scale < min(expected_vol_scale, expected_dd_scale) + 0.01 else '否'}")
    return scale, info


def test_14_extreme_vol_clustering_with_drawdown():
    """
    测试14: 极端场景 — 波动率聚集 + 回撤叠加
    经济学场景: 2020年3月疫情冲击 — 市场突然进入高波动 + 组合回撤
    """
    print("\n" + "#"*70)
    print("  TEST SUITE 14: 极端场景 — 波动率聚集+回撤叠加")
    print("  场景: 2020年3月疫情 — 低波动转高波动 + 15%回撤")
    print("#"*70)

    cfg = make_config()
    ctrl = GARCHPositionController(cfg)

    T = 250
    rets = gen_volatility_clustering(T, seed=42)
    returns = rets.reshape(-1, 1)
    weights = np.array([1.0])
    nav = 850_000  # 15%回撤
    ctrl._nav_peak = 1_000_000

    scale, annual_vol, info = ctrl.compute(weights, returns, nav)
    print_result("波动率聚集+回撤15%", scale, annual_vol, info, expected_range=(0.10, 0.30))

    pre_vol = np.std(rets[:T//2]) * np.sqrt(252)
    post_vol = np.std(rets[T//2:]) * np.sqrt(252)
    print(f"\n  [验证] 前半段年化波动率 = {pre_vol:.4f} ({pre_vol*100:.2f}%)")
    print(f"  [验证] 后半段年化波动率 = {post_vol:.4f} ({post_vol*100:.2f}%)")
    print(f"  [验证] GARCH年化波动率 = {annual_vol:.4f} ({annual_vol*100:.2f}%)")
    print(f"  [验证] vol_scale = clip(0.20/{annual_vol:.4f}, 0.25, 1.0) = {info['vol_scale']:.4f}")
    print(f"  [验证] dd = 15%")
    print(f"  [验证] dd_scale = max(0.25, 1-0.15/0.20) = {info['dd_scale']:.4f}")
    print(f"  [验证] final = min(vol_scale, dd_scale) = {scale:.4f}")
    print(f"  [验证] 双重约束生效 = {'是' if scale <= 0.30 else '否'}")
    return scale, info


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("  GARCH 仓位控制 & 回撤叠加保护 — 功能测试套件 v2")
    print("  Alpha191 Strategy Risk Control Verification")
    print("=" * 70)

    results = {}
    results['t1'] = test_1_normal_low_volatility()
    results['t2'] = test_2_high_volatility()
    results['t3'] = test_3_moderate_drawdown()
    results['t4'] = test_4_severe_drawdown()
    results['t5'] = test_5_cooldown_protection()
    results['t6'] = test_6_volatility_clustering()
    results['t7'] = test_7_crash_event()
    results['t8'] = test_8_ewma_fallback()
    results['t9'] = test_9_insufficient_data()
    results['t10'] = test_10_full_simulation()
    results['t11'] = test_11_drawdown_sensitivity()
    results['t12'] = test_12_garch_vs_ewma_accuracy()
    results['t13'] = test_13_combined_vol_and_drawdown()
    results['t14'] = test_14_extreme_vol_clustering_with_drawdown()

    print("\n\n" + "=" * 70)
    print("  测试总结")
    print("=" * 70)
    print(f"  共执行 14 个测试套件")
    print(f"  所有测试完成")

    return results


if __name__ == '__main__':
    main()
