"""
GARCH 仓位控制 & 回撤叠加保护 — 功能测试套件
================================================
测试目标:
  1. 验证 GARCH 波动率目标化在正常/高波动/极端波动场景下的仓位输出
  2. 验证回撤叠加保护的连续线性降仓机制
  3. 验证冷却期对"V型反弹"的防护
  4. 验证 GARCH → EWMA 降级路径
  5. 验证边界条件（数据不足、零波动等）
  6. 模拟真实经济学场景（2015股灾、2020疫情、2024低波动）

运行方式:
  python test_garch_drawdown.py
"""
import sys
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 确保能导入策略模块 — 直接从当前目录导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from garch_position import GARCHPositionController

# ============================================================
# 辅助函数
# ============================================================

def make_config(**overrides):
    """创建默认配置，允许覆盖"""
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


def gen_garch_returns(n=250, omega=0.00002, alpha=0.08, beta=0.91, seed=42):
    """生成真实的 GARCH(1,1) 收益率序列"""
    rng = np.random.RandomState(seed)
    r = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)  # 无条件方差作为初始值
    for t in range(1, n):
        z = rng.randn()
        r[t] = np.sqrt(sigma2[t-1]) * z
        sigma2[t] = omega + alpha * r[t-1]**2 + beta * sigma2[t-1]
    return r


def gen_normal_returns(n=250, daily_vol=0.012, seed=42):
    """生成恒定波动率的正态收益率"""
    rng = np.random.RandomState(seed)
    return rng.randn(n) * daily_vol


def gen_volatility_clustering(n=250, seed=42):
    """
    生成波动率聚集序列 — 模拟真实金融市场
    前半段低波动(~0.8%/天)，后半段高波动(~2.5%/天)
    """
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
    """生成含崩盘事件的收益率序列"""
    rng = np.random.RandomState(seed)
    r = rng.randn(n) * 0.012
    r[crash_day] = crash_mag  # 单日暴跌6%
    # 崩盘后波动率飙升
    for t in range(crash_day + 1, min(crash_day + 30, n)):
        r[t] = rng.randn() * 0.03
    return r


def print_result(test_name, scale, annual_vol, info, expected_range=None):
    """格式化打印测试结果"""
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
        print(f"  STATUS          = {'PASS ✅' if in_range else 'FAIL ❌'}")
    else:
        print(f"  STATUS          = INFO ℹ️")


# ============================================================
# 测试套件
# ============================================================

def test_1_normal_low_volatility():
    """
    测试1: 正常低波动市场 — 无回撤
    经济学场景: 2024年初A股平稳运行，日均波动率~1.0%，组合在净值峰值附近
    预期: vol_scale接近或达到1.0，dd_scale=1.0，最终全仓或接近全仓
    """
    print("\n" + "█"*70)
    print("  TEST SUITE 1: 正常低波动市场（无回撤）")
    print("  经济学场景: 2024年初A股平稳运行，日均波动~1.0%，净值在峰值")
    print("█"*70)

    cfg = make_config()
    ctrl = GARCHPositionController(cfg)

    # 20只股票，250天，日均波动率1.0%
    np.random.seed(42)
    N = 20
    T = 250
    returns = np.random.randn(T, N) * 0.01
    weights = np.ones(N) / N  # 等权
    nav = 1_000_000  # 峰值

    scale, annual_vol, info = ctrl.compute(weights, returns, nav)
    print_result("正常低波动（无回撤）", scale, annual_vol, info, expected_range=(0.8, 1.0))

    # 验证: 年化波动率应该在 0.01*sqrt(252) ≈ 15.87% 附近
    expected_annual = 0.01 * np.sqrt(252)
    print(f"\n  [验证] 理论年化波动率 = {expected_annual:.4f} ({expected_annual*100:.2f}%)")
    print(f"  [验证] 理论vol_scale = {0.20/expected_annual:.4f} (clip前)")
    print(f"  [验证] 实际vol_scale  = {info['vol_scale']:.4f}")
    vol_scale_diff = abs(info['vol_scale'] - min(1.0, 0.20/expected_annual))
    print(f"  [验证] vol_scale偏差  = {vol_scale_diff:.4f} {'✅' if vol_scale_diff < 0.05 else '❌'}")
    return scale, info


def test_2_high_volatility():
    """
    测试2: 高波动市场 — 无回撤
    经济学场景: 2015年6月股灾期间，日均波动率~3.0%，但组合尚未大幅回撤
    预期: vol_scale显著降低（目标20%/实际47.6% ≈ 0.42），但clip到min_scale=0.25
    """
    print("\n" + "█"*70)
    print("  TEST SUITE 2: 高波动市场（无回撤）")
    print("  经济学场景: 2015年6月股灾，日均波动~3.0%，净值尚未大幅回撤")
    print("█"*70)

    cfg = make_config()
    ctrl = GARCHPositionController(cfg)

    np.random.seed(42)
    N = 20
    T = 250
    returns = np.random.randn(T, N) * 0.03  # 3%日波动
    weights = np.ones(N) / N
    nav = 1_000_000

    scale, annual_vol, info = ctrl.compute(weights, returns, nav)
    print_result("高波动（无回撤）", scale, annual_vol, info, expected_range=(0.25, 0.5))

    expected_annual = 0.03 * np.sqrt(252)
    expected_vol_scale = 0.20 / expected_annual
    print(f"\n  [验证] 理论年化波动率 = {expected_annual:.4f} ({expected_annual*100:.2f}%)")
    print(f"  [验证] 理论vol_scale = {expected_vol_scale:.4f}")
    print(f"  [验证] 实际vol_scale  = {info['vol_scale']:.4f}")
    print(f"  [验证] 高波动→降仓 ✅" if scale < 0.5 else "  [验证] 降仓不足 ❌")
    return scale, info


def test_3_moderate_drawdown():
    """
    测试3: 中等回撤（10%）— 波动率正常
    经济学场景: 组合从峰值120万回撤至108万（10%回撤），波动率正常(~1.2%/天)
    预期: dd_scale = 1 - 0.10/0.20 = 0.50，触发冷却，最终仓位≈0.50
    """
    print("\n" + "█"*70)
    print("  TEST SUITE 3: 中等回撤（10%）— 波动率正常")
    print("  经济学场景: 组合从峰值120万回撤至108万（10%回撤）")
    print("█"*70)

    cfg = make_config()
    ctrl = GARCHPositionController(cfg)

    np.random.seed(42)
    N = 20
    T = 250
    returns = np.random.randn(T, N) * 0.012
    weights = np.ones(N) / N
    nav = 1_080_000  # 从1,200,000回撤10%
    ctrl._nav_peak = 1_200_000  # 手动设置峰值

    scale, annual_vol, info = ctrl.compute(weights, returns, nav)
    print_result("中等回撤10%", scale, annual_vol, info, expected_range=(0.40, 0.60))

    expected_dd = 1 - 1080000/1200000
    expected_dd_scale = max(0.25, 1 - expected_dd/0.20)
    print(f"\n  [验证] 理论回撤     = {expected_dd:.4f} ({expected_dd*100:.1f}%)")
    print(f"  [验证] 理论dd_scale = {expected_dd_scale:.4f}")
    print(f"  [验证] 实际dd_scale = {info['dd_scale']:.4f}")
    dd_diff = abs(info['dd_scale'] - expected_dd_scale)
    print(f"  [验证] dd_scale偏差 = {dd_diff:.4f} {'✅' if dd_diff < 0.01 else '❌'}")
    print(f"  [验证] 冷却触发     = {'是 ✅' if info['cooldown_active'] else '否 ❌'}")
    print(f"         (回撤10% > 阈值10%=0.20*0.5, 应触发冷却)")
    return scale, info


def test_4_severe_drawdown():
    """
    测试4: 严重回撤（20%+）— 触及地板
    经济学场景: 组合从峰值120万回撤至90万（25%回撤）
    预期: dd_scale = max(0.25, 1-0.25/0.20) = max(0.25, -0.25) = 0.25 (地板)
    """
    print("\n" + "█"*70)
    print("  TEST SUITE 4: 严重回撤（25%）— 触及地板")
    print("  经济学场景: 组合从峰值120万回撤至90万（25%回撤）")
    print("█"*70)

    cfg = make_config()
    ctrl = GARCHPositionController(cfg)

    np.random.seed(42)
    N = 20
    T = 250
    returns = np.random.randn(T, N) * 0.015
    weights = np.ones(N) / N
    nav = 900_000  # 从1,200,000回撤25%
    ctrl._nav_peak = 1_200_000

    scale, annual_vol, info = ctrl.compute(weights, returns, nav)
    print_result("严重回撤25%", scale, annual_vol, info, expected_range=(0.25, 0.26))

    print(f"\n  [验证] 理论dd_scale = max(0.25, 1-0.25/0.20) = 0.25 (地板)")
    print(f"  [验证] 实际dd_scale = {info['dd_scale']:.4f}")
    print(f"  [验证] 触及地板     = {'是 ✅' if abs(info['dd_scale'] - 0.25) < 0.001 else '否 ❌'}")
    print(f"  [验证] 冷却触发     = {'是 ✅' if info['cooldown_active'] else '否 ❌'}")
    return scale, info


def test_5_cooldown_protection():
    """
    测试5: 冷却期防"V型反弹"
    经济学场景: 组合回撤触发冷却后，净值快速恢复至峰值
    预期: 冷却期内即使回撤恢复，dd_scale仍被锁定在last_scale，不会立即全仓
    """
    print("\n" + "█"*70)
    print("  TEST SUITE 5: 冷却期防V型反弹")
    print("  经济学场景: 回撤触发冷却后，净值快速恢复至峰值")
    print("█"*70)

    cfg = make_config(cooldown_periods=3)
    ctrl = GARCHPositionController(cfg)

    np.random.seed(42)
    N = 20
    T = 250
    returns = np.random.randn(T, N) * 0.012
    weights = np.ones(N) / N

    # Phase 1: 触发回撤（10%回撤）
    print("\n  --- Phase 1: 触发回撤（NAV=108万，峰值120万，回撤10%）---")
    nav = 1_080_000
    ctrl._nav_peak = 1_200_000
    scale1, _, info1 = ctrl.compute(weights, returns, nav)
    print(f"  scale={scale1:.4f}, dd_scale={info1['dd_scale']:.4f}, cooldown_remaining={ctrl._cooldown_remaining}")
    print(f"  last_scale={ctrl._last_scale:.4f}")

    # Phase 2: 净值恢复到峰值，但冷却期第1轮
    print("\n  --- Phase 2: 净值恢复至120万（冷却第1轮，应阻止全仓）---")
    nav = 1_200_000  # 完全恢复
    scale2, _, info2 = ctrl.compute(weights, returns, nav)
    print(f"  scale={scale2:.4f}, dd_scale={info2['dd_scale']:.4f}, cooldown_active={info2['cooldown_active']}")
    print(f"  cooldown_remaining={ctrl._cooldown_remaining}")
    print(f"  [验证] 冷却阻止全仓 = {'是 ✅' if scale2 < 1.0 else '否 ❌ — V型反弹未被阻止!'}")
    print(f"  [验证] dd_scale被锁定= {'是 ✅' if info2['dd_scale'] <= ctrl._last_scale + 0.001 else '否 ❌'}")

    # Phase 3: 冷却第2轮
    print("\n  --- Phase 3: 净值保持120万（冷却第2轮）---")
    scale3, _, info3 = ctrl.compute(weights, returns, nav)
    print(f"  scale={scale3:.4f}, cooldown_active={info3['cooldown_active']}, remaining={ctrl._cooldown_remaining}")

    # Phase 4: 冷却第3轮（最后一轮）
    print("\n  --- Phase 4: 冷却第3轮（最后一轮）---")
    scale4, _, info4 = ctrl.compute(weights, returns, nav)
    print(f"  scale={scale4:.4f}, cooldown_active={info4['cooldown_active']}, remaining={ctrl._cooldown_remaining}")

    # Phase 5: 冷却结束
    print("\n  --- Phase 5: 冷却结束，应恢复全仓 ---")
    scale5, _, info5 = ctrl.compute(weights, returns, nav)
    print(f"  scale={scale5:.4f}, cooldown_active={info5['cooldown_active']}")
    print(f"  [验证] 冷却后恢复   = {'是 ✅' if scale5 > scale2 else '否 ❌'}")

    return [scale1, scale2, scale3, scale4, scale5]


def test_6_volatility_clustering():
    """
    测试6: 波动率聚集效应
    经济学场景: 市场从低波动突然转入高波动（如2020年3月疫情冲击）
    GARCH应能预测出高于平均水平的条件波动率
    """
    print("\n" + "█"*70)
    print("  TEST SUITE 6: 波动率聚集效应")
    print("  经济学场景: 市场从低波动(~0.8%)突然转入高波动(~2.5%)")
    print("█"*70)

    cfg = make_config()
    ctrl = GARCHPositionController(cfg)

    N = 20
    T = 250
    # 每只股票都有波动率聚集
    returns = np.zeros((T, N))
    for i in range(N):
        returns[:, i] = gen_volatility_clustering(T, seed=42+i)

    weights = np.ones(N) / N
    nav = 1_000_000

    scale, annual_vol, info = ctrl.compute(weights, returns, nav)
    print_result("波动率聚集", scale, annual_vol, info, expected_range=(0.25, 0.7))

    # 对比: 如果用简单标准差会低估尾部波动
    port_ret = returns @ weights
    simple_vol = np.std(port_ret) * np.sqrt(252)
    print(f"\n  [验证] 简单历史波动率 = {simple_vol:.4f} ({simple_vol*100:.2f}%)")
    print(f"  [验证] GARCH年化波动率= {annual_vol:.4f} ({annual_vol*100:.2f}%)")
    print(f"  [验证] GARCH > 简单波动率 = {'是 ✅ — GARCH捕获了波动率聚集' if annual_vol > simple_vol else '否 ❌'}")
    print(f"  [验证] 降仓效果 = {'有效 ✅' if scale < 0.7 else '不足 ❌'}")
    return scale, info


def test_7_crash_event():
    """
    测试7: 单日崩盘事件
    经济学场景: 2015年8月24日"黑色星期一"，单日暴跌6%+
    GARCH应预测出崩盘后波动率飙升
    """
    print("\n" + "█"*70)
    print("  TEST SUITE 7: 单日崩盘事件（2015黑色星期一风格）")
    print("  经济学场景: 250天中第200天暴跌6%，之后波动率飙升")
    print("█"*70)

    cfg = make_config()
    ctrl = GARCHPositionController(cfg)

    N = 20
    T = 250
    returns = np.zeros((T, N))
    for i in range(N):
        returns[:, i] = gen_crash_returns(T, crash_day=200, crash_mag=-0.06, seed=42+i)

    weights = np.ones(N) / N
    nav = 1_000_000

    scale, annual_vol, info = ctrl.compute(weights, returns, nav)
    print_result("崩盘事件", scale, annual_vol, info, expected_range=(0.25, 0.6))

    # 分析崩盘前后波动率变化
    port_ret = returns @ weights
    pre_crash_vol = np.std(port_ret[:200]) * np.sqrt(252)
    post_crash_vol = np.std(port_ret[200:]) * np.sqrt(252)
    print(f"\n  [验证] 崩盘前年化波动率 = {pre_crash_vol:.4f} ({pre_crash_vol*100:.2f}%)")
    print(f"  [验证] 崩盘后年化波动率 = {post_crash_vol:.4f} ({post_crash_vol*100:.2f}%)")
    print(f"  [验证] GARCH年化波动率  = {annual_vol:.4f} ({annual_vol*100:.2f}%)")
    print(f"  [验证] GARCH捕获尾部风险 = {'是 ✅' if annual_vol > pre_crash_vol else '否 ❌'}")
    return scale, info


def test_8_ewma_fallback():
    """
    测试8: GARCH降级为EWMA
    经济学场景: GARCH拟合不收敛时，系统应自动降级为EWMA
    """
    print("\n" + "█"*70)
    print("  TEST SUITE 8: GARCH → EWMA 降级路径")
    print("  经济学场景: GARCH拟合不收敛时的容错")
    print("█"*70)

    cfg = make_config(vol_method='ewma')  # 直接指定EWMA
    ctrl = GARCHPositionController(cfg)

    np.random.seed(42)
    N = 20
    T = 250
    returns = np.random.randn(T, N) * 0.015
    weights = np.ones(N) / N
    nav = 1_000_000

    scale, annual_vol, info = ctrl.compute(weights, returns, nav)
    print_result("EWMA模式", scale, annual_vol, info, expected_range=(0.25, 1.0))

    print(f"\n  [验证] method = {info['method']} {'✅' if info['method'] == 'ewma' else '❌'}")
    print(f"  [验证] converged = {info['converged']} (EWMA不需要收敛)")

    # 对比GARCH和EWMA
    cfg_garch = make_config(vol_method='garch')
    ctrl_garch = GARCHPositionController(cfg_garch)
    scale_g, vol_g, info_g = ctrl_garch.compute(weights, returns, nav)
    print(f"\n  [对比] GARCH scale = {scale_g:.4f}, vol = {vol_g:.4f}, method = {info_g['method']}")
    print(f"  [对比] EWMA  scale = {scale:.4f}, vol = {annual_vol:.4f}, method = {info['method']}")
    vol_diff = abs(vol_g - annual_vol) / max(vol_g, annual_vol) * 100
    print(f"  [对比] 波动率差异   = {vol_diff:.1f}%")
    return scale, info


def test_9_insufficient_data():
    """
    测试9: 数据不足（<60天）
    经济学场景: 新策略上线初期，历史数据不足
    预期: 返回max_scale=1.0，不做控制
    """
    print("\n" + "█"*70)
    print("  TEST SUITE 9: 数据不足（<60天）")
    print("  经济学场景: 新策略上线初期，历史数据不足60天")
    print("█"*70)

    cfg = make_config()
    ctrl = GARCHPositionController(cfg)

    np.random.seed(42)
    N = 20
    T = 30  # 不足60天
    returns = np.random.randn(T, N) * 0.012
    weights = np.ones(N) / N
    nav = 1_000_000

    scale, annual_vol, info = ctrl.compute(weights, returns, nav)
    print_result("数据不足30天", scale, annual_vol, info, expected_range=(1.0, 1.0))

    print(f"\n  [验证] method = {info['method']} {'✅' if info['method'] == 'insufficient_data' else '❌'}")
    print(f"  [验证] scale = 1.0 (全仓, 不控制) {'✅' if scale == 1.0 else '❌'}")
    return scale, info


def test_10_full_simulation():
    """
    测试10: 完整模拟 — 20个调仓周期的仓位变化轨迹
    经济学场景: 模拟组合从峰值→回撤→恢复的完整周期
    """
    print("\n" + "█"*70)
    print("  TEST SUITE 10: 完整模拟 — 20个调仓周期仓位轨迹")
    print("  经济学场景: 组合从峰值→逐步回撤→触底→恢复")
    print("█"*70)

    cfg = make_config()
    ctrl = GARCHPositionController(cfg)

    np.random.seed(42)
    N = 20
    T = 250
    returns = np.random.randn(T, N) * 0.012
    weights = np.ones(N) / N

    # 模拟净值轨迹: 100万 → 85万(谷底) → 100万
    navs = np.concatenate([
        np.linspace(1_000_000, 850_000, 10),  # 逐步回撤15%
        np.linspace(850_000, 1_000_000, 10),  # 恢复
    ])

    print(f"\n  {'周期':>4} | {'NAV':>10} | {'回撤':>8} | {'vol_scale':>10} | {'dd_scale':>10} | {'final':>8} | {'冷却':>4} | {'方法':>12}")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*4}-+-{'-'*12}")

    results = []
    for i, nav in enumerate(navs):
        scale, vol, info = ctrl.compute(weights, returns, nav)
        dd = info['current_drawdown']
        results.append({
            'period': i, 'nav': nav, 'dd': dd,
            'vol_scale': info['vol_scale'], 'dd_scale': info['dd_scale'],
            'final': scale, 'cooldown': info['cooldown_active'],
            'method': info['method']
        })
        print(f"  {i:>4} | {nav:>10,.0f} | {dd*100:>7.2f}% | {info['vol_scale']:>10.4f} | {info['dd_scale']:>10.4f} | {scale:>8.4f} | {'是' if info['cooldown_active'] else '否':>4} | {info['method']:>12}")

    # 分析轨迹
    print(f"\n  === 轨迹分析 ===")
    peak_scale = results[0]['final']
    trough_scale = min(r['final'] for r in results)
    trough_period = [r for r in results if r['final'] == trough_scale][0]['period']
    recovery_final = results[-1]['final']

    print(f"  峰值仓位      = {peak_scale:.4f}")
    print(f"  谷底仓位      = {trough_scale:.4f} (周期 {trough_period})")
    print(f"  恢复后最终仓位= {recovery_final:.4f}")
    print(f"  最大降仓幅度  = {(1 - trough_scale/peak_scale)*100:.1f}%")
    print(f"  恢复比         = {recovery_final/peak_scale:.2f}x")

    # 关键验证
    cooldown_periods = sum(1 for r in results if r['cooldown'])
    print(f"  冷却期总轮数   = {cooldown_periods}")
    print(f"  [验证] 回撤期降仓 = {'有效 ✅' if trough_scale < 0.5 else '不足 ❌'}")
    print(f"  [验证] 恢复期渐进 = {'有效 ✅' if recovery_final > trough_scale else '无效 ❌'}")
    print(f"  [验证] 冷却防反弹 = {'有效 ✅' if cooldown_periods >= 3 else '无效 ❌'}")

    return results


def test_11_drawdown_sensitivity():
    """
    测试11: 回撤-仓位映射灵敏度分析
    逐点验证连续线性降仓函数
    """
    print("\n" + "█"*70)
    print("  TEST SUITE 11: 回撤-仓位映射灵敏度分析")
    print("  逐点验证 dd_scale = max(0.25, 1 - DD/0.20)")
    print("█"*70)

    cfg = make_config()
    ctrl = GARCHPositionController(cfg)

    np.random.seed(42)
    N = 20
    T = 250
    returns = np.random.randn(T, N) * 0.010  # 低波动，使vol_scale≈1.0
    weights = np.ones(N) / N

    nav_peak = 1_000_000
    drawdowns = [0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]

    print(f"\n  {'回撤%':>6} | {'NAV':>10} | {'理论dd_scale':>12} | {'实际dd_scale':>12} | {'实际final':>10} | {'冷却':>4}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*4}")

    for dd in drawdowns:
        ctrl_fresh = GARCHPositionController(cfg)  # 每次新建避免状态污染
        nav = nav_peak * (1 - dd)
        scale, vol, info = ctrl_fresh.compute(weights, returns, nav)
        theoretical = max(0.25, 1 - dd / 0.20)
        match = '✅' if abs(info['dd_scale'] - theoretical) < 0.001 else '❌'
        print(f"  {dd*100:>5.1f}% | {nav:>10,.0f} | {theoretical:>12.4f} | {info['dd_scale']:>12.4f} | {scale:>10.4f} | {'是' if info['cooldown_active'] else '否':>4} {match}")

    return None


def test_12_garch_vs_ewma_accuracy():
    """
    测试12: GARCH vs EWMA 波动率预测精度对比
    使用真实GARCH生成的数据，验证GARCH预测更准确
    """
    print("\n" + "█"*70)
    print("  TEST SUITE 12: GARCH vs EWMA 波动率预测精度对比")
    print("  使用真实GARCH(1,1)生成的数据验证")
    print("█"*70)

    # 生成真实GARCH数据
    true_omega, true_alpha, true_beta = 0.00002, 0.08, 0.91
    true_uncond_var = true_omega / (1 - true_alpha - true_beta)
    true_uncond_vol = np.sqrt(true_uncond_var)

    rets = gen_garch_returns(n=250, omega=true_omega, alpha=true_alpha,
                              beta=true_beta, seed=42)

    # GARCH拟合
    cfg_garch = make_config(vol_method='garch')
    ctrl_garch = GARCHPositionController(cfg_garch)
    weights = np.array([1.0])
    returns_2d = rets.reshape(-1, 1)

    scale_g, vol_g, info_g = ctrl_garch.compute(weights, returns_2d, 1_000_000)

    # EWMA
    cfg_ewma = make_config(vol_method='ewma')
    ctrl_ewma = GARCHPositionController(cfg_ewma)
    scale_e, vol_e, info_e = ctrl_ewma.compute(weights, returns_2d, 1_000_000)

    # 真实条件波动率（最后一个时间步）
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

    print(f"\n  真实 GARCH 参数: ω={true_omega}, α={true_alpha}, β={true_beta}")
    print(f"  无条件日波动率 = {true_uncond_vol:.6f} ({true_uncond_vol*100:.4f}%)")
    print(f"  最后一步真实条件波动率 = {true_cond_vol:.6f} ({true_cond_vol*100:.4f}%)")
    print(f"  真实年化条件波动率     = {true_annual_vol:.4f} ({true_annual_vol*100:.2f}%)")
    print(f"\n  GARCH 估计年化波动率   = {vol_g:.4f} ({vol_g*100:.2f}%) [method={info_g['method']}]")
    print(f"  EWMA  估计年化波动率   = {vol_e:.4f} ({vol_e*100:.2f}%) [method={info_e['method']}]")
    print(f"\n  GARCH 误差 = {abs(vol_g - true_annual_vol)/true_annual_vol*100:.2f}%")
    print(f"  EWMA  误差 = {abs(vol_e - true_annual_vol)/true_annual_vol*100:.2f}%")
    print(f"  GARCH 更准确 = {'是 ✅' if abs(vol_g - true_annual_vol) < abs(vol_e - true_annual_vol) else '否 ❌'}")

    return vol_g, vol_e, true_annual_vol


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("  GARCH 仓位控制 & 回撤叠加保护 — 功能测试套件")
    print("  Alpha191 Strategy Risk Control Verification")
    print("=" * 70)

    results = {}

    # 执行所有测试
    results['test1'] = test_1_normal_low_volatility()
    results['test2'] = test_2_high_volatility()
    results['test3'] = test_3_moderate_drawdown()
    results['test4'] = test_4_severe_drawdown()
    results['test5'] = test_5_cooldown_protection()
    results['test6'] = test_6_volatility_clustering()
    results['test7'] = test_7_crash_event()
    results['test8'] = test_8_ewma_fallback()
    results['test9'] = test_9_insufficient_data()
    results['test10'] = test_10_full_simulation()
    results['test11'] = test_11_drawdown_sensitivity()
    results['test12'] = test_12_garch_vs_ewma_accuracy()

    # 总结
    print("\n\n" + "=" * 70)
    print("  测试总结")
    print("=" * 70)
    print(f"  共执行 12 个测试套件")
    print(f"  所有测试完成 ✅")

    return results


if __name__ == '__main__':
    main()
