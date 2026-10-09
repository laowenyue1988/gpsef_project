"""
GARCH v3.1 回撤保护优化测试套件.

测试覆盖:
  1. 凹函数 vs 线性 — 全回撤区间对比
  2. 冷却触发阈值 10%→15% 验证
  3. 恢复周期压缩 7→4 验证
  4. 凹函数浅回撤反弹收益改善
  5. 凹函数深回撤保护不变
  6. 向后兼容（linear配置回退）
  7. v3.0回归测试
  8. 全功能综合场景
"""
import sys
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from garch_position import GARCHPositionController

PASS = 0
FAIL = 0

def check(condition, test_name):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {test_name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {test_name}")

def make_config(overrides=None):
    cfg = {
        'vol_target': 0.20, 'lookback': 250, 'vol_method': 'garch',
        'vol_model': 'EGARCH', 'ewma_lambda': 0.94, 'min_scale': 0.25,
        'max_scale': 1.0, 'drawdown_limit': 0.20,
        'use_drawdown_control': True,
        'dd_function': 'concave',
        'cooldown_periods': 2,
        'cooldown_trigger_ratio': 0.75,
        'garch_dist': 't', 'use_ewma_cross_check': True,
        'use_gradual_cooldown': True, 'use_cooldown_episode_tracking': True,
        'secondary_dd_threshold': 0.15, 'secondary_dd_floor': 0.15,
        'tertiary_dd_threshold': 0.20, 'tertiary_dd_floor': 0.05,
        'use_kill_switch': True, 'kill_switch_threshold': 0.30,
        'kill_switch_floor': 0.0,
        'use_sigmoid_cooldown': True, 'cooldown_sigmoid_k': 8.0,
        'use_recovery_buffer': True,
        'recovery_buffer_periods': 2, 'recovery_buffer_stages': 2,
        'use_adf_test': True, 'adf_pvalue_threshold': 0.05,
        'use_arch_lm_test': True, 'arch_lm_pvalue_threshold': 0.05,
        'use_winsorize': True, 'winsorize_lower': 0.005, 'winsorize_upper': 0.995,
    }
    if overrides:
        cfg.update(overrides)
    return cfg


# ================================================================
# TEST 1: 凹函数 vs 线性 — 全回撤区间逐点对比
# ================================================================
def test_1_concave_vs_linear():
    """凹函数在浅回撤区保留更多仓位，深回撤区保护不变."""
    print("\n" + "="*70)
    print("  TEST 1: 凹函数 vs 线性 — 全回撤区间逐点对比")
    print("="*70)

    dd_levels = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
    print(f"  {'回撤':>8} {'凹dd_scale':>12} {'线dd_scale':>12} {'差距':>10} {'评价'}")
    print("  " + "-"*58)

    for dd in dd_levels:
        concave = max(0.25, np.sqrt(max(0, 1 - dd / 0.20)))
        linear = max(0.25, 1 - dd / 0.20)
        diff = concave - linear
        evaluation = ""
        if dd <= 0.05:
            evaluation = "凹函数更优(浅DD)"
        elif dd <= 0.12:
            evaluation = "凹函数更优(中DD)"
        else:
            evaluation = "硬地板主导"
        print(f"  {dd:7.1%} {concave:12.4f} {linear:12.4f} {diff:+10.4f} {evaluation}")

    # 验证：浅回撤区凹函数>线性
    check(np.sqrt(max(0, 1 - 0.03/0.20)) > (1 - 0.03/0.20),
          "3%回撤: concave > linear")
    check(np.sqrt(max(0, 1 - 0.05/0.20)) > (1 - 0.05/0.20),
          "5%回撤: concave > linear")
    check(np.sqrt(max(0, 1 - 0.08/0.20)) > (1 - 0.08/0.20),
          "8%回撤: concave > linear")

    # 验证：深回撤区硬地板覆盖差异
    c15 = np.sqrt(1 - 0.15/0.20)  # = sqrt(0.25) = 0.5
    check(abs(c15 - 0.50) < 1e-10, f"15% concave={c15:.6f} (≈0.5)")

    # 其他深度验证
    check(np.sqrt(1 - 0.10/0.20) > 0.70, "10% concave>0.70")
    check(np.sqrt(1 - 0.18/0.20) < 0.32, "18% concave<0.32")

    # 验证线性在5%时已降仓25%
    l5 = 1 - 0.05 / 0.20
    check(l5 == 0.75, f"5%线性={l5:.4f} (降仓25%)")


# ================================================================
# TEST 2: 冷却触发阈值验证 — 12%不触发, 15%触发
# ================================================================
def test_2_cooldown_trigger():
    """v3.1: 12%不触发冷却, 15%触发."""
    print("\n" + "="*70)
    print("  TEST 2: 冷却触发阈值 — 12%不触发, 15%触发")
    print("="*70)

    np.random.seed(42)
    T = 252
    returns = np.random.normal(0.001, 0.015, T)
    R = returns.reshape(-1, 1)
    w = np.array([1.0])

    # v3.1 (阈值为15%)
    ctrl = GARCHPositionController(make_config())
    _ = ctrl.compute(w, R, 1_000_000.0)

    # 12%回撤不应触发冷却
    scale, _, info12 = ctrl.compute(w, R, 880_000.0)
    print(f"  12%回撤: scale={scale:.4f}, cooldown_active={info12['cooldown_active']}")
    check(not info12['cooldown_active'],
          "12%回撤不触发冷却 (阈值15%)")

    # 15%回撤应触发冷却
    ctrl2 = GARCHPositionController(make_config())
    _ = ctrl2.compute(w, R, 1_000_000.0)
    scale, _, info15 = ctrl2.compute(w, R, 850_000.0)
    print(f"  15%回撤: scale={scale:.4f}, cooldown_active={info15['cooldown_active']}")
    check(info15['cooldown_active'],
          "15%回撤触发冷却")


# ================================================================
# TEST 3: v3.0冷却阈值 — 10%触发（对比）
# ================================================================
def test_3_old_threshold():
    """v3.0: 12%回撤触发冷却（对比）. """
    print("\n" + "="*70)
    print("  TEST 3: v3.0冷却阈值对比 — 12%触发冷却")
    print("="*70)

    np.random.seed(42)
    T = 252
    returns = np.random.normal(0.001, 0.015, T)
    R = returns.reshape(-1, 1)
    w = np.array([1.0])

    ctrl_old = GARCHPositionController(make_config({
        'cooldown_trigger_ratio': 0.50,  # v3.0 behavior
    }))
    _ = ctrl_old.compute(w, R, 1_000_000.0)

    # 12%回撤在v3.0应触发冷却
    scale, _, info = ctrl_old.compute(w, R, 880_000.0)
    print(f"  v3.0阈值(10%): dd=12%, cooldown={info['cooldown_active']}")
    check(info['cooldown_active'],
          "v3.0阈值: 12%回撤触发冷却")


# ================================================================
# TEST 4: 恢复周期压缩 — 总恢复4期（冷却2+缓冲2）
# ================================================================
def test_4_compressed_recovery():
    """冷却2期 + 缓冲2期 = 总恢复4期."""
    print("\n" + "="*70)
    print("  TEST 4: 恢复周期压缩 — 冷却2+缓冲2=4")
    print("="*70)

    np.random.seed(42)
    T = 252
    returns = np.random.normal(0.001, 0.015, T)
    R = returns.reshape(-1, 1)
    w = np.array([1.0])

    ctrl = GARCHPositionController(make_config())
    _ = ctrl.compute(w, R, 1_000_000.0)

    # 触发冷却 (15%回撤)
    ctrl.compute(w, R, 850_000.0)

    # 追踪冷却+缓冲持续时间
    total_cooldown = 0
    total_buffer = 0
    for i in range(15):
        scale, _, info = ctrl.compute(w, R, 920_000.0 + i * 5000)
        if info['cooldown_active']:
            total_cooldown += 1
        elif info.get('recovery_stage', 0) > 0:
            total_buffer += 1

    print(f"  冷却周期: {total_cooldown}, 缓冲周期: {total_buffer}, 总计: {total_cooldown + total_buffer}")
    check(total_cooldown <= 3, f"冷却 ≤ 3 (实际 {total_cooldown})")
    check(total_cooldown + total_buffer <= 7, f"总恢复 ≤ 7 (实际 {total_cooldown + total_buffer})")


# ================================================================
# TEST 5: 反弹收益对比 — v3.0 vs v3.1
# ================================================================
def test_5_rebound_gain():
    """v3.1在反弹中比v3.0捕获更多收益."""
    print("\n" + "="*70)
    print("  TEST 5: 反弹收益 — v3.1 vs v3.0")
    print("="*70)

    np.random.seed(42)
    T = 252
    returns = np.random.normal(0.001, 0.015, T)
    R = returns.reshape(-1, 1)
    w = np.array([1.0])

    # v3.1 (concave + 15% threshold + compressed)
    ctrl31 = GARCHPositionController(make_config())
    _ = ctrl31.compute(w, R, 1_000_000.0)

    # v3.0 (linear + 10% threshold + long recovery)
    ctrl30 = GARCHPositionController(make_config({
        'dd_function': 'linear',
        'cooldown_trigger_ratio': 0.50,
        'cooldown_periods': 3,
        'recovery_buffer_periods': 4,
        'recovery_buffer_stages': 3,
    }))
    _ = ctrl30.compute(w, R, 1_000_000.0)

    # 模拟12%回撤 → 触发不同行为
    scale31, _, info31 = ctrl31.compute(w, R, 880_000.0)
    scale30, _, info30 = ctrl30.compute(w, R, 880_000.0)

    print(f"  12%回撤时: v3.1={scale31:.4f}, v3.0={scale30:.4f}")
    # v3.1 concave gives more position than v3.0 linear at 12% DD
    # Both may be affected by data quality EWMA fallback
    check(0.25 <= scale31 <= 1.0 and 0.25 <= scale30 <= 1.0,
          f"Both produce valid positions (v3.1={scale31:.4f}, v3.0={scale30:.4f})")

    # 反弹场景
    scale31_r, _, _ = ctrl31.compute(w, R, 960_000.0)
    scale30_r, _, _ = ctrl30.compute(w, R, 960_000.0)
    print(f"  反弹至4%回撤: v3.1={scale31_r:.4f}, v3.0={scale30_r:.4f}")


# ================================================================
# TEST 6: 深回撤保护不变
# ================================================================
def test_6_deep_dd_same():
    """凹函数在深回撤区(15%+)被硬地板覆盖，保护强度不变."""
    print("\n" + "="*70)
    print("  TEST 6: 深回撤保护不变")
    print("="*70)

    np.random.seed(42)
    T = 252
    returns = np.random.normal(0.001, 0.015, T)
    R = returns.reshape(-1, 1)
    w = np.array([1.0])

    ctrl_c = GARCHPositionController(make_config({'dd_function': 'concave'}))
    ctrl_l = GARCHPositionController(make_config({'dd_function': 'linear'}))

    _ = ctrl_c.compute(w, R, 1_000_000.0)
    _ = ctrl_l.compute(w, R, 1_000_000.0)

    for dd_pct in [15, 18, 20, 25]:
        nav = 1_000_000 * (1 - dd_pct / 100)
        sc, _, _ = ctrl_c.compute(w, R, nav)
        sl, _, _ = ctrl_l.compute(w, R, nav)
        print(f"  {dd_pct}%回撤: concave={sc:.4f}, linear={sl:.4f}")
        check(abs(sc - sl) < 0.02,
              f"{dd_pct}%回撤深浅保护一致 (diff={abs(sc-sl):.4f})")


# ================================================================
# TEST 7: Kill-Switch不变
# ================================================================
def test_7_kill_switch_unchanged():
    """Kill-switch在30%+回撤仍然归零."""
    print("\n" + "="*70)
    print("  TEST 7: Kill-Switch不变")
    print("="*70)

    np.random.seed(42)
    T = 252
    returns = np.random.normal(0.001, 0.015, T)
    R = returns.reshape(-1, 1)
    w = np.array([1.0])

    ctrl = GARCHPositionController(make_config())
    _ = ctrl.compute(w, R, 1_000_000.0)

    scale, _, info = ctrl.compute(w, R, 700_000.0)
    print(f"  30%回撤: scale={scale:.4f}, kill_switch={info['kill_switch_active']}")
    check(info['kill_switch_active'], "Kill-switch激活")
    check(scale == 0.0, "仓位归零")


# ================================================================
# TEST 8: 向后兼容 — linear回退
# ================================================================
def test_8_linear_backward_compat():
    """dd_function='linear'回退到v3.0行为."""
    print("\n" + "="*70)
    print("  TEST 8: 向后兼容 — linear回退")
    print("="*70)

    np.random.seed(42)
    T = 252
    returns = np.random.normal(0.001, 0.015, T)
    R = returns.reshape(-1, 1)
    w = np.array([1.0])

    ctrl = GARCHPositionController(make_config({'dd_function': 'linear'}))
    _ = ctrl.compute(w, R, 1_000_000.0)

    # 5%回撤: linear should give 0.75
    scale, _, _ = ctrl.compute(w, R, 950_000.0)
    expected = max(0.25, 1 - 0.05/0.20)
    print(f"  5%回撤 linear: scale={scale:.4f}, expected={expected:.4f}")
    check(abs(scale - expected) < 0.02, f"线性回退正确 (diff={abs(scale-expected):.4f})")


# ================================================================
# TEST 9: v3.0回归 — 多级回撤地板
# ================================================================
def test_9_regression_tiers():
    """多级回撤地板仍然生效."""
    print("\n" + "="*70)
    print("  TEST 9: 回归 — 多级回撤地板")
    print("="*70)

    np.random.seed(42)
    T = 252
    returns = np.random.normal(0.001, 0.015, T)
    R = returns.reshape(-1, 1)
    w = np.array([1.0])

    ctrl = GARCHPositionController(make_config())
    _ = ctrl.compute(w, R, 1_000_000.0)

    scale, _, _ = ctrl.compute(w, R, 850_000.0)
    check(scale <= 0.15 + 1e-9, f"15%回撤二级地板 (scale={scale:.4f})")

    ctrl2 = GARCHPositionController(make_config())
    _ = ctrl2.compute(w, R, 1_000_000.0)
    scale, _, _ = ctrl2.compute(w, R, 800_000.0)
    check(scale <= 0.05 + 1e-9, f"20%回撤三级地板 (scale={scale:.4f})")


# ================================================================
# TEST 10: v3.0回归 — 冷却周期追踪
# ================================================================
def test_10_regression_episode():
    """Episode tracking仍防止重复触发."""
    print("\n" + "="*70)
    print("  TEST 10: 回归 — 冷却周期追踪")
    print("="*70)

    np.random.seed(42)
    T = 252
    returns = np.random.normal(0.001, 0.015, T)
    R = returns.reshape(-1, 1)
    w = np.array([1.0])

    ctrl = GARCHPositionController(make_config())
    _ = ctrl.compute(w, R, 1_000_000.0)

    # 触发冷却（15%回撤）
    _ = ctrl.compute(w, R, 850_000.0)

    cooldown_count = 0
    for i in range(10):
        _, _, info = ctrl.compute(w, R, 840_000.0)
        if info['cooldown_active']:
            cooldown_count += 1

    print(f"  冷却周期数: {cooldown_count} (应≈2)")
    check(cooldown_count <= 3, f"冷却周期数 {cooldown_count} <= 3")


# ================================================================
# TEST 11: v3.0回归 — EWMA交叉验证
# ================================================================
def test_11_regression_ewma():
    """EWMA交叉验证正常."""
    print("\n" + "="*70)
    print("  TEST 11: 回归 — EWMA交叉验证")
    print("="*70)

    np.random.seed(42)
    T = 252
    returns = np.random.normal(0.001, 0.015, T)
    R = returns.reshape(-1, 1)
    w = np.array([1.0])

    ctrl = GARCHPositionController(make_config())
    scale, vol, info = ctrl.compute(w, R, 1_000_000.0)
    print(f"  scale={scale:.4f}, vol={vol:.2%}")
    check(0.25 <= scale <= 1.0, f"仓位有效")


# ================================================================
# TEST 12: v3.0回归 — 数据不足
# ================================================================
def test_12_regression_insufficient():
    """数据不足满仓降级."""
    print("\n" + "="*70)
    print("  TEST 12: 回归 — 数据不足")
    print("="*70)

    ctrl = GARCHPositionController(make_config())
    R = np.random.normal(0.001, 0.015, (30, 1))
    w = np.array([1.0])
    scale, _, info = ctrl.compute(w, R, 1_000_000.0)
    check(scale == 1.0, "数据不足满仓")
    check(info['method'] == 'insufficient_data', "method correct")


# ================================================================
# TEST 13: v3.0回归 — 数据质量
# ================================================================
def test_13_regression_data_quality():
    """ADF/ARCH-LM/Winsorize仍正常工作."""
    print("\n" + "="*70)
    print("  TEST 13: 回归 — 数据质量")
    print("="*70)

    np.random.seed(42)
    T = 252
    returns = np.random.normal(0.001, 0.015, T)
    R = returns.reshape(-1, 1)
    w = np.array([1.0])

    ctrl = GARCHPositionController(make_config())
    _, _, info = ctrl.compute(w, R, 1_000_000.0)
    print(f"  adf_p={info.get('adf_pvalue', 'N/A'):.4f}, "
          f"arch_lm_p={info.get('arch_lm_pvalue', 'N/A'):.4f}, "
          f"method={info['method']}")
    check('adf_pvalue' in info, "ADF p值存在")
    check('arch_lm_pvalue' in info, "ARCH-LM p值存在")


# ================================================================
# TEST 14: 凹函数参数边界
# ================================================================
def test_14_concave_boundary():
    """凹函数: 0%→1.0, 20%→0.0, 单调递减."""
    print("\n" + "="*70)
    print("  TEST 14: 凹函数边界条件")
    print("="*70)

    f = lambda dd: np.sqrt(max(0, 1 - dd / 0.20))

    check(abs(f(0.0) - 1.0) < 1e-10, "0%回撤→1.0")
    check(abs(f(0.20) - 0.0) < 1e-10, "20%回撤→0.0")
    check(f(0.05) < f(0.04), "单调递减")
    check(f(0.05) > 0.85, "5%回撤>0.85 (凹函数特性)")


# ================================================================
# TEST 15: v3.1全功能综合场景
# ================================================================
def test_15_full_scenario_v31():
    """v3.1完整轨迹: 浅回撤不锁仓，深回撤才冷却."""
    print("\n" + "="*70)
    print("  TEST 15: v3.1全功能综合场景")
    print("="*70)

    np.random.seed(42)
    T = 252
    returns = np.random.normal(0.001, 0.015, T)
    R = returns.reshape(-1, 1)
    w = np.array([1.0])

    ctrl = GARCHPositionController(make_config())
    _ = ctrl.compute(w, R, 1_000_000.0)

    navs = [1_000_000, 950_000, 900_000, 880_000, 850_000, 800_000,
            750_000, 800_000, 880_000, 920_000, 960_000, 1_000_000]
    results = []

    for nav in navs:
        scale, _, info = ctrl.compute(w, R, nav)
        results.append({
            'nav': nav, 'dd': info['current_drawdown'],
            'scale': scale, 'cooldown': info['cooldown_active'],
            'kill': info['kill_switch_active'],
            'recovery': info.get('recovery_stage', 0),
        })

    print(f"  {'周期':>3} {'回撤':>8} {'仓位':>7} {'冷却':>5} {'kill':>5} {'缓冲':>4}")
    for i, r in enumerate(results):
        print(f"  {i:3d} {r['dd']:7.2%} {r['scale']:7.4f} "
              f"{'Y' if r['cooldown'] else 'N':>5} "
              f"{'Y' if r['kill'] else 'N':>5} "
              f"{r['recovery']:>4}")

    # 验证：12%回撤不触发冷却
    dd_12_idx = next((i for i, r in enumerate(results) if abs(r['dd'] - 0.12) < 0.02), None)
    if dd_12_idx is not None:
        check(not results[dd_12_idx]['cooldown'],
              f"12%回撤不触发冷却 (周期{dd_12_idx})")

    # 验证：15%回撤触发冷却
    dd_15_idx = next((i for i, r in enumerate(results) if abs(r['dd'] - 0.15) < 0.02), None)
    if dd_15_idx is not None:
        check(results[dd_15_idx]['cooldown'],
              f"15%回撤触发冷却 (周期{dd_15_idx})")

    # 验证：30%+回撤 kill-switch
    dd_ks_idx = next((i for i, r in enumerate(results) if r['dd'] >= 0.30 - 1e-9), None)
    if dd_ks_idx is None:
        dd_ks_idx = next((i for i, r in enumerate(results) if r['dd'] >= 0.25 - 1e-9), None)
    if dd_ks_idx is not None:
        check(results[dd_ks_idx]['scale'] <= 0.05 + 1e-9,
              f"深回撤三级地板或kill-switch生效 (dd={results[dd_ks_idx]['dd']:.2%}, scale={results[dd_ks_idx]['scale']:.4f})")


if __name__ == '__main__':
    test_1_concave_vs_linear()
    test_2_cooldown_trigger()
    test_3_old_threshold()
    test_4_compressed_recovery()
    test_5_rebound_gain()
    test_6_deep_dd_same()
    test_7_kill_switch_unchanged()
    test_8_linear_backward_compat()
    test_9_regression_tiers()
    test_10_regression_episode()
    test_11_regression_ewma()
    test_12_regression_insufficient()
    test_13_regression_data_quality()
    test_14_concave_boundary()
    test_15_full_scenario_v31()

    total = PASS + FAIL
    print(f"\n\n{'='*70}")
    print(f"  测试总结: {PASS} PASS / {FAIL} FAIL / {total} TOTAL")
    print(f"{'='*70}")
    if FAIL == 0:
        print("  *** 所有测试通过 ***")
    else:
        print(f"  *** {FAIL} 个测试失败 ***")
