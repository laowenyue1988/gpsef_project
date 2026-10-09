"""
GARCH 仓位控制 v2.0 — 改进验证测试套件
=========================================
验证5项改进:
  1. Student-t分布 vs Normal分布 (肥尾数据拟合精度)
  2. EWMA交叉验证 (max(GARCH, EWMA) 防止低估)
  3. 冷却渐进解锁 (线性恢复 vs 硬锁定)
  4. 冷却回撤周期追踪 (单次触发, 回撤恢复前不重置)
  5. 多级回撤地板 (15%→0.15, 20%→0.05)

同时包含回归测试确保原有功能不受影响。
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

def make_v2_config(**overrides):
    """v2.0 默认配置"""
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
        # v2.0 new
        'garch_dist': 't',
        'use_ewma_cross_check': True,
        'use_gradual_cooldown': True,
        'secondary_dd_threshold': 0.15,
        'secondary_dd_floor': 0.15,
        'tertiary_dd_threshold': 0.20,
        'tertiary_dd_floor': 0.05,
    }
    cfg.update(overrides)
    return cfg


def make_v1_config(**overrides):
    """v1.0 兼容配置（禁用所有v2.0新特性）"""
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
        # v2.0 disabled
        'garch_dist': 'normal',
        'use_ewma_cross_check': False,
        'use_gradual_cooldown': False,
        'use_cooldown_episode_tracking': False,  # v1.0 behavior: reset every period
        'secondary_dd_threshold': 1.0,   # effectively disabled
        'secondary_dd_floor': 1.0,
        'tertiary_dd_threshold': 1.0,
        'tertiary_dd_floor': 1.0,
    }
    cfg.update(overrides)
    return cfg


def gen_correlated_returns(n=250, n_stocks=20, daily_vol=0.02, corr=0.6, seed=42):
    rng = np.random.RandomState(seed)
    cov = np.full((n_stocks, n_stocks), corr * daily_vol**2)
    np.fill_diagonal(cov, daily_vol**2)
    L = np.linalg.cholesky(cov)
    returns = rng.randn(n, n_stocks) @ L.T
    return returns


def gen_volatility_clustering(n=250, seed=42):
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
    rng = np.random.RandomState(seed)
    r = rng.randn(n) * 0.012
    r[crash_day] = crash_mag
    for t in range(crash_day + 1, min(crash_day + 30, n)):
        r[t] = rng.randn() * 0.03
    return r


def gen_fat_tail_returns(n=250, df=3, daily_vol=0.015, seed=42):
    """生成Student-t分布收益率（自由度df=3, 肥尾）"""
    rng = np.random.RandomState(seed)
    r = rng.standard_t(df, size=n) * daily_vol
    return r


PASS_COUNT = 0
FAIL_COUNT = 0

def check(condition, test_name, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  [PASS] {test_name} {detail}")
    else:
        FAIL_COUNT += 1
        print(f"  [FAIL] {test_name} {detail}")


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ============================================================
# 测试套件
# ============================================================

def test_1_student_t_vs_normal_fat_tail():
    """
    测试1: Student-t vs Normal 分布在肥尾数据上的拟合精度对比
    经济学场景: 金融收益率具有肥尾特征(excess kurtosis > 0)
    Student-t GARCH应能更好地拟合肥尾数据, 估计更高的条件波动率
    """
    print_header("TEST 1: Student-t vs Normal — 肥尾数据拟合")

    T = 250
    rets = gen_fat_tail_returns(T, df=3, daily_vol=0.015, seed=42)
    returns = rets.reshape(-1, 1)
    weights = np.array([1.0])
    nav = 1_000_000

    from scipy.stats import kurtosis
    data_kurt = kurtosis(rets)

    # Normal distribution (v1.0)
    ctrl_normal = GARCHPositionController(make_v1_config(
        garch_dist='normal',
        use_drawdown_control=False,
    ))
    scale_n, vol_n, info_n = ctrl_normal.compute(weights, returns, nav)

    # Student-t distribution (v2.0)
    ctrl_t = GARCHPositionController(make_v2_config(
        garch_dist='t',
        use_ewma_cross_check=False,  # isolate distribution effect
        use_drawdown_control=False,
    ))
    scale_t, vol_t, info_t = ctrl_t.compute(weights, returns, nav)

    # Extract fitted nu (degrees of freedom) if available
    fitted_nu = None
    try:
        from arch import arch_model
        ret_scaled = (rets - rets.mean()) / rets.std() * 100
        model_t = arch_model(ret_scaled, vol='Garch', p=1, q=1, dist='t', rescale=False)
        result_t = model_t.fit(disp='off', show_warning=False)
        fitted_nu = result_t.params.get('nu', None)
    except:
        pass

    print(f"  数据超额峰度 = {data_kurt:.4f} (Normal=0, t(3)理论=∞)")
    print(f"  Normal GARCH 年化波动率 = {vol_n:.4f} ({vol_n*100:.2f}%)")
    print(f"  Student-t GARCH 年化波动率 = {vol_t:.4f} ({vol_t*100:.2f}%)")
    if fitted_nu:
        print(f"  拟合自由度 nu = {fitted_nu:.2f} (越低越肥尾, 正态=∞)")

    check(data_kurt > 1.0, "数据具有显著肥尾特征", f"(kurtosis={data_kurt:.4f})")
    check(vol_t >= vol_n * 0.98, "Student-t 在肥尾数据上波动率>=Normal",
          f"(t={vol_t:.4f} >= n={vol_n:.4f})")
    check(scale_t <= scale_n + 0.001, "Student-t 仓位<=Normal (更保守)",
          f"(t={scale_t:.4f} <= n={scale_n:.4f})")

    return {'normal_vol': vol_n, 't_vol': vol_t, 'kurtosis': data_kurt, 'nu': fitted_nu}


def test_2_student_t_vs_normal_crash():
    """
    测试2: Student-t vs Normal 在崩盘场景下的波动率预测
    经济学场景: 2015年8月24日黑色星期一，单日暴跌6%，之后波动率飙升
    Student-t应能更好地处理极端值, 减少低估
    """
    print_header("TEST 2: Student-t vs Normal — 崩盘场景")

    T = 250
    rets = gen_crash_returns(T, crash_day=200, crash_mag=-0.06, seed=42)
    returns = rets.reshape(-1, 1)
    weights = np.array([1.0])
    nav = 1_000_000

    # Normal
    ctrl_normal = GARCHPositionController(make_v1_config(
        garch_dist='normal',
        use_ewma_cross_check=False,
        use_drawdown_control=False,
    ))
    scale_n, vol_n, _ = ctrl_normal.compute(weights, returns, nav)

    # Student-t
    ctrl_t = GARCHPositionController(make_v2_config(
        garch_dist='t',
        use_ewma_cross_check=False,
        use_drawdown_control=False,
    ))
    scale_t, vol_t, _ = ctrl_t.compute(weights, returns, nav)

    # EWMA for reference
    ewma_vol_val = np.sqrt(np.sum(
        (1-0.94) * 0.94**np.arange(T)[::-1] / ((1-0.94) * 0.94**np.arange(T)[::-1]).sum() * rets**2
    )) * np.sqrt(252)

    post_vol = np.std(rets[200:]) * np.sqrt(252)

    print(f"  崩盘后年化波动率 = {post_vol:.4f} ({post_vol*100:.2f}%)")
    print(f"  EWMA年化波动率   = {ewma_vol_val:.4f} ({ewma_vol_val*100:.2f}%)")
    print(f"  Normal GARCH年化波动率   = {vol_n:.4f} ({vol_n*100:.2f}%)")
    print(f"  Student-t GARCH年化波动率 = {vol_t:.4f} ({vol_t*100:.2f}%)")

    # Both should underestimate post-crash vol (GARCH limitation), but neither should be wildly off
    n_err = abs(vol_n - post_vol) / post_vol
    t_err = abs(vol_t - post_vol) / post_vol
    print(f"\n  Normal 误差   = {n_err*100:.1f}%")
    print(f"  Student-t 误差 = {t_err*100:.1f}%")

    # Student-t should not be worse than Normal
    check(t_err <= n_err * 1.1, "Student-t 误差不大于Normal的110%",
          f"(t_err={t_err*100:.1f}% <= n_err*1.1={n_err*110:.1f}%)")

    return {'normal_vol': vol_n, 't_vol': vol_t, 'post_vol': post_vol}


def test_3_ewma_cross_validation():
    """
    测试3: EWMA交叉验证 — max(GARCH, EWMA) 防止GARCH低估
    经济学场景: GARCH在快速regime change时可能低估，EWMA对近期更敏感
    """
    print_header("TEST 3: EWMA交叉验证 — 防止GARCH低估")

    T = 250
    rets = gen_volatility_clustering(T, seed=42)
    returns = rets.reshape(-1, 1)
    weights = np.array([1.0])
    nav = 1_000_000

    # Without cross-validation
    ctrl_no_xval = GARCHPositionController(make_v2_config(
        use_ewma_cross_check=False,
        use_drawdown_control=False,
    ))
    scale_no, vol_no, info_no = ctrl_no_xval.compute(weights, returns, nav)

    # With cross-validation
    ctrl_xval = GARCHPositionController(make_v2_config(
        use_ewma_cross_check=True,
        use_drawdown_control=False,
    ))
    scale_xval, vol_xval, info_xval = ctrl_xval.compute(weights, returns, nav)

    # EWMA standalone
    ctrl_ewma = GARCHPositionController(make_v2_config(vol_method='ewma', use_drawdown_control=False))
    _, vol_ewma, _ = ctrl_ewma.compute(weights, returns, nav)

    print(f"  GARCH年化波动率 (无交叉验证) = {vol_no:.4f} ({vol_no*100:.2f}%)")
    print(f"  EWMA年化波动率               = {vol_ewma:.4f} ({vol_ewma*100:.2f}%)")
    print(f"  GARCH年化波动率 (有交叉验证) = {vol_xval:.4f} ({vol_xval*100:.2f}%)")
    print(f"  info['ewma_vol']             = {info_xval.get('ewma_vol',0)*np.sqrt(252):.4f}")
    print(f"  method (交叉验证)            = {info_xval['method']}")

    # Final vol should be max(GARCH, EWMA)
    expected_vol = max(vol_no, vol_ewma)
    check(abs(vol_xval - expected_vol) < 0.005, "最终波动率 = max(GARCH, EWMA)",
          f"(got={vol_xval:.4f}, expected_max={expected_vol:.4f})")

    # If EWMA > GARCH, cross-validation should pick EWMA and mark it
    if vol_ewma > vol_no:
        check('xval' in info_xval['method'], "method标记了交叉验证生效",
              f"(method={info_xval['method']})")
        check(scale_xval <= scale_no + 0.001, "交叉验证后仓位更保守",
              f"(xval={scale_xval:.4f} <= no_xval={scale_no:.4f})")
    else:
        check('xval' not in info_xval['method'], "GARCH>=EWMA时交叉验证不改变结果",
              f"(method={info_xval['method']})")

    return {'garch_vol': vol_no, 'ewma_vol': vol_ewma, 'xval_vol': vol_xval}


def test_4_cooldown_gradual_unlock():
    """
    测试4: 冷却渐进解锁 — 验证仓位在冷却期内线性恢复
    对比v1.0硬锁定 vs v2.0渐进解锁
    """
    print_header("TEST 4: 冷却渐进解锁 vs 硬锁定")

    N, T = 20, 250
    returns = gen_correlated_returns(T, N, daily_vol=0.012, corr=0.6, seed=42)
    weights = np.ones(N) / N

    # --- v1.0: Hard lock cooldown ---
    print("\n  --- v1.0 硬锁定冷却 ---")
    ctrl_v1 = GARCHPositionController(make_v1_config(cooldown_periods=3))
    ctrl_v1._nav_peak = 1_200_000
    scale1_v1, _, info1_v1 = ctrl_v1.compute(weights, returns, 1_056_000)  # 12% dd
    print(f"  Phase1 (12% dd): scale={scale1_v1:.4f}, cooldown={info1_v1['cooldown_active']}")
    v1_scales = [scale1_v1]
    for i in range(4):
        s, _, info = ctrl_v1.compute(weights, returns, 1_200_000)  # recovered
        v1_scales.append(s)
        print(f"  Phase{2+i} (recovered, cd={info['cooldown_active']}): scale={s:.4f}")

    # --- v2.0: Gradual unlock cooldown ---
    print("\n  --- v2.0 渐进解锁冷却 ---")
    ctrl_v2 = GARCHPositionController(make_v2_config(cooldown_periods=3))
    ctrl_v2._nav_peak = 1_200_000
    scale1_v2, _, info1_v2 = ctrl_v2.compute(weights, returns, 1_056_000)  # 12% dd
    print(f"  Phase1 (12% dd): scale={scale1_v2:.4f}, cooldown={info1_v2['cooldown_active']}")
    v2_scales = [scale1_v2]
    for i in range(4):
        s, _, info = ctrl_v2.compute(weights, returns, 1_200_000)
        v2_scales.append(s)
        print(f"  Phase{2+i} (recovered, cd={info['cooldown_active']}): scale={s:.4f}, "
              f"unlock={info.get('cooldown_unlock', 0):.2f}")

    print(f"\n  --- 对比 ---")
    print(f"  {'Phase':>6} | {'v1.0硬锁':>10} | {'v2.0渐进':>10} | {'差异':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for i in range(5):
        diff = v2_scales[i] - v1_scales[i]
        print(f"  {i+1:>6} | {v1_scales[i]:>10.4f} | {v2_scales[i]:>10.4f} | {diff:>+10.4f}")

    # v2.0 should show gradual recovery during cooldown (Phase 2 < Phase 3 < Phase 4)
    check(v2_scales[1] <= v2_scales[2] + 0.001, "v2.0 Phase2 <= Phase3 (渐进恢复)",
          f"({v2_scales[1]:.4f} <= {v2_scales[2]:.4f})")
    check(v2_scales[2] <= v2_scales[3] + 0.001, "v2.0 Phase3 <= Phase4 (渐进恢复)",
          f"({v2_scales[2]:.4f} <= {v2_scales[3]:.4f})")

    # v2.0 Phase 2 should have HIGHER scale than v1.0 Phase 2 (gradual unlock allows partial recovery)
    check(v2_scales[1] >= v1_scales[1] - 0.001, "v2.0 Phase2 >= v1.0 Phase2 (渐进解锁允许部分恢复)",
          f"(v2={v2_scales[1]:.4f} >= v1={v1_scales[1]:.4f})")

    # Both should fully recover after cooldown ends
    check(v1_scales[4] >= 0.8 and v2_scales[4] >= 0.8, "冷却结束后两种方式都恢复高仓位",
          f"(v1={v1_scales[4]:.4f}, v2={v2_scales[4]:.4f})")

    # Unlock factors should be monotonically increasing during cooldown
    # Include trigger call as first cooldown period
    ctrl_v2_test = GARCHPositionController(make_v2_config(cooldown_periods=3))
    ctrl_v2_test._nav_peak = 1_200_000
    unlocks = []
    # Trigger call (first cooldown period)
    _, _, info_trig = ctrl_v2_test.compute(weights, returns, 1_056_000)
    if info_trig['cooldown_active']:
        unlocks.append(info_trig.get('cooldown_unlock', 0))
    # Subsequent calls
    for i in range(3):
        _, _, info = ctrl_v2_test.compute(weights, returns, 1_200_000)
        if info['cooldown_active']:
            unlocks.append(info.get('cooldown_unlock', 0))
    print(f"\n  冷却期内解锁因子序列: {unlocks}")
    check(len(unlocks) == 3, "冷却期3轮全部激活", f"(激活轮数={len(unlocks)})")
    if len(unlocks) >= 2:
        check(unlocks == sorted(unlocks), "解锁因子单调递增", f"({unlocks})")

    return {'v1_scales': v1_scales, 'v2_scales': v2_scales}


def test_5_cooldown_episode_tracking():
    """
    测试5: 冷却回撤周期追踪 — 回撤持续>10%时冷却不重复触发
    v1.0 bug: 每次dd>10%都重置cooldown_remaining=3, 导致冷却永不结束
    v2.0 fix: 回撤周期追踪, 只在回撤恢复后才能重新触发
    """
    print_header("TEST 5: 冷却回撤周期追踪 — 不重复触发")

    N, T = 20, 250
    returns = gen_correlated_returns(T, N, daily_vol=0.012, corr=0.6, seed=42)
    weights = np.ones(N) / N

    # --- v1.0: cooldown keeps resetting ---
    print("\n  --- v1.0 冷却持续重置测试 ---")
    ctrl_v1 = GARCHPositionController(make_v1_config(cooldown_periods=3))
    ctrl_v1._nav_peak = 1_200_000
    v1_cd_active = []
    v1_scales = []
    for i in range(8):
        s, _, info = ctrl_v1.compute(weights, returns, 1_056_000)  # persistent 12% dd
        v1_cd_active.append(info['cooldown_active'])
        v1_scales.append(s)
        print(f"  周期{i}: scale={s:.4f}, cd_remaining={ctrl_v1._cooldown_remaining}, "
              f"cd_active={info['cooldown_active']}")

    v1_cooldown_count = sum(1 for a in v1_cd_active if a)
    print(f"  v1.0 冷却持续周期数: {v1_cooldown_count}/8 (配置=3)")

    # --- v2.0: cooldown triggers once, no re-trigger until recovery ---
    print("\n  --- v2.0 回撤周期追踪测试 ---")
    ctrl_v2 = GARCHPositionController(make_v2_config(cooldown_periods=3))
    ctrl_v2._nav_peak = 1_200_000
    v2_cd_active = []
    v2_scales = []
    for i in range(8):
        s, _, info = ctrl_v2.compute(weights, returns, 1_056_000)  # persistent 12% dd
        v2_cd_active.append(info['cooldown_active'])
        v2_scales.append(s)
        print(f"  周期{i}: scale={s:.4f}, cd_remaining={ctrl_v2._cooldown_remaining}, "
              f"cd_active={info['cooldown_active']}, episode={ctrl_v2._cooldown_episode_active}")

    v2_cooldown_count = sum(1 for a in v2_cd_active if a)
    print(f"  v2.0 冷却持续周期数: {v2_cooldown_count}/8 (配置=3)")

    # v1.0: cooldown should persist much longer than 3 periods (bug)
    check(v1_cooldown_count > 3, "v1.0 冷却持续>3期 (重置bug)",
          f"(实际={v1_cooldown_count})")

    # v2.0: cooldown should be exactly 3 periods (no re-trigger while dd persists)
    check(v2_cooldown_count == 3, "v2.0 冷却恰好3期 (回撤周期追踪修复)",
          f"(实际={v2_cooldown_count})")

    # After cooldown expires, v2.0 should NOT be in cooldown even with continued drawdown
    check(not v2_cd_active[4], "v2.0 周期5冷却已结束 (即使回撤持续)",
          f"(cd_active={v2_cd_active[4]})")

    # v2.0 should not re-trigger at period 3 (dd still > 10%)
    check(not v2_cd_active[3], "v2.0 周期4不在冷却 (回撤持续但不重复触发)",
          f"(cd_active={v2_cd_active[3]})")

    print(f"\n  --- v2.0 回撤恢复后可重新触发 ---")
    # Now recover below 10% dd, then re-drawdown should trigger again
    ctrl_v2b = GARCHPositionController(make_v2_config(cooldown_periods=3))
    ctrl_v2b._nav_peak = 1_200_000
    # Trigger
    s1, _, _ = ctrl_v2b.compute(weights, returns, 1_056_000)
    # Wait for cooldown to pass
    for _ in range(3):
        ctrl_v2b.compute(weights, returns, 1_056_000)
    # Recover below 10%
    ctrl_v2b.compute(weights, returns, 1_100_000)  # 8.3% dd < 10%
    # Now re-drawdown to 12%
    s_re, _, info_re = ctrl_v2b.compute(weights, returns, 1_056_000)
    print(f"  恢复后重新触发: scale={s_re:.4f}, cd_active={info_re['cooldown_active']}, "
          f"cd_remaining={ctrl_v2b._cooldown_remaining}")
    check(info_re['cooldown_active'], "回撤恢复后可重新触发冷却", "")

    return {'v1_cooldown_periods': v1_cooldown_count, 'v2_cooldown_periods': v2_cooldown_count}


def test_6_secondary_dd_floor():
    """
    测试6: 二级回撤地板 — dd>=15%时仓位<=15%
    """
    print_header("TEST 6: 二级回撤地板 (15% → 0.15)")

    N, T = 20, 250
    returns = gen_correlated_returns(T, N, daily_vol=0.012, corr=0.6, seed=42)
    weights = np.ones(N) / N
    nav_peak = 1_000_000

    # Test at exactly 15% drawdown
    dd = 0.15
    nav = nav_peak * (1 - dd)

    # v1.0 (no secondary floor)
    ctrl_v1 = GARCHPositionController(make_v1_config())
    ctrl_v1._nav_peak = nav_peak
    scale_v1, _, info_v1 = ctrl_v1.compute(weights, returns, nav)

    # v2.0 (with secondary floor) - disable cooldown to isolate floor effect
    ctrl_v2 = GARCHPositionController(make_v2_config(cooldown_periods=0))
    ctrl_v2._nav_peak = nav_peak
    scale_v2, _, info_v2 = ctrl_v2.compute(weights, returns, nav)

    v1_theory = max(0.25, 1 - dd / 0.20)
    v2_theory = min(v1_theory, 0.15)

    print(f"  回撤 = {dd*100:.0f}%")
    print(f"  v1.0 理论 dd_scale = max(0.25, 1-{dd}/0.20) = {v1_theory:.4f}")
    print(f"  v2.0 理论 dd_scale = min({v1_theory:.4f}, 0.15) = {v2_theory:.4f}")
    print(f"  v1.0 实际 scale = {scale_v1:.4f}, dd_scale = {info_v1['dd_scale']:.4f}")
    print(f"  v2.0 实际 scale = {scale_v2:.4f}, dd_scale = {info_v2['dd_scale']:.4f}")

    check(scale_v2 <= 0.15 + 0.01, "v2.0 在15%回撤时仓位<=0.15 (二级地板)",
          f"(scale={scale_v2:.4f})")
    check(scale_v2 < scale_v1, "v2.0 在15%回撤时仓位 < v1.0",
          f"(v2={scale_v2:.4f} < v1={scale_v1:.4f})")

    # Test at 14% (below secondary threshold)
    dd2 = 0.14
    nav2 = nav_peak * (1 - dd2)
    ctrl_v2_14 = GARCHPositionController(make_v2_config(cooldown_periods=0))
    ctrl_v2_14._nav_peak = nav_peak
    scale_v2_14, _, _ = ctrl_v2_14.compute(weights, returns, nav2)
    print(f"\n  回撤 = {dd2*100:.0f}% (低于二级阈值15%)")
    print(f"  v2.0 scale = {scale_v2_14:.4f}")
    check(scale_v2_14 > 0.15, "14%回撤时二级地板不触发",
          f"(scale={scale_v2_14:.4f} > 0.15)")

    return {'v1_15pct': scale_v1, 'v2_15pct': scale_v2, 'v2_14pct': scale_v2_14}


def test_7_tertiary_dd_floor():
    """
    测试7: 三级回撤地板 — dd>=20%时仓位<=5%
    """
    print_header("TEST 7: 三级回撤地板 (20% → 0.05)")

    N, T = 20, 250
    returns = gen_correlated_returns(T, N, daily_vol=0.012, corr=0.6, seed=42)
    weights = np.ones(N) / N
    nav_peak = 1_000_000

    for dd in [0.18, 0.20, 0.22, 0.25, 0.30]:
        nav = nav_peak * (1 - dd)
        ctrl = GARCHPositionController(make_v2_config(cooldown_periods=0))
        ctrl._nav_peak = nav_peak
        scale, _, info = ctrl.compute(weights, returns, nav)

        v1_dd_scale = max(0.25, 1 - dd / 0.20)
        v2_dd_scale = v1_dd_scale
        if dd >= 0.15:
            v2_dd_scale = min(v2_dd_scale, 0.15)
        if dd >= 0.20:
            v2_dd_scale = min(v2_dd_scale, 0.05)

        print(f"  dd={dd*100:>5.1f}%: v1_dd={v1_dd_scale:.4f}, v2_dd={v2_dd_scale:.4f}, "
              f"actual_scale={scale:.4f}, info_dd_scale={info['dd_scale']:.4f}")

        if dd >= 0.20:
            check(scale <= 0.05 + 0.01, f"dd={dd*100:.0f}% 时仓位<=0.05 (三级地板)",
                  f"(scale={scale:.4f})")

    return True


def test_8_multi_tier_dd_sweep():
    """
    测试8: 多级回撤地板完整灵敏度扫描
    """
    print_header("TEST 8: 多级回撤地板灵敏度扫描")

    cfg = make_v2_config(cooldown_periods=0)
    N, T = 20, 250
    returns = gen_correlated_returns(T, N, daily_vol=0.008, corr=0.6, seed=42)
    weights = np.ones(N) / N
    nav_peak = 1_000_000

    drawdowns = [0.00, 0.05, 0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.19, 0.20, 0.22, 0.25, 0.30]

    print(f"\n  {'dd%':>5} | {'tier1':>8} | {'tier2':>8} | {'tier3':>8} | {'v2_theory':>10} | {'actual':>8} | {'match':>5}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*5}")

    all_match = True
    for dd in drawdowns:
        ctrl = GARCHPositionController(cfg)
        ctrl._nav_peak = nav_peak
        nav = nav_peak * (1 - dd)
        scale, vol, info = ctrl.compute(weights, returns, nav)

        tier1 = max(0.25, 1 - dd / 0.20)
        tier2 = 0.15 if dd >= 0.15 else 1.0
        tier3 = 0.05 if dd >= 0.20 else 1.0
        v2_theory = min(tier1, tier2, tier3)

        match = abs(info['dd_scale'] - v2_theory) < 0.01
        if not match:
            all_match = False

        print(f"  {dd*100:>4.0f}% | {tier1:>8.4f} | {tier2:>8.4f} | {tier3:>8.4f} | {v2_theory:>10.4f} | {info['dd_scale']:>8.4f} | {'OK' if match else 'XX'}")

    check(all_match, "所有回撤点dd_scale匹配理论值", "")
    return all_match


def test_9_full_simulation_v2():
    """
    测试9: v1.0 vs v2.0 完整模拟对比 — 20个调仓周期仓位轨迹
    """
    print_header("TEST 9: v1.0 vs v2.0 完整模拟对比")

    N, T = 20, 250
    returns = gen_correlated_returns(T, N, daily_vol=0.012, corr=0.6, seed=42)
    weights = np.ones(N) / N

    navs = np.concatenate([
        np.linspace(1_000_000, 820_000, 10),  # 18% drawdown
        np.linspace(820_000, 1_000_000, 10),  # recovery
    ])

    ctrl_v1 = GARCHPositionController(make_v1_config())
    ctrl_v2 = GARCHPositionController(make_v2_config())

    print(f"\n  {'周期':>4} | {'NAV':>10} | {'dd%':>6} | {'v1_scale':>9} | {'v2_scale':>9} | {'v1_cd':>5} | {'v2_cd':>5} | {'unlock':>7}")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*6}-+-{'-'*9}-+-{'-'*9}-+-{'-'*5}-+-{'-'*5}-+-{'-'*7}")

    v1_scales = []
    v2_scales = []
    for i, nav in enumerate(navs):
        s1, _, i1 = ctrl_v1.compute(weights, returns, nav)
        s2, _, i2 = ctrl_v2.compute(weights, returns, nav)
        v1_scales.append(s1)
        v2_scales.append(s2)
        dd = i2['current_drawdown']
        print(f"  {i:>4} | {nav:>10,.0f} | {dd*100:>5.1f}% | {s1:>9.4f} | {s2:>9.4f} | "
              f"{'Y' if i1['cooldown_active'] else 'N':>5} | {'Y' if i2['cooldown_active'] else 'N':>5} | "
              f"{i2.get('cooldown_unlock',0):>7.2f}")

    print(f"\n  --- 统计对比 ---")
    print(f"  v1.0 最低仓位 = {min(v1_scales):.4f}")
    print(f"  v2.0 最低仓位 = {min(v2_scales):.4f}")
    print(f"  v1.0 最终仓位 = {v1_scales[-1]:.4f}")
    print(f"  v2.0 最终仓位 = {v2_scales[-1]:.4f}")

    # v2.0 should have faster recovery in final periods (gradual unlock)
    check(np.mean(v2_scales[-3:]) >= np.mean(v1_scales[-3:]) - 0.05,
          "v2.0恢复期仓位 >= v1.0",
          f"(v2={np.mean(v2_scales[-3:]):.4f} >= v1={np.mean(v1_scales[-3:]):.4f})")

    return {'v1_scales': v1_scales, 'v2_scales': v2_scales}


def test_10_crash_scenario_comprehensive():
    """
    测试10: 崩盘场景综合测试 — Student-t + EWMA交叉 + 多级回撤
    """
    print_header("TEST 10: 崩盘综合场景 (Student-t + EWMA交叉 + 多级回撤)")

    N, T = 20, 250
    returns = gen_correlated_returns(T, N, daily_vol=0.03, corr=0.8, seed=42)
    weights = np.ones(N) / N

    # 18% drawdown
    nav_peak = 1_200_000
    nav = nav_peak * (1 - 0.18)  # 984,000

    # v1.0
    ctrl_v1 = GARCHPositionController(make_v1_config())
    ctrl_v1._nav_peak = nav_peak
    scale_v1, vol_v1, info_v1 = ctrl_v1.compute(weights, returns, nav)

    # v2.0 (disable cooldown to isolate floor + vol effects)
    ctrl_v2 = GARCHPositionController(make_v2_config(cooldown_periods=0))
    ctrl_v2._nav_peak = nav_peak
    scale_v2, vol_v2, info_v2 = ctrl_v2.compute(weights, returns, nav)

    print(f"  年化波动率: v1.0={vol_v1:.4f} ({vol_v1*100:.2f}%), v2.0={vol_v2:.4f} ({vol_v2*100:.2f}%)")
    print(f"  波动率差异: {vol_v2 - vol_v1:+.4f} ({(vol_v2-vol_v1)/vol_v1*100:+.1f}%)")
    print(f"  v2.0 method: {info_v2['method']}")
    print(f"  v2.0 ewma_vol: {info_v2.get('ewma_vol',0)*np.sqrt(252):.4f}")
    print(f"\n  仓位: v1.0={scale_v1:.4f}, v2.0={scale_v2:.4f}")
    print(f"  dd_scale: v1.0={info_v1['dd_scale']:.4f}, v2.0={info_v2['dd_scale']:.4f}")

    # v2.0 vol should be >= v1.0 (Student-t + EWMA cross-check)
    check(vol_v2 >= vol_v1 * 0.95, "v2.0 崩盘场景波动率 >= v1.0",
          f"(v2={vol_v2:.4f} >= v1={vol_v1:.4f})")

    # 18% drawdown: v2.0 secondary floor (15% threshold) should apply
    check(info_v2['dd_scale'] <= 0.15 + 0.01, "v2.0 18%回撤触发二级地板",
          f"(dd_scale={info_v2['dd_scale']:.4f} <= 0.15)")

    # v2.0 should be more conservative due to secondary floor
    check(scale_v2 <= scale_v1 + 0.01, "v2.0 崩盘场景仓位 <= v1.0 (二级地板更保守)",
          f"(v2={scale_v2:.4f} <= v1={scale_v1:.4f})")

    return {'v1_scale': scale_v1, 'v2_scale': scale_v2, 'v1_vol': vol_v1, 'v2_vol': vol_v2}


def test_11_regression_normal_market():
    """测试11: 回归测试 — 正常市场行为不受影响"""
    print_header("TEST 11: 回归测试 — 正常市场（无回撤）")

    N, T = 20, 250
    returns = gen_correlated_returns(T, N, daily_vol=0.012, corr=0.6, seed=42)
    weights = np.ones(N) / N
    nav = 1_000_000

    ctrl_v2 = GARCHPositionController(make_v2_config())
    scale, vol, info = ctrl_v2.compute(weights, returns, nav)

    print(f"  scale={scale:.4f}, vol={vol:.4f} ({vol*100:.2f}%), method={info['method']}")
    print(f"  dd_scale={info['dd_scale']:.4f}, cooldown={info['cooldown_active']}")

    check(abs(info['dd_scale'] - 1.0) < 0.01, "无回撤时dd_scale=1.0",
          f"(dd_scale={info['dd_scale']:.4f})")
    check(scale >= 0.8, "正常市场高仓位", f"(scale={scale:.4f})")
    check(not info['cooldown_active'], "正常市场无冷却", "")

    return scale


def test_12_regression_insufficient_data():
    """测试12: 回归测试 — 数据不足"""
    print_header("TEST 12: 回归测试 — 数据不足(<60天)")

    cfg = make_v2_config()
    ctrl = GARCHPositionController(cfg)

    N, T = 20, 30
    returns = gen_correlated_returns(T, N, daily_vol=0.012, corr=0.6, seed=42)
    weights = np.ones(N) / N

    scale, vol, info = ctrl.compute(weights, returns, 1_000_000)

    print(f"  scale={scale}, vol={vol}, method={info['method']}")
    check(scale == 1.0, "数据不足时返回全仓", f"(scale={scale})")
    check(info['method'] == 'insufficient_data', "method=insufficient_data", "")

    return scale


def test_13_v2_full_features_trajectory():
    """
    测试13: v2.0全功能轨迹 — 极端崩盘+恢复
    """
    print_header("TEST 13: v2.0全功能极端场景轨迹")

    N, T = 20, 250
    returns = gen_correlated_returns(T, N, daily_vol=0.025, corr=0.7, seed=42)
    weights = np.ones(N) / N

    navs = np.concatenate([
        np.linspace(1_000_000, 780_000, 8),   # 22% drawdown
        np.linspace(780_000, 950_000, 6),     # partial recovery
        np.linspace(950_000, 1_050_000, 6),   # full recovery
    ])

    ctrl = GARCHPositionController(make_v2_config())

    print(f"\n  {'周期':>4} | {'NAV':>10} | {'dd%':>6} | {'vol_sc':>7} | {'dd_sc':>7} | {'final':>7} | {'cd':>3} | {'unlock':>7} | {'method':>10}")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*3}-+-{'-'*7}-+-{'-'*10}")

    results = []
    for i, nav in enumerate(navs):
        scale, vol, info = ctrl.compute(weights, returns, nav)
        results.append({
            'period': i, 'nav': nav, 'dd': info['current_drawdown'],
            'vol_scale': info['vol_scale'], 'dd_scale': info['dd_scale'],
            'final': scale, 'cooldown': info['cooldown_active'],
            'unlock': info.get('cooldown_unlock', 0),
            'method': info['method'],
        })
        print(f"  {i:>4} | {nav:>10,.0f} | {info['current_drawdown']*100:>5.1f}% | "
              f"{info['vol_scale']:>7.4f} | {info['dd_scale']:>7.4f} | {scale:>7.4f} | "
              f"{'Y' if info['cooldown_active'] else 'N':>3} | "
              f"{info.get('cooldown_unlock',0):>7.2f} | {info['method']:>10}")

    trough = min(r['final'] for r in results)
    trough_period = [r for r in results if r['final'] == trough][0]
    recovery = results[-1]['final']

    print(f"\n  --- 分析 ---")
    print(f"  谷底仓位 = {trough:.4f} (周期{trough_period['period']}, dd={trough_period['dd']*100:.1f}%)")
    print(f"  恢复后仓位 = {recovery:.4f}")

    # With 22% drawdown, tertiary floor (0.05) should be active at trough
    check(trough <= 0.10, "22%回撤时仓位极低 (三级地板生效)",
          f"(trough={trough:.4f})")

    # Recovery should be gradual (not V-shape)
    mid_recovery = results[11]['final']
    late_recovery = results[15]['final']
    check(late_recovery >= mid_recovery, "恢复期仓位单调递增",
          f"(period11={mid_recovery:.4f} <= period15={late_recovery:.4f})")

    return results


def test_14_ewma_cross_check_disabled():
    """测试14: EWMA交叉验证可关闭"""
    print_header("TEST 14: EWMA交叉验证可关闭")

    T = 250
    rets = gen_volatility_clustering(T, seed=42)
    returns = rets.reshape(-1, 1)
    weights = np.array([1.0])

    ctrl_on = GARCHPositionController(make_v2_config(use_ewma_cross_check=True, use_drawdown_control=False))
    _, vol_on, info_on = ctrl_on.compute(weights, returns, 1_000_000)

    ctrl_off = GARCHPositionController(make_v2_config(use_ewma_cross_check=False, use_drawdown_control=False))
    _, vol_off, info_off = ctrl_off.compute(weights, returns, 1_000_000)

    print(f"  交叉验证开启: vol={vol_on:.4f}, method={info_on['method']}")
    print(f"  交叉验证关闭: vol={vol_off:.4f}, method={info_off['method']}")

    check('xval' not in info_off['method'], "关闭后method不含xval标记", "")

    return True


def test_15_config_backward_compatibility():
    """测试15: 向后兼容 — 旧config（无v2.0参数）仍可运行"""
    print_header("TEST 15: 向后兼容 — 旧config")

    old_config = {
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

    ctrl = GARCHPositionController(old_config)
    N, T = 20, 250
    returns = gen_correlated_returns(T, N, daily_vol=0.012, corr=0.6, seed=42)
    weights = np.ones(N) / N

    scale, vol, info = ctrl.compute(weights, returns, 1_000_000)
    print(f"  scale={scale:.4f}, vol={vol:.4f}, method={info['method']}")
    print(f"  garch_dist={ctrl.garch_dist} (default='t')")
    print(f"  use_ewma_cross_check={ctrl.use_ewma_cross_check} (default=True)")
    print(f"  use_gradual_cooldown={ctrl.use_gradual_cooldown} (default=True)")
    print(f"  secondary_dd_threshold={ctrl.secondary_dd_threshold} (default=0.15)")

    check(ctrl.garch_dist == 't', "默认garch_dist='t'", "")
    check(ctrl.use_ewma_cross_check == True, "默认use_ewma_cross_check=True", "")
    check(ctrl.use_gradual_cooldown == True, "默认use_gradual_cooldown=True", "")
    check(ctrl.secondary_dd_threshold == 0.15, "默认secondary_dd_threshold=0.15", "")
    check(scale > 0, "旧config正常运行", f"(scale={scale:.4f})")

    return scale


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("  GARCH 仓位控制 v2.0 — 改进验证测试套件")
    print("  Alpha191 Strategy Risk Control Improvement Verification")
    print("=" * 70)

    results = {}
    results['t1'] = test_1_student_t_vs_normal_fat_tail()
    results['t2'] = test_2_student_t_vs_normal_crash()
    results['t3'] = test_3_ewma_cross_validation()
    results['t4'] = test_4_cooldown_gradual_unlock()
    results['t5'] = test_5_cooldown_episode_tracking()
    results['t6'] = test_6_secondary_dd_floor()
    results['t7'] = test_7_tertiary_dd_floor()
    results['t8'] = test_8_multi_tier_dd_sweep()
    results['t9'] = test_9_full_simulation_v2()
    results['t10'] = test_10_crash_scenario_comprehensive()
    results['t11'] = test_11_regression_normal_market()
    results['t12'] = test_12_regression_insufficient_data()
    results['t13'] = test_13_v2_full_features_trajectory()
    results['t14'] = test_14_ewma_cross_check_disabled()
    results['t15'] = test_15_config_backward_compatibility()

    print("\n\n" + "=" * 70)
    print(f"  测试总结: {PASS_COUNT} PASS / {FAIL_COUNT} FAIL / {PASS_COUNT+FAIL_COUNT} TOTAL")
    print("=" * 70)

    if FAIL_COUNT == 0:
        print("  *** 所有测试通过 ***")
    else:
        print(f"  *** {FAIL_COUNT} 个测试失败，请检查 ***")

    return results


if __name__ == '__main__':
    main()
