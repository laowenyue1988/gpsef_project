"""
GARCH Position Controller for Alpha191 (v3.2).
Holding Volatility-based position scaling.

Methodology:
  1. Build synthetic portfolio returns: r_portfolio = sum(w_i * r_i) for past T days
  2. Fit EGARCH(1,1) (default) or GARCH(1,1) with Student-t distribution
  3. Scale: exposure = min(max_scale, max(min_scale, target_vol / annualized_vol))
  4. Multi-tier drawdown overlay (aligned with SignalPilot/HyperQuant):
     - Tier 1 (concave sqrt): dd_scale = sqrt(1 - dd/dd_limit), min 0.25
     - Tier 2 (hard floor): dd >= 15% -> floor = 0.50
     - Tier 3 (hard floor): dd >= 20% -> floor = 0.25  
     - Kill-Switch: dd >= 30% -> floor = 0.00
       Release: time lock (N periods) + nav_peak reset = automatic restart
  5. Cooldown with Sigmoid unlock + compressed recovery

v3.2 Improvements over v3.1:
  - Kill-Switch deadlock fix: time-based lock + nav_peak reset (Solution D)
  - Hard floors aligned with industry: T2 0.15→0.50, T3 0.05→0.25
  - Recovery buffer compressed: stages 2→1, periods 2→1
  - Reduced lock time for alpha-confidence strategies
     - Data quality: ADF stationarity + ARCH-LM effect + Winsorize outliers
  3. Scale: exposure = min(max_scale, max(min_scale, target_vol / annualized_vol))
  4. Apply multi-tier drawdown overlay:
     - Tier 1 (concave sqrt): dd_scale = sqrt(1 - dd/dd_limit), min 0.25
       Concave shape: shallow DD → minimal reduction, deep DD → accelerated protection
     - Tier 2 (hard floor): dd >= 15% -> floor = 0.15
     - Tier 3 (hard floor): dd >= 20% -> floor = 0.05
     - Kill-switch: dd >= 30% -> floor = 0.0 (complete flatten)
  5. Cooldown with Sigmoid unlock + compressed recovery:
     - Cooldown triggers at 75% of drawdown_limit (15% DD) — v3.1 improvement
     - 2 cooldown + 2 buffer = 4 total recovery periods (was 7)

v3.0 Improvements over v2.0:
  - EGARCH(1,1) with leverage effect capture (Nelson 1991)
  - Kill-switch at 30% drawdown (complete position flattening)
  - Sigmoid cooldown unlock (S-curve) replacing linear
  - Post-cooldown recovery buffer (staged recovery)
  - ADF stationarity test before volatility model fitting
  - ARCH-LM test for heteroskedasticity verification
  - Winsorize outlier returns (0.5%/99.5% quantile)

References:
  De Prado, AFML (2018) Ch.3
  RiskMetrics (1996) - EWMA covariance
  Engle (1982/2001) - ARCH/GARCH
  Bollerslev (1987) - Student-t GARCH
  Nelson (1991) - EGARCH leverage effect
  Cont (2001) - Stylized facts
  MCPA (2024) - Production GARCH best practices
  SignalPilot (2024) - Drawdown recovery protocol

All parameters configurable via config.yaml / garch_control section.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Graceful imports
try:
    from arch import arch_model
    _ARCH_AVAILABLE = True
except ImportError:
    _ARCH_AVAILABLE = False
    logger.warning(
        "arch library not installed. GARCH/EGARCH path unavailable. "
        "Strategy will use EWMA fallback. To enable: pip install arch"
    )

try:
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import het_arch as arch_lm_test
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False
    logger.warning(
        "statsmodels not installed. ADF/ARCH-LM data quality checks disabled. "
        "To enable: pip install statsmodels"
    )


class GARCHPositionController:
    """
    Holding Volatility-based position scaling (v3.1).

    v3.1 Improvements over v3.0:
      - Concave drawdown function sqrt(1-dd/dd_limit): less reduction at shallow DD
      - Cooldown trigger raised: 10% -> 15% (0.75 × drawdown_limit)
      - Recovery compressed: cooldown 2 + buffer 2 = 4 total (was 7)
      - Industry alignment: vol-driven dominates shallow DD, DD-driven kicks in at 15%+

    Improvements over v2.0:
      - EGARCH(1,1)-t with leverage effect (Nelson 1991): negative shocks better captured
      - Kill-switch at 30%+ drawdown: complete position flattening
      - Sigmoid cooldown unlock (S-curve): smoother recovery profile
      - Post-cooldown recovery buffer: staged recovery (50%->75%->100%)
      - ADF stationarity test: reject non-stationary returns before GARCH fitting
      - ARCH-LM test: verify heteroskedasticity before fitting
      - Winsorize extreme returns: 0.5%/99.5% quantile clipping

    On each rebalance day:
      1. Build synthetic portfolio returns: r_portfolio = sum(w_i * r_i) for past T days
      2. Data quality checks: ADF + ARCH-LM + Winsorize
      3. Fit EGARCH/GARCH(1,1)-t, cross-validate with EWMA
      4. Scale: exposure = target_vol / annualized_vol, clipped to [min_scale, max_scale]
      5. Apply multi-tier drawdown overlay + kill-switch
      6. Cooldown with Sigmoid unlock + recovery buffer
    """

    # ============================================================
    # GARCH Position Controller for Alpha191 (v3.1).
    # Holding Volatility-based position scaling.
    #
    # v3.1 Improvements over v3.0:
    #   - Concave drawdown function (sqrt) replacing linear: less reduction at shallow DD
    #   - Cooldown trigger raised: 10% -> 15% (75% of drawdown_limit)
    #   - Recovery period compressed: cooldown 3->2, buffer 4->2, stages 3->2
    #   - Industry alignment: vol-driven dominates shallow DD, DD-driven kicks in at 15%+
    # ============================================================

    def __init__(self, config: dict):
        # Core parameters
        self.vol_target = config.get('vol_target', 0.15)
        self.lookback = config.get('lookback', 250)
        self.vol_method = config.get('vol_method', 'garch')      # 'garch' | 'ewma'
        self.vol_model = config.get('vol_model', 'EGARCH')      # 'Garch' | 'EGARCH' | 'GJR-GARCH'
        self.ewma_lambda = config.get('ewma_lambda', 0.94)
        self.min_scale = config.get('min_scale', 0.0)             # v3.3: 0.0 default
        self.max_scale = config.get('max_scale', 1.0)
        self.drawdown_limit = config.get('drawdown_limit', 0.20)
        self.use_drawdown_control = config.get('use_drawdown_control', True)
        self.cooldown_periods = config.get('cooldown_periods', 2)  # v3.1: 3->2

        # --- v2.0 Parameters (retained) ---
        self.garch_dist = config.get('garch_dist', 't')
        self.use_ewma_cross_check = config.get('use_ewma_cross_check', True)
        self.secondary_dd_threshold = config.get('secondary_dd_threshold', 0.15)
        self.secondary_dd_floor = config.get('secondary_dd_floor', 0.50)
        self.tertiary_dd_threshold = config.get('tertiary_dd_threshold', 0.20)
        self.tertiary_dd_floor = config.get('tertiary_dd_floor', 0.25)

        # --- v3.4: Recovery buffer (replaces cooldown) ---
        # Buffer activates when DD drops from >=15% back below 15%
        # It gradually eases position from the hard floor (0.50) to concave function
        self.recovery_buffer_periods = config.get('recovery_buffer_periods', 3)
        self.use_recovery_buffer = config.get('use_recovery_buffer', True)

        # --- v3.0 Parameters ---
        self.vol_model = config.get('vol_model', 'EGARCH')

        # Kill-switch
        self.kill_switch_threshold = config.get('kill_switch_threshold', 0.30)
        self.kill_switch_floor = config.get('kill_switch_floor', 0.0)
        self.use_kill_switch = config.get('use_kill_switch', True)
        self.kill_switch_lock_periods = config.get('kill_switch_lock_periods', 5)

        # --- v3.1 Parameters ---
        self.dd_function = config.get('dd_function', 'concave')

        # Data quality
        self.use_adf_test = config.get('use_adf_test', True)
        self.adf_pvalue_threshold = config.get('adf_pvalue_threshold', 0.05)
        self.use_arch_lm_test = config.get('use_arch_lm_test', True)
        self.arch_lm_pvalue_threshold = config.get('arch_lm_pvalue_threshold', 0.05)
        self.use_winsorize = config.get('use_winsorize', True)
        self.winsorize_lower = config.get('winsorize_lower', 0.005)
        self.winsorize_upper = config.get('winsorize_upper', 0.995)

        # Internal state
        self._nav_peak = None
        self._last_scale = 1.0
        self._last_dd = 0.0
        # [AUDIT-FIX 2026-09-14] 上一期的 dd_scale（回撤叠加项本身），
        #   供恢复缓冲取"从哪个地板开始恢复" —— 原实现误用 _last_scale
        #   （含 vol_scale 与 min_scale 夹取），语义偏移。
        self._last_dd_scale = 1.0
        # v3.4: Recovery buffer state (replaces cooldown)
        self._buffer_remaining = 0
        self._buffer_total = 0
        self._buffer_from = 1.0
        self._kill_switch_active = False
        self._kill_switch_lock = 0

    def compute(self, weights, returns, portfolio_value):
        """
        Compute position scaling factor.

        Args:
            weights: (N,) ndarray of target portfolio weights for N stocks
            returns: (T, N) ndarray of past T days individual stock returns
            portfolio_value: current portfolio total value

        Returns:
            (scale, annual_vol, info_dict)
        """
        # ---- Step 0: Input validation ----
        T, N = returns.shape if len(returns.shape) == 2 else (len(returns), 1)
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1) if N == 1 else weights

        min_obs = 60
        if T < min_obs or N == 0:
            return self.max_scale, 0.0, {
                'method': 'insufficient_data', 'daily_vol': 0.0,
                'converged': False, 'cooldown_active': False,
                'current_drawdown': 0.0, 'dd_scale': 1.0,
                'vol_scale': 1.0, 'annual_vol': 0.0,
                'cooldown_remaining': 0, 'cooldown_unlock': 0.0,
                'ewma_vol': 0.0, 'kill_switch_active': False,
                'adf_pvalue': 1.0, 'arch_lm_pvalue': 1.0,
            }

        # ---- Step 1: Synthetic portfolio returns (Holding Volatility) ----
        # [E2-R1 2026-09-09] NaN 安全加权: 每日按有效(非NaN)股票权重归一
        #   (原 port_returns = returns @ weights: NaN 传播 + 零填充低估波动)
        w_flat = weights.flatten()[:N].astype(float)
        valid = ~np.isnan(returns)
        w_mat = np.broadcast_to(w_flat, returns.shape).copy()
        w_mat[~valid] = 0.0
        denom = w_mat.sum(axis=1)
        port_returns = np.full(T, np.nan)
        with np.errstate(invalid='ignore', divide='ignore'):
            port_returns = np.nansum(returns * w_mat, axis=1) / np.where(denom > 0, denom, 1.0)
        port_returns[denom <= 0] = np.nan  # 整行无有效股票 → NaN
        port_returns = port_returns[~np.isnan(port_returns)]  # 剔除该日
        if len(port_returns) < min_obs:
            return self.max_scale, 0.0, {
                'method': 'insufficient_data', 'daily_vol': 0.0,
                'converged': False, 'cooldown_active': False,
                'current_drawdown': 0.0, 'dd_scale': 1.0,
                'vol_scale': 1.0, 'annual_vol': 0.0,
                'cooldown_remaining': 0, 'cooldown_unlock': 0.0,
                'ewma_vol': 0.0, 'kill_switch_active': False,
                'adf_pvalue': 1.0, 'arch_lm_pvalue': 1.0,
            }

        # ---- Step 1b: Data quality checks (v3.0) ----
        adf_pvalue = 1.0
        arch_lm_pvalue = 1.0
        data_quality_ok = True

        # Winsorize extreme returns
        if self.use_winsorize and len(port_returns) >= 60:
            lo, hi = np.percentile(port_returns, [
                self.winsorize_lower * 100, self.winsorize_upper * 100
            ])
            n_clipped = np.sum((port_returns < lo) | (port_returns > hi))
            if n_clipped > 0:
                logger.debug(f'[数据质量] Winsorize: {n_clipped}/{T} 天被裁剪 '
                             f'(lo={lo:.4f}, hi={hi:.4f})')
            port_returns = np.clip(port_returns, lo, hi)

        # ADF stationarity test
        if self.use_adf_test and _STATSMODELS_AVAILABLE and len(port_returns) >= 60:
            try:
                adf_result = adfuller(port_returns, maxlag=int(T ** 0.5))
                adf_pvalue = adf_result[1]
                if adf_pvalue > self.adf_pvalue_threshold:
                    logger.warning(f'[数据质量] ADF测试未通过 (p={adf_pvalue:.4f} > '
                                   f'{self.adf_pvalue_threshold}), 降级EWMA')
                    data_quality_ok = False
            except Exception:
                logger.debug('[数据质量] ADF测试异常，跳过')

        # ARCH-LM heteroskedasticity test
        if self.use_arch_lm_test and _STATSMODELS_AVAILABLE and len(port_returns) >= 60:
            try:
                lm_result = arch_lm_test(port_returns)
                arch_lm_pvalue = lm_result[1]
                if arch_lm_pvalue > self.arch_lm_pvalue_threshold:
                    logger.info(f'[数据质量] ARCH-LM未通过 (p={arch_lm_pvalue:.4f} > '
                                f'{self.arch_lm_pvalue_threshold}), '
                                '无显著ARCH效应，降级EWMA')
                    data_quality_ok = False
            except Exception:
                logger.debug('[数据质量] ARCH-LM测试异常，跳过')

        # ---- Step 2: Estimate portfolio volatility ----
        ewma_vol = 0.0
        if self.vol_method == 'ewma':
            daily_vol, converged, method = self._compute_ewma_vol(port_returns)
            ewma_vol = daily_vol
        elif not data_quality_ok:
            # Data quality failed -> fallback to EWMA
            daily_vol, converged, method = self._compute_ewma_vol(port_returns)
            ewma_vol = daily_vol
            method = 'ewma_dq'  # mark as data quality fallback
        else:
            daily_vol, converged, method = self._compute_arch_vol(port_returns)
            ewma_vol, _, _ = self._compute_ewma_vol(port_returns)

            if self.use_ewma_cross_check and method.startswith(('garch', 'egarch')):
                if ewma_vol > daily_vol:
                    method = method + '_xval'
                    daily_vol = ewma_vol

        # Clamp to reasonable range
        daily_vol = max(daily_vol, 1e-8)
        annual_vol = daily_vol * np.sqrt(252)
        # [v3.4 FIX] Safety net: if annual_vol exceeds 1000% (clearly anomalous), fallback to EWMA
        if annual_vol > 10.0:
            logger.error(f'GARCH annual_vol异常({annual_vol:.1f}%), 降级EWMA')
            daily_vol, converged, method = self._compute_ewma_vol(port_returns)
            ewma_vol = daily_vol
            annual_vol = daily_vol * np.sqrt(252)

        # ---- Step 3: Volatility scaling ----
        vol_scale = min(self.max_scale, max(self.min_scale,
                        self.vol_target / max(annual_vol, 1e-8)))

        # ---- Step 4: Drawdown overlay (v3.4 — stair-step hard floors, no cooldown) ----
        dd_scale = 1.0
        current_dd = 0.0
        kill_switch_active = self._kill_switch_active

        if self.use_drawdown_control and portfolio_value > 0:
            if self._nav_peak is None or portfolio_value > self._nav_peak:
                self._nav_peak = portfolio_value
            current_dd = max(0.0, 1.0 - portfolio_value / self._nav_peak)

            if current_dd > 0:
                # Tier 1: Concave function for DD < 15% (smooth, no over-protection)
                if current_dd < self.secondary_dd_threshold:
                    if self.dd_function == 'concave':
                        dd_scale = np.sqrt(max(0.0, 1.0 - current_dd / self.drawdown_limit))
                    else:
                        dd_scale = 1.0 - current_dd / self.drawdown_limit

                # Tier 2: Hard floor at DD >= 15%
                if current_dd >= self.secondary_dd_threshold:
                    dd_scale = min(dd_scale, self.secondary_dd_floor)

                # Tier 3: Hard floor at DD >= 20%
                if current_dd >= self.tertiary_dd_threshold:
                    dd_scale = min(dd_scale, self.tertiary_dd_floor)

                # Kill-switch: DD >= 30%
                if self.use_kill_switch and current_dd >= self.kill_switch_threshold:
                    dd_scale = min(dd_scale, self.kill_switch_floor)

        # ---- Step 5: Recovery buffer (v3.4 — activates when DD drops back below 15%) ----
        # When DD transitions from >=15% back to <15%, the position needs to gradually
        # ease back from the hard floor (0.50) to the concave function value.
        # Without buffer: position jumps 0.50→0.775 instantly — too sudden after a crash.
        # With buffer: position increases over buffer_periods cycles — validates recovery.
        #
        # [AUDIT-FIX 2026-09-14] 原实现缺陷（经 test_garch_position_audit.py 实证）：
        #   触发条件 `was_in_hard_floor and now_recovering` 仅对**过渡首期**成立；
        #   次期起 _last_dd 已 < 15% ⇒ 条件恒为 False ⇒ 缓冲被中断（实际只生效 1 期），
        #   且 _buffer_remaining 冻结在首期值（状态污染）。
        #   现改为「过渡首期触发 + 计数驱动直至耗尽 + 重回硬地板区即取消」。
        buffer_active = False
        buffer_progress = 0.0

        # Detect transition: DD now < 15% AND last period DD was >= 15%
        was_in_hard_floor = (self._last_dd >= self.secondary_dd_threshold - 1e-9)
        now_recovering = (current_dd < self.secondary_dd_threshold - 1e-9)

        if self.use_recovery_buffer:
            if was_in_hard_floor and now_recovering:
                if self._buffer_remaining <= 0:
                    # Fresh recovery: start a new buffer period
                    self._buffer_remaining = self.recovery_buffer_periods
                    self._buffer_total = self.recovery_buffer_periods
                    # [AUDIT-FIX] 起点 = 上一期的 dd_scale（硬地板值）
                    self._buffer_from = self._last_dd_scale
            elif not now_recovering:
                # 重新进入硬地板区 ⇒ 取消未完成的缓冲，避免与阶梯保护互相干扰
                self._buffer_remaining = 0
                self._buffer_total = 0

            if self._buffer_remaining > 0:
                buffer_active = True
                if self._buffer_total > 0:
                    periods_elapsed = self._buffer_total - self._buffer_remaining
                    buffer_progress = min(1.0, max(0.0, (periods_elapsed + 1) / self._buffer_total))
                self._buffer_remaining -= 1
                if self._buffer_remaining < 0:
                    self._buffer_remaining = 0

        if buffer_active and self._buffer_total > 0:
            # Blend: gradually move from floor toward concave function
            # [AUDIT-FIX] 原 line 367 的 `min(dd_scale, dd_scale)` 为恒等死代码，已移除；
            #   单调性由 buffer_progress ∈ [0,1] 的线性混合保证。
            dd_scale = self._buffer_from + (dd_scale - self._buffer_from) * buffer_progress

        # ---- Step 5b: Kill-switch trigger + time lock ----
        if self.use_kill_switch and current_dd >= self.kill_switch_threshold - 1e-9:
            if not self._kill_switch_active:
                self._kill_switch_active = True
                self._kill_switch_lock = self.kill_switch_lock_periods
                kill_switch_active = True
                logger.warning(f'[Kill-Switch] 触发! DD={current_dd:.2%} 锁定{self.kill_switch_lock_periods}期')

        if self._kill_switch_active:
            kill_switch_active = True
            if self._kill_switch_lock > 0:
                self._kill_switch_lock -= 1
            if self._kill_switch_lock <= 0:
                self._kill_switch_active = False
                kill_switch_active = False
                self._nav_peak = portfolio_value
                self._buffer_remaining = 0
                logger.info(f'[Kill-Switch] 锁定期满, nav_peak重置, 策略重启')

        # ---- Step 6: Final scale ----
        final_scale = min(vol_scale, dd_scale)
        final_scale = min(self.max_scale, max(self.min_scale, final_scale))

        self._last_scale = final_scale
        self._last_dd = current_dd
        # [AUDIT-FIX 2026-09-14] 记录本期 dd_scale，作为下期恢复缓冲的起点
        self._last_dd_scale = dd_scale

        # [v3.4] 结构化仓位控制日志
        tier2_active = current_dd >= self.secondary_dd_threshold
        tier3_active = current_dd >= self.tertiary_dd_threshold
        buf_active = self._buffer_remaining > 0 and current_dd < self.secondary_dd_threshold
        if buf_active and self._buffer_total > 0:
            buf_progress = (self._buffer_total - self._buffer_remaining) / self._buffer_total
        else:
            buf_progress = 0.0
        logger.info(
            f"[仓位控制] vol_scale={vol_scale:.3f}(σ={annual_vol:.1%},method={method}) "
            f"dd_scale={dd_scale:.3f}(DD={current_dd:.1%},"
            f"{'T2' if tier2_active else ''}{'T3' if tier3_active else ''}{'KS' if kill_switch_active else ''}) "
            f"buf={'ACTIVE' if buf_active else '---'}({buf_progress:.0%}) "
            f"→ final={final_scale:.3f}"
        )

        info = {
            'method': method,
            'daily_vol': daily_vol,
            'annual_vol': annual_vol,
            'vol_scale': vol_scale,
            'current_drawdown': current_dd,
            'dd_scale': dd_scale,
            'final_scale': final_scale,
            'converged': converged,
            'cooldown_active': False,
            'cooldown_remaining': 0,
            'cooldown_unlock': 0.0,
            'ewma_vol': ewma_vol if self.vol_method != 'ewma' else daily_vol,
            'kill_switch_active': kill_switch_active,
            'adf_pvalue': adf_pvalue,
            'arch_lm_pvalue': arch_lm_pvalue,
            'recovery_stage': 1 if buffer_active else 0,
            'buffer_progress': buffer_progress,
            'buffer_active': buffer_active,
        }
        return final_scale, annual_vol, info

    def _compute_ewma_vol(self, returns):
        """EWMA volatility (RiskMetrics lambda=0.94)."""
        n = len(returns)
        weights = (1 - self.ewma_lambda) * np.power(self.ewma_lambda, np.arange(n)[::-1])
        weights = weights / weights.sum()
        weighted_var = np.sum(weights * (returns ** 2))
        return np.sqrt(max(weighted_var, 1e-16)), False, 'ewma'

    def _compute_arch_vol(self, returns):
        """
        EGARCH/GARCH(1,1) conditional volatility with Student-t distribution.

        v3.0: Default EGARCH captures leverage effect (Nelson 1991).
              Falls back to EWMA on convergence failure or data quality issues.

        Returns:
            (daily_vol, converged, method)
        """
        if not _ARCH_AVAILABLE:
            return self._compute_ewma_vol(returns)

        try:
            ret_centered = returns - np.mean(returns)
            scale_factor = np.std(ret_centered)
            if scale_factor < 1e-8 or np.isnan(scale_factor):
                # near_zero -> fallback to EWMA (v3.0 fix)
                daily_vol, _, _ = self._compute_ewma_vol(returns)
                return daily_vol, False, 'ewma_nz'

            ret_scaled = ret_centered / scale_factor * 100

            model = arch_model(
                ret_scaled,
                vol=self.vol_model,       # 'EGARCH' (default), 'Garch', 'GJR-GARCH'
                p=1, q=1,
                dist=self.garch_dist,     # 't' (Student-t) or 'normal'
                rescale=False,
            )
            result = model.fit(disp='off', show_warning=False,
                               options={'maxiter': 500})

            # Monitor GARCH persistence (for EGARCH, alpha+beta approximate)
            if self.vol_model in ('Garch', 'GJR-GARCH') and 'alpha[1]' in result.params:
                persistence = result.params['alpha[1]'] + result.params.get('beta[1]', 0)
                if persistence > 0.99:
                    logger.warning(f'GARCH persistence={persistence:.4f}, '
                                   '接近IGARCH, 可能非平稳')

            # Monitor Student-t degrees of freedom stability
            if 'nu' in result.params:
                nu = result.params['nu']
                if nu > 30:
                    logger.debug(f'GARCH nu={nu:.1f} > 30, t分布接近正态')
                elif nu < 2.5:
                    logger.warning(f'GARCH nu={nu:.1f} < 2.5, 方差可能不存在')

            forecast = result.forecast(horizon=1)
            cond_var = forecast.variance.values[-1, 0]

            if np.isnan(cond_var) or cond_var <= 0:
                raise ValueError("Invalid forecast")

            daily_vol = np.sqrt(cond_var) / 100 * scale_factor
            # [v3.4 FIX] Upper bound: prevent EGARCH numerical overflow
            # EGARCH's exp(ln(σ²)) can produce astronomically large values when
            # parameters become unstable. Clamp to reasonable max: 50% daily (~800% annual).
            daily_vol = min(daily_vol, 0.50)
            if daily_vol > 0.30:
                logger.warning(f'[{self.vol_model}] EGARCH daily_vol极高({daily_vol:.4f}), 钳制到0.50')
            return max(daily_vol, 1e-8), True, 'egarch' if self.vol_model == 'EGARCH' else 'garch'

        except Exception as e:
            logger.debug(f'ARCH模型拟合失败 ({type(e).__name__}), 降级EWMA')
            daily_vol, _, _ = self._compute_ewma_vol(returns)
            return daily_vol, False, 'ewma_fallback'
