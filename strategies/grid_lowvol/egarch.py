# -*- coding: utf-8 -*-
"""EGARCH(1,1) 波动率预测: 正态 MLE 估计 + h 步累计波动预测。

模型 (Nelson 1991):
    ln σ_t² = ω + α(|z_{t-1}| - √(2/π)) + γ z_{t-1} + β ln σ_{t-1}²,  z_t = r_t/σ_t
    - γ<0: 负收益(利空)比正收益更推升波动 (杠杆效应, A股下跌市波动集聚);
    - β∈(0.9,0.999): 长记忆/平稳性约束;
    - 正态 MLE: ℓ = ½ Σ (ln σ_t² + z_t²) (省略常数);
h 步预测 (长记忆近似):
    E[ln σ_{t+k}²] = ω + β·E[ln σ_{t+k-1}²] → σ_{t+k}² = exp(·)
    累计波动: Σ_{t+1..t+h} = sqrt(Σ_k σ_{t+k}²)   (选股排序输入)

性能: 单只 ~30ms (scipy L-BFGS-B, 250日窗口); 全市场需两阶段 (RV初筛→候选池, 见 strategy)。
"""
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SQRT_2_PI = float(np.sqrt(2.0 / np.pi))

try:
    from numba import njit

    @njit(cache=True)
    def _egarch_filter_nb(r, w, a, g, b):
        """EGARCH 条件方差递推 (numba, 纯循环)。返回 sigma2 序列。"""
        n = len(r)
        sigma2 = np.empty(n)
        s2 = float(np.var(r)) if n > 1 else 1e-6
        for t in range(n):
            if t > 0:
                s2 = np.exp(w + a * (abs(r[t - 1]) - SQRT_2_PI) + g * r[t - 1]
                            + b * np.log(s2))
            else:
                s2 = np.exp(w + b * np.log(max(s2, 1e-12)))
            sigma2[t] = max(s2, 1e-12)
        return sigma2

    @njit(cache=True)
    def _egarch_negloglik_nb(r, w, a, g, b):
        """EGARCH 负对数似然 (numba)。"""
        n = len(r)
        s2 = float(np.var(r)) if n > 1 else 1e-6
        ll = 0.0
        for t in range(n):
            if t > 0:
                s2 = np.exp(w + a * (abs(r[t - 1]) - SQRT_2_PI) + g * r[t - 1]
                            + b * np.log(s2))
            else:
                s2 = np.exp(w + b * np.log(max(s2, 1e-12)))
            s2 = max(s2, 1e-12)
            ll += np.log(s2) + (r[t] * r[t]) / s2
        return 0.5 * ll

    _HAS_NB = True
except Exception:  # pragma: no cover
    _HAS_NB = False


class EGARCHModel:
    """EGARCH(1,1) 单只股票估计器 (numba 加速)。"""

    def __init__(self, window: int = 250, max_iter: int = 200, tol: float = 1e-6):
        self.window = int(window)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.params: Optional[np.ndarray] = None      # [ω, α, γ, β]
        self.last_sigma2: float = np.nan              # 最近一期条件方差
        self.converged: bool = False

    # ------------------------------------------------------------------
    def _neg_loglik(self, p: np.ndarray, r: np.ndarray) -> float:
        w, a, g, b = p
        if _HAS_NB:
            return float(_egarch_negloglik_nb(r, w, a, g, b))
        return self._neg_loglik_py(p, r)

    def _neg_loglik_py(self, p: np.ndarray, r: np.ndarray) -> float:
        w, a, g, b = p
        n = len(r)
        s2 = float(np.var(r)) if n > 1 else 1e-6
        ll = 0.0
        for t in range(n):
            if t > 0:
                s2 = np.exp(w + a * (abs(r[t - 1]) - SQRT_2_PI) + g * r[t - 1]
                            + b * np.log(s2))
            else:
                s2 = np.exp(w + b * np.log(max(s2, 1e-12)))
            s2 = max(s2, 1e-12)
            ll += np.log(s2) + (r[t] ** 2) / s2
        return 0.5 * ll

    def fit(self, returns: np.ndarray) -> bool:
        """用收益率序列 (≤T-1 窗口) 估计 EGARCH(1,1) 参数。失败返回 False。"""
        r = np.asarray(returns, dtype=np.float64)
        r = r[np.isfinite(r)]
        if len(r) < 30:
            return False
        r = r[-self.window:]
        # 初值: ω=ln(RV), α=0.08, γ=-0.05, β=0.95 (GARCH 经验初值)
        rv = float(np.var(r))
        p0 = np.array([np.log(max(rv, 1e-8)), 0.08, -0.05, 0.95])
        bounds = [(-8.0, 2.0), (0.01, 0.30), (-0.30, 0.30), (0.90, 0.999)]
        try:
            from scipy.optimize import minimize
            res = minimize(self._neg_loglik, p0, args=(r,), method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': self.max_iter,
                                                   'ftol': self.tol})
            if not res.success and res.fun >= self._neg_loglik(p0, r):
                # 不收敛且没改善 → 用初值 (RV 兜底)
                self.params = p0
                self.converged = False
            else:
                self.params = res.x
                self.converged = res.success
        except Exception as e:
            logger.warning(f"[egarch] 估计失败: {e}")
            self.params = p0
            self.converged = False
        # 最近一期条件方差
        if _HAS_NB:
            self.last_sigma2 = float(_egarch_filter_nb(r, *self.params)[-1])
        else:
            self.last_sigma2 = self._filter_py(r, self.params)[-1]
        return True

    def _filter(self, r: np.ndarray, p: np.ndarray) -> np.ndarray:
        if _HAS_NB:
            return _egarch_filter_nb(r, *p)
        return self._filter_py(r, p)

    def _filter_py(self, r: np.ndarray, p: np.ndarray) -> np.ndarray:
        w, a, g, b = p
        n = len(r)
        sigma2 = np.empty(n)
        s2 = float(np.var(r)) if n > 1 else 1e-6
        for t in range(n):
            if t > 0:
                s2 = np.exp(w + a * (abs(r[t - 1]) - SQRT_2_PI) + g * r[t - 1]
                            + b * np.log(s2))
            else:
                s2 = np.exp(w + b * np.log(max(s2, 1e-12)))
            sigma2[t] = max(s2, 1e-12)
        return sigma2

    def predict_cumvol(self, horizon: int = 20) -> float:
        """预测未来 horizon 日累计波动率 σ_{t+1..t+h} (年化日频: 日σ, 不乘√252, 供排序)。"""
        if self.params is None:
            return np.nan
        w, a, g, b = self.params
        ln_s2 = np.log(max(self.last_sigma2, 1e-12))
        total = 0.0
        for _ in range(max(1, int(horizon))):
            ln_s2 = w + b * ln_s2      # E[ln σ²] 递推 (α,γ 项期望为 0: E|z|-√2/π=0, E[z]=0)
            total += np.exp(ln_s2)
        return float(np.sqrt(total / horizon))   # 日均波动率 (与 RV20 同量纲)

    def predict_daily_vol(self) -> float:
        """预测下一日波动率 (日σ)。"""
        if self.params is None:
            return np.nan
        w, a, g, b = self.params
        ln_s2 = w + b * np.log(max(self.last_sigma2, 1e-12))
        return float(np.sqrt(np.exp(ln_s2)))
