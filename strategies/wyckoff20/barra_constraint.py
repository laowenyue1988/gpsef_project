# -*- coding: utf-8 -*-
"""
Barra 约束组合优化器 — 机构级二次规划求解

功能：
  1. 加载 Barra 暴露快照
  2. 构造 alpha 向量（IC 缩放 + 截面标准化）
  3. 构造 EWMA 协方差矩阵（decay=0.94, RiskMetrics 标准）
  4. CVXOPT / scipy QP 求解：
     max α^T w - λ·w^T Σ w
     s.t. Σw=1, 0≤w≤0.10
          λ_dev·||B^T w - B_bench||² (风格因子软约束)
          λ_ind·||H^T w - b_ind||²  (行业偏离软约束, 替代旧版 max_ind_dev 硬约束)
  5. 后处理：w < min_weight 剔除 → 归一化

降级条件：
  - 优化器求解失败 → 回退 softmax + warning
  - 持仓 < min_positions → 直接 softmax
  - barra_constraint.enabled=false → 直接 softmax

Author: BULLION GPSEF Framework
Date: 2026-07-10
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# ─── 参数默认值 ──────────────────────────────────────────────
DEFAULT_RISK_AVERSION = 1.0
DEFAULT_MAX_WEIGHT = 0.10
DEFAULT_MIN_WEIGHT = 0.01
DEFAULT_TOLERANCES = [0.3, 0.3, 0.5, 0.5, 0.3, 0.3]
DEFAULT_EXPECTED_IC = 0.05
DEFAULT_VOL_TARGET = 0.15
DEFAULT_EWMA_DECAY = 0.94
DEFAULT_MIN_POSITIONS = 20
DEFAULT_LAMBDA_DEV = 0.5     # 因子偏离惩罚权重
DEFAULT_LAMBDA_L2 = 0.5      # alpha 保留惩罚权重 (0.02→0.5, Part C v4.0)

# [2026-09-08 死参数清理] 已移除 DEFAULT_MAX_IND_DEV: 行业偏离由 λ_ind 软惩罚承担
# (原 hard_constraint 模式的 max_ind_dev 硬约束已废弃)。
# 废弃参数列表 (solve_barra_weights 以 **kwargs 吸收并告警, 兼容旧调用方):
LEGACY_DEPRECATED_KWARGS = {'prev_weights', 'max_ind_dev', 'turnover_bound'}

STYLE_FACTORS = ['size', 'beta', 'value', 'momentum', 'volatility', 'liquidity']
N_STYLE = len(STYLE_FACTORS)


def build_alpha_from_preds(preds: np.ndarray,
                            expected_ic: float = DEFAULT_EXPECTED_IC,
                            vol_target: float = DEFAULT_VOL_TARGET,
                            alpha_scale: float = 10.0) -> np.ndarray:
    """将模型预测转为可优化的 alpha 向量 (Grinold-Kahn 式)。

    alpha = alpha_scale × expected_ic × vol_target × z(preds)

    [Fix-20260908 量纲审核] 原式 expected_ic×vol_target×z 对弱 IC(~0.05) 产出的年化
    alpha 仅 ~0.75%, 相对风险项(年化方差 ~0.25×λ_risk) 被完全压制 → 优化退化为
    纯风险最小化。引入 alpha_scale(默认 10) 使 alpha 与组合风险同量级,
    风险厌恶 λ_risk 才能与 alpha 形成真实权衡。alpha_scale 可配置
    (config barra.alpha_scale)。
    """
    if len(preds) == 0:
        return np.array([])

    alpha_raw = preds - preds.mean()
    std = alpha_raw.std()
    if std < 1e-10:
        return np.zeros(len(preds))

    alpha_z = alpha_raw / std
    alpha = alpha_scale * expected_ic * vol_target * alpha_z

    return alpha


def compute_ewma_covariance(return_matrix: np.ndarray,
                             decay: float = DEFAULT_EWMA_DECAY) -> np.ndarray:
    """
    EWMA 协方差矩阵（RiskMetrics 标准）。

    Σ_t = (1-λ) × r_t r_t^T + λ × Σ_{t-1}

    Args:
        return_matrix: T×N, 日收益率（最近在前），含 NaN 表示数据缺失
        decay: 衰减因子，0.94 半衰期约 11 天
    """
    T, N = return_matrix.shape
    if T < 20 or N < 2:
        return np.diag(np.nanvar(return_matrix, axis=0))

    # RiskMetrics 标准: 最近数据最高权重 (t=T-1 为最新)
    # weights[0]=small (最旧), weights[T-1]=(1-λ)=0.06 (最新)
    # [P0 Fix] 原 decay**t → decay**(T-1-t), 修复权重方向反转
    weights = np.array([(1 - decay) * decay ** (T - 1 - t) for t in range(T)])
    weights = weights / weights.sum()

    # 处理 NaN：
    # [P1-6 FIX] 缺失的收益率用截面均值填充（而非 0）
    # 用 0 填充会系统性低估缺失数据股票的方差→优化器偏好该类股票
    col_mean = np.nanmean(return_matrix, axis=0)
    rets_filled = np.where(np.isnan(return_matrix), col_mean[np.newaxis, :], return_matrix)

    # 去均值
    rets_dm = rets_filled - np.nanmean(rets_filled, axis=0)

    # 加权协方差
    Sigma = np.zeros((N, N))
    for t in range(T):
        w = weights[t]
        if np.isnan(return_matrix[t]).any():
            # 部分缺失 → 只对外积中有效的元素加权
            mask = ~np.isnan(return_matrix[t])
            rt = rets_dm[t]
            for i in range(N):
                if not mask[i]:
                    continue
                for j in range(N):
                    if mask[j]:
                        Sigma[i, j] += w * rt[i] * rt[j]
        else:
            Sigma += w * np.outer(rets_dm[t], rets_dm[t])

    # 年化
    Sigma = Sigma * 252

    # 微弱正则化保证正定
    Sigma += np.eye(N) * 1e-8

    return Sigma


def compute_covariance_from_snapshot(expo: pd.DataFrame,
                                      selected_codes: List[str],
                                      decay: float = DEFAULT_EWMA_DECAY,
                                      B: Optional[np.ndarray] = None) -> np.ndarray:
    """[2026-09-08 P0-2] 因子模型协方差 Σ = B·F·Bᵀ + Δ (BARRA 标准)。

    [Fix-20260908 业界标准] 因子收益 F 在**宇宙截面**(快照全部股票)上逐日 OLS 回归估计:
      R_univ(t) = B_univ·f(t) + ε(t)  →  F(6×T) → EWMA 因子协方差(年化)
      特质风险 Δ = ε 的 EWMA 方差对角(取持仓子集)
    而非此前仅用 20 只持仓做截面回归(噪声巨大、Σ 不可靠)。

    若宇宙样本不足/退化, 降级为个股 EWMA 协方差(原行为)。
    """
    import json as _json

    def _parse_rets(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if isinstance(v, str):
            try:
                return _json.loads(v)
            except Exception:
                return None
        return v

    # ── 宇宙收益矩阵与暴露矩阵 (全快照股票) ─────────────────────
    univ_codes = []
    univ_rets = []
    for _, row in expo.iterrows():
        r = _parse_rets(row.get('daily_rets_252'))
        if r is not None and len(r) >= 50:
            univ_codes.append(row['code'])
            univ_rets.append(r)

    if len(univ_codes) < 30:
        return np.diag(np.ones(len(selected_codes)) * 0.0225)  # 默认 15% vol

    min_len = min(len(r) for r in univ_rets)
    R_u = np.array([r[-min_len:] for r in univ_rets]).T  # T × N_univ

    # 宇宙暴露矩阵 (正交化后因子)
    exp_cols = [f'{f}_exp' for f in STYLE_FACTORS]
    B_u = np.zeros((len(univ_codes), N_STYLE))
    code_to_idx_u = {c: i for i, c in enumerate(univ_codes)}
    for k, f in enumerate(STYLE_FACTORS):
        col = f'{f}_exp'
        if col in expo.columns:
            cv = dict(zip(expo['code'], expo[col].fillna(0.0)))
            for j, c in enumerate(univ_codes):
                B_u[j, k] = float(cv.get(c, 0.0))

    # NaN → 截面均值填充 (与 compute_ewma_covariance 内部一致)
    col_mean = np.nanmean(R_u, axis=0)
    R_u = np.where(np.isnan(R_u), col_mean[np.newaxis, :], R_u)

    # ── 逐日截面 OLS: 含市场因子(截距) + 6 风格因子 ─────────────
    # [Fix-20260908 市场因子] BARRA/CNE5 标准回归必含 Country/市场因子:
    #   R(t) = m(t)·1 + B·f(t) + ε(t)
    # 缺失市场因子时市场共同波动全部进入特质残差 → 特质风险虚高、
    # 组合相关性被稀释、风险结构失真 (T1b 实测: 因子ρ=0.587 但总Σρ≈0)。
    B_u = np.hstack([np.ones((len(univ_codes), 1)), B_u])   # 首列 = 市场因子暴露(1)
    K = B_u.shape[1]
    BtB = B_u.T @ B_u + np.eye(K) * 1e-8
    F_all = np.linalg.solve(BtB, B_u.T @ R_u.T)             # K × T
    F_cov = compute_ewma_covariance(F_all.T, decay)         # 年化 (K×K)

    # 特质残差 (T × N_univ)
    E = R_u - (B_u @ F_all).T
    # [Fix-20260908] 每股票特质方差 = 自身残差 EWMA 方差 (向量化: 沿时间轴)
    resid = E - np.nanmean(E, axis=0)                       # 去均值
    T = resid.shape[0]
    w_ew = np.power(decay, np.arange(T - 1, -1, -1))        # 最新权重最高
    w_ew = w_ew / w_ew.sum()
    delta_u = np.nansum((resid ** 2) * w_ew[:, None], axis=0) * 252.0
    delta_u = np.maximum(delta_u, 1e-8)

    # ── 输出: 持仓子集的 Σ = B_sel·F·B_selᵀ + diag(Δ_sel) ───────
    n = len(selected_codes)
    Sigma = np.zeros((n, n))
    if B is not None and B.shape[0] == n:
        B_sel = np.hstack([np.ones((n, 1)), B])             # 加市场因子列
        Sigma = B_sel @ F_cov @ B_sel.T
    else:
        # 无持仓暴露矩阵时: 从宇宙映射
        B_sel = np.zeros((n, N_STYLE))
        for j, c in enumerate(selected_codes):
            c6 = str(c).split('.')[-1].zfill(6)
            iu = code_to_idx_u.get(c6)
            if iu is not None:
                B_sel[j, :] = B_u[iu, 1:]                   # 跳过市场列
        Sigma = np.hstack([np.ones((n, 1)), B_sel]) @ F_cov @ np.hstack([np.ones((n, 1)), B_sel]).T

    for j, c in enumerate(selected_codes):
        c6 = str(c).split('.')[-1].zfill(6)
        iu = code_to_idx_u.get(c6)
        if iu is not None:
            Sigma[j, j] += float(delta_u[iu])
        elif Sigma[j, j] < 1e-10:
            Sigma[j, j] = 0.0225

    # 微弱正则化保证正定
    Sigma += np.eye(n) * 1e-8
    return Sigma


def compute_industry_deviation(weights: np.ndarray,
                                selected_codes: List[str],
                                sel_industries: List[str],
                                bench_industry_weights: Dict[str, float]) -> float:
    """
    行业总偏离 = 0.5 × Σ |w_i^port - w_i^bench|
    """
    port_ind = defaultdict(float)
    for code, w, ind in zip(selected_codes, weights, sel_industries):
        port_ind[ind] += w

    all_inds = set(port_ind.keys()) | set(bench_industry_weights.keys())
    deviation = sum(
        abs(port_ind.get(i, 0) - bench_industry_weights.get(i, 0))
        for i in all_inds
    )
    return 0.5 * deviation


def optimize_portfolio(
    alpha: np.ndarray,
    Sigma: np.ndarray,
    B_port: np.ndarray,
    B_bench: np.ndarray,
    tolerances: List[float],
    selected_codes: List[str],
    sel_industries: List[str],
    bench_industry_weights: Dict[str, float],
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    method: str = 'factor_neutral',    # 'factor_neutral' | 'hard_constraint'
    w_target: np.ndarray = None,       # 仅在 factor_neutral 模式下使用
    lambda_dev: float = DEFAULT_LAMBDA_DEV,
    lambda_l2: float = DEFAULT_LAMBDA_L2,
    total_exposure: float = 1.0,       # [P0-2] 总敞口（GARCH 前置确定）
    lambda_ind: float = 0.3,           # [P0-1] 行业偏离惩罚系数
) -> Optional[np.ndarray]:
    """
    Barra 组合优化。

    [2026-09-08 死参数清理] 已删除 prev_weights/max_ind_dev/turnover_bound
    (三者从未在函数体内使用; 行业控制由 λ_ind 软惩罚承担)。

    method='factor_neutral' (推荐):
      min  λ_dev ||B^T w - B_bench||² + λ_l2 ||w - w_target||²
           + λ_ind ||H^T w - b_ind||²  [P0-1 行业约束]
      s.t. Σw = total_exposure, 0 ≤ w ≤ max_weight
           [P0-2 total_exposure 由 GARCH 前置确定, 默认=1.0]
      适用场景: alpha 有效时保留信号 + 约束因子偏离 + 行业集中度控制

    method='hard_constraint' (旧版):
      max α^T w - λ w^T Σ w
      s.t. |B^T w - B_bench| ≤ k, Σw = total_exposure, 0 ≤ w ≤ max_weight
    """
    n = len(alpha)
    if n == 0:
        return None

    # ── w_target: 默认 = softmax(alpha) ────────────────────
    if w_target is None:
        exp_a = np.exp(np.clip(alpha - alpha.max(), -10, 0) / 0.3)
        w_target = exp_a / exp_a.sum()

    try:
        from cvxopt import matrix, solvers
        solvers.options['show_progress'] = False
        solvers.options['maxiters'] = 200
        solvers.options['abstol'] = 1e-8
        solvers.options['reltol'] = 1e-7
        solvers.options['feastol'] = 1e-7
    except ImportError:
        logger.error("[Barra QP] cvxopt not installed")
        return None

    if method == 'factor_neutral':
        # ── 因子中性化: 双目标软优化 + 行业偏离 ─────────
        # P = 2 × (λ_dev B B^T + λ_l2 I + λ_ind H H^T)
        # q = -2 × (λ_dev B B_bench + λ_l2 w_target + λ_ind H b_ind)
        BBt = B_port @ B_port.T
        P = 2.0 * (lambda_dev * BBt + lambda_l2 * np.eye(n))

        # [P0-1] 行业偏离惩罚矩阵
        unique_inds = sorted(set(sel_industries))
        if lambda_ind > 0 and len(unique_inds) > 1:
            n_ind = len(unique_inds)
            ind_to_idx = {ind: i for i, ind in enumerate(unique_inds)}
            H_ind = np.zeros((n, n_ind))
            for j, ind in enumerate(sel_industries):
                if ind in ind_to_idx:
                    H_ind[j, ind_to_idx[ind]] = 1.0
            b_ind = np.array([bench_industry_weights.get(ind, 0.0)
                             for ind in unique_inds])
            P += 2.0 * lambda_ind * (H_ind @ H_ind.T)
            q_ind = -2.0 * lambda_ind * (H_ind @ b_ind)
        else:
            q_ind = np.zeros(n)

        P = P + np.eye(n) * 1e-10

        q_vec = -2.0 * (lambda_dev * (B_port @ B_bench) + lambda_l2 * w_target) + q_ind

        # 约束: Σw = 1.0 (全投资约束) + 边界
        G = np.vstack([-np.eye(n), np.eye(n)])
        h = np.hstack([np.zeros(n), np.ones(n) * max_weight])
        A_eq = np.ones((1, n))
        b_eq = np.array([1.0])

        result = solvers.qp(
            matrix(P, tc='d'), matrix(q_vec, tc='d'),
            matrix(G, tc='d'), matrix(h, tc='d'),
            matrix(A_eq, tc='d'), matrix(b_eq, tc='d'),
        )
        # [P1-4 FIX] 检查求解状态, 非最优解时 fallback 到 softmax
        if result['status'] != 'optimal':
            logger.warning(f"[Barra Neutral] Solver status={result['status']}, "
                           f"not optimal, falling back to softmax")
            return None

        weights = np.array(result['x']).flatten()

        # 后验: 因子偏离
        port_expo = B_port.T @ weights
        deviations = [abs(port_expo[k] - B_bench[k]) for k in range(N_STYLE)]
        n_pos = int((weights > 0.001).sum())
        logger.info(f"[Barra Neutral] Converged: {n_pos}/{n} positions, "
                    f"dev=[{', '.join(f'{d:.3f}' for d in deviations)}]")

    elif method == 'corrected_soft':
        # ── [Part C] 修正软约束: max α^T w - λ_risk·w^TΣw - λ_dev·||B^T w - B_bench||² ──
        # QP form: min 0.5 w^T P w + q^T w
        # P = 2·λ_risk·Σ + 2·λ_dev·BB^T + 2·λ_ind·HH^T + 2·λ_l2·I
        # q = -α - 2·λ_dev·B·B_bench - 2·λ_ind·H·b_ind
        lambda_risk = risk_aversion  # from config, default 1.0
        BBt = B_port @ B_port.T
        P = 2.0 * (lambda_risk * Sigma + lambda_dev * BBt + lambda_l2 * np.eye(n))

        # Industry deviation
        unique_inds = sorted(set(sel_industries))
        if lambda_ind > 0 and len(unique_inds) > 1:
            H_ind = np.zeros((n, len(unique_inds)))
            ind_to_idx = {ind: i for i, ind in enumerate(unique_inds)}
            for j, ind in enumerate(sel_industries):
                if ind in ind_to_idx:
                    H_ind[j, ind_to_idx[ind]] = 1.0
            b_ind = np.array([bench_industry_weights.get(ind, 0.0) for ind in unique_inds])
            P += 2.0 * lambda_ind * (H_ind @ H_ind.T)
            q_ind = -2.0 * lambda_ind * (H_ind @ b_ind)
        else:
            q_ind = np.zeros(n)

        P = P + np.eye(n) * 1e-10
        q_vec = -alpha - 2.0 * lambda_dev * (B_port @ B_bench) + q_ind

        G = np.vstack([-np.eye(n), np.eye(n)])
        h = np.hstack([np.zeros(n), np.ones(n) * max_weight])
        A_eq = np.ones((1, n))
        b_eq = np.array([1.0])

        result = solvers.qp(
            matrix(P, tc='d'), matrix(q_vec, tc='d'),
            matrix(G, tc='d'), matrix(h, tc='d'),
            matrix(A_eq, tc='d'), matrix(b_eq, tc='d'),
        )
        if result['status'] != 'optimal':
            logger.warning(f"[Barra CorrectedSoft] Solver status={result['status']}, falling back")
            return None

        weights = np.array(result['x']).flatten()
        port_expo = B_port.T @ weights
        deviations = [abs(port_expo[k] - B_bench[k]) for k in range(N_STYLE)]
        n_pos = int((weights > 0.001).sum())
        logger.info(f"[Barra CorrectedSoft] Converged: {n_pos}/{n} positions, "
                    f"dev=[{', '.join(f'{d:.3f}' for d in deviations)}], "
                    f"risk_aversion={lambda_risk}")

    else:
        # ── 硬约束模式 (旧版) ────────────────────────────
        P = 2.0 * risk_aversion * Sigma + np.eye(n) * 1e-10
        q_vec = -alpha

        G_list = [-np.eye(n), np.eye(n)]
        h_list = [np.zeros(n), np.ones(n) * max_weight]

        for k in range(N_STYLE):
            tol = tolerances[k] if k < len(tolerances) else 0.3
            G_list.append(B_port[:, k:k+1].T)
            h_list.append(np.array([B_bench[k] + tol]))
            G_list.append(-B_port[:, k:k+1].T)
            h_list.append(np.array([tol - B_bench[k]]))

        G = np.vstack(G_list)
        h = np.hstack(h_list)
        # 约束: Σw = 1.0 (全投资约束)
        A_eq = np.ones((1, n))
        b_eq = np.array([1.0])

        result = solvers.qp(
            matrix(P, tc='d'), matrix(q_vec, tc='d'),
            matrix(G, tc='d'), matrix(h, tc='d'),
            matrix(A_eq, tc='d'), matrix(b_eq, tc='d'),
        )
        # [P1-4 FIX] hard_constraint 模式同样检查求解状态
        if result['status'] != 'optimal':
            logger.warning(f"[Barra Hard] Solver status={result['status']}, "
                           f"not optimal, falling back to softmax")
            return None
        weights = np.array(result['x']).flatten()

    # ── 后处理 ──────────────────────────────────────────
    weights = np.maximum(weights, 0)  # 确保非负
    weights = np.minimum(weights, max_weight)  # 确保不超上限
    # [P0-2 FIX] weights 全零时的 fallback: 等权分布而非引用未定义变量 x0
    n_total = len(weights)
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(n_total) / n_total

    # 剔除 < min_weight 的仓位
    weights[weights < min_weight] = 0
    if weights.sum() > 0:
        weights = weights / weights.sum()

    return weights


def softmax_weights(preds: np.ndarray, temperature: float = 0.3,
                     max_pct: float = 0.10) -> np.ndarray:
    """退路方案：原 softmax 加权。"""
    if len(preds) == 0:
        return np.array([])
    exp_preds = np.exp((preds - preds.max()) / temperature)
    w = exp_preds / exp_preds.sum()
    w = np.minimum(w, max_pct)
    return w / w.sum()


def solve_barra_weights(
    preds: np.ndarray,
    selected_codes: List[str],
    barra_expo: pd.DataFrame,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    tolerances: Optional[List[float]] = None,
    expected_ic: float = DEFAULT_EXPECTED_IC,
    vol_target: float = DEFAULT_VOL_TARGET,
    constraint_date=None,
    total_exposure: float = 1.0,  # [P0-2] GARCH 前置确定的总敞口
    lambda_dev: float = DEFAULT_LAMBDA_DEV,
    lambda_l2: float = DEFAULT_LAMBDA_L2,
    lambda_ind: float = 0.3,
    baseline: str = 'csi800',    # [Fix-20260908 业界标准] 基线指数 (仅用于聚合 B_bench/行业基准)
    alpha_scale: float = 10.0,   # [Fix-20260908] alpha 量纲缩放 (IC×vol×z 放大系数)
    **kwargs,                    # [2026-09-08 死参数清理] 吸收废弃参数 (见 LEGACY_DEPRECATED_KWARGS)
) -> Tuple[np.ndarray, bool]:
    """
    Barra 约束权重分配的顶层接口。

    [2026-09-08 死参数清理] prev_weights/max_ind_dev/turnover_bound 已删除:
    - prev_weights: 从未参与任何计算 (换手约束未实现)
    - max_ind_dev: 行业偏离硬约束已由 λ_ind 软惩罚替代 (见 optimize_portfolio)
    - turnover_bound: 从未使用
    旧调用方若仍传这些参数, 经 **kwargs 静默吸收并告警一次 (兼容兄弟策略)。

    对于选股池中不在预构建快照中的股票，即时补充计算其 Barra 暴露
    （使用与构建快照完全相同的逻辑，只能使用 constraint_date 之前的数据）。
    """
    if tolerances is None:
        tolerances = DEFAULT_TOLERANCES

    deprecated = LEGACY_DEPRECATED_KWARGS & set(kwargs)
    if deprecated:
        logger.warning(f"[Barra] 废弃参数已忽略 (死参数清理 2026-09-08): "
                       f"{sorted(deprecated)}; 行业控制请使用 lambda_ind 软惩罚")

    # [Fix-20260908] code 统一 6 位匹配: 快照与 on-the-fly 内部均为 6 位,
    # selected_codes 来自策略层可为 ts_code 格式 ('sz.300588')。此前混用导致:
    # ①持仓股 isin 匹配全失败 → on-the-fly 被 ts code 取数失败 → 暴露全 NaN;
    # ②B_port 全 0 → λ_dev/基线切换零传导; ③Σ 取数失败退化为默认对角矩阵。
    code6_of = {c.split('.')[-1].zfill(6): c for c in selected_codes}
    sel6 = list(code6_of.keys())

    # 检测缺失股票，即时补充计算
    missing_codes = [c6 for c6 in sel6 if c6 not in barra_expo['code'].values]
    if missing_codes and constraint_date is not None:
        try:
            from strategies.wyckoff20.barra_builder import compute_on_the_fly_exposures
            logger.info(f"[Barra] Computing on-the-fly exposures for {len(missing_codes)} "
                        f"missing stocks (constraint={constraint_date.date()})")
            otf_expo = compute_on_the_fly_exposures(
                missing_codes, constraint_date, barra_expo
            )
            if len(otf_expo) > 0:
                barra_expo = pd.concat([barra_expo, otf_expo], ignore_index=True)
                logger.info(f"[Barra] Added {len(otf_expo)} on-the-fly exposures")
        except Exception as e:
            logger.warning(f"[Barra] On-the-fly exposure failed: {e}")

    # 对齐快照 (6 位匹配)
    expo_sel = barra_expo[barra_expo['code'].isin(sel6)].copy()
    if len(expo_sel) < DEFAULT_MIN_POSITIONS:
        logger.info(f"[Barra] Only {len(expo_sel)} stocks with Barra data, "
                    f"falling back to softmax")
        return softmax_weights(preds), False

    # 对齐顺序 (expo_sel 行 → selected_codes 原始位置)
    sel6_pos = {c6: i for i, c6 in enumerate(sel6)}
    idx_order = [sel6_pos.get(c) for c in expo_sel['code']]
    idx_order = [i for i in idx_order if i is not None]
    if len(idx_order) < DEFAULT_MIN_POSITIONS:
        logger.warning(f"[Barra] Only {len(idx_order)} position matches, "
                       f"falling back to softmax")
        return softmax_weights(preds), False

    # [Fix-20260908] 防御: 有效因子暴露行不足 → 暴露计算异常, 禁止 B_port≈0 静默继续
    exp_cols_chk = [f'{f}_exp' for f in STYLE_FACTORS]
    expo_ok = expo_sel[exp_cols_chk].notna().sum(axis=1)
    n_ok = int((expo_ok >= 3).sum())
    if n_ok < DEFAULT_MIN_POSITIONS:
        logger.warning(f"[Barra] Exposures mostly NaN ({n_ok} valid rows), "
                       f"falling back to softmax")
        return softmax_weights(preds), False

    # 子集数据
    alpha_full = build_alpha_from_preds(preds, expected_ic, vol_target, alpha_scale)
    alpha_sel = alpha_full[idx_order]
    codes_sel = [selected_codes[i] for i in idx_order]
    codes_sel6 = [c.split('.')[-1].zfill(6) for c in codes_sel]

    # 因子暴露矩阵 N×K (code 统一 6 位匹配)
    B_port = np.zeros((len(codes_sel), N_STYLE))
    for k, f in enumerate(STYLE_FACTORS):
        col = f'{f}_exp'
        if col in expo_sel.columns:
            code_vals = dict(zip(expo_sel['code'], expo_sel[col].fillna(0)))
            for j, c6 in enumerate(codes_sel6):
                B_port[j, k] = float(code_vals.get(c6, 0))

    # 基准暴露 (业界标准: 基线成分 ∩ 宇宙快照 的市值加权, 见 barra_builder.compute_benchmark_exposures)
    try:
        from strategies.wyckoff20.barra_builder import compute_benchmark_exposures as _cbe
        B_bench = _cbe(barra_expo, baseline=baseline, constraint_date=constraint_date)
    except Exception:
        B_bench = compute_benchmark_exposures_from_df(barra_expo)

    # [P0-2] 因子模型协方差 (先构建 B_port 再估计 Σ = B·F·Bᵀ + Δ)
    # [Fix-20260908] 传入 6 位 code 列表, 保证能取到各股 daily_rets_252 (此前 ts 码匹配失败 → Σ 退化为默认对角)
    Sigma = compute_covariance_from_snapshot(barra_expo, codes_sel6, B=B_port)

    # 行业 (code 统一 6 位匹配)
    ind_map = dict(zip(expo_sel['code'], expo_sel['industry'].fillna('other')))
    sel_industries = [ind_map.get(c6, 'other') for c6 in codes_sel6]
    bench_ind = _compute_benchmark_industry_weights(
        barra_expo, baseline=baseline, constraint_date=constraint_date)

    # 求解 — corrected_soft 三元目标函数 (Part C v4.0)
    result = optimize_portfolio(
        alpha_sel, Sigma, B_port, B_bench, tolerances,
        codes_sel, sel_industries, bench_ind,
        method='corrected_soft',
        risk_aversion=risk_aversion,
        max_weight=max_weight,
        lambda_dev=lambda_dev,       # config-driven (default 0.5)
        lambda_l2=lambda_l2,         # config-driven (default 0.5)
        lambda_ind=lambda_ind,       # config-driven (default 0.3)
        total_exposure=1.0,
    )

    if result is None:
        logger.warning("[Barra] QP solver failed, falling back to softmax")
        return softmax_weights(preds), False

    # 映射回全量
    full_w = np.zeros(len(selected_codes))
    for j, i in enumerate(idx_order):
        full_w[i] = result[j]

    logger.info(f"[Barra] Optimized: {int((full_w > 0).sum())} stocks, "
                f"max_w={full_w.max():.3f}, sum_w={full_w.sum():.3f}")

    return full_w, True


def compute_benchmark_exposures_from_df(expo: pd.DataFrame, baseline: str = 'csi800',
                                        constraint_date=None) -> np.ndarray:
    """[Fix-20260908] 基线指数市值加权暴露 (兼容接口)。

    优先: 基线成分 ∩ 宇宙快照 → total_mv 加权 (业界标准, 见 barra_builder.compute_benchmark_exposures)。
    旧快照(含 weight_bm): 直接用。兜底: total_mv 全宇宙加权。
    """
    try:
        from strategies.wyckoff20.barra_builder import compute_benchmark_exposures as _cbe
        return _cbe(expo, baseline=baseline, constraint_date=constraint_date)
    except Exception:
        pass
    if 'weight_bm' in expo.columns:
        w = expo['weight_bm'].fillna(0).values
        w = w / w.sum() if w.sum() > 0 else np.ones(len(expo)) / len(expo)
    elif 'total_mv' in expo.columns:
        w = expo['total_mv'].fillna(0).values
        w = w / w.sum() if w.sum() > 0 else np.ones(len(expo)) / len(expo)
    else:
        w = np.ones(len(expo)) / max(len(expo), 1)

    bench = np.zeros(N_STYLE)
    for k, f in enumerate(STYLE_FACTORS):
        col = f'{f}_exp'
        if col in expo.columns:
            bench[k] = float(np.dot(w, expo[col].fillna(0).values))
    return bench


def _compute_benchmark_industry_weights(expo: pd.DataFrame, baseline: str = 'csi800',
                                        constraint_date=None) -> Dict[str, float]:
    """[Fix-20260908] 基线指数行业权重 (兼容接口, 逻辑同 compute_benchmark_exposures)。

    [2026-09-08] sh932000/sh000001 无本地成分缓存 → 宇宙快照派生成分:
      sh932000 = 市值排名 1801-3800; sh000001 = 沪市 (code 以 6 开头)。
    """
    wcol = None
    sub = expo
    if 'weight_bm' not in expo.columns and constraint_date is not None \
            and 'total_mv' in expo.columns:
        if baseline == 'sh932000':
            s = expo.dropna(subset=['total_mv'])
            s = s[s['total_mv'] > 0].sort_values('total_mv', ascending=False)
            sub = s.iloc[1800:3800].copy()
        elif baseline == 'sh000001':
            sub = expo[expo['code'].astype(str).str.startswith('6')].copy()
        else:
            try:
                from strategies.wyckoff20.barra_builder import _get_constituents
                const = _get_constituents(constraint_date, baseline, bs=None)
                if len(const) > 0:
                    codes = set(const['code'].astype(str).str.zfill(6))
                    sub = expo[expo['code'].isin(codes)].copy()
            except Exception:
                pass
    if 'weight_bm' in sub.columns:
        wcol = 'weight_bm'
    elif 'total_mv' in sub.columns:
        wcol = 'total_mv'

    ind_w = defaultdict(float)
    if wcol is None:
        for _, row in sub.iterrows():
            ind_w[str(row.get('industry', 'other'))] += 1.0
    else:
        for _, row in sub.iterrows():
            ind = row.get('industry', 'other')
            if pd.isna(ind):
                ind = 'other'
            wi = row.get(wcol, 0)
            if pd.isna(wi):
                wi = 0
            ind_w[str(ind)] += float(wi)

    total = sum(ind_w.values())
    if total > 0:
        ind_w = {k: v / total for k, v in ind_w.items()}
    return dict(ind_w)
