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
          |B_port^T w - B_bench| ≤ k·σ  (五风格软约束)
          0.5 Σ|w_ind - w_bm| ≤ max_ind_dev (行业总偏离)
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
DEFAULT_MAX_IND_DEV = 0.12
DEFAULT_TOLERANCES = [0.3, 0.3, 0.5, 0.5, 0.3]
DEFAULT_EXPECTED_IC = 0.05
DEFAULT_VOL_TARGET = 0.15
DEFAULT_EWMA_DECAY = 0.94
DEFAULT_MIN_POSITIONS = 20
DEFAULT_LAMBDA_DEV = 0.5     # 因子偏离惩罚权重
DEFAULT_LAMBDA_L2 = 0.5      # alpha 保留惩罚权重 (0.02→0.5, Part C v4.0)

STYLE_FACTORS = ['size', 'beta', 'value', 'momentum', 'volatility']


def build_alpha_from_preds(preds: np.ndarray,
                            expected_ic: float = DEFAULT_EXPECTED_IC,
                            vol_target: float = DEFAULT_VOL_TARGET) -> np.ndarray:
    """
    将模型预测转为可优化的 alpha 向量。

    preds 应为原始预测值（校前），精度高无平局。
    Step 1: 去均值 → 相对强弱信号
    Step 2: 截面 Z-score
    Step 3: 缩放到年化预期收益
    """
    if len(preds) == 0:
        return np.array([])

    alpha_raw = preds - preds.mean()
    std = alpha_raw.std()
    if std < 1e-10:
        return np.zeros(len(preds))

    alpha_z = alpha_raw / std
    alpha = expected_ic * vol_target * alpha_z

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
                                      decay: float = DEFAULT_EWMA_DECAY) -> np.ndarray:
    """
    从 Barra 快照的 daily_rets_252 列构建 EWMA 协方差。
    """
    ret_matrix = []
    valid_codes = []

    for code in selected_codes:
        row = expo[expo['code'] == code]
        if len(row) == 0:
            continue
        rets_json = row['daily_rets_252'].iloc[0]
        if rets_json is None or (isinstance(rets_json, float) and np.isnan(rets_json)):
            continue
        if isinstance(rets_json, str):
            import json
            rets = json.loads(rets_json)
        else:
            rets = rets_json
        if rets is None or len(rets) < 50:
            continue
        ret_matrix.append(rets)
        valid_codes.append(code)

    if len(valid_codes) < 3:
        return np.diag(np.ones(len(selected_codes)) * 0.0225)  # 默认 15% vol

    # 对齐长度
    min_len = min(len(r) for r in ret_matrix)
    aligned = np.array([r[-min_len:] for r in ret_matrix]).T  # T × N

    Sigma_full = compute_ewma_covariance(aligned, decay)

    # 映射回原始顺序
    code_to_idx = {c: i for i, c in enumerate(selected_codes)}
    idx_map = [code_to_idx.get(c) for c in valid_codes]
    Sigma = np.zeros((len(selected_codes), len(selected_codes)))
    for i, mi in enumerate(idx_map):
        if mi is not None:
            for j, mj in enumerate(idx_map):
                if mj is not None:
                    Sigma[mi, mj] = Sigma_full[i, j]

    # 缺失的对角线位置填默认值
    for i in range(len(selected_codes)):
        if Sigma[i, i] < 1e-10:
            Sigma[i, i] = 0.0225  # 15% vol²

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
    prev_weights: Optional[np.ndarray] = None,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    max_ind_dev: float = DEFAULT_MAX_IND_DEV,
    turnover_bound: float = 0.50,
    method: str = 'factor_neutral',    # 'factor_neutral' | 'hard_constraint'
    w_target: np.ndarray = None,       # 仅在 factor_neutral 模式下使用
    lambda_dev: float = DEFAULT_LAMBDA_DEV,
    lambda_l2: float = DEFAULT_LAMBDA_L2,
    total_exposure: float = 1.0,       # [P0-2] 总敞口（GARCH 前置确定）
    lambda_ind: float = 0.3,           # [P0-1] 行业偏离惩罚系数
) -> Optional[np.ndarray]:
    """
    Barra 组合优化。

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
        deviations = [abs(port_expo[k] - B_bench[k]) for k in range(5)]
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
        deviations = [abs(port_expo[k] - B_bench[k]) for k in range(5)]
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

        for k in range(5):
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
    prev_weights: Optional[np.ndarray] = None,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    tolerances: Optional[List[float]] = None,
    expected_ic: float = DEFAULT_EXPECTED_IC,
    vol_target: float = DEFAULT_VOL_TARGET,
    max_ind_dev: float = DEFAULT_MAX_IND_DEV,
    constraint_date=None,
    total_exposure: float = 1.0,  # [P0-2] GARCH 前置确定的总敞口
    lambda_dev: float = DEFAULT_LAMBDA_DEV,
    lambda_l2: float = DEFAULT_LAMBDA_L2,
    lambda_ind: float = 0.3,
) -> Tuple[np.ndarray, bool]:
    """
    Barra 约束权重分配的顶层接口。

    对于选股池中不在预构建快照中的股票，即时补充计算其 Barra 暴露
    （使用与构建快照完全相同的逻辑，只能使用 constraint_date 之前的数据）。
    """
    if tolerances is None:
        tolerances = DEFAULT_TOLERANCES

    # 检测缺失股票，即时补充计算
    missing_codes = [c for c in selected_codes if c not in barra_expo['code'].values]
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

    # 对齐快照
    expo_sel = barra_expo[barra_expo['code'].isin(selected_codes)].copy()
    if len(expo_sel) < DEFAULT_MIN_POSITIONS:
        logger.info(f"[Barra] Only {len(expo_sel)} stocks with Barra data, "
                    f"falling back to softmax")
        return softmax_weights(preds), False

    # 对齐顺序
    code_order = {c: i for i, c in enumerate(selected_codes)}
    idx_order = [code_order.get(c) for c in expo_sel['code']]
    idx_order = [i for i in idx_order if i is not None]
    if len(idx_order) < DEFAULT_MIN_POSITIONS:
        return softmax_weights(preds), False

    # 子集数据
    alpha_full = build_alpha_from_preds(preds, expected_ic, vol_target)
    alpha_sel = alpha_full[idx_order]
    codes_sel = [selected_codes[i] for i in idx_order]

    # 协方差
    Sigma = compute_covariance_from_snapshot(barra_expo, codes_sel)

    # 因子暴露矩阵 N×5
    B_port = np.zeros((len(codes_sel), 5))
    for k, f in enumerate(STYLE_FACTORS):
        col = f'{f}_exp'
        if col in expo_sel.columns:
            code_vals = dict(zip(expo_sel['code'], expo_sel[col].fillna(0)))
            for j, c in enumerate(codes_sel):
                B_port[j, k] = float(code_vals.get(c, 0))

    # 基准暴露
    B_bench = compute_benchmark_exposures_from_df(barra_expo)

    # 行业
    ind_map = dict(zip(expo_sel['code'], expo_sel['industry'].fillna('other')))
    sel_industries = [ind_map.get(c, 'other') for c in codes_sel]
    bench_ind = _compute_benchmark_industry_weights(barra_expo)

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


def compute_benchmark_exposures_from_df(expo: pd.DataFrame) -> np.ndarray:
    """CSI 800 市值加权暴露。"""
    w = expo['weight_bm'].fillna(0).values
    w = w / w.sum() if w.sum() > 0 else np.ones(len(expo)) / len(expo)

    bench = np.zeros(5)
    for k, f in enumerate(STYLE_FACTORS):
        col = f'{f}_exp'
        if col in expo.columns:
            bench[k] = float(np.dot(w, expo[col].fillna(0).values))
    return bench


def _compute_benchmark_industry_weights(expo: pd.DataFrame) -> Dict[str, float]:
    """CSI 800 行业权重。"""
    w = expo['weight_bm'].fillna(0).values
    w = w / w.sum() if w.sum() > 0 else np.ones(len(expo)) / len(expo)

    ind_w = defaultdict(float)
    for _, row in expo.iterrows():
        ind = row.get('industry', 'other')
        if pd.isna(ind):
            ind = 'other'
        wi = row.get('weight_bm', 0)
        if pd.isna(wi):
            wi = 0
        ind_w[str(ind)] += float(wi)

    # 重新归一化
    total = sum(ind_w.values())
    if total > 0:
        ind_w = {k: v / total for k, v in ind_w.items()}

    return dict(ind_w)
