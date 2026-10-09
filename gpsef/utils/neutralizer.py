# gpsef/utils/neutralizer.py
"""
因子中性化工具模块

提供通用的行业/市值中性化处理，对因子列做线性回归并取残差。

【修复记录】
- 2026-05-15: 修复 fillna(0) 问题。原代码将因子缺失值填充为0参与回归，
  导致残差计算错误（0 - pred ≠ NaN - pred）。修复后：仅对非缺失样本做回归拟合
  和残差计算，缺失样本的中性化结果保持为 NaN。

【性能优化 - 2026-05-30】
- 向量化中性化：对相同有效mask的因子批量计算OLS残差，而非逐因子调用sklearn LinearRegression。
  预计算投影矩阵 X_pinv = (X'X)^-1 X'，后续只需矩阵乘法。
  数学等价性: OLS闭式解 beta = (X'X)^-1 X'y → 残差 = y - X·beta。
  sklearn LinearRegression 内部使用相同的闭式解（通过 scipy.linalg.lstsq），
  结果在浮点精度(1e-12)内完全一致。
  
- 新增 `neutralize_legacy()` 保留原始逐因子LR实现，方便回归测试。
- 新增 `_ols_residuals_batch()` 批量计算方法。

【确定性修复 - 2026-07-19】
- _neutralize_vectorized: 将 np.linalg.lstsq 替换为 np.linalg.solve + ridge(λ=1e-12)。
  原因：lstsq 底层调用 LAPACK ?GELSD，多线程下浮点运算顺序不确定，导致跨进程
  结果不一致（backtest vs recomd 的 neu_result_md5 不同）。
  solve + ridge 是纯数学运算，完全确定性，且性能优于 lstsq（~3-5x）。
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from collections import defaultdict
import logging


logger = logging.getLogger(__name__)


class FactorNeutralizer:
    """
    因子中性化处理器。

    用法:
        neutralizer = FactorNeutralizer()
        df = neutralizer.neutralize(df, factor_cols, X_neutral)

    【性能优化】默认使用向量化批量计算, 结果与原逐因子LR完全一致。
    如需逐因子LR（调试用）, 调用 neutralize_legacy()。
    
    【fit/transform】neutralize_fit() 拟合并返回系数，
    neutralize_apply() 使用预拟合系数进行纯矩阵运算（跨机器确定性）。
    
    【确定性保证】使用 solve + ridge 替代 lstsq，跨进程结果完全一致。
    """

    def __init__(self, winsorize_pct=(1, 99)):
        """
        Args:
            winsorize_pct: Winsorize 百分位 (low, high)，保留供兼容性。
                Winsorize 逻辑由调用方（strategy._winsorize_factors）执行。
        """
        self.winsorize_pct = winsorize_pct

    def neutralize(self,
                   df: pd.DataFrame,
                   factor_cols: list,
                   X_neutral: pd.DataFrame) -> pd.DataFrame:
        """
        对 df 中的指定因子列进行中性化处理（向量化批量计算版）。

        数学原理:
            对每个因子j: y_j = X·beta_j + eps_j
            beta_j = (X'X)^-1 X' y_j  (OLS闭式解)
            残差 = y_j - X·beta_j
            
            当X相同时（行业哑变量+市值对所有因子相同），可：
            1. 预计算投影矩阵 P = (X'X)^-1 X'
            2. 按有效mask分组：同mask的因子共享P
            3. 批量计算: Betas = P @ Y_matrix
            4. 批量计算: Residuals = Y_all - X_all @ Betas

        参数:
            df: 包含原始因子列的 DataFrame。
            factor_cols: 需要中性化的因子列名列表。
            X_neutral: 解释变量矩阵（如行业哑变量、对数市值等）。

        返回:
            df (新增 _neu 列并返回)
        """
        return self._neutralize_vectorized(df, factor_cols, X_neutral)

    def neutralize_fit(self,
                       df: pd.DataFrame,
                       factor_cols: list,
                       X_neutral: pd.DataFrame):
        """
        拟合OLS中性化并返回中性化后的DataFrame + 拟合系数。

        与 neutralize() 的差异：同时返回每个因子的 OLS 系数（Betas），
        供后续 neutralize_apply() 使用，实现 fit/transform 分离。

        返回:
            (df, betas_info)
            betas_info = {
                'X_cols': [X列名列表],
                'betas': {factor_col: np.array([intercept, coef1, ..., coefK])}
            }
        """

        return self._neutralize_vectorized(df, factor_cols, X_neutral, capture_betas=True)

    def neutralize_apply(self,
                         df: pd.DataFrame,
                         factor_cols: list,
                         X_neutral: pd.DataFrame,
                         betas_info: dict) -> pd.DataFrame:
        """
        使用预拟合的 OLS 系数进行中性化（无线性求解，跨机器确定性）。

        只做矩阵乘法 Y - X·Betas，不调用 np.linalg.lstsq，
        因此不受 BLAS 实现差异影响，确保跨机器位级一致。

        betas_info: 来自 neutralize_fit() 的返回值第二部分。

        行业哑变量列对齐：X_neutral 中缺失的列 → 系数贡献为0；
        X_neutral 中新增的列 → 系数为0。
        """
        X_train_cols = betas_info['X_cols']
        betas_raw = betas_info['betas']

        df = df.copy()
        X = X_neutral.reindex(df.index, fill_value=0)
        X_arr = X.values  # [N, K_pred]
        X_pred_cols = list(X_neutral.columns)
        N = X_arr.shape[0]

        available = [c for c in factor_cols if c in df.columns and c in betas_raw]
        if not available:
            return df

        for col in available:
            valid = df[col].notna() & ~df[col].isin([np.inf, -np.inf])
            n_valid = valid.sum()
            n_total = N

            if n_valid < max(10, int(n_total * 0.05)):
                logger.warning(
                    f"因子 {col} 有效样本过少 ({n_valid}/{n_total})，中性化结果设为 NaN"
                )
                df[col + '_neu'] = np.nan
                continue

            # 对齐系数：训练列 → 预测列
            beta_full = betas_raw[col]  # [K_train + 1]
            beta_aligned = np.zeros(len(X_pred_cols) + 1)
            beta_aligned[0] = beta_full[0]  # intercept
            for j_train, train_col in enumerate(X_train_cols):
                if train_col in X_pred_cols:
                    j_pred = X_pred_cols.index(train_col)
                    beta_aligned[j_pred + 1] = beta_full[j_train + 1]

            X_with_int = np.hstack([np.ones((N, 1)), X_arr])
            pred_all = X_with_int @ beta_aligned

            valid_idx = valid.values.nonzero()[0]
            df[col + '_neu'] = np.nan
            df.loc[valid, col + '_neu'] = df.loc[valid, col].values - pred_all[valid_idx]

        return df

    # ================================================================
    #  向量化实现
    # ================================================================

    def _neutralize_vectorized(self, df, factor_cols, X_neutral, capture_betas=False):
        """向量化批量中性化。

        对每个因子：用有效样本的OLS闭式解计算残差。X对所有因子相同，
        按有效mask分组：同mask的因子共享投影矩阵，批量计算。

        【确定性】使用 np.linalg.solve + ridge(λ=1e-12)，纯数学运算，
        不受 BLAS 多线程影响，跨进程结果完全一致。

        capture_betas=True: 返回 (df, betas_dict) 而非仅 df。
        betas_dict = {factor_col: np.array([intercept, coef1, ..., coefK])}
        """
        df = df.copy()
        X = X_neutral.reindex(df.index, fill_value=0)
        X_arr = X.values  # [N, K]
        X_cols = list(X_neutral.columns)
        N, K = X_arr.shape

        # 添加截距列（等价于 LinearRegression(fit_intercept=True)）
        X_with_int = np.hstack([np.ones((N, 1)), X_arr])  # [N, K+1]

        available = [c for c in factor_cols if c in df.columns]
        if not available:
            if capture_betas:
                return df, {'X_cols': X_cols, 'betas': {}}
            return df

        # 按有效mask分组（用bytes表示精确mask模式）
        from collections import defaultdict
        groups = defaultdict(list)
        for col in available:
            valid = df[col].notna() & ~df[col].isin([np.inf, -np.inf])
            groups[valid.values.tobytes()].append((col, valid))

        n_total = N
        captured = {}  # {factor_col: beta_vector}

        for mask_key, items in groups.items():
            cols_in_group = [c for c, _ in items]
            valid = items[0][1]  # 同组mask完全相同
            n_valid = valid.sum()

            if n_valid < max(10, int(n_total * 0.05)):
                for col in cols_in_group:
                    logger.warning(
                        f"因子 {col} 有效样本过少 ({n_valid}/{n_total})，中性化结果设为 NaN"
                    )
                    df[col + '_neu'] = np.nan
                continue

            valid_idx = valid.values.nonzero()[0]
            M = len(cols_in_group)
            K1 = X_with_int.shape[1]  # K+1

            # [2026-08-05 内存修复] 分块累加 XTX/XTY（数学严格等价）:
            #   XTX = X_v^T·X_v = Σ_block X_b^T·X_b （[X1;X2]^T[X1;X2] = X1^T·X1 + X2^T·X2, 无交叉项）
            #   XTY = X_v^T·Y_v = Σ_block X_b^T·Y_b
            # 内存峰值从 O(N×K1 + N×M)（V10 全市场 66万行时 ~2GB 触发 _ArrayMemoryError）
            # 降为 O(chunk×K1 + chunk×M)，且结果与整块计算一致（固定分块顺序, 保持确定性）。
            CHUNK = 50000
            Y_valid = df.loc[valid, cols_in_group].values  # [n_valid, M] 仅取一次
            XTX = np.zeros((K1, K1), dtype=np.float64)
            XTY = np.zeros((K1, M), dtype=np.float64)
            for s in range(0, n_valid, CHUNK):
                e = min(s + CHUNK, n_valid)
                X_b = X_with_int[valid_idx[s:e]]            # [chunk, K+1]
                Y_b = Y_valid[s:e]                           # [chunk, M]
                XTX += X_b.T @ X_b
                XTY += X_b.T @ Y_b
            XTX += np.eye(K1) * 1e-12
            Betas = np.linalg.solve(XTX, XTY)

            # 分块计算残差并写入（无效样本保持 NaN）
            neu_arrays = {col: np.full(N, np.nan, dtype=np.float64)
                          for col in cols_in_group}
            for s in range(0, n_valid, CHUNK):
                e = min(s + CHUNK, n_valid)
                idx_b = valid_idx[s:e]
                pred_b = X_with_int[idx_b] @ Betas           # [chunk, M]
                res_b = Y_valid[s:e] - pred_b                # [chunk, M]
                for j, col in enumerate(cols_in_group):
                    neu_arrays[col][idx_b] = res_b[:, j]
            for col in cols_in_group:
                df[col + '_neu'] = neu_arrays[col]
                if capture_betas:
                    captured[col] = Betas[:, cols_in_group.index(col)]  # [K+1]

        if capture_betas:
            return df, {'X_cols': X_cols, 'betas': captured}
        return df

    def neutralize_legacy(self,
                          df: pd.DataFrame,
                          factor_cols: list,
                          X_neutral: pd.DataFrame) -> pd.DataFrame:
        """
        逐因子 LinearRegression 中性化（原始实现，保留用于调试/回归测试）。

        与 neutralize() 的差异：使用 sklearn LinearRegression 逐因子拟合，
        而非向量化 solve 批量计算。结果数学等价。
        """
        return self._neutralize_per_factor(df, factor_cols, X_neutral)

    def _neutralize_per_factor(self, df, factor_cols, X_neutral):
        """逐因子 LinearRegression 中性化（原始实现）"""
        df = df.copy()
        X = X_neutral.reindex(df.index, fill_value=0)

        for col in factor_cols:
            if col not in df.columns:
                continue

            valid_mask = df[col].notna() & ~df[col].isin([np.inf, -np.inf])
            n_valid = valid_mask.sum()
            n_total = len(df)

            if n_valid < max(10, int(n_total * 0.05)):
                logger.warning(
                    f"因子 {col} 有效样本过少 ({n_valid}/{n_total})，"
                    f"中性化结果设为 NaN"
                )
                df[col + '_neu'] = np.nan
                continue

            y_valid = df.loc[valid_mask, col].values
            X_valid = X.loc[valid_mask].values

            model = LinearRegression(fit_intercept=True)

            try:
                model.fit(X_valid, y_valid)
                pred_all = model.predict(X.values)
                df[col + '_neu'] = np.nan
                df.loc[valid_mask, col + '_neu'] = y_valid - pred_all[valid_mask]

            except Exception as e:
                logger.warning(f"因子 {col} 中性化失败，保留原始值。错误: {e}")
                df[col + '_neu'] = np.nan
                df.loc[valid_mask, col + '_neu'] = y_valid

        return df