"""
CSI 300 ETF 期权 QVIX 仓位控制器

数据源: AKShare index_option_300etf_qvix() — 完全免费
预计算: scripts/compute_iv_index.py → data/iv_index.parquet

防未来数据设计:
  - 回测 T 日盘前: 取 parquet 中 < T 的最大日期的 scale 值
  - scale[prev] 基于 QVIX 历史到 prev 日, 全部在 T 日前已知
  - 不对日期做精确匹配, 自动找最近的前一个数据日

架构:
  IVPositionController  ← 顶层（暴露给策略）
    ├── IVDataProvider   ← 数据读取
    ├── IVPercentileCalibrator ← 百分位校准
    └── IVPositionScaler ← 缩放因子计算
"""

import logging
from collections import deque
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class IVDataProvider:
    """IV 数据源抽象层

    回测模式：从预计算的 parquet 读取 scale 列。
    get_scale_at(T) → 返回 parquet 中 < T 的最大日期的 scale 值。
    例如 T=2024-05-22: parquet 中有 2024-05-21 → 返回 scale[2024-05-21]。
    这确保了 T 日盘前使用的全部是 T-1 及之前已知的 QVIX 数据, 无未来数据。

    实时模式：调用 AKShare API → 实时计算。
    """

    def __init__(self, config: dict):
        iv_cfg = config.get('iv', {})
        self.source = iv_cfg.get('source', 'parquet')

        if self.source == 'parquet':
            data_path = iv_cfg.get('data_path', './data/iv_index.parquet')
            self.data_path = Path(data_path)
            self._cache = None
            self._dates = None        # sorted numpy array of dates
            self._scales = None        # corresponding scale values
            self._loaded_count = 0
        elif self.source == 'akshare_realtime':
            self._iv_buffer = deque(maxlen=252)
        else:
            raise ValueError(f"Unknown IV data source: {self.source}")

    def load(self):
        """加载预计算的 IV 指数文件, 构建快速查找数组"""
        if self.source != 'parquet':
            return
        if self._cache is not None:
            return
        if not self.data_path.exists():
            logger.warning(f"IV index file not found: {self.data_path}")
            self._cache = pd.DataFrame()
            self._dates = np.array([])
            self._scales = np.array([])
            self._loaded_count = 0
            return
        df = pd.read_parquet(self.data_path)
        self._cache = df
        # 确保按日期排序
        df = df.sort_index()
        self._dates = df.index.values  # numpy datetime64 array
        self._scales = df['scale'].values.astype(float)
        self._loaded_count = len(df)
        logger.info(f"IV index loaded: {self._loaded_count} entries, "
                    f"range [{df.index[0].date()} → {df.index[-1].date()}]")

    def get_scale_at(self, date) -> float:
        """
        获取回测日期 T 的仓位缩放因子（防未来数据）。

        逻辑: 在 parquet 中找到 < T 的最大日期, 返回其 scale 值。
          - T=2024-05-22: parquet 最大<0522 的是 2024-05-21 → scale[0521]
          - T 之前无数据 → 返回 1.0（满仓）

        scale[T-1] 基于 QVIX 历史到 T-1 日, 全部在 T 日盘前已知。
        """
        if self.source != 'parquet':
            return 1.0
        if self._cache is None:
            self.load()
        if self._loaded_count == 0:
            return 1.0

        d = np.datetime64(pd.Timestamp(date).normalize())
        # searchsorted side='left': 返回 d 应插入的位置 (dates < d 的个数)
        # idx = 个数 - 1 = 最大 < d 的索引
        idx = np.searchsorted(self._dates, d, side='left') - 1
        if idx < 0:
            logger.debug(f"[IVCtrl] {pd.Timestamp(date).date()} "
                         f"no prior data → scale=1.0")
            return 1.0
        prev_date = pd.Timestamp(self._dates[idx]).date()
        scale = float(self._scales[idx])
        logger.debug(f"[IVCtrl] {pd.Timestamp(date).date()} "
                     f"→ prev={prev_date} scale={scale:.3f}")
        return scale

    def get_iv(self, date) -> float | None:
        """获取指定日期的 QVIX 原始值"""
        if self.source != 'parquet':
            return None
        if self._cache is None:
            self.load()
        d = pd.Timestamp(date).normalize()
        if d in self._cache.index:
            return float(self._cache.loc[d, 'qvix_raw'])
        return None

    def get_history(self, start_date, end_date) -> pd.Series:
        """获取历史 QVIX 序列"""
        if self.source != 'parquet':
            return pd.Series(dtype=float)
        if self._cache is None:
            self.load()
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        mask = (self._cache.index >= start) & (self._cache.index <= end)
        return self._cache.loc[mask, 'qvix_smooth']


class IVPercentileCalibrator:
    """滚动窗口百分位校准器"""

    def __init__(self, window: int = 252, min_history: int = 20):
        self.window = window
        self.min_history = min_history
        self.history = deque(maxlen=window)

    def update(self, iv: float):
        self.history.append(iv)

    def get_percentile(self, iv: float) -> float:
        if len(self.history) < self.min_history:
            return 0.5
        count_less = sum(1 for h in self.history if h <= iv)
        return count_less / len(self.history)


class IVPositionScaler:
    """IV → 仓位缩放因子"""

    def __init__(self, pct_low=0.25, pct_high=0.75, s_min=0.25,
                 scaling='linear', sigmoid_k=10.0):
        self.pct_low = pct_low
        self.pct_high = pct_high
        self.s_min = s_min
        self.scaling = scaling
        self.sigmoid_k = sigmoid_k

    def scale(self, percentile: float) -> float:
        if percentile <= self.pct_low:
            return 1.0
        if percentile >= self.pct_high:
            return self.s_min

        if self.scaling == 'linear':
            x = (percentile - self.pct_low) / (self.pct_high - self.pct_low)
            return 1.0 - (1.0 - self.s_min) * x

        elif self.scaling == 'sigmoid':
            midpoint = (self.pct_low + self.pct_high) / 2
            x = 1.0 / (1.0 + np.exp(-self.sigmoid_k * (midpoint - percentile)))
            return self.s_min + (1.0 - self.s_min) * x

        return 1.0


class IVPositionController:
    """IV 仓位控制器（顶层接口）

    用法示例:
        ctrl = IVPositionController(config)
        ctrl.initialize_for_backtest()  # 回测初始化

        # 在 before_market_open 中:
        scale = ctrl.get_scaling_factor(context.current_dt)
        for code in target_weights:
            target_weights[code] *= scale

    config 示例:
        position_controller:
          enabled: true
          type: "iv_based"
          iv:
            source: "parquet"
            data_path: "./data/iv_index.parquet"
            pct_low: 0.25
            pct_high: 0.75
            s_min: 0.25
            scaling: "linear"
    """

    def __init__(self, config: dict):
        iv_cfg = config.get('iv', {})
        self.config = iv_cfg
        self.enabled = config.get('enabled', True)

        # 数据源
        self.data_provider = IVDataProvider(config)

        # 仅实时模式需要这些组件
        if iv_cfg.get('source') == 'akshare_realtime':
            self.calibrator = IVPercentileCalibrator(
                window=iv_cfg.get('lookback_days', 252),
                min_history=iv_cfg.get('min_history_days', 20)
            )
            self.scaler = IVPositionScaler(
                pct_low=iv_cfg.get('pct_low', 0.25),
                pct_high=iv_cfg.get('pct_high', 0.75),
                s_min=iv_cfg.get('s_min', 0.25),
                scaling=iv_cfg.get('scaling', 'linear'),
            )
            self._iv_prev = None
            self._initialized = False
        else:
            self.calibrator = None
            self.scaler = None

    def initialize_for_backtest(self):
        """回测模式初始化：加载预计算 parquet"""
        self.data_provider.load()

    def get_scaling_factor(self, date) -> float:
        """获取当日仓位缩放因子（盘前调用, 只用T日之前已知数据）

        回测模式（parquet source）：
          - 查 parquet 中 < date 的最大日期, 返回其 scale 值
          - 例如 date=2024-05-22: parquet 最大<0522 是 0521 → scale[0521]
          - scale[0521] 基于 QVIX 历史到 0521, 全部在 0522 盘前已知

        实时模式（akshare_realtime source）：
          - 调用 AKShare API 获取最新 QVIX
          - 实时计算百分位和缩放因子
        """
        # 回测模式：查最近的前一日 scale
        if self.data_provider.source == 'parquet':
            scale = self.data_provider.get_scale_at(date)
            return scale

        # 实时模式：实时计算
        iv_raw = self.data_provider.get_iv(date)
        if iv_raw is None:
            if not self._initialized:
                return 1.0
            iv_smooth = self._iv_prev
        else:
            alpha = 2.0 / (self.config.get('smooth_days', 5) + 1)
            iv_smooth = (alpha * iv_raw
                         + (1 - alpha) * (self._iv_prev or iv_raw))
        self._iv_prev = iv_smooth

        if not self._initialized:
            self._init_history(date)
            self._initialized = True

        self.calibrator.update(iv_smooth)
        pct = self.calibrator.get_percentile(iv_smooth)
        scale = self.scaler.scale(pct)

        logger.info(f"[IVCtrl] {pd.Timestamp(date).date()} IV={iv_smooth:.2f} "
                    f"pct={pct:.2f} scale={scale:.2f}")
        return scale

    def apply(self, weights: dict, date=None) -> dict:
        """对权重字典应用缩放"""
        scale = self.get_scaling_factor(date) if date else 1.0
        if scale >= 0.99:
            return weights
        return {k: v * scale for k, v in weights.items()}

    def get_state(self) -> dict:
        """返回诊断状态"""
        if self.data_provider.source == 'parquet':
            return {
                'source': 'parquet',
                'entries': self.data_provider._loaded_count,
            }
        return {
            'source': 'akshare_realtime',
            'iv_current': self._iv_prev,
            'history_len': len(self.calibrator.history) if self.calibrator else 0,
        }

    def _init_history(self, current_date):
        """用历史数据初始化滚动窗口（仅实时模式）"""
        start = pd.Timestamp(current_date) - pd.Timedelta(days=365 * 2)
        hist = self.data_provider.get_history(start, current_date)
        for _, iv in hist.items():
            self.calibrator.update(iv)
