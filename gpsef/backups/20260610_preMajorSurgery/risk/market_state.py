"""
市场状态分类与仓位控制模块 (Market State Guard)

基于「威科夫2.0策略_尾部风险对冲方案 V1.0」实现 Layer 1-2。
通过均线趋势 + 波动率（可选贴水率）三维信号，
将市场分为 GREEN / YELLOW / RED / BLACK 四档，按档位控制总仓位。

设计原则:
  - 即插即用: 不依赖特定数据源，通过接口注入
  - 确定性: 相同输入 → 相同输出，可复现
  - 降级优雅: 数据缺失时自动回退到保守模式
  - 可测试: 所有核心方法都可独立测试

用法:
    from gpsef.risk.market_state import MarketStateGuard

    guard = MarketStateGuard(index_close_arr, config={'enable_basis': False})

    for date in trading_dates:
        guard.update(date, index_close_upto(date))
        cap = guard.position_cap          # 1.0 / 0.6 / 0.3 / 0.1
        weight = base_weight * cap

参考:
    威科夫2.0策略_尾部风险对冲方案.md.docx
    Sections 3-5: 三维信号构建, 市场状态分类器, 仓位映射规则

Version: 2.0.0
Date: 2026-06-01
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Union, Sequence
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  数据类型
# ════════════════════════════════════════════════════════════════════

class MarketState(Enum):
    """市场状态枚举"""
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"
    BLACK = "BLACK"

    @property
    def position_cap(self) -> float:
        """每个状态对应的仓位上限"""
        return _STATE_POSITION[self]


@dataclass
class SignalScore:
    """单日三维信号评分"""
    ma_trend: float       # 均线趋势 0-100 (越高越危险)
    volatility: float      # 波动率 0-100
    basis_rate: float = 0.0  # 贴水率 0-100 (仅在3信号模式下有效)
    composite: float = 0.0   # 综合评分 0-100


@dataclass
class GuardStatus:
    """单日 Guard 状态快照"""
    state: MarketState
    score: SignalScore
    position_cap: float
    current_vol_pct: float          # 年化波动率 (%)
    current_basis_pct: Optional[float]  # 贴水率 (%)
    is_upgrade: bool                # 当日是否升级
    is_downgrade: bool              # 当日是否降级


# ════════════════════════════════════════════════════════════════════
#  常量 (文档 Section 5)
# ════════════════════════════════════════════════════════════════════

_STATE_POSITION: Dict[MarketState, float] = {
    MarketState.GREEN: 1.00,
    MarketState.YELLOW: 0.60,
    MarketState.RED: 0.30,
    MarketState.BLACK: 0.10,
}

# 双信号权重 (无贴水率, 文档 Section 3 变体)
_SIGNAL_WEIGHTS_2 = {'ma_trend': 0.35, 'vol': 0.65}

# 三信号权重 (有贴水率, 文档 Section 4.1)
# V3: 波动率核心驱动 (0.50) + 贴水率前置预警 (0.30) + 均线确认 (0.20)
# 配合否决权机制：贴水率>75→RED，贴水率>75且波动率>70→BLACK，波动率>80→至少YELLOW
_SIGNAL_WEIGHTS_3 = {'ma_trend': 0.20, 'basis': 0.30, 'vol': 0.50}

# 动态权重阈值: 贴水率评分超过此值后，权重向贴水率倾斜
_DYNAMIC_BASIS_THRESHOLD = 60  # 贴水率评分>60 → 提升贴水权重
_DYNAMIC_BASIS_MAX_BOOST = 0.15  # 最多从波动率挪0.15给贴水率

# 降级确认天数 (文档 Section 4.3)
_DOWNGRADE_CONFIRM_DAYS: Dict[MarketState, int] = {
    MarketState.GREEN: 0,
    MarketState.YELLOW: 3,
    MarketState.RED: 3,
    MarketState.BLACK: 5,
}

_STATE_ORDER: Tuple[MarketState, ...] = (
    MarketState.GREEN, MarketState.YELLOW,
    MarketState.RED, MarketState.BLACK,
)

# 波动率阈值 (文档 Section 3.3)
_VOL_THRESHOLDS = [
    (18, 0), (25, 15), (32, 40), (40, 70), (float('inf'), 100)
]

# 贴水率阈值 (文档 Section 3.2.3)
_BASIS_THRESHOLDS = [
    (-3, 0), (-6, 10), (-10, 40), (-15, 70), (float('-inf'), 100)
]

# 默认年化波动率 (数据不足时)
_DEFAULT_VOL = 25.0

# 状态emoji
_STATE_EMOJI = {
    MarketState.GREEN: '\U0001f7e2',   # 🟢
    MarketState.YELLOW: '\U0001f7e1',  # 🟡
    MarketState.RED: '\U0001f534',     # 🔴
    MarketState.BLACK: '\u26ab',       # ⚫
}


# ════════════════════════════════════════════════════════════════════
#  核心类
# ════════════════════════════════════════════════════════════════════

class MarketStateGuard:
    """市场状态分类器 + 仓位控制器。

    每日运行 `update(date, close_prices)` 评估市场风险，
    通过状态机 (升级即时 / 降级需确认) 控制总仓位上限。

    仅依赖收盘价序列，不直接访问数据源。
    策略层负责喂入数据，保证解耦。

    Attributes:
        state: 当前市场状态
        position_cap: 当前仓位上限 (0.1-1.0)
        current_vol_pct: 最新年化波动率 (%)
        scores: 最新三维信号评分
    """

    # ═══════════════════════════════════════════════════════════════
    #  初始化与配置
    # ═══════════════════════════════════════════════════════════════

    def __init__(self,
                 index_close_prices: Optional[np.ndarray] = None,
                 *,
                 config: dict = None,
                 basis_date_pct_pairs: Optional[List[Tuple[datetime, float]]] = None):
        """初始化市场状态守卫。

        Args:
            index_close_prices:
                监控指数的历史收盘价序列 (从早到晚排列)。
                若为 None，进入 fallback 模式 (始终 YELLOW)。
            config:
                可选配置字典，支持的 key:
                - enable_basis: 是否启用贴水率信号 (需额外的 basis_date_pct_pairs)
                - debug: 启用调试日志
            basis_date_pct_pairs:
                贴水率历史数据 [(date, basis_pct), ...]。
                basis_pct = (期货收盘-现货收盘)/现货收盘×100，负数=贴水。
                若为 None，自动回退到双信号模式。
        """
        config = config or {}
        self._debug = config.get('debug', False)

        # ── 指数数据 ──
        self._close_arr: Optional[np.ndarray] = None
        if index_close_prices is not None and len(index_close_prices) > 0:
            self._close_arr = np.asarray(index_close_prices, dtype=np.float64)
            self._fallback_mode = False
        else:
            self._fallback_mode = True
            logger.warning(
                "MarketStateGuard: 无指数数据，进入 fallback 模式 (始终 YELLOW, 仓位上限 60%)"
            )

        # ── 贴水率数据 ──
        self._enable_basis = config.get('enable_basis', False)
        self._basis_lookup: Dict[date, float] = {}
        self._basis_values: List[float] = []
        if basis_date_pct_pairs and self._enable_basis:
            for dt, pct in basis_date_pct_pairs:
                d = dt.date() if isinstance(dt, datetime) else dt
                self._basis_lookup[d] = float(pct)
                self._basis_values.append(float(pct))
            if self._basis_lookup:
                logger.info(
                    "MarketStateGuard: 贴水率数据已加载 (%d 条)", len(self._basis_lookup)
                )
            else:
                self._enable_basis = False

        # ── 信号模式 ──
        self._signal_weights = _SIGNAL_WEIGHTS_3 if self._enable_basis else _SIGNAL_WEIGHTS_2
        mode_name = '3-signal (MA+Basis+Vol)' if self._enable_basis else '2-signal (MA+Vol)'
        if self._fallback_mode:
            mode_name = 'FALLBACK (YELLOW only)'

        # ── 状态机 ──
        self._state: MarketState = MarketState.GREEN
        self._downgrade_counter: int = 0

        # ── 当前评分 ──
        self._scores = SignalScore(ma_trend=0, volatility=0, basis_rate=0, composite=0)
        self._current_vol_pct: float = _DEFAULT_VOL
        self._current_basis_pct: Optional[float] = None

        # ── 历史序列 ──
        self._vol_history: List[float] = []
        self._basis_history: List[float] = list(self._basis_values) if self._basis_values else []

        # ── 状态日志 ──
        self._state_log: List[GuardStatus] = []

        # ── 最近一次状态变化标记 ──
        self._last_upgrade: bool = False
        self._last_downgrade: bool = False

        logger.info(
            "MarketStateGuard 初始化: mode=%s, index_bars=%d",
            mode_name,
            len(self._close_arr) if self._close_arr is not None else 0
        )

    # ═══════════════════════════════════════════════════════════════
    #  属性
    # ═══════════════════════════════════════════════════════════════

    @property
    def state(self) -> MarketState:
        """当前市场状态"""
        return self._state

    @property
    def position_cap(self) -> float:
        """当前仓位上限 (0.1-1.0)"""
        if self._fallback_mode:
            return 0.6  # fallback: YELLOW
        return _STATE_POSITION[self._state]

    @property
    def current_vol_pct(self) -> float:
        """最新年化波动率 (%)"""
        return self._current_vol_pct

    @property
    def current_basis_pct(self) -> Optional[float]:
        """最新贴水率 (%), None = 无数据"""
        return self._current_basis_pct

    @property
    def scores(self) -> SignalScore:
        """最新三维信号评分"""
        return self._scores

    @property
    def signal_mode(self) -> str:
        """信号模式: '2-signal', '3-signal', 或 'fallback'"""
        if self._fallback_mode:
            return 'fallback'
        return '3-signal' if self._enable_basis else '2-signal'

    @property
    def state_history(self) -> List[GuardStatus]:
        """完整的状态变化日志"""
        return list(self._state_log)

    @property
    def vol_history(self) -> List[float]:
        """波动率历史序列"""
        return list(self._vol_history)

    # ═══════════════════════════════════════════════════════════════
    #  每日更新 (主入口)
    # ═══════════════════════════════════════════════════════════════

    def update(self,
               dt: Union[datetime, date],
               close_prices_upto: Optional[np.ndarray] = None) -> MarketState:
        """每日盘后更新市场状态。

        应在 handle_bar / before_trading 开始处调用。
        升级即时生效 (T+1执行)，降级需连续确认 (YELLOW/RED 3天, BLACK 5天)。

        Args:
            dt: 当前日期
            close_prices_upto: 截至当日(含)的指数收盘价序列。
                               若初始化时已传入完整序列，可省略。
                               若提供，会更新内部序列 (用于实时/增量场景)。

        Returns:
            更新后的市场状态
        """
        # ── Fallback 模式 ──
        if self._fallback_mode:
            self._state = MarketState.YELLOW
            self._scores = SignalScore(ma_trend=40, volatility=40, composite=45)
            self._log_state(dt, False, False)
            return self._state

        # ── 更新价格序列 ──
        if close_prices_upto is not None and len(close_prices_upto) > 0:
            self._close_arr = np.asarray(close_prices_upto, dtype=np.float64)

        if self._close_arr is None:
            logger.warning("MarketStateGuard.update: 无价格数据，维持 %s", self._state.name)
            return self._state

        # ── 记录状态变更前的状态 ──
        prev_state = self._state

        # ── 1. 计算三维信号 ──
        self._scores.ma_trend = self._compute_ma_trend()
        self._scores.volatility = self._compute_volatility()
        self._scores.basis_rate = self._compute_basis(dt) if self._enable_basis else 0.0

        # ── 2. 综合评分 (三信号时使用动态权重) ──
        w = self._compute_dynamic_weights() if self._enable_basis else self._signal_weights
        self._scores.composite = (
            self._scores.ma_trend * w['ma_trend'] +
            self._scores.basis_rate * w.get('basis', 0.0) +
            self._scores.volatility * w['vol']
        )
        self._scores.composite = float(np.clip(self._scores.composite, 0.0, 100.0))

        # ── 3. 判定目标状态 ──
        target = self._classify(self._scores.composite,
                                self._scores.basis_rate,
                                self._scores.volatility)

        # ── 4. 状态机转换 ──
        self._transition(target)

        # ── 5. 记录 ──
        self._last_upgrade = (self._state.value != prev_state.value and
                              _STATE_ORDER.index(self._state) > _STATE_ORDER.index(prev_state))
        self._last_downgrade = (self._state.value != prev_state.value and
                                _STATE_ORDER.index(self._state) < _STATE_ORDER.index(prev_state))
        self._log_state(dt, self._last_upgrade, self._last_downgrade)

        if self._debug:
            logger.info(
                "[%s] %s | score=%.1f MA=%.1f Basis=%.1f Vol=%.1f | "
                "vol=%.1f%% | cap=%.0f%% | %s",
                dt if isinstance(dt, date) else dt.date(),
                _STATE_EMOJI[self._state], self._scores.composite,
                self._scores.ma_trend, self._scores.basis_rate, self._scores.volatility,
                self._current_vol_pct, self.position_cap * 100,
                '\u2191UPGRADE' if self._last_upgrade else
                ('\u2193DOWNGRADE' if self._last_downgrade else 'hold')
            )

        return self._state

    # ═══════════════════════════════════════════════════════════════
    #  仓位控制接口
    # ═══════════════════════════════════════════════════════════════

    def should_reduce_from(self, prev_state: Union[str, MarketState]) -> bool:
        """判断是否发生了风险升级 (需要紧急减仓)。

        用于非调仓日的紧急减仓判断。升级当日即使不是调仓日也应执行减仓。

        Args:
            prev_state: 上一次 handle_bar 时的状态 (字符串或枚举)

        Returns:
            True = 发生了风险升级，需要紧急减仓

        Example:
            prev = guard.state
            guard.update(today, prices)
            if guard.should_reduce_from(prev):
                # 紧急减仓：将每个持仓降到 target = current_weight * guard.position_cap
                for code in positions:
                    broker.order_target_percent(code, current_weight[code] * guard.position_cap)
        """
        if isinstance(prev_state, str):
            prev_state = MarketState(prev_state)
        return _STATE_ORDER.index(self._state) > _STATE_ORDER.index(prev_state)

    def was_upgraded(self) -> bool:
        """最近一次 update 是否触发了升级"""
        return self._last_upgrade

    def was_downgraded(self) -> bool:
        """最近一次 update 是否触发了降级"""
        return self._last_downgrade

    # ═══════════════════════════════════════════════════════════════
    #  状态查询
    # ═══════════════════════════════════════════════════════════════

    def status_report(self) -> str:
        """生成单行状态摘要 (适合日志)"""
        emoji = _STATE_EMOJI.get(self._state, '??')
        parts = [
            f"{emoji} {self._state.name}",
            f"score={self._scores.composite:.1f}",
            f"MA={self._scores.ma_trend:.1f}",
        ]
        if self._enable_basis:
            parts.append(f"Basis={self._scores.basis_rate:.1f}")
        parts.append(f"Vol={self._scores.volatility:.1f}")
        parts.append(f"cap={self.position_cap*100:.0f}%")
        return " ".join(parts)

    def to_dict(self) -> dict:
        """导出当前状态为字典 (适合序列化/监控)"""
        return {
            'state': self._state.name,
            'position_cap': self.position_cap,
            'composite_score': round(self._scores.composite, 2),
            'ma_score': round(self._scores.ma_trend, 2),
            'vol_score': round(self._scores.volatility, 2),
            'basis_score': round(self._scores.basis_rate, 2) if self._enable_basis else None,
            'current_vol_pct': round(self._current_vol_pct, 2),
            'current_basis_pct': round(self._current_basis_pct, 2) if self._current_basis_pct is not None else None,
            'signal_mode': self.signal_mode,
            'downgrade_counter': self._downgrade_counter,
        }

    # ═══════════════════════════════════════════════════════════════
    #  测试辅助
    # ═══════════════════════════════════════════════════════════════

    def reset(self) -> None:
        """重置状态机 (用于跨验证折叠测试)。

        保留价格序列和波动率历史 (维持数据统计连续性)，
        仅重置状态机回到 GREEN。
        """
        self._state = MarketState.GREEN
        self._downgrade_counter = 0
        self._state_log.clear()
        self._last_upgrade = False
        self._last_downgrade = False

    def reset_all(self) -> None:
        """完全重置 (用于全新测试)。

        清除所有状态、评分、历史和日志。
        """
        self._state = MarketState.GREEN
        self._downgrade_counter = 0
        self._scores = SignalScore(ma_trend=0, volatility=0, basis_rate=0, composite=0)
        self._current_vol_pct = _DEFAULT_VOL
        self._current_basis_pct = None
        self._vol_history.clear()
        self._basis_history.clear()
        self._state_log.clear()
        self._last_upgrade = False
        self._last_downgrade = False
        self._fallback_mode = (self._close_arr is None)

    def force_state(self, state: Union[str, MarketState]) -> None:
        """强置状态 (仅用于测试状态机转换)。"""
        if isinstance(state, str):
            state = MarketState(state)
        self._state = state
        self._downgrade_counter = 0

    def force_vol_history(self, values: List[float]) -> None:
        """注入波动率历史 (仅用于测试)。"""
        self._vol_history = list(values)

    # ═══════════════════════════════════════════════════════════════
    #  信号计算 (文档 Section 3)
    # ═══════════════════════════════════════════════════════════════

    def _compute_dynamic_weights(self) -> Dict[str, float]:
        """动态权重: 贴水率进入预警区间时自动提升其权重。
        
        原则:
        - 平时 (basis≤60): 波动率主导 (0.20/0.30/0.50)
        - 预警 (basis>60): 贴水率权重从0.30→最高0.45，从波动率中扣减
        - 否决权兜底: basis>75→RED, basis>75且vol>70→BLACK (权重无关)
        """
        basis_score = self._scores.basis_rate
        w_ma = _SIGNAL_WEIGHTS_3['ma_trend']
        w_vol = _SIGNAL_WEIGHTS_3['vol']
        w_basis = _SIGNAL_WEIGHTS_3['basis']

        if basis_score > _DYNAMIC_BASIS_THRESHOLD:
            boost = min((basis_score - _DYNAMIC_BASIS_THRESHOLD) / 40.0 * _DYNAMIC_BASIS_MAX_BOOST,
                        _DYNAMIC_BASIS_MAX_BOOST)
            w_basis += boost
            w_vol -= boost

        total = w_ma + w_vol + w_basis
        return {'ma_trend': w_ma / total, 'basis': w_basis / total, 'vol': w_vol / total}

    def _compute_ma_trend(self) -> float:
        """均线趋势评分 (0-100, 越高越空头/危险)。

        文档 Section 3.1 规则:
          - close <= MA20: +40
          - close <= MA60: +35
          - MA20 <= MA60 (死叉): +25
          - 偏离MA20 < -5%:  +15 (上限55)
          - 偏离MA60 < -8%:  +15 (上限50)
          - 偏离MA20 < -10%: +10 (上限65)
          → 总分截断到100
        """
        arr = self._close_arr
        if len(arr) < 60:
            return 40.0

        close = arr[-1]
        ma20 = np.mean(arr[-20:])
        ma60 = np.mean(arr[-60:])

        s1 = 0.0 if close > ma20 else 40.0
        s2 = 0.0 if close > ma60 else 35.0
        s3 = 0.0 if ma20 > ma60 else 25.0

        dist20 = (close / ma20 - 1.0) * 100.0
        dist60 = (close / ma60 - 1.0) * 100.0

        if dist20 < -5.0:
            s1 = min(s1 + 15.0, 55.0)
        if dist60 < -8.0:
            s2 = min(s2 + 15.0, 50.0)
        if dist20 < -10.0:
            s1 = min(s1 + 10.0, 65.0)

        return float(np.clip(s1 + s2 + s3, 0.0, 100.0))

    def _compute_volatility(self) -> float:
        """波动率评分 (0-100, 越高越恐慌)。

        文档 Section 3.3:
          绝对水平 (30%): <18%=0, <25%=15, <32%=40, <40%=70, >=40%=100
          历史分位数 (40%): 252日窗口中当前值的分位
          变化速率 (30%): 5日变动>8%→90, >4%→60, >2%→30

        Returns:
            波动率评分 0-100
        """
        arr = self._close_arr
        if len(arr) < 25:
            self._current_vol_pct = _DEFAULT_VOL
            return 40.0

        # 20日实现波动率 (年化, ddof=1 for unbiased)
        log_ret = np.diff(np.log(arr))
        if len(log_ret) < 20:
            self._current_vol_pct = _DEFAULT_VOL
            return 40.0

        daily_vol = np.std(log_ret[-20:], ddof=1)
        annual_vol = daily_vol * np.sqrt(252) * 100.0
        self._current_vol_pct = annual_vol
        self._vol_history.append(annual_vol)
        vol = annual_vol

        # ── 绝对水平 ──
        abs_score = 100.0
        for threshold, score in _VOL_THRESHOLDS:
            if vol < threshold:
                abs_score = float(score)
                break

        # ── 历史分位数 ──
        lookback = 252
        if len(self._vol_history) >= lookback:
            recent = self._vol_history[-lookback:]
            pct = sum(1.0 for x in recent if x < vol) / lookback * 100.0
            pct_score = float(np.clip(pct, 0.0, 100.0))
        else:
            pct_score = 50.0

        # ── 变化速率 (5日) ──
        if len(self._vol_history) >= 6:
            change_5d = vol - self._vol_history[-6]
            if change_5d > 8.0:
                speed_score = 90.0
            elif change_5d > 4.0:
                speed_score = 60.0
            elif change_5d > 2.0:
                speed_score = 30.0
            else:
                speed_score = 0.0
        else:
            speed_score = 0.0

        return float(np.clip(abs_score * 0.30 + pct_score * 0.40 + speed_score * 0.30, 0.0, 100.0))

    def _compute_basis(self, dt: Union[datetime, date]) -> float:
        """贴水率评分 (0-100, 越贴水/越危险)。

        文档 Section 3.2.3:
          绝对水平 (35%): >-3%=0, >-6%=10, >-10%=40, >-15%=70, <=-15%=100
          历史分位数 (40%): 120日窗口
          变化速率 (25%): 5日走阔<-3%→80, <-1.5%→50, <-0.5%→20

        Returns:
            贴水率评分 0-100
        """
        if not self._basis_history:
            return 0.0

        d = dt.date() if isinstance(dt, datetime) else dt
        basis = self._basis_lookup.get(d)

        if basis is None:
            # 非交易日或数据缺失：用最近值
            all_dates = sorted(self._basis_lookup.keys())
            for past_d in reversed(all_dates):
                if past_d <= d:
                    basis = self._basis_lookup[past_d]
                    break
        if basis is None:
            self._current_basis_pct = None
            return 0.0

        self._current_basis_pct = basis

        # ── 绝对水平 (basis 通常为负数) ──
        abs_score = 100.0
        base_basis = float(basis)
        for threshold, score in _BASIS_THRESHOLDS:
            if base_basis > threshold:
                abs_score = float(score)
                break

        # ── 历史分位数 ──
        lookback = 120
        hist = self._basis_history
        n = len(hist)
        if n >= lookback:
            recent = hist[-lookback:]
            # fraction of history that is LESS dangerous (higher basis) than current
            # → high pct means current is very bad (extreme low basis)
            pct = sum(1.0 for x in recent if x > base_basis) / lookback * 100.0
            pct_score = float(np.clip(pct, 0.0, 100.0))
        else:
            pct_score = 50.0

        # ── 变化速率 (5日) ──
        if n >= 6:
            change_5d = base_basis - hist[-6]
            if change_5d < -3.0:
                speed_score = 80.0
            elif change_5d < -1.5:
                speed_score = 50.0
            elif change_5d < -0.5:
                speed_score = 20.0
            else:
                speed_score = 0.0
        else:
            speed_score = 0.0

        return float(np.clip(abs_score * 0.35 + pct_score * 0.40 + speed_score * 0.25, 0.0, 100.0))

    # ═══════════════════════════════════════════════════════════════
    #  状态分类与状态机 (文档 Section 4)
    # ═══════════════════════════════════════════════════════════════

    def _classify(self, composite: float, basis: float, vol: float) -> MarketState:
        """判定目标市场状态 (文档 Section 4.2)。

        否决权规则:
          - basis > 75 且 vol > 70 → BLACK
          - basis > 75 → RED
          - vol > 80 → RED (composite>=40) 或 YELLOW

        常规规则:
          - composite < 30 → GREEN
          - composite < 60 → YELLOW
          - composite < 80 → RED
          - else → BLACK
          - GREEN 需额外满足 vol < 50

        文档 Section 4.2 设计要点:
          贴水率评分 > 75 → 直接升级 RED (贴水是最强前置信号)
          波动率评分 > 80 → 至少 YELLOW
          两者同时 > 70 → 直接 BLACK
        """
        # 否决权 (仅在3信号模式下)
        if self._enable_basis:
            if basis > 75.0 and vol > 70.0:
                return MarketState.BLACK
            if basis > 75.0:
                return MarketState.RED

        # 常规分类
        if composite < 30.0:
            target = MarketState.GREEN
        elif composite < 60.0:
            target = MarketState.YELLOW
        elif composite < 80.0:
            target = MarketState.RED
        else:
            target = MarketState.BLACK

        # 波动率否决: vol>80 将 GREEN/YELLOW 升级, 但不覆盖 RED/BLACK
        if vol > 80.0 and target in (MarketState.GREEN, MarketState.YELLOW):
            target = MarketState.RED if composite >= 40.0 else MarketState.YELLOW

        # GREEN 安全条件: vol >= 50 不能 GREEN
        if target == MarketState.GREEN and vol >= 50.0:
            target = MarketState.YELLOW

        return target

    def _transition(self, target: MarketState) -> None:
        """状态机转换 (文档 Section 4.3)。

        升级: 即时生效 (无论跳几级)
        降级: 需连续确认天数达标，每次只降一级
        """
        current_idx = _STATE_ORDER.index(self._state)
        target_idx = _STATE_ORDER.index(target)

        if target_idx > current_idx:
            # 升级 → 立即生效
            self._state = target
            self._downgrade_counter = 0
        elif target_idx < current_idx:
            # 降级 → 需连续确认
            self._downgrade_counter += 1
            required = _DOWNGRADE_CONFIRM_DAYS[self._state]
            if self._downgrade_counter >= required:
                # 降一级 (不跳级)
                self._state = _STATE_ORDER[current_idx - 1]
                self._downgrade_counter = 0
        else:
            # 目标=当前 → 计数器归零
            self._downgrade_counter = 0

    # ═══════════════════════════════════════════════════════════════
    #  内部辅助
    # ═══════════════════════════════════════════════════════════════

    def _log_state(self, dt: Union[datetime, date], is_upgrade: bool, is_downgrade: bool):
        """记录状态快照"""
        status = GuardStatus(
            state=self._state,
            score=SignalScore(
                ma_trend=round(self._scores.ma_trend, 2),
                volatility=round(self._scores.volatility, 2),
                basis_rate=round(self._scores.basis_rate, 2),
                composite=round(self._scores.composite, 2),
            ),
            position_cap=self.position_cap,
            current_vol_pct=round(self._current_vol_pct, 2),
            current_basis_pct=(
                round(self._current_basis_pct, 2)
                if self._current_basis_pct is not None else None
            ),
            is_upgrade=is_upgrade,
            is_downgrade=is_downgrade,
        )
        self._state_log.append(status)


# ════════════════════════════════════════════════════════════════════
#  工厂函数 — 从现有 data_provider 创建
# ════════════════════════════════════════════════════════════════════

def create_guard_from_provider(
    data_provider,
    index_code: str = 'sh.000300',
    config: dict = None,
) -> MarketStateGuard:
    """从 LocalDataProvider 创建 MarketStateGuard。

    自动从 index_daily/ 目录加载指数数据，并尝试加载 basis_history.csv。

    Args:
        data_provider: LocalDataProvider 实例
        index_code: 监控指数代码 (如 'sh.000300', 'sh.000905')
        config: 额外配置

    Returns:
        配置好的 MarketStateGuard 实例
    """
    config = dict(config or {})

    # ── 加载指数日线 ──
    import os
    import pandas as pd

    base = os.path.dirname(data_provider.raw_daily_path)
    index_dir = os.path.join(base, 'index_daily')
    safe = index_code.replace('.', '')
    close_prices = None

    for ext in ('.parquet', '.csv'):
        fp = os.path.join(index_dir, f'{safe}{ext}')
        if os.path.exists(fp):
            try:
                if ext == '.parquet':
                    df = pd.read_parquet(fp)
                else:
                    df = pd.read_csv(fp)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                close_prices = df['close'].values.astype(np.float64)
                logger.info("create_guard_from_provider: %s loaded (%d bars)", index_code, len(close_prices))
                break
            except Exception as e:
                logger.warning("Failed to load %s: %s", fp, e)

    # ── 加载贴水率 ──
    basis_pairs = None
    basis_csv = config.pop('basis_csv', os.path.join(base, 'basis_history.csv'))
    if os.path.exists(basis_csv):
        try:
            df = pd.read_csv(basis_csv, parse_dates=['date'])
            basis_pairs = [(row['date'], float(row['ic_basis'])) for _, row in df.iterrows()]
            config['enable_basis'] = True
        except Exception as e:
            logger.warning("Failed to load basis CSV: %s", e)

    return MarketStateGuard(
        index_close_prices=close_prices,
        config=config,
        basis_date_pct_pairs=basis_pairs,
    )
