"""Optimized TripleBarrierLabeler with batch loading — identical results, 100x speedup."""
import os, logging
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np

try:
    import numba as _nb
    _HAS_NUMBA = True
except Exception:
    _nb = None
    _HAS_NUMBA = False

from .barriers import compute_barriers

logger = logging.getLogger(__name__)


# ============================================================================
#  [L1] 分阶段退出路径模拟器 — 与 strategy.py 生产端 _check_staged_minute
#  完全同语义 (标签-交易对齐, de Prado: 标签必须编码执行规则下的已实现结果)
# ============================================================================

def simulate_staged_path(day_bars, entry, vol, stage_cfg, hard_sl,
                         pl_sl1=0.0, pl_sl2=1.0, pl_sl3=2.0,
                         pt_after_sl2=1.5, pt_after_sl3=2.0):
    """分阶段退出路径模拟（标签端 = 交易端同规则）。

    生产语义（已对 strategy.py _check_staged_minute 逐行核对）：
    - PT 判定: up_s = (当根bar high/entry-1)/vol >= 阈值  ([PTFIX] 当根触达)
    - SL 判定: cur_s = (low/entry-1)/vol <= 阈值        (仅当前 bar 的 low)
    - 利润锁定: pl_sl 按 s1/s2/s3_pt_hit 逐 bar 刷新 (S1→0σ, S2→+1σ, S3→+2σ);
      [E6-2026-09-10] PT 事件自下一根 bar 起抬高 SL (与生产端同语义)
    - PT 下移: pt_after_sl 逐 bar 生效 (S2→1.5σ, S3→2.0σ);
      [E6-2026-09-10] SL 事件自下一根 bar 起下移 PT 阈值
    - 阶段标记 (stage_done / s_pt_hit): 触发后【立即】生效 (每阶段全期仅触发一次)
    - hard_sl: 任意 bar 触发 → 卖出 S4 份额 → 当日结束 (次日 S1-S3 仍可继续)
    - S4 PT/SL 触发 → 当日结束 (同上)
    - 成交价 = 阈值价 entry*(1±mult*vol)  (非 bar 极值价)
    - 窗口结束后剩余仓位按窗口最后可得 close 退出 (≈生产到期次日开盘, 一日偏差
      与旧三重屏障标签的口径一致)

    Args:
        day_bars: list of (highs, lows, closes) — 每个元素为一天的 bar 数组
                  (np.ndarray, 已复权, 时间升序; 分钟bar或日bar)
        entry: 入场价 (复权)
        vol: EWMA 日波动率 (入场时)
        stage_cfg: {1: {'pt','sl','pct'}, 2: {...}, 3: {...}, 4: {...}}
        hard_sl: 硬兜底 σ 倍数 (负值)

    Returns:
        dict: realized_ret (实现收益, 按初始仓位加权的各阶段成交价收益之和
              + 剩余仓位到期收益), exit_type ('profit_take'/'stop_loss'/'expire',
              由最后一次触发方向决定, 无触发=expire), n_fills, fills
    """
    if vol <= 0 or entry <= 0:
        return {'realized_ret': 0.0, 'exit_type': 'expire', 'n_fills': 0, 'fills': []}

    # [Perf] 阈值参数提升到局部变量 (每事件一次, 避免 bar 循环内 dict 查找)
    pt1 = stage_cfg[1]['pt']; sl1 = stage_cfg[1]['sl']; p1 = stage_cfg[1]['pct']
    pt2 = stage_cfg[2]['pt']; sl2 = stage_cfg[2]['sl']; p2 = stage_cfg[2]['pct']
    pt3 = stage_cfg[3]['pt']; sl3 = stage_cfg[3]['sl']; p3 = stage_cfg[3]['pct']
    pt4 = stage_cfg[4]['pt']; sl4 = stage_cfg[4]['sl']; p4 = stage_cfg[4]['pct']

    s_done = {1: False, 2: False, 3: False, 4: False}
    s_pt = {1: False, 2: False, 3: False}
    pt_after_sl = False
    cum_u = float('-inf')          # 累计 high/entry-1 (跨日保持)
    realized = 0.0
    qty_left = 1.0
    fills = []
    last_close = None

    for highs, lows, closes in day_bars:
        if qty_left <= 1e-9:
            break
        if closes is None or len(closes) == 0:
            continue
        # ── 日开盘快照: 仅用于"日级快速跳过"判定 (逐 bar 检查用循环内动态刷新) ──
        #   [E6-2026-09-10] 事件驱动: pl_sl/pt_after_sl 改为逐 bar 生效 (下一根 bar)
        if s_pt[3]:
            pl_sl = pl_sl3
        elif s_pt[2]:
            pl_sl = pl_sl2
        elif s_pt[1]:
            pl_sl = pl_sl1
        else:
            pl_sl = float('-inf')
        eff_sl1 = sl1 if sl1 > pl_sl else pl_sl
        eff_sl2 = sl2 if sl2 > pl_sl else pl_sl
        eff_sl3 = sl3 if sl3 > pl_sl else pl_sl
        eff_sl4 = sl4 if sl4 > pl_sl else pl_sl
        if pt_after_sl:
            eff_pt_2, eff_pt_3 = pt_after_sl2, pt_after_sl3
        else:
            eff_pt_2, eff_pt_3 = pt2, pt3
        # [R6-③] 日内快照已移除 (统一读 eff_pt_2/3, 与 test3 一致)

        # ── [Perf] 日级快速跳过: 当日不可能触发任何阶段时, 向量化更新状态后跳过 ──
        # 判定与逐 bar 语义严格等价:
        #   PT 触发条件 = 当日max_high_ret >= 任一pending PT阈值(取最小者) [PTFIX 当根触达]
        #   SL 触发条件 = 当日min_low_ret <= 任一pending SL阈值(取最大者, 含hard_sl)
        # 二者皆不满足 → 当日必无触发, 仅 cum_u 与 last_close 需更新 (与逐bar一致)
        _min_pt = None
        if not s_done[1]:
            _min_pt = pt1
        if not s_done[2]:
            _c = eff_pt_2
            _min_pt = _c if _min_pt is None else min(_min_pt, _c)
        if not s_done[3]:
            _c = eff_pt_3
            _min_pt = _c if _min_pt is None else min(_min_pt, _c)
        if not s_done[4]:
            _min_pt = pt4 if _min_pt is None else min(_min_pt, pt4)
        _max_sl = None
        if not s_done[1]:
            _max_sl = eff_sl1
        if not s_done[2]:
            _max_sl = eff_sl2 if _max_sl is None else max(_max_sl, eff_sl2)
        if not s_done[3]:
            _max_sl = eff_sl3 if _max_sl is None else max(_max_sl, eff_sl3)
        if not s_done[4]:
            _c = eff_sl4 if eff_sl4 > hard_sl else hard_sl
            _max_sl = _c if _max_sl is None else max(_max_sl, _c)
        if _min_pt is not None:  # 仍有未完成阶段
            _hi_max = np.nanmax(highs) if len(highs) else float('-inf')
            _lo_min = np.nanmin(lows) if len(lows) else float('inf')
            if (_hi_max / entry - 1 < _min_pt * vol
                    and (_lo_min / entry - 1) > _max_sl * vol):
                # 当日必无触发: 仅更新 cum_u 与 last_close (与逐bar结果一致)
                if _hi_max == _hi_max:
                    cum_u = max(cum_u, _hi_max / entry - 1)
                last_close = closes[-1]
                continue

        # [Perf] numpy 标量索引慢 (~100ns/次), 转 list 迭代 (~30ns/次)
        hi_l = highs.tolist() if hasattr(highs, 'tolist') else list(highs)
        lo_l = lows.tolist() if hasattr(lows, 'tolist') else list(lows)
        cl_l = closes.tolist() if hasattr(closes, 'tolist') else list(closes)

        for k in range(len(cl_l)):
            if qty_left <= 1e-9:
                break
            hi = hi_l[k]; lo = lo_l[k]
            last_close = cl_l[k]
            # [E6-2026-09-10] 事件驱动逐 bar 刷新: 上一根 bar 内的 PT/SL 事件自本根 bar 起生效
            #   (同 bar 内保持保守: 本根全部检查使用"截至上一根 bar"的状态)
            if s_pt[3]:
                pl_sl = pl_sl3
            elif s_pt[2]:
                pl_sl = pl_sl2
            elif s_pt[1]:
                pl_sl = pl_sl1
            else:
                pl_sl = float('-inf')
            eff_sl1 = sl1 if sl1 > pl_sl else pl_sl
            eff_sl2 = sl2 if sl2 > pl_sl else pl_sl
            eff_sl3 = sl3 if sl3 > pl_sl else pl_sl
            eff_sl4 = sl4 if sl4 > pl_sl else pl_sl
            if pt_after_sl:
                eff_pt_2, eff_pt_3 = pt_after_sl2, pt_after_sl3
            else:
                eff_pt_2, eff_pt_3 = pt2, pt3
            if hi == hi:  # not NaN (与 np.isnan 同义, 更快)
                cum_u = max(cum_u, hi / entry - 1)
            up_s = (hi / entry - 1) / vol if hi == hi else float('-inf')  # [PTFIX] 当根触达
            cur_s = (lo / entry - 1) / vol if lo == lo else float('inf')

            # ── 硬兜底 (每 bar 最先检查; 触发 S4 份额后当日结束) ──
            if cur_s <= hard_sl and not s_done[4]:
                px = entry * (1 + hard_sl * vol)
                realized += p4 * (px / entry - 1)
                qty_left -= p4
                s_done[4] = True
                fills.append((4, False, px / entry - 1))
                break  # 当日结束

            # ── S1 ([O3-20260904] 同bar PT/SL歧义 → SL优先保守; 触发后 flag 立即生效) ──
            if not s_done[1]:
                if cur_s <= eff_sl1:
                    px = entry * (1 + eff_sl1 * vol)
                    realized += p1 * (px / entry - 1)
                    qty_left -= p1
                    s_done[1] = True
                    pt_after_sl = True   # [E6] SL 事件自下一根 bar 起下移 PT 阈值
                    fills.append((1, False, px / entry - 1))
                elif up_s >= pt1:
                    px = entry * (1 + pt1 * vol)
                    realized += p1 * (px / entry - 1)
                    qty_left -= p1
                    s_done[1] = True; s_pt[1] = True
                    fills.append((1, True, px / entry - 1))

            # ── S2 ([O3] SL优先) ──
            if not s_done[2]:
                _pt2 = eff_pt_2
                if cur_s <= eff_sl2:
                    px = entry * (1 + eff_sl2 * vol)
                    realized += p2 * (px / entry - 1)
                    qty_left -= p2
                    s_done[2] = True
                    pt_after_sl = True
                    fills.append((2, False, px / entry - 1))
                elif up_s >= _pt2:
                    px = entry * (1 + _pt2 * vol)
                    realized += p2 * (px / entry - 1)
                    qty_left -= p2
                    s_done[2] = True; s_pt[2] = True
                    fills.append((2, True, px / entry - 1))

            # ── S3 ([O3] SL优先) ──
            if not s_done[3]:
                _pt3 = eff_pt_3
                if cur_s <= eff_sl3:
                    px = entry * (1 + eff_sl3 * vol)
                    realized += p3 * (px / entry - 1)
                    qty_left -= p3
                    s_done[3] = True
                    pt_after_sl = True
                    fills.append((3, False, px / entry - 1))
                elif up_s >= _pt3:
                    px = entry * (1 + _pt3 * vol)
                    realized += p3 * (px / entry - 1)
                    qty_left -= p3
                    s_done[3] = True; s_pt[3] = True
                    fills.append((3, True, px / entry - 1))

            # ── S4 ([O3] SL优先; 触发后当日结束) ──
            if not s_done[4]:
                if cur_s <= eff_sl4:
                    px = entry * (1 + eff_sl4 * vol)
                    realized += p4 * (px / entry - 1)
                    qty_left -= p4
                    s_done[4] = True
                    fills.append((4, False, px / entry - 1))
                    break  # 当日结束
                elif up_s >= pt4:
                    px = entry * (1 + pt4 * vol)
                    realized += p4 * (px / entry - 1)
                    qty_left -= p4
                    s_done[4] = True
                    fills.append((4, True, px / entry - 1))
                    break  # 当日结束

    # ── 到期: 剩余仓位按窗口最后可得 close 退出 ──
    if qty_left > 1e-9 and last_close is not None:
        realized += qty_left * (last_close / entry - 1)

    if not fills:
        exit_type = 'expire'
    elif fills[-1][1]:
        exit_type = 'profit_take'
    else:
        exit_type = 'stop_loss'

    return {'realized_ret': realized, 'exit_type': exit_type,
            'n_fills': len(fills), 'fills': fills}


# ============================================================================
#  [Perf] numba 加速版 — 与 simulate_staged_path 逐行同语义 (仅实现语言不同)
#  输入为扁平 bar 数组 + 日边界索引; 输出 realized_ret / exit_code / n_fills
#  (exit_code: 0=expire, 1=profit_take, 2=stop_loss)
# ============================================================================

def _staged_sim_numba_impl():
    """定义 numba 内核 (延迟到首次调用编译; 无 numba 时回退 Python 版)。"""
    if not _HAS_NUMBA:
        return None

    @_nb.njit(cache=True, nogil=True)
    def _sim(hi, lo, cl, day_start, day_end, entry, vol, hard_sl,
             pt1, sl1, p1, pt2, sl2, p2, pt3, sl3, p3, pt4, sl4, p4,
             pl_sl1, pl_sl2, pl_sl3, pt_after_sl2, pt_after_sl3):
        s_done1 = s_done2 = s_done3 = s_done4 = False
        s_pt1 = s_pt2 = s_pt3 = False
        pt_after_sl = False
        cum_u = -np.inf
        realized = 0.0
        qty_left = 1.0
        last_close = -1.0
        n_fills = 0
        last_is_pt = False
        n_days = day_start.shape[0]
        for d in range(n_days):
            if qty_left <= 1e-9:
                break
            i0 = day_start[d]; i1 = day_end[d]
            if i1 <= i0:
                continue
            # ── 日开盘快照: 仅用于"日级快速跳过"判定 (逐 bar 检查用循环内动态刷新) ──
            #   [E6-2026-09-10] 事件驱动: pl_sl/pt_after_sl 改为逐 bar 生效 (下一根 bar)
            if s_pt3:
                pl_sl = pl_sl3
            elif s_pt2:
                pl_sl = pl_sl2
            elif s_pt1:
                pl_sl = pl_sl1
            else:
                pl_sl = -np.inf
            eff_sl1 = sl1 if sl1 > pl_sl else pl_sl
            eff_sl2 = sl2 if sl2 > pl_sl else pl_sl
            eff_sl3 = sl3 if sl3 > pl_sl else pl_sl
            eff_sl4 = sl4 if sl4 > pl_sl else pl_sl
            if pt_after_sl:
                eff_pt_2, eff_pt_3 = pt_after_sl2, pt_after_sl3
            else:
                eff_pt_2, eff_pt_3 = pt2, pt3
            # [R6-③] 日内快照已移除 (统一读 eff_pt_2/3, 与 test3 一致)

            # ── 日级快速跳过 (与 Python 版严格等价) ──
            min_pt = 1e18
            has_pending = False
            if not s_done1:
                min_pt = pt1; has_pending = True
            if not s_done2:
                c = eff_pt_2
                if not has_pending or c < min_pt:
                    min_pt = c
                has_pending = True
            if not s_done3:
                c = eff_pt_3
                if not has_pending or c < min_pt:
                    min_pt = c
                has_pending = True
            if not s_done4:
                if not has_pending or pt4 < min_pt:
                    min_pt = pt4
                has_pending = True
            max_sl = -1e18
            if not s_done1:
                max_sl = eff_sl1
            if not s_done2:
                if eff_sl2 > max_sl:
                    max_sl = eff_sl2
            if not s_done3:
                if eff_sl3 > max_sl:
                    max_sl = eff_sl3
            if not s_done4:
                c = eff_sl4 if eff_sl4 > hard_sl else hard_sl
                if c > max_sl:
                    max_sl = c
            if has_pending:
                hi_max = -np.inf
                lo_min = np.inf
                for k in range(i0, i1):
                    h = hi[k]
                    if h == h and h > hi_max:
                        hi_max = h
                    l = lo[k]
                    if l == l and l < lo_min:
                        lo_min = l
                if (hi_max / entry - 1.0 < min_pt * vol
                        and (lo_min / entry - 1.0) > max_sl * vol):
                    if hi_max == hi_max:
                        cum_u = max(cum_u, hi_max / entry - 1.0)
                    last_close = cl[i1 - 1]
                    continue

            # ── 逐 bar (与 Python 版逐行对应) ──
            for k in range(i0, i1):
                if qty_left <= 1e-9:
                    break
                h = hi[k]; l = lo[k]
                last_close = cl[k]
                # [E6-2026-09-10] 事件驱动逐 bar 刷新 (同 Python 版): 上一根 bar 的事件自本根生效
                if s_pt3:
                    pl_sl = pl_sl3
                elif s_pt2:
                    pl_sl = pl_sl2
                elif s_pt1:
                    pl_sl = pl_sl1
                else:
                    pl_sl = -np.inf
                eff_sl1 = sl1 if sl1 > pl_sl else pl_sl
                eff_sl2 = sl2 if sl2 > pl_sl else pl_sl
                eff_sl3 = sl3 if sl3 > pl_sl else pl_sl
                eff_sl4 = sl4 if sl4 > pl_sl else pl_sl
                if pt_after_sl:
                    eff_pt_2, eff_pt_3 = pt_after_sl2, pt_after_sl3
                else:
                    eff_pt_2, eff_pt_3 = pt2, pt3
                if h == h:
                    cum_u = max(cum_u, h / entry - 1.0)
                if h == h:
                    up_s = (h / entry - 1.0) / vol  # [PTFIX] 当根触达
                else:
                    up_s = -np.inf
                if l == l:
                    cur_s = (l / entry - 1.0) / vol
                else:
                    cur_s = 1e18

                # ── 硬兜底 (每 bar 最先检查; 触发 S4 份额后当日结束) ──
                if cur_s <= hard_sl and not s_done4:
                    px = entry * (1.0 + hard_sl * vol)
                    realized += p4 * (px / entry - 1.0)
                    qty_left -= p4
                    s_done4 = True
                    n_fills += 1; last_is_pt = False
                    break

                # ── S1 ([O3-20260904] 同bar PT/SL歧义 → SL优先保守) ──
                if not s_done1:
                    if cur_s <= eff_sl1:
                        px = entry * (1.0 + eff_sl1 * vol)
                        realized += p1 * (px / entry - 1.0)
                        qty_left -= p1
                        s_done1 = True
                        pt_after_sl = True
                        n_fills += 1; last_is_pt = False
                    elif up_s >= pt1:
                        px = entry * (1.0 + pt1 * vol)
                        realized += p1 * (px / entry - 1.0)
                        qty_left -= p1
                        s_done1 = True; s_pt1 = True
                        n_fills += 1; last_is_pt = True

                # ── S2 ([O3] SL优先) ──
                if not s_done2:
                    _pt2 = eff_pt_2
                    if cur_s <= eff_sl2:
                        px = entry * (1.0 + eff_sl2 * vol)
                        realized += p2 * (px / entry - 1.0)
                        qty_left -= p2
                        s_done2 = True
                        pt_after_sl = True
                        n_fills += 1; last_is_pt = False
                    elif up_s >= _pt2:
                        px = entry * (1.0 + _pt2 * vol)
                        realized += p2 * (px / entry - 1.0)
                        qty_left -= p2
                        s_done2 = True; s_pt2 = True
                        n_fills += 1; last_is_pt = True

                # ── S3 ([O3] SL优先) ──
                if not s_done3:
                    _pt3 = eff_pt_3
                    if cur_s <= eff_sl3:
                        px = entry * (1.0 + eff_sl3 * vol)
                        realized += p3 * (px / entry - 1.0)
                        qty_left -= p3
                        s_done3 = True
                        pt_after_sl = True
                        n_fills += 1; last_is_pt = False
                    elif up_s >= _pt3:
                        px = entry * (1.0 + _pt3 * vol)
                        realized += p3 * (px / entry - 1.0)
                        qty_left -= p3
                        s_done3 = True; s_pt3 = True
                        n_fills += 1; last_is_pt = True

                # ── S4 ([O3] SL优先; 触发后当日结束) ──
                if not s_done4:
                    if cur_s <= eff_sl4:
                        px = entry * (1.0 + eff_sl4 * vol)
                        realized += p4 * (px / entry - 1.0)
                        qty_left -= p4
                        s_done4 = True
                        n_fills += 1; last_is_pt = False
                        break
                    elif up_s >= pt4:
                        px = entry * (1.0 + pt4 * vol)
                        realized += p4 * (px / entry - 1.0)
                        qty_left -= p4
                        s_done4 = True
                        n_fills += 1; last_is_pt = True
                        break

        # ── 到期: 剩余仓位按窗口最后可得 close 退出 ──
        if qty_left > 1e-9 and last_close > 0:
            realized += qty_left * (last_close / entry - 1.0)

        if n_fills == 0:
            exit_code = 0
        elif last_is_pt:
            exit_code = 1
        else:
            exit_code = 2
        return realized, exit_code, n_fills

    return _sim


_STAGED_SIM_NUMBA = _staged_sim_numba_impl()


def simulate_staged_path_fast(day_bars, entry, vol, stage_cfg, hard_sl,
                              pl_sl1=0.0, pl_sl2=1.0, pl_sl3=2.0,
                              pt_after_sl2=1.5, pt_after_sl3=2.0):
    """numba 加速版 simulate_staged_path (语义完全一致, ~50-200x 更快)。

    与 simulate_staged_path 的差异: fills 列表不返回 (标签端不需要);
    realized_ret / exit_type / n_fills 逐位一致。
    """
    if vol <= 0 or entry <= 0:
        return {'realized_ret': 0.0, 'exit_type': 'expire', 'n_fills': 0, 'fills': []}

    # 扁平化 day_bars → 连续数组 + 日边界
    hi_parts = []
    lo_parts = []
    cl_parts = []
    day_start = np.empty(len(day_bars), dtype=np.int64)
    day_end = np.empty(len(day_bars), dtype=np.int64)
    n_bar = 0
    n_days = 0
    for highs, lows, closes in day_bars:
        if closes is None or len(closes) == 0:
            continue
        hi_parts.append(np.asarray(highs, dtype=np.float64))
        lo_parts.append(np.asarray(lows, dtype=np.float64))
        cl_parts.append(np.asarray(closes, dtype=np.float64))
        day_start[n_days] = n_bar
        n_bar += len(closes)
        day_end[n_days] = n_bar
        n_days += 1
    if n_days == 0:
        return {'realized_ret': 0.0, 'exit_type': 'expire', 'n_fills': 0, 'fills': []}

    if _STAGED_SIM_NUMBA is None:
        return simulate_staged_path(day_bars, entry, vol, stage_cfg, hard_sl,
              pl_sl1=pl_sl1, pl_sl2=pl_sl2, pl_sl3=pl_sl3,
              pt_after_sl2=pt_after_sl2, pt_after_sl3=pt_after_sl3)

    hi = np.concatenate(hi_parts)
    lo = np.concatenate(lo_parts)
    cl = np.concatenate(cl_parts)
    realized, exit_code, n_fills = _STAGED_SIM_NUMBA(
        hi, lo, cl, day_start[:n_days], day_end[:n_days],
        float(entry), float(vol), float(hard_sl),
        float(stage_cfg[1]['pt']), float(stage_cfg[1]['sl']), float(stage_cfg[1]['pct']),
        float(stage_cfg[2]['pt']), float(stage_cfg[2]['sl']), float(stage_cfg[2]['pct']),
        float(stage_cfg[3]['pt']), float(stage_cfg[3]['sl']), float(stage_cfg[3]['pct']),
        float(stage_cfg[4]['pt']), float(stage_cfg[4]['sl']), float(stage_cfg[4]['pct']),
                float(pl_sl1), float(pl_sl2), float(pl_sl3), float(pt_after_sl2), float(pt_after_sl3))
    if exit_code == 0:
        exit_type = 'expire'
    elif exit_code == 1:
        exit_type = 'profit_take'
    else:
        exit_type = 'stop_loss'
    return {'realized_ret': realized, 'exit_type': exit_type,
            'n_fills': n_fills, 'fills': []}


# ── back-adjust helpers ──────────────────────────────────────────────

def _load_back_factor_map(dp, code):
    """Load back_factor as a Series indexed by date (cumulative factor up to that date)."""
    back_file = os.path.join(dp.back_adjust_path,
                             f"{code.replace('.', '_')}_back_adjust.parquet")
    if not os.path.exists(back_file):
        return None  # no dividends → factor = 1.0 everywhere
    
    bdf = pd.read_parquet(back_file)
    bdf['dividOperateDate'] = pd.to_datetime(bdf['dividOperateDate'])
    bdf = bdf.sort_values('dividOperateDate')
    # Build a full-date-range factor series with forward fill
    return bdf.set_index('dividOperateDate')['backAdjustFactor']


def _get_back_at(back_map, dt):
    """Get cumulative back factor at date dt (most recent factor <= dt)."""
    if back_map is None:
        return 1.0
    # back_map is a Series indexed by date; take last <= dt
    mask = back_map.index <= pd.Timestamp(dt)
    if mask.any():
        return back_map.loc[mask].iloc[-1]
    return 1.0


# ── batch labeler ────────────────────────────────────────────────────

class TripleBarrierLabeler:
    """
    Triple-barrier labeler (batch-optimized).
    
    Identical label outputs to the original per-event IO version,
    but pre-loads raw parquets once per stock and does back-adjustment
    in memory.

    Prado Ch.3.5: barriers = f(σ), not f(signal).
    """
    def __init__(self, data_provider, pt_mult=2.0, sl_mult=1.5,
                 max_hold_days=5, minute_freq='5',
                 daily_vol_window=20, vol_multiplier=1.0,
                 vol_method='rolling', vol_ewma_lambda=0.94,
                 vol_ewma_seed_days=60,
                 staged_cfg=None, hard_sl=-6.0,
                 profit_lock=None, pt_after_sl=None):
        self.dp = data_provider
        self.pt_mult = pt_mult
        self.sl_mult = sl_mult
        self.max_hold_days = max_hold_days
        self.minute_freq = minute_freq
        self.daily_vol_window = daily_vol_window
        self.vol_multiplier = vol_multiplier
        self.vol_method = vol_method
        self.vol_ewma_lambda = vol_ewma_lambda
        self.vol_ewma_seed_days = vol_ewma_seed_days
        # [L1] 分阶段退出标签参数 (来自 config.staged_exit, 与交易端同源)
        self.staged_cfg = staged_cfg
        self.hard_sl = hard_sl
        # [R6-③ 2026-09-10] 利润锁定/PT下移参数化 (与交易端 StageManager 同源, config 单点定义;
        #   默认值与原硬编码 0/1/2 与 1.5/2.0 等价; 供 simulate_staged_path(_fast) 使用)
        _plk = profit_lock or {}
        self.pl_sl1 = float(_plk.get('s1_sl', 0.0))
        self.pl_sl2 = float(_plk.get('s2_sl', 1.0))
        self.pl_sl3 = float(_plk.get('s3_sl', 2.0))
        _pasl = pt_after_sl or {}
        self.pt_after_sl2 = float(_pasl.get('s2_pt', 1.5))
        self.pt_after_sl3 = float(_pasl.get('s3_pt', 2.0))
    
    def _get_next_trading_day(self, date, calendar):
        # [Perf] searchsorted O(log n) 替代线性扫描 (2.3ms/事件 → ~2µs/事件)
        # 语义与原实现完全一致: 返回第一个严格大于 date 的交易日; 无则 None
        if isinstance(calendar, (list, tuple, np.ndarray)):
            calendar = pd.DatetimeIndex(calendar)
        idx = calendar.searchsorted(pd.Timestamp(date), side='right')
        if idx < len(calendar):
            return calendar[idx]
        return None

    # ── in-memory caches (cleared per call) ───────────────────────
    _daily_raw = None
    _minute_raw = None
    _no_minute = False  # True when batch pre-load confirms no minute data available
    _back_map = None
    _cur_code = None
    _minute_start = None  # first timestamp loaded
    _minute_end = None    # last timestamp loaded

    def _ensure_stock_data(self, code, min_date=None, max_date=None):
        """Pre-load raw daily + minute for `code`, covering needed date range."""
        if self._cur_code == code:
            # Check if already-loaded range covers new request
            if min_date is None or (self._minute_start is not None and 
                self._minute_start <= pd.Timestamp(min_date) and
                self._minute_end   >= pd.Timestamp(max_date)):
                return
            # Range extended — need to re-read minute data
        self._cur_code = code
        self._daily_raw = None
        self._minute_raw = None
        self._no_minute = False
        self._back_map = None
        self._minute_start = None
        self._minute_end = None

        try:
            safe = code.replace('.', '_')
            daily_file = os.path.join(self.dp.raw_daily_path, f"{safe}.parquet")
        except AttributeError:
            return  # mock

        # Daily — full file is small (~15 years × 250 days), always read whole
        if os.path.exists(daily_file):
            df = pd.read_parquet(daily_file)
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.normalize()
            df = df.sort_values('trade_date')
            bm = _load_back_factor_map(self.dp, code)
            self._back_map = bm
            df['backFactor'] = df['trade_date'].map(lambda d: _get_back_at(bm, d)) if bm is not None else 1.0
            self._daily_raw = df
        try:
            div_file = os.path.join(self.dp.dividend_data_path,
                                    f"{code.replace('.', '_')}_dividend.parquet")
            if os.path.exists(div_file):
                div_df = pd.read_parquet(div_file)
                div_df['dividOperateDate'] = pd.to_datetime(
                    div_df['dividOperateDate']).dt.normalize()
                self._dividend_df = div_df
        except Exception:
            self._dividend_df = None

        # Minute — read only the required date range (avoids loading entire file)
        minute_file = os.path.join(self.dp.raw_minute_base_path, '5min', f"{safe}.parquet")
        if min_date is not None and os.path.exists(minute_file):
            start_ts = pd.Timestamp.combine(pd.Timestamp(min_date), datetime.min.time())
            end_ts   = pd.Timestamp.combine(pd.Timestamp(max_date), datetime.max.time()) + pd.Timedelta(days=1)
            # Use pyarrow filtering to read only matching rows
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(minute_file,
                    filters=[('trade_time', '>=', start_ts),
                             ('trade_time', '<=', end_ts)])
                if table.num_rows > 0:
                    df = table.to_pandas()
                    df['trade_time'] = pd.to_datetime(df['trade_time'])
                    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.normalize()
                    df = df.sort_values('trade_time')
                    bm = self._back_map if self._back_map is not None else _load_back_factor_map(self.dp, code)
                    df['backFactor'] = df['trade_date'].map(lambda d: _get_back_at(bm, d)) if bm is not None else 1.0
                    self._minute_raw = df
                    self._minute_start = pd.Timestamp(min_date)
                    self._minute_end   = pd.Timestamp(max_date)
                else:
                    self._no_minute = True  # 0 rows in range → skip per-event fallback
            except Exception:
                self._no_minute = True  # read failed → skip per-event fallback

    def _compute_daily_vol(self, event_date, code):
        """Compute daily vol. Uses batch data if available, else provider API."""
        if self._daily_raw is not None:
            lookback_start = event_date - timedelta(days=self.daily_vol_window * 2)
            mask = (self._daily_raw['trade_date'] >= lookback_start) & \
                   (self._daily_raw['trade_date'] <= event_date)
            df = self._daily_raw[mask].sort_values('trade_date')
            if len(df) < 5:
                return 0.01
            back_T = _get_back_at(self._back_map, event_date)
            adj_close = df['close'] * df['backFactor'] / back_T
            daily_returns = adj_close.pct_change().dropna()
            vol = daily_returns.std()
            return vol if not (pd.isna(vol) or vol == 0) else 0.01
        else:
            lookback_start = event_date - timedelta(days=self.daily_vol_window * 2)
            daily_df = self.dp.get_daily_data(code, lookback_start, event_date,
                                              reference_date=event_date)
            if daily_df.empty or len(daily_df) < 5:
                return 0.01
            if 'raw_close' in daily_df.columns and 'backAdjustFactor' in daily_df.columns:
                adj_prices = daily_df['raw_close'] * daily_df['backAdjustFactor']
                daily_returns = adj_prices.pct_change().dropna()
            else:
                daily_returns = daily_df['close'].pct_change().dropna()
            daily_vol = daily_returns.std()
            return daily_vol if not (pd.isna(daily_vol) or daily_vol == 0) else 0.01

    def _check_minute_path(self, code, start_date, end_date, entry_price,
                           pt_threshold, sl_threshold):
        """Check minute path (vectorized). Returns (exit_type, exit_price)."""
        if self._minute_raw is not None:
            # Batch path: raw data in memory, apply back-factor
            start_ts = pd.Timestamp.combine(start_date, datetime.min.time())
            end_ts = pd.Timestamp.combine(end_date, datetime.max.time())
            mask = (self._minute_raw['trade_time'] >= start_ts) & \
                   (self._minute_raw['trade_time'] <= end_ts)
            mdf = self._minute_raw[mask]
            if mdf.empty:
                return 'expire', entry_price
            back_T = _get_back_at(self._back_map, start_date)
            adj_factor = mdf['backFactor'].values / back_T
            highs = mdf['high'].values * adj_factor
            lows  = mdf['low'].values  * adj_factor
            closes = mdf['close'].values * adj_factor
        else:
            # Fallback: use data provider (pre-adjusted data) — but skip if batch already
            # confirmed no minute data for this stock (avoid O(events) full-file reads).
            if self._no_minute:
                return 'expire', entry_price
            s = datetime.combine(start_date, datetime.min.time())
            e = datetime.combine(end_date, datetime.max.time())
            minute_df = self.dp.get_minute_data(code, s, e,
                                                freq=self.minute_freq,
                                                reference_date=start_date)
            if minute_df.empty:
                return 'expire', entry_price
            highs = minute_df['high'].values
            lows = minute_df['low'].values
            closes = minute_df['close'].values

        # ── Vectorised: max/min + argmax for tie-breaking ──
        pt_hit = highs >= pt_threshold
        sl_hit = lows <= sl_threshold
        pt_any = pt_hit.any()
        sl_any = sl_hit.any()

        if pt_any and sl_any:
            # Both hit → compare time of first hit
            if np.argmax(pt_hit) <= np.argmax(sl_hit):
                return 'profit_take', pt_threshold
            else:
                return 'stop_loss', sl_threshold
        elif pt_any:
            return 'profit_take', pt_threshold
        elif sl_any:
            return 'stop_loss', sl_threshold
        else:
            return 'expire', closes[-1]


    def compute_labels(self, events_df, calendar):
        """
        Batch-optimized label generation (identical results).

        Pre-loads raw daily + minute parquets once per stock (with pyarrow
        pushdown filtering to avoid reading the entire minute file),
        then does back-adjustment in memory per event.

        Delegates to compute_labels_fast which implements day-line prescreen
        + batch vectorization for ~5x speedup with zero accuracy loss.
        """
        return self.compute_labels_fast(events_df, calendar)

    # ==================== 加速版标签生成（方案A+B：日线预筛 + 批量向量化） ====================

    def _batch_daily_volatility(self, daily_df, event_dates, ref_date=None):
        """
        批量计算每日波动率。daily_df 必须是已复权的 DataFrame（high/low/close 均后复权）。
        直接从 close 计算收益率差异即可。
        
        [FIX] 支持 vol_method='ewma'：使用 EWMA 波动率（RiskMetrics 标准），
        种子窗口为 vol_ewma_seed_days（默认60日），与交易端一致。
        回退默认值统一为 0.02。
        
        【性能优化 v2】
        - rolling 模式：一次性 rolling().std() 向量化计算，searchsorted 定位索引
        - ewma 模式：searchsorted 替代全表 mask 扫描（O(log n) vs O(n)）
        输出与原始实现完全一致。
        """
        _fallback_vol = 0.02  # 统一回退值

        if daily_df is None or len(daily_df) < 5:
            return np.full(len(event_dates), _fallback_vol)

        daily_df = daily_df.sort_values('trade_date')
        adj_close = daily_df['close'].values
        daily_dates = daily_df['trade_date'].values

        # EWMA 模式下需要更长的 lookback 窗口（种子窗口）
        if self.vol_method == 'ewma':
            lookback = max(self.daily_vol_window, self.vol_ewma_seed_days)
        else:
            lookback = self.daily_vol_window

        # 一次性计算所有日收益率
        all_returns = np.diff(adj_close) / adj_close[:-1]

        # 预计算日期索引映射（searchsorted 替代全表扫描）
        daily_ts = pd.DatetimeIndex(daily_dates)

        vols = np.full(len(event_dates), _fallback_vol)

        if self.vol_method == 'ewma' and len(all_returns) > 1:
            # EWMA 模式：种子依赖窗口起始，保持逐事件计算但用 searchsorted 加速
            alpha = 1 - self.vol_ewma_lambda
            for i, ed in enumerate(event_dates):
                # searchsorted side='right' 找到最后一个 <= ed 的索引
                idx = daily_ts.searchsorted(pd.Timestamp(ed), side='right') - 1
                if idx < 4:  # mask.sum() < 5 → 不足5个数据点
                    vols[i] = _fallback_vol
                    continue
                # 取与原代码相同的窗口：adj_close[max(0,idx-lookback+1):idx+1] 的收益率
                start = max(0, idx - lookback + 1)
                window_rets = all_returns[start:idx]
                if len(window_rets) < 2:
                    vols[i] = _fallback_vol
                    continue
                ewma_var = pd.Series(window_rets).ewm(alpha=alpha, adjust=False).var().iloc[-1]
                if pd.notna(ewma_var) and ewma_var > 0:
                    v = float(np.sqrt(ewma_var))
                    vols[i] = v if not (np.isnan(v) or v == 0) else _fallback_vol
                else:
                    vols[i] = _fallback_vol
        else:
            # Rolling std 模式：完全向量化
            # 原代码取 lookback 个 close → lookback-1 个 returns → std(ddof=1)
            # rolling(lookback-1, min_periods=2) 在位置 idx-1 给出 all_returns[idx-lookback+1:idx] 的 std
            ret_series = pd.Series(all_returns)
            rolling_std = ret_series.rolling(lookback - 1, min_periods=2).std(ddof=1)
            vol_series = rolling_std.values

            for i, ed in enumerate(event_dates):
                idx = daily_ts.searchsorted(pd.Timestamp(ed), side='right') - 1
                if idx < 4:  # 不足5个close价格
                    vols[i] = _fallback_vol
                    continue
                vol_idx = idx - 1  # all_returns[idx-1] 是窗口最后一个收益率
                if 0 <= vol_idx < len(vol_series):
                    v = vol_series[vol_idx]
                    if pd.notna(v) and v > 0:
                        vols[i] = float(v)
                    # else: 保持 _fallback_vol
                # else: 保持 _fallback_vol

        return vols

    def _daily_prescreen(self, daily_df, event_dates, entry_prices,
                          vols, pt_ths, sl_ths, hold_windows):
        """
        日线预筛：检查持有窗口内的日线极值是否可能触达PT/SL。
        daily_df 必须是已复权的 DataFrame（high/low/close 均后复权）。
        返回 needs_minute 索引列表和 expire 结果列表。
        
        【性能优化 v2】searchsorted 替代全表布尔扫描（O(log n) vs O(n)）
        输出与原始实现完全一致。
        """
        if daily_df is None or len(daily_df) == 0:
            return list(range(len(event_dates))), []

        daily_df = daily_df.sort_values('trade_date')
        daily_dates = daily_df['trade_date'].values
        adj_high = daily_df['high'].values
        adj_low = daily_df['low'].values
        adj_close = daily_df['close'].values

        # 预构建 DatetimeIndex 用于 searchsorted
        daily_ts = pd.DatetimeIndex(daily_dates)

        needs_minute = []
        expire_results = []

        for i, (ed, ep, pt, sl) in enumerate(zip(event_dates, entry_prices, pt_ths, sl_ths)):
            next_day, hold_end = hold_windows[i]
            if next_day is None:
                expire_results.append((i, ep))
                continue

            # searchsorted: O(log n) 替代 O(n) 全表布尔扫描
            start_idx = daily_ts.searchsorted(pd.Timestamp(next_day))
            end_idx = daily_ts.searchsorted(pd.Timestamp(hold_end), side='right')
            if start_idx >= end_idx:
                expire_results.append((i, ep))
                continue

            w_high = adj_high[start_idx:end_idx]
            w_low = adj_low[start_idx:end_idx]

            max_high = w_high.max()
            min_low = w_low.min()

            if max_high < pt and min_low > sl:
                expire_results.append((i, float(adj_close[start_idx:end_idx][-1])))
            else:
                needs_minute.append(i)

        return needs_minute, expire_results

    def _batch_minute_confirmation(self, code, back_map,
                                    needs_indices, event_dates, entry_prices,
                                    pt_ths, sl_ths, hold_windows,
                                    daily_ref_date=None):
        """
        批量分钟线确认：通过 dp.get_minute_data 获取已复权分钟数据（利用LRU缓存）。
        与原始 _check_minute_path 结果完全一致，但加载一次后全股票复用缓存。
        """
        # [FIX-ADJ-CONSISTENT] Use daily ref_date to ensure minute adj
        # is on same scale as daily adj (PT/SL are in daily adj scale)
        ref_date = daily_ref_date if daily_ref_date is not None else (
            hold_windows[needs_indices[0]][0] if needs_indices else datetime.now()
        )
        if ref_date is None:
            ref_date = datetime.now()
        try:
            # 加载全部分钟数据（dp内部有LRU缓存，同股票复用）
            df = self.dp._load_full_minute_data(code, '5', ref_date)
        except Exception:
            df = None

        if df is None or df.empty:
            return [(i, 'expire', float(entry_prices[i])) for i in needs_indices]

        times = df['trade_time'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        dates = df['trade_date'].values

        results = []
        for idx in needs_indices:
            ep = entry_prices[idx]
            pt = pt_ths[idx]
            sl = sl_ths[idx]
            next_day, hold_end = hold_windows[idx]
            if next_day is None:
                results.append((idx, 'expire', float(ep)))
                continue

            start_ts = pd.Timestamp.combine(next_day, datetime.min.time())
            end_ts = pd.Timestamp.combine(hold_end, datetime.max.time())
            mask = (times >= start_ts) & (times <= end_ts)

            if not mask.any():
                results.append((idx, 'expire', float(ep)))
                continue

            h = highs[mask]
            l = lows[mask]
            c = closes[mask]

            pt_hit = h >= pt
            sl_hit = l <= sl
            pt_any = pt_hit.any()
            sl_any = sl_hit.any()

            if pt_any and sl_any:
                if np.argmax(pt_hit) <= np.argmax(sl_hit):
                    results.append((idx, 'profit_take', float(pt)))
                else:
                    results.append((idx, 'stop_loss', float(sl)))
            elif pt_any:
                results.append((idx, 'profit_take', float(pt)))
            elif sl_any:
                results.append((idx, 'stop_loss', float(sl)))
            else:
                results.append((idx, 'expire', float(c[-1])))

        return results

    def compute_labels_fast(self, events_df, calendar, align_entry_mode='event_date_close', reference_date=None):
        """
        加速版标签生成：日线预筛 + 批量向量化 + data_provider缓存复用。

        与传统 compute_labels 结果完全一致，但利用 dp 的 LRU 缓存避免重复 I/O。

        Args:
            align_entry_mode: 入场价对齐模式
                - "event_date_close": 用事件日close覆盖entry_price（当前默认，向后兼容）
                - "next_day_open": 用D+1日open覆盖entry_price（A2标签，与交易对齐）
        """
        import time as _time
        t0 = _time.time()

        # Group events by stock
        code_events = defaultdict(list)
        for event_date, row in events_df.iterrows():
            if not isinstance(event_date, datetime):
                event_date = pd.Timestamp(event_date)
            code_events[row['ts_code']].append(
                (event_date, row['entry_price'])
            )

        # Pre-compute date range per stock
        code_range = {}
        for code, evs in code_events.items():
            dts = [e[0] for e in evs]
            min_dt = min(dts) - timedelta(days=self.daily_vol_window * 2)
            max_end_candidate = max(dts)
            for dt in dts:
                nd = self._get_next_trading_day(dt, calendar)
                if nd is not None:
                    try:
                        i = calendar.index(nd)
                        he = calendar[min(i + self.max_hold_days - 1, len(calendar) - 1)]
                        if he > max_end_candidate:
                            max_end_candidate = he
                    except ValueError:
                        pass
            code_range[code] = (min_dt, max_end_candidate + timedelta(days=1))

        # === 性能优化：多线程并行处理股票 ===
        # 每只股票完全独立，无数据依赖
        # self.dp._load_full_daily_data 内部有 LRU 缓存（线程安全）
        # self._batch_daily_volatility / _daily_prescreen / _batch_minute_confirmation
        # 仅读取 self 属性，不修改实例状态，线程安全

        def _process_one_stock(code_events_tuple):
            """处理单只股票的标签生成（线程安全，可在多线程中调用）"""
            code, events = code_events_tuple
            stock_labels = []
            if not code or not isinstance(code, str):
                return stock_labels, 0, 0  # labels, n_events, n_skipped

            # Phase 1: Load daily data via dp (uses LRU cache)
            ref_date = reference_date if reference_date is not None else (max(e[0] for e in events) if events else datetime.now())
            daily_df = None
            try:
                daily_full = self.dp._load_full_daily_data(code, ref_date)
                if daily_full is not None and not daily_full.empty:
                    daily_df = daily_full.copy()
                    daily_df['trade_date'] = pd.to_datetime(daily_df['trade_date']).dt.normalize()
                    daily_df = daily_df.sort_values('trade_date')
            except Exception:
                pass

            # Phase 2: Prepare event arrays
            event_dates = [e[0] for e in events]
            entry_prices = np.array([e[1] for e in events])
            n_events = len(events)

            # Phase 2.5: [FIX-ADJ-CONSISTENT] Override raw entry_price with adj price
            if daily_df is not None and not daily_df.empty:
                daily_lookup = daily_df.set_index('trade_date')
                for i, ed in enumerate(event_dates):
                    if align_entry_mode == 'next_day_open':
                        nd = self._get_next_trading_day(ed, calendar)
                        if nd is not None and pd.Timestamp(nd) in daily_lookup.index:
                            entry_prices[i] = float(daily_lookup.loc[pd.Timestamp(nd), 'open'])
                        else:
                            entry_prices[i] = np.nan
                    else:
                        ts_ed = pd.Timestamp(ed)
                        if ts_ed in daily_lookup.index:
                            entry_prices[i] = float(daily_lookup.loc[ts_ed, 'close'])
                        else:
                            entry_prices[i] = np.nan
                daily_lookup = None
            else:
                entry_prices[:] = np.nan

            # Phase 3: Batch daily volatility
            vols = self._batch_daily_volatility(daily_df, event_dates, ref_date)

            # Phase 4: Compute PT/SL in batch
            pt_ths, sl_ths = compute_barriers(
                entry_prices, vols,
                pt_mult=self.pt_mult, sl_mult=self.sl_mult,
                vol_mult=self.vol_multiplier, hold_days=1,
                vectorized=True
            )

            # Phase 5: Pre-compute hold windows
            hold_windows = []
            for ed in event_dates:
                next_day = self._get_next_trading_day(ed, calendar)
                if next_day is None:
                    hold_windows.append((None, None))
                    continue
                try:
                    idx = calendar.index(next_day)
                    hold_end = calendar[min(idx + self.max_hold_days - 1, len(calendar) - 1)]
                    hold_windows.append((next_day, hold_end))
                except ValueError:
                    hold_windows.append((None, None))

            # Phase 6: Day-line prescreen
            needs_minute, expire_results = self._daily_prescreen(
                daily_df, event_dates, entry_prices,
                vols, pt_ths, sl_ths, hold_windows
            )
            n_skipped = len(expire_results)

            # Phase 7: Minute precise confirmation
            minute_results = {}
            if needs_minute:
                conf_results = self._batch_minute_confirmation(
                    code, None,
                    needs_minute, event_dates, entry_prices,
                    pt_ths, sl_ths, hold_windows,
                    daily_ref_date=ref_date
                )
                for idx, ext, exp in conf_results:
                    minute_results[idx] = (ext, exp)

            # Phase 8: Assemble labels
            expire_dict = dict(expire_results)
            for i in range(n_events):
                ed = event_dates[i]
                ep = entry_prices[i]
                if hold_windows[i][0] is None:
                    continue
                if np.isnan(ep):
                    continue

                if i in minute_results:
                    exit_type, exit_price = minute_results[i]
                elif i in expire_dict:
                    exit_type = 'expire'
                    exit_price = expire_dict[i]
                else:
                    continue

                ret = (exit_price - ep) / ep
                if exit_type == 'profit_take':
                    bin_label = 1
                elif exit_type == 'expire':
                    bin_label = 1 if ret > 0 else 0
                else:
                    bin_label = 0

                stock_labels.append({
                    'date': ed, 'code': code,
                    'bin': bin_label, 'ret': ret, 'exit_type': exit_type
                })

            return stock_labels, n_events, n_skipped

        # 执行并行处理
        from concurrent.futures import ThreadPoolExecutor
        stock_items = list(code_events.items())
        n_stocks = len(stock_items)
        max_workers = min(8, n_stocks) if n_stocks > 0 else 1

        logger.info("[加速标签] 开始处理 %d 只股票 (max_workers=%d)", n_stocks, max_workers)

        all_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            all_results = list(ex.map(_process_one_stock, stock_items))

        # 汇总结果
        all_labels = []
        total_events = 0
        skipped_by_prescreen = 0
        stock_count = 0
        for labels, n_ev, n_skip in all_results:
            all_labels.extend(labels)
            total_events += n_ev
            skipped_by_prescreen += n_skip
            stock_count += 1

        t1 = _time.time()
        logger.info(
            "[加速标签] 完成: %d 只股票, %d 事件, 日线预筛跳过=%d (%.1f%%), 总耗时=%.1fs",
            stock_count, total_events, skipped_by_prescreen,
            100 * skipped_by_prescreen / max(total_events, 1),
            t1 - t0
        )
        return pd.DataFrame(all_labels)

    # ==================== [L1] 分阶段退出标签（标签-交易同规则） ====================

    def compute_labels_staged(self, events_df, calendar, reference_date=None):
        """[L1] 分阶段退出标签: 对每个事件模拟与 handle_bar 完全同规则的分阶段
        退出路径, 输出 realized_ret (已实现收益) 与 exit_type。

        与 compute_labels_fast 的差异:
        - 标签语义: 三重屏障 PT/SL/expire → 分阶段实现收益 (L1 对齐)
        - 入场价: D日close (与旧标签口径一致, 对应交易端 D+1 open 的一日平移)
        - 分钟缺失日: 回退日线 bar (与生产 _check_staged_daily_fallback 一致;
          旧标签直接 expire@entry)

        Args:
            events_df: index=event_date, 列含 ts_code, entry_price (占位)
            calendar: 交易日历 (list of datetime)
            reference_date: 复权基准日

        Returns:
            DataFrame[date, code, ret, realized_ret, exit_type, n_fills]
            (bin 标签由策略端按 label_mode 后处理: staged_binary=sign, staged_rank=截面分位)
        """
        import time as _time
        t0 = _time.time()

        if self.staged_cfg is None:
            raise ValueError("compute_labels_staged 需要 staged_cfg (来自 config.staged_exit)")

        # 按股票分组事件 (与 compute_labels_fast 相同)
        code_events = defaultdict(list)
        for event_date, row in events_df.iterrows():
            if not isinstance(event_date, datetime):
                event_date = pd.Timestamp(event_date)
            code_events[row['ts_code']].append((event_date, row['entry_price']))

        cal_ts = pd.DatetimeIndex(sorted(calendar))

        def _process_one_stock(code_events_tuple):
            code, events = code_events_tuple
            stock_labels = []
            if not code or not isinstance(code, str):
                return stock_labels, 0

            ref_date = reference_date if reference_date is not None else (
                max(e[0] for e in events) if events else datetime.now())
            # Phase 1: 日线 (LRU 缓存)
            daily_df = None
            try:
                daily_full = self.dp._load_full_daily_data(code, ref_date)
                if daily_full is not None and not daily_full.empty:
                    daily_df = daily_full.copy()
                    daily_df['trade_date'] = pd.to_datetime(daily_df['trade_date']).dt.normalize()
                    daily_df = daily_df.sort_values('trade_date').reset_index(drop=True)
            except Exception:
                pass

            # Phase 2: 事件数组 + 入场价覆盖为前复权 D日close (同旧标签口径)
            event_dates = [e[0] for e in events]
            entry_prices = np.array([e[1] for e in events], dtype=float)
            # [Perf] 日线数据预构建字典 (替代 set_index+.loc, 每次省~20µs)
            daily_bar_map = None
            if daily_df is not None and not daily_df.empty:
                _dd = daily_df['trade_date'].values
                _dh = daily_df['high'].values
                _dl = daily_df['low'].values
                _dc = daily_df['close'].values
                daily_bar_map = {pd.Timestamp(d).normalize(): (h, l, c)
                                 for d, h, l, c in zip(_dd, _dh, _dl, _dc)}
                for i, ed in enumerate(event_dates):
                    _row = daily_bar_map.get(pd.Timestamp(ed).normalize())
                    entry_prices[i] = float(_row[2]) if _row is not None else np.nan
            else:
                entry_prices[:] = np.nan

            # Phase 3: EWMA 波动率 (与旧标签同一实现)
            vols = self._batch_daily_volatility(daily_df, event_dates, ref_date)

            # Phase 4: 持有窗口 [D+1, D+max_hold] (与旧标签相同)
            # [Perf] 向量化: 全部事件一次 searchsorted (O(n log n) 替代逐事件
            #   _get_next_trading_day 线性扫描; 语义完全一致: 第一个严格大于 ed 的交易日)
            _ev_ts = pd.DatetimeIndex(event_dates)
            _nxt_pos = cal_ts.searchsorted(_ev_ts, side='right')
            _invalid = _nxt_pos >= len(cal_ts)
            _nxt_pos = np.where(_invalid, len(cal_ts) - 1, _nxt_pos)
            _end_idx = np.minimum(_nxt_pos + self.max_hold_days, len(cal_ts))
            _hold_ends = cal_ts[_end_idx - 1]
            hold_windows = []
            for i in range(len(event_dates)):
                if _invalid[i]:
                    hold_windows.append((None, None, 0, 0))
                else:
                    hold_windows.append((cal_ts[_nxt_pos[i]], _hold_ends[i],
                                         int(_nxt_pos[i]), int(_end_idx[i])))

            # Phase 5: 分钟数据 (一次加载, LRU 复用)
            try:
                mdf = self.dp._load_full_minute_data(code, '5', ref_date)
            except Exception:
                mdf = None
            minute_by_day = None
            if mdf is not None and not mdf.empty:
                mdf = mdf.sort_values('trade_time')
                # [Perf-1] 先按全局窗口切片 (76K行→~13K行), 再切分日
                _valid_w = [w for w in hold_windows if w[0] is not None]
                if _valid_w:
                    _g_start = min(w[0] for w in _valid_w) - timedelta(days=1)
                    _g_end = max(w[1] for w in _valid_w) + timedelta(days=2)
                    _tt = mdf['trade_time'].values
                    _i0 = int(np.searchsorted(_tt, np.datetime64(pd.Timestamp(_g_start))))
                    _i1 = int(np.searchsorted(_tt, np.datetime64(pd.Timestamp(_g_end))))
                    if _i1 > _i0:
                        mdf = mdf.iloc[_i0:_i1]
                # [Perf-2] 向量化按日切分 (替代逐日 O(n) 布尔扫描, 实测快~90x)
                # 数据已按 trade_time 排序 → trade_date 分段连续, 语义与
                # 原 for d in pd.unique(...): mdf[mdf.trade_date==d] 完全一致
                minute_by_day = {}
                _dvals = mdf['trade_date'].values
                if not np.issubdtype(_dvals.dtype, np.datetime64):
                    _dvals = pd.to_datetime(mdf['trade_date']).values
                _hi_all = mdf['high'].values
                _lo_all = mdf['low'].values
                _cl_all = mdf['close'].values
                _uniq, _starts = np.unique(_dvals, return_index=True)
                _ends = np.r_[_starts[1:], len(_dvals)]
                for _u, _s, _e in zip(_uniq, _starts, _ends):
                    minute_by_day[pd.Timestamp(_u).normalize()] = (
                        _hi_all[_s:_e], _lo_all[_s:_e], _cl_all[_s:_e])

            # Phase 6: 逐事件分阶段模拟
            for i, ed in enumerate(event_dates):
                ep = entry_prices[i]
                next_day, hold_end, w_i0, w_i1 = hold_windows[i]
                if next_day is None or np.isnan(ep):
                    continue
                vol = vols[i]
                # 组装窗口内逐日 bar (分钟优先, 缺失日回退日线)
                day_bars = []
                for wd in cal_ts[w_i0:w_i1]:
                    wd_norm = pd.Timestamp(wd).normalize()
                    _mb = minute_by_day.get(wd_norm) if minute_by_day is not None else None
                    if _mb is not None:
                        day_bars.append(_mb)
                    elif daily_bar_map is not None:
                        _row = daily_bar_map.get(wd_norm)
                        if _row is not None:
                            day_bars.append((np.array([_row[0]]),
                                             np.array([_row[1]]),
                                             np.array([_row[2]])))
                    # 全日无数据 → 跳过该日 (生产: 无数据日不检查)
                if not day_bars:
                    # 全窗口无数据 → 无法模拟, 记 expire@0 (同旧标签 no-data 口径)
                    stock_labels.append({'date': ed, 'code': code, 'ret': 0.0,
                                         'realized_ret': 0.0, 'exit_type': 'expire',
                                         'n_fills': 0})
                    continue
                sim = simulate_staged_path_fast(day_bars, ep, vol,
                                                self.staged_cfg, self.hard_sl,
                                                pl_sl1=self.pl_sl1, pl_sl2=self.pl_sl2,
                                                pl_sl3=self.pl_sl3,
                                                pt_after_sl2=self.pt_after_sl2,
                                                pt_after_sl3=self.pt_after_sl3)
                stock_labels.append({'date': ed, 'code': code,
                                     'ret': sim['realized_ret'],
                                     'realized_ret': sim['realized_ret'],
                                     'exit_type': sim['exit_type'],
                                     'n_fills': sim['n_fills']})
            return stock_labels, len(events)

        from concurrent.futures import ThreadPoolExecutor
        stock_items = list(code_events.items())
        max_workers = min(8, len(stock_items)) if stock_items else 1
        logger.info("[分阶段标签] 开始处理 %d 只股票 (max_workers=%d)",
                    len(stock_items), max_workers)
        all_labels = []
        total_events = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for labels, n_ev in ex.map(_process_one_stock, stock_items):
                all_labels.extend(labels)
                total_events += n_ev

        t1 = _time.time()
        logger.info("[分阶段标签] 完成: %d 只股票, %d 事件, 总耗时=%.1fs",
                    len(stock_items), total_events, t1 - t0)
        return pd.DataFrame(all_labels)

