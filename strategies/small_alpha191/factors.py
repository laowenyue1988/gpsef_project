# -*- coding: utf-8 -*-
"""
Alpha191 因子计算模块（Numba 加速版）
基于 pandas 和 numpy 实现 WorldQuant Alpha191 全部 191 个因子 + Alpha101 扩展因子

优化内容：
1. 使用多进程并行计算加速因子计算
2. 优化数据类型减少内存占用
3. 使用增量写入避免一次性合并大量数据
4. 优化滚动窗口计算，减少重复计算
5. 使用向量化操作替代循环
6. 【2026-03-16优化】批量加载时根据日期范围过滤，避免加载过多历史数据导致内存溢出
7. 【2026-03-25关键修复】批量加载时增加复权因子版本检查，过期因子自动跳过并触发重算
8. 【2026-03-26关键修复】批量加载后检测完全缺失的股票（本地无因子文件），并触发补充计算
9. 【2026-03-26关键修复】批量加载后检测日期范围不完整的股票，并触发补充计算
10. 【2026-04-03关键修复】修复多进程随机种子设置，确保因子计算结果确定性
11. 【2026-04-09 Numba加速】使用 Numba 加速滚动窗口自定义函数，提升因子计算速度约 2-3 倍
"""

import random
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings
import logging
import os
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from datetime import datetime, timedelta
from gpsef.utils.fracdiff import fracDiff_FFD, getWeights_FFD

# 尝试导入 Numba
try:
    from numba import jit, prange, float64, int64, boolean
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # 定义空的装饰器，保证在不安装 Numba 时也能运行
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range
    float64 = int64 = boolean = None
    logging.warning("Numba 未安装，将使用纯 Python 回退，建议安装: pip install numba")

SEED = 42  # 与 main_strategy_fixed.py 保持一致
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# ==================== Numba 加速的辅助函数 ====================
@jit(nopython=True, cache=True)
def _rank_last_numba(x):
    """计算窗口内最后一个值在窗口中的百分等级（0-1）"""
    n = len(x)
    if n == 0:
        return np.nan
    last_val = x[-1]
    # 计算小于 last_val 的个数 + 等于 last_val 的个数的一半
    less_count = 0
    eq_count = 0
    for i in range(n):
        if x[i] < last_val:
            less_count += 1
        elif x[i] == last_val:
            eq_count += 1
    # 百分等级公式
    return (less_count + 0.5 * eq_count) / n


@jit(nopython=True, cache=True)
def _argmin_last_numba(x, window):
    """返回最近窗口内最小值距离末尾的天数（固定窗口大小）。
    [A7-20260905] NaN 感知：跳过 NaN，min_idx 为原始窗口坐标（等价 np.nanargmin 语义），
    与 pandas 回退路径统一；全 NaN 窗口返回 NaN。"""
    n = len(x)
    if n == 0:
        return np.nan
    min_val = np.inf
    min_idx = -1
    for i in range(n):
        if not np.isnan(x[i]) and x[i] < min_val:
            min_val = x[i]
            min_idx = i
    if min_idx < 0:
        return np.nan
    return window - 1 - min_idx

@jit(nopython=True, cache=True)
def _argmax_last_numba(x, window):
    """返回最近窗口内最大值距离末尾的天数（固定窗口大小）。
    [A7-20260905] NaN 感知：跳过 NaN，max_idx 为原始窗口坐标（等价 np.nanargmax 语义）。"""
    n = len(x)
    if n == 0:
        return np.nan
    max_val = -np.inf
    max_idx = -1
    for i in range(n):
        if not np.isnan(x[i]) and x[i] > max_val:
            max_val = x[i]
            max_idx = i
    if max_idx < 0:
        return np.nan
    return window - 1 - max_idx


@jit(nopython=True, cache=True)
def _linear_slope_numba(x):
    """计算序列的线性回归斜率（最小二乘法）"""
    n = len(x)
    if n < 2:
        return np.nan
    # x 值: 1,2,...,n
    sum_x = n * (n + 1) / 2.0
    sum_y = np.sum(x)
    sum_xy = 0.0
    for i in range(n):
        sum_xy += (i + 1) * x[i]
    sum_x2 = n * (n + 1) * (2 * n + 1) / 6.0
    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return np.nan
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return slope


@jit(nopython=True, cache=True)
def _polyfit_slope_numba(x):
    """与 _linear_slope_numba 相同，用于 polyfit 场景"""
    return _linear_slope_numba(x)


@jit(nopython=True, cache=True)
def _decay_linear_weighted_avg(x):
    n = len(x)
    if n == 0:
        return np.nan
    total_weight = n * (n + 1) / 2.0
    weighted_sum = 0.0
    for i in range(n):
        weighted_sum += x[i] * (i + 1)   # 修正：最近一天权重最大
    return weighted_sum / total_weight

@jit(nopython=True, cache=True)
def _wma_numba(x):
    """加权移动平均（权重 1,2,...,n）"""
    n = len(x)
    if n == 0:
        return np.nan
    total_weight = n * (n + 1) / 2.0
    weighted_sum = 0.0
    for i in range(n):
        weighted_sum += x[i] * (i + 1)
    return weighted_sum / total_weight


@jit(nopython=True, cache=True)
def _product_numba(x):
    """数组元素乘积"""
    prod = 1.0
    for val in x:
        prod *= val
    return prod


@jit(nopython=True, cache=True)
def _sma_numba(arr, window, weight):
    """Wilder 平滑 (EWMA) 使用 Numba 循环实现"""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    alpha = weight / window
    result[0] = arr[0]
    for i in range(1, n):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    return result


# ==================== 基础工具函数（优化版）====================

# ==================== [A1-20260905] 横截面 rank 两遍法基础设施 ====================
# 设计: 'timeseries' 旧行为(个股时序 pct rank, 默认, 兼容 test2/生产);
#       'collect'    记录每个 rank() 调用点的参数序列到 _RANK_REGISTRY;
#       'replay'     从预构建的截面 rank memmap 面板查表返回真实截面 rank 序列。
# rank_key = f"{suffix}#{调用序号}"; 调用序号在每股票计算入口重置,
# 公式均为无条件赋值(AST 验证) → collect/replay 两遍调用序列严格一致。
_RANK_MODE = 'timeseries'
_RANK_REGISTRY = None          # collect: {key: {ts_code: (dates np.datetime64[:], vals np.float64[:])}}
_RANK_PANEL = None             # replay: {'arr': memmap3d, 'key_idx': dict, 'code_idx': dict, 'dates': np.array}
_RANK_PANEL_DIR = None
_RANK_COUNTER = [0]
_RANK_CTX = {'ts_code': None, 'dates': None, 'suffix': ''}
_RANK_MISSING_REPORTED = set()   # [终审加固-20260905] replay 漂移告警去重: {(kind, name)}


def _rank_begin_stock(ts_code, dates, suffix):
    """每股票计算入口调用: 重置计数器并设置上下文"""
    _RANK_COUNTER[0] = 0
    _RANK_CTX['ts_code'] = ts_code
    _RANK_CTX['dates'] = dates
    _RANK_CTX['suffix'] = suffix


def _rank_close_panel():
    """显式关闭本进程持有的 rank 面板 memmap 底层文件句柄。
    [mmapfix-20260906] 仅置 None 不够: numpy memmap 的 __del__ 在 Windows 下不一定
    释放句柄, 需显式 _mmap.close(); 否则 day_dir 文件被占用, 主进程 rmtree 静默失败,
    目录逐日累积撑爆磁盘。"""
    global _RANK_PANEL, _RANK_PANEL_DIR
    if _RANK_PANEL is not None:
        arr = _RANK_PANEL.get('arr')
        if arr is not None:
            try:
                arr._mmap.close()
            except Exception:
                pass
    _RANK_PANEL = None
    _RANK_PANEL_DIR = None


def _rank_load_panel(panel_dir):
    """replay 模式下 worker 进程惰性加载截面 rank 面板(每进程每目录一次)。
    [fix] panel_dir 变化时强制重载 — 串行模式/同进程多日连跑时防止沿用上一日期的陈旧面板。"""
    global _RANK_PANEL, _RANK_PANEL_DIR
    if _RANK_PANEL is not None and _RANK_PANEL_DIR == panel_dir:
        return
    _rank_close_panel()
    arr = np.load(os.path.join(panel_dir, 'cs_rank_panel.npy'), mmap_mode='r')
    meta = np.load(os.path.join(panel_dir, 'cs_rank_meta.npz'), allow_pickle=False)
    codes = [str(c) for c in meta['codes']]
    keys = [str(k) for k in meta['keys']]
    _RANK_PANEL = {
        'arr': arr,
        'key_idx': {k: i for i, k in enumerate(keys)},
        'code_idx': {c: i for i, c in enumerate(codes)},
        'dates': meta['dates'],
    }
    _RANK_PANEL_DIR = panel_dir


def _rank_release_panel():
    """[A12-20260905] 释放本进程持有的 rank 面板 memmap 句柄。
    Windows 下 memmap 句柄不释放则面板目录无法删除 (串行模式父进程自身加载了面板)。
    [mmapfix-20260906] 显式 _mmap.close() (置 None 在 numpy 2.x 下不释放句柄)。"""
    _rank_close_panel()


def _rank_panel_lookup(key, ts_code, dates):
    """从面板查 (key, ts_code, dates) 对应的截面 rank 序列
    [终审加固-20260905] collect/replay 调用序列漂移时不再静默: 未知 key/股票一次性 error 告警
    (正常链路永不触发; 触发即表示公式控制流在两遍间不一致, 因子将被静默污染)。"""
    p = _RANK_PANEL
    ki = p['key_idx'].get(key)
    ci = p['code_idx'].get(ts_code)
    n = len(dates)
    if ki is None:
        _k = ('key', key)
        if _k not in _RANK_MISSING_REPORTED:
            _RANK_MISSING_REPORTED.add(_k)
            logger.error(f"[A1] rank 面板缺失调用点 key='{key}' → 返回全NaN "
                         f"(collect/replay 调用序列漂移, 该调用点因子被静默污染!)")
        return np.full(n, np.nan, dtype=np.float64)
    if ci is None:
        _k = ('code', ts_code)
        if _k not in _RANK_MISSING_REPORTED:
            _RANK_MISSING_REPORTED.add(_k)
            logger.error(f"[A1] rank 面板缺失股票 {ts_code} → 返回全NaN "
                         f"(collect/replay 股票集漂移, 该股因子被静默污染!)")
        return np.full(n, np.nan, dtype=np.float64)
    pos = np.searchsorted(p['dates'], dates)
    return np.asarray(p['arr'][ki, pos, ci], dtype=np.float64)


def rank(series: pd.Series) -> pd.Series:
    """横截面排名（归一化到 0-1）
    [A1-20260905] 三模式: timeseries=旧行为(默认); collect=记录参数序列; replay=返回真实截面 rank。"""
    if _RANK_MODE == 'timeseries':
        return series.rank(pct=True)
    key = f"{_RANK_CTX['suffix']}#{_RANK_COUNTER[0]}"
    _RANK_COUNTER[0] += 1
    if _RANK_MODE == 'collect':
        _RANK_REGISTRY.setdefault(key, {})[_RANK_CTX['ts_code']] = (
            _RANK_CTX['dates'], np.asarray(series.values, dtype=np.float64))
        return series.rank(pct=True)   # collect 遍产出丢弃, 返回值无关
    # replay
    vals = _rank_panel_lookup(key, _RANK_CTX['ts_code'], _RANK_CTX['dates'])
    return pd.Series(vals, index=series.index)


def build_cs_rank_panels(registry: dict, df_daily: pd.DataFrame, work_dir: str,
                         rank_output: bool = True) -> str:
    """[A1] 由 collect 遍 registry 构建截面 rank memmap 面板。
    每日截面仅含当日 tradestatus=='1' 的股票(停牌剔除, 陈旧价量不污染截面)。
    返回 panel_dir (含 cs_rank_panel.npy + cs_rank_meta.npz)。
    [PERF-20260905] rank_output=False: 导出未 rank 的原始参数矩阵 (cs_arg_panel.npy),
    供 snapshot_fast 逐日打补丁+重排名使用; 默认 True 行为与之前完全一致。"""
    os.makedirs(work_dir, exist_ok=True)
    all_dates = np.unique(df_daily['trade_date'].values.astype('datetime64[ns]'))  # 面板日期轴(去重排序)
    codes = sorted(df_daily['ts_code'].unique())
    code_pos = {c: i for i, c in enumerate(codes)}

    # 停牌面板: [n_dates, n_codes], True=当日正常交易
    # [终审加固-20260905] 外部以 cross_section 直调且数据无 tradestatus 列时,
    #   不再 KeyError: 回退为全部参与截面并告警一次 (快照链路必含该列, 不受影响)。
    if 'tradestatus' in df_daily.columns:
        ts_frame = df_daily[['trade_date', 'ts_code', 'tradestatus']].copy()
        ts_frame['tradestatus'] = ts_frame['tradestatus'].astype(str)
        ts_pivot = ts_frame.pivot_table(index='trade_date', columns='ts_code',
                                        values='tradestatus', aggfunc='first')
        ts_pivot = ts_pivot.reindex(index=pd.DatetimeIndex(all_dates), columns=codes)
        ts_ok = (ts_pivot.values == '1')
    else:
        logger.warning("[A1] df_daily 无 tradestatus 列: 停牌掩码回退为全部参与截面 (仅外部直调场景)")
        ts_ok = np.ones((len(all_dates), len(codes)), dtype=bool)

    key_list = sorted(registry.keys())
    if rank_output:
        panel_path = os.path.join(work_dir, 'cs_rank_panel.npy')
        meta_name = 'cs_rank_meta.npz'
    else:
        panel_path = os.path.join(work_dir, 'cs_arg_panel.npy')
        meta_name = 'cs_arg_meta.npz'
    panel = np.lib.format.open_memmap(
        panel_path, mode='w+', dtype=np.float32,
        shape=(len(key_list), len(all_dates), len(codes)))

    for ki, key in enumerate(key_list):
        mat = np.full((len(all_dates), len(codes)), np.nan, dtype=np.float32)
        for code, (dates, vals) in registry[key].items():
            pos = np.searchsorted(all_dates, dates)
            n = min(len(pos), len(vals))
            mat[pos[:n], code_pos[code]] = vals[:n].astype(np.float32)
        if rank_output:
            mat[~ts_ok] = np.nan   # 停牌日剔除出截面
            ranked = pd.DataFrame(mat).rank(axis=1, pct=True).values.astype(np.float32)
            panel[ki] = ranked
        else:
            panel[ki] = mat        # [PERF] 原始参数矩阵 (不打掩码不 rank, 逐日重建时处理)
        if (ki + 1) % 20 == 0:
            logger.info(f"  [A1] 截面 rank 面板构建: {ki + 1}/{len(key_list)}")
    panel.flush()

    np.savez(os.path.join(work_dir, meta_name),
             keys=np.array(key_list), codes=np.array(codes), dates=all_dates)
    logger.info(f"[A1] 截面 rank 面板完成: {len(key_list)} keys × {len(all_dates)} dates × {len(codes)} codes"
                f"{'' if rank_output else ' (raw arg 模式)'}")
    return work_dir
# ====================================================================


def ts_rank(series: pd.Series, window: int) -> pd.Series:
    """时间序列排名：对过去 window 个值（含当前）计算当前值的百分等级"""
    if NUMBA_AVAILABLE:
        vals = series.values
        n = len(vals)
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            start = max(0, i - window + 1)
            win = vals[start:i+1]
            # 剔除 NaN
            valid = win[~np.isnan(win)]
            if len(valid) == 0:
                out[i] = np.nan
            else:
                out[i] = _rank_last_numba(valid)
        return pd.Series(out, index=series.index)
    else:
        def _rank_last(x):
            if len(x) == 0:
                return np.nan
            return (x.rank(pct=True).iloc[-1])
        return series.rolling(window=window, min_periods=1).apply(_rank_last, raw=False)


def ts_sum(series: pd.Series, window: int) -> pd.Series:
    """时间序列求和"""
    return series.rolling(window=window, min_periods=1).sum()


def ts_mean(series: pd.Series, window: int) -> pd.Series:
    """时间序列均值"""
    return series.rolling(window=window, min_periods=1).mean()


def ts_std(series: pd.Series, window: int) -> pd.Series:
    """时间序列标准差"""
    return series.rolling(window=window, min_periods=1).std()


def ts_max(series: pd.Series, window: int) -> pd.Series:
    """时间序列最大值"""
    return series.rolling(window=window, min_periods=1).max()


def ts_min(series: pd.Series, window: int) -> pd.Series:
    """时间序列最小值"""
    return series.rolling(window=window, min_periods=1).min()


def ts_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """时间序列相关系数
    [A6-20260905] min_periods=1 → window: 暖机期不足窗口输出 NaN，
    消除前 2 样本 corr 恒 ±1 的伪相关（下游缺失率过滤处理 NaN）。"""
    return x.rolling(window=window, min_periods=window).corr(y)


def ts_cov(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """时间序列协方差
    [A6-20260905] min_periods=1 → window: 同 ts_corr，消除暖机伪值。"""
    return x.rolling(window=window, min_periods=window).cov(y)


def ts_delta(series: pd.Series, period: int) -> pd.Series:
    """时间序列差分"""
    return series.diff(period)


def ts_delay(series: pd.Series, period: int) -> pd.Series:
    """时间序列延迟"""
    return series.shift(period)


def ts_sma(series: pd.Series, window: int, weight: float) -> pd.Series:
    """指数加权移动平均（Wilder 平滑），直接调用 pandas ewm 保证与原始脚本完全一致"""
    return series.ewm(alpha=weight / window, adjust=False).mean()


def decay_linear(series: pd.Series, window: int) -> pd.Series:
    """线性衰减加权平均"""
    if NUMBA_AVAILABLE:
        vals = series.values
        n = len(vals)
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            if i >= window - 1:
                win = vals[i-window+1:i+1]
                valid = win[~np.isnan(win)]
                if len(valid) == window:  # 必须满窗口
                    out[i] = _decay_linear_weighted_avg(valid)
                else:
                    out[i] = np.nan
        return pd.Series(out, index=series.index)
    else:
        weights = np.arange(1, window + 1)
        weights = weights / weights.sum()
        return series.rolling(window=window, min_periods=1).apply(
            lambda x: np.dot(x, weights[::-1]) if len(x) == window else np.nan, raw=True
        )


def signed_power(series: pd.Series, power: float) -> pd.Series:
    """带符号的幂运算"""
    return np.sign(series) * np.abs(series) ** power


def wma(series: pd.Series, window: int) -> pd.Series:
    """加权移动平均（线性权重）"""
    if NUMBA_AVAILABLE:
        vals = series.values
        n = len(vals)
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            if i >= window - 1:
                win = vals[i-window+1:i+1]
                valid = win[~np.isnan(win)]
                if len(valid) == window:
                    out[i] = _wma_numba(valid)
                else:
                    out[i] = np.nan
        return pd.Series(out, index=series.index)
    else:
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=1).apply(
            lambda x: np.average(x, weights=weights[:len(x)]) if len(x) > 0 else np.nan, raw=True
        )

def lowday(series: pd.Series, window: int) -> pd.Series:
    """最近 window 天内最低价至今的天数
    [A7-20260905] 两路径统一 nanargmin 语义：忽略 NaN、极值位置取原始窗口坐标。"""
    if NUMBA_AVAILABLE:
        vals = series.values
        n = len(vals)
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            start = max(0, i - window + 1)
            win = vals[start:i+1]
            out[i] = _argmin_last_numba(win, window)   # [A7] 传完整窗口, 由内核跳过 NaN
        return pd.Series(out, index=series.index)
    else:
        def _argmin_last(x):
            if len(x) == 0 or np.isnan(x).all():
                return np.nan
            return window - 1 - np.nanargmin(x)   # [A7] nanargmin 忽略 NaN, 保持原始坐标
        return series.rolling(window=window, min_periods=1).apply(_argmin_last, raw=True)

def highday(series: pd.Series, window: int) -> pd.Series:
    """最近 window 天内最高价至今的天数
    [A7-20260905] 两路径统一 nanargmax 语义。"""
    if NUMBA_AVAILABLE:
        vals = series.values
        n = len(vals)
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            start = max(0, i - window + 1)
            win = vals[start:i+1]
            out[i] = _argmax_last_numba(win, window)   # [A7]
        return pd.Series(out, index=series.index)
    else:
        def _argmax_last(x):
            if len(x) == 0 or np.isnan(x).all():
                return np.nan
            return window - 1 - np.nanargmax(x)   # [A7]
        return series.rolling(window=window, min_periods=1).apply(_argmax_last, raw=True)


def count(condition: pd.Series, window: int) -> pd.Series:
    """滚动窗口内条件为真的次数"""
    return condition.rolling(window=window, min_periods=1).sum()


def sumif(condition: pd.Series, value: pd.Series, window: int) -> pd.Series:
    """滚动窗口内满足条件的值的和"""
    masked = value.where(condition, 0)
    return masked.rolling(window=window, min_periods=1).sum()


def sumac(series: pd.Series) -> pd.Series:
    """累积和（cumulative sum）"""
    return series.cumsum()


def scale(series: pd.Series) -> pd.Series:
    """将序列缩放到和为1（横截面归一化）"""
    s = series.abs().sum()
    if s == 0 or np.isnan(s):
        return pd.Series(np.nan, index=series.index)
    return series / s


def product(series: pd.Series, window: int) -> pd.Series:
    """滚动窗口内值的乘积"""
    if NUMBA_AVAILABLE:
        vals = series.values
        n = len(vals)
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            if i >= window - 1:
                win = vals[i-window+1:i+1]
                valid = win[~np.isnan(win)]
                if len(valid) == window:
                    out[i] = _product_numba(valid)
                else:
                    out[i] = np.nan
        return pd.Series(out, index=series.index)
    else:
        return series.rolling(window=window, min_periods=1).apply(
            lambda x: np.prod(x) if len(x) > 0 else np.nan, raw=True
        )


# ==================== 优化的单股票因子计算函数（用于多进程）====================
def _compute_all_alphas(OPEN, HIGH, LOW, CLOSE, VOLUME, VWAP, AMOUNT, RET,
                        INDEX_CLOSE, INDEX_OPEN, suffix=''):
    """
    核心因子计算逻辑，返回一个包含所有因子列的 DataFrame。
    参数中的序列为 pandas Series，后缀 suffix 用于构造列名。
    """
    import numpy as np
    import pandas as pd

    # 内部辅助函数
    def if_else(cond, a, b):
        return pd.Series(np.where(cond, a, b), index=cond.index)

    # 预计算常用中间变量（与原版完全一致）
    delay_close = ts_delay(CLOSE, 1)
    delay_open = ts_delay(OPEN, 1)
    delay_high = ts_delay(HIGH, 1)
    delay_low = ts_delay(LOW, 1)
    delay_volume = ts_delay(VOLUME, 1)

    rank_high = rank(HIGH)
    rank_low = rank(LOW)
    rank_close = rank(CLOSE)
    rank_open = rank(OPEN)
    rank_volume = rank(VOLUME)
    rank_vwap = rank(VWAP)

    mean_volume_20 = ts_mean(VOLUME, 20)
    mean_volume_40 = ts_mean(VOLUME, 40)
    mean_volume_50 = ts_mean(VOLUME, 50)
    mean_volume_60 = ts_mean(VOLUME, 60)
    mean_volume_80 = ts_mean(VOLUME, 80)
    mean_volume_150 = ts_mean(VOLUME, 150)
    mean_volume_180 = ts_mean(VOLUME, 180)

    mean_close_6 = ts_mean(CLOSE, 6)
    mean_close_7 = ts_mean(CLOSE, 7)
    mean_close_12 = ts_mean(CLOSE, 12)
    mean_close_20 = ts_mean(CLOSE, 20)
    mean_close_24 = ts_mean(CLOSE, 24)

    std_close_20 = ts_std(CLOSE, 20)

    result = {}

    # ========== 批量计算所有因子（完全复制原代码，表头已替换） ==========
    # 以下为完整的 191 个因子，每个 result['alpha_XXX'] 已替换为 result[f'alpha_XXX{suffix}']
    result[f'alpha_001{suffix}'] = -1 * ts_corr(rank(ts_delta(np.log(VOLUME), 1)), rank((CLOSE - OPEN) / OPEN), 6)
    result[f'alpha_002{suffix}'] = -1 * ts_delta(((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW), 1)

    cond_eq = (CLOSE == delay_close)
    cond_gt = (CLOSE > delay_close)
    part1 = CLOSE - np.minimum(LOW, delay_close)
    part2 = CLOSE - np.maximum(HIGH, delay_close)
    val = if_else(cond_gt, part1, part2)
    val = if_else(cond_eq, 0, val)
    result[f'alpha_003{suffix}'] = ts_sum(val, 6)

    ma8 = ts_mean(CLOSE, 8)
    std8 = ts_std(CLOSE, 8)
    ma2 = ts_mean(CLOSE, 2)
    cond1 = (ma8 + std8) < ma2
    cond2 = ma2 < (ma8 - std8)
    cond3 = (VOLUME / mean_volume_20) >= 1
    result[f'alpha_004{suffix}'] = if_else(cond1, -1, if_else(cond2, 1, if_else(cond3, 1, -1)))

    result[f'alpha_005{suffix}'] = -1 * ts_max(ts_corr(ts_rank(VOLUME, 5), ts_rank(HIGH, 5), 5), 3)
    result[f'alpha_006{suffix}'] = -1 * rank(ts_delta(OPEN * 0.85 + HIGH * 0.15, 4))
    result[f'alpha_007{suffix}'] = (rank(ts_max(VWAP - CLOSE, 3)) + rank(ts_min(VWAP - CLOSE, 3))) * rank(
        ts_delta(VOLUME, 3))
    result[f'alpha_008{suffix}'] = -1 * rank(ts_delta((HIGH + LOW) / 2 * 0.2 + VWAP * 0.8, 4))
    result[f'alpha_009{suffix}'] = ts_sma(((HIGH + LOW) / 2 - (delay_high + delay_low) / 2) * (HIGH - LOW) / VOLUME, 7, 2)
    result[f'alpha_010{suffix}'] = rank(
        ts_max(pd.Series(np.where(RET < 0, ts_std(RET, 20), CLOSE) ** 2, index=CLOSE.index), 5))
    result[f'alpha_011{suffix}'] = ts_sum(((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW) * VOLUME, 6)
    result[f'alpha_012{suffix}'] = rank(OPEN - ts_mean(VWAP, 10)) * (-1 * rank(np.abs(CLOSE - VWAP)))
    result[f'alpha_013{suffix}'] = np.sqrt(HIGH * LOW) - VWAP
    result[f'alpha_014{suffix}'] = CLOSE - ts_delay(CLOSE, 5)
    result[f'alpha_015{suffix}'] = OPEN / delay_close - 1
    result[f'alpha_016{suffix}']= -1 * ts_max(rank(ts_corr(rank_volume, rank_vwap, 5)), 5)
    result[f'alpha_017{suffix}'] = rank(VWAP - ts_max(VWAP, 15)) ** ts_delta(CLOSE, 5)
    result[f'alpha_018{suffix}'] = CLOSE / ts_delay(CLOSE, 5)

    cond_lt = (CLOSE < ts_delay(CLOSE, 5))
    cond_eq = (CLOSE == ts_delay(CLOSE, 5))
    part1 = (CLOSE - ts_delay(CLOSE, 5)) / ts_delay(CLOSE, 5)
    part2 = (CLOSE - ts_delay(CLOSE, 5)) / CLOSE
    result[f'alpha_019{suffix}'] = if_else(cond_lt, part1, if_else(cond_eq, 0, part2))

    result[f'alpha_020{suffix}'] = (CLOSE - ts_delay(CLOSE, 6)) / ts_delay(CLOSE, 6) * 100
    # 使用 Numba 加速的线性回归斜率
    if NUMBA_AVAILABLE:
        vals = mean_close_6.values
        n = len(vals)
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            if i >= 5:
                win = vals[i - 5:i + 1]
                valid = win[~np.isnan(win)]
                if len(valid) >= 2:  # 至少2个点才能拟合斜率
                    out[i] = _linear_slope_numba(valid)
        result[f'alpha_021{suffix}'] = pd.Series(out, index=mean_close_6.index)
    else:
        result[f'alpha_021{suffix}'] = mean_close_6.rolling(6, min_periods=1).apply(
            lambda x: np.polyfit(np.arange(1, len(x) + 1), x, 1)[0] if len(x) >= 2 else np.nan, raw=True
        )

    val = (CLOSE - mean_close_6) / mean_close_6
    val_delay3 = ts_delay(val, 3)
    result[f'alpha_022{suffix}'] = ts_sma(val - val_delay3, 12, 1)

    up = ts_sma(if_else(CLOSE > delay_close, std_close_20, 0), 20, 1)
    down = ts_sma(if_else(CLOSE <= delay_close, std_close_20, 0), 20, 1)
    result[f'alpha_023{suffix}']= up / (up + down) * 100

    result[f'alpha_024{suffix}'] = ts_sma(CLOSE - ts_delay(CLOSE, 5), 5, 1)
    result[f'alpha_025{suffix}'] = -1 * rank(ts_delta(CLOSE, 7) * (1 - rank(decay_linear(VOLUME / mean_volume_20, 9)))) * (
            1 + rank(ts_sum(RET, 250)))
    result[f'alpha_026{suffix}'] = (mean_close_7 - CLOSE) + ts_corr(VWAP, ts_delay(CLOSE, 5), 230)
    result[f'alpha_027{suffix}']= wma(
        (CLOSE - ts_delay(CLOSE, 3)) / ts_delay(CLOSE, 3) * 100 + (CLOSE - ts_delay(CLOSE, 6)) / ts_delay(CLOSE,
                                                                                                          6) * 100,
        12)

    numerator = CLOSE - ts_min(LOW, 9)
    denominator = ts_max(HIGH, 9) - ts_min(LOW, 9)
    val = numerator / denominator * 100
    sma3 = ts_sma(val, 3, 1)
    result[f'alpha_028{suffix}'] = 3 * sma3 - 2 * ts_sma(sma3, 3, 1)

    result[f'alpha_029{suffix}'] = (CLOSE - ts_delay(CLOSE, 6)) / ts_delay(CLOSE, 6) * VOLUME
    result[f'alpha_030{suffix}']= np.nan
    result[f'alpha_031{suffix}']= (CLOSE - mean_close_12) / mean_close_12 * 100
    result[f'alpha_032{suffix}'] = -1 * ts_sum(rank(ts_corr(rank_high, rank_volume, 3)), 3)
    result[f'alpha_033{suffix}'] = (-1 * ts_min(LOW, 5) + ts_delay(ts_min(LOW, 5), 5)) * rank(
        (ts_sum(RET, 240) - ts_sum(RET, 20)) / 220) * ts_rank(VOLUME, 5)
    result[f'alpha_034{suffix}'] = mean_close_12 / CLOSE
    result[f'alpha_035{suffix}'] = -1 * np.minimum(rank(decay_linear(ts_delta(OPEN, 1), 15)),
                                          rank(decay_linear(ts_corr(VOLUME, OPEN, 17), 7)))
    result[f'alpha_036{suffix}'] = rank(ts_sum(ts_corr(rank_volume, rank_vwap, 6), 2))
    result[f'alpha_037{suffix}'] = -1 * rank(
        ts_sum(OPEN, 5) * ts_sum(RET, 5) - ts_delay(ts_sum(OPEN, 5) * ts_sum(RET, 5), 10))
    result[f'alpha_038{suffix}'] = if_else(ts_mean(HIGH, 20) < HIGH, -1 * ts_delta(HIGH, 2), 0)
    result[f'alpha_039{suffix}'] = -1 * (rank(decay_linear(ts_delta(CLOSE, 2), 8)) - rank(
        decay_linear(ts_corr(VWAP * 0.3 + OPEN * 0.7, ts_sum(mean_volume_180, 37), 14), 12)))

    up_vol = ts_sum(if_else(CLOSE > delay_close, VOLUME, 0), 26)
    down_vol = ts_sum(if_else(CLOSE <= delay_close, VOLUME, 0), 26)
    result[f'alpha_040{suffix}'] = up_vol / down_vol * 100

    result[f'alpha_041{suffix}'] = -1 * rank(ts_max(ts_delta(VWAP, 3), 5))
    result[f'alpha_042{suffix}'] = -1 * rank(ts_std(HIGH, 10)) * ts_corr(HIGH, VOLUME, 10)
    result[f'alpha_043{suffix}'] = ts_sum(if_else(CLOSE > delay_close, VOLUME, if_else(CLOSE < delay_close, -VOLUME, 0)), 6)
    result[f'alpha_044{suffix}'] = ts_rank(decay_linear(ts_corr(LOW, ts_mean(VOLUME, 10), 7), 6), 4) + ts_rank(
        decay_linear(ts_delta(VWAP, 3), 10), 15)
    result[f'alpha_045{suffix}'] = rank(ts_delta(CLOSE * 0.6 + OPEN * 0.4, 1)) * rank(ts_corr(VWAP, mean_volume_150, 15))
    result[f'alpha_046{suffix}'] = (ts_mean(CLOSE, 3) + mean_close_6 + mean_close_12 + ts_mean(CLOSE, 24)) / (4 * CLOSE)
    result[f'alpha_047{suffix}'] = ts_sma((ts_max(HIGH, 6) - CLOSE) / (ts_max(HIGH, 6) - ts_min(LOW, 6)) * 100, 9, 1)

    sign1 = np.sign(CLOSE - delay_close)
    sign2 = np.sign(delay_close - ts_delay(CLOSE, 2))
    sign3 = np.sign(ts_delay(CLOSE, 2) - ts_delay(CLOSE, 3))
    result[f'alpha_048{suffix}'] = -1 * (rank(sign1 + sign2 + sign3) * ts_sum(VOLUME, 5)) / ts_sum(VOLUME, 20)

    result[f'alpha_049{suffix}'] = ts_sum(if_else((HIGH + LOW) >= (delay_high + delay_low), 0,
                                         np.maximum(np.abs(HIGH - delay_high), np.abs(LOW - delay_low))), 12) / \
                          ts_sum(np.maximum(np.abs(HIGH - delay_high), np.abs(LOW - delay_low)), 12)

    result[f'alpha_050{suffix}'] = ts_sum(if_else((HIGH + LOW) <= (delay_high + delay_low), 0,
                                         np.maximum(np.abs(HIGH - delay_high), np.abs(LOW - delay_low))), 12) / \
                          ts_sum(np.maximum(np.abs(HIGH - delay_high), np.abs(LOW - delay_low)), 12) - \
                          ts_sum(if_else((HIGH + LOW) >= (delay_high + delay_low), 0,
                                         np.maximum(np.abs(HIGH - delay_high), np.abs(LOW - delay_low))), 12) / \
                          ts_sum(np.maximum(np.abs(HIGH - delay_high), np.abs(LOW - delay_low)), 12)

    result[f'alpha_051{suffix}'] = ts_sum(if_else((HIGH + LOW) <= (delay_high + delay_low), 0,
                                         np.maximum(np.abs(HIGH - delay_high), np.abs(LOW - delay_low))), 12) / \
                          ts_sum(np.maximum(np.abs(HIGH - delay_high), np.abs(LOW - delay_low)), 12)

    tp = (HIGH + LOW + CLOSE) / 3
    delay_tp = ts_delay(tp, 1)
    result[f'alpha_052{suffix}'] = ts_sum(np.maximum(0, HIGH - delay_tp), 26) / ts_sum(np.maximum(0, delay_tp - LOW),
                                                                              26) * 100
    result[f'alpha_053{suffix}'] = count(CLOSE > delay_close, 12) / 12 * 100
    result[f'alpha_054{suffix}'] = -1 * rank(ts_std(np.abs(CLOSE - OPEN), 10) + (CLOSE - OPEN) + ts_corr(CLOSE, OPEN, 10))

    result[f'alpha_055{suffix}'] = ts_sum(16 * (CLOSE - delay_close + (CLOSE - OPEN) / 2 + delay_close - delay_open) / \
                                 np.maximum(np.maximum(HIGH - LOW, np.abs(delay_close - HIGH)),
                                            np.abs(delay_close - LOW)) * \
                                 np.maximum(np.abs(HIGH - delay_high), np.abs(LOW - delay_low)), 20)

    rank1 = rank(OPEN - ts_min(OPEN, 12))
    rank2 = rank(ts_corr(ts_sum((HIGH + LOW) / 2, 19), ts_sum(ts_mean(VOLUME, 40), 19), 13)) ** 5
    result[f'alpha_056{suffix}'] = (rank1 < rank2).astype(float)
    result[f'alpha_057{suffix}'] = ts_sma((CLOSE - ts_min(LOW, 9)) / (ts_max(HIGH, 9) - ts_min(LOW, 9)) * 100, 3, 1)
    result[f'alpha_058{suffix}'] = count(CLOSE > delay_close, 20) / 20 * 100

    cond_eq = (CLOSE == delay_close)
    cond_gt = (CLOSE > delay_close)
    val = if_else(cond_gt, CLOSE - np.minimum(LOW, delay_close), CLOSE - np.maximum(HIGH, delay_close))
    val = if_else(cond_eq, 0, val)
    result[f'alpha_059{suffix}'] = ts_sum(val, 20)

    result[f'alpha_060{suffix}'] = ts_sum(((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW) * VOLUME, 20)
    result[f'alpha_061{suffix}'] = -1 * pd.Series(np.maximum(rank(decay_linear(ts_delta(VWAP, 1), 12)),
                                                    rank(decay_linear(rank(ts_corr(LOW, mean_volume_80, 8)), 17))),
                                         index=OPEN.index)
    result[f'alpha_062{suffix}'] = -1 * ts_corr(HIGH, rank_volume, 5)
    result[f'alpha_063{suffix}'] = ts_sma(np.maximum(CLOSE - delay_close, 0), 6, 1) / ts_sma(np.abs(CLOSE - delay_close), 6,
                                                                                    1) * 100
    result[f'alpha_064{suffix}'] = -1 * np.maximum(rank(decay_linear(ts_corr(rank_vwap, rank_volume, 4), 4)),
                                          rank(
                                              decay_linear(ts_max(ts_corr(rank_close, rank(mean_volume_60), 4), 13),
                                                           14)))
    result[f'alpha_065{suffix}'] = mean_close_6 / CLOSE
    result[f'alpha_066{suffix}'] = (CLOSE - mean_close_6) / mean_close_6 * 100
    result[f'alpha_067{suffix}'] = ts_sma(np.maximum(CLOSE - delay_close, 0), 24, 1) / ts_sma(np.abs(CLOSE - delay_close),
                                                                                     24, 1) * 100
    result[f'alpha_068{suffix}'] = ts_sma(((HIGH + LOW) / 2 - (delay_high + delay_low) / 2) * (HIGH - LOW) / VOLUME, 15, 2)

    dtm = if_else(OPEN <= delay_open, 0, np.maximum(HIGH - OPEN, OPEN - delay_open))
    dbm = if_else(OPEN >= delay_open, 0, np.maximum(OPEN - LOW, delay_open - OPEN))
    sum_dtm20 = ts_sum(dtm, 20)
    sum_dbm20 = ts_sum(dbm, 20)
    result[f'alpha_069{suffix}'] = if_else(sum_dtm20 > sum_dbm20, (sum_dtm20 - sum_dbm20) / sum_dtm20,
                                  if_else(sum_dtm20 == sum_dbm20, 0, (sum_dtm20 - sum_dbm20) / sum_dbm20))

    result[f'alpha_070{suffix}'] = ts_std(AMOUNT, 6)
    result[f'alpha_071{suffix}'] = (CLOSE - mean_close_24) / mean_close_24 * 100
    result[f'alpha_072{suffix}'] = ts_sma((ts_max(HIGH, 6) - CLOSE) / (ts_max(HIGH, 6) - ts_min(LOW, 6)) * 100, 15, 1)
    result[f'alpha_073{suffix}'] = -1 * (ts_rank(decay_linear(decay_linear(ts_corr(CLOSE, VOLUME, 10), 16), 4), 5) -
                                rank(decay_linear(ts_corr(VWAP, ts_mean(VOLUME, 30), 4), 3)))
    rank(decay_linear(ts_corr(VWAP, ts_mean(VOLUME, 30), 4), 3))
    result[f'alpha_074{suffix}'] = rank(ts_corr(ts_sum(LOW * 0.35 + VWAP * 0.65, 20), ts_sum(ts_mean(VOLUME, 40), 20), 7)) + \
                          rank(ts_corr(rank_vwap, rank_volume, 6))
    result[f'alpha_075{suffix}'] = np.nan if INDEX_CLOSE.isna().all() else count(INDEX_CLOSE < INDEX_OPEN, 50) / 50
    result[f'alpha_076{suffix}'] = ts_std(np.abs(CLOSE / delay_close - 1) / VOLUME, 20) / ts_mean(
        np.abs(CLOSE / delay_close - 1) / VOLUME, 20)
    result[f'alpha_077{suffix}'] = np.minimum(rank(decay_linear(((HIGH + LOW) / 2 + HIGH) - (VWAP + HIGH), 20)),
                                     rank(decay_linear(ts_corr((HIGH + LOW) / 2, ts_mean(VOLUME, 40), 3), 6)))

    tp = (HIGH + LOW + CLOSE) / 3
    ma12 = ts_mean(tp, 12)
    result[f'alpha_078{suffix}'] = (tp - ma12) / (0.015 * ts_mean(np.abs(CLOSE - ma12), 12))

    up = ts_sma(np.maximum(CLOSE - delay_close, 0), 12, 1)
    down = ts_sma(np.abs(CLOSE - delay_close), 12, 1)
    result[f'alpha_079{suffix}'] = up / down * 100
    result[f'alpha_080{suffix}'] = (VOLUME - ts_delay(VOLUME, 5)) / ts_delay(VOLUME, 5) * 100
    result[f'alpha_081{suffix}'] = ts_sma(VOLUME, 21, 2)
    result[f'alpha_082{suffix}'] = ts_sma((ts_max(HIGH, 6) - CLOSE) / (ts_max(HIGH, 6) - ts_min(LOW, 6)) * 100, 20, 1)
    result[f'alpha_083{suffix}'] = -1 * rank(ts_cov(rank_high, rank_volume, 5))
    result[f'alpha_084{suffix}'] = ts_sum(if_else(CLOSE > delay_close, VOLUME, if_else(CLOSE < delay_close, -VOLUME, 0)), 20)
    result[f'alpha_085{suffix}'] = ts_rank(VOLUME / mean_volume_20, 20) * ts_rank(-1 * ts_delta(CLOSE, 7), 8)

    term1 = (ts_delay(CLOSE, 20) - ts_delay(CLOSE, 10)) / 10
    term2 = (ts_delay(CLOSE, 10) - CLOSE) / 10
    diff = term1 - term2
    result[f'alpha_086{suffix}'] = if_else(diff > 0.25, -1, if_else(diff < 0, 1, -1 * (CLOSE - delay_close)))

    result[f'alpha_087{suffix}'] = -1 * (rank(decay_linear(ts_delta(VWAP, 4), 7)) + ts_rank(
        decay_linear((LOW * 0.9 + LOW * 0.1 - VWAP) / (OPEN - (HIGH + LOW) / 2), 11), 7))
    result[f'alpha_088{suffix}'] = (CLOSE - ts_delay(CLOSE, 20)) / ts_delay(CLOSE, 20) * 100

    sma13 = ts_sma(CLOSE, 13, 2)
    sma27 = ts_sma(CLOSE, 27, 2)
    diff = sma13 - sma27
    sma10 = ts_sma(diff, 10, 2)
    result[f'alpha_089{suffix}'] = 2 * (diff - sma10)

    result[f'alpha_090{suffix}'] = -1 * rank(ts_corr(rank_vwap, rank_volume, 5))
    result[f'alpha_091{suffix}'] = -1 * rank(CLOSE - ts_max(CLOSE, 5)) * rank(ts_corr(mean_volume_40, LOW, 5))
    result[f'alpha_092{suffix}'] = -1 * np.maximum(rank(decay_linear(ts_delta(CLOSE * 0.35 + VWAP * 0.65, 2), 3)),
                                          ts_rank(decay_linear(np.abs(ts_corr(mean_volume_180, CLOSE, 13)), 5), 15))
    result[f'alpha_093{suffix}'] = ts_sum(if_else(OPEN >= delay_open, 0, np.maximum(OPEN - LOW, OPEN - delay_open)), 20)
    result[f'alpha_094{suffix}'] = ts_sum(if_else(CLOSE > delay_close, VOLUME, if_else(CLOSE < delay_close, -VOLUME, 0)), 30)
    result[f'alpha_095{suffix}'] = ts_std(AMOUNT, 20)

    val = (CLOSE - ts_min(LOW, 9)) / (ts_max(HIGH, 9) - ts_min(LOW, 9)) * 100
    sma3 = ts_sma(val, 3, 1)
    result[f'alpha_096{suffix}'] = ts_sma(sma3, 3, 1)

    result[f'alpha_097{suffix}'] = ts_std(VOLUME, 10)
    mean100 = ts_mean(CLOSE, 100)
    delta100 = ts_delta(mean100, 100)
    ratio = delta100 / ts_delay(CLOSE, 100)
    result[f'alpha_098{suffix}'] = if_else(ratio <= 0.05, -1 * (CLOSE - ts_min(CLOSE, 100)), -1 * ts_delta(CLOSE, 3))
    result[f'alpha_099{suffix}'] = -1 * rank(ts_corr(rank_close, rank_volume, 5))
    result[f'alpha_100{suffix}'] = ts_std(VOLUME, 20)
    result[f'alpha_101{suffix}'] = -1 * (rank(ts_corr(CLOSE, ts_sum(ts_mean(VOLUME, 30), 37), 15)) <
                                rank(ts_corr(rank(HIGH * 0.1 + VWAP * 0.9), rank_volume, 11))).astype(float)
    result[f'alpha_102{suffix}'] = ts_sma(pd.Series(np.maximum(VOLUME - delay_volume, 0), index=VOLUME.index), 6, 1) / \
                          ts_sma(pd.Series(np.abs(VOLUME - delay_volume), index=VOLUME.index), 6, 1) * 100
    result[f'alpha_103{suffix}'] = ((20 - lowday(LOW, 20)) / 20) * 100
    result[f'alpha_104{suffix}'] = -1 * ts_delta(ts_corr(HIGH, VOLUME, 5), 5) * rank(ts_std(CLOSE, 20))
    result[f'alpha_105{suffix}'] = -1 * ts_corr(rank_open, rank_volume, 10)
    result[f'alpha_106{suffix}'] = CLOSE - ts_delay(CLOSE, 20)
    result[f'alpha_107{suffix}'] = rank(OPEN - delay_high) * rank(OPEN - delay_close) * rank(OPEN - delay_low)
    result[f'alpha_108{suffix}'] = -1 * rank(HIGH - ts_min(HIGH, 2)) ** rank(ts_corr(VWAP, ts_mean(VOLUME, 120), 6))

    hl = HIGH - LOW
    sma10_2 = ts_sma(hl, 10, 2)
    result[f'alpha_109{suffix}'] = sma10_2 / ts_sma(sma10_2, 10, 2)

    result[f'alpha_110{suffix}'] = ts_sum(np.maximum(0, HIGH - delay_close), 20) / ts_sum(np.maximum(0, delay_close - LOW),
                                                                                 20) * 100

    val = VOLUME * ((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)
    result[f'alpha_111{suffix}'] = ts_sma(val, 11, 2) - ts_sma(val, 4, 2)

    up = ts_sum(np.maximum(CLOSE - delay_close, 0), 12)
    down = ts_sum(np.abs(np.minimum(CLOSE - delay_close, 0)), 12)
    result[f'alpha_112{suffix}'] = (up - down) / (up + down) * 100

    result[f'alpha_113{suffix}'] = -1 * rank(ts_mean(ts_delay(CLOSE, 5), 20)) * ts_corr(CLOSE, VOLUME, 2) * rank(
        ts_corr(ts_sum(CLOSE, 5), ts_sum(CLOSE, 20), 2))
    # [A5-20260905] 除零守卫: (VWAP-CLOSE)=0 或 (HIGH-LOW)=0 时分母链为 0 → NaN
    _hl_114 = (HIGH - LOW) / ts_mean(CLOSE, 5)
    _denom_114 = (_hl_114 / (VWAP - CLOSE).replace(0, np.nan)).replace(0, np.nan)
    result[f'alpha_114{suffix}'] = rank(ts_delay(_hl_114, 2)) * rank(rank_volume) / _denom_114
    result[f'alpha_115{suffix}'] = rank(ts_corr(HIGH * 0.9 + CLOSE * 0.1, ts_mean(VOLUME, 30), 10)) ** rank(
        ts_corr(ts_rank((HIGH + LOW) / 2, 4), ts_rank(VOLUME, 10), 7))

    result[f'alpha_116{suffix}'] = CLOSE.rolling(20, min_periods=1).apply(
        lambda x: np.polyfit(np.arange(1, len(x) + 1), x, 1)[0] if len(x) >= 2 else np.nan, raw=True
    )

    result[f'alpha_117{suffix}'] = ts_rank(VOLUME, 32) * (1 - ts_rank(CLOSE + HIGH - LOW, 16)) * (1 - ts_rank(RET, 32))
    result[f'alpha_118{suffix}'] = ts_sum(HIGH - OPEN, 20) / ts_sum(OPEN - LOW, 20) * 100
    result[f'alpha_119{suffix}'] = rank(decay_linear(ts_corr(VWAP, ts_sum(ts_mean(VOLUME, 5), 26), 5), 7)) - rank(
        decay_linear(ts_rank(ts_min(ts_corr(rank_open, rank(ts_mean(VOLUME, 15)), 21), 9), 7), 8))
    result[f'alpha_120{suffix}'] = rank(VWAP - CLOSE) / rank(VWAP + CLOSE)
    result[f'alpha_121{suffix}'] = -1 * (rank(VWAP - ts_min(VWAP, 12)) ** ts_rank(
        ts_corr(ts_rank(VWAP, 20), ts_rank(mean_volume_60, 2), 18), 3))

    log_close = np.log(CLOSE)
    sma13_2 = ts_sma(log_close, 13, 2)
    sma13_2_2 = ts_sma(sma13_2, 13, 2)
    sma13_2_3 = ts_sma(sma13_2_2, 13, 2)
    delay = ts_delay(sma13_2_3, 1)
    result[f'alpha_122{suffix}'] = (sma13_2_3 - delay) / delay

    result[f'alpha_123{suffix}'] = rank(ts_corr(ts_sum((HIGH + LOW) / 2, 20), ts_sum(ts_mean(VOLUME, 60), 20), 9))
    result[f'alpha_124{suffix}'] = (CLOSE - VWAP) / decay_linear(rank(ts_max(CLOSE, 30)), 2)
    result[f'alpha_125{suffix}'] = rank(decay_linear(ts_corr(VWAP, ts_mean(VOLUME, 80), 17), 20)) / rank(
        decay_linear(ts_delta(CLOSE * 0.5 + VWAP * 0.5, 3), 16))
    result[f'alpha_126{suffix}'] = (CLOSE + HIGH + LOW) / 3

    max12 = ts_max(CLOSE, 12)
    term = 100 * (CLOSE - max12) / max12
    result[f'alpha_127{suffix}'] = np.sqrt(ts_mean(term ** 2, 12))

    tp = (HIGH + LOW + CLOSE) / 3
    up = ts_sum(if_else(tp > ts_delay(tp, 1), tp * VOLUME, 0), 14)
    down = ts_sum(if_else(tp < ts_delay(tp, 1), tp * VOLUME, 0), 14)
    result[f'alpha_128{suffix}'] = 100 - 100 / (1 + up / down)
    result[f'alpha_129{suffix}'] = ts_sum(np.abs(np.minimum(CLOSE - delay_close, 0)), 12)
    result[f'alpha_130{suffix}'] = rank(decay_linear(ts_corr((HIGH + LOW) / 2, ts_mean(VOLUME, 40), 9), 10)) / rank(
        decay_linear(ts_corr(rank_vwap, rank_volume, 7), 3))
    result[f'alpha_131{suffix}'] = ts_rank(ts_delta(VWAP, 1), 1) ** ts_rank(ts_corr(CLOSE, mean_volume_50, 18), 18)
    result[f'alpha_132{suffix}'] = ts_mean(AMOUNT, 20)
    result[f'alpha_133{suffix}'] = ((20 - highday(HIGH, 20)) / 20) * 100 - ((20 - lowday(LOW, 20)) / 20) * 100
    result[f'alpha_134{suffix}'] = (CLOSE - ts_delay(CLOSE, 12)) / ts_delay(CLOSE, 12) * VOLUME
    result[f'alpha_135{suffix}'] = -1 * ts_sma(ts_delay(CLOSE / ts_delay(CLOSE, 20), 1), 20, 1)
    result[f'alpha_136{suffix}'] = -1 * rank(ts_delta(RET, 3)) * ts_corr(OPEN, VOLUME, 10)
    result[f'alpha_137{suffix}'] = np.nan
    result[f'alpha_138{suffix}'] = -1 * (rank(decay_linear(ts_delta(LOW * 0.7 + VWAP * 0.3, 3), 20)) - ts_rank(
        decay_linear(ts_rank(ts_corr(ts_rank(LOW, 8), ts_rank(mean_volume_60, 17), 5), 19), 16), 7))
    result[f'alpha_139{suffix}'] = -1 * ts_corr(OPEN, VOLUME, 10)

    min_low_close_6 = ts_sum(np.minimum(LOW, delay_close), 6)
    max_high_close_6 = ts_sum(np.maximum(HIGH, delay_close) - np.minimum(LOW, delay_close), 6)
    part_6 = (CLOSE - min_low_close_6) / max_high_close_6 * 12 * 24
    min_low_close_12 = ts_sum(np.minimum(LOW, delay_close), 12)
    max_high_close_12 = ts_sum(np.maximum(HIGH, delay_close) - np.minimum(LOW, delay_close), 12)
    part_12 = (CLOSE - min_low_close_12) / max_high_close_12 * 6 * 24
    min_low_close_24 = ts_sum(np.minimum(LOW, delay_close), 24)
    max_high_close_24 = ts_sum(np.maximum(HIGH, delay_close) - np.minimum(LOW, delay_close), 24)
    part_24 = (CLOSE - min_low_close_24) / max_high_close_24 * 6 * 24
    result[f'alpha_140{suffix}'] = pd.Series(
        np.minimum(rank(decay_linear((rank_open + rank_low) - (rank_high + rank_close), 8)),
                   ts_rank(decay_linear(ts_corr(ts_rank(CLOSE, 8), ts_rank(mean_volume_60, 20), 8), 7), 3)),
        index=OPEN.index)
    result[f'alpha_141{suffix}'] = -1 * rank(ts_corr(rank_high, rank(ts_mean(VOLUME, 15)), 9))
    result[f'alpha_142{suffix}'] = -1 * rank(ts_rank(CLOSE, 10)) * rank(ts_delta(ts_delta(CLOSE, 1), 1)) * rank(
        ts_rank(VOLUME / mean_volume_20, 5))
    result[f'alpha_143{suffix}'] = np.nan
    result[f'alpha_144{suffix}'] = sumif(CLOSE < delay_close, np.abs(CLOSE / delay_close - 1) / AMOUNT, 20) / count(
        CLOSE < delay_close, 20)
    result[f'alpha_145{suffix}'] = (ts_mean(VOLUME, 9) - ts_mean(VOLUME, 26)) / ts_mean(VOLUME, 12) * 100
    result[f'alpha_146{suffix}'] = ts_mean((CLOSE / delay_close - 1) - ts_sma(CLOSE / delay_close - 1, 61, 2), 20) * (
            (CLOSE / delay_close - 1) - ts_sma(CLOSE / delay_close - 1, 61, 2)) / ts_sma(
        ((CLOSE / delay_close - 1) - ts_sma(CLOSE / delay_close - 1, 61, 2)) ** 2, 60, 2)

    mean12 = ts_mean(CLOSE, 12)
    result[f'alpha_147{suffix}'] = mean12.rolling(12, min_periods=1).apply(
        lambda x: np.polyfit(np.arange(1, len(x) + 1), x, 1)[0] if len(x) >= 2 else np.nan, raw=True
    )

    result[f'alpha_148{suffix}'] = -1 * (
            rank(ts_corr(OPEN, ts_sum(ts_mean(VOLUME, 60), 9), 6)) < rank(OPEN - ts_min(OPEN, 14))).astype(
        float)
    result[f'alpha_149{suffix}'] = np.nan
    result[f'alpha_150{suffix}'] = (CLOSE + HIGH + LOW) / 3 * VOLUME
    result[f'alpha_151{suffix}'] = ts_sma(CLOSE - ts_delay(CLOSE, 20), 20, 1)
    result[f'alpha_152{suffix}'] = ts_sma(ts_mean(ts_sma(ts_delay(CLOSE / ts_delay(CLOSE, 9), 1), 9, 1), 12) - ts_mean(
        ts_sma(ts_delay(CLOSE / ts_delay(CLOSE, 9), 1), 9, 1), 26), 9, 1)
    result[f'alpha_153{suffix}'] = (ts_mean(CLOSE, 3) + mean_close_6 + mean_close_12 + ts_mean(CLOSE, 24)) / 4
    result[f'alpha_154{suffix}'] = (VWAP - ts_min(VWAP, 16) < ts_corr(VWAP, ts_mean(VOLUME, 180), 18)).astype(float)
    result[f'alpha_155{suffix}'] = ts_sma(ts_sma(VOLUME, 13, 2) - ts_sma(VOLUME, 27, 2), 10, 2) - (
            ts_sma(VOLUME, 13, 2) - ts_sma(VOLUME, 27, 2))
    result[f'alpha_156{suffix}'] = -1 * np.maximum(rank(decay_linear(ts_delta(VWAP, 5), 3)), rank(
        decay_linear(-1 * ts_delta(OPEN * 0.15 + LOW * 0.85, 2) / (OPEN * 0.15 + LOW * 0.85), 3)))
    result[f'alpha_157{suffix}'] = ts_min(
        rank(rank(np.log(ts_sum(ts_min(rank(rank(-1 * rank(ts_delta(CLOSE - 1, 5)))), 2), 1)))), 5) + ts_rank(
        ts_delay(-1 * RET, 6), 5)
    result[f'alpha_158{suffix}'] = ((HIGH - ts_sma(CLOSE, 15, 2)) - (LOW - ts_sma(CLOSE, 15, 2))) / CLOSE
    result[f'alpha_159{suffix}'] = (part_6 + part_12 + part_24) * 100 / (6 * 12 + 6 * 24 + 12 * 24)
    result[f'alpha_160{suffix}'] = ts_sma(if_else(CLOSE <= delay_close, std_close_20, 0), 20, 1)

    up = ts_sma(np.maximum(CLOSE - delay_close, 0), 12, 1)
    down = ts_sma(np.abs(CLOSE - delay_close), 12, 1)
    rsi = up / down * 100
    result[f'alpha_162{suffix}'] = (rsi - ts_min(rsi, 12)) / (ts_max(rsi, 12) - ts_min(rsi, 12))

    result[f'alpha_163{suffix}'] = rank(-1 * RET * ts_mean(VOLUME, 20) * VWAP * (HIGH - CLOSE))

    cond = CLOSE > delay_close
    inv = 1 / (CLOSE - delay_close)
    val = if_else(cond, inv, 1)
    result[f'alpha_164{suffix}'] = ts_sma((val - ts_min(val, 12)) / (HIGH - LOW) * 100, 13, 2)

    mean48 = ts_mean(CLOSE, 48)
    cum = sumac(CLOSE - mean48)
    result[f'alpha_165{suffix}'] = (ts_max(cum, 48) - ts_min(cum, 48)) / ts_std(CLOSE, 48)

    ret1 = CLOSE / delay_close - 1
    mean20 = ts_mean(ret1, 20)
    diff = ret1 - mean20
    sum20 = ts_sum(diff, 20)
    sum_sq20 = ts_sum(ret1 ** 2, 20)
    result[f'alpha_166{suffix}'] = (-20 * (20 - 1) ** 1.5 * sum20) / ((20 - 1) * (20 - 2) * (sum_sq20 ** 1.5))
    result[f'alpha_167{suffix}'] = ts_sum(
        pd.Series(np.where(CLOSE - delay_close > 0, CLOSE - delay_close, 0), index=CLOSE.index), 12)
    result[f'alpha_168{suffix}'] = -1 * VOLUME / mean_volume_20

    delay1 = ts_delay(ts_sma(CLOSE - delay_close, 9, 1), 1)
    result[f'alpha_169{suffix}'] = ts_sma(ts_mean(delay1, 12) - ts_mean(delay1, 26), 10, 1)
    result[f'alpha_170{suffix}'] = rank(1 / CLOSE) * VOLUME / mean_volume_20 * (HIGH * rank(HIGH - CLOSE)) / (
            ts_sum(HIGH, 5) / 5) - rank(VWAP - ts_delay(VWAP, 5))
    # [A5-20260905] 除零守卫: 一字板 CLOSE==HIGH 时分母为 0 → NaN (原产生 inf, 实证 20220801 单日 210 例)
    _denom_171 = (CLOSE - HIGH) * (CLOSE ** 5)
    result[f'alpha_171{suffix}'] = np.where(
        np.abs(_denom_171) < 1e-12, np.nan,
        (-1 * (LOW - CLOSE) * (OPEN ** 5)) / _denom_171.replace(0, np.nan))

    tr = np.maximum(np.maximum(HIGH - LOW, np.abs(delay_close - HIGH)), np.abs(delay_close - LOW))
    hd = HIGH - delay_high
    ld = delay_low - LOW
    sum_tr14 = ts_sum(tr, 14)
    sum_ld_gt = ts_sum(if_else((ld > 0) & (ld > hd), ld, 0), 14)
    sum_hd_gt = ts_sum(if_else((hd > 0) & (hd > ld), hd, 0), 14)
    p1 = sum_ld_gt * 100 / sum_tr14
    p2 = sum_hd_gt * 100 / sum_tr14
    val = np.abs(p1 - p2) / (p1 + p2) * 100
    result[f'alpha_172{suffix}'] = ts_mean(val, 6)

    log_close = np.log(CLOSE)
    sma13_2 = ts_sma(CLOSE, 13, 2)
    sma13_2_2 = ts_sma(sma13_2, 13, 2)
    sma13_2_log = ts_sma(log_close, 13, 2)
    sma13_2_log_2 = ts_sma(sma13_2_log, 13, 2)
    sma13_2_log_3 = ts_sma(sma13_2_log_2, 13, 2)
    result[f'alpha_173{suffix}'] = 3 * sma13_2 - 2 * sma13_2_2 + sma13_2_log_3

    result[f'alpha_174{suffix}'] = ts_sma(if_else(CLOSE > delay_close, std_close_20, 0), 20, 1)
    result[f'alpha_175{suffix}'] = ts_mean(
        np.maximum(np.maximum(HIGH - LOW, np.abs(delay_close - HIGH)), np.abs(delay_close - LOW)), 6)
    result[f'alpha_176{suffix}'] = ts_corr(rank((CLOSE - ts_min(LOW, 12)) / (ts_max(HIGH, 12) - ts_min(LOW, 12))),
                                  rank_volume, 6)
    result[f'alpha_177{suffix}'] = ((20 - highday(HIGH, 20)) / 20) * 100
    result[f'alpha_178{suffix}'] = (CLOSE - delay_close) / delay_close * VOLUME
    result[f'alpha_179{suffix}'] = rank(ts_corr(VWAP, VOLUME, 4)) * rank(ts_corr(rank_low, rank(mean_volume_50), 12))
    result[f'alpha_180{suffix}'] = if_else(mean_volume_20 < VOLUME,
                                  -1 * ts_rank(np.abs(ts_delta(CLOSE, 7)), 60) * np.sign(ts_delta(CLOSE, 7)),
                                  -1 * VOLUME)
    result[f'alpha_181{suffix}'] = np.nan if INDEX_CLOSE.isna().all() else ts_sum(((CLOSE / delay_close - 1) - ts_mean(
        CLOSE / delay_close - 1, 20) - ((INDEX_CLOSE / ts_delay(INDEX_CLOSE, 1) - 1) - ts_mean(
        INDEX_CLOSE / ts_delay(INDEX_CLOSE, 1) - 1, 20))) ** 2, 20) / ts_sum(((CLOSE / delay_close - 1) - ts_mean(
        CLOSE / delay_close - 1, 20)) ** 3, 20)
    result[f'alpha_182{suffix}'] = np.nan if INDEX_CLOSE.isna().all() else count(
        (CLOSE > OPEN) & (INDEX_CLOSE > INDEX_OPEN) | (CLOSE < OPEN) & (INDEX_CLOSE < INDEX_OPEN), 20) / 20

    mean24 = ts_mean(CLOSE, 24)
    cum = sumac(CLOSE - mean24)
    result[f'alpha_183{suffix}'] = (ts_max(cum, 24) - ts_min(cum, 24)) / ts_std(CLOSE, 24)

    result[f'alpha_184{suffix}'] = rank(ts_corr(ts_delay(OPEN - CLOSE, 1), CLOSE, 200)) + rank(OPEN - CLOSE)
    result[f'alpha_185{suffix}'] = rank(-1 * (1 - (OPEN / CLOSE)) ** 2)

    tr = np.maximum(np.maximum(HIGH - LOW, np.abs(delay_close - HIGH)), np.abs(delay_close - LOW))
    hd = HIGH - delay_high
    ld = delay_low - LOW
    sum_tr14 = ts_sum(tr, 14)
    sum_ld_gt = ts_sum(if_else((ld > 0) & (ld > hd), ld, 0), 14)
    sum_hd_gt = ts_sum(if_else((hd > 0) & (hd > ld), hd, 0), 14)
    p1 = sum_ld_gt * 100 / sum_tr14
    p2 = sum_hd_gt * 100 / sum_tr14
    val = np.abs(p1 - p2) / (p1 + p2) * 100
    mean6 = ts_mean(val, 6)
    result[f'alpha_186{suffix}'] = (mean6 + ts_delay(mean6, 6)) / 2

    result[f'alpha_187{suffix}'] = ts_sum(if_else(OPEN <= delay_open, 0, np.maximum(HIGH - OPEN, OPEN - delay_open)), 20)
    result[f'alpha_188{suffix}'] = ((HIGH - LOW) - ts_sma(HIGH - LOW, 11, 2)) / ts_sma(HIGH - LOW, 11, 2) * 100
    result[f'alpha_189{suffix}'] = ts_mean(np.abs(CLOSE - mean_close_6), 6)
    result[f'alpha_190{suffix}'] = np.nan
    result[f'alpha_191{suffix}'] = ts_corr(mean_volume_20, LOW, 5) + (HIGH + LOW) / 2 - CLOSE

    # 最后，构建 DataFrame
    # 将 result 字典转为 DataFrame，并确保索引正确
    result_df = pd.DataFrame(result, index=CLOSE.index)
    return result_df

def _compute_ffd_selected(OPEN, HIGH, LOW, CLOSE, VOLUME, VWAP, AMOUNT, RET,
                          INDEX_CLOSE, INDEX_OPEN):
    """
    仅计算用户筛选出的 21 个有效 FFD 因子，以大幅提升计算速度。
    """
    import numpy as np
    import pandas as pd

    # ---- 内部辅助函数 ----
    def if_else(cond, a, b):
        return pd.Series(np.where(cond, a, b), index=cond.index)

    # ---- 预计算常用中间变量 ----
    delay_close = ts_delay(CLOSE, 1)
    delay_open = ts_delay(OPEN, 1)
    delay_high = ts_delay(HIGH, 1)
    delay_low = ts_delay(LOW, 1)
    delay_volume = ts_delay(VOLUME, 1)

    rank_high = rank(HIGH)
    rank_low = rank(LOW)
    rank_close = rank(CLOSE)
    rank_open = rank(OPEN)
    rank_volume = rank(VOLUME)
    rank_vwap = rank(VWAP)

    mean_volume_20 = ts_mean(VOLUME, 20)
    mean_volume_40 = ts_mean(VOLUME, 40)
    mean_volume_50 = ts_mean(VOLUME, 50)
    mean_volume_60 = ts_mean(VOLUME, 60)
    mean_volume_80 = ts_mean(VOLUME, 80)
    mean_volume_150 = ts_mean(VOLUME, 150)
    mean_volume_180 = ts_mean(VOLUME, 180)

    mean_close_6 = ts_mean(CLOSE, 6)
    mean_close_7 = ts_mean(CLOSE, 7)
    mean_close_12 = ts_mean(CLOSE, 12)
    mean_close_20 = ts_mean(CLOSE, 20)
    mean_close_24 = ts_mean(CLOSE, 24)

    std_close_20 = ts_std(CLOSE, 20)

    # ---- 提前定义 dtm 和 dbm（alpha_069 需要） ----
    dtm = if_else(OPEN <= delay_open, 0, np.maximum(HIGH - OPEN, OPEN - delay_open))
    dbm = if_else(OPEN >= delay_open, 0, np.maximum(OPEN - LOW, delay_open - OPEN))

    result = {}

    # ---- 白名单因子（20 个） ----
    result['alpha_006_FFD'] = -1 * rank(ts_delta(OPEN * 0.85 + HIGH * 0.15, 4))
    result['alpha_008_FFD'] = -1 * rank(ts_delta((HIGH + LOW) / 2 * 0.2 + VWAP * 0.8, 4))
    result['alpha_013_FFD'] = np.sqrt(HIGH * LOW) - VWAP
    result['alpha_037_FFD'] = -1 * rank(
        ts_sum(OPEN, 5) * ts_sum(RET, 5) - ts_delay(ts_sum(OPEN, 5) * ts_sum(RET, 5), 10)
    )
    result['alpha_038_FFD'] = if_else(ts_mean(HIGH, 20) < HIGH, -1 * ts_delta(HIGH, 2), 0)
    result['alpha_041_FFD'] = -1 * rank(ts_max(ts_delta(VWAP, 3), 5))
    result['alpha_049_FFD'] = ts_sum(if_else((HIGH + LOW) >= (delay_high + delay_low), 0,
                                             np.maximum(np.abs(HIGH - delay_high), np.abs(LOW - delay_low))), 12) / \
                              ts_sum(np.maximum(np.abs(HIGH - delay_high), np.abs(LOW - delay_low)), 12)
    result['alpha_069_FFD'] = if_else(ts_sum(dtm, 20) > ts_sum(dbm, 20),
                                      (ts_sum(dtm, 20) - ts_sum(dbm, 20)) / ts_sum(dtm, 20),
                                      if_else(ts_sum(dtm, 20) == ts_sum(dbm, 20), 0,
                                              (ts_sum(dtm, 20) - ts_sum(dbm, 20)) / ts_sum(dbm, 20)))
    result['alpha_070_FFD'] = ts_std(AMOUNT, 6)
    result['alpha_087_FFD'] = -1 * (rank(decay_linear(ts_delta(VWAP, 4), 7)) + ts_rank(
        decay_linear((LOW * 0.9 + LOW * 0.1 - VWAP) / (OPEN - (HIGH + LOW) / 2), 11), 7))
    result['alpha_093_FFD'] = ts_sum(if_else(OPEN >= delay_open, 0,
                                             np.maximum(OPEN - LOW, OPEN - delay_open)), 20)
    result['alpha_095_FFD'] = ts_std(AMOUNT, 20)
    result['alpha_103_FFD'] = ((20 - lowday(LOW, 20)) / 20) * 100
    result['alpha_109_FFD'] = ts_sma(HIGH - LOW, 10, 2) / ts_sma(ts_sma(HIGH - LOW, 10, 2), 10, 2)
    result['alpha_118_FFD'] = ts_sum(HIGH - OPEN, 20) / ts_sum(OPEN - LOW, 20) * 100
    result['alpha_132_FFD'] = ts_mean(AMOUNT, 20)
    result['alpha_133_FFD'] = ((20 - highday(HIGH, 20)) / 20) * 100 - ((20 - lowday(LOW, 20)) / 20) * 100
    result['alpha_177_FFD'] = ((20 - highday(HIGH, 20)) / 20) * 100
    result['alpha_187_FFD'] = ts_sum(if_else(OPEN <= delay_open, 0,
                                             np.maximum(HIGH - OPEN, OPEN - delay_open)), 20)
    result['alpha_188_FFD'] = ((HIGH - LOW) - ts_sma(HIGH - LOW, 11, 2)) / ts_sma(HIGH - LOW, 11, 2) * 100

    result_df = pd.DataFrame(result, index=CLOSE.index)
    return result_df

def _setup_rank_ctx(rank_ctx):
    """[A1] worker 进程入口: 按任务上下文设置 rank 模式"""
    global _RANK_MODE, _RANK_REGISTRY
    mode = (rank_ctx or {}).get('mode', 'timeseries')
    _RANK_MODE = mode
    if mode == 'collect':
        _RANK_REGISTRY = {}
    elif mode == 'replay':
        _rank_load_panel(rank_ctx['panel_dir'])


def calculate_factors_for_stock_single(args):
    # [A1] args 兼容: (code, data_dict) 或 (code, data_dict, rank_ctx)
    if len(args) == 3:
        code, stock_data_dict, rank_ctx = args
        _setup_rank_ctx(rank_ctx)
    else:
        code, stock_data_dict = args
    # 数据准备与之前完全相同
    df = pd.DataFrame(stock_data_dict)
    if len(df) < 10:
        return None
    df = df.sort_values('trade_date').reset_index(drop=True)

    OPEN = df['open']
    HIGH = df['high']
    LOW = df['low']
    CLOSE = df['close']
    VOLUME = df['volume']
    VWAP = df['vwap'] if 'vwap' in df.columns else df['amount'] / (df['volume'] + 1e-10)
    AMOUNT = df['amount']
    RET = df['close'].pct_change()
    INDEX_CLOSE = df.get('index_close', pd.Series(np.nan, index=df.index))
    INDEX_OPEN = df.get('index_open', pd.Series(np.nan, index=df.index))

    # [A1] 设置 rank 上下文 (计数器重置, suffix 命名空间)
    if _RANK_MODE != 'timeseries':
        _rank_begin_stock(code, df['trade_date'].values.astype('datetime64[ns]'), suffix='')

    # 调用核心计算函数，后缀为空（原始因子）
    result = _compute_all_alphas(
        OPEN, HIGH, LOW, CLOSE, VOLUME, VWAP, AMOUNT, RET,
        INDEX_CLOSE, INDEX_OPEN, suffix=''
    )
    # 添加基础信息列
    result.insert(0, 'ts_code', code)
    result.insert(1, 'trade_date', df['trade_date'].values)
    # 保留原始价格列（可选，用于快照或日后分析）
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'vwap', 'tradestatus', 'isST']:
        if col in df.columns:
            result[col] = df[col].values
    # [A1] collect 模式附带 registry 块返回
    if _RANK_MODE == 'collect':
        return result, _RANK_REGISTRY
    return result

def calculate_ffd_factors_for_stock_single(args):
    # [A1] args 兼容: (code, data_dict, d_star) 或 (code, data_dict, d_star, rank_ctx)
    if len(args) == 4:
        code, stock_data_dict, d_star, rank_ctx = args
        _setup_rank_ctx(rank_ctx)
    else:
        code, stock_data_dict, d_star = args   # d_star 为全局统一中位数
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(stock_data_dict)
    if len(df) < 10:
        return None
    df = df.sort_values('trade_date').reset_index(drop=True)

    OPEN = df['open']
    HIGH = df['high']
    LOW = df['low']
    CLOSE = df['close']
    VOLUME = df['volume']
    VWAP = df['vwap'] if 'vwap' in df.columns else df['amount'] / (df['volume'] + 1e-10)
    AMOUNT = df['amount']
    RET = df['close'].pct_change()
    INDEX_CLOSE = df.get('index_close', pd.Series(np.nan, index=df.index))
    INDEX_OPEN = df.get('index_open', pd.Series(np.nan, index=df.index))

    # 对 close 和 volume 进行 FFD 变换
    if d_star > 0:
        CLOSE_FFD = fracDiff_FFD(CLOSE, d_star)
        VOLUME_FFD = fracDiff_FFD(VOLUME, d_star)
    else:
        CLOSE_FFD = CLOSE.copy()
        VOLUME_FFD = VOLUME.copy()

    # [A1] 设置 rank 上下文 (FFD 命名空间 '_FFD')
    if _RANK_MODE != 'timeseries':
        _rank_begin_stock(code, df['trade_date'].values.astype('datetime64[ns]'), suffix='_FFD')

    # 调用核心计算函数，后缀为 '_FFD'
    result = _compute_ffd_selected(
        OPEN, HIGH, LOW, CLOSE_FFD, VOLUME_FFD, VWAP, AMOUNT, RET,
        INDEX_CLOSE, INDEX_OPEN
    )
    result.insert(0, 'ts_code', code)
    result.insert(1, 'trade_date', df['trade_date'].values)
    # 同时保存原始 close，便于比对
    result['close'] = CLOSE.values
    result['volume'] = VOLUME.values
    # [A1] collect 模式附带 registry 块返回
    if _RANK_MODE == 'collect':
        return result, _RANK_REGISTRY
    return result

# ==================== Alpha191 因子类（向后兼容）====================

class Alpha191Calculator:
    """Alpha191 因子计算器（完整版）"""

    FACTOR_NAMES = [f'alpha_{i:03d}' for i in range(1, 293)]

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._validate_columns()

    def _validate_columns(self):
        """验证必要的列是否存在，并补充计算列"""
        required_cols = ['ts_code', 'trade_date', 'open', 'high',
                         'low', 'close', 'volume', 'amount']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"缺少必要的列: {missing}")

        self.df['trade_date'] = pd.to_datetime(self.df['trade_date'])

        # 若没有 vwap 列，则根据 amount 和 volume 估算
        if 'vwap' not in self.df.columns:
            self.df['vwap'] = self.df['amount'] / (self.df['volume'] + 1e-10)

        # 收益率
        self.df['ret'] = self.df['close'].pct_change()

        # 指数相关列（若无则置 NaN）
        if 'index_close' not in self.df.columns:
            self.df['index_close'] = np.nan
        if 'index_open' not in self.df.columns:
            self.df['index_open'] = np.nan

        self.has_ff3 = all(col in self.df.columns for col in ['MKT', 'SMB', 'HML'])

    def calculate_all_factors(self) -> pd.DataFrame:
        """计算所有 191 个因子（使用多进程并行）"""
        # 将数据转换为字典列表以便多进程处理
        stock_data_list = []
        for code, group in self.df.groupby('ts_code'):
            stock_data_list.append((code, group.to_dict('list')))

        results = []
        # 使用多进程并行计算，进程数改为 10
        num_workers = 10

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(calculate_factors_for_stock_single, args): args[0]
                       for args in stock_data_list}

            for future in as_completed(futures):
                code = futures[future]
                try:
                    result = future.result()
                    if result is not None and not result.empty:
                        results.append(result)
                except Exception as e:
                    logger.error(f"处理股票 {code} 时出错: {e}")

        if not results:
            raise ValueError("没有成功计算任何因子")

        final_result = pd.concat(results, ignore_index=True)
        return final_result


# ==================== 优化的批量计算函数 ====================
def calculate_alpha_factors(df_daily: pd.DataFrame, local_manager=None, batch_size: int = 500,
                            use_parallel: bool = True, max_workers: int = None,
                            calculate_factors: str = 'on',
                            incremental: bool = True,
                            reference_date: Optional[datetime] = None,
                            save_to_local: bool = True,
                            enable_ffd: bool = True,
                            ffd_only: bool = False,      # 新增
                            data_base_path: str = './data',
                            rank_mode: str = 'timeseries',   # [A1-20260905] 'timeseries'(默认,旧行为) | 'cross_section'(两遍法截面rank)
                            rank_work_dir: str = None) -> pd.DataFrame:  # [A1] 面板工作目录(默认系统临时目录)
    # ... 文档字符串根据情况可保留，参数说明添加 ffd_only
    if df_daily is None or df_daily.empty:
        raise ValueError("输入数据为空")
    df_daily['trade_date'] = pd.to_datetime(df_daily['trade_date'])
    start_date = df_daily['trade_date'].min()
    end_date = df_daily['trade_date'].max()
    logger.info(f"【{'FFD增量' if ffd_only else '全量重算'}模式】请求因子数据范围: {start_date.date()} 至 {end_date.date()}")
    if reference_date is None:
        reference_date = end_date
    logger.info(f"前复权基准日: {reference_date.date()}")

    # 读取 FFD 统一中位数
    d_star_median = 0.35
    effective_ffd = enable_ffd or ffd_only
    if effective_ffd:
        base_path = local_manager.base_path if local_manager else data_base_path
        median_file = os.path.join(base_path, 'ffd_params', 'd_star_median.txt')
        if os.path.exists(median_file):
            try:
                with open(median_file, 'r') as f:
                    d_star_median = float(f.read().strip())
                logger.info(f"已读取全局 FFD 中位数 d* = {d_star_median:.4f}")
            except Exception as e:
                # [A9-20260905] warning → error: 读取失败静默回退 0.35 会造成跨期 FFD 口径不一致
                logger.error(f"[A9] 读取 d_star_median.txt 失败: {e}，使用默认 0.35 (跨期口径风险!)")
        else:
            # [A9-20260905] warning → error (文件缺失告警已存在, 级别不足)
            logger.error("[A9] d_star_median.txt 未找到，使用默认 d=0.35 (跨期口径风险!)")

    required_cols = ['open', 'high', 'low', 'close', 'volume', 'amount',
                     'vwap', 'index_close', 'index_open', 'trade_date', 'ts_code',
                     'tradestatus', 'isST']   # [A3-20260905] 保留交易状态列供快照停牌过滤
    avail = [c for c in required_cols if c in df_daily.columns]

    def _build_tasks(rank_ctx):
        tasks = []
        for code, group in df_daily.groupby('ts_code'):
            stock_dict = group[avail].to_dict('list')
            if not ffd_only:
                tasks.append(('orig', code, stock_dict, rank_ctx))
            if effective_ffd:
                tasks.append(('ffd', code, stock_dict, d_star_median, rank_ctx))
        return tasks

    if rank_mode == 'cross_section':
        # ============ [A1] 两遍法: collect → 建面板 → replay ============
        import tempfile
        _own_work_dir = rank_work_dir is None          # [A12] 仅清理自建临时目录
        work_dir = rank_work_dir or tempfile.mkdtemp(prefix='cs_rank_')
        logger.info(f"[A1] 横截面 rank 两遍法启动, 工作目录: {work_dir}")
        try:
            logger.info("[A1] Pass 1/2: collect (记录 rank 参数序列)...")
            registry = {}
            _ = _execute_tasks(_build_tasks({'mode': 'collect'}), use_parallel, max_workers,
                               collect_registry=registry)

            logger.info(f"[A1] 构建截面 rank 面板 ({len(registry)} 个 rank 调用点)...")
            panel_dir = build_cs_rank_panels(registry, df_daily, work_dir)
            del registry
            gc.collect()

            logger.info("[A1] Pass 2/2: replay (应用真实截面 rank)...")
            df_factors = _execute_tasks(_build_tasks({'mode': 'replay', 'panel_dir': panel_dir}),
                                        use_parallel, max_workers)
        finally:
            # [A12-20260905] 用完即清: 面板 memmap ~600MB/日, 长期全量再生会堆积数十 GB
            # (此时所有 replay worker 已 join, memmap 无读者; 外部传入的 rank_work_dir 不动)
            if _own_work_dir:
                try:
                    import shutil
                    _rank_release_panel()      # Windows: 先释放本进程 memmap 句柄再删目录
                    gc.collect()
                    shutil.rmtree(work_dir, ignore_errors=True)
                    logger.info(f"[A12] 已清理临时 rank 面板目录: {work_dir}")
                except Exception as _e:
                    logger.warning(f"[A12] 清理 {work_dir} 失败(可手动删除): {_e}")
    else:
        df_factors = _execute_tasks(_build_tasks(None), use_parallel, max_workers)

    if df_factors.empty:
        raise ValueError("因子计算失败，返回空数据")

    # ffd_only 模式下，我们返回的数据只包含 _FFD 列和基础列，不需要再保存（由调用方自行处理）
    if not ffd_only and save_to_local and local_manager is not None:
        logger.info("覆盖保存因子到本地...")
        for code, group in df_factors.groupby('ts_code'):
            local_manager.save_stock_factors(code, group, reference_date=reference_date)
        logger.info(f"因子已保存，涉及 {df_factors['ts_code'].nunique()} 只股票")
    elif not save_to_local:
        logger.info("根据配置跳过本地保存，仅内存缓存")

    logger.info(f"因子数据准备完成: {len(df_factors)}条记录，{df_factors['ts_code'].nunique()}只股票")
    return df_factors

def _execute_tasks(tasks, use_parallel, max_workers, collect_registry=None):
    """[A1-20260905] 统一执行入口: tasks 为 ('orig', code, dict, rank_ctx) / ('ffd', code, dict, d, rank_ctx)。
    collect_registry 非 None 时, 合并各 worker 返回的 (df, registry_chunk) 中的 chunk。"""
    logger.info(f"总任务数: {len(tasks)}")
    if use_parallel:
        return _calculate_factors_parallel(tasks, max_workers=max_workers,
                                           collect_registry=collect_registry)
    results = []
    for task in tasks:
        if task[0] == 'orig':
            _, code, data_dict, rank_ctx = task
            res = calculate_factors_for_stock_single((code, data_dict, rank_ctx))
        else:
            _, code, data_dict, d, rank_ctx = task
            res = calculate_ffd_factors_for_stock_single((code, data_dict, d, rank_ctx))
        if collect_registry is not None and isinstance(res, tuple):
            res, chunk = res
            for k, v in chunk.items():
                collect_registry.setdefault(k, {}).update(v)
        if res is not None and not res.empty:
            results.append(res)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def _calculate_factors_parallel(tasks: list, max_workers: int = None, collect_registry=None) -> pd.DataFrame:
    """
    使用多进程并行计算所有任务（包括原始因子和 FFD 因子）。
    tasks 元素格式：
        ('orig', code, data_dict, rank_ctx)  或
        ('ffd',  code, data_dict, d_star, rank_ctx)
    [A1-20260905] rank_ctx 随任务传入 worker; collect_registry 非 None 时合并 registry 块。
    """
    if max_workers is None:
        max_workers = 10

    logger.info(f"使用 {max_workers} 个进程并行计算因子...")

    # 任务映射：futures -> task index (用于排序)，且记录股票代码
    results_dict = {}
    future_to_task = {}          # 用于追踪进度
    task_idx = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for task in tasks:
            if task[0] == 'orig':
                _, code, data_dict, rank_ctx = task
                fut = executor.submit(calculate_factors_for_stock_single, (code, data_dict, rank_ctx))
            else:
                _, code, data_dict, d_star, rank_ctx = task
                fut = executor.submit(calculate_ffd_factors_for_stock_single, (code, data_dict, d_star, rank_ctx))
            future_to_task[fut] = (code, task_idx)
            task_idx += 1

        completed = 0
        total = len(tasks)
        for future in as_completed(future_to_task):
            code, idx = future_to_task[future]
            completed += 1
            try:
                result = future.result()
                # [A1] collect 模式: 返回 (df, registry_chunk)
                if collect_registry is not None and isinstance(result, tuple):
                    result, chunk = result
                    for k, v in chunk.items():
                        collect_registry.setdefault(k, {}).update(v)
                if result is not None and not result.empty:
                    results_dict[idx] = result
                    if completed % 200 == 0:
                        logger.info(f"  进度: {completed}/{total} ({completed / total * 100:.1f}%)")
                else:
                    logger.debug(f"股票 {code} 计算返回空结果，跳过")
            except Exception as e:
                logger.error(f"处理股票 {code} 时出错: {e}")

    if not results_dict:
        logger.warning("所有任务均未成功完成")
        return pd.DataFrame()

    logger.info(f"合并 {len(results_dict)} 个结果...")
    sorted_results = [results_dict[i] for i in sorted(results_dict.keys())]

    # 分批合并（复用原函数）
    final_result = _merge_results_in_batches(sorted_results, batch_size=500)

    # 转换因子列为 float32
    for col in final_result.columns:
        if col.startswith('alpha_'):
            final_result[col] = final_result[col].astype(np.float32)

    gc.collect()
    return final_result

def _merge_results_in_batches(results: List[pd.DataFrame], batch_size: int = 500) -> pd.DataFrame:
    """
    分批合并结果DataFrame列表，避免内存溢出
    """
    if not results:
        return pd.DataFrame()

    if len(results) <= batch_size:
        return pd.concat(results, ignore_index=True)

    # 分批合并
    merged_batches = []
    for i in range(0, len(results), batch_size):
        batch = results[i:i + batch_size]
        merged = pd.concat(batch, ignore_index=True)
        merged_batches.append(merged)
        logger.info(f"  合并批次 {i // batch_size + 1}/{(len(results) - 1) // batch_size + 1}: {len(merged)}条")

    # 递归合并
    if len(merged_batches) > 1:
        return _merge_results_in_batches(merged_batches, batch_size)
    else:
        return merged_batches[0]




# ==================== [PERF-20260905] rank-only 快速路径函数 (自动生成, 勿手改) ====================
def _compute_rank_alphas(OPEN, HIGH, LOW, CLOSE, VOLUME, VWAP, AMOUNT, RET,
                        INDEX_CLOSE, INDEX_OPEN, suffix=''):
    """[PERF-20260905] 由 _compute_all_alphas 自动抽取: 仅含 rank 传导闭包的 62 列 + 全部前导/中间语句 (原顺序)。
    rank() 调用序列与原函数一致 ⇒ collect/replay 的 rank_key 天然对齐。
    等价性由 U1 单测逐位验证。"""
    import numpy as np

    import pandas as pd


    # 内部辅助函数

    def if_else(cond, a, b):
        return pd.Series(np.where(cond, a, b), index=cond.index)


    # 预计算常用中间变量（与原版完全一致）

    delay_close = ts_delay(CLOSE, 1)

    delay_open = ts_delay(OPEN, 1)

    delay_high = ts_delay(HIGH, 1)

    delay_low = ts_delay(LOW, 1)

    delay_volume = ts_delay(VOLUME, 1)


    rank_high = rank(HIGH)

    rank_low = rank(LOW)

    rank_close = rank(CLOSE)

    rank_open = rank(OPEN)

    rank_volume = rank(VOLUME)

    rank_vwap = rank(VWAP)


    mean_volume_20 = ts_mean(VOLUME, 20)

    mean_volume_40 = ts_mean(VOLUME, 40)

    mean_volume_50 = ts_mean(VOLUME, 50)

    mean_volume_60 = ts_mean(VOLUME, 60)

    mean_volume_80 = ts_mean(VOLUME, 80)

    mean_volume_150 = ts_mean(VOLUME, 150)

    mean_volume_180 = ts_mean(VOLUME, 180)


    mean_close_6 = ts_mean(CLOSE, 6)

    mean_close_7 = ts_mean(CLOSE, 7)

    mean_close_12 = ts_mean(CLOSE, 12)

    mean_close_20 = ts_mean(CLOSE, 20)

    mean_close_24 = ts_mean(CLOSE, 24)


    std_close_20 = ts_std(CLOSE, 20)


    result = {}


    # ========== 批量计算所有因子（完全复制原代码，表头已替换） ==========

    # 以下为完整的 191 个因子，每个 result['alpha_XXX'] 已替换为 result[f'alpha_XXX{suffix}']

    result[f'alpha_001{suffix}'] = -1 * ts_corr(rank(ts_delta(np.log(VOLUME), 1)), rank((CLOSE - OPEN) / OPEN), 6)

    cond_eq = (CLOSE == delay_close)

    cond_gt = (CLOSE > delay_close)

    part1 = CLOSE - np.minimum(LOW, delay_close)

    part2 = CLOSE - np.maximum(HIGH, delay_close)

    val = if_else(cond_gt, part1, part2)

    val = if_else(cond_eq, 0, val)

    ma8 = ts_mean(CLOSE, 8)

    std8 = ts_std(CLOSE, 8)

    ma2 = ts_mean(CLOSE, 2)

    cond1 = (ma8 + std8) < ma2

    cond2 = ma2 < (ma8 - std8)

    cond3 = (VOLUME / mean_volume_20) >= 1

    result[f'alpha_006{suffix}'] = -1 * rank(ts_delta(OPEN * 0.85 + HIGH * 0.15, 4))

    result[f'alpha_007{suffix}'] = (rank(ts_max(VWAP - CLOSE, 3)) + rank(ts_min(VWAP - CLOSE, 3))) * rank(
        ts_delta(VOLUME, 3))

    result[f'alpha_008{suffix}'] = -1 * rank(ts_delta((HIGH + LOW) / 2 * 0.2 + VWAP * 0.8, 4))

    result[f'alpha_010{suffix}'] = rank(
        ts_max(pd.Series(np.where(RET < 0, ts_std(RET, 20), CLOSE) ** 2, index=CLOSE.index), 5))

    result[f'alpha_012{suffix}'] = rank(OPEN - ts_mean(VWAP, 10)) * (-1 * rank(np.abs(CLOSE - VWAP)))

    result[f'alpha_016{suffix}']= -1 * ts_max(rank(ts_corr(rank_volume, rank_vwap, 5)), 5)

    result[f'alpha_017{suffix}'] = rank(VWAP - ts_max(VWAP, 15)) ** ts_delta(CLOSE, 5)

    cond_lt = (CLOSE < ts_delay(CLOSE, 5))

    cond_eq = (CLOSE == ts_delay(CLOSE, 5))

    part1 = (CLOSE - ts_delay(CLOSE, 5)) / ts_delay(CLOSE, 5)

    part2 = (CLOSE - ts_delay(CLOSE, 5)) / CLOSE

    # 使用 Numba 加速的线性回归斜率

    if NUMBA_AVAILABLE:
        vals = mean_close_6.values
        n = len(vals)
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            if i >= 5:
                win = vals[i - 5:i + 1]
                valid = win[~np.isnan(win)]
                if len(valid) >= 2:  # 至少2个点才能拟合斜率
                    out[i] = _linear_slope_numba(valid)
        result[f'alpha_021{suffix}'] = pd.Series(out, index=mean_close_6.index)

    else:
        result[f'alpha_021{suffix}'] = mean_close_6.rolling(6, min_periods=1).apply(
            lambda x: np.polyfit(np.arange(1, len(x) + 1), x, 1)[0] if len(x) >= 2 else np.nan, raw=True
        )


    val = (CLOSE - mean_close_6) / mean_close_6

    val_delay3 = ts_delay(val, 3)

    up = ts_sma(if_else(CLOSE > delay_close, std_close_20, 0), 20, 1)

    down = ts_sma(if_else(CLOSE <= delay_close, std_close_20, 0), 20, 1)

    result[f'alpha_025{suffix}'] = -1 * rank(ts_delta(CLOSE, 7) * (1 - rank(decay_linear(VOLUME / mean_volume_20, 9)))) * (
            1 + rank(ts_sum(RET, 250)))

    numerator = CLOSE - ts_min(LOW, 9)

    denominator = ts_max(HIGH, 9) - ts_min(LOW, 9)

    val = numerator / denominator * 100

    sma3 = ts_sma(val, 3, 1)

    result[f'alpha_032{suffix}'] = -1 * ts_sum(rank(ts_corr(rank_high, rank_volume, 3)), 3)

    result[f'alpha_033{suffix}'] = (-1 * ts_min(LOW, 5) + ts_delay(ts_min(LOW, 5), 5)) * rank(
        (ts_sum(RET, 240) - ts_sum(RET, 20)) / 220) * ts_rank(VOLUME, 5)

    result[f'alpha_035{suffix}'] = -1 * np.minimum(rank(decay_linear(ts_delta(OPEN, 1), 15)),
                                          rank(decay_linear(ts_corr(VOLUME, OPEN, 17), 7)))

    result[f'alpha_036{suffix}'] = rank(ts_sum(ts_corr(rank_volume, rank_vwap, 6), 2))

    result[f'alpha_037{suffix}'] = -1 * rank(
        ts_sum(OPEN, 5) * ts_sum(RET, 5) - ts_delay(ts_sum(OPEN, 5) * ts_sum(RET, 5), 10))

    result[f'alpha_039{suffix}'] = -1 * (rank(decay_linear(ts_delta(CLOSE, 2), 8)) - rank(
        decay_linear(ts_corr(VWAP * 0.3 + OPEN * 0.7, ts_sum(mean_volume_180, 37), 14), 12)))


    up_vol = ts_sum(if_else(CLOSE > delay_close, VOLUME, 0), 26)

    down_vol = ts_sum(if_else(CLOSE <= delay_close, VOLUME, 0), 26)

    result[f'alpha_041{suffix}'] = -1 * rank(ts_max(ts_delta(VWAP, 3), 5))

    result[f'alpha_042{suffix}'] = -1 * rank(ts_std(HIGH, 10)) * ts_corr(HIGH, VOLUME, 10)

    result[f'alpha_045{suffix}'] = rank(ts_delta(CLOSE * 0.6 + OPEN * 0.4, 1)) * rank(ts_corr(VWAP, mean_volume_150, 15))

    sign1 = np.sign(CLOSE - delay_close)

    sign2 = np.sign(delay_close - ts_delay(CLOSE, 2))

    sign3 = np.sign(ts_delay(CLOSE, 2) - ts_delay(CLOSE, 3))

    result[f'alpha_048{suffix}'] = -1 * (rank(sign1 + sign2 + sign3) * ts_sum(VOLUME, 5)) / ts_sum(VOLUME, 20)


    tp = (HIGH + LOW + CLOSE) / 3

    delay_tp = ts_delay(tp, 1)

    result[f'alpha_054{suffix}'] = -1 * rank(ts_std(np.abs(CLOSE - OPEN), 10) + (CLOSE - OPEN) + ts_corr(CLOSE, OPEN, 10))


    rank1 = rank(OPEN - ts_min(OPEN, 12))

    rank2 = rank(ts_corr(ts_sum((HIGH + LOW) / 2, 19), ts_sum(ts_mean(VOLUME, 40), 19), 13)) ** 5

    result[f'alpha_056{suffix}'] = (rank1 < rank2).astype(float)

    cond_eq = (CLOSE == delay_close)

    cond_gt = (CLOSE > delay_close)

    val = if_else(cond_gt, CLOSE - np.minimum(LOW, delay_close), CLOSE - np.maximum(HIGH, delay_close))

    val = if_else(cond_eq, 0, val)

    result[f'alpha_061{suffix}'] = -1 * pd.Series(np.maximum(rank(decay_linear(ts_delta(VWAP, 1), 12)),
                                                    rank(decay_linear(rank(ts_corr(LOW, mean_volume_80, 8)), 17))),
                                         index=OPEN.index)

    result[f'alpha_062{suffix}'] = -1 * ts_corr(HIGH, rank_volume, 5)

    result[f'alpha_064{suffix}'] = -1 * np.maximum(rank(decay_linear(ts_corr(rank_vwap, rank_volume, 4), 4)),
                                          rank(
                                              decay_linear(ts_max(ts_corr(rank_close, rank(mean_volume_60), 4), 13),
                                                           14)))

    dtm = if_else(OPEN <= delay_open, 0, np.maximum(HIGH - OPEN, OPEN - delay_open))

    dbm = if_else(OPEN >= delay_open, 0, np.maximum(OPEN - LOW, delay_open - OPEN))

    sum_dtm20 = ts_sum(dtm, 20)

    sum_dbm20 = ts_sum(dbm, 20)

    result[f'alpha_073{suffix}'] = -1 * (ts_rank(decay_linear(decay_linear(ts_corr(CLOSE, VOLUME, 10), 16), 4), 5) -
                                rank(decay_linear(ts_corr(VWAP, ts_mean(VOLUME, 30), 4), 3)))

    rank(decay_linear(ts_corr(VWAP, ts_mean(VOLUME, 30), 4), 3))

    result[f'alpha_074{suffix}'] = rank(ts_corr(ts_sum(LOW * 0.35 + VWAP * 0.65, 20), ts_sum(ts_mean(VOLUME, 40), 20), 7)) + \
                          rank(ts_corr(rank_vwap, rank_volume, 6))

    result[f'alpha_077{suffix}'] = np.minimum(rank(decay_linear(((HIGH + LOW) / 2 + HIGH) - (VWAP + HIGH), 20)),
                                     rank(decay_linear(ts_corr((HIGH + LOW) / 2, ts_mean(VOLUME, 40), 3), 6)))


    tp = (HIGH + LOW + CLOSE) / 3

    ma12 = ts_mean(tp, 12)

    up = ts_sma(np.maximum(CLOSE - delay_close, 0), 12, 1)

    down = ts_sma(np.abs(CLOSE - delay_close), 12, 1)

    result[f'alpha_083{suffix}'] = -1 * rank(ts_cov(rank_high, rank_volume, 5))

    term1 = (ts_delay(CLOSE, 20) - ts_delay(CLOSE, 10)) / 10

    term2 = (ts_delay(CLOSE, 10) - CLOSE) / 10

    diff = term1 - term2

    result[f'alpha_087{suffix}'] = -1 * (rank(decay_linear(ts_delta(VWAP, 4), 7)) + ts_rank(
        decay_linear((LOW * 0.9 + LOW * 0.1 - VWAP) / (OPEN - (HIGH + LOW) / 2), 11), 7))

    sma13 = ts_sma(CLOSE, 13, 2)

    sma27 = ts_sma(CLOSE, 27, 2)

    diff = sma13 - sma27

    sma10 = ts_sma(diff, 10, 2)

    result[f'alpha_090{suffix}'] = -1 * rank(ts_corr(rank_vwap, rank_volume, 5))

    result[f'alpha_091{suffix}'] = -1 * rank(CLOSE - ts_max(CLOSE, 5)) * rank(ts_corr(mean_volume_40, LOW, 5))

    result[f'alpha_092{suffix}'] = -1 * np.maximum(rank(decay_linear(ts_delta(CLOSE * 0.35 + VWAP * 0.65, 2), 3)),
                                          ts_rank(decay_linear(np.abs(ts_corr(mean_volume_180, CLOSE, 13)), 5), 15))

    val = (CLOSE - ts_min(LOW, 9)) / (ts_max(HIGH, 9) - ts_min(LOW, 9)) * 100

    sma3 = ts_sma(val, 3, 1)

    mean100 = ts_mean(CLOSE, 100)

    delta100 = ts_delta(mean100, 100)

    ratio = delta100 / ts_delay(CLOSE, 100)

    result[f'alpha_099{suffix}'] = -1 * rank(ts_corr(rank_close, rank_volume, 5))

    result[f'alpha_101{suffix}'] = -1 * (rank(ts_corr(CLOSE, ts_sum(ts_mean(VOLUME, 30), 37), 15)) <
                                rank(ts_corr(rank(HIGH * 0.1 + VWAP * 0.9), rank_volume, 11))).astype(float)

    result[f'alpha_104{suffix}'] = -1 * ts_delta(ts_corr(HIGH, VOLUME, 5), 5) * rank(ts_std(CLOSE, 20))

    result[f'alpha_105{suffix}'] = -1 * ts_corr(rank_open, rank_volume, 10)

    result[f'alpha_107{suffix}'] = rank(OPEN - delay_high) * rank(OPEN - delay_close) * rank(OPEN - delay_low)

    result[f'alpha_108{suffix}'] = -1 * rank(HIGH - ts_min(HIGH, 2)) ** rank(ts_corr(VWAP, ts_mean(VOLUME, 120), 6))


    hl = HIGH - LOW

    sma10_2 = ts_sma(hl, 10, 2)

    val = VOLUME * ((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)

    up = ts_sum(np.maximum(CLOSE - delay_close, 0), 12)

    down = ts_sum(np.abs(np.minimum(CLOSE - delay_close, 0)), 12)

    result[f'alpha_113{suffix}'] = -1 * rank(ts_mean(ts_delay(CLOSE, 5), 20)) * ts_corr(CLOSE, VOLUME, 2) * rank(
        ts_corr(ts_sum(CLOSE, 5), ts_sum(CLOSE, 20), 2))

    # [A5-20260905] 除零守卫: (VWAP-CLOSE)=0 或 (HIGH-LOW)=0 时分母链为 0 → NaN

    _hl_114 = (HIGH - LOW) / ts_mean(CLOSE, 5)

    _denom_114 = (_hl_114 / (VWAP - CLOSE).replace(0, np.nan)).replace(0, np.nan)

    result[f'alpha_114{suffix}'] = rank(ts_delay(_hl_114, 2)) * rank(rank_volume) / _denom_114

    result[f'alpha_115{suffix}'] = rank(ts_corr(HIGH * 0.9 + CLOSE * 0.1, ts_mean(VOLUME, 30), 10)) ** rank(
        ts_corr(ts_rank((HIGH + LOW) / 2, 4), ts_rank(VOLUME, 10), 7))


    result[f'alpha_119{suffix}'] = rank(decay_linear(ts_corr(VWAP, ts_sum(ts_mean(VOLUME, 5), 26), 5), 7)) - rank(
        decay_linear(ts_rank(ts_min(ts_corr(rank_open, rank(ts_mean(VOLUME, 15)), 21), 9), 7), 8))

    result[f'alpha_120{suffix}'] = rank(VWAP - CLOSE) / rank(VWAP + CLOSE)

    result[f'alpha_121{suffix}'] = -1 * (rank(VWAP - ts_min(VWAP, 12)) ** ts_rank(
        ts_corr(ts_rank(VWAP, 20), ts_rank(mean_volume_60, 2), 18), 3))


    log_close = np.log(CLOSE)

    sma13_2 = ts_sma(log_close, 13, 2)

    sma13_2_2 = ts_sma(sma13_2, 13, 2)

    sma13_2_3 = ts_sma(sma13_2_2, 13, 2)

    delay = ts_delay(sma13_2_3, 1)

    result[f'alpha_123{suffix}'] = rank(ts_corr(ts_sum((HIGH + LOW) / 2, 20), ts_sum(ts_mean(VOLUME, 60), 20), 9))

    result[f'alpha_124{suffix}'] = (CLOSE - VWAP) / decay_linear(rank(ts_max(CLOSE, 30)), 2)

    result[f'alpha_125{suffix}'] = rank(decay_linear(ts_corr(VWAP, ts_mean(VOLUME, 80), 17), 20)) / rank(
        decay_linear(ts_delta(CLOSE * 0.5 + VWAP * 0.5, 3), 16))

    max12 = ts_max(CLOSE, 12)

    term = 100 * (CLOSE - max12) / max12

    tp = (HIGH + LOW + CLOSE) / 3

    up = ts_sum(if_else(tp > ts_delay(tp, 1), tp * VOLUME, 0), 14)

    down = ts_sum(if_else(tp < ts_delay(tp, 1), tp * VOLUME, 0), 14)

    result[f'alpha_130{suffix}'] = rank(decay_linear(ts_corr((HIGH + LOW) / 2, ts_mean(VOLUME, 40), 9), 10)) / rank(
        decay_linear(ts_corr(rank_vwap, rank_volume, 7), 3))

    result[f'alpha_136{suffix}'] = -1 * rank(ts_delta(RET, 3)) * ts_corr(OPEN, VOLUME, 10)

    result[f'alpha_138{suffix}'] = -1 * (rank(decay_linear(ts_delta(LOW * 0.7 + VWAP * 0.3, 3), 20)) - ts_rank(
        decay_linear(ts_rank(ts_corr(ts_rank(LOW, 8), ts_rank(mean_volume_60, 17), 5), 19), 16), 7))

    min_low_close_6 = ts_sum(np.minimum(LOW, delay_close), 6)

    max_high_close_6 = ts_sum(np.maximum(HIGH, delay_close) - np.minimum(LOW, delay_close), 6)

    part_6 = (CLOSE - min_low_close_6) / max_high_close_6 * 12 * 24

    min_low_close_12 = ts_sum(np.minimum(LOW, delay_close), 12)

    max_high_close_12 = ts_sum(np.maximum(HIGH, delay_close) - np.minimum(LOW, delay_close), 12)

    part_12 = (CLOSE - min_low_close_12) / max_high_close_12 * 6 * 24

    min_low_close_24 = ts_sum(np.minimum(LOW, delay_close), 24)

    max_high_close_24 = ts_sum(np.maximum(HIGH, delay_close) - np.minimum(LOW, delay_close), 24)

    part_24 = (CLOSE - min_low_close_24) / max_high_close_24 * 6 * 24

    result[f'alpha_140{suffix}'] = pd.Series(
        np.minimum(rank(decay_linear((rank_open + rank_low) - (rank_high + rank_close), 8)),
                   ts_rank(decay_linear(ts_corr(ts_rank(CLOSE, 8), ts_rank(mean_volume_60, 20), 8), 7), 3)),
        index=OPEN.index)

    result[f'alpha_141{suffix}'] = -1 * rank(ts_corr(rank_high, rank(ts_mean(VOLUME, 15)), 9))

    result[f'alpha_142{suffix}'] = -1 * rank(ts_rank(CLOSE, 10)) * rank(ts_delta(ts_delta(CLOSE, 1), 1)) * rank(
        ts_rank(VOLUME / mean_volume_20, 5))

    mean12 = ts_mean(CLOSE, 12)

    result[f'alpha_148{suffix}'] = -1 * (
            rank(ts_corr(OPEN, ts_sum(ts_mean(VOLUME, 60), 9), 6)) < rank(OPEN - ts_min(OPEN, 14))).astype(
        float)

    result[f'alpha_156{suffix}'] = -1 * np.maximum(rank(decay_linear(ts_delta(VWAP, 5), 3)), rank(
        decay_linear(-1 * ts_delta(OPEN * 0.15 + LOW * 0.85, 2) / (OPEN * 0.15 + LOW * 0.85), 3)))

    result[f'alpha_157{suffix}'] = ts_min(
        rank(rank(np.log(ts_sum(ts_min(rank(rank(-1 * rank(ts_delta(CLOSE - 1, 5)))), 2), 1)))), 5) + ts_rank(
        ts_delay(-1 * RET, 6), 5)

    up = ts_sma(np.maximum(CLOSE - delay_close, 0), 12, 1)

    down = ts_sma(np.abs(CLOSE - delay_close), 12, 1)

    rsi = up / down * 100

    result[f'alpha_163{suffix}'] = rank(-1 * RET * ts_mean(VOLUME, 20) * VWAP * (HIGH - CLOSE))


    cond = CLOSE > delay_close

    inv = 1 / (CLOSE - delay_close)

    val = if_else(cond, inv, 1)

    mean48 = ts_mean(CLOSE, 48)

    cum = sumac(CLOSE - mean48)

    ret1 = CLOSE / delay_close - 1

    mean20 = ts_mean(ret1, 20)

    diff = ret1 - mean20

    sum20 = ts_sum(diff, 20)

    sum_sq20 = ts_sum(ret1 ** 2, 20)

    delay1 = ts_delay(ts_sma(CLOSE - delay_close, 9, 1), 1)

    result[f'alpha_170{suffix}'] = rank(1 / CLOSE) * VOLUME / mean_volume_20 * (HIGH * rank(HIGH - CLOSE)) / (
            ts_sum(HIGH, 5) / 5) - rank(VWAP - ts_delay(VWAP, 5))

    # [A5-20260905] 除零守卫: 一字板 CLOSE==HIGH 时分母为 0 → NaN (原产生 inf, 实证 20220801 单日 210 例)

    _denom_171 = (CLOSE - HIGH) * (CLOSE ** 5)

    tr = np.maximum(np.maximum(HIGH - LOW, np.abs(delay_close - HIGH)), np.abs(delay_close - LOW))

    hd = HIGH - delay_high

    ld = delay_low - LOW

    sum_tr14 = ts_sum(tr, 14)

    sum_ld_gt = ts_sum(if_else((ld > 0) & (ld > hd), ld, 0), 14)

    sum_hd_gt = ts_sum(if_else((hd > 0) & (hd > ld), hd, 0), 14)

    p1 = sum_ld_gt * 100 / sum_tr14

    p2 = sum_hd_gt * 100 / sum_tr14

    val = np.abs(p1 - p2) / (p1 + p2) * 100

    log_close = np.log(CLOSE)

    sma13_2 = ts_sma(CLOSE, 13, 2)

    sma13_2_2 = ts_sma(sma13_2, 13, 2)

    sma13_2_log = ts_sma(log_close, 13, 2)

    sma13_2_log_2 = ts_sma(sma13_2_log, 13, 2)

    sma13_2_log_3 = ts_sma(sma13_2_log_2, 13, 2)

    result[f'alpha_176{suffix}'] = ts_corr(rank((CLOSE - ts_min(LOW, 12)) / (ts_max(HIGH, 12) - ts_min(LOW, 12))),
                                  rank_volume, 6)

    result[f'alpha_179{suffix}'] = rank(ts_corr(VWAP, VOLUME, 4)) * rank(ts_corr(rank_low, rank(mean_volume_50), 12))

    mean24 = ts_mean(CLOSE, 24)

    cum = sumac(CLOSE - mean24)

    result[f'alpha_184{suffix}'] = rank(ts_corr(ts_delay(OPEN - CLOSE, 1), CLOSE, 200)) + rank(OPEN - CLOSE)

    result[f'alpha_185{suffix}'] = rank(-1 * (1 - (OPEN / CLOSE)) ** 2)


    tr = np.maximum(np.maximum(HIGH - LOW, np.abs(delay_close - HIGH)), np.abs(delay_close - LOW))

    hd = HIGH - delay_high

    ld = delay_low - LOW

    sum_tr14 = ts_sum(tr, 14)

    sum_ld_gt = ts_sum(if_else((ld > 0) & (ld > hd), ld, 0), 14)

    sum_hd_gt = ts_sum(if_else((hd > 0) & (hd > ld), hd, 0), 14)

    p1 = sum_ld_gt * 100 / sum_tr14

    p2 = sum_hd_gt * 100 / sum_tr14

    val = np.abs(p1 - p2) / (p1 + p2) * 100

    mean6 = ts_mean(val, 6)

    # 最后，构建 DataFrame

    # 将 result 字典转为 DataFrame，并确保索引正确

    result_df = pd.DataFrame(result, index=CLOSE.index)
    return result_df
RANK_ALPHA_COLS = ['alpha_001', 'alpha_006', 'alpha_007', 'alpha_008', 'alpha_010', 'alpha_012', 'alpha_016', 'alpha_017', 'alpha_025', 'alpha_032', 'alpha_033', 'alpha_035', 'alpha_036', 'alpha_037', 'alpha_039', 'alpha_041', 'alpha_042', 'alpha_045', 'alpha_048', 'alpha_054', 'alpha_056', 'alpha_061', 'alpha_062', 'alpha_064', 'alpha_073', 'alpha_074', 'alpha_077', 'alpha_083', 'alpha_087', 'alpha_090', 'alpha_091', 'alpha_092', 'alpha_099', 'alpha_101', 'alpha_104', 'alpha_105', 'alpha_107', 'alpha_108', 'alpha_113', 'alpha_114', 'alpha_115', 'alpha_119', 'alpha_120', 'alpha_121', 'alpha_123', 'alpha_124', 'alpha_125', 'alpha_130', 'alpha_136', 'alpha_138', 'alpha_140', 'alpha_141', 'alpha_142', 'alpha_148', 'alpha_156', 'alpha_157', 'alpha_163', 'alpha_170', 'alpha_176', 'alpha_179', 'alpha_184', 'alpha_185']
def _compute_ffd_rank_only(OPEN, HIGH, LOW, CLOSE, VOLUME, VWAP, AMOUNT, RET,
                        INDEX_CLOSE, INDEX_OPEN, suffix=''):
    """[PERF-20260905] 由 _compute_ffd_selected 自动抽取: 仅含 rank 传导闭包的 5 列 + 全部前导/中间语句 (原顺序)。
    rank() 调用序列与原函数一致 ⇒ collect/replay 的 rank_key 天然对齐。
    等价性由 U1 单测逐位验证。"""
    import numpy as np

    import pandas as pd


    # ---- 内部辅助函数 ----

    def if_else(cond, a, b):
        return pd.Series(np.where(cond, a, b), index=cond.index)


    # ---- 预计算常用中间变量 ----

    delay_close = ts_delay(CLOSE, 1)

    delay_open = ts_delay(OPEN, 1)

    delay_high = ts_delay(HIGH, 1)

    delay_low = ts_delay(LOW, 1)

    delay_volume = ts_delay(VOLUME, 1)


    rank_high = rank(HIGH)

    rank_low = rank(LOW)

    rank_close = rank(CLOSE)

    rank_open = rank(OPEN)

    rank_volume = rank(VOLUME)

    rank_vwap = rank(VWAP)


    mean_volume_20 = ts_mean(VOLUME, 20)

    mean_volume_40 = ts_mean(VOLUME, 40)

    mean_volume_50 = ts_mean(VOLUME, 50)

    mean_volume_60 = ts_mean(VOLUME, 60)

    mean_volume_80 = ts_mean(VOLUME, 80)

    mean_volume_150 = ts_mean(VOLUME, 150)

    mean_volume_180 = ts_mean(VOLUME, 180)


    mean_close_6 = ts_mean(CLOSE, 6)

    mean_close_7 = ts_mean(CLOSE, 7)

    mean_close_12 = ts_mean(CLOSE, 12)

    mean_close_20 = ts_mean(CLOSE, 20)

    mean_close_24 = ts_mean(CLOSE, 24)


    std_close_20 = ts_std(CLOSE, 20)


    # ---- 提前定义 dtm 和 dbm（alpha_069 需要） ----

    dtm = if_else(OPEN <= delay_open, 0, np.maximum(HIGH - OPEN, OPEN - delay_open))

    dbm = if_else(OPEN >= delay_open, 0, np.maximum(OPEN - LOW, delay_open - OPEN))


    result = {}


    # ---- 白名单因子（20 个） ----

    result['alpha_006_FFD'] = -1 * rank(ts_delta(OPEN * 0.85 + HIGH * 0.15, 4))

    result['alpha_008_FFD'] = -1 * rank(ts_delta((HIGH + LOW) / 2 * 0.2 + VWAP * 0.8, 4))

    result['alpha_037_FFD'] = -1 * rank(
        ts_sum(OPEN, 5) * ts_sum(RET, 5) - ts_delay(ts_sum(OPEN, 5) * ts_sum(RET, 5), 10)
    )

    result['alpha_041_FFD'] = -1 * rank(ts_max(ts_delta(VWAP, 3), 5))

    result['alpha_087_FFD'] = -1 * (rank(decay_linear(ts_delta(VWAP, 4), 7)) + ts_rank(
        decay_linear((LOW * 0.9 + LOW * 0.1 - VWAP) / (OPEN - (HIGH + LOW) / 2), 11), 7))

    result_df = pd.DataFrame(result, index=CLOSE.index)
    return result_df
RANK_FFD_COLS = ['alpha_006_FFD', 'alpha_008_FFD', 'alpha_037_FFD', 'alpha_041_FFD', 'alpha_087_FFD']

def calculate_rank_factors_for_stock_single(args):
    """[PERF-20260905] replay 专用: 仅 62 列 rank orig 因子。守卫/排序与原 worker 逐字一致。"""
    code, stock_data_dict, rank_ctx = args
    _setup_rank_ctx(rank_ctx)
    df = pd.DataFrame(stock_data_dict)
    if len(df) < 10:
        return None
    df = df.sort_values('trade_date').reset_index(drop=True)

    OPEN = df['open']
    HIGH = df['high']
    LOW = df['low']
    CLOSE = df['close']
    VOLUME = df['volume']
    VWAP = df['vwap'] if 'vwap' in df.columns else df['amount'] / (df['volume'] + 1e-10)
    AMOUNT = df['amount']
    RET = df['close'].pct_change()
    INDEX_CLOSE = df.get('index_close', pd.Series(np.nan, index=df.index))
    INDEX_OPEN = df.get('index_open', pd.Series(np.nan, index=df.index))

    if _RANK_MODE != 'timeseries':
        _rank_begin_stock(code, df['trade_date'].values.astype('datetime64[ns]'), suffix='')

    result = _compute_rank_alphas(
        OPEN, HIGH, LOW, CLOSE, VOLUME, VWAP, AMOUNT, RET,
        INDEX_CLOSE, INDEX_OPEN, suffix=''
    )
    result.insert(0, 'ts_code', code)
    result.insert(1, 'trade_date', df['trade_date'].values)
    return result


def calculate_ffd_rank_factors_for_stock_single(args):
    """[PERF-20260905] replay 专用: 仅 5 列 rank FFD 因子。守卫/排序/FFD 变换与原 worker 逐字一致。"""
    code, stock_data_dict, d_star, rank_ctx = args
    _setup_rank_ctx(rank_ctx)
    df = pd.DataFrame(stock_data_dict)
    if len(df) < 10:
        return None
    df = df.sort_values('trade_date').reset_index(drop=True)

    OPEN = df['open']
    HIGH = df['high']
    LOW = df['low']
    CLOSE = df['close']
    VOLUME = df['volume']
    VWAP = df['vwap'] if 'vwap' in df.columns else df['amount'] / (df['volume'] + 1e-10)
    AMOUNT = df['amount']
    RET = df['close'].pct_change()
    INDEX_CLOSE = df.get('index_close', pd.Series(np.nan, index=df.index))
    INDEX_OPEN = df.get('index_open', pd.Series(np.nan, index=df.index))

    if d_star > 0:
        CLOSE_FFD = fracDiff_FFD(CLOSE, d_star)
        VOLUME_FFD = fracDiff_FFD(VOLUME, d_star)
    else:
        CLOSE_FFD = CLOSE.copy()
        VOLUME_FFD = VOLUME.copy()

    if _RANK_MODE != 'timeseries':
        _rank_begin_stock(code, df['trade_date'].values.astype('datetime64[ns]'), suffix='_FFD')

    result = _compute_ffd_rank_only(
        OPEN, HIGH, LOW, CLOSE_FFD, VOLUME_FFD, VWAP, AMOUNT, RET,
        INDEX_CLOSE, INDEX_OPEN
    )
    result.insert(0, 'ts_code', code)
    result.insert(1, 'trade_date', df['trade_date'].values)
    return result
