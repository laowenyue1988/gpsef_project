# -*- coding: utf-8 -*-
"""
[PERF-20260905] test3 快速因子快照生成: 面板引擎两遍制 + 全函数 Pass2 (与 snapshots.py 逐日路径输出逐位一致)

架构 (方案文档: SmallAlpha191Test3_快照生成性能优化方案_20260905.md):
  Chunk 级 (每 CHUNK_DAYS 个交易日一次):
    C1 并行加载 [c0-400d, ce], ref=ce (线程池 + 线程局部 provider)
    C3' 数据面板: 9 原始行情面板 + 2 FFD 面板 (dates × codes)
    C4 除权股分段: T=smooth_bf 在 chunk 内有变化的股票, 按段 ref=段末日 加载
    [C2 批量 collect 帧已废除: chunk 窗口的 rolling/ewm 增量内核路径残留 (Class B)
       与基线逐日 [D-400d] 窗口产生 1e-15~1e-9 级偏差, 无法构造性消除]
  逐日级 (每个交易日, 输入面板按 [D-400d, D] 切片 + 除权股补丁):
    D1 Pass1 (collect 语义): 记录 111+11 个 rank arg (NEED_TS key 返回时序 rank),
       缺口股 arg 用 mini-collect 补丁 → float32 → 停牌掩码 → 截面 rank 面板
    D2 Pass2 (replay 语义, 全函数): rank()=面板查找, 收割全部 190+20 列当日值
       (rank-free 列同获逐日窗口语义 ⇒ rolling 补偿态/ewm 递归/cumsum 起点
        与基线逐日 worker 逐步一致 — Class B 的构造性根治)
    D3 装配: 引擎当日行 (缺口/回退股用逐股全量 replay 行)
    D4 <10bar 守卫 → A11 补齐 → A3 停牌过滤 → A5 净化 → 保存 (全部复用 snapshots.py 语义)
  运行时安全网: 每日 5 只安全股全列 replay 抽检, 不一致 → 当日全量 replay 回退。

精确性: 见方案文档 §3 构造性论证; U4a/U4b/U4c/U4d/U5 探针 + E2E 全表逐位比对为硬门槛。
"""

import os
import sys
import gc
import time
import random
import shutil
import pickle
import logging
import threading
import tempfile
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpsef.data import LocalDataProvider
from strategies.small_alpha191_test3.data_utils import get_trade_calendar, VALID_STOCK_PREFIXES
from strategies.small_alpha191_test3 import factors as F
from strategies.small_alpha191_test3 import snapshots as S
from strategies.small_alpha191_test3 import panel_engine as PE

logger = logging.getLogger(__name__)

# ==================== 配置 ====================
DATA_BASE_PATH = r'E:\quant\gpsef_project\data'
SNAPSHOT_DIR_NAME = 'factor_snapshots_test'
CHUNK_DAYS = 20                  # 每 chunk 交易日数 (越大摊销越多, 内存越大)
MAX_WORKERS = 14                 # 持久进程池 (22 逻辑核)
LOAD_THREADS = 16
LOAD_WINDOW_DAYS = 400
SKIP_EXISTING = True
# [全函数Pass2-20260905] BOUNDARY_COLS 机制已废除: 全部列均由引擎在逐日窗口重算,
# 不再存在"帧值 + 边界列补丁"的混合路径 (原机制对 ewm/cumsum 之外的 rolling 补偿态
# 路径残留 (Class B, 1e-15 级) 无能为力, 凡 [D-400d] 与 [c0-400d] 首个交易日不同即污染)。

REQUIRED_COLS = ['open', 'high', 'low', 'close', 'volume', 'amount',
                 'vwap', 'index_close', 'index_open', 'trade_date', 'ts_code',
                 'tradestatus', 'isST']
QUOTE_COLS = ['open', 'high', 'low', 'close', 'volume', 'amount', 'vwap', 'tradestatus', 'isST']

# ==================== 持久进程池 ====================
_POOL = None
_POOL_WORKERS = 0


def get_pool(workers):
    global _POOL, _POOL_WORKERS
    if _POOL is None or _POOL_WORKERS != workers:
        if _POOL is not None:
            _POOL.shutdown(wait=True)
        _POOL = ProcessPoolExecutor(max_workers=workers)
        _POOL_WORKERS = workers
    return _POOL


def shutdown_pool():
    global _POOL, _POOL_WORKERS
    if _POOL is not None:
        _POOL.shutdown(wait=True)
        _POOL = None
        _POOL_WORKERS = 0


def _worker_release_panel(_ignored=None):
    """worker 子进程内释放 rank 面板 memmap 句柄 (幂等)。
    [mmapfix-20260906] Windows 下 worker 经 np.load(mmap_mode='r') 加载 cs_rank_panel.npy
    后持有文件句柄, 主进程 rmtree(day_dir) 因文件被占用而静默失败 → day 目录逐日累积撑爆磁盘。
    每日 replay 结束后向所有 worker 派发本任务, 令其释放句柄后再删目录。"""
    F._rank_release_panel()
    return True


def _release_worker_panels(pool):
    """向进程池每个 worker 派发一次 _worker_release_panel, 确保全部释放 memmap 句柄。"""
    n = _POOL_WORKERS if _POOL_WORKERS else 1
    futs = [pool.submit(_worker_release_panel) for _ in range(n)]
    for f in futs:
        try:
            f.result()
        except Exception:
            pass


# ==================== C4: 复权 T 值与除权分段 ====================

def _smooth_bf_one(args):
    """复制 LocalDataProvider._load_full_daily_data 的 merge_asof + smooth_bf 逻辑。
    返回 (code, bar_dates datetime64[ns], smooth_bf float64)。"""
    code, data_dir = args
    try:
        raw_file = os.path.join(data_dir, 'raw_daily', code.replace('.', '_') + '.parquet')
        if not os.path.exists(raw_file):
            return code, None, None
        df_raw = pd.read_parquet(raw_file, columns=['trade_date'])
        if df_raw.empty:
            return code, None, None
        df_raw['trade_date'] = pd.to_datetime(df_raw['trade_date'])
        df_raw = df_raw.sort_values('trade_date').reset_index(drop=True)

        back_file = os.path.join(data_dir, 'back_adjust_factors',
                                 code.replace('.', '_') + '_back_adjust.parquet')
        if os.path.exists(back_file):
            df_back = pd.read_parquet(back_file, columns=['dividOperateDate', 'backAdjustFactor'])
            df_back['dividOperateDate'] = pd.to_datetime(df_back['dividOperateDate'])
            df_merged = pd.merge_asof(df_raw, df_back[['dividOperateDate', 'backAdjustFactor']],
                                      left_on='trade_date', right_on='dividOperateDate',
                                      direction='backward')
            df_merged['backAdjustFactor'] = df_merged['backAdjustFactor'].fillna(1.0)
        else:
            df_merged = df_raw.copy()
            df_merged['backAdjustFactor'] = 1.0

        bf_vals = df_merged['backAdjustFactor'].values
        smooth_bf = np.empty_like(bf_vals)
        smooth_bf[0] = bf_vals[0]
        for i in range(1, len(bf_vals)):
            if bf_vals[i] > 0 and bf_vals[i - 1] > 0:
                ratio = bf_vals[i] / bf_vals[i - 1]
                if ratio >= 1.0:
                    smooth_bf[i] = smooth_bf[i - 1] * ratio
                else:
                    smooth_bf[i] = smooth_bf[i - 1]
            else:
                smooth_bf[i] = smooth_bf[i - 1]
        return code, df_raw['trade_date'].values.astype('datetime64[ns]'), smooth_bf
    except Exception as e:
        logger.warning(f"[C4] {code} smooth_bf 计算失败: {e}")
        return code, None, None


def build_bf_table(codes, data_dir, threads=LOAD_THREADS):
    """{code: (bar_dates, smooth_bf)}，带磁盘缓存 (raw/back 文件 mtime 失效)。"""
    cache_file = os.path.join(data_dir, 'stock_list', 'bf_smooth_cache_test3.pkl')
    cache = {}
    mtimes = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cache, mtimes = pickle.load(f)
        except Exception:
            cache, mtimes = {}, {}
    todo = []
    for code in codes:
        raw_file = os.path.join(data_dir, 'raw_daily', code.replace('.', '_') + '.parquet')
        back_file = os.path.join(data_dir, 'back_adjust_factors',
                                 code.replace('.', '_') + '_back_adjust.parquet')
        if not os.path.exists(raw_file):
            continue
        mt = (os.path.getmtime(raw_file),
              os.path.getmtime(back_file) if os.path.exists(back_file) else 0)
        if code in cache and mtimes.get(code) == mt:
            continue
        mtimes[code] = mt
        todo.append((code, data_dir))
    if todo:
        logger.info(f"[C4] 计算 smooth_bf: {len(todo)} 只 (缓存命中 {len(codes) - len(todo)} 只)")
        with ThreadPoolExecutor(max_workers=threads) as ex:
            for code, dates, smooth in ex.map(_smooth_bf_one, todo, chunksize=32):
                cache[code] = (dates, smooth)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump((cache, mtimes), f)
    return cache


def T_at(bf_entry, D):
    """复制 provider: T(D) = smooth_bf 在最后 ≤D 的 bar; D 早于首 bar 则取首值; 0→1.0"""
    dates, smooth = bf_entry
    if dates is None:
        return 1.0
    idx = np.searchsorted(dates, np.datetime64(D), side='right') - 1
    t = smooth[idx] if idx >= 0 else smooth[0]
    return 1.0 if t == 0 else t


def compute_action_segments(bf_tbl, chunk_dates):
    """{code: [(seg_start_di, seg_end_di, ref_date), ...]} — 仅 T 在 chunk 日期上有变化的股票。
    seg_*_di 为 chunk_dates 下标 (闭区间); ref_date = 段末日期。"""
    ce = chunk_dates[-1]
    segs = {}
    for code, entry in bf_tbl.items():
        ts = np.array([T_at(entry, D) for D in chunk_dates])
        if (ts == ts[-1]).all():
            continue                      # 无变化: 批量数据全 chunk 有效
        runs = []
        start = 0
        for i in range(1, len(ts)):
            if ts[i] != ts[start]:
                runs.append((start, i - 1, chunk_dates[i - 1]))
                start = i
        runs.append((start, len(ts) - 1, chunk_dates[-1]))
        segs[code] = runs
    return segs


# ==================== C1: 并行加载 ====================
_THREAD_LOCAL = threading.local()


def _thread_provider():
    p = getattr(_THREAD_LOCAL, 'provider', None)
    if p is None:
        p = LocalDataProvider(DATA_BASE_PATH)
        _THREAD_LOCAL.provider = p
    return p


def _load_one(args):
    code, start, end, ref = args
    try:
        df = _thread_provider().get_daily_data(code, start, end, reference_date=ref)
        if df is not None and not df.empty:
            df['ts_code'] = code
            return code, df
    except Exception as e:
        logger.debug(f"[C1] 加载 {code} 失败: {e}")
    return code, None


def load_parallel(codes, start, end, ref, threads=LOAD_THREADS):
    """{code: df}；df 内容与串行逐日加载逐位一致 (同 provider 逻辑)。"""
    out = {}
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for code, df in ex.map(_load_one,
                               [(c, start, end, ref) for c in codes], chunksize=16):
            if df is not None:
                out[code] = df
    return out


# ==================== C2/C4: collect 执行 ====================

def _d_star(data_dir):
    f = os.path.join(data_dir, 'ffd_params', 'd_star_median.txt')
    with open(f, 'r') as fh:
        return float(fh.read().strip())


def run_collect(data_by_code, d_star, pool, tag=''):
    """全量 collect: 返回 (frames {code: {'orig':df,'ffd':df}}, registry {key:{code:(dates,vals)}})"""
    tasks = []
    for code in sorted(data_by_code.keys()):
        df = data_by_code[code]
        avail = [c for c in REQUIRED_COLS if c in df.columns]
        sd = df[avail].to_dict('list')
        tasks.append(('orig', code, sd))
        tasks.append(('ffd', code, sd))
    registry = {}
    frames = {}
    futs = []
    for kind, code, sd in tasks:
        if kind == 'orig':
            futs.append((code, 'orig', pool.submit(
                F.calculate_factors_for_stock_single, (code, sd, {'mode': 'collect'}))))
        else:
            futs.append((code, 'ffd', pool.submit(
                F.calculate_ffd_factors_for_stock_single, (code, sd, d_star, {'mode': 'collect'}))))
    n_done = 0
    for code, kind, fut in futs:
        try:
            res = fut.result()
        except Exception as e:
            logger.error(f"[collect{tag}] {code}/{kind} 失败: {e}")
            continue
        if res is None:
            continue
        df_res, chunk_reg = res
        for k, v in chunk_reg.items():
            registry.setdefault(k, {}).update(v)
        if df_res is not None and not df_res.empty:
            frames.setdefault(code, {})[kind] = df_res
        n_done += 1
        if n_done % 1000 == 0:
            logger.info(f"  [collect{tag}] {n_done}/{len(futs)}")
    return frames, registry


# ==================== C3': chunk 级数据面板 (面板引擎输入) ====================
PANEL_COLS = ['open', 'high', 'low', 'close', 'volume', 'amount', 'vwap',
              'index_close', 'index_open']


def build_chunk_panels(df_daily, data_by_code, all_dates, codes, d_star):
    """9 个原始行情面板 + 2 个 FFD 面板 (dates × codes, T(ce) 口径)。
    vwap 缺失列按 worker 语义回退 amount/(volume+1e-10)。"""
    dt_idx = pd.DatetimeIndex(all_dates)
    raw = {}
    for c in PANEL_COLS:
        if c in df_daily.columns:
            p = df_daily.pivot_table(index='trade_date', columns='ts_code',
                                     values=c, aggfunc='first')
            raw[c] = p.reindex(index=dt_idx, columns=codes)
        else:
            # 数据源无此列 (如 index_close/index_open): 全 NaN,
            # 与 worker df.get('index_close', NaN) 回退语义一致
            raw[c] = pd.DataFrame(np.nan, index=dt_idx, columns=codes)
    # vwap 列缺失回退 (与 worker: df['vwap'] if 'vwap' in df.columns else amount/(volume+1e-10) 一致)
    for code, df in data_by_code.items():
        if 'vwap' not in df.columns and code in raw['vwap'].columns:
            s = (df['amount'] / (df['volume'] + 1e-10))
            raw['vwap'].loc[df['trade_date'].values, code] = s.values
    ffd_w = PE.ffd_width(d_star)
    close_ffd = PE.ffd_panel(raw['close'], d_star)
    volume_ffd = PE.ffd_panel(raw['volume'], d_star)
    return raw, close_ffd, volume_ffd, ffd_w


# ==================== 主流程 ====================

def generate_snapshots_fast(start_date, end_date, data_dir=DATA_BASE_PATH,
                            snapshot_dir_name=SNAPSHOT_DIR_NAME,
                            chunk_days=CHUNK_DAYS, max_workers=MAX_WORKERS,
                            skip_existing=SKIP_EXISTING, work_root=None):
    t_run = time.time()
    logger.info("=" * 70)
    logger.info("【[test3-PERF] 快速快照生成: 批量采集 + 逐日精放】")
    logger.info(f"区间 {start_date}~{end_date} | chunk={chunk_days}日 | workers={max_workers}")
    logger.info("=" * 70)

    # [A9] FFD d* 硬校验
    d_star = _d_star(data_dir)
    logger.info(f"[A9] FFD d* = {d_star}")

    cal = get_trade_calendar(start_date, end_date)
    trade_dates = [pd.to_datetime(d) for d in sorted(cal['calendar_date'].tolist())]
    if not trade_dates:
        logger.error("区间内无交易日")
        return
    logger.info(f"交易日: {len(trade_dates)}")

    universe = S.build_stock_universe(data_dir)
    universe['list_date'] = pd.to_datetime(universe['list_date'])

    provider = LocalDataProvider(data_dir)
    provider.snapshot_path = os.path.join(data_dir, snapshot_dir_name)
    os.makedirs(provider.snapshot_path, exist_ok=True)

    pool = get_pool(max_workers)
    bf_tbl = build_bf_table(sorted(universe['code']), data_dir)
    engine = PE.PanelEngine(F)          # 面板引擎 (exec 编译一次, 全程复用)

    work_root = work_root or tempfile.mkdtemp(prefix='snapfast_')
    logger.info(f"工作目录: {work_root}")

    n_ok = n_skip = n_fail = 0
    try:
        for c0i in range(0, len(trade_dates), chunk_days):
            chunk_dates = trade_dates[c0i:c0i + chunk_days]
            c0, ce = chunk_dates[0], chunk_dates[-1]
            # 本 chunk 待处理日期 (skip_existing)
            todo_dates = []
            for D in chunk_dates:
                if skip_existing:
                    ex = provider.get_factor_snapshot(D)
                    if ex is not None and any(str(c).endswith('_FFD') for c in ex.columns):
                        n_skip += 1
                        continue
                todo_dates.append(D)
            if not todo_dates:
                logger.info(f"[chunk {c0.date()}~{ce.date()}] 全部已存在, 跳过")
                continue
            t_chunk = time.time()
            logger.info("=" * 60)
            logger.info(f"[chunk {c0.date()}~{ce.date()}] {len(todo_dates)}/{len(chunk_dates)} 日待生成")

            # ---- C1 并行加载 (ref=ce) ----
            listed = universe[universe['list_date'] <= ce]['code'].tolist()
            ext_start = c0 - timedelta(days=LOAD_WINDOW_DAYS)
            t0 = time.time()
            data_by_code = load_parallel(listed, ext_start, ce, ce)
            data_by_code = {c: d for c, d in data_by_code.items()
                            if c.startswith(VALID_STOCK_PREFIXES)}
            logger.info(f"[C1] 加载 {len(data_by_code)} 股: {time.time()-t0:.1f}s")

            # [C2 已废除-20260905] 批量 collect 帧 (chunk 窗口) 被全函数 Pass2 取代:
            # chunk 帧的 rolling/ewm 增量内核携带 [c0-400d] 起点的路径残留, 与基线逐日
            # [D-400d] 窗口在 ewm 递归/补偿状态上产生 1e-15~1e-9 级偏差 (Class B),
            # 凡 [D-400d] 与 [c0-400d] 首个交易日不同的日子即污染快照 (20 日 chunk ~19/20 天)。
            # 全函数 Pass2 在逐日 [D-400d, D] 面板切片上收割全部 190+20 列, 构造性逐位。

            # ---- C3' 数据面板 (引擎输入) ----
            t0 = time.time()
            df_daily = pd.concat(list(data_by_code.values()), ignore_index=True)
            df_daily = df_daily.drop_duplicates(subset=['ts_code', 'trade_date'])
            all_dates = np.unique(df_daily['trade_date'].values.astype('datetime64[ns]'))
            codes = sorted(df_daily['ts_code'].unique())
            raw_panels, close_ffd_p, volume_ffd_p, ffd_w = build_chunk_panels(
                df_daily, data_by_code, all_dates, codes, d_star)
            logger.info(f"[C3'] 数据面板: {time.time()-t0:.1f}s "
                        f"({len(all_dates)} dates × {len(codes)} codes, ffd_w={ffd_w})")

            # ---- 停牌掩码 (chunk 级) ----
            ts_frame = df_daily[['trade_date', 'ts_code', 'tradestatus']].copy()
            ts_frame['tradestatus'] = ts_frame['tradestatus'].astype(str)
            ts_pivot = ts_frame.pivot_table(index='trade_date', columns='ts_code',
                                            values='tradestatus', aggfunc='first')
            ts_pivot = ts_pivot.reindex(index=pd.DatetimeIndex(all_dates),
                                        columns=codes)
            ts_ok_all = (ts_pivot.values == '1')

            # ---- C4 除权股分段 (仅加载分段数据; 分段 collect 已随 C2 一并废除) ----
            t0 = time.time()
            segs = compute_action_segments(bf_tbl, chunk_dates)
            seg_data = {}       # (code, seg_idx) -> df (ref=段末日 加载)
            for code, runs in sorted(segs.items()):
                ce_sdi = len(runs) - 1          # ce 恒在最后一段, 与主数据重复 → 跳过
                for sdi, (a, b, ref_d) in enumerate(runs):
                    if sdi == ce_sdi:
                        continue
                    d = load_parallel([code], ext_start, ref_d, ref_d, threads=2)
                    if code in d:
                        seg_data[(code, sdi)] = d[code]
            logger.info(f"[C4] 除权股 {len(segs)} 只 / {len(seg_data)} 段(含跳过ce段): {time.time()-t0:.1f}s")

            # ---- 逐日处理 ----
            for D in todo_dates:
                tD = time.time()
                win_start = D - timedelta(days=LOAD_WINDOW_DAYS)
                ok = _process_one_date(
                    D, win_start, c0, ce, chunk_dates, data_by_code,
                    all_dates, codes, raw_panels, close_ffd_p, volume_ffd_p, ffd_w,
                    ts_ok_all, segs, seg_data,
                    df_daily, d_star, pool, engine,
                    provider, work_root)
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1
                logger.info(f"  [{D.date()}] 完成 ({time.time()-tD:.1f}s) | "
                            f"累计 成功{n_ok} 跳过{n_skip} 失败{n_fail}")
                gc.collect()

            # chunk 收尾清理
            del df_daily
            del raw_panels, close_ffd_p, volume_ffd_p
            F._rank_release_panel()
            gc.collect()
            logger.info(f"[chunk {c0.date()}~{ce.date()}] 完成, 耗时 {time.time()-t_chunk:.1f}s")
    finally:
        shutdown_pool()
        # 清理逐日面板目录
        try:
            shutil.rmtree(work_root, ignore_errors=True)
        except Exception as e:
            logger.warning(f"清理工作目录失败: {e}")

    logger.info("=" * 70)
    logger.info(f"【快速生成完成】 成功 {n_ok} | 跳过 {n_skip} | 失败 {n_fail} | "
                f"总耗时 {(time.time()-t_run)/60:.1f} 分钟")
    logger.info("=" * 70)


def _replay_rows(codes_list, day_bar_src, day_dir, d_star, pool, D):
    """逐股全量 replay (读 day_dir 面板, rank()=面板查找; 与基线逐日 replay 同一路径),
    返回 ({code: 当日行(全列)}, {code: 当日行(FFD全列)})。
    [全函数Pass2-20260905] 由 rank-only replay 改为全量 replay: 缺口/回退股的
    rank-free 列同样必须来自逐日窗口的 per-stock 计算 (面板语义对缺口股不等价)。"""
    ctx = {'mode': 'replay', 'panel_dir': day_dir}
    futs = []
    for code in codes_list:
        sl = day_bar_src[code]
        avail = [c for c in REQUIRED_COLS if c in sl.columns]
        sd = sl[avail].to_dict('list')
        futs.append(('orig', code, pool.submit(
            F.calculate_factors_for_stock_single, (code, sd, ctx))))
        futs.append(('ffd', code, pool.submit(
            F.calculate_ffd_rank_factors_for_stock_single, (code, sd, d_star, ctx))))
    rank_orig, rank_ffd = {}, {}
    for kind, code, fut in futs:
        try:
            res = fut.result()
        except Exception as e:
            logger.error(f"  [replay] {code}/{kind} 失败: {e}")
            continue
        if res is None or res.empty:
            continue
        row = res[res['trade_date'].dt.date == D.date()]
        if row.empty:
            continue
        if kind == 'orig':
            rank_orig[code] = row.iloc[0]
        else:
            rank_ffd[code] = row.iloc[0]
    return rank_orig, rank_ffd


def _process_one_date(D, win_start, c0, ce, chunk_dates, data_by_code,
                      all_dates, codes, raw_panels, close_ffd_p, volume_ffd_p, ffd_w,
                      ts_ok_all, segs, seg_data,
                      df_daily, d_star, pool, engine,
                      provider, work_root):
    try:
        # ---- 窗口行 (连续区间) ----
        i0 = np.searchsorted(all_dates, np.datetime64(win_start), side='left')
        i1 = np.searchsorted(all_dates, np.datetime64(D), side='right')
        win_dates = all_dates[i0:i1]
        ts_ok_win = ts_ok_all[i0:i1, :]
        code_pos = {c: i for i, c in enumerate(codes)}

        # ---- 除权分段定位 ----
        ce_seg = {code: _seg_idx_of(runs, chunk_dates, ce) for code, runs in segs.items()}
        seg_of_stock = {code: _seg_idx_of(runs, chunk_dates, D) for code, runs in segs.items()}

        def _is_seg_stock(code):
            sdi = seg_of_stock.get(code)
            return sdi is not None and sdi != ce_seg.get(code)

        # ---- 当日数据切片 + <10bar 守卫 + 内部缺口分类 ----
        day_bar_src = {}
        exclude_codes = set()       # <10bar: 截面剔除 (与 collect 守卫 return None 一致)
        gap_codes = []              # 内部缺口: 面板语义不等价 → arg 补丁 + 逐股 replay
        for code in sorted(data_by_code.keys()):
            if _is_seg_stock(code):
                df = seg_data.get((code, seg_of_stock[code]))
            else:
                df = data_by_code[code]
            if df is None:
                continue
            sl = df[(df['trade_date'] >= win_start) & (df['trade_date'] <= D)]
            if len(sl) < 10:
                exclude_codes.add(code)
                continue
            day_bar_src[code] = sl
            first = sl['trade_date'].values.astype('datetime64[ns]')[0]
            n_expected = len(win_dates) - np.searchsorted(win_dates, first, side='left')
            if len(sl) != n_expected:
                gap_codes.append(code)

        # ---- 引擎输入切片 + 除权股补丁 (T(D) 口径) ----
        # 注意: 直接在 ndarray 上补丁 (pandas CoW 下 DataFrame.values 只读),
        # 补丁完成后再构 DataFrame
        win_idx = pd.DatetimeIndex(win_dates)
        arr = {c: raw_panels[c].values[i0:i1].copy() for c in PANEL_COLS}
        close_ffd_a = close_ffd_p.values[i0:i1].copy()
        volume_ffd_a = volume_ffd_p.values[i0:i1].copy()
        for code in day_bar_src:
            if not _is_seg_stock(code):
                continue
            sl = day_bar_src[code]
            ci = code_pos[code]
            pos = np.searchsorted(win_dates,
                                  sl['trade_date'].values.astype('datetime64[ns]'))
            for c in PANEL_COLS:
                if c in sl.columns:
                    arr[c][pos, ci] = sl[c].values
            # FFD 逐股窗口重算 (与逐日 per-stock 同一函数同一输入 ⇒ 逐位)
            close_ffd_a[pos, ci] = PE.fracDiff_FFD(sl['close'], d_star).values
            volume_ffd_a[pos, ci] = PE.fracDiff_FFD(sl['volume'], d_star).values
        close_ffd_a[:ffd_w] = np.nan          # 逐日窗口 FFD 暖机 NaN (固定窗口语义)
        volume_ffd_a[:ffd_w] = np.nan
        slices = {c: pd.DataFrame(arr[c], index=win_idx, columns=codes) for c in PANEL_COLS}
        close_ffd_s = pd.DataFrame(close_ffd_a, index=win_idx, columns=codes)
        volume_ffd_s = pd.DataFrame(volume_ffd_a, index=win_idx, columns=codes)
        ret_s = slices['close'].pct_change()

        # ---- 缺口股 mini-collect → rank 参数补丁 ----
        patch_orig = patch_ffd = None
        if gap_codes:
            gap_data = {code: day_bar_src[code] for code in gap_codes}
            _, gap_reg = run_collect(gap_data, d_star, pool, tag='-gap')
            patch_orig = {k: v for k, v in gap_reg.items() if not k.startswith('_FFD')}
            patch_ffd = {k: v for k, v in gap_reg.items() if k.startswith('_FFD')}

        # ---- D1+D2 面板引擎 (全函数 Pass2): 全部 190+20 列当日值 + 全 key 截面面板 ----
        t_eng = time.time()
        inputs = {'OPEN': slices['open'], 'HIGH': slices['high'], 'LOW': slices['low'],
                  'CLOSE': slices['close'], 'VOLUME': slices['volume'],
                  'VWAP': slices['vwap'], 'AMOUNT': slices['amount'], 'RET': ret_s,
                  'INDEX_CLOSE': slices['index_close'], 'INDEX_OPEN': slices['index_open']}
        ffd_inputs = dict(inputs)
        ffd_inputs['CLOSE'] = close_ffd_s
        ffd_inputs['VOLUME'] = volume_ffd_s
        res_o, res_f = engine.run(inputs, ffd_inputs, ts_ok_win, win_dates, codes,
                                  exclude_codes=exclude_codes,
                                  patch_orig=patch_orig, patch_ffd=patch_ffd, full=True)
        eng_secs = time.time() - t_eng
        logger.info(f"  [{D.date()}] 面板引擎 {eng_secs:.1f}s | "
                    f"缺口 {len(gap_codes)} | 除权 {sum(1 for c in day_bar_src if _is_seg_stock(c))} "
                    f"| <10bar 守卫 {len(exclude_codes)}")

        # ---- 写当日 rank 面板 (缺口 replay / 抽检 / 回退共用) ----
        day_dir = os.path.join(work_root, f'day_{D:%Y%m%d}')
        engine.write_rank_panel(day_dir, win_dates, codes)

        # ---- 缺口股逐股全量 replay (全部列均取自逐日窗口 per-stock 计算) ----
        replay_o, replay_f = {}, {}
        if gap_codes:
            replay_o, replay_f = _replay_rows(gap_codes, day_bar_src, day_dir,
                                              d_star, pool, D)

        # ---- 抽检: 引擎安全股每日 5 只逐股全量 replay 对拍 (全部列), 不一致 → 全量回退 ----
        safe_codes = [c for c in day_bar_src if c not in set(gap_codes)]
        eng_vals_o = {col: p.df.iloc[-1] for col, p in res_o.items()}
        eng_vals_f = {col: p.df.iloc[-1] for col, p in res_f.items()}
        rng = random.Random(int(D.strftime('%Y%m%d')))
        spot = rng.sample(safe_codes, min(5, len(safe_codes)))
        n_mismatch = 0
        if spot:
            spot_o, spot_f = _replay_rows(spot, day_bar_src, day_dir, d_star, pool, D)
            for code in spot:
                if code not in spot_o:
                    continue
                for col, ser in eng_vals_o.items():
                    if col not in spot_o[code].index:
                        continue
                    ev, rv = ser[code], spot_o[code][col]
                    if not ((ev == rv) or (pd.isna(ev) and pd.isna(rv))):
                        n_mismatch += 1
                if code in spot_f:
                    for col, ser in eng_vals_f.items():
                        if col not in spot_f[code].index:
                            continue
                        ev, rv = ser[code], spot_f[code][col]
                        if not ((ev == rv) or (pd.isna(ev) and pd.isna(rv))):
                            n_mismatch += 1
        if n_mismatch:
            logger.error(f"  [{D.date()}] 面板引擎抽检不一致 {n_mismatch} 格 "
                         f"→ 当日全量逐股 replay 回退 (结果仍精确, 仅速度退化)")
            fb_o, fb_f = _replay_rows(safe_codes, day_bar_src, day_dir, d_star, pool, D)
            replay_o.update(fb_o)
            replay_f.update(fb_f)

        F._rank_release_panel()
        # [mmapfix-20260906] 释放所有 worker 进程持有的 rank 面板 memmap 句柄,
        # 否则 Windows 下 day_dir 文件被占用, 下面 rmtree 静默失败 → 目录逐日累积。
        _release_worker_panels(pool)
        try:
            shutil.rmtree(day_dir, ignore_errors=True)
        except Exception:
            pass

        # ---- D3 装配: 全部列 = 引擎全函数 Pass2 当日行 (缺口/回退股用 replay 全行) ----
        # [全函数Pass2-20260905] C2 帧装配与边界列覆盖已废除: 所有 alpha 列均在
        # 逐日 [D-400d, D] 面板切片上产出, 与基线逐日 worker 构造性逐位等价。
        engine_codes = sorted(day_bar_src.keys())
        rows = []
        d64 = np.datetime64(D)
        for code in engine_codes:
            sl = day_bar_src[code]
            if sl['trade_date'].values.astype('datetime64[ns]')[-1] != d64:
                continue                    # 当日无行情行 → 不在快照 (与逐日路径一致)
            if code in replay_o:
                row = replay_o[code].to_dict()
                row['ts_code'] = code
                if code in replay_f:
                    for c, v in replay_f[code].items():
                        if str(c).endswith('_FFD'):
                            row[c] = v
                rows.append(row)
                continue
            row = {'ts_code': code, 'trade_date': sl['trade_date'].values[-1]}
            for col, ser in eng_vals_o.items():
                row[col] = ser[code]
            for qc in QUOTE_COLS:         # worker 从 df 复制的行情列 (仅存在的列)
                if qc in sl.columns:
                    row[qc] = sl[qc].values[-1]
            for col, ser in eng_vals_f.items():
                row[col] = ser[code]
            rows.append(row)
        if not rows:
            logger.warning(f"  [{D.date()}] 无引擎行")
            return False
        day_data = pd.DataFrame(rows)

        # ---- A11 新股补齐 (除权股行情用分段数据覆盖, 保持 T(D) 口径) ----
        day_bars = df_daily[df_daily['trade_date'] == D]
        missing = sorted(set(day_bars['ts_code']) - set(day_data['ts_code']))
        if missing:
            supp = day_bars[day_bars['ts_code'].isin(missing)].copy()
            for c in day_data.columns:
                if c not in supp.columns:
                    supp[c] = np.nan
            for code in missing:
                if _is_seg_stock(code):
                    sdf = seg_data.get((code, seg_of_stock[code]))
                    if sdf is not None:
                        srow = sdf[sdf['trade_date'] == D]
                        if not srow.empty:
                            for c in QUOTE_COLS:
                                if c in srow.columns and c in supp.columns:
                                    supp.loc[supp['ts_code'] == code, c] = srow.iloc[0][c]
            day_data = pd.concat([day_data, supp[day_data.columns]], ignore_index=True)
            logger.info(f"  [A11] 补齐 {len(missing)} 只: {missing[:5]}"
                        f"{'...' if len(missing) > 5 else ''}")

        # ---- A3 / A5 / 保存 ----
        day_data = S._filter_suspended(day_data)
        day_data = S._sanitize_snapshot(day_data)
        provider.save_factor_snapshot(day_data, D)
        return True
    except Exception as e:
        logger.error(f"  [{D.date()}] 处理失败: {e}", exc_info=True)
        return False


def _seg_idx_of(runs, chunk_dates, D):
    for sdi, (a, b, _) in enumerate(runs):
        if chunk_dates[a] <= D <= chunk_dates[b]:
            return sdi
    return None


# ==================== 主入口 (argparse CLI) ====================
def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description='test3 快速因子快照生成 (面板引擎两遍制 + 全函数 Pass2, '
                    '与 snapshots.py 逐日路径输出逐位一致)')
    p.add_argument('--start', required=True, metavar='YYYYMMDD',
                   help='起始交易日 (含), 如 20200101')
    p.add_argument('--end', required=True, metavar='YYYYMMDD',
                   help='结束交易日 (含), 如 20260901')
    p.add_argument('--out', default=SNAPSHOT_DIR_NAME,
                   help='输出目录名 (位于 data/ 下, 默认 %(default)s)')
    p.add_argument('--data-dir', default=DATA_BASE_PATH,
                   help='数据根目录 (默认 %(default)s)')
    p.add_argument('--chunk-days', type=int, default=CHUNK_DAYS,
                   help='每 chunk 交易日数 (默认 %(default)s)')
    p.add_argument('--workers', type=int, default=MAX_WORKERS,
                   help='持久进程池 worker 数 (默认 %(default)s)')
    p.add_argument('--work-root', default=None,
                   help='临时工作目录 (默认系统 Temp; 如需写到数据盘可指定, 如 '
                        'E:\\quant\\gpsef_project\\data\\_snapfast_work)')
    p.add_argument('--skip-existing', dest='skip_existing', action='store_true',
                   default=SKIP_EXISTING, help='已存在快照则跳过 (默认开启)')
    p.add_argument('--no-skip-existing', dest='skip_existing', action='store_false',
                   help='强制重算已存在日期')
    args = p.parse_args(argv)

    generate_snapshots_fast(args.start, args.end,
                            data_dir=args.data_dir,
                            snapshot_dir_name=args.out,
                            chunk_days=args.chunk_days,
                            max_workers=args.workers,
                            skip_existing=args.skip_existing,
                            work_root=args.work_root)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    main()
