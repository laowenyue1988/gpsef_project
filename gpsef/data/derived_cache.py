# -*- coding: utf-8 -*-
"""派生缓存一致性工具箱（框架层 · 单一实现来源）。

设计原则（2026-09-10 定稿）::

    结果     = f( 原始数据 , 过滤规则 , 代码 )     ← 唯一真值来源
    派生缓存 = g( 原始数据 , 过滤规则 )            ← 只是物化视图，随时可重建

只要 g 是**幂等**纯函数、输入未变，那么无论缓存处于什么状态
（陈旧 / 缺失 / 被重建 / 被部分重写），结果都必须相同。

本模块为"派生缓存"提供三类基础设施（全部为纯函数或自包含工具）:

  ① 输入指纹      : md5_file / files_fingerprint
     用途 = 判定"派生缓存是否需要重算"。
     ⚠ 指纹（含 mtime 成分）**只决定是否重算**；因为重算是幂等的，
       mtime 抖动至多多做一次无用功，**绝不改变结果语义**。
       （反例即 2026-09-09 事故：用 mtime 决定**缓存复用**且重算**非幂等**。）

  ② 缓存指纹      : pool_cache_fingerprint
     用途 = 判定"选股池缓存是否可复用"。必须基于**内容**（mv_digest）与
     **一切决定池成员的规则**（filters / size / mode / index / prefixes /
     生效前缀表 eff_prefixes）；绝不得使用文件 mtime。

  ③ 并发安全落盘  : merge_pool_cache + file_lock（调用方配合 os.replace 原子替换）
     派生缓存的"读-改-写"必须在排他锁内完成 → 既不覆盖他人成果，
     也不产生"部分新/部分旧"的混合版本。

调用方（策略）职责:
  · 用 files_fingerprint 判定派生缓存是否需要重算（重算必须幂等）；
  · 用 pool_cache_fingerprint 计算缓存复用指纹（内容寻址）；
  · 落盘时持 file_lock、锁内重新读取并合并、再 os.replace 原子替换。

历史教训（2026-09-09 事故，全族共享同一 total_mv_cache.parquet）:
  用文件 mtime 决定**缓存复用**、且重算**非幂等**（覆盖同日期旧行）→
  同一份配置的两次运行读到不同历史名单（+6.90% vs +6.27%），
  且长跑实验在两次运行之间改写共享文件。
"""

import contextlib
import hashlib
import json
import os
import time
from datetime import datetime

__all__ = [
    'FINGERPRINT_VERSION',
    'FileLockTimeout',
    'file_lock',
    'files_fingerprint',
    'md5_file',
    'merge_pool_cache',
    'pool_cache_fingerprint',
    'rank_pool_codes',
]

_MB = 1 << 20

# 进程内文件摘要缓存: {path: (size, mtime_ns, digest)}
# 键含 size/mtime_ns → 文件在进程内被改写时自动失效重算（不会返回陈旧摘要）。
_DIGEST_MEMO: dict = {}


def md5_file(path: str) -> 'str | None':
    """文件内容 md5（分块读取，进程内按 (path,size,mtime_ns) 记忆化）。

    返回 None 表示文件不存在或不可读。
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = os.path.abspath(path)
    memo = _DIGEST_MEMO.get(key)
    if memo is not None and memo[0] == st.st_size and memo[1] == st.st_mtime_ns:
        return memo[2]
    h = hashlib.md5()
    try:
        with open(path, 'rb') as f:
            while True:
                blk = f.read(8 * _MB)
                if not blk:
                    break
                h.update(blk)
    except OSError:
        return None
    digest = h.hexdigest()
    _DIGEST_MEMO[key] = (st.st_size, st.st_mtime_ns, digest)
    return digest


# --------------------------------------------------------------------------
# 选股池缓存指纹（纯函数，便于单元测试）
# --------------------------------------------------------------------------

# 指纹算法版本：任何对指纹构成的修改都必须 +1
#   —— 否则新旧算法可能产生相同哈希，使"算法变了"被误判为"输入没变"。
#   v1: filters / pool_size / mv_mtime(文件 mtime) / pool_mode / pool_index / pool_prefixes
#   v2: [FIX-DET-2026-09-10] mv_mtime → mv_digest（文件**内容** md5）
#   v3: [R6-E2-2026-09-10]  纳入生效前缀表 eff_prefixes
#       （框架 VALID_PREFIXES 变化 → 旧缓存自动失效重算，避免读到缺新段名单的旧缓存）
FINGERPRINT_VERSION = 3


def pool_cache_fingerprint(*, filters, pool_size, mv_digest,
                           pool_mode, pool_index, pool_prefixes,
                           eff_prefixes=None) -> str:
    """选股池缓存指纹（纯函数）。

    语义：只要"决定池成员的一切"不变，指纹就不变 → 缓存可复用 → 结果可复现。
    关键点：
      · `mv_digest` 必须是市值缓存的【内容】摘要，**不得**使用文件 mtime；
      · `eff_prefixes` 必须是**框架级生效前缀表**（DynamicStockPool.VALID_PREFIXES）,
        它是"哪些代码段进入股票宇宙"的决定性规则之一。
    """
    raw = json.dumps({
        'fp_version': FINGERPRINT_VERSION,
        'filters': list(filters or []),
        'pool_size': pool_size,
        'mv_digest': mv_digest,
        'pool_mode': pool_mode,
        'pool_index': pool_index,
        'pool_prefixes': sorted(pool_prefixes or []),
        'eff_prefixes': (sorted(eff_prefixes) if eff_prefixes is not None else None),
    }, sort_keys=True)
    return hashlib.md5(raw.encode('utf-8')).hexdigest()


# --------------------------------------------------------------------------
# 选股池缓存合并（派生缓存语义）—— 纯函数，便于单元测试
# --------------------------------------------------------------------------

def merge_pool_cache(old_df, new_df, on_conflict='replace'):
    """把新算的池行并入已有缓存。

    设计原则（[FIX-DET7-2026-09-10] 定稿）：
        池缓存是**派生缓存**，其真值是 `pool(D) = f(候选池(D), T-1 市值, 池配置)`。
        ∴ **结果必须由输入决定，而不是由"缓存里恰好存了什么"决定。**

    因此默认 `on_conflict='replace'`：
      · 本次重算到的日期 → 以新行替换（因为重算已经用了**当前输入**）；
      · 未重算到的日期 → 保留；
      · 替换后仍与旧行不同 → 作为**诊断**返回 conflicts（供人工判断是"输入已变化"
        还是"同一输入下结果不同=非确定性"）。

    `on_conflict='keep'`（冻结语义）仅为兼容/降级保留：**会违反上述原则**，不推荐。

    返回 (merged_df, stats, conflicts):
      stats     = {'n_frozen_dates', 'n_append_dates', 'n_replaced_dates', 'n_conflict_dates'}
      conflicts = {date_str: {'n_new', 'n_old', 'n_overlap'}}
    """
    import pandas as pd

    if new_df is None or len(new_df) == 0:
        return old_df, {'n_frozen_dates': 0, 'n_append_dates': 0,
                        'n_replaced_dates': 0, 'n_conflict_dates': 0}, {}

    new_df = new_df.copy()
    new_df['trade_date'] = new_df['trade_date'].astype(str)
    conflicts: dict = {}
    n_frozen = n_append = n_replaced = 0

    if old_df is None or len(old_df) == 0:
        out = new_df.copy()
        n_append = int(out['trade_date'].nunique())
        out = out.sort_values(['trade_date', 'mv_rank']).reset_index(drop=True)
        return out, {'n_frozen_dates': 0, 'n_append_dates': n_append,
                     'n_replaced_dates': 0, 'n_conflict_dates': 0}, conflicts

    old = old_df.copy()
    old['trade_date'] = old['trade_date'].astype(str)
    old_dates = set(old['trade_date'])

    overlap = new_df['trade_date'].isin(old_dates)
    # 诊断：重叠日期上，新行与旧行是否不同
    if overlap.any():
        for d, g in new_df[overlap].groupby('trade_date'):
            new_codes = [c for _, c in sorted(zip(g['mv_rank'], g['ts_code']))]
            og = old[old['trade_date'] == d]
            old_codes = [c for _, c in sorted(zip(og['mv_rank'], og['ts_code']))] if len(og) else []
            if old_codes and new_codes != old_codes:
                conflicts[str(d)] = {
                    'n_new': len(new_codes), 'n_old': len(old_codes),
                    'n_overlap': len(set(new_codes) & set(old_codes)),
                }
    n_replaced = int(new_df[overlap]['trade_date'].nunique())
    n_append = int(new_df[~overlap]['trade_date'].nunique())

    if on_conflict == 'keep':
        # 降级语义：已存在日期冻结不覆盖（会破坏"结果=f(输入)"，不推荐）
        n_frozen = n_replaced
        n_replaced = 0
        # 重叠日期 → 保留**旧行**；非重叠日期 → 追加新行
        body = pd.concat([old, new_df[~overlap]], ignore_index=True) if n_append else old
    else:
        # 默认语义：重算到的日期以新行替换（派生缓存应反映当前输入）
        kept = old[~old['trade_date'].isin(new_df['trade_date'])]
        body = pd.concat([kept, new_df], ignore_index=True)

    body = body.sort_values(['trade_date', 'mv_rank']).reset_index(drop=True)
    return body, {'n_frozen_dates': n_frozen, 'n_append_dates': n_append,
                  'n_replaced_dates': n_replaced, 'n_conflict_dates': len(conflicts)}, conflicts


# --------------------------------------------------------------------------
# 池排序（确定性二级键）—— 纯函数，便于单元测试
# --------------------------------------------------------------------------

def rank_pool_codes(candidates, mv_map, pool_size, mode='smallcap'):
    """按市值排序取前 N 只，**并列时以 ts_code 升序为确定性二级键**。

    返回 (codes, boundary_tie):
      codes        选中的前 N 只 (list)
      boundary_tie None 或 (第N名, 第N+1名, 并列市值) —— 边界并列时给出，供审计

    为什么需要二级键:
      Python 的 `list.sort` 是**稳定排序**，原实现只用市值作 key →
      市值**完全并列**的股票，其相对名次由**输入 `candidates` 的顺序**决定。
      一旦候选集顺序不确定（或候选集组成变化），池的边界成员就会抖动 → 结果不可复现。
      实测该项目确实存在并列：`sz.300114` 与 `sz.302132`（同一公司双代码）
      在 2023-04~2024-04 窗口内**每日一组且全程市值完全相同**。

    注：对"市值降序"（bigcap）用 -mv 作主键，二级键仍为 ts_code 升序
        —— 保证两个方向都**与输入顺序无关**。
    """
    pooled = [c for c in candidates if c in mv_map]
    if not pooled:
        return [], None
    if mode == 'bigcap':
        pooled.sort(key=lambda c: (-mv_map[c], c))
    else:
        pooled.sort(key=lambda c: (mv_map[c], c))
    b = int(pool_size)
    boundary_tie = None
    if b >= 1 and len(pooled) > b and mv_map[pooled[b - 1]] == mv_map[pooled[b]]:
        boundary_tie = (pooled[b - 1], pooled[b], float(mv_map[pooled[b - 1]]))
    return pooled[:b], boundary_tie


# --------------------------------------------------------------------------
# 输入指纹（判定派生缓存是否需要重算）
# --------------------------------------------------------------------------

def files_fingerprint(paths) -> str:
    """一组文件的**输入指纹**：排序后的 (文件名, 字节数, mtime_ns) → md5。

    用途：判定"派生缓存是否需要重算"。
    重要区分（框架设计的关键）：
      · 此指纹【只决定是否重算】，而重算是**幂等的**（缓存 = 输入的函数），
        因此 mtime 抖动至多多做一次无用功，**绝不改变结果语义**；
      · 与之相对，"用 mtime 决定缓存复用从而改变结果"才是缺陷（2026-09-09 事故）。
    返回 32 位十六进制串。
    """
    items = []
    for p in paths:
        try:
            st = os.stat(p)
            items.append((os.path.basename(p), st.st_size, st.st_mtime_ns))
        except OSError:
            items.append((os.path.basename(p), -1, -1))
    items.sort()
    h = hashlib.md5()
    for name, size, mt in items:
        h.update(f'{name}|{size}|{mt}\n'.encode('utf-8'))
    return h.hexdigest()


# --------------------------------------------------------------------------
# 跨进程文件锁（并发写者安全）
# --------------------------------------------------------------------------

class FileLockTimeout(RuntimeError):
    pass


@contextlib.contextmanager
def file_lock(lock_path: str, timeout: float = 120.0, stale_after: float = 900.0,
              poll: float = 0.2):
    """跨进程**排他锁**（原子创建独占锁文件；含陈旧锁自动回收）。

    用途：派生缓存的"读-改-写"必须在锁内完成，避免两个写者互相覆盖，
    也避免写者与读者交错产生混合版本。

    实现要点（业界通用做法，Windows/Linux 均可）:
      · `os.open(..., O_CREAT|O_EXCL)` 原子创建 → 成功者持锁；
      · 锁文件内写入 pid/时间，用于诊断；
      · 超时未获取 → 抛 FileLockTimeout（调用方决定降级或失败）；
      · 陈旧锁（存在时间 > stale_after）自动回收，防止进程被杀后永久死锁。
    """
    deadline = time.time() + timeout
    fd = None
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f'{os.getpid()}|{datetime.now().isoformat()}\n'.encode())
            break
        except FileExistsError:
            # 陈旧锁回收
            try:
                age = time.time() - os.path.getmtime(lock_path)
                if age > stale_after:
                    os.unlink(lock_path)
                    continue
            except OSError:
                pass
            if time.time() >= deadline:
                raise FileLockTimeout(f'获取锁超时: {lock_path}')
            time.sleep(poll)
    try:
        yield
    finally:
        try:
            if fd is not None:
                os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(lock_path)
        except OSError:
            pass
