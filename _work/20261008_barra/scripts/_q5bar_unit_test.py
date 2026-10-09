# -*- coding: utf-8 -*-
"""[Q5BAR] 单元测试：`_apply_barra_risk_budget` 与 `_load_barra_expo_pit`。

覆盖纪律：
  · R-127 行为级断言（不是"函数被调用"，而是"输出确实按预期变化"）
  · R-132 判别力自证：每条断言都要能"故意弄错就失败"
  · R-196/R-197 逐键行为可达探针（每个配置键单独回退 ⇒ 输出必须改变）
  · R-168 口诀：该是常数时若出现散布/变化 ⇒ 先怀疑自己
  · 反自证：结构退化 ⇒ 必须"数值不动但不返回 None"

用法: python _q5bar_unit_test.py
退出码: 0 = 全绿; 1 = 有失败
"""
import glob
import io
import os
import sys
from importlib import import_module

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))

STRAT = import_module('strategies.small_alpha191.strategy')
CLS = getattr(STRAT, 'SmallAlpha191Strategy')
F = CLS._apply_barra_risk_budget          # staticmethod，直接取

PASS, FAIL = [], []


def ck(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print('  [%s] %-58s %s' % ('PASS' if cond else 'FAIL', name, detail))


def banner(t):
    print('\n' + '=' * 78)
    print(t)
    print('=' * 78)


# ────────────────────────────────────────────────────────────────
# 夹具：从**真实快照**取 60 只（只读）
# ────────────────────────────────────────────────────────────────
SNAP_DIR = 'data/barra_exposures'
_snap_files = sorted(glob.glob(os.path.join(SNAP_DIR, 'barra_expo_*.parquet')))
assert _snap_files, '找不到 BARRA 快照'
EXPO_FULL = pd.read_parquet(_snap_files[len(_snap_files) // 2])
SUB = EXPO_FULL.head(60).copy()
CODES = ['sh.' + str(c).zfill(6) if str(c)[0] in '69' else 'sz.' + str(c).zfill(6)
         for c in SUB['code'].astype(str)]
N = len(CODES)

FACT = ['volatility', 'beta']
LAM = 0.30

# 基准调用（真实暴露，λ=0.3）
SC0 = np.linspace(0.9, 0.1, N)             # 递减分数：score 与"顺序"解耦，便于观察
BASE_NEW, BASE_INFO = F(CODES, SC0, SUB, FACT, LAM)
assert BASE_NEW is not None, '基准调用应生效: %s' % BASE_INFO

# 真实风险分（用于构造"高风险⇒高分"场景）
R_REAL = None


def _risk_of(scores, expo=SUB, factors=FACT, lam=LAM, **kw):
    new, info = F(CODES, scores, expo, factors, lam, **kw)
    return new, info


# 反解 R（用 λ→小 的两次调用做差分不可靠；改为直接用暴露自算稳健 z）
def _robust_R(expo, factors):
    cols = [f + '_exp' for f in factors]
    X = expo[cols].to_numpy(dtype=float)
    Z = np.zeros_like(X)
    for j in range(X.shape[1]):
        xj = X[:, j]
        m = np.isfinite(xj)
        med = np.median(xj[m])
        mad = np.median(np.abs(xj[m] - med))
        s = 1.4826 * mad
        if s <= 0:
            s = np.mean(np.abs(xj[m] - med)) or 1.0
        Z[:, j] = np.where(m, (xj - med) / s, 0.0)
    return Z.sum(axis=1)


R_REAL = _robust_R(SUB, FACT)

# ────────────────────────────────────────────────────────────────
banner('T1 · 零影响闭合（任一即返回 None）')
# ────────────────────────────────────────────────────────────────
for nm, kw in [
    ('lam=0', dict(lam=0.0)),
    ('lam<0', dict(lam=-0.5)),
]:
    new, info = F(CODES, SC0, SUB, FACT, kw['lam'])
    ck('T1.a %s ⇒ None' % nm, new is None, str(info.get('reason')))

new, info = F(CODES, SC0, None, FACT, LAM)
ck('T1.b exposures=None ⇒ None', new is None, str(info.get('reason')))

new, info = F(CODES, SC0, pd.DataFrame(), FACT, LAM)
ck('T1.c 空暴露 ⇒ None', new is None, str(info.get('reason')))

new, info = F(CODES, SC0, SUB, ['不存在的因子'], LAM)
ck('T1.d 因子列全缺 ⇒ None', new is None, str(info.get('reason')))

new, info = F(CODES, SC0, SUB, FACT, LAM, min_coverage=2.0)
ck('T1.e 覆盖门槛不可达 ⇒ None', new is None, str(info.get('reason')))

new, info = F([], [], SUB, FACT, LAM)
ck('T1.f 空候选 ⇒ None', new is None, str(info.get('reason')))

new, info = F(CODES, SC0[:10], SUB, FACT, LAM)
ck('T1.g 长度不一致 ⇒ None', new is None, str(info.get('reason')))

# ────────────────────────────────────────────────────────────────
banner('T2 · 行为级可达：λ>0 ⇒ 输出必须真的变')
# ────────────────────────────────────────────────────────────────
ck('T2.a 基准调用返回非 None', BASE_NEW is not None)
ck('T2.b 长度守恒', len(BASE_NEW) == N, '%d vs %d' % (len(BASE_NEW), N))
ck('T2.c 分数确实被改变（score_changed>0）', BASE_INFO['score_changed'] > 0,
   'changed=%d' % BASE_INFO['score_changed'])
_pen = np.asarray(BASE_NEW) / SC0
ck('T2.d 乘子 ∈ (0,1]', bool(np.all(_pen > 0) and np.all(_pen <= 1 + 1e-12)),
   '[%.4f,%.4f]' % (_pen.min(), _pen.max()))
ck('T2.e R<=0 的乘子恒为 1（只罚高风险侧）',
   bool(np.allclose(_pen[R_REAL <= 0], 1.0)),
   'n(R<=0)=%d, max|dev|=%.2e' % (int((R_REAL <= 0).sum()),
                                  float(np.max(np.abs(_pen[R_REAL <= 0] - 1.0))) if (R_REAL <= 0).any() else 0.0))

# 方向性断言：高风险股票排名必须下降
_sc_high = np.asarray(BASE_NEW)
_ord_before = np.argsort(SC0)[::-1]
_ord_after = np.argsort(_sc_high)[::-1]
_top_before = set(_ord_before[:20])
_top_after = set(_ord_after[:20])
_r_in_before = float(np.mean(R_REAL[list(_top_before)]))
_r_in_after = float(np.mean(R_REAL[list(_top_after)]))
ck('T2.f Top-20 平均风险分下降（机制方向正确）', _r_in_after < _r_in_before,
   '%.4f → %.4f (Δ=%.4f)' % (_r_in_before, _r_in_after, _r_in_after - _r_in_before))
ck('T2.g Top-20 集合确实改变', len(_top_before ^ _top_after) > 0,
   '对称差=%d' % len(_top_before ^ _top_after))

# ────────────────────────────────────────────────────────────────
banner('T3 · 反自证（判别力）—— 每条都必须"故意弄错就失败"')
# ────────────────────────────────────────────────────────────────
# T3.a 常数分数 ⇒ 降权后分数必须**解析式可预测** = exp(−λ·max(0,R))，方向判别力最强
CONST = np.ones(N)
new_c, info_c = F(CODES, CONST, SUB, FACT, LAM)
_expect_c = np.exp(-LAM * np.maximum(0.0, R_REAL))
ck('T3.a 常数分数 ⇒ 输出 == exp(−λ·max(0,R))（解析判别力）',
   bool(new_c is not None and np.allclose(np.asarray(new_c), _expect_c, rtol=1e-12, atol=1e-15)),
   'max|Δ|=%.3e' % (float(np.max(np.abs(np.asarray(new_c) - _expect_c))) if new_c else float('nan')))
# 方向：按 R 升序排列后，分数必须**单调不增**（允许平台 —— R≤0 段 penalty 恒 1）
_order = np.argsort(R_REAL, kind='stable')
_out_sorted = np.asarray(new_c)[_order]
ck('T3.a 分数随 R 单调不增（允许平台段）',
   bool(np.all(np.diff(_out_sorted) <= 1e-15)),
   'min(Δ)=%.3e (R 升序)' % float(np.min(np.diff(_out_sorted))))
ck('T3.a R≤0 段分数恒等（平台且 =1）',
   bool(np.allclose(_out_sorted[R_REAL[_order] <= 0], 1.0, rtol=0, atol=0)),
   'n=%d' % int((R_REAL <= 0).sum()))

# T3.b 暴露结构退化（全同值 ⇒ MAD=0 ⇒ 稳健 z 无定义）⇒ 判"不可计算/不生效"（返回 None）
#     ★语义选择：MAD=0 时 z 无定义 ⇒ 返回 None（诚实"不生效"）优于硬凑一个乘子。
#     ★关键断言：**两条路径对最终排序分数的效果完全等价**（都 ⇒ 不改分数）。
DEGEN = SUB.copy()
for f in FACT:
    DEGEN[f + '_exp'] = 0.0
new_d, info_d = F(CODES, SC0, DEGEN, FACT, LAM)
ck('T3.b 暴露全同值 ⇒ None（稳健 z 无定义，诚实不生效）',
   new_d is None, str(info_d.get('reason')))
ck('T3.b 退化 reason 正确', info_d.get('reason') == 'all_factor_cols_degenerate',
   str(info_d.get('reason')))
# 等价性：None ⇒ 调用方保持原分数 ⇒ 与"乘子全 1"效果逐位相同
_eff_degen = np.asarray(SC0) if new_d is None else np.asarray(new_d)
ck('T3.b 退化路径 ⇒ 最终排序分数与输入逐位相同（该是常数时不动）',
   bool(np.allclose(_eff_degen, SC0, rtol=0, atol=0)),
   'max|Δ|=%.3e' % float(np.max(np.abs(_eff_degen - SC0))))
# 反自证：若把"退化"误判成"生效且乘子≠1"，本断言必须失败
ck('T3.b 反自证：乘子若≠1 则本组断言会失败（判别力）',
   bool(new_d is None or np.allclose(np.asarray(new_d) / SC0, 1.0)))

# T3.c sham 判别力：置换 seed ⇒ 输出**必须不同**
new_s1, info_s1 = F(CODES, SC0, SUB, FACT, LAM, 0.0, 12345)
new_s2, info_s2 = F(CODES, SC0, SUB, FACT, LAM, 0.0, 54321)
ck('T3.c shuffle 与 No-shuffle 输出不同（sham 有判别力）',
   bool(not np.allclose(np.asarray(new_s1), np.asarray(BASE_NEW))),
   'max|Δ|=%.6f' % float(np.max(np.abs(np.asarray(new_s1) - np.asarray(BASE_NEW)))))
ck('T3.c 不同 seed ⇒ 输出不同', bool(not np.allclose(np.asarray(new_s1), np.asarray(new_s2))),
   'max|Δ|=%.6f' % float(np.max(np.abs(np.asarray(new_s1) - np.asarray(new_s2)))))
ck('T3.c sham 保留惩罚多重集（同分布置换）',
   bool(np.allclose(np.sort(np.asarray(new_s1) / SC0), np.sort(np.asarray(BASE_NEW) / SC0))),
   'sort(max|Δ|)=%.3e' % float(np.max(np.abs(np.sort(np.asarray(new_s1) / SC0)
                                              - np.sort(np.asarray(BASE_NEW) / SC0)))))

# ────────────────────────────────────────────────────────────────
banner('T4 · 剔除档（exclude_quantile）')
# ────────────────────────────────────────────────────────────────
for eq in (0.10, 0.25):
    new_e, info_e = F(CODES, SC0, SUB, FACT, LAM, eq)
    k_expect = int(np.floor(eq * N))
    n_inf = int(np.sum(~np.isfinite(np.asarray(new_e))))
    ck('T4 eq=%.2f ⇒ 恰 %d 只被剔除' % (eq, k_expect), n_inf == k_expect,
       'actual=%d' % n_inf)
    ck('T4 eq=%.2f ⇒ 被剔除的正是 R 最高者' % eq,
       bool(np.all(R_REAL[~np.isfinite(np.asarray(new_e))] >=
                   np.partition(R_REAL, N - k_expect)[N - k_expect] - 1e-12)),
       'min R(excluded)=%.4f' % float(np.min(R_REAL[~np.isfinite(np.asarray(new_e))])))
new_e0, _ = F(CODES, SC0, SUB, FACT, LAM, 0.0)
ck('T4 eq=0 ⇒ 无剔除', bool(np.all(np.isfinite(np.asarray(new_e0)))))

# ────────────────────────────────────────────────────────────────
banner('T5 · R-197 逐键行为可达探针（每键单独回退 ⇒ 输出必须改变）')
# ────────────────────────────────────────────────────────────────
_kw_full = dict(lam=0.30, exclude_quantile=0.10, shuffle_seed=777, min_coverage=0.80)
_kw_full.pop('shuffle_seed')                        # shuffle 单独另测
_REF, _ = F(CODES, SC0, SUB, FACT, **_kw_full)
_probe = [
    ('factors: [volatility,beta] → [volatility]',
     lambda: F(CODES, SC0, SUB, ['volatility'], **_kw_full)[0]),
    ('strength: 0.30 → 0.60',
     lambda: F(CODES, SC0, SUB, FACT, lam=0.60, exclude_quantile=0.10, min_coverage=0.80)[0]),
    ('exclude_quantile: 0.10 → 0.00',
     lambda: F(CODES, SC0, SUB, FACT, lam=0.30, exclude_quantile=0.00, min_coverage=0.80)[0]),
    ('shuffle_seed: None → 777',
     lambda: F(CODES, SC0, SUB, FACT, 0.30, 0.10, 777, 0.80)[0]),
]
for nm, fn in _probe:
    out = fn()
    changed = (out is None) or (not np.allclose(np.asarray(out), np.asarray(_REF)))
    _e = 'n/a'
    if out is not None:
        _a, _b = np.asarray(out), np.asarray(_REF)
        _m = np.isfinite(_a) & np.isfinite(_b)
        _mx = ('%.6f' % float(np.max(np.abs(_a[_m] - _b[_m])))) if _m.any() else '0'
        _infdiff = int(np.sum(np.isfinite(_a) != np.isfinite(_b)))
        _e = 'max|Δ|=%-9s 剔除集差异=%d' % (_mx, _infdiff)
    ck('T5 回退「%s」⇒ 输出改变' % nm, bool(changed), _e)

# 剂量单调（R-181：剂量非退化）
seq = []
for lam in (0.0, 0.15, 0.30, 0.60, 1.20):
    o, _ = F(CODES, SC0, SUB, FACT, lam)
    seq.append(None if o is None else float(np.mean(np.asarray(o) / SC0)))
ck('T5.b 剂量 λ ↑ ⇒ 平均乘子单调 ↓',
   all((seq[i] is None) or (seq[i + 1] is None) or (seq[i + 1] < seq[i] - 1e-9)
       for i in range(len(seq) - 1)),
   ' '.join('None' if x is None else '%.4f' % x for x in seq))

# 结构比例恒定（R-181④）：λ 变化不改变"谁被剔除"
new_l1, _ = F(CODES, SC0, SUB, FACT, 0.15, 0.10)
new_l2, _ = F(CODES, SC0, SUB, FACT, 1.20, 0.10)
ck('T5.c 剔除集合与 λ 无关（结构比例恒定）',
   bool(np.array_equal(~np.isfinite(np.asarray(new_l1)), ~np.isfinite(np.asarray(new_l2)))))

# ────────────────────────────────────────────────────────────────
banner('T6 · `_load_barra_expo_pit`：PIT 语义 + 只读 + 缓存')
# ────────────────────────────────────────────────────────────────
try:
    obj = object.__new__(CLS)
    obj._barra_rb_cfg = {}
    obj.config = {'barra': {'snapshot_dir': 'data/barra_exposures'}}
    obj._barra_rb_expo_cache = {}
    obj._barra_rb_used = set()

    _before = sorted(os.listdir(SNAP_DIR))
    d1 = obj._load_barra_expo_pit(pd.Timestamp('2024-04-10'))
    _after = sorted(os.listdir(SNAP_DIR))
    ck('T6.a 返回非空快照', d1 is not None and len(d1) > 100,
       'shape=%s' % (None if d1 is None else str(d1.shape),))
    ck('T6.b 目录清单前后一致（绝不写盘）', _before == _after,
       'before=%d after=%d' % (len(_before), len(_after)))
    # PIT：选中日期必须严格 < 调仓日（★用 basename 日期串比较，避开 Windows 路径分隔符陷阱）
    _dates_avail = sorted(os.path.basename(f)[len('barra_expo_'):-len('.parquet')]
                          for f in _snap_files)
    _expect_date = max(d for d in _dates_avail if d < '20240410')
    ck('T6.c 选中严格 < 日期的最近一份',
       bool(obj._barra_rb_used and list(obj._barra_rb_used)[0] == _expect_date),
       'used=%s expect=%s' % (sorted(obj._barra_rb_used), _expect_date))
    d2 = obj._load_barra_expo_pit(pd.Timestamp('2024-04-10'))
    ck('T6.d 同日期缓存命中（同一对象）', d2 is d1)
    ck('T6.e 缓存条目数=1', len(obj._barra_rb_expo_cache) == 1)
    d0 = obj._load_barra_expo_pit(pd.Timestamp('2021-04-01'))
    ck('T6.f 早于全部快照 ⇒ None', d0 is None)
except Exception as _e:
    ck('T6 系列执行未抛异常', False, repr(_e))

# ────────────────────────────────────────────────────────────────
banner('总判定')
print('PASS = %d   FAIL = %d' % (len(PASS), len(FAIL)))
if FAIL:
    print('失败项:')
    for f in FAIL:
        print('   ✗ %s' % f)
    print('\n❌ FAIL')
    sys.exit(1)
print('\n✅ ALL PASS —— 单元测试全绿')
sys.exit(0)
