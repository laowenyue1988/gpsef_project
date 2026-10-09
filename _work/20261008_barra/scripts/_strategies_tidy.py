# -*- coding: utf-8 -*-
"""[TIDY] 策略目录与项目根「杂物」盘点 + 归档（★同卷 rename，不删）。

分类：
  bak_files   : *.bak_* / *.BROKEN_* / *.bak（历史备份，我造成的）
  pycache     : __pycache__/ 目录（可重建缓存）
  empty_dirs  : 空目录
  stray_pyc   : 包【非】__pycache__ 内的 .pyc（★注意：determinism_provenance.pyc 是必需，不在此列）

★硬白名单（绝不触碰）：
  · 所有 *.py
  · strategies/small_alpha191/determinism_provenance.pyc（sourceless 运行模块，必需）
  · config.yaml
  · strategies/ 以外的目录（gpsef/ data/ 等）

用法: python _strategies_tidy.py --dry | --apply
"""
import datetime
import hashlib
import io
import os
import shutil
import sys

ROOT = r'E:\quant\gpsef_project'
os.chdir(ROOT)
TS = datetime.datetime.now().strftime('%Y%m%d_%H%M')
ARCH = r'E:\quant\gpsef_archive_%s_strategies' % TS
DRY = '--dry' in sys.argv

# ★必需文件白名单（绝对不动）
KEEP_FILES = {
    os.path.join('strategies', 'small_alpha191', 'determinism_provenance.pyc'),
}
KEEP_NAMES = {'config.yaml'}


def is_keep(rel):
    if rel.replace('/', os.sep) in {k.replace('/', os.sep) for k in KEEP_FILES}:
        return True
    b = os.path.basename(rel)
    if b in KEEP_NAMES:
        return True
    if b.endswith('.py'):
        return True
    return False


plan = {'bak_files': [], 'pycache': [], 'empty_dirs': [], 'stray_pyc': []}
out = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    out.append(s)
    print(s, flush=True)


P('=' * 96)
P('[TIDY] 策略目录/项目根 杂物盘点  %s  模式=%s' % (datetime.datetime.now().strftime('%F %T'), 'DRY' if DRY else 'APPLY'))
P('=' * 96)

# ── 1) strategies/ 全树（限 3 层，小目录，安全） ─────────────
SROOT = 'strategies'
for dirpath, dirnames, filenames in os.walk(SROOT):
    rel_dir = os.path.relpath(dirpath, '.')
    depth = rel_dir.count(os.sep)
    if depth > 2:
        dirnames[:] = []
        continue
    # __pycache__
    for dn in list(dirnames):
        if dn == '__pycache__':
            plan['pycache'].append(os.path.join(rel_dir, dn))
            dirnames.remove(dn)
    for fn in filenames:
        rel = os.path.join(rel_dir, fn)
        base = fn
        if '.bak_' in base or '.BROKEN_' in base or base.endswith('.bak'):
            if not is_keep(rel):
                plan['bak_files'].append(rel)
        elif base.endswith('.pyc') and not is_keep(rel):
            plan['stray_pyc'].append(rel)

# 空目录（strategies/ 下 2 层内）
for dirpath, dirnames, filenames in os.walk(SROOT, topdown=False):
    rel = os.path.relpath(dirpath, '.')
    if rel == SROOT or rel.count(os.sep) > 2:
        continue
    try:
        if not os.listdir(dirpath):
            plan['empty_dirs'].append(rel)
    except OSError:
        pass

# ── 2) 项目根（复查，排除已知必需） ─────────────────────────
KEEP_ROOT = {
    'run_backtest.py', 'requirements.txt', 'conftest.py', 'setup.py', 'pyproject.toml',
    'pytest.ini', 'setup.cfg', 'data_quality_registry.json', 'download_data.py',
    'rebuild_all_barra.py', 'run_optimize.py', 'run_select.py', 'warmup_cache.py',
    'estimate_ffd_params.py', 'alpha191_gate_fix_unit_test.py',
}
root_extra = []
for fn in sorted(os.listdir('.')):
    if os.path.isdir(fn):
        continue
    if fn in KEEP_ROOT or fn.startswith('_q5bar'):
        continue
    if fn.endswith(('.bak', '.bak_')) or '.bak_' in fn:
        root_extra.append(fn)

# ── 汇总 ─────────────────────────────────────────────────────
P('')
P('[1] 分类结果')
tot_b = 0
for k in ('bak_files', 'pycache', 'empty_dirs', 'stray_pyc'):
    n = len(plan[k])
    sz = 0
    for p in plan[k]:
        if os.path.isfile(p):
            sz += os.path.getsize(p)
    tot_b += sz
    P('    %-12s %3d 项  %s' % (k, n, ('%.2f KB' % (sz / 1024)) if sz else '—'))
P('    根目录 .bak 类 %d 项' % len(root_extra))

P('')
P('[2] 明细（前 25 项）')
shown = 0
for k in ('bak_files', 'pycache', 'stray_pyc', 'empty_dirs'):
    for p in plan[k][:25]:
        if shown >= 25:
            break
        P('    [%-10s] %s' % (k, p))
        shown += 1

P('')
P('[3] ★白名单自证（必须全部存在、且不被搬走）')
for f in sorted(KEEP_FILES):
    P('    %-56s %s' % (f, '存在 ✓' if os.path.exists(f) else '★缺失'))

# 策略包 .py 清单（归档前后须一致）
sdir = os.path.join('strategies', 'small_alpha191')
pys = sorted(f for f in os.listdir(sdir) if f.endswith(('.py', '.pyc')))
P('')
P('[4] strategies/small_alpha191 运行文件清单（%d 项，归档后须逐字节不变）' % len(pys))
for f in pys:
    p = os.path.join(sdir, f)
    P('    %-42s %9d B' % (f, os.path.getsize(p)))
fp = {f: hashlib.md5(open(os.path.join(sdir, f), 'rb').read()).hexdigest() for f in pys}
with io.open('_work/20261008_barra/tmp/strategies_runtime_fp.json', 'w', encoding='utf-8') as fh:
    import json
    json.dump(fp, fh, ensure_ascii=False, indent=1)
P('    → 指纹已写 _work/20261008_barra/tmp/strategies_runtime_fp.json')

if DRY:
    P('')
    P('[DRY] 未做任何改动。加 --apply 执行归档。')
    with io.open('_work/20261008_barra/analysis/tidy_plan.txt', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(out))
    sys.exit(0)

# ── APPLY ────────────────────────────────────────────────────
P('')
P('[5] 执行归档 → %s' % ARCH)
moved, failed = [], []
manifest = ['target,archived_to,size,mtime,category']


def move(rel, cat):
    src = rel
    if not os.path.exists(src):
        return
    dst = os.path.join(ARCH, cat, rel.replace(os.sep, '__'))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        st = os.stat(src)
        shutil.move(src, dst)
        moved.append(rel)
        manifest.append('%s,%s,%d,%d,%s' % (rel, dst, st.st_size, int(st.st_mtime), cat))
    except Exception as e:                                   # noqa: BLE001
        failed.append((rel, str(e)[:80]))
        P('    ⚠ 跳过 %s: %s' % (rel, str(e)[:70]))


for cat in ('bak_files', 'stray_pyc', 'pycache', 'empty_dirs'):
    for rel in plan[cat]:
        move(rel, cat)
for fn in root_extra:
    move(fn, 'root_bak')

with io.open(os.path.join(ARCH, 'MANIFEST.csv'), 'w', encoding='utf-8', newline='\n') as fh:
    fh.write('\n'.join(manifest) + '\n')

P('')
P('[6] 结果：搬走 %d 项，失败 %d 项' % (len(moved), len(failed)))
for r, e in failed[:8]:
    P('    ⚠ %s — %s' % (r, e))

# 后置自证：运行文件指纹必须逐字节不变
P('')
P('[7] ★后置自证：strategies/small_alpha191 运行文件逐字节不变')
import json
ok = True
for f, h in fp.items():
    p = os.path.join(sdir, f)
    if not os.path.exists(p):
        P('    ★丢失 %s' % f)
        ok = False
        continue
    h2 = hashlib.md5(open(p, 'rb').read()).hexdigest()
    if h2 != h:
        P('    ★变化 %s (%s → %s)' % (f, h[:8], h2[:8]))
        ok = False
P('    ⇒ %s（%d 项）' % ('✅ 全部逐字节不变' if ok else '❌ 有变化', len(fp)))

P('')
P('[done] 归档目录 = %s' % ARCH)
with io.open('_work/20261008_barra/analysis/tidy_result.txt', 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(out))
