# -*- coding: utf-8 -*-
"""[TIDY2] 归档 strategies/ 下的历史 backup/调试 目录（★同卷 move，可还原）。

范围（**只归档目录，绝不碰任何 .py / config.yaml / data/**）：
  · strategies/*/backup*        （历史代码快照目录）
  · strategies/wyckoff20/ols_debug （调试目录）

用法: python _strategies_tidy2.py --dry | --apply
"""
import datetime
import io
import os
import shutil
import sys

ROOT = r'E:\quant\gpsef_project'
os.chdir(ROOT)
TS = datetime.datetime.now().strftime('%Y%m%d_%H%M')
ARCH = r'E:\quant\gpsef_archive_%s_strategies_backupdirs' % TS
DRY = '--dry' in sys.argv

# ★目录名模式（只匹配这些）
PATTERNS = ('backup', 'ols_debug')

targets = []
for pkg in sorted(os.listdir('strategies')):
    pd = os.path.join('strategies', pkg)
    if not os.path.isdir(pd):
        continue
    for d in sorted(os.listdir(pd)):
        full = os.path.join(pd, d)
        if not os.path.isdir(full):
            continue
        low = d.lower()
        if any(low.startswith(p) or p in low for p in PATTERNS):
            targets.append(full)

out = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    out.append(s)
    print(s, flush=True)


def du(p):
    t = 0
    n = 0
    for dp, dn, fn in os.walk(p):
        for f in fn:
            try:
                t += os.path.getsize(os.path.join(dp, f))
                n += 1
            except OSError:
                pass
    return t, n


P('=' * 96)
P('[TIDY2] 归档 strategies/ 历史备份目录  %s  模式=%s' % (datetime.datetime.now().strftime('%F %T'), 'DRY' if DRY else 'APPLY'))
P('=' * 96)
P('')
P('[1] 待归档目录（%d 个）' % len(targets))
P('%-52s %8s %6s' % ('目录', '体积', '文件数'))
P('-' * 72)
tot_sz = tot_n = 0
for t in targets:
    s, n = du(t)
    tot_sz += s
    tot_n += n
    P('%-52s %7.2f MB %6d' % (t, s / 1024 / 1024, n))
P('-' * 72)
P('%-52s %7.2f MB %6d' % ('合计', tot_sz / 1024 / 1024, tot_n))

P('')
P('[2] ★白名单自证（这些必须仍在）')
for f in ('strategies/wyckoff20/barra_constraint.py',
          'strategies/wyckoff20/barra_builder.py',
          'strategies/small_alpha191/strategy.py',
          'strategies/small_alpha191/determinism_provenance.pyc',
          'strategies/alpha191/config.yaml'):
    P('    %-58s %s' % (f, '存在 ✓' if os.path.exists(f) else '★缺失'))

if DRY:
    P('')
    P('[DRY] 未做任何改动。加 --apply 执行。')
    with io.open('_work/20261008_barra/analysis/tidy2_plan.txt', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(out))
    sys.exit(0)

P('')
P('[3] 执行归档 → %s' % ARCH)
os.makedirs(ARCH, exist_ok=True)
moved, failed = [], []
man = ['target,archived_to,files,bytes']
for t in targets:
    dst = os.path.join(ARCH, t.replace(os.sep, '__'))
    s, n = du(t)
    try:
        shutil.move(t, dst)
        moved.append(t)
        man.append('%s,%s,%d,%d' % (t, dst, n, s))
    except Exception as e:                                   # noqa: BLE001
        failed.append((t, str(e)[:90]))
        P('    ⚠ 跳过 %s: %s' % (t, str(e)[:80]))

with io.open(os.path.join(ARCH, 'MANIFEST.csv'), 'w', encoding='utf-8', newline='\n') as fh:
    fh.write('\n'.join(man) + '\n')

P('')
P('[4] 结果：搬走 %d 个目录，失败 %d' % (len(moved), len(failed)))
for t, e in failed:
    P('    ⚠ %s — %s' % (t, e))

P('')
P('[5] ★后置自证：白名单文件仍在')
ok = True
for f in ('strategies/wyckoff20/barra_constraint.py',
          'strategies/small_alpha191/strategy.py',
          'strategies/small_alpha191/determinism_provenance.pyc'):
    e = os.path.exists(f)
    P('    %-58s %s' % (f, '✓' if e else '★丢失'))
    ok &= e
P('    ⇒ %s' % ('✅ 全部在' if ok else '❌ 有丢失'))

P('')
P('[6] 归档后 strategies/ 结构')
for pkg in sorted(os.listdir('strategies')):
    pd = os.path.join('strategies', pkg)
    if os.path.isdir(pd):
        P('    %-28s %3d 项' % (pkg + '/', len(os.listdir(pd))))

P('')
P('[done] 归档 = %s' % ARCH)
with io.open('_work/20261008_barra/analysis/tidy2_result.txt', 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(out))
