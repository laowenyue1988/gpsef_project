# -*- coding: utf-8 -*-
"""[Q5BAR] 搭台：生成 3 份臂配置 + 启动器（不跑回测）。

  Z1  = 零影响验证臂：新代码 + 采纳配置（无 risk_budget 键 ⇒ mode='off'）
        ⇒ 与 D-342 的 A 臂产物逐字节比对（若相同 ⇒ 零影响闭合）
  B1  = 真实暴露臂：risk_budget {factors:[volatility,beta], strength:0.30}
  S1  = sham 臂（R-146）：同 B1 + shuffle_seed ⇒ 暴露同分布随机置换

窗口 = 标准烟测窗 20240401~20240801（5 轮，全链重训）
★三配置除 barra.risk_budget 外**逐键相同**（断言）
★启动器预置 7 个确定性环境变量（R-202：绕开 run_backtest.py 的自我 exec 重启）
"""
import datetime
import hashlib
import io
import json
import os
import sys

import yaml

PROJ = r'E:\quant\gpsef_project'
os.chdir(PROJ)
ANCHOR = 'results/SmallAlpha191Test3_Q5R_OPT_dump_full_20260923/config.yaml'
WD = '_work/20261008_barra'
OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def flat(o, p='', out=None):
    if out is None:
        out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            flat(v, '%s%s.' % (p, k), out)
    elif isinstance(o, list):
        out[p[:-1]] = json.dumps(o, sort_keys=True, ensure_ascii=False)
    else:
        out[p[:-1]] = o
    return out


P('=' * 100)
P('[Q5BAR] 搭台  %s' % (datetime.datetime.now().strftime('%F %T'),))
P('=' * 100)

# ── 1) 基线配置 ──────────────────────────────────────────────
base = yaml.safe_load(io.open(ANCHOR, encoding='utf-8'))
P('[1] 基线 = %s' % ANCHOR)
P('    顶层键 %d | strategy.class=%s | barra.enabled=%s'
  % (len(base), base['strategy']['class'], base['barra'].get('enabled')))
P('    experiment 段（剔除前） = %s' % json.dumps(base.get('experiment'), ensure_ascii=False))

# ── 2) 三份配置 ──────────────────────────────────────────────
RB = {
    'mode': 'risk_budget',
    'factors': ['volatility', 'beta'],
    'strength': 0.30,
    'exclude_quantile': 0.0,
    'min_coverage': 0.80,
    'shuffle_seed': None,
    'snapshot_dir': None,
}
ARMS = {
    'Z1': dict(rb=None,               desc='零影响验证（mode=off）'),
    'B1': dict(rb=dict(RB),           desc='真实暴露 λ=0.30'),
    'S1': dict(rb=dict(RB, shuffle_seed=20261008), desc='sham：暴露同分布置换'),
}
P('')
P('[2] 生成臂配置')
cfgs = {}
for k, spec in ARMS.items():
    c = yaml.safe_load(io.open(ANCHOR, encoding='utf-8'))
    c['strategy']['class'] = 'strategies.small_alpha191.strategy.SmallAlpha191Strategy'
    c['backtest']['start_date'] = '20240401'
    c['backtest']['end_date'] = '20240801'
    # ★剔除可能造成缓存复用的键（与 D-342 一致）
    _e = c.get('experiment') or {}
    _e.pop('preds_mode', None)
    _e.pop('cache_dir', None)
    c['experiment'] = _e
    if spec['rb'] is not None:
        c['barra']['risk_budget'] = spec['rb']
    else:
        c['barra'].pop('risk_budget', None)
    p = '%s/configs/config_%s.yaml' % (WD, k)
    with io.open(p, 'w', encoding='utf-8', newline='\n') as f:
        yaml.safe_dump(c, f, allow_unicode=True, sort_keys=False, width=200)
    cfgs[k] = p
    P('    %-3s %-42s %s' % (k, p, spec['desc']))

# ── 3) 断言：除 barra.risk_budget 外逐键相同 ────────────────
P('')
P('[3] 配置互比（差异键必须**只**在 barra.risk_budget 内）')
fz = flat(yaml.safe_load(io.open(cfgs['Z1'], encoding='utf-8')))
for k in ('B1', 'S1'):
    fk = flat(yaml.safe_load(io.open(cfgs[k], encoding='utf-8')))
    d = sorted(x for x in set(fz) | set(fk) if fz.get(x) != fk.get(x))
    bad = [x for x in d if not x.startswith('barra.risk_budget')]
    P('    Z1 vs %s: 差异 %d 键 ⇒ %s' % (k, len(d), d))
    P('        非 barra.risk_budget 的差异 = %s  %s'
      % (bad if bad else '无', '✅' if not bad else '❌ FAIL'))
    if bad:
        raise SystemExit('配置纯度断言失败')

# ── 4) 启动器（R-202：预置 7 个变量） ──────────────────────
lines = [
    '#!/usr/bin/env bash',
    'set -u',
    '# ★R-202：必须预置 7 个确定性环境变量，绕开 run_backtest.py:27-30 的自我 exec 重启',
    'export TEMP=E:/temp TMP=E:/temp TMPDIR=E:/temp',
    'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
    'export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1',
    'export PYTHONHASHSEED=0',
    'cd /e/quant/gpsef_project',
    'PY=.venv/Scripts/python.exe',
    'WD=_work/20261008_barra',
    'mkdir -p "$WD/logs" "$WD/tmp"',
    'echo "=== Q5BAR batch start $(date +%F_%T) ==="',
    'PIDS=()',
]
for k in ('Z1', 'B1', 'S1'):
    lines += [
        'rm -rf "results/Q5BAR_%s_20261008"' % k,
        '("$PY" -u run_backtest.py --config "%s/configs/config_%s.yaml" '
        '--output "results/Q5BAR_%s_20261008" 2>&1 '
        '| tee "$WD/logs/arm_%s.log") &' % (WD, k, k, k),
        'PIDS+=($!)',
        'echo "  ARM %s pid=${PIDS[-1]}"' % k,
    ]
lines += [
    'for p in "${PIDS[@]}"; do wait "$p"; echo "  waited $p rc=$?"; done',
    'echo "=== Q5BAR batch end $(date +%F_%T) ==="',
]
launcher = '%s/run_smoke.sh' % WD
with io.open(launcher, 'w', encoding='utf-8', newline='\n') as f:
    f.write('\n'.join(lines) + '\n')
P('')
P('[4] 启动器 %s（三臂并行；预置 7 环境变量；逐臂 tee 到 %s/logs/）' % (launcher, WD))

# ── 5) 落地指纹 ─────────────────────────────────────────────
P('')
P('[5] 代码指纹')
for f in ('strategies/small_alpha191/strategy.py',):
    P('    %-44s md5=%s  %s B' % (f, md5(f)[:16], os.path.getsize(f)))

with io.open('%s/analysis/q5bar_setup.txt' % WD, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(OUT))
P('')
P('[done] %s/analysis/q5bar_setup.txt' % WD)
