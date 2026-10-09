# -*- coding: utf-8 -*-
"""[Q5BAR] 长窗搭台：生成 B1/S1 全窗配置 + 启动器（烟测过闸后执行）。

  B1_full = 真实暴露（factors=[volatility,beta], λ=0.30）
  S1_full = sham（同 B1 + shuffle_seed）
窗口 = 全窗 20210401~20260901（与采纳包锚**逐字一致**）
★不跑 Z1_full：零影响已由烟测（Z1 vs D-342 A 臂逐字节）证明；锚产物即 mode=off 的长窗等价物
"""
import datetime
import hashlib
import io
import json
import os

import yaml

os.chdir(r'E:\quant\gpsef_project')
ANCHOR = 'results/SmallAlpha191Test3_Q5R_OPT_dump_full_20260923/config.yaml'
WD = '_work/20261008_barra'
OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)


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
P('[Q5BAR] 长窗搭台  %s' % (datetime.datetime.now().strftime('%F %T'),))
P('=' * 100)

anc = yaml.safe_load(io.open(ANCHOR, encoding='utf-8'))
W0 = anc['backtest']['start_date']
W1 = anc['backtest']['end_date']
P('[1] 锚窗口 = %s ~ %s（逐字沿用）' % (W0, W1))

RB = {'mode': 'risk_budget', 'factors': ['volatility', 'beta'], 'strength': 0.30,
      'exclude_quantile': 0.0, 'min_coverage': 0.80, 'shuffle_seed': None,
      'snapshot_dir': None}
ARMS = {'B1': dict(RB), 'S1': dict(RB, shuffle_seed=20261008)}

P('')
P('[2] 生成全窗臂配置')
cfgs = {}
for k, rb in ARMS.items():
    c = yaml.safe_load(io.open(ANCHOR, encoding='utf-8'))
    c['strategy']['class'] = 'strategies.small_alpha191.strategy.SmallAlpha191Strategy'
    c['backtest']['start_date'] = W0
    c['backtest']['end_date'] = W1
    _e = c.get('experiment') or {}
    _e.pop('preds_mode', None)
    _e.pop('cache_dir', None)
    c['experiment'] = _e
    c['barra']['risk_budget'] = rb
    p = '%s/configs/config_%s_full.yaml' % (WD, k)
    with io.open(p, 'w', encoding='utf-8', newline='\n') as f:
        yaml.safe_dump(c, f, allow_unicode=True, sort_keys=False, width=200)
    cfgs[k] = p
    P('    %-3s %-46s λ=%.2f shuffle=%s' % (k, p, rb['strength'], rb['shuffle_seed']))

# 纯度：两配置差异只能在 barra.risk_budget 内
fb = flat(yaml.safe_load(io.open(cfgs['B1'], encoding='utf-8')))
fs = flat(yaml.safe_load(io.open(cfgs['S1'], encoding='utf-8')))
d = sorted(x for x in set(fb) | set(fs) if fb.get(x) != fs.get(x))
bad = [x for x in d if not x.startswith('barra.risk_budget')]
P('')
P('[3] 纯度断言 B1 vs S1: 差异 %d 键 ⇒ %s' % (len(d), d))
P('    非 barra.risk_budget 差异 = %s  %s' % (bad if bad else '无', '✅' if not bad else '❌'))
if bad:
    raise SystemExit('纯度断言失败')

# 启动器
lines = [
    '#!/usr/bin/env bash',
    'set -u',
    '# ★R-202：预置 7 个确定性环境变量（绕开 run_backtest.py 自我 exec 重启）',
    'export TEMP=E:/temp TMP=E:/temp TMPDIR=E:/temp',
    'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
    'export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1',
    'export PYTHONHASHSEED=0',
    'cd /e/quant/gpsef_project',
    'PY=.venv/Scripts/python.exe',
    'WD=_work/20261008_barra',
    'mkdir -p "$WD/logs" "$WD/tmp"',
    'echo "=== Q5BAR FULL batch start $(date +%F_%T) ==="',
    'PIDS=()',
]
for k in ('B1', 'S1'):
    lines += [
        'rm -rf "results/Q5BAR_%s_full_20261008"' % k,
        '("$PY" -u run_backtest.py --config "%s/configs/config_%s_full.yaml" '
        '--output "results/Q5BAR_%s_full_20261008" 2>&1 '
        '| tee "$WD/logs/full_%s.log") &' % (WD, k, k, k),
        'PIDS+=($!)',
        'echo "  FULL ARM %s pid=${PIDS[-1]}"' % k,
    ]
lines += [
    'for p in "${PIDS[@]}"; do wait "$p"; echo "  waited $p rc=$?"; done',
    'echo "=== Q5BAR FULL batch end $(date +%F_%T) ==="',
]
L = '%s/run_full.sh' % WD
with io.open(L, 'w', encoding='utf-8', newline='\n') as f:
    f.write('\n'.join(lines) + '\n')
P('')
P('[4] 长窗启动器 = %s（2 臂并行；日志落 %s/logs/）' % (L, WD))

with io.open('%s/analysis/q5bar_setup_full.txt' % WD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(OUT))
P('')
P('[done] %s/analysis/q5bar_setup_full.txt' % WD)
