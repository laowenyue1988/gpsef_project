# -*- coding: utf-8 -*-
"""[Q5BAR] 烟测 → 长窗：生成 B1/S1 全窗配置 + 启动器（★含 gate0 预检）。

窗口 = 全窗 20210401~20260901（与采纳包锚**逐字一致**）
"""
import datetime
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


P('=' * 96)
P('[Q5BAR] 长窗搭台  %s' % (datetime.datetime.now().strftime('%F %T'),))
P('=' * 96)

anc = yaml.safe_load(io.open(ANCHOR, encoding='utf-8'))
W0, W1 = anc['backtest']['start_date'], anc['backtest']['end_date']
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

fb = flat(yaml.safe_load(io.open(cfgs['B1'], encoding='utf-8')))
fs = flat(yaml.safe_load(io.open(cfgs['S1'], encoding='utf-8')))
d = sorted(x for x in set(fb) | set(fs) if fb.get(x) != fs.get(x))
bad = [x for x in d if not x.startswith('barra.risk_budget')]
P('')
P('[3] 纯度断言 B1 vs S1: 差异 %d 键 ⇒ %s' % (len(d), d))
P('    非 barra.risk_budget 差异 = %s  %s' % (bad if bad else '无', '✅' if not bad else '❌'))
if bad:
    raise SystemExit('纯度断言失败')

# 启动器：2 长窗 + 1 烟测（V1）⇒ 3 槽
lines = [
    '#!/usr/bin/env bash',
    'set -u',
    'export TEMP=E:/temp TMP=E:/temp TMPDIR=E:/temp',
    'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
    'export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1',
    'export PYTHONHASHSEED=0',
    'cd /e/quant/gpsef_project',
    'PY=.venv/Scripts/python.exe',
    'WD=_work/20261008_barra',
    'mkdir -p "$WD/logs" "$WD/tmp"',
    'echo "=== Q5BAR FULL batch start $(date +%F_%T) ==="',
    'echo "--- [gate0] 运行时依赖闭包预检 ---"',
    '"$PY" -u "$WD/scripts/_q5bar_preimport_check.py" || { echo "★ gate0 FAIL ⇒ 拒起臂"; exit 1; }',
    'echo "--- [gate0] PASS ---"',
    'PIDS=()',
]
JOBS = [
    ('B1', '%s/configs/config_B1_full.yaml' % WD, 'results/Q5BAR_B1_full_20261008', 'full_B1'),
    ('S1', '%s/configs/config_S1_full.yaml' % WD, 'results/Q5BAR_S1_full_20261008', 'full_S1'),
    ('V1', '%s/configs/config_V1.yaml' % WD, 'results/Q5BARB_V1_20261008', 'v1'),
]
for tag, cfg, outp, log in JOBS:
    lines += [
        'rm -rf "%s"' % outp,
        '("$PY" -u run_backtest.py --config "%s" --output "%s" 2>&1 '
        '| tee "$WD/logs/%s.log") &' % (cfg, outp, log),
        'PIDS+=($!)',
        'echo "  ARM %s pid=${PIDS[-1]}  (%s)"' % (tag, outp),
    ]
lines += [
    'for p in "${PIDS[@]}"; do wait "$p"; echo "  waited $p rc=$?"; done',
    'echo "=== Q5BAR FULL batch end $(date +%F_%T) ==="',
]
L = '%s/run_full.sh' % WD
with io.open(L, 'w', encoding='utf-8', newline='\n') as f:
    f.write('\n'.join(lines) + '\n')
P('')
P('[4] 启动器 = %s' % L)
P('    ★3 槽：B1_full（~14h）＋ S1_full（~14h）＋ V1 烟测（~1h15m）')

with io.open('%s/analysis/q5bar_setup_full2.txt' % WD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(OUT))
P('')
P('[done] %s/analysis/q5bar_setup_full2.txt' % WD)
