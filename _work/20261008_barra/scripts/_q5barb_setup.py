# -*- coding: utf-8 -*-
"""[Q5BAR-B] 搭台：生成 V1/V2/V3 臂配置（求解器配权族）+ 快照内容级基线 + 启动器。

★零代码改动：仅把臂配置里的 `barra.enabled` 置 true 并设定软参数。
★与采纳包逐键一致（除 barra 段）；断言纯度。
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
P('[Q5BAR-B] 搭台  %s' % (datetime.datetime.now().strftime('%F %T'),))
P('=' * 100)

anc = yaml.safe_load(io.open(ANCHOR, encoding='utf-8'))
P('[1] 锚配置 barra 段 = %s' % json.dumps(anc.get('barra'), ensure_ascii=False))

BASE_BARRA = {
    'enabled': True,                      # ★本波唯一的功能开关
    'snapshot_dir': './data/barra_exposures',
    'baseline_index_code': 'sh932000',    # 中证2000 微盘（设计档 §1.3 论证）
    'max_weight': 0.10,
    'expected_ic': 0.05,                  # ⚠️ 假杠杆（运行时被 meta_auc 覆盖）
    'alpha_scale': 10.0,
    'vol_target': 0.15,
}
ARMS = {
    'V1': dict(BASE_BARRA, risk_aversion=0.5,  lambda_dev=0.2,  lambda_l2=0.3,  lambda_ind=0.3,  _tag='当年档复现'),
    'V2': dict(BASE_BARRA, risk_aversion=2.0,  lambda_dev=0.2,  lambda_l2=1.0,  lambda_ind=0.3,  _tag='风控主导档'),
    'V3': dict(BASE_BARRA, risk_aversion=0.05, lambda_dev=0.02, lambda_l2=0.05, lambda_ind=0.05, _tag='纯alpha端（R-195 替代探针）'),
}

P('')
P('[2] 生成臂配置（烟测窗 20240401~20240801）')
cfgs = {}
for k, spec in ARMS.items():
    tag = spec.pop('_tag')
    c = yaml.safe_load(io.open(ANCHOR, encoding='utf-8'))
    c['strategy']['class'] = 'strategies.small_alpha191.strategy.SmallAlpha191Strategy'
    c['backtest']['start_date'] = '20240401'
    c['backtest']['end_date'] = '20240801'
    _e = c.get('experiment') or {}
    _e.pop('preds_mode', None)
    _e.pop('cache_dir', None)
    c['experiment'] = _e
    c['barra'] = dict(spec)
    c['barra'].pop('risk_budget', None)   # 本族不用波 5 的机制
    p = '%s/configs/config_%s.yaml' % (WD, k)
    with io.open(p, 'w', encoding='utf-8', newline='\n') as f:
        yaml.safe_dump(c, f, allow_unicode=True, sort_keys=False, width=200)
    cfgs[k] = p
    P('    %-3s %-42s risk_av=%.2f λ_dev=%.2f λ_l2=%.2f λ_ind=%.2f  %s'
      % (k, p, spec['risk_aversion'], spec['lambda_dev'], spec['lambda_l2'], spec['lambda_ind'], tag))

P('')
P('[3] 纯度断言：三配置差异必须**只**在 barra 段内')
f1 = flat(yaml.safe_load(io.open(cfgs['V1'], encoding='utf-8')))
for k in ('V2', 'V3'):
    fk = flat(yaml.safe_load(io.open(cfgs[k], encoding='utf-8')))
    d = sorted(x for x in set(f1) | set(fk) if f1.get(x) != fk.get(x))
    bad = [x for x in d if not x.startswith('barra.')]
    P('    V1 vs %s: 差异 %d 键 ⇒ %s' % (k, len(d), d))
    P('        非 barra.* 差异 = %s  %s' % (bad if bad else '无', '✅' if not bad else '❌'))
    if bad:
        raise SystemExit('纯度断言失败')

# ── 快照内容级基线（R-156：供跑后比对，检出运行期重建） ──────
P('')
P('[4] 快照目录内容级基线（R-156）')
snap = 'data/barra_exposures'
files = sorted(f for f in os.listdir(snap) if f.startswith('barra_expo_') and f.endswith('.parquet'))
rows = []
for f in files:
    p = os.path.join(snap, f)
    st = os.stat(p)
    rows.append('%s,%d,%d' % (f, st.st_size, int(st.st_mtime)))
bl = '%s/analysis/snapshot_baseline_%s.csv' % (WD, datetime.datetime.now().strftime('%Y%m%d_%H%M'))
with io.open(bl, 'w', encoding='utf-8', newline='\n') as fh:
    fh.write('file,size,mtime\n' + '\n'.join(rows) + '\n')
P('    快照文件数 = %d  基线已写 %s' % (len(rows), bl))
P('    md5(清单) = %s' % hashlib.md5('\n'.join(rows).encode()).hexdigest()[:16])

# ── 启动器（预置 7 变量 + gate0 预检；**不自动执行**） ───────
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
    'echo "=== Q5BAR-B batch start $(date +%F_%T) ==="',
    'echo "--- [gate0] 运行时依赖闭包预检 ---"',
    '"$PY" -u "$WD/scripts/_q5bar_preimport_check.py" || { echo "★ gate0 FAIL ⇒ 拒起臂"; exit 1; }',
    'echo "--- [gate0] PASS ---"',
    'PIDS=()',
]
for k in ('V1', 'V2'):
    lines += [
        'rm -rf "results/Q5BARB_%s_20261008"' % k,
        '("$PY" -u run_backtest.py --config "%s/configs/config_%s.yaml" '
        '--output "results/Q5BARB_%s_20261008" 2>&1 '
        '| tee "$WD/logs/w_%s.log") &' % (WD, k, k, k),
        'PIDS+=($!)',
        'echo "  ARM %s pid=${PIDS[-1]}"' % k,
    ]
lines += [
    'for p in "${PIDS[@]}"; do wait "$p"; echo "  waited $p rc=$?"; done',
    'echo "=== Q5BAR-B batch end $(date +%F_%T) ==="',
]
L = '%s/run_solveweight.sh' % WD
with io.open(L, 'w', encoding='utf-8', newline='\n') as f:
    f.write('\n'.join(lines) + '\n')
P('')
P('[5] 启动器 = %s（V1+V2 两臂；★待槽位腾出后手动起）' % L)

with io.open('%s/analysis/q5barb_setup.txt' % WD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(OUT))
P('')
P('[done] %s/analysis/q5barb_setup.txt' % WD)
