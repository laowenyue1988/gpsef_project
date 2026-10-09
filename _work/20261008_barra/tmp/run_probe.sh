#!/usr/bin/env bash
set -u
export TEMP=E:/temp TMP=E:/temp TMPDIR=E:/temp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0
cd /e/quant/gpsef_project
.venv/Scripts/python.exe -u run_backtest.py --config _work/20261008_barra/tmp/config_probe.yaml --output results/_Q5BAR_probe > _work/20261008_barra/tmp/probe.log 2>&1
echo "PROBE_RC=$?"
