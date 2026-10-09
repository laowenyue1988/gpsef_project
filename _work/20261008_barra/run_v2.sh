#!/usr/bin/env bash
set -u
export TEMP=E:/temp TMP=E:/temp TMPDIR=E:/temp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0
cd /e/quant/gpsef_project
PY=.venv/Scripts/python.exe
WD=_work/20261008_barra
mkdir -p "$WD/logs" "$WD/tmp"
echo "=== Q5BAR-B V2 single-arm start $(date +%F_%T) ==="
echo "--- [gate0] 运行时依赖闭包预检（★判据=标志文件，禁用返回码）---"
rm -f "$WD/tmp/gate0.result"
"$PY" -u "$WD/scripts/_q5bar_preimport_check.py"
_G=$(cat "$WD/tmp/gate0.result" 2>/dev/null | tr -d '\r\n')
[ "$_G" = "PASS" ] || { echo "★ gate0 FAIL (sentinel='$_G') ⇒ 拒起臂"; exit 1; }
echo "--- [gate0] PASS ---"
PIDS=()
rm -rf "results/Q5BARB_V2_20261008"
("$PY" -u run_backtest.py --config "_work/20261008_barra/configs/config_V2.yaml" --output "results/Q5BARB_V2_20261008" 2>&1 | tee "$WD/logs/w_V2.log") &
PIDS+=($!)
echo "  ARM V2 pid=${PIDS[-1]}"
for p in "${PIDS[@]}"; do wait "$p"; echo "  waited $p rc=$?"; done
echo "=== Q5BAR-B V2 single-arm end $(date +%F_%T) ==="
