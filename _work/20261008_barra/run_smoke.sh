#!/usr/bin/env bash
set -u
# ★R-202：必须预置 7 个确定性环境变量，绕开 run_backtest.py:27-30 的自我 exec 重启
export TEMP=E:/temp TMP=E:/temp TMPDIR=E:/temp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0
cd /e/quant/gpsef_project
PY=.venv/Scripts/python.exe
WD=_work/20261008_barra
mkdir -p "$WD/logs" "$WD/tmp"
echo "=== Q5BAR batch start $(date +%F_%T) ==="

# ★门控 0：运行时依赖闭包预检（含**函数内** import）—— 来自 2026-10-08 INCIDENT（函数内 import 是盲区）
echo "--- [gate0] 运行时依赖闭包预检（★判据=标志文件，禁用返回码）---"
rm -f "$WD/tmp/gate0.result"
"$PY" -u "$WD/scripts/_q5bar_preimport_check.py"
_G=$(cat "$WD/tmp/gate0.result" 2>/dev/null | tr -d '\r\n')
[ "$_G" = "PASS" ] || { echo "★ gate0 FAIL (sentinel='$_G') ⇒ 拒起臂"; exit 1; }
echo "--- [gate0] PASS ---"

PIDS=()
rm -rf "results/Q5BAR_Z1_20261008"
("$PY" -u run_backtest.py --config "_work/20261008_barra/configs/config_Z1.yaml" --output "results/Q5BAR_Z1_20261008" 2>&1 | tee "$WD/logs/arm_Z1.log") &
PIDS+=($!)
echo "  ARM Z1 pid=${PIDS[-1]}"
rm -rf "results/Q5BAR_B1_20261008"
("$PY" -u run_backtest.py --config "_work/20261008_barra/configs/config_B1.yaml" --output "results/Q5BAR_B1_20261008" 2>&1 | tee "$WD/logs/arm_B1.log") &
PIDS+=($!)
echo "  ARM B1 pid=${PIDS[-1]}"
rm -rf "results/Q5BAR_S1_20261008"
("$PY" -u run_backtest.py --config "_work/20261008_barra/configs/config_S1.yaml" --output "results/Q5BAR_S1_20261008" 2>&1 | tee "$WD/logs/arm_S1.log") &
PIDS+=($!)
echo "  ARM S1 pid=${PIDS[-1]}"
for p in "${PIDS[@]}"; do wait "$p"; echo "  waited $p rc=$?"; done
echo "=== Q5BAR batch end $(date +%F_%T) ==="
