#!/usr/bin/env bash
# Q5BAR-A 全周期批次（第 2 次起跑）—— 只含 B1_full ＋ S1_full 两臂
#   ① V1 已完成烟测（D-347/B 族），不重跑；
#   ② 内存门控：2 臂长窗峰值 ≈2×5GB，第 3 臂会使 commit_free 跌破 20GB 门槛 ⇒ 严格 2 臂；
#   ③ 判据 = gate0 **标志文件**（禁用返回码，R-208）；日志落 _work（禁堆根目录）。
set -u
export TEMP=E:/temp TMP=E:/temp TMPDIR=E:/temp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0
cd /e/quant/gpsef_project
PY=.venv/Scripts/python.exe
WD=_work/20261008_barra
mkdir -p "$WD/logs" "$WD/tmp"
echo "=== Q5BAR-A FULL batch#2 start $(date +%F_%T) ==="
echo "--- [gate0] 运行时依赖闭包预检（★判据=标志文件，禁用返回码）---"
rm -f "$WD/tmp/gate0.result"
"$PY" -u "$WD/scripts/_q5bar_preimport_check.py"
_G=$(cat "$WD/tmp/gate0.result" 2>/dev/null | tr -d '\r\n')
[ "$_G" = "PASS" ] || { echo "★ gate0 FAIL (sentinel='$_G') ⇒ 拒起臂"; exit 1; }
echo "--- [gate0] PASS ---"
PIDS=()
for A in B1 S1; do
  rm -rf "results/Q5BAR_${A}_full_20261008"
  ("$PY" -u run_backtest.py --config "$WD/configs/config_${A}_full.yaml" \
      --output "results/Q5BAR_${A}_full_20261008" 2>&1 | tee "$WD/logs/full_${A}.log") &
  PIDS+=($!)
  echo "  ARM ${A}_full pid=${PIDS[-1]}  (results/Q5BAR_${A}_full_20261008)"
done
for p in "${PIDS[@]}"; do wait "$p"; echo "  waited $p rc=$?"; done
echo "=== Q5BAR-A FULL batch#2 end $(date +%F_%T) ==="
