#!/usr/bin/env bash
# Q5BAR-A 单臂重起：仅 S1_full（因第 2 批起跑时 rm -rf 静默失败 ⇒ S1 产物目录混入第 1 批残留 ⇒ 重起）
#   ★新增 fail-closed 判据：rm -rf 后**必须断言目录不存在**，否则拒起臂（不信 rm 的返回码，R-208/R-209）
#   B1_full 不受影响（继续跑）⇒ 本脚本只动 S1。
set -u
export TEMP=E:/temp TMP=E:/temp TMPDIR=E:/temp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0
cd /e/quant/gpsef_project
PY=.venv/Scripts/python.exe
WD=_work/20261008_barra
mkdir -p "$WD/logs" "$WD/tmp"
echo "=== Q5BAR-A S1 single-arm restart $(date +%F_%T) ==="
echo "--- [gate0] 运行时依赖闭包预检（★判据=标志文件，禁用返回码）---"
rm -f "$WD/tmp/gate0.result"
"$PY" -u "$WD/scripts/_q5bar_preimport_check.py"
_G=$(cat "$WD/tmp/gate0.result" 2>/dev/null | tr -d '\r\n')
[ "$_G" = "PASS" ] || { echo "★ gate0 FAIL (sentinel='$_G') ⇒ 拒起臂"; exit 1; }
echo "--- [gate0] PASS ---"

OUT=results/Q5BAR_S1_full_20261008
rm -rf "$OUT"
sleep 3
if [ -d "$OUT" ]; then
  echo "★ 输出目录清理失败（仍存在）⇒ 拒起臂：$OUT"
  echo "   提示：旧批进程可能仍持句柄 ⇒ 先生成 kill 清单并逐杀后再试"
  ls -la "$OUT"
  exit 1
fi
echo "--- [clean] 输出目录已确认不存在 ✓ ---"

("$PY" -u run_backtest.py --config "$WD/configs/config_S1_full.yaml" --output "$OUT" 2>&1 | tee "$WD/logs/full_S1.log") &
P1=$!
echo "  ARM S1_full pid=$P1  ($OUT)"
wait "$P1"; echo "  waited $P1 rc=$?"
echo "=== Q5BAR-A S1 single-arm end $(date +%F_%T) ==="
