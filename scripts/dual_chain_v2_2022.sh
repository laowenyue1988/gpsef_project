#!/bin/bash
# 对接链 v2 (2026-09-01 修复后重跑):
#   gpsef 回测完成 → 导出真实信号 → Eureka(修复后) 同步回测
cd /e/quant/gpsef_project
DUAL_DIR=results/SmallAlpha191Test2_20220801_20221001_20260901012724
LOG=results/dual_chain_v2_2022.log
: > "$LOG"

echo "[$(date '+%m-%d %H:%M:%S')] 等待 gpsef v2 回测完成 (目录: $DUAL_DIR)..." | tee -a "$LOG"
for i in $(seq 1 900); do
  if [ -f "$DUAL_DIR/report.txt" ] && [ -f "$DUAL_DIR/orders.csv" ] && \
     grep -q "绩效报告已保存" "$DUAL_DIR/backtest.log" 2>/dev/null; then
    echo "[$(date '+%m-%d %H:%M:%S')] gpsef v2 回测完成" | tee -a "$LOG"
    break
  fi
  sleep 30
done
if [ ! -f "$DUAL_DIR/report.txt" ]; then
  echo "[$(date '+%m-%d %H:%M:%S')] ERROR: 等待超时" | tee -a "$LOG"
  echo "CHAIN_V2_FAILED" > results/dual_chain_v2_done.txt
  exit 1
fi

echo "[$(date '+%m-%d %H:%M:%S')] 导出真实信号..." | tee -a "$LOG"
./.venv/Scripts/python.exe scripts/export_signals.py \
  --orders "$DUAL_DIR/orders.csv" \
  --log "$DUAL_DIR/backtest.log" \
  --config strategies/small_alpha191_test2/config_dual2022.yaml \
  --out E:/quant/Eureka/signals 2>&1 | tee -a "$LOG"

N=$(ls E:/quant/Eureka/signals/*.json 2>/dev/null | wc -l)
echo "[$(date '+%m-%d %H:%M:%S')] 信号文件数: $N" | tee -a "$LOG"
if [ "$N" -lt 1 ]; then
  echo "[$(date '+%m-%d %H:%M:%S')] ERROR: 信号导出为空" | tee -a "$LOG"
  echo "CHAIN_V2_FAILED" > results/dual_chain_v2_done.txt
  exit 1
fi

echo "[$(date '+%m-%d %H:%M:%S')] 启动 Eureka v2 同步回测..." | tee -a "$LOG"
cd /e/quant/Eureka
./.venv/Scripts/python.exe run.py strategies/small_alpha191_test2_live 2>&1 | tee -a /e/quant/gpsef_project/results/dual_chain_v2_2022.log

echo "[$(date '+%m-%d %H:%M:%S')] 对接链 v2 完成" | tee -a "$LOG"
echo "CHAIN_V2_DONE $(date '+%m-%d %H:%M:%S')" > /e/quant/gpsef_project/results/dual_chain_v2_done.txt
