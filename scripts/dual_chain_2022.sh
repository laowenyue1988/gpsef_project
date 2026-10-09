#!/bin/bash
# ============================================================
# 真实流程对接链: gpsef 2022段回测 → 导出真实信号 → Eureka 同步回测
# 严格按真实流水线: 回测产出 orders/backtest.log → export_signals.py
# 生成 signals → Eureka 读取信号跑同步回测。不手工导出、不重用旧信号。
# ============================================================
cd /e/quant/gpsef_project

DUAL_DIR=results/SmallAlpha191Test2_20220801_20221001_20260831235455
CHAIN_LOG=results/dual_chain_2022.log
: > "$CHAIN_LOG"

echo "[$(date '+%m-%d %H:%M:%S')] 等待 gpsef 回测完成..." | tee -a "$CHAIN_LOG"
for i in $(seq 1 600); do
  if [ -f results/dual_gpsef_done.txt ]; then
    echo "[$(date '+%m-%d %H:%M:%S')] gpsef 回测已完成" | tee -a "$CHAIN_LOG"
    break
  fi
  sleep 30
done
if [ ! -f results/dual_gpsef_done.txt ]; then
  echo "[$(date '+%m-%d %H:%M:%S')] ERROR: 等待超时" | tee -a "$CHAIN_LOG"
  echo "CHAIN_FAILED" > results/dual_chain_done.txt
  exit 1
fi

# 1) 从这轮真实回测结果导出信号 (orders.csv + backtest.log 均来自本轮 dual 回测)
echo "[$(date '+%m-%d %H:%M:%S')] 导出真实信号..." | tee -a "$CHAIN_LOG"
./.venv/Scripts/python.exe scripts/export_signals.py \
  --orders "$DUAL_DIR/orders.csv" \
  --log "$DUAL_DIR/backtest.log" \
  --config strategies/small_alpha191_test2/config_dual2022.yaml \
  --out E:/quant/Eureka/signals 2>&1 | tee -a "$CHAIN_LOG"

N=$(ls E:/quant/Eureka/signals/*.json 2>/dev/null | wc -l)
echo "[$(date '+%m-%d %H:%M:%S')] 信号文件数: $N" | tee -a "$CHAIN_LOG"
if [ "$N" -lt 1 ]; then
  echo "[$(date '+%m-%d %H:%M:%S')] ERROR: 信号导出为空" | tee -a "$CHAIN_LOG"
  echo "CHAIN_FAILED" > results/dual_chain_done.txt
  exit 1
fi

# 2) Eureka 读取真实信号跑同步回测
echo "[$(date '+%m-%d %H:%M:%S')] 启动 Eureka 同步回测..." | tee -a "$CHAIN_LOG"
cd /e/quant/Eureka
./.venv/Scripts/python.exe run.py strategies/small_alpha191_test2_live 2>&1 | tee -a /e/quant/gpsef_project/results/dual_chain_2022.log

echo "[$(date '+%m-%d %H:%M:%S')] 对接链完成" | tee -a "$CHAIN_LOG"
echo "CHAIN_DONE $(date '+%m-%d %H:%M:%S')" > /e/quant/gpsef_project/results/dual_chain_done.txt
