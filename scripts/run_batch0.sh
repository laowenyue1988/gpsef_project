#!/bin/bash
# 批0: R0 + R1 串行回测 (断点续跑: 每组独立, 失败继续下一组)
cd E:/quant/gpsef_project
ST=results/batch0_status.txt
echo "=== BATCH0 START $(date '+%Y-%m-%d %H:%M:%S') ===" > $ST
echo "[R0] start $(date '+%H:%M:%S')" >> $ST
.venv/Scripts/python.exe -u run_backtest.py --config strategies/small_alpha191_test2/config_R0.yaml > results/R0.log 2>&1
echo "[R0] exit=$? end $(date '+%H:%M:%S')" >> $ST
echo "[R1] start $(date '+%H:%M:%S')" >> $ST
.venv/Scripts/python.exe -u run_backtest.py --config strategies/small_alpha191_test2/config_R1.yaml > results/R1.log 2>&1
echo "[R1] exit=$? end $(date '+%H:%M:%S')" >> $ST
echo "=== BATCH0 END $(date '+%Y-%m-%d %H:%M:%S') ===" >> $ST
