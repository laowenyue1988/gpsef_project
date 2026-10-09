#!/bin/bash
# 等待 R12 全区间回测完成 (日志出现完成标志)
cd E:/quant/gpsef_project
for i in $(seq 1 300); do
  if grep -q "绩效报告已保存" results/R12_full.log 2>/dev/null; then
    echo "R12_FULL_DONE $(date '+%m-%d %H:%M:%S')" > results/R12_full_done.txt
    echo "DONE" >> results/R12_full_done.txt
    exit 0
  fi
  sleep 120
done
echo "TIMEOUT" > results/R12_full_done.txt
