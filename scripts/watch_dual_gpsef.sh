#!/bin/bash
# 等待 gpsef 2022段回测完成
cd E:/quant/gpsef_project
for i in $(seq 1 400); do
  if grep -q "绩效报告已保存" results/dual_gpsef_2022.log 2>/dev/null; then
    echo "GPSEF_DUAL_DONE $(date '+%m-%d %H:%M:%S')" > results/dual_gpsef_done.txt
    exit 0
  fi
  sleep 30
done
echo "TIMEOUT" > results/dual_gpsef_done.txt
