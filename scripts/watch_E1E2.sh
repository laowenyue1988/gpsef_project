#!/bin/bash
# 监控 E1/E2 全窗口回测完成 (与 R12 修复版同窗口 2022-08-01~2026-08-04)
cd E:/quant/gpsef_project
for i in $(seq 1 400); do   # 400*120s ≈ 13h
  A=$(grep -c "绩效报告已保存" results/E1.log 2>/dev/null)
  B=$(grep -c "绩效报告已保存" results/E2.log 2>/dev/null)
  if [ "$A" -ge 1 ] && [ "$B" -ge 1 ]; then
    echo "E1E2_DONE $(date '+%m-%d %H:%M:%S')" > results/E1E2_done.txt
    exit 0
  fi
  sleep 120
done
echo "E1E2_TIMEOUT" > results/E1E2_done.txt
