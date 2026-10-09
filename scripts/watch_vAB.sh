#!/bin/bash
# 监控微型验证 A/B 首训标签分布, 对比一致性
cd E:/quant/gpsef_project
OUT=results/vAB_label_check.txt
echo "=== $(date '+%m-%d %H:%M:%S') 等待首训标签 ===" > $OUT
for i in $(seq 1 60); do
  A=$(grep -oE "训练样本总数: [0-9]+, 标签分布: [^)]*" results/vA.log 2>/dev/null | head -1)
  B=$(grep -oE "训练样本总数: [0-9]+, 标签分布: [^)]*" results/vB.log 2>/dev/null | head -1)
  if [ -n "$A" ] && [ -n "$B" ]; then
    echo "vA(ref 2024-03-01): $A" >> $OUT
    echo "vB(ref 2026-08-04): $B" >> $OUT
    if [ "$A" = "$B" ]; then
      echo "RESULT: ✅ 一致 (修复生效)" >> $OUT
    else
      echo "RESULT: ❌ 不一致 (修复未生效)" >> $OUT
    fi
    echo "DONE" >> $OUT
    exit 0
  fi
  sleep 60
done
echo "TIMEOUT" >> $OUT
