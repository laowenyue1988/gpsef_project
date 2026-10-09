#!/bin/bash
# 批0 完成监控: 轮询 R0/R1 日志直到"绩效报告已保存" (每120s), 全部完成后退出
# 用途: 将异步回测的完成时机转为本后台任务的完成通知
cd E:/quant/gpsef_project
ST=results/batch0_monitor.txt
echo "=== MONITOR START $(date '+%Y-%m-%d %H:%M:%S') ===" > $ST
for tag in R0 R1; do
  echo "[$tag] 等待完成 $(date '+%H:%M:%S')" >> $ST
  cnt=0
  while ! grep -q "绩效报告已保存" results/$tag.log 2>/dev/null; do
    cnt=$((cnt+1))
    if [ $cnt -ge 400 ]; then   # 400*120s ≈ 13.3h 上限
      echo "[$tag] TIMEOUT (13h 未完成)" >> $ST
      break
    fi
    sleep 120
  done
  if grep -q "绩效报告已保存" results/$tag.log 2>/dev/null; then
    echo "[$tag] 完成 $(date '+%H:%M:%S') $(wc -l < results/$tag.log) 行" >> $ST
  fi
done
echo "=== BATCH0 全部完成 $(date '+%Y-%m-%d %H:%M:%S') ===" >> $ST
echo "DONE" >> $ST
