#!/bin/bash
# 排程器看门狗 v2: DONE_ALL 检测 + 心跳告警 (不再自动重启排程器——v4 已自愈)
# - 每 120s: 心跳 >900s 未更新 → 写告警日志(人工介入线索)
# - 检测 DONE_ALL.txt → 跑 final_analyze.py → 退出(通知主代理做最终结论)
# 用法: bash scripts/watchdog.sh (后台运行)
cd E:/quant/gpsef_project
LG=results/watchdog.log
FA_LAST=results/final_analyze_last.txt
echo "=== WATCHDOG v3 START $(date '+%m-%d %H:%M:%S') ===" >> $LG
for i in $(seq 1 1080); do   # 1080*120s = 36h
  # 1) DONE_ALL → 最终分析 → 退出
  if [ -f results/DONE_ALL.txt ]; then
    echo "$(date '+%m-%d %H:%M:%S') DONE_ALL 检测到, 运行最终分析" >> $LG
    .venv/Scripts/python.exe scripts/final_analyze.py >> $LG 2>&1
    echo "$(date '+%m-%d %H:%M:%S') 最终分析完成, watchdog 退出" >> $LG
    exit 0
  fi
  # 2) 每小时自动更新 grid_summary (幂等, 中间阶段也持续输出分析)
  if [ ! -f "$FA_LAST" ] || [ $(( $(date +%s) - $(stat -c %Y "$FA_LAST" 2>/dev/null || echo 0) )) -gt 3600 ]; then
    touch "$FA_LAST"
    echo "$(date '+%m-%d %H:%M:%S') 每小时汇总更新" >> $LG
    .venv/Scripts/python.exe scripts/final_analyze.py >> results/final_analyze_hourly.log 2>&1
    echo "$(date '+%m-%d %H:%M:%S') 汇总完成 (exit=$?)" >> $LG
  fi
  # 3) 心跳告警
  if [ -f results/auto_runner_heartbeat.txt ]; then
    MT=$(stat -c %Y results/auto_runner_heartbeat.txt 2>/dev/null || echo 0)
    NOW=$(date +%s)
    AGE=$((NOW - MT))
    if [ "$AGE" -gt 900 ]; then
      echo "$(date '+%m-%d %H:%M:%S') ⚠️ 心跳 ${AGE}s 未更新 (排程器可能已死, 需人工介入)" >> $LG
    fi
  fi
  sleep 120
done
echo "$(date '+%m-%d %H:%M:%S') watchdog 36h 超时退出 (无 DONE_ALL)" >> $LG
exit 1
