#!/bin/bash
# supervisor_v4.sh —— v4 训练/评测的常驻监督（每 5 分钟一次，直到评测完成）
#
# 它只做三件事，全部带时间戳写进 status：
#   1) 训练还在跑：记录 step 与各卡空闲显存；step 连续 8 次（约 40 分钟）不变 -> 告警
#   2) 训练已退出但没人在评：如果看门狗也不在了，就把它重新拉起来（自愈）
#   3) 评测跑完（results/V4_EVAL_DONE 出现）：记录并退出
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
STATUS=$B/logs/supervisor_v4.status
log() { echo "[sup $(date '+%F %T')] $*" | tee -a "$STATUS"; }

last_step=""; stall=0
log "supervisor 启动 pid=$$"
while true; do
  if [ -f "$B/results/V4_EVAL_DONE" ]; then
    log "评测已完成（V4_EVAL_DONE 存在），supervisor 退出"
    exit 0
  fi
  if pgrep -f "train_latent_memory.py --router" >/dev/null 2>&1; then
    step=$(grep -oE "^\[step [0-9]+" "$B/logs/train_v4.log" | tail -1 | tr -dc '0-9')
    if [ "${step:-}" = "$last_step" ]; then stall=$((stall+1)); else stall=0; fi
    last_step="${step:-}"
    gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tr '\n' ';')
    log "训练中 step=${step:-?} 停滞计数=$stall 空闲显存=[$gpu]"
    if [ "$stall" -ge 8 ]; then
      log "告警：训练 step 连续 8 次未变（约 40 分钟），可能已卡死；进程未动，不做干预"
    fi
  else
    if pgrep -f "eval_latent_table.py" >/dev/null 2>&1; then
      log "训练已结束，评测进行中"
    elif pgrep -f "watch_v4_eval.sh" >/dev/null 2>&1; then
      log "训练已结束，看门狗在等（挑卡/退避中）"
    else
      log "训练已结束、无评测、看门狗不在 -> 重新拉起看门狗"
      ( cd "$R" && nohup bash runs/watch_v4_eval.sh >> "$B/logs/watch_v4_eval.log" 2>&1 & )
      sleep 5
      log "看门狗 pid=$(pgrep -f 'watch_v4_eval.sh' | head -1)"
    fi
  fi
  sleep 300
done
