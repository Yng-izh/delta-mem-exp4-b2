#!/bin/bash
# watch_v4_eval.sh —— v4 训练看门狗：训练进程一退出就自动把评测挂上去。
#
# 设计约束（都是踩过的坑）：
#   1) 训练和评测都在这台共享机上跑，卡经常被别人占满 -> 每次开跑前**按显存挑卡**，
#      不写死 GPU 1；没卡就等，等到为止。
#   2) 评测会 OOM（4B + adapter 约 14 GB）-> 每个模式最多重试 3 次，失败间隔 5 分钟。
#   3) 训练可能是正常结束、也可能是崩了 -> 先查日志收尾标记 + ckpt 是否在，
#      两种情况都记进 status，不许静默。
#   4) 一切状态写到 $B/logs/watch_v4_eval.status（带时间戳），人不用守着终端。
#
# 跑两个模式（同一 ckpt）：
#   router        -> 端到端 EM，与 v3 的 12.6/12.8 同表
#   oracle_latent -> 金标槽位直送 bridge 的读出上界，与 v3 的 17.0 对照
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=/data/zhangjiayang/miniconda3/envs/flux/bin/python
CKPT=$B/ckpt/hotpot_router_10k_v4.pt
STATUS=$B/logs/watch_v4_eval.status
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$R" || exit 1

log() { echo "[watch $(date '+%F %T')] $*" | tee -a "$STATUS"; }

log "看门狗启动 pid=$$"
while pgrep -f "train_latent_memory.py --router" >/dev/null 2>&1; do sleep 60; done
log "训练进程已退出"

if grep -q "ROUTER-10K-V4 DONE" "$B/logs/train_v4.log" 2>/dev/null; then
  log "训练日志：有正常收尾标记 ROUTER-10K-V4 DONE"
else
  log "警告：训练日志里没有 ROUTER-10K-V4 DONE（可能异常退出），仍尝试评测"
fi
if [ ! -f "$CKPT" ]; then log "致命：找不到 $CKPT"; exit 1; fi
log "ckpt=$(ls -la "$CKPT" | awk '{print $5" bytes "$6" "$7" "$8}')"

pick_gpu() {
  local need=${1:-16000} best idx i f
  while true; do
    best=-1; idx=-1
    while IFS=, read -r i f; do
      i="${i// /}"; f="${f// /}"
      if [ "$f" -ge "$need" ] && [ "$f" -gt "$best" ]; then best=$f; idx=$i; fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    if [ "$idx" -ge 0 ]; then echo "$idx"; return 0; fi
    log "没有 >= ${need}MiB 空闲的卡，等 120 秒"
    sleep 120
  done
}

run_mode() {
  local mode=$1 label=$2 out=$3 attempt gpu rc
  for attempt in 1 2 3; do
    gpu=$(pick_gpu 16000)
    log "mode=$mode 第 $attempt 次，GPU=$gpu 空闲，开跑"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" eval_latent_table.py --mode "$mode" \
      --ckpt "$CKPT" --data $B/synth/hotpot_val500.jsonl --limit 500 \
      --max-new-tokens 16 --gpu 0 \
      --base $B/models/Qwen3-4B-Instruct-2507 \
      --adapter $B/models/delta-mem_qwen3_4b_instruct \
      --label "$label" --save-json "$out" >> "$B/logs/eval_${mode}_v4.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -f "$out" ]; then
      log "mode=$mode 完成 rc=0 -> $out"
      grep -E "^\| $label " "$B/logs/eval_${mode}_v4.log" | tail -1 | tee -a "$STATUS"
      return 0
    fi
    log "mode=$mode 失败 rc=$rc，5 分钟后重试"
    sleep 300
  done
  log "mode=$mode 三次都失败，放弃"
  return 1
}

run_mode router        v4-router        "$B/results/v4_router_500.json"
run_mode oracle_latent v4-oracle_latent "$B/results/v4_oracle_latent_500.json"
log "全部结束"
touch "$B/results/V4_EVAL_DONE"
