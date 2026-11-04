#!/bin/bash
# 阶段二产物的 500 题评测：先端到端 router，再 oracle_latent 读出上界（完全绕过 Router）。
# 两个数一对照就能分出"寻址错"与"读出/压缩丢信息"——差得多是寻址，差得少是读出。
#
# ---- 用法 ----
#   CKPT=$B/ckpt/<你的阶段二权重>.pt bash runs/eval_stage2.sh
#   可选：TAG=名字（默认取 ckpt 文件名）LIMIT=题数（默认 500）GPU=0/1
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=/data/zhangjiayang/miniconda3/envs/flux/bin/python
CKPT=${CKPT:?必须显式指定 CKPT（没有默认权重了）}
TAG=${TAG:-$(basename "$CKPT" .pt)}
LIMIT=${LIMIT:-500}
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd "$R" || exit 1

run() {
  local mode=$1 label=$2 out=$3 log=$4
  echo "[eval $(date '+%F %T')] $mode 开始"
  "$PY" eval_latent_table.py --mode "$mode" --ckpt "$CKPT" \
    --data $B/synth/hotpot_val500.jsonl --limit "$LIMIT" --max-new-tokens 16 \
    --topk-per-hop 4 \
    --base $B/models/Qwen3-4B-Instruct-2507 \
    --adapter $B/models/delta-mem_qwen3_4b_instruct \
    --label "$label" --save-json "$out" > "$log" 2>&1
  echo "[eval $(date '+%F %T')] $mode 结束 rc=$?"
}

run router        "$TAG-router"        "$B/results/${TAG}_router_${LIMIT}.json"        "$B/logs/eval_${TAG}_router.log"
run oracle_latent "$TAG-oracle_latent" "$B/results/${TAG}_oracle_latent_${LIMIT}.json" "$B/logs/eval_${TAG}_oracle_latent.log"
echo "[eval $(date '+%F %T')] 全部完成"
