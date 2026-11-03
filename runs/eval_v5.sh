#!/bin/bash
# 阶段二产物（v5）的 500 题评测：先端到端 router，再 oracle_latent 读出上界。
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=/data/zhangjiayang/miniconda3/envs/flux/bin/python
CKPT=$B/ckpt/hotpot_router_10k_v5.pt
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd "$R" || exit 1

run() {
  local mode=$1 label=$2 out=$3 log=$4
  echo "[eval $(date '+%F %T')] $mode 开始"
  "$PY" eval_latent_table.py --mode "$mode" --ckpt "$CKPT" \
    --data $B/synth/hotpot_val500.jsonl --limit 500 --max-new-tokens 16 --gpu 0 \
    --topk-per-hop 4 \
    --base $B/models/Qwen3-4B-Instruct-2507 \
    --adapter $B/models/delta-mem_qwen3_4b_instruct \
    --label "$label" --save-json "$out" > "$log" 2>&1
  echo "[eval $(date '+%F %T')] $mode 结束 rc=$?"
}

run router        v5-router        $B/results/v5_router_500.json        $B/logs/eval_v5_router.log
run oracle_latent v5-oracle_latent $B/results/v5_oracle_latent_500.json $B/logs/eval_v5_oracle_latent.log
echo "[eval $(date '+%F %T')] 全部完成"
