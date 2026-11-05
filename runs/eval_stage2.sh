#!/bin/bash
# 阶段二产物的评测：默认先端到端 router，再 oracle_latent 读出上界（完全绕过 Router）。
# 两个数一对照就能分出"寻址错"与"读出/压缩丢信息"——差得多是寻址，差得少是读出。
#
# ---- 用法 ----
#   CKPT=$B/ckpt/<权重>.pt bash runs/eval_stage2.sh
#   只跑某几条读法：MODES="oracle_lex oracle_sem oracle_latent" CKPT=... bash runs/eval_stage2.sh
#   可选：TAG=名字（默认取 ckpt 文件名）LIMIT=题数（默认 500）GPU=0/1
#
# 只跑"读法"（不给 router 权重时）用这条更直接：
#   python eval_latent_table.py --mode oracle_sem --ckpt <ckpt> --limit 200 --stratify
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=${PY:-/data/zhangjiayang/miniconda3/envs/flux/bin/python}
CKPT=${CKPT:?必须显式指定 CKPT（没有默认权重了）}
TAG=${TAG:-$(basename "$CKPT" .pt)}
LIMIT=${LIMIT:-500}
MODES=${MODES:-"router oracle_latent"}
mkdir -p "$B/results" "$B/logs"
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd "$R" || exit 1

for mode in $MODES; do
  echo "[eval $(date '+%F %T')] $mode 开始"
  "$PY" eval_latent_table.py --mode "$mode" --ckpt "$CKPT" \
    --data $B/synth/hotpot_val500.jsonl --limit "$LIMIT" --max-new-tokens 16 \
    --topk-per-hop 4 --stratify \
    --base $B/models/Qwen3-4B-Instruct-2507 \
    --adapter $B/models/delta-mem_qwen3_4b_instruct \
    --label "$TAG-$mode" --save-json "$B/results/${TAG}_${mode}_${LIMIT}.json" \
    > "$B/logs/eval_${TAG}_${mode}.log" 2>&1
  echo "[eval $(date '+%F %T')] $mode 结束 rc=$?"
done
echo "[eval $(date '+%F %T')] 全部完成 -> $B/results/${TAG}_*_${LIMIT}.json"
