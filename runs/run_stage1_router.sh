#!/bin/bash
# 阶段一：只训 Router（冻结主干 + 冻结特征），训完立刻算完整证据召回率。
# 目的：先把"召回率"从 15.6% 拉到 45%+，再谈读出。跑不动 45% 就不进阶段二。
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=/data/zhangjiayang/miniconda3/envs/flux/bin/python
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd $R || exit 1

$PY tools/train_router_stage1.py \
  --base $B/models/Qwen3-4B-Instruct-2507 \
  --adapter $B/models/delta-mem_qwen3_4b_instruct \
  --data $B/synth/hotpot_train_10k.jsonl \
  --val-data $B/synth/hotpot_val500.jsonl --val-limit 200 \
  --n-train 10000 --epochs 3 --accum 8 --lr 2e-3 \
  --m 4 --layer 18 --inner-dim 1024 --heads 8 --bottleneck 512 \
  --max-unit-tokens 32 --max-slots 0 \
  --router-kind hybrid --addr-gamma 0.5 --topk-per-hop 2 \
  --eval-every 1000 \
  --out $B/ckpt/router_stage1.pt
