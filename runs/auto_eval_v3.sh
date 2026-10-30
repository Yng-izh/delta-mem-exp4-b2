#!/bin/bash
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
export DELTA_MEM_SCAN_IMPL=torch PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd $R
PY=/data/zhangjiayang/miniconda3/envs/flux/bin/python

# 等两次训练都结束、两个 ckpt 都落盘（进程消失但文件没写完的窗口也要避开）
echo "[$(date +%H:%M:%S)] AUTO-EVAL 等待 v3 训练完成…"
while pgrep -f "train_latent_memory.py" >/dev/null 2>&1 \
      || [ ! -f $B/ckpt/hotpot_router_10k_v3.pt ] \
      || [ ! -f $B/ckpt/hotpot_bridge_10k_v3.pt ]; do sleep 60; done
sleep 60
echo "[$(date +%H:%M:%S)] v3 ckpt 就绪："
ls -la $B/ckpt/hotpot_*_10k_v3.pt

echo "[$(date +%H:%M:%S)] ==== BRIDGE-V3 EVAL (500 题) ===="
$PY eval_latent_table.py \
  --method bridge --ckpt $B/ckpt/hotpot_bridge_10k_v3.pt \
  --data $B/synth/hotpot_val500.jsonl --limit 500 --seed 0 \
  --base $B/models/Qwen3-4B-Instruct-2507 --adapter $B/models/delta-mem_qwen3_4b_instruct \
  --gpu 0 --layer 18 --max-unit-tokens 32 \
  --label bridge-v3 --save-json $B/results/exp1_bridge_v3.json 2>&1 | tail -12
echo "[$(date +%H:%M:%S)] ==== BRIDGE-V3 EVAL DONE ===="

echo "[$(date +%H:%M:%S)] ==== ROUTER-V3 EVAL (500 题) ===="
$PY eval_latent_table.py \
  --method router --ckpt $B/ckpt/hotpot_router_10k_v3.pt \
  --data $B/synth/hotpot_val500.jsonl --limit 500 --seed 0 \
  --base $B/models/Qwen3-4B-Instruct-2507 --adapter $B/models/delta-mem_qwen3_4b_instruct \
  --gpu 0 --layer 18 --max-unit-tokens 32 \
  --label router-v3 --save-json $B/results/exp1_router_v3.json 2>&1 | tail -12
echo "[$(date +%H:%M:%S)] ==== ROUTER-V3 EVAL DONE ===="
echo "[$(date +%H:%M:%S)] AUTO-EVAL ALL DONE"
