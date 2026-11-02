#!/bin/bash
# watch_v4_eval.sh —— 等 v4 训练进程退出后，自动跑评测（无人值守）。
#
# 为什么要在服务器上挂着等、而不是手动跑：
#   训练约 2 小时，跑完那一刻没人盯着；自动接力能保证"训练用的就是被评测的那个 ckpt"，
#   不会出现"训练完忘了评"或"评的是中途 ckpt"。
#
# 跑两个模式：
#   router        —— 主方法，端到端 EM（与 v3 的 router 同表对照）
#   oracle_latent —— 金标槽位直送 bridge，读出上界（v3 上是 17.0%，用来判断
#                    压缩保真有没有被 v4 的 inner_dim 1024 + h0 直通修好）
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=/data/zhangjiayang/miniconda3/envs/flux/bin/python
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=1          # GPU 0 常年被别的任务占着
cd $R

echo "[watch] $(date '+%F %T') 开始等待训练进程退出"
while pgrep -f "train_latent_memory.py --router" >/dev/null 2>&1; do sleep 60; done
echo "[watch] $(date '+%F %T') 训练进程已退出"
tail -3 $B/logs/train_v4.log

if [ ! -f $B/ckpt/hotpot_router_10k_v4.pt ]; then
  echo "[watch] 找不到 $B/ckpt/hotpot_router_10k_v4.pt，终止"
  exit 1
fi

run() {
  local mode=$1 label=$2 out=$3
  echo "[watch] $(date '+%T') 开始 mode=$mode"
  $PY eval_latent_table.py --mode "$mode" \
    --ckpt $B/ckpt/hotpot_router_10k_v4.pt \
    --data $B/synth/hotpot_val500.jsonl --limit 500 --max-new-tokens 16 --gpu 0 \
    --base $B/models/Qwen3-4B-Instruct-2507 \
    --adapter $B/models/delta-mem_qwen3_4b_instruct \
    --label "$label" --save-json "$out"
  echo "[watch] $(date '+%T') 结束 mode=$mode rc=$?"
}

run router        v4-router        $B/results/v4_router_500.json
run oracle_latent v4-oracle_latent $B/results/v4_oracle_latent_500.json
echo "[watch] $(date '+%F %T') 全部完成"
