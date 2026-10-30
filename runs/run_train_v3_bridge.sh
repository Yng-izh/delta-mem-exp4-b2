#!/bin/bash
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd $R

# 严格串行：必须等 router 的 ckpt 落盘**且** router 进程退出后再开。
# 只等进程是不够的 —— router 启动初期还在自己的等待循环里，pgrep 会瞬间返回空，
# 两个训练一旦并行就是 21+15 GiB 挤 39 GiB 卡的 OOM（上一轮实测）。
echo "[$(date +%H:%M:%S)] 等待 router 训练完成…"
while pgrep -f "train_latent_memory.py --router" >/dev/null 2>&1 \
      || [ ! -f $B/ckpt/hotpot_router_10k_v3.pt ]; do sleep 20; done
sleep 30
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "[$(date +%H:%M:%S)] BRIDGE-10K-V3 START (全槽块级保真读出 + 槽位边界锚点)"
/data/zhangjiayang/miniconda3/envs/flux/bin/python train_latent_memory.py \
  --base $B/models/Qwen3-4B-Instruct-2507 \
  --adapter $B/models/delta-mem_qwen3_4b_instruct \
  --data $B/synth/hotpot_train_10k.jsonl --val-data $B/synth/hotpot_val500.jsonl \
  --n-train 10000 --n-val 120 --epochs 1 --accum 4 --pack 4 \
  --eval-every 250 --eval-limit 40 --log-every 10 \
  --enc-batch 24 \
  --recon-weight 0.3 --recon-units 2 --addr-weight 1.0 --kl-weight 0 \
  --gold-readout-ratio 0.7 --ent-weight 5.0 --stop-weight 0.2 --recon-corrupt 0.5 \
  --m 4 --layer 18 --inner-dim 512 --heads 8 --bottleneck 512 \
  --max-unit-tokens 32 --max-slots 0 --lr 2e-4 \
  --out $B/ckpt/hotpot_bridge_10k_v3.pt --save-every 250
echo "[$(date +%H:%M:%S)] BRIDGE-10K-V3 DONE"
