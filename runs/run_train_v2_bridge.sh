#!/bin/bash
# run_train_v2_bridge.sh —— 实验一「Compressor + Bridge」分支训练（KV cache 补丁后的新一版）
#
# 与 run_train_v2.sh（Router 分支）的关系：
#   逐字相同，唯一实质差异是**不带 --router**。为方便 diff，其余参数（含 --addr-weight）
#   保持字面一致；命中顺序也刻意与 router 脚本一致，避免"看起来一样其实不同序"。
#
#   为什么要拆成两条训练：`--router` 打开的是 LatentRouter 集合内容寻址 + 寻址交叉熵
#   监督（train_latent_memory.py:2393）。Bridge 分支没有这套检索头，两者不能共用一个
#   ckpt —— 共用会让其中一条变成另一条的退化版，对照就失效了。公平对照的前提是：
#   同数据、同 questions、同 epoch/步数、同 max-unit-tokens(32)，只差这一个开关。
#
# 串行理由：GPU0 一次只能跑一个训练（ckpt 峰值 ~15G，卡被别人占着一部分）。本脚本默认
#   先等 Router 那条结束再开跑，所以现在挂着不占资源。
#
# 用法：
#   bash run_train_v2_bridge.sh > logs/train_10k_v2_bridge.log 2>&1 &      # 自动等 router
#   WAIT_ROUTER=0 bash run_train_v2_bridge.sh ...                          # 不等，直接跑
#   WAIT_ROUTER=0 GRAD_CKPT=1 bash ...                                     # 与另一条训练并行时省显存
#
# 并行注意：pack 4 的单条训练峰值约 15G，GPU0 只有 40G 且常驻别人的 ~6G 进程，
#   两条训练并行会顶到 ~36G。GRAD_CKPT=1 用重算换显存（数值等价，只影响步时），
#   是并行时唯一安全的做法；对 bridge/router 公平对照无影响。
#
# 训练完的评测（口径与 router 一致，同一份 500 题）：
#   python eval_latent_table.py --method bridge --ckpt ckpt/hotpot_bridge_10k_v2.pt \
#     --data synth/hotpot_val500.jsonl --limit 500 --seed 0 \
#     --base models/Qwen3-4B-Instruct-2507 --adapter models/delta-mem_qwen3_4b_instruct \
#     --gpu 0 --label bridge --save-json results/exp1_bridge.json

set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd $R

# 等 Router 分支训练退出（只匹配训练进程，不匹配评测）
if [ "${WAIT_ROUTER:-1}" != "0" ]; then
  echo "[$(date +%H:%M:%S)] 等待 Router 分支训练结束…"
  while pgrep -f "train_latent_memory.py --router" >/dev/null 2>&1; do sleep 30; done
  sleep 10
fi

EXTRA=""
if [ "${GRAD_CKPT:-0}" != "0" ]; then
  EXTRA="--grad-checkpointing"
  echo "[mem] GRAD_CKPT=1：开启 gradient checkpointing（并行训练省显存）"
fi

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "[$(date +%H:%M:%S)] BRIDGE-10K-V2 START"

/data/zhangjiayang/miniconda3/envs/flux/bin/python train_latent_memory.py \
  --base $B/models/Qwen3-4B-Instruct-2507 \
  --adapter $B/models/delta-mem_qwen3_4b_instruct \
  --data $B/synth/hotpot_train_10k.jsonl --val-data $B/synth/hotpot_val500.jsonl \
  --n-train 10000 --n-val 120 --epochs 1 --accum 4 --pack 4 \
  --eval-every 250 --eval-limit 40 --log-every 10 \
  --enc-batch 24 \
  --recon-weight 0.3 --recon-units 2 --addr-weight 1.0 --kl-weight 0 \
  --m 4 --layer 18 --inner-dim 512 --heads 8 --bottleneck 512 \
  --max-unit-tokens 32 --max-slots 0 --lr 2e-4 \
  --out $B/ckpt/hotpot_bridge_10k_v2.pt --save-every 250 $EXTRA
# 注：--router 缺失即 Bridge 分支；--addr-weight 在无 router 时不产生损失项，
#     这里保留 1.0 只是为了让本文件与 run_train_v2.sh 逐行可 diff，不代表 bridge 受了寻址监督。
#     --prefetch 同样未开：delta_write_disabled 改的是模块级共享 bool，无线程隔离，
#     在写入策略仍是实验变量时不冒这个险（代价约 +15~20% 步时）。

echo "[$(date +%H:%M:%S)] BRIDGE-10K-V2 DONE rc=$?"
