#!/bin/bash
# run_train_v4.sh —— v4 配方（对照 v3 的四处改动，逐条对应诊断 P01~P12）
#
# 与 v3 的差异（**每一条都是为了让训练分布等于推理分布**）：
#   1) --kl-weight 1.0（v3 是 0）：蒸馏回来看住"对齐到冻结模型"这个约束；
#      --pack 4 仍然开着 —— 打包与蒸馏现在可以并存（teacher 一起打包，
#      只在答案位置算 KL），不再二选一。
#   2) --recon-weight 0.5（v3 是 0.3）：压缩保真给更高权重，P06 的落点。
#   3) --epochs 3（v3 是 1）：v3 的 10k×1ep 在 step 2500 就停了，
#      而金标门控让**训练里从没出现过干扰槽**，EM 从 v2 的 16.4 掉到 12.8。
#   4) --hard-neg-k 2 + --gold-readout-ratio 0.1：前 10% 步数纯金标热启动，
#      之后每一步都混入 router 打分最高的 2 个**非金标**槽并打乱槽序 —— 把
#      "在 4 槽里无视 2 个高相似干扰"变成每一步都被训练到的东西。
#   5) --router-kind hybrid + --addr-gamma 0.5：寻址键改由冻结 h0/h18 构造，
#      h0 Token-MaxSim 打底（不需要训练就有的字面匹配能力）。
#   6) --inner-dim 1024 + h0 直通（默认开，--no-lex-skip 可关做消融）：
#      inner_dim 512 是"双向注意力压到 m=4 个槽"的瓶颈，P06 的量化证据在那儿。
#   7) --topk-per-hop 2：两跳共 4 槽读出（旧实现上限 2 槽，多跳题物理上读不全）。
#
# 显存/时间：--pack 4 --accum 4，enc-batch 24。
# 若要把槽位块对角隔离真正打开（需要 eager 注意力，sdpa 会静默丢弃 4D 掩码）：
#   在 _load_backbone 里把 attn_implementation 改成 "eager" 并加 --slot-isolation 1；
#   隔离是否真的生效会写进每条样本 meta["slot_isolation"]，不会假装成功。
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 用哪张卡由外部指定：GPU=1 bash runs/run_train_v4.sh（默认 0）。
# 本机 GPU 0 常被其他任务占满，硬编码 0 会在加载 4B 权重时直接 OOM。
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd $R

while pgrep -f "eval_latent_table.py --method" >/dev/null 2>&1; do sleep 20; done
while pgrep -f "train_latent_memory.py --router" >/dev/null 2>&1; do sleep 20; done
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "[$(date +%H:%M:%S)] ROUTER-10K-V4 START (金标+难负例 / hybrid 键 / h0 直通 / 4 槽读出 / kl=1.0)"
/data/zhangjiayang/miniconda3/envs/flux/bin/python train_latent_memory.py --router \
  --base $B/models/Qwen3-4B-Instruct-2507 \
  --adapter $B/models/delta-mem_qwen3_4b_instruct \
  --data $B/synth/hotpot_train_10k.jsonl --val-data $B/synth/hotpot_val500.jsonl \
  --n-train 10000 --n-val 120 --epochs 3 --accum 4 --pack 4 \
  --eval-every 250 --eval-limit 40 --log-every 10 \
  --enc-batch 24 \
  --recon-weight 0.5 --recon-units 2 --addr-weight 1.0 --kl-weight 1.0 \
  --gold-readout-ratio 0.1 --hard-neg-k 2 --topk-per-hop 4 \
  --router-kind hybrid --addr-gamma 0.5 --addr-maxsim 1 \
  --ent-weight 5.0 --stop-weight 0.2 --recon-corrupt 0.5 \
  --m 4 --layer 18 --inner-dim 1024 --heads 8 --bottleneck 512 \
  --max-unit-tokens 32 --max-slots 0 --lr 2e-4 \
  --out $B/ckpt/hotpot_router_10k_v4.pt --save-every 250
echo "[$(date +%H:%M:%S)] ROUTER-10K-V4 DONE"
