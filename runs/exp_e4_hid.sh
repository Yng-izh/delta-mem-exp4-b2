#!/bin/bash
# E4（条件性）：隐状态蒸馏 —— 明文 teacher 与潜层 student 在**答案生成前那一位**对齐。
#
# **只在 E3 发现"独立语义 token 仍然表达不了关系"时才做。**
# 先只对齐**最后一层**那一个位置（--hid-layer last），不同时加第 18 层：
# 两处一起改，出问题分不清是谁的锅。
# 保留现有 QA / KL / 重建监督；不引入新的加权重建（recon_loss_units 已有实体加权）。
#
# 对照：HID=0 与 HID=0.1 各 1000 步，其余超参逐字相同。
#   HID=0.0 OUT=$B/ckpt/probe_e4_base.pt     bash runs/exp_e4_hid.sh
#   HID=0.1 OUT=$B/ckpt/probe_e4_hid0p1.pt   bash runs/exp_e4_hid.sh
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=/data/zhangjiayang/miniconda3/envs/flux/bin/python
OUT=${OUT:-$B/ckpt/probe_e4_hid.pt}
HID=${HID:-0.1}
HID_LAYER=${HID_LAYER:-last}
COV=${COV:-0.1}
LEX_HEAD=${LEX_HEAD:-1}
LEX_TOKEN_MODE=${LEX_TOKEN_MODE:-mean}
SPLIT_INJECT=${SPLIT_INJECT:-0}
SPLIT_N_LEX=${SPLIT_N_LEX:-3}
M=${M:-4}
RESUME=${RESUME:-0}
STEPS=${STEPS:-1000}
SAVE_EVERY=${SAVE_EVERY:-100}
EVAL_EVERY=${EVAL_EVERY:-500}
TEACHER_PROBE=${TEACHER_PROBE:-16}
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd $R || exit 1

RESUME_ARG=""
if [ "$RESUME" = "1" ]; then
  if [ ! -f "$OUT" ]; then echo "[resume] 找不到 $OUT" >&2; exit 1; fi
  RESUME_ARG="--resume $OUT"
  echo "[resume] 从 $OUT 接着训"
fi

$PY train_latent_memory.py --router \
  --base $B/models/Qwen3-4B-Instruct-2507 \
  --adapter $B/models/delta-mem_qwen3_4b_instruct \
  --data $B/synth/hotpot_train_10k.jsonl --val-data $B/synth/hotpot_val500.jsonl \
  --n-train 10000 --n-val 120 --epochs 2 --accum 4 --pack 4 \
  --max-steps $STEPS \
  --eval-every $EVAL_EVERY --eval-limit 40 --log-every 10 --enc-batch 24 \
  --recon-weight 0.5 --recon-units 2 --recon-lex 1 --addr-weight 1.0 --kl-weight 1.0 \
  --gold-readout-ratio 0.5 --readout-mode mixed --router-readout-p 0.5 --hard-neg-k 2 \
  --teacher-scope gold --teacher-probe-n $TEACHER_PROBE \
  --topk-per-hop 4 \
  --router-kind hybrid --addr-gamma 0.5 --addr-maxsim 1 \
  --init-router $B/ckpt/router_stage1_8slot.pt --freeze-router \
  --ent-weight 5.0 --stop-weight 0.2 --recon-corrupt 0.5 \
  --cov-weight $COV --cov-targets ${COV_TARGETS:-0} \
  --lex-head $LEX_HEAD --lex-token-mode $LEX_TOKEN_MODE \
  --split-inject $SPLIT_INJECT --split-n-lex $SPLIT_N_LEX \
  --hid-weight $HID --hid-layer $HID_LAYER \
  --alpha-max 0.5 --lex-gate-init 0.05 --diag-every 50 \
  --inject-mode user --fence-style bracket \
  --m $M --layer 18 --inner-dim 1024 --heads 8 --bottleneck 512 \
  --max-unit-tokens 32 --max-slots 0 --lr 2e-4 \
  --out "$OUT" --save-every $SAVE_EVERY $RESUME_ARG
