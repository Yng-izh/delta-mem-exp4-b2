#!/bin/bash
# E3：词形与语义**分开注入** —— [lex_1..lex_k] + [sem]，总预算仍是 n_slots。
#
# 动机：残差叠加 `E = z_lex + tanh(α)·Bridge(z_sem)` 把两种信息压在同一个位置，
# 语义只能以 α≈0.12 存在（再大就偏离词表流形）。分开注入给语义一个独立位置、
# 一个完整范数（整向量 L2 对齐到 target_norm ≈ 真实 embedding 范数 ≈1.087）。
#
# **只有 E1/E2 有明确收益时再做这一轮。** 对照组用前面表现最好的那版权重
# （同一套超参、同样 1000 步，只差 --split-inject / --split-n-lex）。
#
#   3 + 1（默认）：--split-inject 1 --split-n-lex 3
#   4 + 2（仅在 3+1 语义容量明显不足时才考虑）：--split-n-lex 4 --m 6
#   注意：6 token 与原来 4 token 的 EM **不能**直接当架构收益，必须同预算对照。
#
# 核心指标：oracle_latent 是否真的优于单分支（不是"token 多了所以分高了"）。
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=/data/zhangjiayang/miniconda3/envs/flux/bin/python
OUT=${OUT:-$B/ckpt/probe_e3_split.pt}
COV=${COV:-0.1}
LEX_HEAD=${LEX_HEAD:-1}
LEX_TOKEN_MODE=${LEX_TOKEN_MODE:-mean}     # E2 若证明 bpe 更好，这里改 bpe
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

if [ "$SPLIT_N_LEX" -ge "$M" ]; then
  echo "[warn] --split-n-lex($SPLIT_N_LEX) >= m($M)：语义 token 没有位置了" >&2
fi
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
  --split-inject 1 --split-n-lex $SPLIT_N_LEX \
  --alpha-max 0.5 --lex-gate-init 0.05 --diag-every 50 \
  --inject-mode user --fence-style bracket \
  --m $M --layer 18 --inner-dim 1024 --heads 8 --bottleneck 512 \
  --max-unit-tokens 32 --max-slots 0 --lr 2e-4 \
  --out "$OUT" --save-every $SAVE_EVERY $RESUME_ARG
