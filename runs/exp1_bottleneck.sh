#!/bin/bash
# 实验 1：定位压缩记忆的主要瓶颈 —— 符号损坏，还是连续表示读不出来。
#
# 两个对照（对应 EXPERIMENTS.md 实验 1 的 A / B）：
#   A. 原生 BPE 保真：同一个 `oracle_lex` 读出，只换词形编码口径
#        mean = 词内子词均值（词表里不存在的向量）
#        bpe  = 完整 BPE 整词向量按原文顺序铺进去（精确符号）
#      位置数**不变**（每块仍是 m 个），所以这是"同一预算换编码"的干净对照。
#   B. 纯连续语义训练：`--no-lex-skip` 从头训 1000 步，没有任何词形直通，
#      训练前向 / QA 损失 / 评测 oracle_sem 走同一个函数同一种输入。
#
# 判据：
#   A 显著改善精确事实（尤其日期/数字）-> 符号保真是必要条件，混合表示必须有符号槽
#   B 仍然贴着 none                    -> 连续表示"不可读"，需要专门解决可读性
#   两者都失败                          -> 压缩本身没救，先别做混合表示
#
# ---- 用法 ----
#   CKPT=$B/ckpt/probe_mut8_1000.pt bash runs/exp1_bottleneck.sh
#   只做 A：STAGE=A ...；只做 B：STAGE=B ...
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=${PY:-/data/zhangjiayang/miniconda3/envs/flux/bin/python}
CKPT=${CKPT:-$B/ckpt/probe_mut8_1000.pt}
STAGE=${STAGE:-AB}
LIMIT=${LIMIT:-200}
MAX_UNIT=${MAX_UNIT:-8}
OUT=${OUT:-$B/results/exp1}
mkdir -p "$OUT"
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-1}
cd "$R" || exit 1

_eval() {   # $1=mode  $2=tag  $3=extra
  echo "[exp1 $(date '+%F %T')] eval $2"
  "$PY" eval_latent_table.py --mode "$1" --ckpt "$CKPT" \
    --data $B/synth/hotpot_val500.jsonl --limit "$LIMIT" --max-new-tokens 16 \
    --topk-per-hop 4 --slot-isolation 1 --stratify \
    --base $B/models/Qwen3-4B-Instruct-2507 \
    --adapter $B/models/delta-mem_qwen3_4b_instruct \
    --label "exp1-$2" --save-json "$OUT/$2.json" $3 \
    > "$OUT/$2.log" 2>&1
  echo "[exp1 $(date '+%F %T')] $2 rc=$?"
}

if [ "${STAGE#*A}" != "$STAGE" ]; then
  # ---- A：同一读出、两种词形编码 ----
  _eval oracle_lex lex_mean "--lex-token-mode mean"
  _eval oracle_lex lex_bpe  "--lex-token-mode bpe"
  echo "[exp1] A 完成；配对比较："
  "$PY" tools/cmp_readout.py "$OUT/lex_mean.json" "$OUT/lex_bpe.json" \
    --label-a lex_mean --label-b lex_bpe | tail -6
fi

if [ "${STAGE#*B}" != "$STAGE" ]; then
  # ---- B：纯连续语义（无词形直通）从头训 1000 步 ----
  SEM=${SEM:-$B/ckpt/probe_puresem_mut${MAX_UNIT}.pt}
  if [ ! -f "$SEM" ]; then
    echo "[exp1 $(date '+%F %T')] 训练纯语义臂 -> $SEM"
    OUT=$SEM MAX_UNIT=$MAX_UNIT STEPS=1000 COV=0 LEX_HEAD=0 \
      bash runs/train_arm.sh > "$OUT.puresem_train.log" 2>&1
    echo "[exp1 $(date '+%F %T')] 训练结束 rc=$? -> $OUT.puresem_train.log"
  fi
  CKPT=$SEM _eval oracle_sem puresem ""
  CKPT=$SEM _eval none       puresem_none ""
fi
echo "[exp1 $(date '+%F %T')] 全部完成 -> $OUT"
