#!/bin/bash
# 实验 2（核心）：符号槽 + 连续关系槽，2+2 —— 连续关系表示到底有没有净贡献。
#
# 每块固定 4 个位置，切成 2 个符号位（原生 BPE 整词，内容精确）+ 2 个关系位
# （连续潜向量）。总位置数与"纯符号 4 位"完全相同，所以是**同预算**对照。
#
# 四条臂（同一批 200 题、同一份权重、同一 prompt 骨架）：
#   split_full  2 符号 + 2 关系（完整混合表示）
#   split_blank 2 符号 + 2 **无信息占位**（同位置、同范数，但不带内容）
#   split_sym   4 符号 + 0 关系（纯符号上界，位置数不变）
#   raw_word    原生整词 4 位（不训练，Query-Agnostic 挑词）
#
# 判据：split_full 必须**显著超过 split_blank**（关系槽的净贡献），并且最好超过
#       split_sym 与 raw_word。如果 full ≈ blank，说明多给的两个位置本身有用、
#       但"连续关系向量"没用 —— 那就是这条方法路线不成立的证据。
#
# ---- 用法 ----
#   bash runs/exp2_hybrid.sh                 # 训练 + 全部评测
#   CKPT=<已训好的 split 权重> SKIP_TRAIN=1 bash runs/exp2_hybrid.sh
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=${PY:-/data/zhangjiayang/miniconda3/envs/flux/bin/python}
MAX_UNIT=${MAX_UNIT:-8}
N_SYM=${N_SYM:-2}
N_REL=${N_REL:-2}
CKPT=${CKPT:-$B/ckpt/probe_split${N_SYM}x${N_REL}_mut${MAX_UNIT}.pt}
LIMIT=${LIMIT:-200}
OUT=${OUT:-$B/results/exp2}
mkdir -p "$OUT"
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-1}
cd "$R" || exit 1

# ---- 0) 关系绑定子集（只算一次；判定与训练无关，纯文本启发式）----
if [ ! -f "$OUT/binding_labels.json" ]; then
  echo "[exp2 $(date '+%F %T')] 划分关系绑定子集"
  "$PY" tools/make_binding_subset.py --data $B/synth/hotpot_val500.jsonl --limit "$LIMIT" \
    --out-labels "$OUT/binding_labels.json" --out-subset "$OUT/binding_subset.jsonl" \
    > "$OUT/binding.log" 2>&1
  echo "[exp2] 绑定子集 rc=$? -> $OUT/binding_labels.json"
fi

# ---- 1) 训练 2+2 ----
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  echo "[exp2 $(date '+%F %T')] 训练 split ${N_SYM}+${N_REL}（max_unit=${MAX_UNIT}）-> $CKPT"
  OUT=$CKPT MAX_UNIT=$MAX_UNIT COMPOSE=split N_SYM=$N_SYM N_REL=$N_REL REL_MODE=latent \
    STEPS=${STEPS:-1000} COV=0 LEX_HEAD=0 \
    bash runs/train_arm.sh > "$CKPT.train.log" 2>&1
  echo "[exp2 $(date '+%F %T')] 训练结束 rc=$?"
fi

_eval() {   # $1=mode  $2=tag  $3=extra
  echo "[exp2 $(date '+%F %T')] eval $2"
  "$PY" eval_latent_table.py --mode "$1" --ckpt "$CKPT" \
    --data $B/synth/hotpot_val500.jsonl --limit "$LIMIT" --max-new-tokens 16 \
    --topk-per-hop 4 --slot-isolation 1 --stratify \
    --binding-labels "$OUT/binding_labels.json" \
    --base $B/models/Qwen3-4B-Instruct-2507 \
    --adapter $B/models/delta-mem_qwen3_4b_instruct \
    --label "exp2-$2" --save-json "$OUT/$2.json" $3 \
    > "$OUT/$2.log" 2>&1
  echo "[exp2 $(date '+%F %T')] $2 rc=$?"
}

# ---- 2) 三条混合臂（同一份权重，只换关系槽内容）----
_eval split_full  split_full  ""
_eval split_blank split_blank ""
_eval split_sym   split_sym   ""

# ---- 3) 原生整词基线（不训练；同预算、Query-Agnostic）----
echo "[exp2 $(date '+%F %T')] eval raw_word 基线"
CKPT=$CKPT LIMIT=$LIMIT TAG=exp2_raw ARMS=latent,raw_prefix,raw_salient,raw_word,raw_full,text,none \
  SWEEP=${SWEEP:-4,8,16} bash runs/exp_a_capacity.sh > "$OUT/raw.log" 2>&1
echo "[exp2] raw rc=$?"

# ---- 4) 配对比较 ----
echo "[exp2 $(date '+%F %T')] 配对比较"
"$PY" tools/cmp_readout.py "$OUT/split_blank.json" "$OUT/split_full.json" \
  --label-a split_blank --label-b split_full | tail -6
"$PY" tools/cmp_readout.py "$OUT/split_sym.json" "$OUT/split_full.json" \
  --label-a split_sym --label-b split_full | tail -6
echo "[exp2 $(date '+%F %T')] 全部完成 -> $OUT"
grep -hE "^\| \`?(split|oracle)" "$OUT"/*.log | tail -12
