#!/bin/bash
# E2：原生 BPE 直通 vs 词内子词均值（**零训练**，复用 E1 的 checkpoint）。
#
# 只改"一个词变成什么向量"：
#   mean（A，现行为）：该词全部子词 h0 取均值 -> 1 个向量
#   bpe （B，新口径）：按原文顺序复制**完整** BPE 向量，每块最多 m 个 token，
#                      整词放不下就跳过（绝不截断单词），有效词数可能变少
#
# 注意这是一处**复合对照**：聚合方式与容量分配同时变了。它回答的是
# "固定预算下哪种词形表示更有效"，不能把两者的贡献分开算。
#
# 重点看：
#   1. oracle_lex 的 EM（cmp_readout 会给配对 McNemar）；
#   2. 完整答案硬覆盖率、数字/专有名词的精确恢复（符号保真 json 里的
#      full_answer / numbers，以及日志里打印的 1838 / YG Entertainment 这两类样本）；
#   3. 每块保留了几个完整词（日志里的 rows_used / words_kept 均值）。
#
# 判据：bpe 若提高 oracle_lex -> 纳入训练架构；持平或下降 -> 不必再考虑 STE。
#
# ---- 用法 ----
#   CKPT=$B/ckpt/probe_e1_lexhead.pt bash runs/exp_e2_lex_tokens.sh
#   可选：LIMIT=200 GPU=0 TAG=名字 MODES="lex latent"
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=/data/zhangjiayang/miniconda3/envs/flux/bin/python
CKPT=${CKPT:?必须显式指定 CKPT（E2 用 E1 的权重，不重训）}
LIMIT=${LIMIT:-200}
TAG=${TAG:-$(basename "$CKPT" .pt)}
OUT=${OUT:-$B/results/e2_lex_tokens/$TAG}
MODES=${MODES:-"lex latent"}
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU:-0}
mkdir -p "$OUT"
cd "$R" || exit 1

# oracle_lex / oracle_latent 名字 -> eval_latent_table 的 --mode
_mode() {
  case "$1" in
    lex) echo "oracle_lex" ;;
    latent) echo "oracle_latent" ;;
    sem) echo "oracle_sem" ;;
    *) echo "$1" ;;
  esac
}

for m in $MODES; do
  mode=$(_mode "$m")
  for tok in mean bpe; do
    echo "[e2 $(date '+%F %T')] $mode / $tok 开始"
    "$PY" eval_latent_table.py --mode "$mode" --ckpt "$CKPT" \
      --data $B/synth/hotpot_val500.jsonl --limit "$LIMIT" --max-new-tokens 16 \
      --topk-per-hop 4 --slot-isolation 1 --progress-every 50 \
      --lex-token-mode "$tok" \
      --base $B/models/Qwen3-4B-Instruct-2507 \
      --adapter $B/models/delta-mem_qwen3_4b_instruct \
      --label "e2-${TAG}-${m}-${tok}" \
      --save-json "$OUT/${m}_${tok}.json" > "$OUT/${m}_${tok}.log" 2>&1
    echo "[e2 $(date '+%F %T')] $mode / $tok 结束 rc=$?"
  done
  echo "[e2 $(date '+%F %T')] $m 配对比较（mean vs bpe）"
  "$PY" tools/cmp_readout.py "$OUT/${m}_mean.json" "$OUT/${m}_bpe.json" \
    --label-a "mean" --label-b "bpe" || true
done

# 符号保真：硬选择下"完整答案覆盖 / 数字 / 专有名词"和"每块保留几个完整词"
for tok in mean bpe; do
  echo "[e2 $(date '+%F %T')] 符号保真 / $tok 开始"
  "$PY" tools/diag_symbol_fidelity.py --ckpt "$CKPT" \
    --data $B/synth/hotpot_val500.jsonl --limit "$LIMIT" --scan 300 \
    --lex-token-mode "$tok" \
    --base $B/models/Qwen3-4B-Instruct-2507 \
    --adapter $B/models/delta-mem_qwen3_4b_instruct \
    --save-json "$OUT/symbol_${tok}.json" > "$OUT/symbol_${tok}.log" 2>&1
  echo "[e2 $(date '+%F %T')] 符号保真 / $tok 结束 rc=$?"
done
echo "[e2 $(date '+%F %T')] 全部完成：$OUT"
