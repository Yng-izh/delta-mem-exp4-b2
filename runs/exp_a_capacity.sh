#!/bin/bash
# 实验 A：读出接口的**能力上限**（不训练，纯对照）。
#
# 问题：把金标事实写成潜层记忆之后读不回来，是"冻结模型只肯从几十个向量里读"
# 这个接口本身的上限，还是我们的压缩器在这几十个向量里塞得比上限更少？
#
# 做法：同一道题、同一个 prompt 骨架、**逐槽位相同的 token 数**，只换"那些位置
# 上放什么向量"：
#   latent       当前方法（z_lex + α·Bridge(z_sem)）
#   raw_prefix   金标原文的前 c_j 个真实词向量（傻截断，会把句尾新值切掉）
#   raw_salient  同等预算下按显著性挑词、按原序拼回（**判断容量主看这条**）
#   raw_full     整条原文的真实词向量全给（不压缩的上限）
#   text         金标原文直接写进 system（参照上界）
#   none         不给记忆（下界）
#
# 判据（写在 EXPERIMENTS.md「实验 A」）：
#   raw_salient ≈ latent      -> 接口/容量是主要限制，优化编码器收益有限
#   raw_salient ≫ latent      -> 压缩确实丢了东西，值得继续做编码保真
#   raw_full 也上不去          -> 连"给足原生 token"都救不回来，问题在读出协议
#
# 预算口径：raw_* 每槽的 token 数**逐槽等于** latent 的内容向量数（不是平均值，
# 不是按块数公式估算），围栏用同一个字符串、同一个位置；不一致会直接报错。
#
# ---- 用法 ----
#   CKPT=$B/ckpt/probe_e1_lexhead.pt bash runs/exp_a_capacity.sh
#   ARMS=latent,raw_salient LIMIT=50 bash runs/exp_a_capacity.sh   # 先跑小样本
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=${PY:-/data/zhangjiayang/miniconda3/envs/flux/bin/python}
CKPT=${CKPT:?必须显式指定 CKPT}
TAG=${TAG:-exp_a}
LIMIT=${LIMIT:-200}
ARMS=${ARMS:-none,text,latent,raw_prefix,raw_salient,raw_full}
OUT=${OUT:-$B/results/$TAG}
mkdir -p "$OUT"
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd "$R" || exit 1

echo "[exp_a $(date '+%F %T')] ckpt=$CKPT limit=$LIMIT arms=$ARMS -> $OUT"
$PY tools/probe_readout_capacity.py \
  --ckpt "$CKPT" \
  --data $B/synth/hotpot_val500.jsonl --limit "$LIMIT" --arms "$ARMS" \
  --max-new-tokens 16 --slot-isolation 1 \
  --base $B/models/Qwen3-4B-Instruct-2507 \
  --adapter $B/models/delta-mem_qwen3_4b_instruct \
  --progress-every 25 --partial-every 25 \
  --save-json "$OUT/capacity.json" \
  > "$OUT/capacity.log" 2>&1
echo "[exp_a $(date '+%F %T')] 结束 rc=$? -> $OUT/capacity.json"
tail -40 "$OUT/capacity.log"
