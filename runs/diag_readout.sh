#!/bin/bash
# P0-2：Oracle 读出分解（200 题，五种读法同一批数据、同一预算口径）
#   明文 oracle / 原始词嵌入 oracle_embed / 只用 z_lex / 只用 Bridge(z_sem) / 当前 oracle_latent
# 判定路径（见 results/ 里对应的 README）：
#   明文高、embed 低      -> 查注入位置/软 token 预算/上下文格式
#   embed 高、lex 低      -> 查 Compressor 的选词与实体覆盖
#   lex 高、latent 低     -> 查 Bridge 残差是否破坏原生词向量
#   全部潜层都低          -> 信息容量/实体绑定问题，转 P1-4 过拟合实验
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=/data/zhangjiayang/miniconda3/envs/flux/bin/python
CKPT=$B/ckpt/hotpot_router_10k_v5.pt
OUT=${OUT:-$B/results/diag_readout_200}
mkdir -p "$OUT"
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd "$R" || exit 1

run() {
  local mode=$1
  echo "[diag $(date '+%F %T')] $mode 开始"
  # --ckpt 对 oracle_embed 也要传：它按 ckpt 的 max_unit_tokens/m 对齐预算，
  # 不传就会用 CLI 默认 64，与训练时的 32 不一致，两行不可比。
  "$PY" eval_latent_table.py --mode "$mode" --ckpt "$CKPT" \
    --data $B/synth/hotpot_val500.jsonl --limit 200 --max-new-tokens 16 \
    --topk-per-hop 4 --slot-isolation 1 --progress-every 25 \
    --base $B/models/Qwen3-4B-Instruct-2507 \
    --adapter $B/models/delta-mem_qwen3_4b_instruct \
    --label "diag-$mode" --save-json "$OUT/$mode.json" \
    > "$OUT/$mode.log" 2>&1
  echo "[diag $(date '+%F %T')] $mode 结束 rc=$?  -> $OUT/$mode.json"
}

# MODES="oracle_embed" 可只重跑某一档（例如改了预算口径后只需重算 oracle_embed）
for m in ${MODES:-"oracle oracle_embed oracle_lex oracle_sem oracle_latent"}; do
  run "$m"
done
echo "[diag $(date '+%F %T')] 全部完成：$OUT"
