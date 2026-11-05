#!/bin/bash
# 实验 3：把实验 2 里有效的表示放回**完整系统**（Router + LatentSlotBank）。
#
# 目的：证明实验 2 的优势在真实检索 + 多跳推理下仍然存在，而且**不能由 Router 的
# 召回差异解释**。所以必须同时报三列：
#   - 端到端 EM/F1（router）
#   - Router 完整证据召回率（read_acc / full_recall_leq2）
#   - **命中完整证据时的条件问答准确率**（acc_hit）—— 固定"检索已成功"这个条件，
#     剩下的差才是读出端的差。
#
# 对照臂 `router_raw`：**同一次寻址**，只把载荷换成同预算的原生整词。
# 这样两条线只差载荷形态，端到端差距不可能来自"router 选得不一样"。
#
# ---- 用法 ----
#   CKPT=$B/ckpt/probe_split2x2_mut8.pt bash runs/exp3_e2e.sh
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=${PY:-/data/zhangjiayang/miniconda3/envs/flux/bin/python}
CKPT=${CKPT:?必须显式指定 CKPT（实验 2 训出来的 split 权重）}
LIMIT=${LIMIT:-200}
OUT=${OUT:-$B/results/exp3}
mkdir -p "$OUT"
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-1}
cd "$R" || exit 1

_eval() {   # $1=mode  $2=tag  $3=extra
  echo "[exp3 $(date '+%F %T')] eval $2"
  "$PY" eval_latent_table.py --mode "$1" --ckpt "$CKPT" \
    --data $B/synth/hotpot_val500.jsonl --limit "$LIMIT" --max-new-tokens 16 \
    --topk-per-hop 4 --slot-isolation 1 --stratify \
    ${BIND:+--binding-labels "$BIND"} \
    --base $B/models/Qwen3-4B-Instruct-2507 \
    --adapter $B/models/delta-mem_qwen3_4b_instruct \
    --label "exp3-$2" --save-json "$OUT/$2.json" $3 \
    > "$OUT/$2.log" 2>&1
  echo "[exp3 $(date '+%F %T')] $2 rc=$?"
}

# 关系绑定子集（若实验 2 已经生成过就直接复用）
BIND=${BIND:-$B/results/exp2/binding_labels.json}
if [ ! -f "$BIND" ]; then
  "$PY" tools/make_binding_subset.py --data $B/synth/hotpot_val500.jsonl --limit "$LIMIT" \
    --out-labels "$BIND" > "$OUT/binding.log" 2>&1
fi

_eval router      router      ""
_eval router_raw  router_raw  ""
_eval oracle_latent oracle_latent ""      # 读出上界（寻址全对），用来分离两段损失
_eval none        none        ""

echo "[exp3 $(date '+%F %T')] 配对比较（router vs router_raw，同一次寻址）"
"$PY" tools/cmp_readout.py "$OUT/router.json" "$OUT/router_raw.json" \
  --label-a router --label-b router_raw | tail -6

echo "[exp3 $(date '+%F %T')] 三列汇总（EM / 完整召回 / 命中条件准确率）"
for f in router router_raw oracle_latent none; do
  [ -f "$OUT/$f.json" ] || continue
  "$PY" - "$OUT/$f.json" "$f" <<'PYEOF'
import json, sys
blob = json.load(open(sys.argv[1]))
s = blob.get("summary") or {}
def p(x):
    return "n/a" if x is None else f"{100*x:.1f}"
print(f"{sys.argv[2]:<14} EM={p(s.get('em'))}%  F1={p(s.get('f1'))}%  "
      f"完整召回={p(s.get('full_recall_leq2'))}%  "
      f"命中条件EM={p(s.get('acc_hit'))}% (n={s.get('n_hit')})")
PYEOF
done
echo "[exp3 $(date '+%F %T')] 全部完成 -> $OUT"
