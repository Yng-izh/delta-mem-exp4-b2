# KNOWN_ISSUES.md — 当前仍未解决的问题

**最后更新**：2026-11-04（读出端对齐轮）

只登记**确实存在、已确认、但尚未修复**的问题，每条写「怎么发现的 / 影响什么 / 怎么修」。
已修的问题记在 [`CHANGELOG.md`](CHANGELOG.md)，不再在这里重复。

---

## P0 — 会影响实验结论

### 1. `v5-router` 的 500 题数字作废，尚未重跑

**现象**：`results/v5_2026-11-04/v5_router_500.json` 里 `read_acc=6.0%`，而同一份权重在
阶段一单独评测是 52.0%（28/28 张量已逐位核对相同）。

**根因**（已修，`3815379`）：`encode_item_slots_full()` 的缓存分支返回 `kfeat=None`；
评测脚本传 `cache={}`，于是 `HybridLatentRouter` 退回"潜载荷键"，MaxSim 与新信息抑制全部失效。

**现状**：`runs/diag_router_cache.sh`（逐题对拍）与 `runs/eval_v5.sh`（重跑 500 题）**都还没执行**。
在看到这两个结果之前，任何"Router 行/不行"的结论都不成立。

### 2. 训练侧的槽位隔离只有围栏，没有块对角

**位置**：`train_latent_memory.py::build_student_inputs`（不接受 4D 掩码）

**现象**：训练期 `--slot-isolation 1` 只让 `compose_routed_soft` 插入 `-- Fact j --` 前后围栏；
`build_block_diag_mask` 只在**推理**的 `_emit_latent` 里生效。

**影响**：当前训练与评测都用 SDPA，两端都只有围栏，**暂时一致**。一旦把评测切到
`attn_implementation="eager"`，评测会真的做块对角隔离而训练从未做过，两端口径分叉，
掉点会被误读成"隔离有害"。

**修法**：让 `build_student_inputs` 接收 `seg_ids` 并把 4D 加性掩码传给主干（需先确认 HF
eager 路径接受 4D `attention_mask`），或显式禁止"训练 SDPA + 评测 eager"的组合。

### 3. 残差上限 0.5 只是先验，没有实验支撑

`--alpha-max 0.5` 的依据是"v5 的 α=0.264 已在退化"，但 0.5 是否合适、是否仍偏大，
需要 `runs/overfit_probe.sh` 或一轮短训练给证据。它现在是一条**未验证的约束**。

---

## P1 — 会让指标解释变模糊

### 4. `encode_keys` 层没有真正的"锚点文本过滤"

标题区加成（前 3 个有效 token）用 `h18` 偏离度做**代理**，没有把"这一槽的标题文本"显式
送进键编码。跨槽同实体（"张三-雇主" vs "张三-居住地"）时，键仍可能靠池化语义相似度区分，
而不是靠标题字面。修法：把标题串单独编码成一路键，或在键编码里加字面匹配项。

### 5. `--fst` 分支不支持断点续训

阶段一（`tools/train_router_stage1.py`）与阶段二（`train_latent_memory.py`）都有 `--resume`
（权重 + AdamW 动量 + step/epoch 位置 + RNG）；`--fst` 的独立循环没有，长跑中断只能从头再来。

### 6. `summarize()` 没有聚合 `slot_isolation`

逐样本已落盘 `info["slot_isolation"]`（`block_diag:eager` / `fence_only:attn_impl=sdpa`），
但汇总行没有"多少比例真的用上了块对角隔离"。降级会打印，却没进结果表，审计时容易漏看。

### 7. `oracle_embed` 与 `oracle_latent` 的预算只是"按块数估算对齐"

`oracle_embed` 按 `ceil(unit_tokens / max_unit_tokens) × m` 估注入 token 数，
`oracle_latent` 的 `mem_tokens` 是实际 `n_soft`。块边界上会差几个 token，
做"注入机制 vs 压缩损失"归因时必须核对结果表这两列的实际值，不能默认相等。

### 8. `--hard-neg-k` 的负例随 router 变强而漂移（课程效应未量化）

负例是"当前 router 认为最像金标的非金标槽"。router 变强 → 负例变难；也可能全变成同类干扰。
若 EM 曲线后期掉点，先看 `--hard-neg-k` 与 `--gold-readout-ratio` 的交互，再怀疑压缩器。
新增的 `--readout-mode router` 会把"router 自选时缺了几条金标"记进 `readout_stats`，
可用于区分"负例太难"与"读出分布本身有缺证据"。

### 9. `HybridLatentRouter` 的 MaxSim 底盘依赖 `LATENT_KEY_TOKENS=64` 的截断

`slot_token_features` 把每槽 token 特征截到 64 个（超出丢弃，非池化）。对
`max_unit_tokens=32` 够用；换成更长单元（>64 token）后字面匹配覆盖会变差，
且目前只在 `n_units` 汇总里体现、没有逐样本告警。

### 10. 训练期验证与产品评测的等价性靠全局 `_ARGS_REF`

`evaluate()` 从 `_ARGS_REF[0]` 取 `topk_per_hop/slot_isolation`，等价性依赖
"`train()` 先设好全局再调 `evaluate()`"。若从其它脚本直接调 `evaluate()`，这两个参数会退回
默认值（1/0），日志数字与 `eval_latent_table.py` 不可比。

### 11. FST 没有两跳链式读出

`generate_latent_answer_fst` 在 `hops>1` 时**显式报错**（静态身份键没有"客体值接力"那一路，
按单跳跑会得到"看起来正常的错答案"）。因此 FST 在 CR 类多跳题上是**单槽对两跳**的
不对等比较，那一列只能与"读对槽位率"一起解读。

### 12. 历史容量与保真度的权衡未量化

`LATENT_HIST_MAX`（默认 4）是拍的常数：超容量 FIFO 淘汰最旧版本，但"保留几条够用"
"压缩后的历史丢多少信息""历史变长后定位精度如何退化"一个都没测。
修法：`hist_max ∈ {0,1,2,4,8}` × 时间跨度 ∈ {3,6,12} 网格，报 history 探针准确率曲线。

---

## P2 — 数据性质与环境（不是 bug，但必须写进论文/结果表）

### 13. HotpotQA 的答案几乎总在上下文里

含黄金段落时 95.0%、加 8 段干扰后 96.3% 的样本答案能在上下文某句里找到。因此 exact-match
混入了"字符串检索能力"。**处理**：结果表必须同时给 `none`（不给记忆）这一行 —— 实测 11.2%，
"有记忆的准确率"要与 11.2% 比，而不是与 0 比；并同时给"读对槽位率"。

### 14. 干扰事实可能包含与答案相同的值

这类样本"读错槽也能答对"，会**高估**准确率且无法检验内容寻址。当前数据生成器
（`tools/prep_hotpotqa.py` 的 `select_facts_stream`）**不按答案挑候选**，但也不排除同值干扰。
修法（二选一）：生成时排除同值干扰（干净，旧 ckpt 作废）；或评估时额外报一列
"无歧义子集"准确率（不破坏已有结果）。

### 15. 槽位容量与 benchmark 规模不匹配（设计级限制）

一条事实一个槽、每槽一次编码前向：HotpotQA（~35 句/题）每题 4~5s 可接受；
MAB-CR 6k context 只能按 `--cap` 评测；LongMemEval-S（~115k token）当前设计不适用。
论文里必须写成局限，不能含糊。

### 16. 服务器环境（当前机器）

```bash
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=/data/zhangjiayang/miniconda3/envs/flux/bin/python
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
```

- GPU 0/1 与别人的作业共享（曾见 34–39 GB 的 PRLM-C 训练与 YOLO 作业）：启动前用
  `nvidia-smi` 挑空卡，每步日志已打 `peakmem`，用于区分"自己涨"与"被挤"。
- checkpoint 不入库（`ckpt/` 已 gitignore）：阶段一 100,991,977 B、v5 224,505,677 B
  （含 AdamW 状态），只存在于服务器。
- `deltamem.core` 顶层硬依赖 `transformers.models.smollm3`（4.51 无此子包），
  必须经 `deltamem_compat` 注入占位后 import。

---

## 登记规范

新增一条前先确认：能复现（给命令/最小片段）、说清影响（污染哪些结论）、给出修法（哪怕只是方向）、
标注优先级（`P0` 影响结论正确性 / `P1` 让某路径静默失效 / `P2` 数据性质与环境）。
