# DATASETS.md — 实验数据盘点

**最后更新**：2026-10-27（潜层 Router 重构轮）

**一句话**：潜层读出实验实际只用了 **3 套自造玩具数据（共 280 条）**；服务器上另有 **5 套更大的数据从未进过潜层实验**（含 2998 条多跳）；外部 benchmark **一套都没有**。

---

## 1. 潜层读出实验正在用的（本地 `synth/`，已入库）

| 数据集 | 条数 | 槽位/条 | 反事实对 | 关系 | 实体 | 用途 |
|---|---|---|---|---|---|---|
| `latent_min16.jsonl` | 16 | **1** | 8 | 8 种 | 8 | 实验 1 单槽健全性（曾达 15/16） |
| `latent_exp2_train.jsonl` | 200 | **4**（1 目标 + 3 干扰） | 100 | 6 种 | 100 | 实验 2 多槽训练 |
| `latent_exp2_val.jsonl` | 64 | **4** | 32 | 6 种 | 32（与训练**不重叠**） | 实验 2 验证 |

三套全部 `hops=1`（单跳）。

**字段**

```jsonc
{
  "kind": "single", "hops": 1,
  "memory": ["Harish is married to Elian.", ...],   // 事实列表
  "units":  ["Harish is married to Elian.", ...],   // 每条事实一个记忆单元 = 一个槽位
  "question": "What did Branish found?",
  "answer": "Ondrey",
  "evidence": ["Branish founded Ondrey."],          // 目标事实原文（唯一可靠的槽位定位依据）
  "subject": "Branish", "relation": "founded"
}
```

注意 `load_synth_jsonl` 的产出与原始文件不同：`memory` 会被 join 成**单个字符串**（teacher 明文输入 / 重建目标用），槽位切分以 `units` 为准。**取"第一条事实"必须用 `units[0]`，用 `memory[0]` 会取到第一个字符**（2026-10-27 修掉的诊断 bug）。

**规模极小，必须拿基线对照**：答案值池每个关系只有 6 个，全库仅 **18 个不同答案值**、
`(relation, answer)` 只有 **36 种组合**，平均每个值出现 11 次。
→ 按"关系"猜值就能达到 **31.2%**，任何低于这个数的准确率都是噪声。评估必须带这一列基线。

---

## 2. 服务器上有、潜层实验一次没用过的（`/share/exp4group/synth/`）

| 文件 | 条数 | 说明 |
|---|---|---|
| `train_15k.jsonl` | **15000** | 12002 单跳 + **2998 多跳**；15 种关系；8076 个组件；memory 4 条 |
| `val_2k.jsonl` | 2000 | 同构验证集 |
| `facts_2000.jsonl` | **32000** 条事实 | cluster 0–1800 的事实池（文件名叫 2000 但装 32000 条，`text` + `cluster` 两字段） |
| `overfit200.jsonl` | 200 | 过拟合自查 |
| `slice_train.jsonl` / `slice_val.jsonl` | 30 / 10 | 调试切片 |

`train_15k` 是唯一有真实多跳的数据，格式完整：

```jsonc
{"kind": "multi", "hops": 2,
 "relations": ["was born in", "is the capital of"],
 "memory": ["Kearkniol Purkpham was born in Sturkbres Vrastriom.", ...],  // 5 条
 "bridge_entity": "Sturkbres Vrastriom",     // 中间实体
 "question": "Which entity has the birthplace of Kearkniol Purkpham as its capital?",
 "answer": "Vrust Ziam"}
```

**这条链正是 `LatentRouter.chain()` 需要验证的数据，目前完全没跑过。**

---

## 3. 外部 benchmark：0 套

`/share/exp4group` 里**没有** MemoryAgentBench / LongMemEval / EventQA 的原始数据。
`results/` 下只有历史跑分的 jsonl 输出（`ar_v7_eventqa_*`、`ar_v8_longmemeval_*`、`groupA_*`）。

→ 方案里的「实验 2：MemoryAgentBench CR/SF + AR」**当前无数据可跑**，需先取数据。

---

## 4. 三个硬缺口

1. **没有单槽拆分数据** → Phase 1（单槽预训练）无法按方案开跑。现在只能拿 4 槽数据或 16 条 min16 凑。
   需要 `tools/explode_units.py`：把 4 槽样本拆成 4 条单槽样本。
2. **多跳潜层数据为 0** → `train_15k` 有 2998 条，但从未被 `load_synth_jsonl` + 潜层管线跑过。
3. **数据太玩具** → 假名实体（音节拼接）、18 个答案值、6 种关系，与真正要评估的 MemoryAgentBench 分布差得远。

---

## 5. 两个已知数据缺陷（未擅自改数据集）

### 缺陷 A：答案值在干扰事实里重复出现 → 高估准确率

| 数据集 | 歧义样本 | 占比 |
|---|---|---|
| `latent_exp2_train` | 45/200 | **22.5%** |
| `latent_exp2_val` | 11/64 | **17.2%** |

模型若读错槽、恰好读到含相同干扰值的那条事实，也能吐出正确答案 → 准确率被高估，
且这类样本**无法检验内容寻址是否 work**。

两条修法（**待定，尚未执行**）：
- **改生成器**：`tools/gen_latent_exp2.py` 生成干扰时排除与答案同值的事实，重新出数据（干净，但旧 ckpt 全部作废）
- **改评估**：保留数据，额外报一列「无歧义子集」的准确率（不破坏已有结果）

### 缺陷 B：已修——寻址监督标签错误 13.5%

`target_slot_index` 原先只用答案串匹配，受缺陷 A 影响把 27/200（13.5%）的寻址标签标到错误槽位。
已改为优先用 `evidence` 精确匹配。**标签错误 200→0、64→0。** 详见
[`results/latent_router_2026-10-27/README.md`](results/latent_router_2026-10-27/README.md) 第三节问题 7。

---

## 6. 数据生成脚本

| 脚本 | 作用 | 关键参数 |
|---|---|---|
| `tools/gen_fact_universe.py` | 事实池：音节拼假名 + 15 种关系模板 | `--clusters 1000 --jsonl` |
| `tools/gen_synth_qa.py` | 事实文本 → 单跳/多跳问答（`parse_fact_triple` 切三元组） | 见脚本头 docstring |
| `tools/gen_latent_min16.py` | 实验 1：16 条单槽反事实配对 | — |
| `tools/gen_latent_exp2.py` | 实验 2：4 槽（1 目标 + 3 干扰） | `--n-train-pairs 100 --n-val-pairs 32 --seed 7` |

重生成命令：

```bash
python tools/gen_latent_exp2.py --n-train-pairs 100 --n-val-pairs 32 --seed 7
```

---

## 7. 评估纪律（写死的硬规矩，血泪换来的）

1. **复用训练常量，绝不手抄**。system prompt 必须从源码提取：
   ```python
   re.search(r'"--system-prompt",\s*\n\s*default="([^"]+)"', open("train_latent_memory.py").read())
   ```
   手抄曾导致 17% vs 0% 的假故障（`"You are a memory agent..."` vs `"You are a helpful assistant..."`）。
2. **每次评估前必须** `set_delta_mem_write_enabled(model, False)` **且逐条** `reset_delta_mem_states(model)`。
   否则 delta_state 跨样本累积，前几条正常、后面整体崩。
3. **比对数字前先对齐**：prompt + 数据 + `layer` + `max_unit_tokens` + `m`。任何一项不同都不是同一个实验。
4. **必须同时报四列**：train / val / plaintext / oracle，外加 **pair-acc** 与 **relation-prior 基线（31.2%）**。
   单看一个 val 数字会得出反向结论。
5. **完整答案匹配（规范化后）**，不用子串匹配。子串会把 `"Ironwood Energy Nilork, a"` 判成对，系统性高估。
