# DATASETS.md — 实验数据盘点

**最后更新**：2026-10-27（真实数据接入轮）

**一句话**：玩具数据退居 smoke 测试；**正式训练与评测改用公开数据**——HotpotQA（训练 + 读出评测）、
MemoryAgentBench Conflict Resolution（冲突覆盖）、LongMemEval cleaned（知识更新/时间推理/多会话）。

---

## 0. 公开数据（v9.7 起，正式实验只用这些）

| 数据 | 服务器位置 | 规模 | 用途 | 转换器 |
|---|---|---|---|---|
| HotpotQA distractor `train` | `/share/exp4group/data/hotpot/distractor/train-0000{0,1}-of-00002.parquet` | 90447 | **训练** Compressor+Router（真实文本/问题/答案/支持事实，2 跳） | `tools/prep_hotpotqa.py` |
| HotpotQA distractor `validation` | 同目录 `validation-00000-of-00001.parquet` | 7405 | **评测**（严格与训练隔离） | 同上 |
| MAB Conflict Resolution | `/share/latent/MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet` | 8 context × 100 问 | **评测**：事实冲突覆盖（"更新到最新值"） | `tools/prep_mab.py` |
| MAB Accurate Retrieval | 同目录 `Accurate_Retrieval-*.parquet` | 22 context → 2000 问 | 基础检索对照（选择性使用） | 待接 |
| LongMemEval cleaned | `/share/latent2/vcrm/data/longmemeval_s_cleaned.json`（264MB, 500 问） | multi-session 133 / temporal 133 / knowledge-update 78 / single-session 156 | **评测**：更新与时间推理 | 待接 |

获取方式：服务器**没有** huggingface.co 网络，但 **`hf-mirror.com` 可达**（`pypi.org` 也可达，
`github.com` 不可达）。下载命令：
`HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=0 python -c "from huggingface_hub import hf_hub_download; ..."`

### 0.1 HotpotQA 转换后的三条硬事实（必须带进论文）

1. **答案在上下文里是常态**：含黄金段落时 95.0% 的样本答案能在同段其它句子找到；
   加入干扰段落（8 段）后 96.3%。这就是数据集性质，**统计并报告**（`meta.leak` 只统计干扰段落），
   不许当成缺陷去"修"。正因如此，结果表必须同时给 `none`（不给记忆）这一行。
2. **0.017% 的 `supporting_facts` 越界/标题缺失**（实测 1/6000）：转换器**跳过并计数**，
   不 clamp（clamp 会把 evidence 指到别的句子，寻址标签直接错）；比例 >1% 直接拒绝产出。
3. **96% 是 2 跳（bridge/comparison 全为 multi-hop）**：`hops=2` 的链式选槽必须开，
   否则两跳题永远不可能对。

### 0.2 MAB-CR 转换结果（实测，`tools/prep_mab.py --cap 32`）

| 项 | 值 |
|---|---|
| 样本数 | 800（8 个 context 组 × 100 问，0 条丢弃） |
| 每问可见事实 | 固定 32（`--cap`，写入 `meta.cap`；原始 context 有几百条） |
| **含冲突版本样本** | **76.8%**（同一 `(entity, relation)` 有 ≥2 个不同取值 → "用最新值覆盖旧值"真正被考到） |
| 关系解析失败率 | 0.2%（模板表解析，不是 LLM 抽取） |
| 训练 / 验证切分 | 500 / 300，按 context 源切（`{sh,mh}_{6k,32k,64k}` 训练，`{sh,mh}_262k` + `mh_6k` 验证） |
| FST 字段 | `unit_meta[i] = {fact, entity, relation, time=事实编号, distractor}`；`entity/relation`（金标那条）与 `answer_fact_time` 供两级寻址监督 |
| 时间轴 | **事实编号**（MAB 没有时间戳，编号越大越新） |

### 0.3 MAB-CR 的容量限制（写进报告，不许隐瞒）

CR 的一条 6k context 有几百条事实，而槽位记忆**一条事实一个槽、每槽一次编码前向**
→ 每问几十秒。因此按 `--cap` 限定每问可见事实数（默认 32），选法是
"金标 + **同键全部冲突版本** + BM25 相关事实"。同键版本必须整组进入，否则
"用最新值覆盖旧值"这条被测能力会被选择策略本身消掉。`meta.cap` / `meta.n_facts_total`
会写进每条样本。

---

## 1. 潜层读出实验正在用的（本地 `synth/`，已入库）

| 数据集 | 条数 | 槽位/条 | 反事实对 | 关系 | 实体 | 用途 |
|---|---|---|---|---|---|---|
| `latent_min16.jsonl` | 16 | **1** | 8 | 8 种 | 8 | 实验 1 单槽健全性（曾达 15/16） |
| `latent_exp2_train.jsonl` | 200 | **4**（1 目标 + 3 干扰） | 100 | 6 种 | 100 | 实验 2 多槽训练 |
| `latent_exp2_val.jsonl` | 64 | **4** | 32 | 6 种 | 32（与训练**不重叠**） | 实验 2 验证 |

三套全部 `hops=1`（单跳）。

**第四套（v9.6 新增，动态事实）**：

| 数据集 | 条数 | 事实键/条 | 版本 | 反事实变体 | 实体 | 用途 |
|---|---|---|---|---|---|---|
| `fst_update_train.jsonl` | **192** | 5（2 个目标属性键 + 3 个干扰键） | 雇主键 **2 个时间版本** | 2（雇主值互换） | 32 | FST-Mem 阶段 B 训练 |
| `fst_update_val.jsonl` | **72** | 同上 | 同上 | 2 | 12（与训练**不重叠**） | FST-Mem 验证 |

它不是"4 条互不相关的事实"的放大版，而是**新问题**：同一实体的多个属性各自被多次覆写。
行数与探针一一对应：`{current: 64, history: 64, unrelated: 64}`（训练集）。
`unrelated` 探针就是**更新局域性**的测量：雇主怎么改，居住地的答案都不该变。

**这道数据集的三个硬约束（生成时校验，不满足直接拒绝落盘）**

1. **答案值不得出现在非目标单元里**。旧版干扰事实与答案取自同一个 `VALUES` 池，实测
   train **42/192（21.9%）**、val **10/72（13.9%）** 的行里答案字符串直接躺在干扰事实中 ——
   那等于"值在场即可答"，不需要读对槽位/版本（exp2 旧集同类缺陷 22.5%）。现已在取值时排除本行全部答案值。
2. **干扰事实的实体只来自同侧切分**。否则"train/val 实体不重叠"只对**目标键**成立，
   干扰文本里仍会出现对侧实体名，"未见实体泛化"这个说法站不住。
3. **反事实配对只对 `current` / `history` 成立**。`unrelated` 探针在两个变体里 gold 相同，
   评估侧必须按"同问题且 gold 不同"筛配对，否则 pair_acc 被稀释（评估脚本已按此实现，
   且按**整组问题**截断评估集，避免配对被切两半）。

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

`fst_update_*` 的字段在上面的基础上多一层结构化元信息（**逐条与 `units` 对齐**）：

```jsonc
{
  "kind": "update", "hops": 1,
  "entity": "Ludarn", "relation": "works at", "fact_key": "ludarn||works at",
  "unit_meta": [                                    // 与 units 逐条对齐，缺一条就报错
    {"entity": "Ludarn", "relation": "works at", "time": "2021",
     "fact": "Ludarn works at Fenwick.", "distractor": false},
    {"entity": "Ludarn", "relation": "lives in", "time": "2023", ...},
    {"entity": "Ludarn", "relation": "works at", "time": "2025", ...},   // 当前雇主
    {"entity": "Islik",  "relation": "works at", "time": "", "distractor": true}
  ],
  "timeline":  [{"time": "2021", "relation": "works at", "value": "Fenwick", ...}, ...],
  "question": "Where does Ludarn work?",   // 或 "In 2021, where did Ludarn work?"
  "answer": "Velcor",
  "as_of": "",                              // 历史题填 "2021"（= 查询的时间条件）
  "answer_fact_time": "2025",               // 正确版本的时间戳（版本级监督用）
  "probe": "current | history | unrelated",
  "evidence": ["Ludarn works at Velcor."]
}
```

**为什么必须有 `unit_meta`**：靠正则从句子里猜 (实体,属性,时间) 三元组，猜错一个属性
在本方法里等价于"更新打到了别的事实键"——静默、且指标上只表现为"莫名其妙变差"。
干扰项的事实键**保证不与目标键冲突**：同一个键写成两条同时刻版本会让"该读哪个版本"
本身有歧义，这种样本无法归因。

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
| `tools/gen_fst_update.py` | 实验 3（v9.6）：同实体多属性 + 多次覆写 + 时间条件；干扰项键不与目标键冲突 | `--n-train-entities 32 --n-val-entities 12 --years 2021,2023,2025 --seed 0` |

重生成命令：

```bash
python tools/gen_latent_exp2.py --n-train-pairs 100 --n-val-pairs 32 --seed 7
python tools/gen_fst_update.py            # -> synth/fst_update_{train,val}.jsonl
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
