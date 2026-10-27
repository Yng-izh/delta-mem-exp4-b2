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

## 1. 本地 `synth/`（已入库，只留 smoke）

| 文件 | 条数 | 用途 |
|---|---|---|
| `fst_update_train.jsonl` | 192 | FST 通路 smoke 训练（`--smoke-model` CPU 自检） |
| `fst_update_val.jsonl` | 72 | 同上验证；三探针各 64 条（当前/历史/无关），实体与训练不相交 |

## 2. 服务器上未用的历史合成数据（不入库，随废案作废）

`/share/exp4group/synth/` 下的 `train_15k`、hotpot 中间产物之外的合成集属于已废管线；
`hotpot_*` 转换产物由 `tools/prep_hotpotqa.py` 随时可重建，以脚本为源，不手工维护副本。

## 3. 外部 benchmark：2 套已接入，1 套在服务器待接

| 基准 | 服务器位置 | 状态 |
|---|---|---|
| MemoryAgentBench CR（800 题） | `/share/latent/MemoryAgentBench/data/Conflict_Resolution-*.parquet` | ✅ `tools/prep_mab.py --cap 32` 已转换并跑分 |
| HotpotQA distractor（train 90447 / val 7405） | `/share/exp4group/data/hotpot/distractor/*.parquet` | ✅ `tools/prep_hotpotqa.py` 已转换并跑分 |
| LongMemEval cleaned（500 题） | `/share/latent2/vcrm/data/longmemeval_s_cleaned.json` | ⏸ 未接：会话 ~115k token vs 64-token 槽位，容量不匹配 |

→ 已接入的两套就是结果表 `results/RESULTS.md` 的全部数据来源。

---

## 4. 玩具数据只留 smoke

旧合成数据（`latent_exp2`、`latent_min16`、15k 事实宇宙）及其生成器已随废案移除。
保留两个 smoke 用途：

| 文件 | 用途 |
|---|---|
| `tools/gen_fst_update.py` → `synth/fst_update_{train,val}.jsonl` | FST 训练/评估通路的 CPU smoke（`--smoke-model`） |
| `tools/gen_synth_qa.py` | `load_synth_jsonl` 的外部语料通路自检 |

已知且保留的两个数据事实（写进报告口径）：
- 寻址标签优先 evidence 匹配：干扰事实常与答案同值（旧合成数据 22.5%），纯答案串匹配会把 13.5% 标错槽；
- HotpotQA 答案在上下文 96.3% 是数据集性质，`meta.leak` 只统计干扰段落，必须与 none 地板、读对槽位列同报。
