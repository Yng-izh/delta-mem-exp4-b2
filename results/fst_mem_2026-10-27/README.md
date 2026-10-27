# results/fst_mem_2026-10-27 — FST-Mem 阶段 B：机制落地与通路验证（**没有任何效果数字**）

**一句话**：本轮把研究定位从"给潜层记忆加个路由器"改成"**压缩之后如何做局部更新与时间一致性**"，
并把三个机制全部落成代码 + 88 项单测；`--fst` 训练入口在 CPU 极小随机模型上跑通
训练→评估→存盘→重载→评估全链路。**真实训练一步没跑**，所以本目录里所有 `acc/group/ver` 数字
都是**随机权重下的噪声**，唯一用途是证明"管线没有断"。
第二轮核查（用户要求"核查代码"）又揪出 **6 个静默 bug + 3 条恒真断言 + 3 个数据缺陷**，
详见"二·五"节与 `audit_mutations.log`。

## 一、为什么改研究定位（不是"加模块"）

| 原方案 | 问题 |
|---|---|
| `Compressor + Router + Bridge + 软 token` | 四个组件都有成熟研究基础，组合本身不构成贡献；且只解决**静态多事实读取** |
| 完全没用到项目已有的能力 | 已有**事实覆写**（`StatePool`）与**时间记忆**（`V9_TEMPORAL_*`）能力，潜层路径一个都没接 |

新问题（论文主线）：**事实被压缩成潜向量后，在不断新增、局部更新、历史回溯的过程中，
如何保持实体—属性绑定、避免无关事实干扰、并按时间取回历史值。**

`LatentRouter` 因此降级为**静态多事实内容寻址基线**。

## 二、三个机制 ↔ 代码 ↔ 断言

| 机制 | 代码 | 单测直接断言的东西 |
|---|---|---|
| 一 · 实体—属性级稳定寻址 | `LatentFactKeyEncoder`（`k_{e,r}=norm(out([h_e,h_r,h_e⊙h_r]))`） | 键**与值无关**（值变了键不变）；不同属性→不同键；实体/属性**不可交换**；梯度到四头 |
| 二 · 状态—轨迹解耦槽位 | `LatentStateTrajectoryBank`（`cur + hist`，超容量 FIFO 淘汰并**真释放**） | 写新值→旧值下沉；**改 A 键时 B 键载荷 `torch.equal` 逐位相等**（更新局域性）；`hist_max=0` 退化为硬覆写；`read(key,time)` 的四级匹配优先级 |
| 三 · 时序条件两级读出 | `FstRouter`（先选键，再在组内选版本） | 事实级打分**与组数/顺序无关**；`logits_ver` 长度 = 组内版本数（非 G×V）；掩码 `-1e4` 不会被选中；**STE 前向 `torch.equal` 等于硬选** |

**更新局域性是数据结构保证的，不是训练出来的** —— 这是本轮最重要的设计取舍：
不能把"改了雇主不该影响居住地"寄托在梯度上。

## 二·五、核查（用户要求"核查代码"后的第二轮自查）

三路并行审查（`latent_memory.py` FST 层 / `train_latent_memory.py` FST 路径 / 数据与测试）
共 25 条 findings，去重后**确认 6 个真 bug + 3 条恒真断言 + 3 个数据缺陷**，全部已修：

| # | 问题 | 危害 | 性质 |
|---|---|---|---|
| 1 | 训练侧用**原始大小写**实体名构键特征，读出侧从（已小写的）事实键反推 | 训练/评估看到不同的键特征，评估等于喂分布外输入，指标上看不出来 | 静默 |
| 2 | STE 软期望分支**没 detach 载荷** | 梯度按 `wg⊗wv` 铺到全部 G×V 条载荷上，压缩器被迫同时满足所有事实（与"软加权前向"同一个病挪到梯度） | 静默 |
| 3 | 版本级监督 / 指标用 `ver_picked`（被选中组内下标）与目标组下标比较 | 组选错时下标碰巧相等 → **虚记命中** | 指标虚高 |
| 4 | `pair_acc` 用 `all(v)`，单成员对恒真；`unrelated` 探针两变体 gold 相同也算配对 | 配对准确率虚高 | 指标虚高 |
| 5 | `viable=False` 把"版本被容量淘汰"与"数据缺时间戳"混成一个数 | hist_max 小的时候看不出是机制问题还是数据问题 | 可归因性 |
| 6 | 并列时间戳时 `tgt_ver` 取**第一个**匹配（最旧），而 `bank.read()` 取最新的 | 监督指向 `read()` 永远不会选的槽位 | 静默 |

**3 条恒真断言**（测了等于没测）：`fact_key(x,y)==fact_key(x,y)`（同函数自比）、
`ver_z.grad is None`（本来就不需要梯度）、"秘密值不在 tokenizer 文本里"（秘密值从未进过任何文本）。
已改成有分辨力的断言，并新增 `tools/audit_mutations.py`：对 6 个"应该被抓的错误"做最小变异，
**要求测试必须失败**。变异存活 = 断言没约束力。当前 6/6 全部被抓住（`audit_mutations.log`）。

**3 个数据缺陷**：① 21.9% 的行里答案字符串出现在干扰事实中（"值在场即可答"，与 exp2 的
22.5% 缺陷同类）→ 干扰值池排除本行全部答案值；② 干扰事实的实体可以来自对侧切分
（"train/val 实体不重叠"只对目标键成立）→ 改成同侧实体池；③ 生成器现在**自检后落盘**：
答案值无泄漏、干扰键无冲突、反事实配对数与 gold 差异、`unit_meta` 与 `units` 逐条一致，任一不成立直接 SystemExit。

指标侧同时补了**常数基线**：`group_const_acc`（永远选第一组）与 `ver_newest_acc`（永远选最新版）。
这在本轮 smoke 里立刻起了作用——随机权重下 `group=0.333` 看着"还行"，而常数基线是 **0.667**，
说明当时还不如盲选。没有基线的话这个数会被误读成"有点效果"。

## 三、数据：`synth/fst_update_{train,val}.jsonl`（192 / 72）

不是"4 条无关事实"的放大版，而是**同实体多属性 + 多次覆写 + 时间条件**：

```
Ludarn works at Fenwick.   (2021)
Ludarn lives in Ostrel.    (2023)
Ludarn works at Velcor.    (2025)  ← 当前值
Islik works at Fenwick.    → 干扰项（键不与目标键冲突）
```

每实体 2 个反事实变体 × 3 个探针 = 6 行；训练集三探针各 **64** 条；
`unit_meta` 与 `units` 逐条对齐（缺一条直接报错，不猜）。

## 四、验证（全部是**通路**验证）

见本目录日志：

| 日志 | 内容 | 结论 |
|---|---|---|
| `test_fst_mem.log` | 机制单测 | **ALL PASS (88 checks)**（原 67 + 核查新增 21） |
| `audit_mutations.log` | 变异审计（故意改坏，要求测试失败） | **6/6 变异全部被抓住** |
| `test_synth_data.log` | 数据管线单测 | **32 项全过** |
| `test_latent_readout.log` | 旧单测（`_emit_latent` 重构后） | **ALL PASS (89 checks)**，不回归 |
| `smoke_fst.log` | 完整 FST 全链路（6 步） | 训练/评估/存盘/重载/评估全通；`grad-check` 五头全 true；`n_soft=4`(=m)；`pair_n=4` 真配对齐全 |
| `smoke_fst_hist0.log` | `--hist-max 0` 硬覆写对照 | `ev=8`（8 条历史题目标版本被淘汰）、`ver_multi=n/a(0)`、`skip_evicted` 递增 |
| `smoke_fst_ignore_time.log` | 去时序条件消融 | 正常跑通，可与完整版分栏对比 |
| `gen_data.log` | 数据生成 | 192/72，探针分布 `{current:64, history:64, unrelated:64}` |

**这些数字不能当结论**：smoke 用的是 `--smoke-model`（hidden=64 的随机 2 层模型），
`acc=0.000` 是必然的（随机权重 + StubTok 解码），`group/ver` 也在随机水平附近抖动 ——
而且注意 `group=0.333` **低于**常数基线 `const=0.667`，即当时连"永远选第一组"都不如。

## 五、本轮修掉的 3 个真 bug（都会静默失效）

1. `LatentStateTrajectoryBank.versions()` 与父类 `self.versions` 字典同名 →
   `TypeError: 'dict' object is not callable`。改名 `list_versions/count_versions`。
2. 版本级监督打在 `logits_ver`（**被选中组**那行）→ 初期选错组时监督指向另一组的版本布局，
   `loss_ver` 从 0 跳到 **9990**。改为 `logits_ver_all[tgt_group]`。
3. STE 软期望 fp32 / 硬选 bf16 相加被提升 → 先 `.to(dtype=z_hard.dtype)` 再做梯度直通。

另有 1 个"测试自伤"：把秘密值写进问题文本，然后断言"值没进 tokenizer" —— 测试自己造的失败。

## 六、下一步（在 v100-2 上，按此顺序）

```bash
# 环境（服务器 Python 环境被其他用户破坏，必须带这些前缀）
export CUDA_VISIBLE_DEVICES=7 HF_HOME=/share/hf_home HF_HUB_OFFLINE=1 \
       DELTA_MEM_SCAN_IMPL=torch \
       PYTHONPATH=/share/exp4group/shim19:/share/exp4group:/share/delta-Mem
PY=/share/conda/envs/latent2/bin/python

# ① 阶段 A 基线（静态多事实内容寻址）
$PY train_latent_memory.py --router --data synth/latent_exp2_train.jsonl \
    --val-data synth/latent_exp2_val.jsonl --epochs 3 --accum 16 \
    --out runs/fst/a_router.pt

# ② 硬覆写对照（只有稳定键，无历史、无时间选择）
$PY train_latent_memory.py --fst --hist-max 0 --data synth/fst_update_train.jsonl \
    --val-data synth/fst_update_val.jsonl --epochs 3 --accum 16 --out runs/fst/b_hard.pt

# ③ FST 完整（历史 + 时序两级读出）
$PY train_latent_memory.py --fst --hist-max 4 --data synth/fst_update_train.jsonl \
    --val-data synth/fst_update_val.jsonl --epochs 3 --accum 16 --out runs/fst/c_fst.pt

# ④ 时序条件消融
$PY train_latent_memory.py --fst --hist-max 4 --ignore-query-time \
    --data synth/fst_update_train.jsonl --val-data synth/fst_update_val.jsonl \
    --epochs 3 --accum 16 --out runs/fst/d_no_time.pt
```

每档至少 3 个 seed，必报 `acc / group_acc / ver_multi_acc / 三探针 / pair_acc / order_stable`。
**在跑完 ②③ 之前，"FST 有效"这句话一个字都不能写。**

## 六·五、本轮修掉的 6 个真 bug（详见"二·五"核查一节）

| # | 一行说明 | 修法 |
|---|---|---|
| 1 | 训练/读出实体文本不同源（大小写） | 库里存**原始文本**，读出从 `bank.group_text()` 取；新增 M5/M6/M6b 断言 + 变异捕获 |
| 2 | STE 未 detach 载荷 | `z.detach()`，并把"选中组 wv 套全组"改成**每组各算一次** softmax |
| 3 | 版本命中不绑组别 | `fst_scores()` 里 `g_ok and ver_picked==tgt_ver`；抽出纯函数便于单测（N1） |
| 4 | 配对恒真 / 同值配对 | 要求 `len>=2` 且 gold 有差异；评估集按"整组问题"截断，保证配对完整 |
| 5 | 淘汰 vs 缺标签混为一谈 | `build_fst_bank` 返回 `evicted/label_missing/label_mismatch`；训练分别处理，`label_mismatch` 直接报错 |
| 6 | 并列时间戳监督取最旧 | 改成取最后一个匹配（= 最新），与 `bank.read()` 的并列规则同源（N5） |

## 七、目录内容

```
README.md                      ← 本文件
test_fst_mem.log               88 项机制单测
test_latent_readout.log        89 项旧单测（回归）
test_synth_data.log            32 项数据管线单测
audit_mutations.log            变异审计（6 个变异必须被抓住）
smoke_fst.log                  全链路（含 save/reload）
smoke_fst_hist0.log            硬覆写对照
smoke_fst_ignore_time.log      去时序条件消融
gen_data.log                   数据生成
```
