# CHANGELOG

## 架构修正轮（2026-10-28）——按代码评审修掉 9 处"不报错但让结论作废"的实现缺陷

**一句话**：2026-10-27 的四方法结果表（HotpotQA dis4 100q：none 12.0 / text-top4 36.0 /
oracle 58.0 / bridge 17.5 / router 17.5；MAB-CR cap32：none 4.0 / text-top4 17.0 /
oracle 32.0 / router 2.5 / fst 2.5）**在实现层面站不住**：潜层两法都低于文本 RAG，
根因不是"潜层表征不够强"，而是训练/推理两条路各自缺半边。本轮只修实现、**未重跑**，
因此上面那些数字仍是旧口径，需在下一轮重训后整体替换。

| # | 缺陷（旧实现） | 为什么会让结论作废 | 修法 |
|---|---|---|---|
| 1 | 多跳**第二跳无监督**：训练只用第 1 跳的 CE，`k_out` 直接被丢掉；推理却用 `k_out` 接力 | 推理走一条训练里从没被监督过的网络 —— 两跳题注定接近随机，而指标只表现为"acc 低"，看不出根因 | `LatentRouter.forward_hops` 逐跳返回 logits，训练按 `evidence` 顺序**逐跳 CE**；训练与推理共用 `compose_routed_soft`（槽间 `"\n"`，与推理逐位一致） |
| 2 | 重建辅助任务**目标错位**：随机抽单元当重建目标，输入却只有"路由选中那一个槽"的 soft | 上下文 A/B/C、路由选 A 时，任务是"据 A 恢复 B 和 C"，信息上不可能完成；实测 recon token-acc 22.8% vs 平凡基线 19.8% | 新增 `recon_loss_units`：每个被抽中单元**用自己的载荷重建自己的原文**，输入与目标严格同位 |
| 3 | `prep_mab.select_facts` 的 BM25 **升序**（漏 `reverse=True`） | 按 cap 截断时优先塞入**最不相关**的事实，等于把选法用意做反；MAB 的"抗干扰"难度被人为放大 | 改降序；并新增 `chain_indices`（值→实体图上的 2 跳确定性扩张）+ 问题实体锚定，优先级：金标与同键版本 > 锚点 > 推理链 > BM25 |
| 4 | FST 读出**硬编码单槽**（`hops>1` 直接抛错） | 与 Router 的 multi-hop 比较不是同一件事，"FST 更差"有一部分只是"只读一条" | `FstRouter.forward(k_groups=)` 贪心取 k 个不同事实键；`generate_latent_answer_fst` 用 `bank.value_at()` 做**确定性**值→实体接力（不引入未训练网络），多读槽位经 `compose_routed_soft` 拼接 |
| 5 | `LatentRouter` 的 STE 软期望**未 detach** 载荷 | 梯度顺着"软期望"回流到载荷本身，等于让压缩器去迎合路由器当前的混合权重（耦合，不是监督） | `z_soft = einsum(w, z.detach())` |
| 6 | HotpotQA 单元**没有实体锚点**（纯句子） | "He was married to Peggy Seeger." 这类句子没有实体绑定，第二跳即使读对也无法使用 | `prep_hotpotqa` 产出 `unit_meta`（`anchor`=段落标题）；编码时以 `[Title] sentence` 进编码器（`unit_encode_text`），**不改** `units`/`evidence`，字符串匹配口径不变 |
| 7 | 版本级**重新学"取最新"**（`TemporalStateSelector` 去拟合归一化里已给定的规则） | 把结构规则交给分类器：多一个可错点、多一份容量、还要给它造 CE 监督 | 新增 `--ver-policy current|time|learned`：默认 `current`（取当前版本，结构规则）/`time`（按查询时间最近匹配）；只有 `learned` 才算并监督版本 CE（消融） |
| 8 | `build_fst_bank` / 编码路径**每题重复编码**整段上下文 | MAB-CR 上 FST 每题 11.0s vs text-top4 1.46s，主要成本是重复前向，而非方法本身 | `encode_item_slots(cache=)` / `build_fst_bank(cache=)` 文本级载荷复用；`eval_latent_table` 每进程一个 `unit_cache`，训练/评估的写入路径不变 |
| 9 | 评测侧两处口径错：`--method bridge` 会把 ckpt 里的 router 塞进 ctx（实跑成"有路由的单槽读出"）；多跳 `read_acc` 用"选中任意一条黄金事实"判定 | 前者让 bridge 与 router 两行**同源**，"Bridge vs Router"这一列不成立；后者把"只读了半条链"算成功，read_acc 虚高、与 acc 的落差被误读 | `use_router` 只给 `--method router`；多跳 `read_ok` 要求 `gold ⊆ picked`（**整条链**），text/oracle 列同口径 |

**接口变化（向后兼容）**

- 新增：`LatentRouter.forward_hops`、模块级 `compose_routed_soft`、
  `LatentStateTrajectoryBank.value_at/slot_values`、`FstRouter.pick_version`、
  `FstRouter.forward(k_groups=, ver_policy=, ver_times=, query_time=)`（返回 `z_reads/groups/versions`，
  `z_read` 仍是第一个槽）、`train_latent_memory.unit_encode_text / evidence_chain_indices /
  recon_loss_units`、`encode_item_slots(cache=)`、`build_fst_bank(value=/cache=)`、
  `prep_mab.chain_indices`、`prep_hotpotqa` 的 `unit_meta`、`--ver-policy` / `--read-groups`
  （训练与评测两侧同名同义）。
- 模块级默认 `ver_policy="learned"`（保持旧调用行为，便于单测与历史配置复现）；
  **CLI 默认 `current`**：方法默认已改回显式结构规则。

**未做（刻意）**

- 未运行任何训练/评测/测试（本轮只改代码并上传）。`tests/test_fst_mem.py::J8` 的旧断言
  （"多跳未实现时显式报错"）已按新行为改写；其余测试是否仍全绿**未经执行验证**，
  下次开工第一件事就是跑一遍。
- 未改历史结果表：旧数字原样保留，仅在本条与下次结果表里标注"旧口径、待重跑"。

## 仓库清理（2026-10-27 下午）——废除方案代码移除，文档对齐当前主线

- **删除**（历史见 git log ≤ e290382）：`v6/v7/v8/v9_slotpool`、`v9_cr`、
  `run_group_b/b2/b2_v6/b2_v7`、`run_ar_v7`、`run_manifest`、`ar_diag`、
  `eval_longmemeval_judge`、`diag1/diag2/smoke_b2`、对应 8 个测试文件、
  `tools/gen_fact_universe|gen_latent_exp2|gen_latent_min16|summarize_sweep|audit_mutations|demo_facts`、
  `versions/`、`experiments/`、`synth/latent_exp2_*|latent_min16`。
  原因：自造数据管线不支撑论文核心结论，主线收束为三个问题（用户决策）。
- **内联**：checkpoint 指纹逻辑（原 `run_manifest.file_fingerprint`）就地内联进
  `latent_memory.checkpoint_fingerprint`，口径不变；`test_latent_readout.py` G 组改为
  直接测内联实现（85 项全过）。
- **文档对齐**：README（142 行，历史轮次压缩为废除注记）、ARCHITECTURE（§1–§9 按当前
  三层结构/不变量/参数重写，§11 登记废除清单）、KNOWN_ISSUES（§3.6、§6 对齐现状）。
- 测试全绿：88+89+32+20+14+24 = 263 项。

## [v9.7-real-data] — 2026-10-27 — 真实数据接入：HotpotQA / MAB-CR 转换器、统一评测入口、两处污染性 bug

**一句话**：按"实验主线收束为三个问题 + 只留两个主基准 + 训练换真实数据 + 方法对比压缩到四组"
的要求落地第一步：接入公开数据、打通真实训练与评测通路，并修掉两个**会让结果整体作废**
的实现问题（Delta 写入状态不恢复、KL 项覆盖寻址损失）。

### 一、两处污染性 bug（已修，均有回归测试）

| # | Bug | 后果 | 修法 / 证据 |
|---|---|---|---|
| 1 | `delta_write_disabled` 在 `finally` 里恒 `fn(model, False)` | 第一次调用之后**进程内永久关闭 Delta 写入**：后续所有 forward 都在"没有写入"的模型上跑，多轮实验口径不一致 | 改为**逐模块快照/恢复**各自原本的 `write_enabled`（不碰 `read_context_mask` / `write_message_ids`）；`tests/test_train_objectives.py` Q1–Q6（含"每个模块各自的值"、嵌套、异常路径） |
| 2 | KL 分支 `loss = loss_qa + kl_weight * loss_kl` 直接**覆盖**了前面累加好的 `addr_weight * loss_addr` | 一旦启用 KL，寻址监督静默消失，Router 退化成无监督 | 抽出纯函数 `assemble_loss(...)` 统一累加；P3 是这条的定点回归（KL 打开后 addr 项必须仍在） |

同时新增 `reset_delta_state(model)`：**每条样本前清空 Delta 在线状态**。在线状态是挂在模块上的
跨 forward 累积量，不清会表现为"显存缓慢上涨到最后 OOM"或"后面的样本受前面影响"，两者都不报错。

### 二、真实数据接入（正式基准与训练数据严格隔离）

| 数据 | 位置（服务器） | 用途 | 转换器 |
|---|---|---|---|
| HotpotQA distractor（train 90447 / validation 7405） | `/share/exp4group/data/hotpot/distractor/`（经 `hf-mirror.com` 下载） | **训练**：真实文本+问题+答案+支持事实；**评测**：dev 7405 | `tools/prep_hotpotqa.py` |
| MemoryAgentBench Conflict Resolution（8 context × 100 问） | `/share/latent/MemoryAgentBench/data/Conflict_Resolution-*.parquet` | 评测：事实冲突覆盖（主线问题 2、3） | `tools/prep_mab.py` |
| LongMemEval cleaned（500 问，6 类） | `/share/latent2/vcrm/data/longmemeval_s_cleaned.json` | 评测：knowledge-update / temporal / multi-session | 待接入（容量见下） |

转换器的关键约束（**写错不会报错，只会让实验失去意义**）：
- `units` 必须逐条保留：一条事实一个槽位。`load_synth_jsonl` 原先只认 `memory` 列表，
  真实数据的 `memory` 往往是拼好的整串，会被压成 1 个槽位 —— 已改为**优先用显式 `units`**。
- `evidence` 必须是支持事实**原文**：丢了就只能靠答案串定位槽位，而 HotpotQA 的答案
  经常出现在干扰段落里，寻址标签会被标错。
- HotpotQA 的"答案出现在上下文里"是数据集性质（含黄金段落时 95%、纯干扰段落时 96.3%），
  **必须统计并报告**（`meta.leak` 只统计干扰段落），不能当成数据缺陷去"修"。
- MAB-CR 没有金标事实编号、没有时间戳：`derive_gold` 用"含答案值且编号最大"机械推导，
  同一事实键的**所有版本必须一起进上下文**（`select_facts` 强制，违反即报错），否则
  "最新值覆盖旧值"这条被测能力会被选择策略本身消掉；可见事实数按 `--cap` 限定并写入 meta。

### 三、统一评测入口 `eval_latent_table.py`（结果表的一行）

四组方法 + 两个参照，**同一冻结主模型、同一 prompt 模板、同一 generate 调用**，差别只是
"记忆以什么形式进入"：

| method | 记忆怎么进 | 注入 token |
|---|---|---|
| `none` | 不给记忆（纯参数知识下界） | 0 |
| `text` | system 段写明文事实（BM25 top-k） | 明文 token |
| `oracle` | 只给支持事实原文（上界） | 明文 token |
| `bridge` | 全部槽位载荷过 Bridge | K × m |
| `router` | Router 选 1 槽 → Bridge | m |
| `fst` | 两级寻址选 (事实, 版本) → Bridge | m |

指标只保留五项：答案准确率、**读对槽位率**（text 用检索命中、router 用选中槽、fst 用组命中，
可直接横向比较）、更新准确率、历史访问准确率、注入 token 数、查询耗时。
`none` 这一行是**必须**的：HotpotQA 答案大量存在于 Qwen 参数知识里，没有它就无法判断记忆是否真被读到。

### 三·五、真实数据暴露的第三个实现问题：变槽数课程在多跳样本上被关掉了

`sample_slot_subset` 为了"别把多跳链剪断"，在 `hops>1` 时直接 `keep_all=True` 把 20 多个槽
全留下。合成数据只有 4 槽时看不出差别，**真实 HotpotQA（一条 23 槽）上等于把课程完全关掉**：
Router 从第一步就要在 23 个槽里选中第 1 跳（chance 4.3%），实测 200 步后仍停在 chance
（dev 5.0%，低于"不给记忆"的 12.0%）。

修法：新增 `must_keep`（= 证据链每一跳的下标，`evidence_slot_indices` 解析），
**证据链全留、其余随机丢**。既保住链条，又让"变槽数 + 随机顺序"真正生效。
回归测试 `tests/test_train_objectives.py` R1–R7（含"课程确实产生多种槽数"和"目标与证据链始终保留"）。

### 四、连通性修复：整段 recon 在长记忆上 OOM

`--recon-weight>0` 时用**整段记忆**（HotpotQA 一条 ~35 句、上千 token）做一次带梯度的 LM
前向+反传，实测峰值 21.8GB 并逐步涨到 30GB，第 3 步 OOM。改为 `--recon-units N`（默认 2）
只重建少量随机单元：峰值从 21.8GB 钉回 9.19GB 且 6 步稳定，recon 的用途（检查压缩器符号保真）不变。


## [v9.6.1-fst-audit] — 2026-10-27 — 核查轮：6 个静默 bug + 3 条恒真断言 + 3 个数据缺陷

**一句话**：用户要求"核查代码"，三路并行审查 FST 代码后逐条在源码里复核，
确认并修掉 **6 个不会报错、只会让指标悄悄失效的 bug**、**3 条测了等于没测的断言**、
**3 个会把准确率系统性虚高的数据缺陷**，并新增"变异审计"（故意改坏代码，要求单测必须失败）。

### 一、6 个静默 bug

| # | Bug | 为什么会静默 | 修法 |
|---|---|---|---|
| 1 | 训练侧用**原始大小写**实体名算键特征（`unit_meta`），读出侧从**已规范化（小写）**的事实键反推实体文本 | 两条路径都跑得通、指标照常输出，只是评估时喂给键编码器的向量分布和训练时不一样；单测里形状全对 | 库里存 `entity/relation` **原始文本**，读出改走 `bank.group_text()`；缓存键补 `layer/device`；新增 M5/M6/M6b（含"共享缓存时读出侧不再重算"的反向验证） |
| 2 | `FstRouter` 的 STE 软期望分支**没有 detach 载荷** | 前向数值正确（`torch.equal(z_read, z_hard)` 仍成立），只有梯度被污染：软加权把梯度按 `wg⊗wv` 铺到**全部 G×V 条**载荷上，压缩器被迫同时满足所有事实（与 2.5 节实测的"软加权前向"同一个病，从数值挪到了梯度） | `z.detach()`；版本分布改成**每组各算一次** softmax（原先把选中组的 `wv` 套到所有组上，向量不再是任何一致的期望量） |
| 3 | 版本级监督与 `ver_acc` 用 `ver_picked`（**被选中组内部**的下标）跟目标组的下标比较 | 组选错时下标碰巧相等 → 记成"版本选对了"，`ver_acc` / `loss_ver` 都虚高 | 统一成 `g_ok and ver_picked == tgt_ver`；把指标抽成纯函数 `fst_scores(recs)` 以便单测（N1–N4）；`loss_ver` 改用 `logits_ver_all[tgt_group]` |
| 4 | `pair_acc` 用 `all(v)` 统计，**单成员配对恒真**；`unrelated` 探针两变体 gold 相同也算配对 | 配对准确率虚高，且"反事实"这层含义被稀释 | 要求成员数 ≥2 且 gold 有差异；评估集按**整组问题**截断（`fst_eval_subset`），避免配对被切成两半 |
| 5 | `viable=False` 把"版本被容量淘汰"与"数据缺时间戳/答案不在库里"混成一个计数 | `--hist-max` 调小时，看不出是容量机制在起作用还是数据有洞 | 拆成 `evicted / label_missing / label_mismatch`；训练只跳过 `evicted`，`label_mismatch` 直接报错；日志加 `skip_evicted/skip_ver`，指标加 `evicted_n/label_missing_n` |
| 6 | 组内有**并列时间戳**时，`build_fst_bank` 取**第一个**匹配（最旧），而 `bank.read()` 取**最新** | 监督指向 `read()` 永远不会选的槽位；只有并列时间戳样本才触发，正常跑完全看不出来 | 改成取最后一个匹配；单测 N5 断言 `tgt_ver` 与 `bank.read()` 的选择一致 |

### 二、3 条恒真断言（测了等于没测）

| 原断言 | 为什么恒真 | 换成 |
|---|---|---|
| F2 `fact_key("Zhang","employer") == fact_key("Zhang","employer")` | 同函数同参数自比 | 同键写两个不同值 → 仍只有一组两版本（`group_keys/count_versions`） |
| I11 `ver_z.grad is None` | 载荷来自 `torch.randn`（非叶子、不需要梯度） | 叶子 + 切断"打分依赖载荷"这一路后，**STE 的载荷梯度必须与 `hard=True` 逐位相同** |
| J4 "秘密值不在 tokenizer 文本里" | 秘密值从未进入任何文本 | "没有事实原文进 tokenizer"（查 `works at`/`lives in`）+ 反向确认问题/system 确实进过，避免空转 |

### 三、3 个数据缺陷（准确率系统性虚高）

| # | 缺陷 | 实测占比 | 修法 |
|---|---|---|---|
| 1 | 干扰事实的值与本行答案取自同一个 `VALUES` 池，"答案字符串出现在干扰事实里"= 值在场即可答 | train 42/192（21.9%）、val 10/72（13.9%）（exp2 旧集同类缺陷 22.5%） | 干扰值池排除本行全部答案值 |
| 2 | 干扰事实的实体可以来自**对侧切分**，"train/val 实体不重叠"只对目标键成立 | — | 干扰实体只从**同侧**实体池取（`entities + extra`） |
| 3 | 生成器只打印，不校验 | — | 新增 `validate()`：答案值无泄漏、干扰键不冲突不重复、`unit_meta` 与 `units` 逐条一致、反事实配对行数与 gold 差异、跨侧实体名不出现 → 任一不成立 **SystemExit，拒绝落盘** |

数据已重新生成（192 / 72，探针分布仍 64/64/64），`unit_meta` 字段结构不变。

### 四、新增：变异审计 `tools/audit_mutations.py`

只跑单测不够 —— 上面 3 条恒真断言让 6 个 bug 中的 4 个在 67 项断言下安然无恙。
该脚本对每个"应该被抓的错误"做最小变异（不 detach / 组序反转 / 不掩码 / 读出退回键反推 /
版本指标不绑组别 / 并列取最旧），在临时副本里跑单测，**要求必须失败**；变异存活即判失败。
当前 **6/6 全部被抓住**（`results/fst_mem_2026-10-27/audit_mutations.log`）。

### 五、指标补了常数基线

`group_const_acc`（永远选第一组）、`ver_newest_acc`（永远选最新版）。
本轮 smoke 立刻见效：随机权重下 `group=0.333` 看起来"还行"，而常数基线是 **0.667** ——
即当时连盲选都不如。没有基线，这个数一定会被误读成"有点效果"。

### 六、本轮**没有**产生任何效果证据

仍是一步真实训练都没跑；所有数字来自 `--smoke-model`（hidden=64 随机模型 + StubTok 解码），
`acc=0.000` 是必然结果。核查轮的价值只在于**此前的"管线通了"不再包含已知的静默错误**。


## [v9.6-fst-mem] — 2026-10-27 — FST-Mem：事实键 + 状态/轨迹解耦 + 时序两级读出（阶段 B，**未训练**）

**一句话**：把 `LatentRouter` 从"候选方法"降级为**静态多事实内容寻址基线**，在它之上加三件事：
① 寻址键从"整条事实"改成 **(实体,属性)** —— 键**与值无关**，值变了键不变；
② 每个事实键拆成**当前状态 + 有限历史轨迹**，写新值 = 旧值下沉、新值上位，**其他键逐位不动**；
③ 读出改成**两级**：先选 (实体,属性)，再在被选中组内部按时间条件选版本，Reader 只收到 1 个版本的
m 个软 token。**真实训练一步没跑，没有任何效果数字**；`hops>1` 未实现（显式报错）；
未接进 AR/CR 线上管线。

### 一、为什么要动（研究定位的修正）

原方案的 `LatentRouter` = Compressor + 内容寻址 + Bridge + 软 token，三个组件都有成熟研究基础，
不足以撑起论文贡献；而且它只解决**静态多事实读取**，完全没用到项目已有的**事实覆写**与**时间记忆**能力。
新的研究问题是：**事实被压缩成潜向量之后，在不断新增、局部更新、历史回溯的过程中，
如何保持实体—属性绑定、避免无关事实干扰、并按时间取回历史值。**

### 二、三个机制 ↔ 代码

| 机制 | 代码 | 关键契约（单测直接断言） |
|---|---|---|
| 一 · 实体—属性级稳定寻址 | `latent_memory.LatentFactKeyEncoder` | `k_{e,r}=norm(out([h_e,h_r,h_e⊙h_r]))`；不同属性给不同键；实体/属性**不可交换**（有交互项）；梯度到四个头 |
| 二 · 状态—轨迹解耦槽位 | `latent_memory.LatentStateTrajectoryBank(LatentSlotBank)` | 写新值→旧值进 hist；**改 A 键时 B 键载荷 `torch.equal` 逐位相等**；超容量**真正释放**槽位；`read(key, time)` 的匹配优先级 |
| 三 · 时序条件两级读出 | `latent_memory.FstRouter` | 事实级打分**与组数/顺序无关**；`logits_ver` 长度 = 组内版本数（不是 G×V）；掩码位 `-1e4` 且不会被选中；**STE 前向 `torch.equal` 等于硬选** |

### 三、本轮新增/改动

| 文件 | 内容 |
|---|---|
| `latent_memory.py` | `LATENT_HIST_MAX`；`fact_key/split_fact_key/_norm_time/_time_match_score/_as_year`；`extract_text_feature`；`LatentFactKeyEncoder`、`TemporalStateSelector`、`FstRouter`、`_payload_md`、`pad_version_stack`；`LatentStateTrajectoryBank`（`write/read/list_versions/gather_versions/gather_all_versions/stats`）；`_emit_latent`（把"算好的 soft 注入并生成"抽出来，供两条读出路径共用）；`generate_latent_answer_fst`；ckpt 支持 `has_fst` + `fst_router`（缺权重**报错**，不许退回裸拼） |
| `train_latent_memory.py` | `--fst / --hist-max / --ver-weight / --ignore-query-time`；`_load_backbone()`（常规与 FST 共用一份加载逻辑）；`build_fst_bank / fst_group_tensors / fst_build_inputs / evaluate_fst / train_fst`；`load_synth_jsonl` 保留 `unit_meta/as_of/probe/fact_key/timeline/answer_fact_time`；`SmokeModel.generate()`（此前没有，`--smoke-model` 一评估就 AttributeError） |
| `tools/gen_fst_update.py` | 新数据集：同实体多属性 + 多次覆写 + 时间条件；192 训练 / 72 验证，实体不相交；干扰项键不与目标键冲突 |
| `tests/test_fst_mem.py` | **67 项**机制断言（F 事实键 / G 键编码器 / H 库语义与更新局域性 / I 两级寻址与 STE / J 读出只注入 1 个版本且值不进 tokenizer / K ckpt 往返） |
| `results/fst_mem_2026-10-27/` | 本轮完整记录 + 7 份日志 |

### 四、本轮踩到并修掉的 3 个真 bug（都会静默失效）

| # | Bug | 后果 | 修法 |
|---|---|---|---|
| 1 | `LatentStateTrajectoryBank.versions()` 与父类 `self.versions`（`{slot_id: 覆写次数}` 字典）**同名** | `TypeError: 'dict' object is not callable`，库直接不可用 | 改名 `list_versions/count_versions`，并在代码里写明为什么不能叫 `versions` |
| 2 | 版本级监督打在 `logits_ver`（**被选中组**那一行）上 | 路由初期选错组时，监督指向另一组的版本布局（甚至填充位），`loss_ver` 从 0 跳到 **9990** | `FstRouter` 额外返回 `logits_ver_all[G,V]`，监督取 `[tgt_group]` 那一行 |
| 3 | STE 软期望是 fp32、硬选是 bf16，相加被提升成 fp32 | bridge 的 bf16 权重直接报错（或悄悄全精度往返） | `z_soft.to(dtype=z_hard.dtype)` 后再做"梯度直通" |

另外实现时**刻意没做**：块对角注意力掩码、绑定子空间、正交键约束 —— 两级读出已经在**输入层**
把"只送一个版本"做到位；在证明这一步不够之前改冻结模型的注意力结构属于"加模块 ≠ 加贡献"。

### 五、验证（都是通路验证，**不是**效果）

| 项 | 结果 |
|---|---|
| `tests/test_fst_mem.py` | **ALL PASS (67 checks)** |
| `tests/test_latent_readout.py` | **ALL PASS (89 checks)**（重构 `_emit_latent` 后逐条不回归） |
| `--fst --smoke-model` 全链路 | 训练 6 步 → 评估 → 存盘 → 重载 → 评估，`grad-check` 五个头全 true，`n_soft=4`（= m，与版本数无关） |
| `--fst --hist-max 0`（硬覆写对照） | `skip_evicted` 递增（历史题无解 → 跳过）、`ver_multi=n/a(0)`：符合预期 |
| `--fst --ignore-query-time`（时序条件消融） | 正常跑通，指标与完整版可分栏对比 |
| 阶段 A（`--router`）回归 | `[step]`/`[eval]` 输出与改动前一致（本地仅 `deltamem` 导入失败，该包只在服务器上有） |

### 六、没有做的事（不要误读）

- **一步真实训练都没跑**：`acc/group_acc/ver_multi_acc` 在 smoke 里都是随机权重下的噪声，
  **不能**作为"FST 有效"或"FST 无效"的任何证据。
- `hops>1` 未实现；FST 未接进 AR/CR 线上管线（需从 `StatePool`/`EpisodicMemory` 构造事实键与时间线）。
- 历史容量 `LATENT_HIST_MAX` 的容量—保真权衡未量化。
- 阶段 A 的 22.5% 数据歧义缺陷在新数据集已规避，但**旧数据集未重生成**。

## [v9.5-latent-router] — 2026-10-27 — "17% 之谜"水落石出（4 个自伤脚本 bug）→ 位置敏感根因 → LatentRouter 重构 + 7 个自查失效修复

**一句话**：训练期 17.2% 与重载后 0% 的落差**不是产品 bug**，是我自己诊断脚本的 4 个 bug；
修干净后真相是**读出完全由"槽位排第几位"决定、与内容无关**（同一事实：单槽 15/16 → 4 槽 target_only
仍 15/16 → 4 槽 full 1/16 → 4 槽 target_end **0/16**）。已按「键值解耦 + 集合内容寻址」重构
`LatentRouter`，**但一步训练都没跑，效果未知**。重构后两轮自查又揪出 **7 个会静默失效的问题**
（含 1 个我自己的设计错误），全部修复。完整记录见
[`results/latent_router_2026-10-27/`](results/latent_router_2026-10-27/README.md)。

### 一、17% 之谜：4 个自伤脚本 bug（结论：checkpoint 存取没问题）

| # | Bug | 后果 |
|---|---|---|
| 1 | 诊断脚本硬编码 system prompt，与训练默认值不同 | 输入分布不同，指标不可比 |
| 2 | 诊断脚本未关闭 Delta 写入 → `delta_state` 跨样本累积 | 前几条正常，之后整体崩到 0 |
| 3 | `generate(inputs_embeds=...)` 返回序列不含输入前缀，脚本却切 `out[0, n_in:]` | 解出空串 → 假 0/32 |
| 4 | `load_synth_jsonl` 把 `memory` 变成拼接字符串，脚本用 `memory[0]` 取到第一个**字符** | 造出 `["T","M","S"]` 畸形槽位 |

已直接测量排除（全部无关）：delta_state 污染最终分数、`comp/bridge` 的 train-vs-eval 模式、
`torch_dtype=` vs `dtype=`、`attn_implementation="sdpa"` vs `"eager"`、adapter 路径默认值、
`load_delta_mem_adapter` 位置/关键字参数、forward 是否改写权重（722 个参数张量哈希前后一致）、
`load_state_dict` 严格性。

### 二、真正的根因（干净重测，训练同款 prompt + Delta 写入关闭 + 每条 reset）

| ckpt | 条件 | 严格全匹配 | pair-acc |
|---|---|---|---|
| 1 事实 | 单槽 | **15/16 (93.8%)** | 7/8 |
| 1 事实 | 4 槽 target_only（内容相同，只是外壳是 4 槽） | **15/16 (93.8%)** | — |
| 1 事实 | 4 槽 full（+3 干扰，目标首位） | **1/16 (6.2%)** | — |
| 1 事实 | 4 槽 target_end（目标末位） | **0/16 (0%)** | — |
| 4 事实 | 训练集 full / 验证集 full | 79/200 (39.5%) / **11/64 (17.2%)** | 16/100 / **0/32** |
| 4 事实 | 训练集 target_only / 验证集 target_only | **0/200 / 0/64** | 0/100 / 0/32 |
| — | 明文 4 槽（32 条）/ Oracle（32 条） | 24/32 (75%) / 17/32 (53%) | — |

三条判据：①**target_only 只喂正确那条反而 0/200**——若真在按内容寻址，这是最容易的情形；
②1 槽 ckpt 套 4 槽外壳就崩（15/16→1/16），内容一模一样 → 纯位置效应；
③**17.2% 低于 31.2% 的 relation-prior 基线**（6 种关系、18 个答案值，按关系猜就能到 31.2%）→ 是噪声不是能力。
瓶颈在 **Compressor + Bridge**，不在 Reader（明文 75%、Oracle 53% 都远高于 17.2%）。

### 三、重构（`latent_memory.py` / `train_latent_memory.py`）

`LatentRouter`：`s_k = <q, k_in_k>/τ` 无位置内积打分；**推理 argmax 只送 1 个槽**（m 个软 token），
让 bridge 永远看到 Phase 1 单槽形状 → 位置先验被彻底去掉。键值解耦：`k_in` 编码主语+关系（寻址）、
`k_out` 编码客体值（多跳接力 `chain()`）。查询取 `[L0 mean ‖ L18 mean]`（词形 + 语境）。
训练侧新增 `--router/--d-key/--addr-weight/--freeze-compressor/--max-slots`，
`sample_slot_subset` 做变槽数 + 随机顺序（必含目标槽），`loss_addr` 显式监督寻址。

### 四、两轮自查揪出的 7 个失效（含 1 个我自己的设计错误）

| # | 问题 | 证据 |
|---|---|---|
| 1 | **软门控聚合做训练前向，与推理硬选不同分布**（我的设计错误） | `‖z_soft‖=61.9` vs `‖z_hard‖=100.2`，距离 54.6%，`w=[0.18,0.05,0.23,0.54]` 远非 one-hot。改为直通估计 `z_hard + (z_soft - z_soft.detach())`，前向与推理逐位相同 |
| 2 | 原有的 `torch.as_tensor(payloads)` 短路新加的 stack 分支 | 多块列表下 `ValueError: only one element tensors can be converted to scalars` |
| 3 | 槽边界丢失：cat 后 Σk>槽数时把"块"当"槽" | 4 槽其中 1 槽切 2 块 → `cat=(5,4,2560)` 当成 5 个槽 |
| 4 | 评测管线不传 router（`run_ar_v7`/`run_group_b2_v7`） | Phase 2 ckpt 会退回 K*m 裸拼，位置敏感原样复现 |
| 5 | `evaluate_recon` 不走路由 | 训练 recon 用 4 token，评估用 16 token，口径不一致 |
| 6 | 多跳样本被随机丢槽会剪断推理链 | 现有数据全 `hops=1`，此路径未验证 |
| 7 | **寻址监督 13.5% 标签是错的**：`target_slot_index` 只用答案串匹配，而 22.5% 的样本答案值在干扰事实里也出现 | 27/200 标错槽位；`loss_addr` 把 router 往错方向训且损失照样下降。根因是 `load_synth_jsonl` 丢了 `evidence`。修后 200→0、64→0 |

### 五、验证

- `tests/test_latent_readout.py` **86 → 89 项全过**（A1 白名单纳入 `router/q_feat/layer/hops` 并保留明文禁令；新增 A8 有 router 时 `n_soft=m`、A9 `routed` 下标、A10 `n_slots_in`）。
- 真实 Qwen3-4B + Delta-Mem 端到端：`slot_z=(4,4,2560)`、`logits=(1,4)`、`chain(hops=2)=[0,1]`、`n_soft=4`（hops=1）/`9`（hops=2 含换行）、无 router 回归 `n_soft=16`、ckpt 往返 `has_router→d_key=256`、老 ckpt `router=None`。
- `--max-unit-tokens 8` 强制多块：修复后 `soft_tokens=4`（槽边界保住）。
- Phase 2 冻结：`gnorm_comp=0.000e+00`；init ckpt 在 router 随机初始化下仍 3/4（未破坏已有通路）。
- 4 槽训练冒烟：`loss_addr` 0.68→1.33→0.67、`gnorm_router` 6.1→0.53（梯度确实存在）。

### 六、新增文档

[`DATASETS.md`](DATASETS.md)（数据盘点：3 套在用 / 5 套未用 / 0 外部 benchmark / 3 个缺口 / 2 个数据缺陷）
与 [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md)（9 条未修问题，含 P0 的 `delta_write_disabled` 永久关写）。

### 七、明确没有做的事

**一步训练都没跑**，所以"路由能否救回 4 槽"无任何证据；多跳 `chain` 无真实数据验证；
bottleneck 未扩容（512 未动，单槽已 93.8%，瓶颈不是容量）；评测管线 3 处已知修复未动。

## [v9.4-latent-gpu-run] — 2026-10-22 — 阶段 A 首次真机训练：跑完但没学会（0/32），瓶颈定位到压缩器

**一句话**：在 v100-2 上用 15,000 条合成语料跑完 1,875 步（3.7 h），loss_qa 12.6→3.87，
但**验证 exact-match = 0/32**。四组对照实验把瓶颈钉死在**压缩器学不会编码**，
并顺带修掉 3 个只有真机才会暴露的问题。日志与完整数字见
[`results/latent_stage_a_2026-10-21/`](results/latent_stage_a_2026-10-21/README.md)。

### 训练结果

| 项 | 值 |
|---|---|
| 数据 / 超参 | 15,000 条（12,002 单跳 + 2,998 多跳）；stage a，m=4，layer=18，accum=8，lr=1e-4，1 epoch |
| 步数 / 耗时 | 1,875 步 / 13,454 s（3.7 h），V100-32G，显存 9.8 GB |
| 可训练参数 | 7.62 M（compressor 4.99 M + bridge 2.63 M），冻结 4,027.9 M |
| loss_qa | 12.59 → 8.73 → 7.73（分段均值）→ **3.87**（末步） |
| **验证 exact-match** | **0.0（n=32）** |

生成不是乱码，是**格式对、名字错**（`Kestrel Analytics Vrindkraulmiothd, a` vs 正确
`Ironwood Energy Nilork`）——所以不是读出端崩了，是记忆没被编码进去。

### 三个只有真机才暴露的问题（已修）

1. **软 token 尺度差两个量级**（`latent_memory.py`）：实测 `bridge` 输出每向量范数 **96–159**，
   而 Qwen3-4B 真实 token embedding 只有 **1.05**。软 token 直接拼进 `inputs_embeds`，
   尺度差两个量级就成了 **attention sink**，把真实 token 淹没。
   修法：`LatentBridge` 输出**方向归一化 + 受限可学习尺度**（初值取真实 embedding 范数，
   夹在 [0.25, 4] 倍，防止训练把尺度推回爆炸区）。新增 C12–C14 三条断言。
2. **掩码设备错配**（`latent_memory.py`）：`encode_latent_units` 为省显存把 `h0/h18` 留在 CPU 返回，
   压缩器里才搬到 `p.device`；但 `key_padding_mask` 跟的是 `h0.device`（CPU），GPU 上 SDPA 直接报
   `attn_bias is on cpu, different from other tensors on cuda`。修法：掩码锚定**参数设备**。
   CPU 单测里两者都是 CPU，这个 bug 恒不触发。
3. **`deltamem.core` 在 transformers 4.51 上根本 import 不了**：`backbone_compat.py` 顶层无条件
   import `transformers.models.smollm3`（5.x 才有），于是 `HFDeltaMemConfig` 等入口全部拿不到 ——
   尽管本实验只用 Qwen3。新增 `deltamem_compat.py` 注入最小占位（3 个名字），真实模块存在时
   完全不介入；占位类不实现 forward，真被用到会显式报错而不是静默算错。

另：长跑原先只在结束时落盘（3+ 小时一次，中途崩掉全丢），新增 `--save-every`（默认 500 步）。

### 瓶颈定位（这是本轮真正有价值的产出）

同一批 200 条样本、8 epoch、400 步，**全部在训练集上评**（最宽松判据）：

| 条件 | QA exact-match | 结论 |
|---|---|---|
| m=4（16 软 token） | 1/200 | 连 200 条都记不住 |
| m=16（64 软 token，加宽 4×） | 3/200 | **不是容量宽度问题** |
| m=4 + 范数对齐 | 0/200 | 尺度是真问题，但事后修不够 |
| m=4 + `--recon-weight 1.0`（强制重建原文） | 0/200，重建 token-acc **22.8%** | **逼不出来** |

真模型上的三条对照（新工具 `tools/diag_latent_capacity.py`）：

| 子命令 | 结果 | 排除了什么 |
|---|---|---|
| `--sub text`（明文记忆） | **6/6** | 数据、模型、任务、评测口径 |
| `--sub oracle`（记忆原文 embedding 当软 token，跳过压缩器） | **4/4** | 软 token 注入通路（位置/mask/拼接） |
| `--sub scale` | 软 token 范数 96–159 vs 真实 1.05 | 尺度（已修） |
| `--sub probe` | 重建 token-acc 22.8%；平凡基线 top1 6.0% / **top10 19.8%** | **压缩器几乎没编码进内容** |

最后一行是决定性的：模板化文本里功能词占比极高，"永远只吐最高频 10 个 token"就能拿 19.8%，
实测 22.8% 只高 3 个点 —— 等于没编码。而 oracle 4/4 证明**只要软 token 里真有信息，模型一定能用**。

### 两次外部事故（不是代码问题）

- 卡 1 上跑到 step 532 被**外部杀掉**（无 traceback，同刻全机 10 张卡任务全被清空，机器没重启），
  从 step 400 的 checkpoint 续训救回 45 分钟。
- 换回卡 1 重试时 OOM（该卡已被另一任务占满 31.7/32.7 GB），改跑空闲卡 5。
- 2026-10-21 21:25 env 的 `transformers` 被**别人改坏**（editable 指向不存在的
  `/root/transformers-main`），全环境 `import transformers` 失败；训练已在 20:46 结束不受影响，
  后续全部改用 vendored 5.19.0（`PYTHONPATH=/share/exp4group/shim19:...`）。

### 未验证（必须保留的披露）

- **没有跑更长训练**：只跑了 400 步（过拟合）与 1,875 步（全量）。soft-prompt 类方法通常要几千步、
  lr 大一个量级。**"没训够"与"结构学不出"尚未区分** —— 这是下一步实验。
- **没有跑 stage B（KL 蒸馏）**：所有数字都来自 stage a 的纯 QA 交叉熵。
- **没有在评测集上跑过**：0/32 是训练期验证观察值，不是 MemoryAgentBench / LongMemEval 指标。
- **探针的对照组失效**：200 样本 × 4 万维输入，线性层怎么都能硬背（随机输入也能 100%），
  所以探针数字只能当"有信号"的弱证据；真正的判据是重建准确率对平凡基线。

## [v9.4-synth-data] — 2026-10-21 — 训练语料生成器 + 训练脚本 `--data` 接入（含 2 个真实 Bug 修复）

**一句话**：训练语料不再只能来自内部 20 行模板 —— 新增"虚构事实宇宙 → 自然陈述句 →
`parse_fact_triple` 抽取 → 单跳/多跳问答"的完整通路，v100-2 上 **32 进程并发**产出
**15,000 训练 + 2,000 验证**；`train_latent_memory.py` 用 `--data/--val-data` 直接消费，
**每条事实一个记忆槽位**（与推理期 bank 布局一致）。

### 新增文件

| 文件 | 行数 | 作用 |
|---|---|---|
| `tools/gen_fact_universe.py` | ~175 | 虚构事实宇宙：1 簇 = 1 人 + 雇主/出生地/母校/配偶/著作/职位（16 条事实） |
| `tools/gen_synth_qa.py` | ~560 | 三元组 → 单跳/多跳问答；29 条关系模板 + 同义问法池；按连通分量并发；唯一性/泄漏审计 |
| `tools/demo_facts.txt` | 30 | 手写演示语料（28 条可解析 + 2 条故意噪声，用于验证过滤器） |
| `tests/test_synth_data.py` | ~185 | **32 条断言**：加载 / 记忆单元 / 切分泄漏 / 编码拼接 / CLI 契约（v100-2 上 32/32 PASS） |

### 两个真实 Bug（都不是本轮引入，是上轮"只改不跑"遗留）

1. **训练时没走 bridge**（`train_latent_memory.py` 训练循环）：`build_student_inputs` 直接吃压缩器输出 `z`，
   bridge 根本不在计算图里 —— 训完是随机投影，而推理期 `generate_latent_answer` 内部正是
   `soft = bridge(payloads)`。表现：`[grad-check] bridge_last_linear=false`。
   修法：`soft = bridge(z)` 之后再进 student / 重建任务。
   **这类 bug 不会报错**，只会让"训练过的模型"和"推理用的模型"是两个东西。
2. **`SmokeModel` 两处错误**：`forward` 里 `self.emb(x)` 引用未赋值变量（`input_ids` 路径必然
   `UnboundLocalError`）；`output_hidden_states=True` 时仍返回 `hidden_states=None`
   （编码器直接抛"模型没有返回 hidden_states"）。两处都只在 `--smoke-model` 路径触发，
   所以"从未执行过"= 从未暴露。另：smoke 模型只有 2 层，`--layer 18` 会越界，
   现在显式降级并打印说明（不做"重复末层"之类的假包装）。

### 训练脚本接入

- `--data PATH` / `--val-data PATH`：读 `gen_synth_qa.py` 的 jsonl；
  `--n-train 0`（新默认）= `--data` 时全量，不给 `--data` 时仍用内部 `make_dataset`（2000 条）。
- `load_synth_jsonl()` 同时保留两种粒度：`units`（逐条事实，用于切槽）与 `memory`（拼接串，
  用于 teacher 明文与重建任务）。**按事实切槽而不是整段压一个槽**：推理期一个槽位就是一条事实，
  整段压缩会让"检索到某一条事实"在训练里根本不存在。
- `encode_item()`：多单元沿**块轴** `cat` → `[Σk, m, d]`，单单元原样返回；空 payload 抛错。
- `check_split_leakage()`：加载时检查 train/val 的**簇重叠**与**重复问句**并打印告警；
  未给 `--val-data` 时日志明确写"该通路检查数字不是验证指标"。

### 生成侧踩过的三个坑（都已修）

1. **并发退化成单进程**：按"共享实体"连边时，`chief engineer`（12 个取值）、作品名（144 种组合）
   把 1,800 个簇连成 1 个连通块 → `workers=1`，2m11s 全串行。改成只在**实体型关系**上连边后，
   32 进程真并行（`user/real = 12.5×`）。
2. **6.71% 样本答案出现在两条记忆里**：干扰事实可能自带答案（两个人同为 chief engineer）。
   加"干扰事实客体 ≠ 答案"后降到 **0.01%**。
3. **机构名/作品名跨簇撞名** → train/val 共享实体 1 个。两者都进全局唯一池后交集归零。

### 实测（v100-2，`/share/exp4group/synth/`）

| 项 | 数字 |
|---|---|
| `train_15k.jsonl` | 15,000（12,002 单跳 + 2,998 多跳），10.2 MB |
| `val_2k.jsonl` | 2,000（1,680 单跳 + 320 多跳） |
| 事实宇宙 | 32,000 条事实 / 2,000 簇 / 20,000 唯一实体名 |
| 生成耗时 | 15k：**14.7s** wall（CPU 3m3s，32 进程，加速 12.5×）；val：3.9s |
| 审计 | 唯一 (问,答) 15000/15000；答案≠1 条记忆 **1 条（0.01%）**；泄漏进问句 0；不在证据行 0；train/val 实体交集 **0**、簇交集 **0**；27 种关系 |
| CPU 通路自检 | 阶段 A / 阶段 B(+KL) 均 `[dry-run] OK`；4 项梯度检查全 `true`；`test_synth_data` 32/32 PASS |

### 首次执行上轮单测：80/82 通过，2 条失败各指向一个真问题

上轮（v9.3-latent）的 `tests/test_latent_readout.py` **从未被执行过**，本轮首次运行：

| 失败项 | 根因 | 处理 |
|---|---|---|
| `C6 bridge 末层 Linear 拿到梯度` | 训练时直接把 `z` 当 soft token，bridge 不在计算图里（就是上面第 1 个 Bug） | 测试改为 `soft = bridge(z)` 后再进 student，与修复后的训练契约一致 |
| `C4 compressor.queries 拿到梯度` | 测试桩 `StubModel` 是**点式** head（无跨位置混合），软 token 不可能影响答案位置 → 梯度**恒为 0**，该断言恒假 | 桩件加跨位置均值混合（模拟真实 attention）；断言从 `is not None` 收紧为**非零** |

修复后 **83/83 全通过**。另外新增 `C11`：绕过 bridge 时 bridge 必须拿不到梯度 ——
把"训练必须过 bridge"这条不变量变成可失败的回归守卫。

> 教训：`is not None` 这种断言挡不住"全零梯度"。真正判据是**梯度非零** + 一条"反向情形也必须失败"的守卫。

### 披露（必读）

1. **没跑 GPU 训练**：`--data` 通路只在 `--smoke-model`（随机小模型，CPU）上验证。
2. **15k 是数据量，不是准确率**：本轮没有任何 latent 效果数字。
3. **语料是虚构的**：27 种关系、用词分布窄于真实文本；泛化到真实评测集的表现未测。
4. **未用真实语料**：服务器上唯一的本地数据集就是评测集（`Conflict_Resolution-*.parquet`），
   不能用于训练；`--facts` 指向任意独立语料即可替换。
5. **`--variants 2`** 只用了每关系 2 种问法；多跳只有 2 跳，3 跳以上未生成。

## [v9.3-latent] — 2026-10-21 — 潜层读出（路径 B）代码落地：压缩器 + 桥接器 + 软 token 注入

**一句话**：`AR_READOUT_MODE=latent` 从"直接 exit 2 的占位开关"变成**真正可运行的潜层读出通路** ——
证据不再以明文进入生成端，而是先压成 `Z ∈ R^{m×d}`（m=4 个软 token/槽位），
经桥接器投影后直接拼进 `inputs_embeds`；配套训练脚本与机制性单测。

### 新增文件

| 文件 | 行数 | 作用 |
|---|---|---|
| `latent_memory.py` | 1068 | 压缩器 / 桥接器 / 载荷库 / 编码器 / 读出器 / CR 路径预算 / checkpoint |
| `train_latent_memory.py` | 642 | 冻结 Qwen + Delta-Mem adapter，只训 Compressor + Bridge（两阶段） |
| `tests/test_latent_readout.py` | 540 | 10 组机制断言（见下"测试"） |

### 核心接口

- `LatentCompressor`：m 个**可学习 query** 对 token 级 `[h0; h18]` 做 cross-attention
  → `[B, m, d]`。为什么不用现成的 `_layer18_provider()`：那个接口返回 mean-pool 后的 `[N,d]`，
  token 轴已被压掉，拿它做 token-level 压缩等于对着一句向量再注意一次。
- `LatentBridge`：`LayerNorm → Linear(d→512) → SiLU → Linear(512→d)`，把 `[k,m,d]` 映射成
  k×m 个软 token 的**嵌入**（与词嵌入同空间，可直接拼 `inputs_embeds`）。
- `LatentSlotBank`：以**稳定槽位 ID**（`epi:3` / `state:0` / `cr:12`）为键存载荷；
  `append` 重复 ID 报错、`overwrite` 缺失 ID 报错、淘汰后 `shift_index()` 重编号。
- `generate_latent_answer(model, tokenizer, bridge, payloads, question, system_prompt, ...)`：
  签名里**没有任何**能塞回明文证据块的参数；返回 `decode_mode` 记录后端 `generate`
  返回的是"新 token"还是"整条序列"（两种真实实现都存在，切错会静默错位）。

### 接线

- `run_ar_v7.py`：`AR_READOUT_MODE=latent` 全链路（建池 → 建载荷库 → 逐题寻址 → 取载荷 → 生成），
  逐题记录 `readout.mode="latent"` / `latent.slot_ids` / `n_vectors` / `budget_used` /
  `dropped_epi_ids` / `checkpoint` / `decode_mode`；新增 `latent_dry_run()`（CPU 自检）。
- `run_group_b2_v7.py`：`CR_READOUT_MODE=prompt|latent`，latent 下 CR 生成**从潜载荷构造**，
  不再拼接事实块；多跳路径按 `select_latent_cr_paths()` 整条保留/整条跳过。
- `v9_slotpool.py`：`slot_uid()`、`attach_latent_bank()`、`latent_payload()` 与
  `_store` / `_replace` / `_drop` 三处同步钩子；新增 `LatentIndexedPool`。
- `v9_cr.py`：`retrieve_multihop_ids()`（纯寻址，**不渲染明文**）/ `render_context()`（明文基线）
  / `retrieve_multihop_context()`（兼容包装，`last_dbg` 键全保留）。
- `run_manifest.py`：`file_fingerprint(path, label)` → `{label, exists, bytes, sha256_16}`，
  checkpoint 进 `config_id`（换了权重不会撞同一个结果文件）。

### 三个设计决策（都写在代码注释里）

1. **分块而非截断**：单元长于 `LATENT_MAX_UNIT_TOKENS=64` 时切成不重叠块并记父槽位 ID。
   截断可能正好丢掉答案，指标下降却查不出原因。
2. **载荷以 ID 为键**：`_replace` 是原位覆写（下标稳定，版本号 +1），但 `_drop`/`_evict`
   会移动下标 → 必须 `shift_index(namespace, drop_index)`，否则 `state:5` 会指向另一条事实。
3. **`detach` 分两条路**：存储路径默认 `detach=True`（不把训练图带进库）；
   训练路径必须 `detach=False` 且 student 前向**不能**放进 `torch.no_grad()`。

### 默认行为变化

**没有默认行为变化**。所有新开关默认关闭（`AR_READOUT_MODE=prompt` / `CR_READOUT_MODE=prompt`），
且指纹字段是**条件写入**：`latent` 节只在 latent 模式加、`readout` 节只在 CR 非 prompt 时加、
`qtxt_no_gold` 只在显式设过环境变量时加 —— 既有 `config_id` 全部不变，旧结果文件与续跑校验不受影响。

### 测试

`tests/test_latent_readout.py` 10 组 / **82 条断言**（75 个断言点，其中签名黑名单 1 处展开为 8 条），覆盖：
读出接口无明文（签名 + 实际送进 tokenizer 的文本）、软 token 位置/掩码/内容、
梯度到 Compressor/Bridge 而冻结模型无梯度、长单元分块不截断（`Σn_tokens` 守恒、span 连续）、
载荷库语义（重复/缺失/版本/淘汰重编号/命名空间隔离）、CR 路径预算（不出现半条路径）、
`config_id` 隔离 + `file_fingerprint`、全 padding 行不产生 NaN、checkpoint 往返与三种报错。

**未做的部分（如实披露）**：

1. **本轮没有运行任何代码**（用户明确要求"先统一修改好代码，不允许运行"）。只做了
   `py_compile` 静态语法检查（**受影响的 8 个文件**全过，全仓 34 个 `.py` 未逐一重跑）与
   AST 未定义名扫描 —— 二者都不导入、不执行模块。
   因此 `test_latent_readout.py` **从未被执行过**，首次运行可能有个别 API 细节要修。
2. **压缩器与桥接器至今未训练一步**。`LATENT_CKPT` 缺失时两个 runner 直接报错退出，
   不会拿随机投影跑对照 —— 随机初始化在维度上完全合法，但冻结 Qwen 没有理由读懂它，
   指标会接近"没有记忆"，一旦被当成"方法无效"就是误判。
3. **没有任何 latent 指标**。训练超参（`LATENT_LR=1e-4`、`LATENT_GRAD_ACCUM=16`、KL 权重、
   `--counterfactual`）都是设计值，收敛性与收益均未验证。
4. `answer_in_context` 在 latent 模式下是 **None**（`used_texts` 保持空），不是 `False`；
   检索侧命中率与潜层读出准确率必须分开报，不能拿明文匹配当"答案进了上下文"。

## [v9.2-l18] — 2026-10-21 — 第 18 层稠密通道：池内 whitening（默认开）+ 复用流式中间态

**一句话**：L18 稠密通道从"纯 L2 余弦"升级为"池内中心化 + 协方差白化"，并把调用方
已算好的向量做成可按文本复用的预计算矩阵 —— **默认行为变化**（见下），旧口径保留为消融对照。

### 默认行为变化（AR_V9_L18=1 时）

| 项 | 旧（v9.0/v9.1） | 新（v9.2，默认） |
|---|---|---|
| 稠密余弦口径 | 原始隐空间 L2 归一化后点积 | 池内中心化 + 白化后再点积（`V9_L18_WHITEN=1`） |
| 池内两两余弦 | 被"共同方向"抬高（合成数据实测均值 0.86、max 0.98） | 均值≈0、分布显著分散（排序才有区分度） |
| 状态池向量 | 对重叠文本再前向一次 | 复用 epi 池已算好的向量（`V9_L18_REUSE_EPI=1`） |

- `V9_L18_WHITEN=0` 恢复旧的纯 L2 行为（逐位一致，已加断言）；`V9_L18_WHITEN_EPS`
  （默认 1e-4）同时是协方差特征值下限，保证数值稳定。
- 数值稳定性：`d>=N` 或 `N<=1`（协方差秩不足，白化会放大噪声）自动退化为**只中心化**，
  并把实际走的模式写进 `layer18_spec()` / `l18_whiten_fit()` 的 `info.mode`。
- 复用接口：`set_layer18_provider(fn, precomputed=[N,d], precomputed_texts=[...])` /
  `set_layer18_precomputed(hidden, texts=...)`。查表按**文本**匹配，因此 epi 池与状态池
  共享同一份矩阵也不会串行号；没有预计算时退回按需前向并**告警一次**。
- 指纹：`l18_whiten / l18_whiten_eps / l18_reuse_epi` 已进 `run_manifest` 的 config_id，
  WHITEN=0/1 两次跑不会撞同一个 id（否则续跑会把两种口径混进一个文件）。

### 测试

`test_v9_slotpool.py` 249 → 284 条断言（新增 6 个用例 / 35 条断言）：whitening 前后余弦
分布变化、`d>N`/`d==N`/`N<=1` 退化路径、`WHITEN=0` 与旧口径逐位一致、预计算矩阵命中时
provider 零调用（计数器）、按文本查表不串行号、状态池复用 epi 向量（开/关对照）。

**未做的部分（如实披露）**：本轮无 GPU 机器，whitening 的**效果**只在合成/CPU 单测上验证，
EventQA/LongMemEval 上的真实指标对照尚未跑；`V9_L18_WHITEN` 默认改为 1 属于口径变更，
与旧跑分不可直接横比，跨口径比较请显式设 `V9_L18_WHITEN=0` 复现旧数字。

## [v9.2-audit] — 2026-10-21 — 20 项审计逐条核验表

逐条核验用户 2026-10-21 审计报告的 20 项要求（代码 / 测试 / 文档三方对齐），产出
`experiments/audit_2026-10-21.md`：20 行核验表（每行带 `文件:行` + 测试名）+ 状态统计
（已修 12 / 部分修 2 / 仅合成验证 1 / 待接线 1 / 已知风险 1 / 方向未定 1 / 未修 1 / 未验证 1）
+ 风险表（P0：Delta 效果未验证、证据级召回缺金标；P1：四格仅合成验证、LME 未接线、CR 双向风险；
P2：块内 serial 回退、v8/v9 时间语义双份实现）+ t6 预算口径 + 复现命令 + 文档遗留清单。

**这不是成绩单**：表里明确区分"已修 / 部分修 / 仅合成验证 / 待接线 / 未修 / 未验证"，
不得把 `topk_refresh`（词面 IDF 近似）写成"检索器驱动刷新"、不得把 Delta 写成"已验证有效"、
不得把窗口方向写成已定论、不得把合成探针结果表述成真实数据集的证据级召回。

## [v9.2-t19] — 2026-10-21 — CR 两处硬门控的反向风险量化（纯解析，无 GPU）

**一句话**：把"漏链 / 误拒覆写"从口头风险变成数字 —— 属性门控的代价是 **0**，
2 跳桥接的代价**测不出来**（判定样本太小），两者都**不触发**归一改动。

### 指标与结论（`experiments/results/cr_link_recall.json`）

| 项 | 口径 | 结果 | 结论 |
|---|---|---|---|
| B 覆写丢失 | **槽级**：同主语 + 同 loose 属性下池中并存多个不同值的活槽 | **0 / 260 = 0.00%** | 未超 5% 阈值 → 不动 `_cr_verb_core` |
| B 覆写丢失（pair 级） | 应覆写对里属性键不同的比例 | 0 / 47 = 0.00% | 同上 |
| B（粗代理，不采用） | old/new 两值都作为子串出现在池文本 | 29 / 47 = 61.7% | **噪声**，会被别的事实的同名字符串污染，不可引用 |
| A 2 跳漏链 | 需链题目里因归一化不等而无法成链的比例 | mh 0.00%（n=10）/ sh 8.33%（n=36） | **未定论**：覆盖率仅 6% / 17% |

### 为什么 A 记"未定论"而不是"未超阈"

172（mh）/ 207（sh）条"需链"里，解析层有 **162 / 171 条找不到桥接候选**，真正被判过的只有
10 / 36 条。8.33% 是"36 条里 3 条"，不足以支撑一次归一改动。脚本内置样本量门槛
（`MIN_JUDGED=30`、`MIN_COVERAGE=30%`），不达标时结论字段直接写 `A_underpowered: true`。
要提高覆盖率需换事实库更全的切片（`sh_262k` / `mh_262k` 本机未落盘，JSON 里标 `unavailable`，不编造）。

### 数据来源

`results/groupA_factconsolidation_mh_6k.jsonl` 的 fact pool（100 行共享）+ mh_6k 100 题 +
`results/v6groupB2_factconsolidation_sh_6k.jsonl` 的 sh_6k 100 题；真实 CR 跑分产物，非合成。

## [v9.2-closeout] — 2026-10-21 — 审计收口：Delta 写入策略 / 时间进 prompt / token 预算 / CR Delta / LME 读出 / 联合归因 / 本地 Judge

**一句话**：把审计报告里"机制缺失"的项全部补上并接线，同时把**默认行为的两处变化**和
**五项仍未验证的效果**写清楚 —— 本轮仍然**没有任何可用于论文的指标**。

### 新增能力

| 能力 | 开关（默认） | 状态 |
|---|---|---|
| Delta 有效写入策略 | `AR_DELTA_WRITE_STRATEGY=all\|filtered\|topk_refresh`（`all`） | `all`/`filtered` 已实现；`topk_refresh` 是**问题词面 IDF 近似**（写入先于建池，拿不到真实检索结果）→ 部分实现 |
| 时间进生成 prompt | `AR_V9_PROMPT_DATE=1` | 行格式 `<serial>. [<date>] <原文>`；日期继承 Session 头；解析不出则不伪造（只到年份写 `[1997]`）。`=0` 与旧输出**逐字节一致** |
| 2500 token 证据预算 | `AR_V9_CTX_TOKEN_BUDGET=2500`、`AR_V9_CTX_MAX_UNITS=0` | 按分数降序装填、超预算即停、装完按位置排序（I2 不变）；OOM 重试改按证据优先级整行丢，删除"保留最后 N token"的静默截尾 |
| CR Delta 参与查询 | `CR_DELTA_MODE=off\|context\|refresh`（`off`） | 删除死开关 `B2_KEEP_DELTA`（只进指纹、从不控制 reset）；非法值 `SystemExit` 不静默退回 |
| CR 逐题证据日志 | 默认开启 | `retrieved_*` / `used_*` / 统一 ID `cr:<idx>` / `rec["delta"]`；FINAL-CR 行给 `answer_hit@K` 与 `answer_hit_ctx` |
| LongMemEval 读出 | `readout(mode="lme")`，开关 `V9_LME_TURN_WINDOW=1` / `V9_MAX_PER_SESSION=2` / `V9_SESSION_QUOTA=1` | Session/Turn 内扩展、跨 Session Max-2 配额、first/last/before/after 时间约束；`before/after` 是硬过滤、`first/last` 是重排、解析失败即 none。**runner 尚未接线 → 线上不生效** |
| v8 时间语义 | `V8_SESSION_RESET=1` | Session 边界重置 `last_ts`；`TIMESTAMP_RE` 扩到 yesterday / last week / 星期几；滑窗不跨 Session |
| 四格联合归因 | `ar_diag --joint-k`（默认 1） | `Reader Acc \| Recall=1`、检索失败率、读到但答错率；`evidence_recall_at_k` 全 None 时标 `evidence_based=False` |
| 本地 Judge | `--judge-backend local` | 与 http 共用 `build_prompt` / `parse_verdict`；缺 GPU / 缺权重 / 缺 `--judge-model` 一律 exit≠0 且**不回退 http**；`--compare-backends` 做等价性验证 |

### 修掉的确定性 Bug

1. **CR 系表谓词解析把不同属性判成同一谓词**：`Alice is a doctor.` 与 `Alice is vegetarian.` 都解析成
   `('Alice','is')`，写入后 `size=1 / n_dup=1`，医生事实被素食事实吃掉。现表语冠词+头词并入谓词键
   （`is a doctor` / `is vegetarian`），并加属性类别门控 `_attr_compatible` + `n_attr_rejected` 诊断。
2. **CR 双跳评分与路径渲染错位**：`chain_score` 改为显式 `(fact1, fact2)` 二元结构，
   `linked()` 要求 `obj(f1) == subj(f2)` 归一化相等，不再用软门控糊过去。
3. **L18 预计算矩阵按 offset 盲切 → 静默取错向量**：`StatePool._replace` 传的是状态池自己的槽 id，
   多池共享同一矩阵时会拿到别人的向量且不报错。改为按**文本**查表（见 [v9.2-l18]）。
4. **`extract_units` 的 `ts` 只到年**（`2023/05/20` → `2023`），使 LME 的 `before/after` 在同一年内
   毫无判别力。新增 `anchor_ts`（Session 头日期，整日精度），`unit_date` 取最精确者。
5. **块行缺日期**：时间此前只进检索打分、不进模型输入；现每行带 `[date]` 前缀。

### 默认行为变化（对比旧跑分前必看）

- `AR_V9_CTX_MAX_UNITS` **40 → 0**：拼块改由 token 预算决定，复现旧口径设 `AR_V9_CTX_TOKEN_BUDGET=0`。
- `V9_L18_WHITEN` **默认 1**（仅 `AR_V9_L18=1` 时生效）：复现旧 L18 口径设 `=0`。
- 两处均已进 `run_manifest` 配置指纹，不同口径不会共用结果文件。
- 预算模式下 **Top-K 不再决定拼块集合**，`hits` / `recall@K` 仅用于诊断。

### 未验证（硬约束）

1. `AR_DELTA_WRITE=0/1` × 三种策略 —— **Delta 到底有没有用，无任何对照**。
2. `AR_V9_PROMPT_DATE=1/0` 配对 —— 日期进 prompt 的收益未测。
3. `AR_V9_L18=0/1`（含 whitening）—— 稠密通道收益未测（只有合成池几何证据）。
4. `CR_DELTA_MODE=off/context/refresh` —— CR 里 Delta 参不参与查询的差别未测。
5. 本地 Judge 真实跑分未做；`evidence_recall_at_k` 在真实数据上仍恒为 None
   （`results/*.jsonl` 18 文件 1800 行中带 `diag` 字段 0 条）。

### 文档修正

- 撤回"端到端 SF 对检索内容敏感、是有效探针"：配对后 1 句 vs 默认 −3.85pp（p=0.625）、
  宽窗 vs 默认 +8.70pp（p=0.500），**均不显著**；n=100 时 MDE=9.7pp。
- 修正数字张冠李戴："默认 21 句 = 0.810" 应为 **0.730**（0.810 属 b0_a20 / b4_a16）；
  b16_a4 的 all_rows 0.750 应为 **0.7483**；交集口径 n=144 → **145**。
- 窗口方向明确写"**未定**"：12 对比较 Holm 后全不显著（最小 Holm p=0.0764），默认 16/4 仅暂留。

## [v9-validate] — 2026-10-21 — 首次真实评测：窗口扫描中止 + 两个口径问题 + 潜层读出立项

**一句话**：10 项修复全部落地并双环境验证；首次跑真实评测（EventQA_65536）暴露出
"指标口径失效"与"服务器环境跑不起来"两个必须先解决的问题；窗口方向实验中止，潜层读出立项。

### 新增文档

| 文件 | 内容 |
|---|---|
| `experiments/eventqa_window_sweep.md` | 窗口扫描设计、指标口径失效的实测证据、四组 + 两个对照的真实数字、中止决策 |
| `experiments/run_eventqa_window_sweep.sh` | 四组顺序扫描脚本（内置 overlay `PYTHONPATH`，每组独立结果文件 + manifest） |
| `tools/summarize_sweep.py` | 扫描结果汇总（window/n/SF/anchor/in_ctx/AR@1/3/10/run_id） |
| `README.md` | 新增"v9 验证轮"条目：修复清单、首次评测数字、口径/环境问题、P4 计划 |
| `ARCHITECTURE.md` | 状态矩阵补 4 行（潜层读出未实现 / 窗口验证不成立 / 证据级召回缺失）、风险补 7–8 条、P4 拆成 A/B/C |
| `test_v9_cr.py` | 新增 `test_subject_index_survives_eviction`：淘汰/覆写后主语索引必须指向活槽 |

### 实测数字（EventQA_65536，单卡 V100，ctx0 各 100 题）

| 窗口 | n=100 SF | n=145（ctx0+ctx1 交集） |
|---|---|---|
| b0_a20 | 0.810 | 0.8276 |
| b4_a16 | 0.810 | 0.8207 |
| b10_a10 | 0.800 | 0.8069 |
| b16_a4（默认） | 0.730 | 0.7517 |

> **数字修正（2026-10-21 收口轮）**：本节原写"n=144（ctx0+1）"与 b16_a4 `0.750`，实为
> **n=145 交集**与 `0.7517`（文件级 all_rows 为 0.7483）；"默认 21 句 0.810"是笔误，默认是 **0.730**。
> 配对检验见 `experiments/results/paired_window_test.json`。

对照（配对口径）：同 52 题 1 句窗口 0.6346 vs 默认 0.6731；同 23 题 ~350 句宽窗 0.7826 vs 默认 0.6957。

### 关键发现（必须披露）

1. **EventQA 金标不在原文中**：`gold in context`（原文 / 归一化）均 False → 选项是生成式改写。
   `answer_in_context` / `anchor_answer_hit` 750 条命中 0/750，**口径失效**，不是检索失败。
2. ~~**端到端 SF 是有效探针但粗**：1 句窗口掉 17.5pp，说明依赖检索~~
   **→ 已撤回（2026-10-21 收口轮）**：17.5pp 来自**未配对**样本（1 句 0.635 vs 默认 0.810，而两者题号不同）。
   配对后同 52 题 1 句 vs 默认仅 **−3.85pp（p=0.625）**，同 23 题宽窗 vs 默认 **+8.70pp（p=0.500）**，
   均不显著。结论改为：**现有数据无法证明 SF 能区分检索好坏**，需换自带金标 span 的合成探针。
3. **窗口方向差异不显著**（配对 McNemar：b0_a20 / b4_a16 vs b16_a4 各 +8.00pp，raw p=0.0386，
   **Holm p=0.2314**；MDE@80% power = 9.7pp，观测 8.0pp < MDE）→ 方向**未定**，默认值暂留，实验中止。
4. **服务器环境原本跑不起来**：transformers 4.51.3 缺 `smollm3`（`import deltamem` 失败）+ 缺
   `nltk/rouge_score/editdistance/tiktoken`；用 overlay `/share/exp4group/pylib_tf57` 解决，跨版本数字不可比。
5. **宿主机负载**：四进程并行时 load 180，每题 5s → 200s，9 小时只推进到第 2 个 context。

### 未做 / 未验证

- 窗口扫描未跑满 5 个 context（中止，见上）。
- `evidence_recall_at_k` 仍无数据源（EventQA 无 span，LongMemEval 未接）。
- **潜层读出（P4-A/B/C）未开工**：`AR_READOUT_MODE=latent` 仍直接 exit 2，所有数字都是明文 prompt 读出。
- 第 18 层稠密通道的真实效果、Delta 写入的真实效果，仍无证据。

本仓库的实验记录按版本追加。指标一律注明"由哪一版代码产出"，避免把重构前数字误当重构后结果。

## [v9-fix] — 2026-10-20 — 设备不一致 / 覆写丢槽 / 结果文件复用（P0）+ 诊断诚实化 + CR 性能与路径读出

**性质：正确性与可复现性修复，不含任何新指标。** 未跑真实评测、未加载模型权重、未初始化 CUDA。
所有结论来自 CPU 单测与 `--dry-run`；EventQA 窗口扫描脚本已就绪但**未执行**（见
`experiments/eventqa_window_sweep.md`，表中数值留空）。

### Fixed — P0（会让结果错误却看起来正常）

- **第 18 层稠密通道设备不一致**（`v9_slotpool.py:545` `hybrid_scores`）：旧版把 `dense_scores` 的结果
  （provider 在 GPU 上算）与 `lexical_scores`（CPU/另一设备）直接相加，触发
  `RuntimeError: Expected all tensors to be on the same device`。该异常被 `run_ar_v7.py` 里
  宽 `except` 吞掉，于是**整题退化成空证据块**，指标下降被误读为"方法变差"。
  改为：`l18` 向量一律常驻 **CPU fp32**（262K 单元 × d 放显存会爆），`dense_scores` 返回 CPU 张量、
  `hybrid_scores` 在相加前把两侧对齐到 `s.device/s.dtype`，并加形状断言。
- **CR 覆写可能找不到正确的旧槽**（`v9_slotpool.py:843` `_merge_target`）：旧版在全池 `mean` 上取
  `argmax` 选候选，再按主语过滤——当别的槽主语向量更接近时，同主语的旧槽**根本没进候选**，
  于是"覆写"变成新增，池里留下过期事实。改为 `_subj_index` 主语倒排表（`_reindex_subject`），
  先 O(1) 取同主语槽，再在其中按关系谓词余弦 + `V9_CR_REL_THETA` 双阈值判定。
- **不同实验配置复用同一个结果文件**（`run_ar_v7.py:224` `init_manifest` + 新增 `run_manifest.py`）：
  旧版输出固定 `ar_{MODE}_{SRC}.jsonl` 且 `AR_RESUME=1` 默认开，`AR_DELTA_WRITE=0/1`、
  `AR_V9_L18=0/1`、不同 EventQA 窗口全部落到同一文件，第二次运行直接跳过已完成题并打印第一次的分数。
  改为：配置指纹（8 位）→ 文件名 `ar_{MODE}_{SRC}_{id}.jsonl`；同目录 `<out>.manifest.json` 记录
  完整参数 + git commit + 环境；续跑前校验，不一致**拒绝恢复**（`AR_RESUME_MANIFEST=0` 可关，不建议）。

### Fixed — P1

- **时间戳跨 Session 继承**（`v9_slotpool.py:135` `split_sessions` / `:182` `extract_units`）：
  违反不变量 I3。现在按 `Session N` / `## Session` / `=== Session` / `Date:` / `YYYY/MM/DD` 切 Session，
  每个 Session 重置 `last_ts`，单元新增 `session_id` / `event_time` / `ingestion_order` 字段。
  无显式标记的上下文仍视为单 Session（行为不变）。
- **CR 覆写是 drop+append**（`v9_slotpool.py:909` `_replace`）：旧版删除旧槽再追加新槽，槽 id 漂移、
  图键与槽下标错位、`cr_E_*` 索引失效。改为原位替换（texts/index_texts/metas/tokv/words/mean/years/df/
  CR 图键/l18 全部就地更新），槽 id 与池序稳定。`v9_cr.py:179` 覆写 `_replace` 同步 CR 图结构与
  主语/宾语键表，并清空 `_subj_mat_cache`。
- **StatePool 建池 O(N²)**：旧版每条事实都对全池 stack 一次再打分。改为主语倒排索引 +
  批量内 pending 冲突合并（`_merge_target` 返回 `("stored"|"pending", idx)`），单条事实只碰同主语槽。
- **第 18 层编码失败静默降级**（`v9_slotpool.py:305` `set_layer18_provider(fn, strict=True)`、
  `:358` `_call_l18`）：runner 注入 provider 时传 `strict=True`，失败/形状不符直接抛错，不再退化成纯词面。
  默认 `strict=None` 时仍按 `V9_L18_STRICT` 决定，保持单测与旧脚本可用。
- **检索异常被宽 except 吞掉**（`run_ar_v7.py:244` `strict_retrieval_enabled`）：`AR_STRICT_RETRIEVAL=auto`
  时，只要稠密通道 / delta 写入 / 潜层读出任一开启，检索异常直接 `raise`，不再产出空证据块。
- **`inputs_embeds` 被当成潜层读出**（`run_ar_v7.py:395` `generate_readout`）：实际是 `EMB[ids]`——
  **同一段明文 prompt** 的查表嵌入，没有注入任何槽位状态。新增 `AR_READOUT_MODE=prompt|text_embeds|latent`，
  `latent` **显式报错退出**（未实现）；每条记录写入 `readout.mode` 与 `readout.latent_slots=false`，
  避免论文里把明文嵌入说成 latent memory。

### Fixed — P2

- **诊断 `Recall@K` 名不副实**（`ar_diag.py`）：它是"gold answer 字符串是否出现在第 K 个槽位"的代理，
  不是证据级召回。规范键改为 `answer_recall_at_k`（旧键 `recall_at_k` 保留为别名以兼容历史 JSONL），
  新增 `evidence_recall_at_k`：没有金标证据 span 时恒为 `None`，**不伪造**。
- **Delta 空快照允许实验继续**（`run_ar_v7.py:357` `prepare_delta_memory`）：写入开启但写后快照为空时，
  逐题恢复退化为 no-op、题目互相污染。改为直接 `RuntimeError`；确需放行用 `AR_ALLOW_EMPTY_DELTA_SNAP=1`。

### Added

- **`run_manifest.py`**：配置指纹、原子写 sidecar、`check_resume`（config_id / 参数 / commit 三级校验）、
  `default_out`（文件名带指纹）、`describe`。纯标准库，不依赖 torch。
- **`run_group_b2_v7.py`** 接入 manifest：`CR_MODE`/`BEAM`/`EGO_K`/`REL_THETA`/`HOP1_M`/`PATH_VIEW`/
  `CF_PROMPT`/`THETA`/`POOL_MAX` 全部进指纹；续跑按 `question_idx` 跳过已完成题；汇总改为**读文件**
  而非只算本轮 `results`（续跑时旧算法会把总分算成本轮小样本）。
- **`v9_cr.hop1_candidates`**（`v9_cr.py:241`）：Hop1 先用主语均值矩阵 `@ q_k` 初筛至多
  `V9_CR_HOP1_M=96` 个候选（`n<=M` 时全量，等价旧行为），再逐 token MaxSim；1 万槽池的 Python
  循环从 1 万次降到 96 次。gold 槽不丢（`test_hop1_matrix_prefilter`）。
- **CR 路径式读出**（`v9_cr.py:337` / `:437` `_render_paths`）：每条完整路径渲染成
  `PATH i: 种子 --关系--> 中间实体 --关系--> 目标`，附实体链与关系分，再列其余已选事实；
  事实仍用**全局编号**，反事实 Prompt 的"编号越大越新"规则不变。`V9_CR_PATH_VIEW=1` 默认开
  （v9 CR 从未跑分，无历史可比性问题），`=0` 回到平铺块。
- **`experiments/eventqa_window_sweep.md` + `run_eventqa_window_sweep.sh`**：Back/Ahead = 0/20、4/16、
  10/10、16/4 四组，记录 `anchor_accuracy`（`anchor_answer_hit`）与 `answer_in_context`。
- **`tools/summarize_sweep.py`**：读若干 jsonl，输出 `window/n/SF/anchor/in_ctx/AR@1/3/10/run_id` 表。
- **`test_run_manifest.py`**（27 断言）：指纹随消融开关变化、文件隔离、续跑三级校验、
  `latent` 读出被拒、delta 空快照报错。

### Changed

- `run_ar_v7.py` 记录新增 `run_id` / `readout` / `window` / `anchor_answer_hit` / `anchor_window_hit`；
  `AR_INPUTS_EMBEDS=1` 仍可用，等价于 `AR_READOUT_MODE=text_embeds`。
- `ARCHITECTURE.md` 重写为**三层结构**（输入事件流 / State Memory / Episodic Memory / Evidence Router /
  Latent Readout），新增不变量 I5（结果是配置的函数），并明确"读出层仍是明文"这一事实边界。

### Tests

| 套件 | 断言 | 状态 |
|---|---|---|
| `test_v9_slotpool.py` | 129 | 全绿（含同主语覆写 P0 回归、设备对齐、Session 边界、strict 模式） |
| `test_v9_cr.py` | 80 | 全绿（含原位覆写图键一致、Hop1 初筛、路径读出/平铺回退） |
| `test_ar_v9_wiring.py` | 36 | 全绿 |
| `test_ar_delta_alignment.py` | 27 | 全绿 |
| `test_run_manifest.py` | 27 | 全绿 |
| `test_v8_slotpool.py` | 57 | 全绿 |
| `ar_diag.py --selftest` | 57 | 全绿 |

### 环境阻塞（2026-10-20 实测，非本仓库问题）

- `latent2` 的 transformers 4.51.3 **缺 `models/smollm3`**，而 `/share/delta-Mem/deltamem/core/
  backbone_compat.py:3` 无条件 import 它 → `import deltamem` 直接 `ImportError`，
  **任何评测都跑不起来**（`run_ar_v7.py` 会打印"缺少真实跑分依赖"）。
- 同环境还缺 `nltk` / `rouge_score` / `editdistance` / `tiktoken`（`post_process` 依赖）。
- 处理：不改共享 conda 环境，装 overlay 目录 `/share/exp4group/pylib_tf57`
  （transformers 4.57.1 + tokenizers 0.22.2 + 上述四个包），运行时
  `PYTHONPATH=/share/exp4group/pylib_tf57:/share/delta-Mem:/share/MemoryAgentBench`。
  已实测 `import deltamem` + 模型加载 + 3 题冒烟通过（SF=1.0）。
- 副作用：transformers 版本与历史 v7 结果不同，**跨版本数字不可直接比较**；
  窗口扫描四组同栈，内部可比。

### 未做 / 未验证（不得在论文中声称）

1. 真实 GPU 跑分：v9 AR、CR v9、EventQA 窗口扫描四组**均未执行**。
2. 第 18 层稠密通道的**效果**（只有设备与失败路径被验证）。
3. Delta 状态 + 快照恢复在真实模型上的行为（仅单测 stub）。
4. 槽位潜向量表征（路径 B）——**无实现**，`AR_READOUT_MODE=latent` 直接报错。

## [docs] — 2026-10-20 — 新增 `ARCHITECTURE.md`（预期架构）+ 环境路径订正

### Added

- **`ARCHITECTURE.md`**：描述当前**预期架构**（DM-ELSM 双模态弹性潜槽位记忆），而非完成清单。
  含 4 条设计不变量（全池不截断 / 先分数后位置 / 时间是数据 / 读出走潜层）、数据流图、组件契约表、
  两条合法潜层读出路径（A: delta 状态 + 快照恢复；B: `inputs_embeds` + 槽位潜向量）、参数表、
  实现状态矩阵（`已实现` / `已接线·未验证` / `规划中`）、5 项已知风险与 P0–P5 演进路线。
- README 顶部与"文件"章节加入指向该文档的入口。

### Changed

- README「运行」章节补充**环境订正**：`/share/conda/envs/latent2/bin/python`（Python 3.12.4 +
  torch 2.8.0+cu126 + transformers 4.51.3，2026-10-20 重建）；明确**不得再带 `t28` / `pylib5`
  叠加层**（新环境下会使 `import torch` 失败、dry-run 退化为 `cuda_initialized=n/a`）。
- 「v9 真实跑分」示例命令由 `/opt/conda/bin/python` 改为重建后的解释器路径。

## [v9] — 2026-10-20 — 确定性 Bug 修复 + 逐题诊断 + AR 检索重写 + CR 强化 + Delta 对齐

**性质：代码 / 可诊断性重构，不含任何新指标。** 本轮按硬约束只做 dry-run（`py_compile` / CPU 单测 / `--dry-run`），
未跑真实评测、未加载模型权重、未初始化 CUDA、未占用 GPU。

### Fixed — v8 三个确定性 Bug（`v8_slotpool.py`，接口零变更）

- **Bug1 `cur_ts` 第二轮污染**（`v8_slotpool.py:79-138`）：旧版首轮扫描结束后 `cur_ts` 停在全篇最后一个时间戳，
  第二轮窗口/状态判定直接复用它（状态轮还继续改写它），使位于任何时间标记之前的单元被写入"文档末尾"时间戳。
  改为首轮一次性预计算逐句 `resolved_ts[i]`（= 该位置之前最近一次显式时间戳），窗口与状态两轮只读；
  窗口无显式标记时继承 `resolved_ts[i-1]`。合成 150 句上下文：旧版 31 个情景单元被写入末尾时间戳，新版仅 2 个
  （窗口内确实含该时间戳句，属正常显式标记）。
- **Bug2 512 词 span 偏移**（`v8_slotpool.py:159,222-236,401`）：旧版 `spans` 用未截断的 `len(ws)` 记账、却只
  `extend(ws[:512])`，长单元 span 越过自身边界吃进后续单元的词；零 token 兜底分支还 `off += 1`，使后续单元
  `tokv` 整体错位。改为**先截断再记账**（`UNIT_MAX_WORDS=512`，env `V8_UNIT_MAX_WORDS`），零 token 分支不推进 `off`；
  `StatePool._mean_only` 复用同一常量。
- **Bug3 `ev` 时间触发**（`v8_slotpool.py:123,131-138`）：旧版 `ev` 只看事件动词，纯时间句（含 `In 2001` 但无过去时动词）
  落入 state，与"时间标记或事件动词 → episodic"的设计矛盾。改为 `ev` = 事件动词 **or** 窗口内显式时间标记；
  state 只收无显式时间标记且无事件动词的稳定陈述句。仅"继承到时间戳"、窗口内无显式标记且无事件动词的窗口仍归 state。
- 接口：episodic unit 字段不变；state unit 只新增附加字段 `ts_inherited`；`run_ar_v7.py` 无需改动。
- 回归：`test_v8_slotpool.py` 从 9 组扩到 16 组（+7 个 Bug 回归用例）。有效性已验证——把 git HEAD 的旧版
  `v8_slotpool.py` 放到临时目录跑新测试，bug1/bug2/bug3 用例均按预期失败，新版全绿。

### Added

- **`v9_slotpool.py`**（777 行）— v9 AR 检索：
  - 非重叠索引 `extract_units`（`:124`）：`stride == block_sents`，`V9_BLOCK_SENTS=1` 即单句索引，每句恰好属于一个单元；
    修掉 v8 3 句滑窗 stride=1 的 3 倍池膨胀、Top-K 只覆盖 K/3 句、位置排序有歧义三个问题。
  - 分数优先：`retrieve`（`:466`）只按分数降序返回；`retrieve_ordered`（`:477`）/`readout`（`:550`）只对**已选中的 K 个**
    按 `start` 排序，绝不做"全池按位置排序后截断"。
  - EventQA 后向非对称扩展 `expand_backward`（`:515`）：默认 `back=16`、`ahead=4`（`V9_EQA_BACK` / `V9_EQA_AHEAD`），
    含边界裁剪；`anchor_events` 走 last/last2 重锚。
  - 时间戳参与打分与重排：单元编码文本 `In <ts>: <原文>`（`build_index_text`，`:117`），日期 token 进入 IDF-MaxSim；
    `texts` 仍保存原文；`temporal_rerank`（`:444`）按 `|year_u − year_q|` 指数衰减加成
    （`V9_TEMPORAL_TAU=2.0` / `V9_TEMPORAL_BONUS=0.15`），无年份单元不加分不罚分。
  - 第 18 层混合稠密打分：`set_layer18_provider`（`:223`）/`mean_pool_hidden`/`attach_layer18`/`dense_scores`/
    `hybrid_scores`（`:433`），`hybrid=(1−α)·IDF-MaxSim + α·((cos+1)/2)`，`V9_L18_ALPHA=0.3`、`V9_L18_LAYER=18`；
    provider 缺失/抛错/形状不符一律静默退化为纯词面。
  - `StatePool`（`:647`）只作 AR 可选补充与 CR v9 扩展点，`_should_merge` 可覆写。
- **`v9_cr.py`**（359 行）— CR 强化：主语+关系双阈值覆写（`V9_CR_REL_THETA=0.80`）、Hop1 `Beam=6` + ego-graph
  （`V9_CR_EGO_K=12`，同主语属性 + 出边实体）、双跳链联合打分
  `0.40·subj_cov + 0.30·(bridge1+bridge2)/2 + 0.30·(rel_mid+rel_tgt)/2`、反事实强约束 Prompt（`build_cr_prompt`，`:61`）。
- **`ar_diag.py`**（823 行）— 逐题诊断：`record()`（`:306`）产出 `retrieved_slot_ids / retrieved_scores / retrieved_texts /
  retrieved_starts / used_slot_ids / answer_in_retrieved / answer_in_context / answer_hit_slot / answer_matches /
  recall_at_k / diag_status`；`hits` 必须分数降序（Recall@K 依赖）。口径：NFKC + 小写 + 撇号删除 + 连字符折叠后子串匹配；
  rubric（偏好类）gold 标 `rubric_not_applicable` 并剔除分母；只有 `idxs` 的 legacy 记录标 `legacy_idxs_only`。
  CLI：`--input / --summary-out / --enrich / --selftest`。
- **`eval_longmemeval_judge.py`**（836 行）— LongMemEval 官方 `evaluate_qa` 口径的 LLM-as-Judge 后处理：6 题型模板
  （通用 / temporal off-by-one / knowledge-update 旧值容忍 / preference rubric）+ abstention 模板；
  `--prompt-source official` 直接 import 官方 `get_anscheck_prompt`；OpenAI 兼容 `/chat/completions`
  （`temperature=0, max_tokens=10`）、线程池并发、重试退避、sha1 缓存断点续跑；verdict 复刻官方 `'yes' in text` 语义，
  另存 `verdict_strict` 审计；abstention 单列不进主分母；`--dry-run` 零网络、`--selftest` 用 monkeypatch 哨兵保证零网络。
- 新单测：`test_v9_slotpool.py`（95 断言）、`test_v9_cr.py`（56）、`test_ar_v9_wiring.py`（36）、
  `test_ar_delta_alignment.py`（27）；`test_v8_slotpool.py` 扩到 56 断言。

### Changed

- **`run_ar_v7.py`**（976 行）：
  - 新增 `AR_MODE=v9`（惰性 `v9_import`，`build_pool_for_context_v9` / `build_block_v9`，`:589` / `:620`）。
  - 逐题诊断字段：v7/v8/v9/plain 四种模式都写；诊断异常只写 `diag_status=diag_error`，不影响主指标。
  - 修正"先排序后截断"：新增 `_cap_before_sort()`（`:490`），先按分数优先裁到情景配额（EventQA 以 anchor 为中心裁窗）
    再按原文位置排序；旧写法会丢高分尾部单元（单测构造 11 个候选、旧实现丢 8 个）。
  - 新增 `--dry-run`（`dry_run()`，`:740`）：不加载权重、不读 parquet、不初始化 CUDA；打印配置与依赖可导入性，
    跑两项纯 CPU 自检（v9 `extract_units` 时间戳继承、`ar_diag.record` 假池命中），末行打印
    `cuda_initialized=False model_loaded=False`；非法 `AR_MODE` / source 返回退出码 2。
    `transformers/deltamem/pyarrow` 改为 try 导入，缺依赖时真实跑分显式报错，dry-run 不受影响。
  - Delta Memory 对齐：写入移到**查询前**（`prepare_delta_memory`，`:255`），每个 context 只 `reset_delta_mem_states` 一次，
    题目循环内不再 reset；每题生成前 `delta_restore(snap)`（`:239`，优先官方 `load_delta_mem_online_state`，
    缺 API 时退回 `iter_delta_mem_modules` 手工 clone/restore，两套都缺则 `[warn]` 退化为 no-op）；
    新增 `AR_INPUTS_EMBEDS=1` 的 `inputs_embeds` 读出路径（`generate_readout`，`:288`），OOM 渐进截短两条路径都生效；
    每题 JSONL 新增 `delta: {write, restore, inputs_embeds, snapshot_layers}`。
- **`run_group_b2_v7.py`**：新增 `CR_MODE=v7|v9`（默认 v7，v7 分支与历史完全一致）；v9 分支注入 `v9_cr`、
  可选 CF Prompt、每题追加 `cr_dbg`、OUT 改 `v9groupB2_*.jsonl`、FINAL 打印 `CR_MODE`；非法值直接退出。

### Docs

- `README.md`：新增 v9 章节（改动清单 + dry-run 结论 + 未验证项声明 + v9 env 表），v8 章节标注"指标为修复前数字"，
  补 `--dry-run` 与 v9 运行命令，更新文件清单。
- 本 `CHANGELOG.md`。

### Verification（dry-run，本机 Python 3.9.6 + torch 2.8.0 CPU，`device_count=0`）

| 检查 | 结果 |
|---|---|
| `python3 -m py_compile *.py` | 21/21 通过 |
| CPU 单测（5 文件） | 95 + 56 + 56 + 36 + 27 = 270 断言全部 PASS |
| `ar_diag.py --selftest` | 43/43 |
| `eval_longmemeval_judge.py --selftest` | 45/45（零网络） |

> **当前数字（v9 收口轮）**：`py_compile` **28/28**；CPU 单测 **9 个文件** —— `test_v9_slotpool` 284 /
> `test_v9_cr` 143 / `test_v8_slotpool` 86 / `test_ar_v9_wiring` 105 / `test_ar_delta_alignment` 37 /
> `test_run_manifest` 27 / `test_ar_delta_write_strategy` 47 / `test_cr_delta_log` 84 / `test_v6_parse`；
> `ar_diag --selftest` **121/121**；`eval_longmemeval_judge --selftest` **76/76**。上表是 v9 验证轮的历史快照。
| `run_ar_v7.py --dry-run`（v7/v8/v9/plain） | 四种均退出码 0，末行 `cuda_initialized=False model_loaded=False` |
| `eval_longmemeval_judge.py --dry-run` | 合成 3 题渲染 3 条 prompt，零网络请求 |
| 守卫断言 | 零网络、零 `transformers/pyarrow/deltamem` 导入、无文件写入、无残留进程 |

### Not verified（硬约束，非"已通过"）

1. **没有任何 v9 指标**：真实 tokenizer/模型下的 v9 建池与生成、五项检索改动对 SF / Recall 的影响均未测量。
2. **v8 历史指标（0.7100 / 0.5120 / 0.3820 / 0.3167）由修复前代码产出**，修复后未重跑。
3. `AR_V9_L18=1` 的 provider 前向未跑（仅静态审查，默认关闭）。
4. Delta Memory：`/share/delta-Mem` 实际版本是否含 `get/load_delta_mem_online_state` 未确认（已做 fallback）；
   真实 Qwen3 `generate(inputs_embeds=...)` 行为、写入/恢复开销与指标影响未测。
5. `run_group_b2_v7.py` 未运行（模块级 `from_pretrained` + `/share` 路径、无 `--dry-run`）；CR v9 只由单测覆盖。
6. `eval_longmemeval_judge.py` 未做真实判题（未配 API key、未联网）。
7. `ar_diag.py` 未跑真实 AR 结果，只在 legacy CR JSONL 上验证不崩（全部 `legacy_idxs_only`，applicable=0）。

## [v8] — 2026-10-19 — 双模态潜槽位记忆 DM-ELSM（AR）

- 新增 `v8_slotpool.py`：`extract_adaptive_units` 情景/状态自适应分流（3 句滑窗微块 + 时间戳继承）、
  EpisodicFastPool 全池无截断 IDF-MaxSim、邻域扩展；`run_ar_v7.py AR_MODE=v8`。
- AR 四切片：eventqa_65536 0.7100 / 131072 0.5120 / full 0.3820 / longmemeval_s* 0.3167
  （v7 基线 0.7240 / 0.5620 / 0.3500 / 0.2767）。**注：这三个 Bug 已于 v9 修复，上表未重跑。**
- 长上下文 OOM 修复：`encode_words` 分批编码 + 逐切片 `.cpu()`；generate 前 `empty_cache` + 渐进截短重试。

## [v7] — 2026-10-18 — 自适应弹性槽位池打通 6K→262K

- 新增 `v7_slotpool.py`：无损弹性容量（2048→…，`B2_POOL_MAX=0`）、二级路由 O(1) 覆写、主语特异性长度加权、
  双向桥接覆盖率、WHERE 倒装解析；CR 8 切片 SF：sh 0.94/0.91/0.89/0.85，mh 0.67/0.63/0.59/0.49。
- AR 四切片：eventqa_65536 0.7240 / 131072 0.5620 / full 0.3500 / longmemeval_s* 0.2767。

## [v6] — 2026-10-18 — 一套代码通吃 sh + mh

- `v6_slotpool.py`：VP_RE 谓词补全 + 长谓词优先、WHERE 倒装从句解析、(subj, rel) gid 精确覆写。
- CR：sh_6k 0.940 / sh_64k 0.890 / mh_6k 0.670 / mh_64k 0.590。

## [v5] — 2026-10-17 — 三元组解耦潜键 + Beam Chaining

- `v5_slotpool.py`：`parse_fact_triple` 三元组解耦、双重命中覆写、主语门控多跳检索（Beam=2）。
- CR mh：6k 0.510 / 64k 0.450。

## [v4] — 2026-10-17 — 大池 + 主谓组去重 + Top-15 检索读出

- 修 4 个瓶颈（POOL_MAX=512 盲淘、全量拼块稀释注意力、KEY_TAIL_CUT 漏覆写、读出前未 reset）。
- CR sh：6k 0.860 / 64k 0.870。

## [v3] 及更早

见 git 历史与 `versions/` 目录。初版（v1）因 5 个致命 Bug（事实切分漏召、模板 KV 污染、decode 步进、跨 chunk RoPE 拼接、
key 切分失效）被判 No-Go，v2 修复后 sh_6k 0.110→0.850。
