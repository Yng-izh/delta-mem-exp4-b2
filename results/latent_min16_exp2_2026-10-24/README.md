# 潜层读出实验 1+2：反事实配对可学习性 → 多事实干扰（2026-10-24/25，v100-2）

**结论先说**：实验 1（16 条反事实配对过拟合）**通过**——15/16、配对 7/8；
实验 2（4 槽多事实，200 训 + 64 验证）**未通过**——训练集读出 0/200、验证 17.2%、配对 0/32。
定位链把瓶颈钉死在**压缩器无法编码多条事实的组合**（容量/对位结构问题，不是泛化、不是训练量、
不是干扰摊薄注意力）。checkpoint 未入库（30 MB/个），配置在各日志 `[save]` 行。

## 实验对照总表（全部实体级/严格双口径，同一批判据）

| 实验 | 设定 | 训练集 | 验证集 | pair-acc（验证） | 结论 |
|---|---|---|---|---|---|
| 通路确认 | 明文 16 条 | — | 16/16 | 8/8 | 任务成立 |
| 通路确认 | Oracle（原文 embedding） | — | 16/16 | 8/8 | 注入通路成立 |
| **实验 1** | 1 槽 × m=4，纯 QA | （评=训练集） | 1/16 | 0/8 | 失败 |
| 消融 A | **跳过压缩器**（原文 embedding→Bridge） | 16/16 实体级 | — | 8/8 实体级 | **Bridge 无瓶颈** |
| **实验 1+recon** | 1 槽 + `--recon-weight 1.0` | — | 6/16（实体 13/16） | 2/8 | 重建目标救回实体信息 |
| **实验 1 v2** | 1 槽 + **EOS 监督 + 两阶段**（recon-only 300 步） | — | **15/16** | **7/8** | **通过（成功线 14/16、7 组）** |
| **实验 2** | 4 槽（1 目标+3 干扰），recon 有 str(list) bug | 读出 0/200 | 9.4% | 0/32 | 失败（目标畸形） |
| **实验 2 v2** | 同上，**recon 修复**，3000 步 | **读出 0/200** | 17.2% | 0/32 | **失败（结构性）** |

## 定位链（实验 2 失败原因逐环排除）

| 假说 | 检验 | 结果 |
|---|---|---|
| 任务/数据不成立 | 明文 4 槽读出 | ✗ 排除（16/16） |
| 干扰摊薄注意力 / Reader 读不出 | Oracle 4 槽（含 3 干扰） | ✗ 排除（16/16） |
| Bridge 映射不行 | 消融 A | ✗ 排除（16/16） |
| 训练量不够 / 欠训练 | 实验学成后**训练集**读出 | ✗ 排除（训练集 0/200，与验证集无差别） |
| recon 目标畸形 | 修 str(list) bug 重训 3000 步 | ✗ 排除（训练集仍 0/200） |
| **压缩器无法编码多条事实组合** | 1 槽 ✓ vs 4 槽 ✗，其他全同 | **✓ 唯一存活** |

机制推断：4 事实共用 16 软 token（m=4 个 query 服务 4 条事实），query 与事实之间**没有对位
关系**，值信息在多条事实混叠时互相干扰——实验 1 单事实下偶发的值绑定混淆（Taron 组），
到 4 槽被放大成系统性失败。

## 本轮代码改动（`train_latent_memory.py`）

1. **EOS 监督**：答案 token 末尾补 eos 进 labels。没有它模型从没被教过"答完就停"，
   失败模式是实体对但拖尾巴（`Velcor`→`Velcora`、`Kestmor`→`Kestmorville`）。
2. **两阶段训练**（`--recon-only-steps N`）：前 N 步只训重建，之后切 QA+recon。
   同时开两个目标会在同一组软 token 上争预算，评估曲线非单调震荡
   （0.0625→0.125→0.0625）；改两阶段后单调爬升至 15/16。
3. **recon 目标 bug 修复**：`item["memory"]` 是列表，`str(list)` 得到
   `"['Taron works at Velcor.']"` 这样的 Python repr——实验 2 首轮全程在重建 repr 字符串。
   改为 `" ".join(item_units(item))`。**教训：给 `build_student_inputs` 传 answer 前先确认类型。**
4. **评估口径**：子串匹配 → 规范化完整匹配 + 逐条打印 + 反事实配对准确率
   （同问题不同记忆值、整组全对才算；纯背 Q→A 做不到）。
5. **梯度范数**：日志里 compressor/bridge 打实际 L2 范数（`.grad is not None` 不足以判断
   梯度是否可用）。
6. **`--bridge-only`**：跳过压缩器、原文 token embedding 直接进 Bridge（消融 A 的实现）。
7. 新数据生成器：`tools/gen_latent_min16.py`（16 条/8 组反事实配对）、
   `tools/gen_latent_exp2.py`（200 训 + 64 验，实体不重叠，每条 4 槽共享干扰组）。

## 日志对照

| 文件 | 对应实验 |
|---|---|
| `exp1_min16_qa.log` | 实验 1：纯 QA，1/16 |
| `exp1_bridgeonly.log` | 消融 A：Bridge-only，loss→0，12/16（实体级 16/16） |
| `exp1_recon.log` | +recon 1.0：6/16，实体级 13/16，配对 2/8 |
| `exp1_eos_twostage.log` | **v2：EOS+两阶段，15/16，配对 7/8（通过）** |
| `exp2_4slot_reconbug.log` | 实验 2：4 槽，recon bug 未修，1500 步，训练集读出 0/200 |
| `exp2_4slot_reconfixed.log` | 实验 2 v2：recon 修复，3000 步，训练集读出仍 0/200 |

## 评估方法说明（为什么信这些数字）

- **训练集读出是关键判据**：只看验证集无法区分"欠训练"和"不泛化"（实验 2 首轮就栽在
  这里——验证集 0 曾被误读为泛化失败，训练集评估后证明是两者皆非的"未学成"）。
- Oracle 与明文是**上界对照**：任何潜层失败都必须先在这两条上拿到满分才有解读资格。
- 严格口径判错的预测若**包含正确实体**，单独记实体级（`Velcora` 与 `Ondrey's Library`
  是生成边界问题，与 `Perthia`（答成同组另一值）性质不同，后者才是信息级失败）。

## 复现

```bash
export HF_HOME=/share/hf_home HF_HUB_OFFLINE=1 DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=/share/exp4group/shim19:/share/exp4group:/share/delta-Mem   # env 的
# transformers 仍是指向不存在目录的 editable，必须用 vendored 5.19.0
P=/share/conda/envs/latent2/bin/python

$P tools/gen_latent_min16.py && $P tools/gen_latent_exp2.py

# 实验 1 v2（通过的那次）
CUDA_VISIBLE_DEVICES=7 $P train_latent_memory.py --stage a \
  --data synth/latent_min16.jsonl --val-data synth/latent_min16.jsonl \
  --m 4 --lr 5e-4 --accum 4 --epochs 200 --max-steps 800 \
  --recon-weight 1.0 --recon-only-steps 300 \
  --eval-limit 16 --eval-every 200 --save-every 200 --out ckpt/latent_min16_v2.pt

# 实验 2 v2（4 槽，recon 修复版）
CUDA_VISIBLE_DEVICES=7 $P train_latent_memory.py --stage a \
  --data synth/latent_exp2_train.jsonl --val-data synth/latent_exp2_val.jsonl \
  --m 4 --lr 3e-4 --accum 4 --epochs 120 --max-steps 3000 \
  --recon-weight 1.0 --recon-only-steps 400 \
  --eval-limit 64 --eval-every 500 --save-every 200 --out ckpt/latent_exp2_v2.pt

# 定位脚本：明文/Oracle/4槽/1槽 四条件对照
$P diag_exp2_locate.py
```

## 下一步（已定位到结构，二选一）

1. **m 4→8 + read budget 32→64**（每事实软 token 翻倍，query 数同步增）：改动最小，
   直接检验容量/对位假说；
2. **per-fact query 对位结构**：每个 query 固定负责一条事实——更根本，改动中等。

按"先定位再改架构"的原则，本轮定位已完成，下一轮应执行上述之一。
