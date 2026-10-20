# EventQA 检索方向验证：窗口扫描

## 为什么做这个实验

v7/v8 的 EventQA 读出是**只向后取**：锚点单元之后 `EQA_WIN=4` 块。于是"答案在锚点之前"
的题目必然检索失败，而失败被记成"方法不行"。v9 改成**非对称后向扩展**
`expand_backward(anchor, back=16, ahead=4)`，方向假设是"事件问句的锚点通常落在答案之后"。

这个假设必须验证，不能只看总分：总分同时受锚点定位、窗口方向、生成质量影响。
所以拆成两个可直接观测的量：

| 指标 | 字段 | 含义 |
|---|---|---|
| `anchor_accuracy` | `anchor_answer_hit` | **锚点单元本身**是否含答案（定位对不对） |
| `answer_in_context` | `answer_in_context` | **最终证据窗口**是否覆盖答案（方向+宽度够不够） |

二者一起看才能区分两类失败：

- `anchor_answer_hit` 低 → 锚点定位错了，调窗口没用，要查 `anchor_from_events` / `best_unit_for`。
- `anchor_answer_hit` 高、`answer_in_context` 低 → 定位对但窗口方向/宽度不够，正是本实验要测的。
- 两者都高、SF 低 → 检索没问题，问题在生成或判分。

## 扫描配置

| 组 | Back | Ahead | 检验的假设 |
|---|---|---|---|
| 1 | 0 | 20 | 纯前向（锚点之后取 20 句）——v7/v8 方向的极端版 |
| 2 | 4 | 16 | 弱后向 |
| 3 | 10 | 10 | 对称 |
| 4 | 16 | 4 | **当前默认**（强后向） |

四组窗口总宽都是 20 句，只改方向，保证可比。

## 运行

```bash
bash experiments/run_eventqa_window_sweep.sh
# 或手动单组：
AR_MODE=v9 AR_V9_EQA_BACK=4 AR_V9_EQA_AHEAD=16 \
  AR_OUT=/share/exp4group/results/sweep/ar_v9_eventqa_65536_b4_a16.jsonl \
  python run_ar_v7.py eventqa_65536 0
```

每组结果文件旁边会自动生成 `<out>.manifest.json`（配置指纹 + 完整参数 + git commit）。
**换配置必须换文件**：若把两个窗口写进同一个 `AR_OUT`，第二次启动会被 manifest 校验拒绝
（`[resume][ERR] 拒绝续跑`），这正是为了不让两组消融混在一起。

## 汇总

```bash
python3 tools/summarize_sweep.py /share/exp4group/results/sweep/ar_v9_eventqa_65536_b*.jsonl
```

输出列：`window / n / SF / anchor / in_ctx / AR@1 / AR@3 / AR@10 / run_id`。

## 记录（待填）

| Back/Ahead | n | SF | anchor_accuracy | answer_in_context | AR@1 | AR@3 | AR@10 | run_id |
|---|---|---|---|---|---|---|---|---|
| 0/20 | | | | | | | | |
| 4/16 | | | | | | | | |
| 10/10 | | | | | | | | |
| 16/4 | | | | | | | | |

> 说明：本表数值必须来自真实运行。CPU 单测只能验证"扩展逻辑正确"，
> 不能代替这四个数字。未跑之前一律留空，不填估计值。

## 判读规则（跑完后照此结论）

1. 若 `16/4` 的 `answer_in_context` 明显高于 `0/20` → 后向假设成立，默认值保持。
2. 若 `10/10` 或 `4/16` 更高 → 后向幅度过大，把默认 `V9_EQA_BACK` 调小。
3. 若四组 `anchor_accuracy` 都低（< 0.5）→ 问题在锚点定位而非窗口，先修
   `anchor_from_events`（`cov_threshold=0.5`、`last/last2` 重锚）再谈方向。
4. 若 `anchor_accuracy` 高但 `answer_in_context` 四组都低 → 窗口总宽 20 不够，
   下一轮扫宽度（如 8/8、16/16、32/32）而不是方向。
