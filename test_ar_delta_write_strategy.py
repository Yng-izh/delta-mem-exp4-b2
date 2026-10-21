# test_ar_delta_write_strategy.py — Delta 有效写入策略 CPU 单测（t4）
#
# 不加载模型、不初始化 CUDA、不读 parquet：stub tokenizer / fake model / stub 分块器。
# 覆盖：
#   1. all（对照基线）= 旧行为：全部分块盲写，skip=0
#   2. filtered：只写状态/事实类单元 —— 情景叙述（显式时间/事件动词）跳过、
#      重复事实去重、同主语+同属性只保留最终值；写入条数 < all
#   3. topk_refresh：在 filtered 之上按问题词面 IDF 只刷新 Top-K
#   4. prepare_delta_memory 真实写入路径：forward 次数 == write_chunks，
#      统计进 DELTA_STATS 与每题 record 的 delta 字段
#   5. 过滤后为空 -> 报错（禁止空写入 + 空快照的静默退化）
#   6. 三档策略在配置指纹上互相区分（不同 config_id，不共用结果文件）
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_ar_v7 as R
import run_manifest


class _Batch(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


class FakeTok:
    eos_token_id = 0

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
        return "PROMPT:" + msgs[-1]["content"]

    def __call__(self, text, return_tensors=None):
        return _Batch(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))

    def decode(self, ids, skip_special_tokens=True):
        return "answer"


class FakeModel:
    def __init__(self, calls=None):
        self.calls = calls if calls is not None else []

    def __call__(self, **kw):
        self.calls.append("forward")
        return None

    def named_modules(self):
        return [("", self)]


def ok(name, cond, detail=""):
    assert cond, f"FAIL: {name}{(' | ' + detail) if detail else ''}"
    print(f"  ok - {name}")


# 合成上下文：含【重复事实】【同主语+同属性的覆写】【纯背景叙述】【不同属性共存】
DEMO = (
    "Session 1\n2021-03-05\n"
    "Alice lives in Paris.\n"          # 后续被覆写 -> 不应写入
    "Alice lives in Paris.\n"          # 完全重复 -> 去重
    "In 2021 Alice moved to Lyon.\n"   # 显式时间 + 事件动词 -> 情景，跳过
    "Alice is a doctor.\n"             # 状态事实，保留
    "Alice is vegetarian.\n"           # 不同属性（表语不同）-> 必须与 doctor 共存
    "She worked there for years.\n"    # 事件动词 -> 情景，跳过
    "Session 2\n2022-01-09\n"
    "Alice lives in London.\n"         # 同主语+同属性的最终值 -> 保留 London
)
# 让 all 与 filtered 在同一粒度（句级）上比较：只让"过滤"这一个变量变化
DEMO_SENTS = [s.strip() for s in R.re.split(r"(?<=[.!?])\s+|\n+", DEMO) if s.strip()]


def setup():
    R.DEV = "cpu"
    R.MAX_NEW_TOKENS = 3
    R.sys_msg = ""
    R.mem_tpl = "{context}"
    R.tok = FakeTok()
    R.EMB = torch.randn(10000, 8)
    R.INPUTS_EMBEDS = False
    R.READOUT_MODE = "prompt"
    R.DELTA_RESTORE = True
    R.DELTA_WRITE = False
    R.DELTA_WRITE_STRATEGY = "all"
    R.model = FakeModel()
    R.DELTA_STATS.clear()


# ---------------- 1. all = 对照基线（旧行为） ----------------
def test_all_strategy_is_old_behaviour():
    setup()
    R.chunk_text_into_sentences = lambda ctx, chunk_size: ["c1", "c2", "c3"]
    texts, st = R.plan_delta_writes("ctx", strategy="all")
    ok("all: 全部分块盲写", texts == ["c1", "c2", "c3"])
    ok("all: 不跳过任何分块", st["skipped_chunks"] == 0 and st["write_chunks"] == 3)
    ok("all: unit_source 标记为 chunk512", st["unit_source"] == "chunk512")
    ok("all: 走 chunk_text_into_sentences（chunk_size=512 默认）",
       st["chunk_size"] == 512)


# ---------------- 2. filtered：过滤 + 去重 + 覆写合并 ----------------
def test_filtered_fewer_and_final_value():
    setup()
    all_texts, all_st = R.plan_delta_writes(DEMO, strategy="all", all_chunks=DEMO_SENTS)
    texts, st = R.plan_delta_writes(DEMO, strategy="filtered")

    ok("filtered: 写入条数 < all（同一句级粒度）", len(texts) < len(all_texts),
       f"filtered={len(texts)} all={len(all_texts)}")
    ok("filtered: 写入条数与 stats 一致", st["write_chunks"] == len(texts))
    ok("filtered: skipped 与 reasons 自洽",
       st["skipped_chunks"] == sum(st["skip_reasons"].values()))
    ok("filtered: candidate_units 覆盖全部单元", st["candidate_units"] == len(all_texts))

    joined = " || ".join(texts)
    ok("filtered: 保留最终值 London", any("London" in t for t in texts), joined)
    ok("filtered: 被覆写的旧值 Paris 不写入", not any("Paris" in t for t in texts), joined)
    ok("filtered: 重复事实只写一次",
       sum(1 for t in texts if "lives in london" in t.lower()) == 1, joined)
    ok("filtered: 情景叙述（显式时间+事件）跳过",
       not any("moved to Lyon" in t for t in texts), joined)
    ok("filtered: 事件句跳过", not any("worked there" in t for t in texts), joined)
    ok("filtered: 不同属性共存（doctor + vegetarian）",
       any("doctor" in t for t in texts) and any("vegetarian" in t for t in texts), joined)
    ok("filtered: skip_reasons 分类正确（episodic/duplicate/overwritten 均 >0）",
       st["skip_reasons"]["episodic"] >= 2 and st["skip_reasons"]["duplicate"] >= 1
       and st["skip_reasons"]["overwritten"] >= 1, str(st["skip_reasons"]))
    ok("filtered: 覆写是原位替换（London 占据被覆写槽位的位置，仍排在 doctor 之前）",
       "London" in texts[0] and "doctor" in texts[1], str(texts))
    ok("filtered: 未覆写单元按原文顺序",
       texts.index([t for t in texts if "doctor" in t][0])
       < texts.index([t for t in texts if "vegetarian" in t][0]), str(texts))


def test_slot_key_separates_attributes():
    """同主语不同属性不得落进同一个槽（'is a doctor' vs 'is vegetarian'）。"""
    k1 = R._delta_slot_key("Alice is a doctor.")
    k2 = R._delta_slot_key("Alice is vegetarian.")
    ok("slot_key: doctor != vegetarian", k1 != k2, f"{k1} vs {k2}")
    k3 = R._delta_slot_key("Alice lives in Paris.")
    k4 = R._delta_slot_key("Alice lives in London.")
    ok("slot_key: 同主语+同属性（lives in）同槽", k3 == k4, f"{k3} vs {k4}")
    k5 = R._delta_slot_key("Alice works at Google.")
    k6 = R._delta_slot_key("Alice works on Project X.")
    ok("slot_key: works at != works on", k5 != k6, f"{k5} vs {k6}")


# ---------------- 3. topk_refresh：按问题词面刷新 Top-K ----------------
def test_topk_refresh_selects_top_k():
    setup()
    filt, fst = R.plan_delta_writes(DEMO, strategy="filtered")
    texts, st = R.plan_delta_writes(DEMO, strategy="topk_refresh",
                                    questions=["Where does Alice live?"], topk=1)
    ok("topk_refresh: 只刷新 Top-K", len(texts) == 1, str(texts))
    ok("topk_refresh: 写入条数 <= filtered", len(texts) <= len(filt))
    ok("topk_refresh: 选中与问题最相关的槽（lives in London）",
       "London" in texts[0], str(texts))
    ok("topk_refresh: skip_reasons.topk 记录被裁掉的条数",
       st["skip_reasons"]["topk"] == len(filt) - 1, str(st["skip_reasons"]))
    ok("topk_refresh: topk 写进 stats", st["topk"] == 1)

    # K 大于候选数时不得凭空多写
    texts2, st2 = R.plan_delta_writes(DEMO, strategy="topk_refresh",
                                      questions=["Where does Alice live?"], topk=99)
    ok("topk_refresh: K > 候选数时不扩容", len(texts2) == len(filt) and
       st2["skip_reasons"]["topk"] == 0, str(texts2))


# ---------------- 4. prepare_delta_memory 真实写入路径 ----------------
def test_prepare_delta_memory_writes_filtered_set():
    setup()
    calls = []
    snap = {"layer.0": torch.ones(2)}
    R.DELTA_WRITE = True
    R.DELTA_WRITE_STRATEGY = "filtered"
    R.reset_delta_mem_states = lambda m: calls.append("reset")
    R.set_delta_mem_write_enabled = lambda m, en: calls.append(("write", en))
    R.get_delta_mem_online_state = lambda m: (calls.append("snapshot"), snap)[1]
    R.collect_delta_mem_state_stats = None
    R.model = FakeModel(calls)

    out = R.prepare_delta_memory(DEMO, questions=["Where does Alice live?"])
    n_fwd = sum(1 for c in calls if c == "forward")
    ok("prepare: 返回快照", out is snap)
    ok("prepare: forward 次数 == write_chunks",
       n_fwd == R.DELTA_STATS["write_chunks"], f"{n_fwd} vs {R.DELTA_STATS}")
    ok("prepare: 写入量少于 all 基线",
       R.DELTA_STATS["write_chunks"] < len(DEMO_SENTS),
       str(R.DELTA_STATS["write_chunks"]))
    ok("prepare: 记录 snapshot_layers", R.DELTA_STATS["snapshot_layers"] == 1)
    ok("prepare: 记录耗时（>=0）", R.DELTA_STATS["write_seconds"] >= 0.0)
    ok("prepare: 记录 strategy", R.DELTA_STATS["strategy"] == "filtered")
    ok("prepare: 写入后关闭写开关", calls[-2] == ("write", False))


def test_prepare_delta_memory_empty_filter_raises():
    setup()
    R.DELTA_WRITE = True
    R.DELTA_WRITE_STRATEGY = "filtered"
    R.reset_delta_mem_states = lambda m: None
    R.set_delta_mem_write_enabled = lambda m, en: None
    R.get_delta_mem_online_state = lambda m: {}
    R.collect_delta_mem_state_stats = None
    os.environ.pop("AR_ALLOW_EMPTY_DELTA_SNAP", None)
    narrative = "In 2021 she moved to Lyon. She worked there for years. Then she left."
    try:
        R.prepare_delta_memory(narrative)
        ok("prepare: 过滤后为空必须报错（禁止空写入静默退化）", False)
    except RuntimeError as e:
        ok("prepare: 过滤后为空必须报错（禁止空写入静默退化）",
           "过滤后没有任何可写单元" in str(e), str(e))


def test_prepare_no_write_keeps_default():
    """AR_DELTA_WRITE=0（默认）时行为不变：不写、不产生统计噪声。"""
    setup()
    R.DELTA_WRITE = False
    R.DELTA_WRITE_STRATEGY = "filtered"
    R.reset_delta_mem_states = lambda m: None
    R.set_delta_mem_write_enabled = lambda m, en: None
    out = R.prepare_delta_memory(DEMO)
    ok("default: AR_DELTA_WRITE=0 仍不写", out == {})
    ok("default: 统计里 write_chunks=0", R.DELTA_STATS["write_chunks"] == 0)


# ---------------- 5. 三档策略在配置指纹上互相区分 ----------------
def test_strategies_have_distinct_fingerprints():
    setup()
    ids = {}
    for st in R.DELTA_WRITE_STRATEGIES:
        R.DELTA_WRITE_STRATEGY = st
        params = R._manifest_params()
        ok(f"fingerprint: {st} 策略进 manifest 参数",
           params.get("delta_write_strategy") == st, str(params.get("delta_write_strategy")))
        ids[st] = run_manifest.build(kind="ar", src="eventqa_65536", params=params)["config_id"]
    ok("fingerprint: 三档策略 config_id 互不相同",
       len(set(ids.values())) == 3, str(ids))
    ok("fingerprint: topk 也进指纹",
       "delta_write_topk" in R._manifest_params())


def test_invalid_strategy_rejected():
    setup()
    try:
        R.plan_delta_writes(DEMO, strategy="bogus")
        ok("非法策略被拒绝", False)
    except ValueError as e:
        ok("非法策略被拒绝", "非法" in str(e), str(e))


def test_record_carries_delta_write_fields():
    """每题 record 必须落盘写入量/跳过量/快照层数（否则无法核对 Delta 是否真参与）。"""
    src = open(R.__file__, encoding="utf-8").read()
    rec = src.split('"delta": {"write": bool(DELTA_WRITE)', 1)[1].split("},", 1)[0]
    for field in ('"write_chunks": DELTA_STATS.get("write_chunks"',
                  '"skipped_chunks": DELTA_STATS.get("skipped_chunks"',
                  '"snapshot_layers": len(snap)',
                  '"write_seconds": DELTA_STATS.get("write_seconds"',
                  '"strategy": DELTA_WRITE_STRATEGY'):
        ok(f"record: delta 含 {field.split(':')[0]}", field in rec, rec)


def main():
    tests = [
        test_all_strategy_is_old_behaviour,
        test_filtered_fewer_and_final_value,
        test_slot_key_separates_attributes,
        test_topk_refresh_selects_top_k,
        test_prepare_delta_memory_writes_filtered_set,
        test_prepare_delta_memory_empty_filter_raises,
        test_prepare_no_write_keeps_default,
        test_strategies_have_distinct_fingerprints,
        test_invalid_strategy_rejected,
        test_record_carries_delta_write_fields,
    ]
    for t in tests:
        print(f"[{t.__name__}]")
        t()
    print(f"ALL PASS ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
