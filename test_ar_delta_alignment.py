# test_ar_delta_alignment.py — Delta Memory 读出对齐 CPU 单测
#
# 不加载模型、不初始化 CUDA：stub tokenizer / fake model / fake delta API。
# 覆盖：
#   1. AR_DELTA_WRITE：context 起点 reset -> 查询前 write -> disable -> snapshot
#   2. 查询期不 reset；逐题 delta_restore(snapshot)
#   3. 快照 clone（官方 API 缺失时的手工 fallback）
#   4. AR_INPUTS_EMBEDS 读出路径开关（inputs_embeds vs input_ids）
#   5. OOM 重试按证据优先级裁剪（记录 keep_ids/dropped_ids），不再保留最后 N token
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_ar_v7 as R


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

    def generate(self, **kw):
        self.calls.append(("generate", sorted(kw.keys())))
        if "inputs_embeds" in kw:
            return torch.tensor([[7, 8, 9]])
        return torch.cat([kw["input_ids"], torch.tensor([[7, 8, 9]])], dim=1)

    def named_modules(self):
        return [("", self)]


def ok(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"  ok - {name}")


def setup():
    R.DEV = "cpu"
    R.MAX_NEW_TOKENS = 3
    R.sys_msg = ""
    R.mem_tpl = "{context}"
    R.tok = FakeTok()
    R.EMB = torch.randn(10000, 8)
    R.INPUTS_EMBEDS = False
    R.READOUT_MODE = "prompt"      # 读出模式默认值（旧开关仅作别名）
    R.DELTA_RESTORE = True
    R.DELTA_WRITE = False
    R.model = FakeModel()


# ---------------- 1. 写入顺序：reset -> write -> disable -> snapshot ----------------
def test_prepare_delta_memory_order():
    setup()
    calls = []
    snap = {"layer.0": torch.ones(2)}
    R.DELTA_WRITE = True
    R.chunk_text_into_sentences = lambda ctx, chunk_size: ["c1", "c2"]
    R.reset_delta_mem_states = lambda m: calls.append("reset")
    R.set_delta_mem_write_enabled = lambda m, en: calls.append(("write", en))
    R.get_delta_mem_online_state = lambda m: (calls.append("snapshot"), snap)[1]
    R.collect_delta_mem_state_stats = None
    R.model = FakeModel(calls)

    R.reset_delta_mem_states(R.model)          # main 在 context 起点调用
    out = R.prepare_delta_memory("ctx")

    ok("prepare returns snapshot", out is snap)
    ok("order: reset -> write(True) -> forwards -> write(False) -> snapshot",
       calls == ["reset", ("write", True), "forward", "forward",
                 ("write", False), "snapshot"])
    ok("write disabled after prepare", calls[-2] == ("write", False))


def test_prepare_delta_memory_no_write():
    setup()
    calls = []
    R.DELTA_WRITE = False
    R.reset_delta_mem_states = lambda m: calls.append("reset")
    R.set_delta_mem_write_enabled = lambda m, en: calls.append(("write", en))
    R.get_delta_mem_online_state = lambda m: calls.append("snapshot") or {}
    R.chunk_text_into_sentences = lambda ctx, chunk_size: ["c1"]
    R.reset_delta_mem_states(R.model)
    out = R.prepare_delta_memory("ctx")
    ok("no write -> empty snapshot", out == {})
    ok("no write -> only reset", calls == ["reset"])


def test_prepare_requires_write_api():
    setup()
    R.DELTA_WRITE = True
    R.reset_delta_mem_states = None
    R.set_delta_mem_write_enabled = None
    try:
        R.prepare_delta_memory("ctx")
        ok("missing write api raises", False)
    except RuntimeError:
        ok("missing write api raises", True)


# ---------------- 2. 快照 clone / 逐题恢复 ----------------
def test_delta_restore():
    setup()
    calls = []
    snap = {"layer.0": torch.ones(2)}
    R.load_delta_mem_online_state = lambda m, s: calls.append(("load", s is snap))
    R.delta_restore(snap)
    ok("restore uses official API", calls == [("load", True)])
    calls.clear()
    R.delta_restore({})
    ok("empty snapshot -> no restore call", calls == [])


class _FakeDeltaMod:
    def __init__(self):
        self.delta_state = torch.arange(4.0)

    def parameters(self):
        yield torch.zeros(1, dtype=torch.bfloat16)


def test_snapshot_fallback_clone():
    setup()
    R.get_delta_mem_online_state = None
    mod = _FakeDeltaMod()
    R.iter_delta_mem_modules = lambda m: [("layer.0", mod)]
    snap = R.delta_snapshot(R.model)
    ok("fallback snapshot key", list(snap) == ["layer.0"])
    ok("fallback snapshot value", torch.equal(snap["layer.0"], mod.delta_state))
    ok("fallback snapshot is a clone",
       snap["layer.0"].data_ptr() != mod.delta_state.data_ptr())

    # 手工 restore：写回 device/dtype 由模块参数决定
    target = _FakeDeltaMod()
    target.delta_state = None
    R.load_delta_mem_online_state = None
    R.model.named_modules = lambda: [("layer.0", target)]
    R.delta_restore(snap)
    ok("fallback restore writes state", target.delta_state is not None)
    ok("fallback restore keeps values",
       torch.equal(target.delta_state.float(), torch.arange(4.0)))


# ---------------- 3. inputs_embeds 读出路径 ----------------
def test_generate_readout_input_ids():
    setup()
    pred, in_len, out_len = R.generate_readout("block", "q")
    ok("input_ids path pred", pred == "answer")
    ok("input_ids path lengths", (in_len, out_len) == (3, 3))
    kw = R.model.calls[-1][1]
    ok("input_ids path uses input_ids", "input_ids" in kw and "inputs_embeds" not in kw)


def test_generate_readout_inputs_embeds():
    setup()
    saved = R.READOUT_MODE
    R.READOUT_MODE = "text_embeds"          # 显式模式（旧的 AR_INPUTS_EMBEDS 等价写法）
    try:
        pred, in_len, out_len = R.generate_readout("block", "q")
    finally:
        R.READOUT_MODE = saved
    ok("embeds path pred", pred == "answer")
    ok("embeds path input_len", in_len == 3)
    ok("embeds path output_len (only new tokens)", out_len == 3)
    kw = R.model.calls[-1][1]
    ok("embeds path uses inputs_embeds",
       "inputs_embeds" in kw and "input_ids" not in kw)
    ok("embeds path passes attention_mask", "attention_mask" in kw)


class _WordTok(FakeTok):
    """按词数给 token 的 tokenizer：让 OOM 重试后的 prompt 长度真的变化。"""

    def __call__(self, text, return_tensors=None):
        n = max(1, len(str(text).split()))
        return _Batch(input_ids=torch.arange(n, dtype=torch.long).unsqueeze(0))


class _OomModel(FakeModel):
    """前 n_oom 次 generate 抛 OOM，之后成功；记录每次 prompt 长度。"""

    def __init__(self, n_oom=1):
        super().__init__()
        self.lengths = []
        self.n_oom = n_oom

    def generate(self, **kw):
        self.lengths.append(kw["input_ids"].shape[1])
        if len(self.lengths) <= self.n_oom:
            raise torch.OutOfMemoryError("simulated oom")
        return torch.cat([kw["input_ids"], torch.tensor([[7, 8, 9]])], dim=1)


def test_generate_readout_oom_retry():
    """OOM 重试必须按证据优先级裁剪，并把 keep_ids/dropped_ids 记录下来。"""
    setup()
    R.tok = _WordTok()
    R.model = _OomModel(n_oom=1)
    lines = ["1. [2023-05-20] Alice likes tea.", "2. She moved to Paris.",
             "3. Bob likes coffee.", "4. Carol runs daily.",
             "5. Dave paints walls.", "6. Erin sings.", "7. Finn reads.",
             "8. Gina jogs."]
    # 分数从高到低，最低分在最后两行 -> 重试必须先丢这两行
    trim = {"lines": list(lines), "prio": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1],
            "ids": [f"epi:{i}" for i in range(8)]}
    pred, in_len, out_len = R.generate_readout("\n".join(lines), "q", trim=trim)

    ok("oom retry succeeds", pred == "answer" and out_len == 3)
    ok("oom retry 重跑一次 generate", len(R.model.lengths) == 2)
    ok("oom retry 记录到 trim['retries']", len(trim.get("retries") or []) == 1)
    r0 = trim["retries"][0]
    ok("oom retry 先丢最低分的两行（epi:6/epi:7，分数 0.3/0.1）",
       r0["dropped_ids"] == ["epi:6", "epi:7"])
    ok("oom retry keep_ids 是剩余高分行",
       r0["keep_ids"] == [f"epi:{i}" for i in range(6)])
    ok("oom retry 记录重试前的 prompt token 数", r0["prompt_tokens"] == R.model.lengths[0])
    ok("oom retry 后 prompt 变短（整行丢，不是截尾 token）",
       R.model.lengths[1] < R.model.lengths[0])
    ok("oom retry final keep/dropped 可追溯",
       trim["final_keep_ids"] == [f"epi:{i}" for i in range(6)]
       and trim["final_dropped_ids"] == ["epi:6", "epi:7"])
    ok("oom retry 丢的不是靠后的 token 而是最低分证据（keep 仍是分数序前缀）",
       trim["final_keep_ids"] == trim["ids"][:6])


def test_generate_readout_oom_raises_without_evidence():
    """只剩一条证据还 OOM 就如实抛出：不静默丢证据、不截尾。"""
    setup()
    R.tok = _WordTok()
    R.model = _OomModel(n_oom=99)
    lines = ["1. Alice likes tea.", "2. She moved to Paris."]
    trim = {"lines": list(lines), "prio": [0.9, 0.1], "ids": ["epi:0", "epi:1"]}
    raised = False
    try:
        R.generate_readout("\n".join(lines), "q", trim=trim)
    except torch.OutOfMemoryError:
        raised = True
    ok("oom 无证据可丢时抛错", raised)
    ok("抛错前已记录一次裁剪", len(trim.get("retries") or []) == 1)


# ---------------- 4. 结构约束：查询期禁止 reset ----------------
def test_main_reset_only_before_queries():
    src = open(R.__file__, encoding="utf-8").read()
    main_body = src.split("def main():", 1)[1]
    ok("main resets delta once per context",
       main_body.count("reset_delta_mem_states(model)") == 1)
    ok("main prepares delta before question loop",
       main_body.index("prepare_delta_memory(ctx") <
       main_body.index("for qi, q in enumerate(qs):"))
    ok("main passes questions to prepare (topk_refresh 写入前粗排)",
       "prepare_delta_memory(ctx, questions=qs)" in main_body)
    ok("main restores snapshot inside question loop",
       main_body.count("delta_restore(snap)") == 1)
    ok("no post-loop delta write block",
       "if DELTA_WRITE:\n            set_delta_mem_write_enabled" not in main_body)


def main():
    print("[1] prepare delta memory order")
    test_prepare_delta_memory_order()
    test_prepare_delta_memory_no_write()
    test_prepare_requires_write_api()
    print("[2] snapshot clone / restore")
    test_delta_restore()
    test_snapshot_fallback_clone()
    print("[3] inputs_embeds readout")
    test_generate_readout_input_ids()
    test_generate_readout_inputs_embeds()
    test_generate_readout_oom_retry()
    test_generate_readout_oom_raises_without_evidence()
    print("[4] query-period no reset")
    test_main_reset_only_before_queries()
    print("ALL PASS")


if __name__ == "__main__":
    main()
