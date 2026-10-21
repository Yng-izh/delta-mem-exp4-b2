# test_ar_v9_wiring.py — run_ar_v7.py 的 v9 接线 / 截断顺序 / 诊断字段 CPU 单测
#
# 不加载模型、不读 parquet、不初始化 CUDA：stub tokenizer + 随机小 EMB。
# 覆盖：
#   1. AR_MODE=v9 建池 + readout(topk) 先分数选 Top-K 再位置排序
#   2. EventQA 后向非对称扩展（anchor_events）与超配额时保留锚点邻域
#   3. 「先截断后排序」：v8 路径按分数优先裁候选，不再截掉高分尾部
#   4. ar_diag.record 逐题诊断字段（hits 分数降序 / answer_in_retrieved）
#   6. 时间信息进生成 Prompt（日期前缀 + 编号，AR_V9_PROMPT_DATE）
#   7. token 级证据预算（2500 token，按分数装填 + dropped_ids 记账）
#      与跨池槽位 ID 对齐（diag_ctx 带 state_pool/used_state_ids -> 'state:<idx>'）
import os
import re
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ar_diag
import run_ar_v7 as R
import v9_slotpool as v9


def _line_payload(line):
    """去掉 '<serial>. [<date>] ' 前缀，取回原文（用于断言顺序与内容）。"""
    return re.sub(r"^\d+\.\s+(\[[^\]]+\]\s+)?", "", line)


class _Batch(dict):
    """同时支持 enc["input_ids"] 与 enc.input_ids（对齐 HF BatchEncoding）。"""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


class StubTok:
    """词级 stub tokenizer：str 或 list[str] -> padding batch。"""

    def __call__(self, texts, add_special_tokens=False, padding=False,
                 truncation=False, max_length=None, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]
        seqs = []
        for t in texts:
            ws = [w for w in v9.WORD_RE.findall(t.lower())] or ["<empty>"]
            if truncation and max_length:
                ws = ws[:max_length]
            seqs.append(ws)
        L = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), L), dtype=torch.long)
        am = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            for j, w in enumerate(s):
                ids[i, j] = hash(w) % 9000 + 10
            am[i, :len(s)] = 1
        return _Batch(input_ids=ids, attention_mask=am)


DIM = 32
torch.manual_seed(7)
EMB_MAT = F.normalize(torch.randn(10000, DIM), dim=-1)

R.tok = StubTok()
R.EMB = EMB_MAT
R.TOP_K = 2
R.V9_CTX_UNITS = 40
R.V9_TOPK_STATE = 0
# 模块导入时的预算默认值（t6 要求 2500），下面各测试按需临时改写
_BUDGET_DEFAULT = R.V9_TOKEN_BUDGET


def ok(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"  ok - {name}")


def fresh_v9(ctx, family="longmemeval"):
    """重置 v9 全局并建池（v9_import 会注入 R.tok / R.EMB）。"""
    v9.set_layer18_provider(None)
    R.FAMILY = family
    R.MODE = "v9"
    R.v9_slotpool = None
    return R.build_pool_for_context_v9(ctx)


# ---------------- 1. v9 topk：先分数选 Top-K 再位置排序 ----------------
def test_v9_topk_score_first_then_position():
    ctx = ("Alpha beta gamma. Delta epsilon zeta. Eta theta iota. "
           "Kappa lambda mu. Nu xi omicron. Pi rho sigma.")
    epi, st, _n = fresh_v9(ctx)
    ok("v9 units non-overlapping", len(epi) == 6)
    # 这一段测的是「先分数选 Top-K、再位置排序」的老口径（AR_V9_CTX_TOKEN_BUDGET=0 时的路径）；
    # token 预算模式（默认 2500）在 test_v9_token_budget_fill 里单独测。
    R.V9_TOKEN_BUDGET = 0
    try:
        block, dbg, dctx = R.build_block_v9(epi, st, "epsilon kappa", {}, 0)

        ok("v9 dbg mode", dbg["mode"] == "topk_score_first")
        ok("v9 selected exactly Top-K", len(dctx["hits"]) == 2)
        scores = [s for _, s in dctx["hits"]]
        ok("v9 hits score-descending", scores == sorted(scores, reverse=True))
        ok("v9 used_ids position-ascending", dctx["used_ids"] == sorted(dctx["used_ids"]))
        ok("v9 used ids == dbg idxs", dctx["used_ids"] == dbg["idxs"])
        ok("budget=0 时不限流（ctx_budget 记录 0）", dbg["ctx_budget"] == 0)
        ok("budget=0 时无预算裁剪", dbg["dropped_ids"] == [] and dbg["dropped_tokens"] == 0)
        # 命中句 1（epsilon）与句 3（kappa）都必须被选中
        ok("v9 both matching sentences selected", set(dctx["used_ids"]) == {1, 3})
        # 拼块行格式（t5）：'<serial>. [<date>] <原文>'；本上下文无时间戳 -> 无方括号
        ok("v9 block follows original order",
           [_line_payload(l) for l in block.split("\n")]
           == [epi.texts[i] for i in dctx["used_ids"]])
        ok("v9 block lines carry serial prefix",
           all(re.match(r"^\d+\. ", l) for l in block.split("\n")))
        ok("v9 no fabricated date brackets when unit has no timestamp",
           "[" not in block)
        ok("v9 serials are global 1-based ingestion order",
           dbg["serials"] == [epi.metas[i]["start"] + 1 for i in dctx["used_ids"]])
        ok("v9 serials strictly ascending (编号越大越新)",
           dbg["serials"] == sorted(dbg["serials"])
           and len(set(dbg["serials"])) == len(dbg["serials"]))

        # 诊断字段：hits 顺序即 Recall@K 顺序
        d = ar_diag.record(answer="epsilon", query="epsilon kappa", flat=True, **dctx)
        ok("diag status ok", d["diag_status"] == "ok")
        ok("diag retrieved ids == hits order",
           d["retrieved_slot_ids"] == [i for i, _ in dctx["hits"]])
        ok("diag answer_in_retrieved", d["answer_in_retrieved"] is True)
        ok("diag answer_in_context", d["answer_in_context"] is True)
        ok("diag answer_hit_slot in Top-2", d["answer_hit_slot"] in (0, 1))
        ok("diag recall@3 hits", d["recall_at_k"]["3"] == 1)
        ok("diag recall@1 consistent with hit slot",
           d["recall_at_k"]["1"] == (1 if d["answer_hit_slot"] == 0 else 0))
        ok("diag used_slot_ids", d["used_slot_ids"] == dctx["used_ids"])

        # 逐题记录必须能直接写 JSONL（dbg 含 stats/hits，diag 平铺）
        import json
        rec = {"context_id": 0, "question_idx": 0, "retrieval": dbg, **d}
        line = json.dumps(rec, ensure_ascii=False)
        ok("diag record json-serializable", '"diag_status": "ok"' in line)
    finally:
        R.V9_TOKEN_BUDGET = _BUDGET_DEFAULT


# ---------------- 2. EventQA 后向非对称扩展 ----------------
def test_v9_eventqa_backward():
    ctx = ("Alice moved to Paris in 1997. Bob painted the wall. "
           "Carol wrote a letter. Dave fixed the roof. Erin sang a song.")
    epi, st, _n = fresh_v9(ctx, family="eventqa")
    R.EQA_BACK, R.EQA_AHEAD = 1, 1
    meta = {"previous_events": ["1. Alice moved to Paris"]}
    block, dbg, dctx = R.build_block_v9(epi, st, "What happened next?", meta, 0)

    ok("eventqa mode", dbg["mode"] == "eventqa_backward")
    ok("eventqa anchor", dbg["anchor"] == 0)
    ok("eventqa asymmetric window", dctx["used_ids"] == [0, 1])
    ok("eventqa dbg back/ahead", (dbg["back"], dbg["ahead"]) == (1, 1))

    # 超配额：保留锚点邻域，而不是排序后砍尾部
    R.V9_CTX_UNITS = 2
    try:
        _b, dbg2, dctx2 = R.build_block_v9(epi, st, "What happened next?", meta, 0)
        ok("eventqa cap keeps anchor neighborhood", dctx2["used_ids"] == [0, 1])
    finally:
        R.V9_CTX_UNITS = 40
        R.EQA_BACK, R.EQA_AHEAD = 16, 4


# ---------------- 3. 先截断后排序 ----------------
def test_cap_before_sort():
    ok("cap keeps score order", R._cap_before_sort([9, 2, 5], 2) == [9, 2])
    ok("cap centered on anchor",
       R._cap_before_sort(list(range(10)), 4, anchor=5) == [3, 4, 5, 6])
    ok("cap dedupes", R._cap_before_sort([3, 3, 1, 2], 3) == [3, 1, 2])


class _FakeV8Epi:
    """v8 路径假池：命中按分数降序给出，邻域不扩展。"""

    def __init__(self, n=12):
        self.texts = [f"Sentence{i} word{i}." for i in range(n)]
        self.metas = [{"start": i, "end": i} for i in range(n)]

    def __len__(self):
        return len(self.texts)

    def retrieve(self, query, top_k=12):
        return [(8, 0.9), (2, 0.8), (11, 0.7), (0, 0.6), (5, 0.5), (3, 0.4)][:top_k]

    def neighbor_expand(self, hits, extra=0):
        return [i for i, _ in hits]

    def best_unit_for(self, text):
        return 0, 0.0


def test_v8_truncate_before_sort():
    R.FAMILY = "longmemeval"
    R.MODE = "v8"
    R.CTX_SENTS = 4
    R.TOP_K = 6
    try:
        epi = _FakeV8Epi()
        block, dbg, dctx = R.build_block_v8(epi, [], "word", {}, 0)
        # 分数序前 4 = [8,2,11,0]；按位置排序后拼块 = [0,2,8,11]
        ok("v8 cap before sort keeps high-score tail",
           dctx["used_ids"] == [0, 2, 8, 11])
        ok("v8 block order follows positions",
           block.split("\n") == [epi.texts[i] for i in [0, 2, 8, 11]])
        # 旧实现会先排序成 [0,2,3,5,8,11] 再截断 -> [0,2,3,5]，丢掉 8/11
        ok("v8 old truncation would have dropped 8/11",
           not set([0, 2, 8, 11]).issubset({0, 2, 3, 5}))
    finally:
        R.CTX_SENTS = 40
        R.TOP_K = 2


# ---------------- 4. v7 ChunkPool 诊断路径 ----------------
def test_v7_diag_path():
    R.FAMILY = "longmemeval"
    R.MODE = "v7"
    R.TOP_K = 2
    R.DEV = "cpu"
    try:
        p = R.ChunkPool()
        p.add_batch(["Alpha beta gamma.", "Delta epsilon zeta.", "Eta theta iota."])
        block, dbg, dctx = R.build_block(p, "epsilon", {}, 0)
        ok("v7 hits score-descending",
           [s for _, s in dctx["hits"]] == sorted([s for _, s in dctx["hits"]],
                                                  reverse=True))
        ok("v7 used ids position-ascending", dctx["used_ids"] == sorted(dctx["used_ids"]))
        ok("v7 block follows position order",
           block.split("\n") == [p.texts[i] for i in dctx["used_ids"]])
        d = ar_diag.record(answer="epsilon", query="epsilon", flat=True, **dctx)
        ok("v7 diag ok", d["diag_status"] == "ok" and d["answer_in_retrieved"] is True)
        ok("v7 pool has no metas -> starts None",
           all(s is None for s in d["retrieved_starts"]))
    finally:
        R.TOP_K = 2


# ---------------- 5. 诊断字段边界 ----------------
def test_diag_edge_cases():
    d = ar_diag.record(answer="Paris", query="q", hits=[], used_ids=[], flat=True)
    ok("empty hits -> no_hits", d["diag_status"] == "no_hits")
    ok("empty hits -> None answer_in_retrieved", d["answer_in_retrieved"] is None)
    rubric = "A long preference answer " * 20
    d2 = ar_diag.record(answer=rubric, query="q", hits=[(0, 1.0)], used_ids=[0],
                        texts=["some text"], flat=True)
    ok("rubric -> not applicable", d2["diag_status"] == "rubric_not_applicable")


# ---------------- 6. 时间信息进 Prompt（t5） ----------------
# 每个 Session 前一行日期；Session 内的句子**没有**自己的时间戳，必须继承 Session 头日期。
DATE_CTX = ("2023-05-20\nAlice likes tea.\nShe moved to Paris.\n"
            "2023-06-01\nBob visited her in June.")


def test_v9_prompt_date_prefix():
    epi, st, _n = fresh_v9(DATE_CTX)
    R.TOP_K = 6
    try:
        block, dbg, dctx = R.build_block_v9(epi, st, "Alice tea Paris Bob", {}, 0)
        lines = block.split("\n")
        ok("date: 每行都是 '<serial>. [YYYY-MM-DD] <原文>'",
           all(re.match(r"^\d+\. \[(?:19|20)\d{2}-\d{2}-\d{2}\] ", l) for l in lines))
        # 继承日期（本句无显式时间戳，取 Session 头日期）必须真的进 Prompt
        ok("date: 继承的 Session 日期进入 Prompt（句子自身无时间戳）",
           "[2023-05-20] Alice likes tea." in block
           and "[2023-06-01] Bob visited her in June." in block)
        ok("date: dbg.unit_dates 与行一一对应",
           dbg["unit_dates"] == ["2023-05-20", "2023-05-20", "2023-05-20",
                                 "2023-06-01", "2023-06-01"])
        ok("date: 日期前缀不改变顺序（payload 顺序 == 位置顺序）",
           [_line_payload(l) for l in lines] == [epi.texts[i] for i in dctx["used_ids"]])
        ok("date: serial 仍全局递增（编号越大越新）",
           dbg["serials"] == sorted(dbg["serials"]) ==
           [epi.metas[i]["start"] + 1 for i in dctx["used_ids"]])
        ok("date: dbg.prompt_date 记录开关状态", dbg["prompt_date"] is True)

        # AR_V9_PROMPT_DATE=0（SERIAL 未显式开）-> 与旧实现逐字节一致
        R.V9_PROMPT_DATE, R.V9_PROMPT_SERIAL = False, False
        try:
            block0, dbg0, dctx0 = R.build_block_v9(epi, st, "Alice tea Paris Bob", {}, 0)
            old = "\n".join(epi.texts[i] for i in dctx0["used_ids"])
            ok("date=0: 块与旧实现逐字节一致", block0 == old)
            ok("date=0: 无编号、无方括号",
               "[" not in block0 and not re.match(r"^\d+\. ", block0.split("\n")[0]))
            ok("date=0: unit_dates 全为 None（不写日期）",
               all(d is None for d in dbg0["unit_dates"]))
            ok("date=0: 选中单元与 DATE=1 相同（只改呈现，不改检索）",
               dctx0["used_ids"] == dctx["used_ids"])
        finally:
            R.V9_PROMPT_DATE, R.V9_PROMPT_SERIAL = True, True
    finally:
        R.TOP_K = 2


def test_prompt_date_no_fabrication():
    ok("date: None -> None", R.format_event_date(None) is None)
    ok("date: 空串 -> None", R.format_event_date("") is None)
    ok("date: ISO 原样", R.format_event_date("2023-05-20") == "2023-05-20")
    ok("date: YYYY/M/D 归一", R.format_event_date("2023/5/7") == "2023-05-07")
    ok("date: '12 January 2005'", R.format_event_date("12 January 2005") == "2005-01-12")
    ok("date: 'January 12, 2005'", R.format_event_date("January 12, 2005") == "2005-01-12")
    ok("date: 只有年份就只写年份（不编造月日）", R.format_event_date("1997") == "1997")
    ok("date: 年代", R.format_event_date("the 1990s") == "1990s")
    ok("date: 解析不出 -> None", R.format_event_date("sometime last week") is None)
    # 无时间单元不得出现方括号（"不要伪造"）
    ok("date: 无时间单元省略方括号",
       R.render_v9_prompt_lines([(1, "Alice moved to Lyon.", None)],
                                with_date=True, with_serial=True)
       == ["1. Alice moved to Lyon."])


def test_prompt_serial_date_ablation():
    rows = [(3, "Alice likes tea.", "2023-05-20")]
    ok("ablate: 只开日期",
       R.render_v9_prompt_lines(rows, with_date=True, with_serial=False)
       == ["[2023-05-20] Alice likes tea."])
    ok("ablate: 只开编号",
       R.render_v9_prompt_lines(rows, with_date=False, with_serial=True)
       == ["3. Alice likes tea."])
    ok("ablate: 全关 == 旧输出",
       R.render_v9_prompt_lines(rows, with_date=False, with_serial=False)
       == ["Alice likes tea."])
    ok("ablate: 全开",
       R.render_v9_prompt_lines(rows, with_date=True, with_serial=True)
       == ["3. [2023-05-20] Alice likes tea."])


def test_prompt_date_manifest_fingerprint():
    import run_manifest
    R.MODE = "v9"
    ids = {}
    try:
        for d in (True, False):
            R.V9_PROMPT_DATE = d
            R.V9_PROMPT_SERIAL = d
            p = R._manifest_params()
            ok(f"fingerprint: prompt_date={int(d)} 进 manifest 参数",
               p["v9"]["prompt_date"] is d)
            ids[d] = run_manifest.build(kind="ar", src="longmemeval_s", params=p)["config_id"]
        ok("fingerprint: DATE=1/0 的 config_id 不同（不共用结果文件）",
           ids[True] != ids[False])
    finally:
        R.V9_PROMPT_DATE = True
        R.V9_PROMPT_SERIAL = True


# ---------------- 7. token 级证据预算（t6） ----------------
BUDGET_CTX = " ".join(f"Fact{i} about item{i} holds." for i in range(12))


def test_v9_token_budget_fill():
    epi, st, _n = fresh_v9(BUDGET_CTX)
    ok("budget: 候选池够大", len(epi) >= 8)
    line_tok = [R.prompt_tokens(R.render_v9_prompt_lines([R._v9_row(epi, i)])[0])
                for i in range(len(epi))]
    # 预算 = 恰好装得下 3 行（第 4 行必然超预算）
    budget = sum(line_tok[:3])
    R.V9_TOKEN_BUDGET = budget
    try:
        block, dbg, dctx = R.build_block_v9(epi, st, "item3 item7", {}, 0)
        ok("budget: ctx_budget 记录预算值", dbg["ctx_budget"] == budget)
        ok("budget: 装入 token 数 ≤ 预算", dbg["ctx_tokens"] <= budget)
        ok("budget: 装入 token 数与块逐行一致",
           dbg["ctx_tokens"] == sum(R.prompt_tokens(l) for l in block.split("\n")))
        # 最高分单元必在块内（预算再紧也保留第一条证据）
        top1 = int(dctx["hits"][0][0])
        ok("budget: 最高分单元必在块内", top1 in dctx["used_ids"])
        # dropped 非空且可解释：都是池内下标、与 used 不相交、token 数 > 0
        dropped = dbg["dropped_ids"]
        ok("budget: dropped_ids 非空", len(dropped) > 0)
        ok("budget: dropped 与 used 不相交", not (set(dropped) & set(dctx["used_ids"])))
        ok("budget: dropped 都是合法池下标",
           all(0 <= i < len(epi) for i in dropped))
        ok("budget: dropped_tokens > 0", dbg["dropped_tokens"] > 0)
        ok("budget: dropped_ids_total ≥ 落盘条数",
           dbg["dropped_ids_total"] >= len(dropped))
        # 「装填按分数降序、超预算即停」：最后一条 kept 的分数 ≥ 第一条 dropped 的分数
        score = dict((int(i), float(s)) for i, s in epi.retrieve("item3 item7",
                                                                top_k=len(epi)))
        ok("budget: kept 是分数序前缀（最后 kept ≥ 第一条 dropped）",
           min(score[i] for i in dctx["used_ids"]) >= max(score[i] for i in dropped))
        # 输出仍是位置序（I2 不被预算破坏）
        ok("budget: 块内仍按原文位置排序",
           dctx["used_ids"] == sorted(dctx["used_ids"]))
        ok("budget: 行数与 used_ids 一致", len(block.split("\n")) == len(dctx["used_ids"]))
        # 预算过小时至少保留一条（不拼空块）
        R.V9_TOKEN_BUDGET = 1
        b2, dbg2, dctx2 = R.build_block_v9(epi, st, "item3 item7", {}, 0)
        ok("budget: 预算小到 1 token 仍保留最高分一条", len(dctx2["used_ids"]) == 1)
        ok("budget: 该条即最高分单元", dctx2["used_ids"][0] == top1)
    finally:
        R.V9_TOKEN_BUDGET = _BUDGET_DEFAULT


def test_v9_budget_is_default_and_in_fingerprint():
    import run_manifest
    ok("budget: 默认预算 2500", _BUDGET_DEFAULT == 2500)
    R.MODE = "v9"
    ids = {}
    try:
        for b in (2500, 0):
            R.V9_TOKEN_BUDGET = b
            p = R._manifest_params()
            ok(f"budget: ctx_token_budget={b} 进 manifest 参数",
               p["v9"]["ctx_token_budget"] == b)
            ids[b] = run_manifest.build(kind="ar", src="longmemeval_s",
                                        params=p)["config_id"]
        ok("budget: 预算 2500/0 的 config_id 不同（不共用结果文件）",
           ids[2500] != ids[0])
    finally:
        R.V9_TOKEN_BUDGET = _BUDGET_DEFAULT


def test_v9_diag_state_pool_wiring():
    """diag_ctx 必须带 state_pool/used_state_ids，状态行才落成 'state:<idx>'。

    场景：预算只装得下 1 条情景 + 1 条状态（V9_CTX_UNITS=1 作为二级上限），
    状态行不是情景行的重复，因此会真的出现在拼块里。
    """
    ctx = ("Alice moved to Paris in 1997. "
           "Bob is a doctor at the big hospital. Carol likes tea.")
    epi, st, _n = fresh_v9(ctx)
    ok("state: 池里有状态单元", len(st) > 0)
    q = "Where did Alice move in 1997?"
    R.V9_TOPK_STATE, R.V9_CTX_UNITS = 2, 1
    epi_top = int(epi.retrieve(q, top_k=1)[0][0])
    st_top = int(st.retrieve(q, top_k=1)[0][0])
    budget = (R.prompt_tokens(R.render_v9_prompt_lines([R._v9_row(epi, epi_top)])[0])
              + R.prompt_tokens(R.render_v9_prompt_lines([R._v9_row(st, st_top)])[0]))
    R.V9_TOKEN_BUDGET = budget
    try:
        block, dbg, dctx = R.build_block_v9(epi, st, q, {}, 0)
        ok("state: diag_ctx 带 state_pool", dctx.get("state_pool") is st)
        ok("state: diag_ctx 带 used_state_ids", len(dctx.get("used_state_ids") or []) > 0)
        ok("state: 状态行确实拼进块（非情景行重复）",
           st.texts[dctx["used_state_ids"][0]] in block)
        ok("state: 预算内只装 1 情景 + 1 状态",
           len(dctx["used_ids"]) == 1 and len(dctx["used_state_ids"]) == 1)
        d = ar_diag.record(answer="Paris", query=q, flat=True, **dctx)
        ok("state: used_slot_uids 与 used_texts 等长",
           len(d["used_slot_uids"]) == len(d["used_texts"]))
        ok("state: used_slot_uids 含 state: 前缀",
           any(str(u).startswith("state:") for u in d["used_slot_uids"]))
        ok("state: 不再退化成 texts_without_ids",
           d["used_alignment"] != "texts_without_ids")
        ok("state: 跨池对齐后 answer_in_context 仍正确",
           d["answer_in_context"] is True)
        ok("state: 预算裁掉的状态槽位有记录", len(dbg["dropped_state_ids"]) > 0)
    finally:
        R.V9_TOPK_STATE, R.V9_CTX_UNITS = 0, 40
        R.V9_TOKEN_BUDGET = _BUDGET_DEFAULT


def test_main_records_budget_fields():
    """记录里必须真的出现 ctx_budget / ctx_tokens / dropped_ids / gen_retries。"""
    src = open(R.__file__, encoding="utf-8").read()
    main_body = src.split("def main():", 1)[1]
    ok("record: ctx_budget 落盘", '"ctx_budget": dbg.get("ctx_budget")' in main_body)
    ok("record: ctx_tokens 落盘", '"ctx_tokens": dbg.get("ctx_tokens")' in main_body)
    ok("record: dropped_ids 落盘", '"dropped_ids": dbg.get("dropped_ids", [])' in main_body)
    ok("record: dropped_tokens 落盘",
       '"dropped_tokens": dbg.get("dropped_tokens", 0)' in main_body)
    ok("record: OOM 重试明细落盘", '"gen_retries"' in main_body)
    ok("record: generate_readout 收到证据优先级 trim",
       "generate_readout(block, qtxt, trim=trim)" in main_body)
    ok("record: block_chunks 按真实拼块行数",
       'len(diag_ctx.get("used_texts") or dbg.get("idxs") or [])' in main_body)


def main():
    print("[1] v9 topk score-first")
    test_v9_topk_score_first_then_position()
    print("[2] v9 eventqa backward")
    test_v9_eventqa_backward()
    print("[3] truncate before sort")
    test_cap_before_sort()
    test_v8_truncate_before_sort()
    print("[4] v7 ChunkPool diag")
    test_v7_diag_path()
    print("[5] diag edge cases")
    test_diag_edge_cases()
    print("[6] 时间信息进 Prompt (t5)")
    test_v9_prompt_date_prefix()
    test_prompt_date_no_fabrication()
    test_prompt_serial_date_ablation()
    test_prompt_date_manifest_fingerprint()
    print("[7] token 级证据预算 (t6)")
    test_v9_token_budget_fill()
    test_v9_budget_is_default_and_in_fingerprint()
    test_v9_diag_state_pool_wiring()
    test_main_records_budget_fields()
    print("ALL PASS")


if __name__ == "__main__":
    main()
