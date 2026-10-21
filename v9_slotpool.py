# v9_slotpool.py — v9 非重叠索引 + 分数优先检索 + EventQA 窗口扩展（方向未定，见 (3)）
#                 + 时间戳参与打分/时序重排 + 第 18 层隐状态混合稠密打分
#
# 相对 v8（v8_slotpool.py）的改动，对应 v9 重构的 5 个检索问题：
#   (1) 非重叠索引 extract_units：句级非重叠单元（stride == block_sents，无覆盖重叠）。
#       v8 用 3 句滑窗 stride=1，同一句最多出现在 3 个单元里 -> 池膨胀 3 倍、
#       Top-K 的"K 个单元"实际只覆盖 K/3 个不同句、位置排序有歧义。
#       v9 每个句子恰好属于一个单元，Recall@K 与位置排序都良定义（block_sents=1 即单句索引）。
#   (2) 先按分数选 Top-K、再按原文位置排序：retrieve() 只按分数返回；
#       retrieve_ordered()/readout() 对【已选中的 K 个】按 start 排序，
#       绝不做"先把全池按位置排序再截断"（那会把高分尾部单元截掉）。
#   (3) EventQA 窗口扩展：expand_backward(anchor, back=16, ahead=4)
#       —— 以"已发生事件"锚点为准，向前取 back 个、向后取 ahead 个（当前默认前多后少）。
#       v7/v8 只向后（anchor..anchor+EQA_WIN）取 4 块，锚点之前的证据全部拿不到。
#       ⚠️ 窗口方向【未定】：配对检验（同 (ctx,qid) 交集 + McNemar 精确 + Newcombe CI + Holm 校正）
#       显示 b0_a20 / b4_a16 vs b16_a4 均 +8.00pp、raw p=0.0386，但 Holm p=0.2314，12 对比较
#       全部不显著（最小 Holm p=0.0764），且观测 8.0pp < MDE 9.7pp（power<0.80）。
#       → 现有数据既不支持改默认值，也不支持声称任何方向更优；保留旧默认、行为不变。
#       证据：experiments/results/paired_window_test.json（脚本 experiments/paired_window_test.py）。
#   (4) 时间戳参与检索打分与时序重排：
#       - 打分：单元编码用 index_text = "In <ts>: <text>"（ts 可为显式或继承），
#         query 里的日期 token 因此能与单元时间戳逐 token MaxSim 命中；
#         texts 仍保存原文，拼给 LLM 的上下文不含人工前缀。
#       - 重排：temporal_rerank 按 |year_unit - year_query| 的指数衰减给分加成
#         （V9_TEMPORAL_TAU / V9_TEMPORAL_BONUS），无年份单元不加分也不罚分。
#   (4b) 相对时间语义（v9.1 补齐）：TIMESTAMP_RE 扩展 yesterday/today/tomorrow、
#       last|this|next <week|month|year|season>、星期名、N days/weeks/months/years ago；
#       相对时间以【本 Session 头日期】为锚解析成确定日期，无锚或精度不足时只标记不解析
#       （ts=None，ts_unresolved=True，绝不编造日期）；Session 标记本身参与 ev 判定。
#   (5) 第 18 层隐状态混合稠密打分接口：
#       - 调用方注入 set_layer18_provider(fn)：fn(texts, kind) -> [N, d] | None，
#         kind ∈ {"units", "query"}；fn 内部用模型 hidden_states[V9_L18_LAYER] 的
#         mean-pool（层号默认 18）。
#       - 或直接 attach_layer18(hidden) + retrieve(..., q_hidden=...)。
#       - hybrid = (1-alpha) * IDF-MaxSim + alpha * ((cos+1)/2)，alpha=V9_L18_ALPHA。
#       - provider 缺失/抛错/形状不符 -> 自动退化为纯词面 IDF-MaxSim（CPU 单测路径）。
#   (5b) 池内 whitening（v9.2，默认开）：稠密余弦在原始隐空间里被"共同方向"支配
#       （实测池内两两余弦均值可达 0.86），排序几乎退化成常数偏移。故对池内 l18 矩阵
#       做中心化 + 协方差白化后再算余弦：
#         V9_L18_WHITEN=1（默认） 中心化 + 白化（C=Hc^T Hc/(N-1)+eps·I，W=V·diag(1/√λ)·V^T）
#         V9_L18_WHITEN=0        纯 L2 归一化（旧的 v9.0/v9.1 行为，仅用于消融对照）
#         d>=N 或 N<=1（协方差秩不足，白化会放大噪声）-> 自动退化为【只中心化】
#         eps（V9_L18_WHITEN_EPS，默认 1e-4）同时是特征值下限，保证数值稳定
#       query 必须过【同一个】(mean, W) 变换，否则余弦不在同一空间（dense_scores 已保证）。
#   (5c) 复用流式前向中间态（避免二次前向）：
#       - set_layer18_provider(fn, precomputed=[N,d], precomputed_texts=[...]) 或
#         set_layer18_precomputed(hidden, texts=...) 可把调用方【建池/流式写入时已算好】的
#         单元侧向量交给本模块，建索引阶段直接查表，完全不调用 fn；
#       - 查表按**文本**匹配（texts 已给时），因此 epi 池与 state 池共享同一份矩阵也不会
#         串行号（早期实现按 offset 盲切，StatePool 传自己的槽 id 会静默取到别人的向量）；
#       - 只有调用方明确给 position 语义（未给 texts）时才退化为按 offset 切片，并告警一次；
#       - 没给 precomputed 时退回"按需前向"的旧行为，并在首次真正前向时告警一次。
#
# 注入约定（同 v6/v7/v8）：v9_slotpool.tok = tok ; v9_slotpool.EMB = EMB。
# CR（FactConsolidation）主路径仍走 v7_slotpool.py；本文件 StatePool 只作为 AR 的
# 可选状态补充与 CR v9 扩展点（_should_merge 可覆写为主语+关系双阈值）。
#
# AR 接入示例（run_ar_v7.py, AR_MODE=v9）：
#   epi, st = v9.build_pools_for_context(ctx)
#   idxs, dbg = epi.readout(q_core, top_k=TOP_K, mode="topk")
#   idxs, dbg = epi.readout(q_core, mode="eventqa", anchor_events=evs,
#                           back=EQA_BACK, ahead=EQA_AHEAD)   # anchor_events: last/last2
#   block = "\n".join(epi.texts[i] for i in idxs)      # texts 是原文，无时间戳前缀
#
# CPU 可测：注入 stub tok + 小 EMB（+ 可选 stub layer18 provider）即可跑全部逻辑单测。

import datetime
import math
import os
import re
import string
from collections import Counter

import torch
import torch.nn.functional as F

# ----------------------------- 环境参数 -----------------------------
BLOCK_SENTS    = int(os.environ.get("V9_BLOCK_SENTS", "1"))       # 非重叠单元句数（1=单句索引）
UNIT_MAX_WORDS = int(os.environ.get("V9_UNIT_MAX_WORDS", "512"))  # 单元词级硬截断
ENCODE_BS      = int(os.environ.get("V9_ENCODE_BS", "1024"))      # 词级编码分批
BATCH          = int(os.environ.get("V9_BATCH", "64"))            # 全池 MaxSim 分批
IDF_UNSEEN     = float(os.environ.get("V9_IDF_UNSEEN", "0"))      # 0 -> 用池内最大 IDF
TOKV_BF16      = os.environ.get("V9_TOKV_BF16", "1") == "1"       # 逐 token 向量 bf16 存储（省内存）
# EventQA 窗口默认值（back=16 / ahead=4，沿用旧默认，不改行为；V9_EQA_BACK/V9_EQA_AHEAD 开关保留）。
# ⚠️ 默认值未经显著验证：配对检验 Holm p=0.2314，MDE=9.7pp，
#    见 experiments/results/paired_window_test.json —— 窗口方向目前【未定】，
#    既不支持改默认值，也不支持声称任何方向更优（12 对比较 Holm 后全部不显著）。
EQA_BACK       = int(os.environ.get("V9_EQA_BACK", "16"))         # EventQA 锚点前取几句（默认值，方向未定）
EQA_AHEAD      = int(os.environ.get("V9_EQA_AHEAD", "4"))         # EventQA 锚点后取几句（默认值，方向未定）
TEMPORAL_TAU   = float(os.environ.get("V9_TEMPORAL_TAU", "2.0"))  # 时序加成衰减（年）
TEMPORAL_BONUS = float(os.environ.get("V9_TEMPORAL_BONUS", "0.15"))  # 时序加成上限（相对）
L18_ALPHA      = float(os.environ.get("V9_L18_ALPHA", "0.3"))     # 稠密通道权重
L18_LAYER      = int(os.environ.get("V9_L18_LAYER", "18"))        # 调用方应取的隐状态层号
L18_WHITEN     = os.environ.get("V9_L18_WHITEN", "1") == "1"      # 池内 whitening（=0 等价旧的纯 L2-only）
L18_WHITEN_EPS = float(os.environ.get("V9_L18_WHITEN_EPS", "1e-4"))  # 白化正则项（特征值下限）
L18_REUSE_EPI  = os.environ.get("V9_L18_REUSE_EPI", "1") == "1"   # 状态池复用 epi 已算好的向量（免二次前向）
STATE_THETA    = float(os.environ.get("V9_STATE_THETA", "0.92"))  # 状态池语义合并阈值
SPLIT_ON_NEWLINE = os.environ.get("V9_SPLIT_ON_NEWLINE", "1") == "1"  # 换行也切句（聊天多轮）
SESSION_RESET  = os.environ.get("V9_SESSION_RESET", "1") == "1"   # Session 边界不继承时间戳
# ---- LongMemEval 专用（readout(mode="lme")；默认不影响 mode="topk"/"eventqa"）----
LME_TURN_WINDOW = int(os.environ.get("V9_LME_TURN_WINDOW", "1"))   # 命中单元在【本 Session 内】前后各扩几个 turn
MAX_PER_SESSION = int(os.environ.get("V9_MAX_PER_SESSION", "2"))   # 同一 Session 最多取几个命中单元（跨 Session 多样性）
LME_QUOTA       = os.environ.get("V9_SESSION_QUOTA", "1") == "1"   # lme 模式默认开启跨 Session 配额

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "is", "was", "are", "were", "be", "been", "what", "which", "who", "where",
    "when", "that", "this", "does", "did", "has", "have", "had", "and", "or",
    "it", "its", "his", "her", "their", "he", "she", "they", "as", "but",
}

# 事件线索：过去时事件动词
EVENT_HINT_RE = re.compile(
    r"\b(?:was|were|had|did|done|became|become|moved|founded|created|born|died|killed|"
    r"wrote|directed|released|won|lost|opened|closed|joined|left|started|"
    r"ended|married|graduated|retired|elected|appointed|discovered|invaded|"
    r"attended|worked|played|traveled|visited|returned|occurred|happened|crossed|"
    r"swam|climbed|escaped|arrived|departed|sent|gave|took|made|built|sold|bought)\b",
    flags=re.IGNORECASE,
)

# ----------------------------- 时间标记（绝对 + 相对） -----------------------------
# ⚠️ 本段（正则 + 锚点解析 + 相对时间求解）与 v8_slotpool.py 中的同名实现保持一致，
#    两边需同步修改，否则 v8/v9 的时间语义会漂移（AR_MODE=v8/v9 结果不可比）。
_MON = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"

# 绝对时间标记：March 3, 2001 / 12 January 2005 / 2001-03-03 / 1997 / the 1990s
# 口径说明：ISO(YYYY-MM-DD) 提到纯年份之前——旧版把 ISO 放在年份之后，年份分支
# 会在同一位置先命中，导致 "2001-03-03" 只留下 "2001"（ISO 分支是死代码）。
ABS_TIMESTAMP_RE = re.compile(
    rf"(?:\b{_MON}[a-z]*\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b"
    rf"|\b\d{{1,2}}\s+{_MON}[a-z]*\.?,?\s+\d{{4}}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b(?:19|20)\d{2}\b"
    r"|\bthe\s+\d{4}s\b)",
    flags=re.IGNORECASE,
)

# 相对时间标记：yesterday / last week / last summer / Monday / 3 days ago ...
# 星期缩写（Mon/Tue/...）只在【首字母大写】时匹配，避免 "the sun is hot" 里的 sun
# 被当成星期天；星期全称小写不歧义，仍大小写不敏感。
REL_TIMESTAMP_RE = re.compile(
    r"\b(?:yesterday|today|tomorrow)\b"
    r"|\b(?:last|this|next)\s+(?:week|month|year|summer|spring|autumn|fall|winter)\b"
    r"|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    r"|(?-i:\b(?:Mon|Tue|Tues|Wed|Thu|Thur|Thurs|Fri|Sat|Sun)\b)"
    r"|\b\d+\s+(?:day|week|month|year)s?\s+ago\b",
    flags=re.IGNORECASE,
)

# 统一时间标记 = 绝对（优先，保持历史 group(0) 口径）+ 相对
TIMESTAMP_RE = re.compile(ABS_TIMESTAMP_RE.pattern + "|" + REL_TIMESTAMP_RE.pattern,
                          flags=re.IGNORECASE)

# ----------------------------- 相对时间解析（以 Session 头日期为锚） -----------------------------
_MONTH_IDX = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}

_DATE_YMD_RE = re.compile(r"\b((?:19|20)\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b")
_DATE_MDY_RE = re.compile(
    rf"\b({_MON})[a-z]*\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+((?:19|20)\d{{2}})\b",
    flags=re.IGNORECASE)
_DATE_DMY_RE = re.compile(
    rf"\b(\d{{1,2}})\s+({_MON})[a-z]*\.?,?\s+((?:19|20)\d{{2}})\b",
    flags=re.IGNORECASE)
_DATE_YM_RE = re.compile(r"\b((?:19|20)\d{2})[-/.](\d{1,2})\b")

_DAY_OFFSET = {"yesterday": -1, "today": 0, "tomorrow": 1}
# 季节没有确切日期，这里给【可复现的代表日】（北半球节气起点）
_SEASON_MD = {"spring": (3, 20), "summer": (6, 21), "autumn": (9, 22),
              "fall": (9, 22), "winter": (12, 21)}
_WEEKDAY_IDX = {"monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1,
                "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thur": 3,
                "thurs": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5,
                "sunday": 6, "sun": 6}


def _days_in_month(y, m):
    if m == 2:
        leap = (y % 4 == 0 and y % 100 != 0) or y % 400 == 0
        return 29 if leap else 28
    return 30 if m in (4, 6, 9, 11) else 31


def _shift_month(y, m, delta):
    t = (y * 12 + (m - 1)) + int(delta)
    return t // 12, t % 12 + 1


def _fmt_ymd(y, m, d):
    return f"{y:04d}-{m:02d}-{min(max(1, int(d)), _days_in_month(y, m)):02d}"


def _fmt_ym(y, m):
    return f"{y:04d}-{m:02d}"


def _shift_year(anchor, delta, prec):
    y = anchor.year + int(delta)
    if prec == "day":
        return _fmt_ymd(y, anchor.month, anchor.day)
    if prec == "month":
        return _fmt_ym(y, anchor.month)
    return f"{y:04d}"


def extract_date_text(text):
    """取文本里【最精确】的日期文本（YYYY-MM-DD > Month D, YYYY > D Month YYYY > YYYY-MM > YYYY）。

    只用于给相对时间找锚点；TIMESTAMP_RE.group(0) 的历史口径不受影响
    （所以 "2021/07/15" 作为时间戳仍是 "2021"，但作为锚点能拿到整日精度）。
    """
    s = str(text or "")
    for rx in (_DATE_YMD_RE, _DATE_MDY_RE, _DATE_DMY_RE):
        m = rx.search(s)
        if m:
            return m.group(0)
    m = _DATE_YM_RE.search(s)
    if m:
        return m.group(0)
    m = YEAR_RE.search(s)
    return m.group(0) if m else None


def parse_anchor_date(anchor_ts):
    """锚点时间戳 -> (datetime.date | None, precision | None)，precision ∈ {"day","month","year"}。

    非法日期（如 2021-02-31）返回 (None, None)：宁可解析不出来，也不编造日期。
    """
    ts = extract_date_text(anchor_ts) if anchor_ts else None
    if not ts:
        return None, None
    m = _DATE_YMD_RE.search(ts)
    if m:
        try:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3))), "day"
        except ValueError:
            return None, None
    for rx, (mo_g, d_g, y_g) in ((_DATE_MDY_RE, (1, 2, 3)), (_DATE_DMY_RE, (2, 1, 3))):
        m = rx.search(ts)
        if not m:
            continue
        mo = _MONTH_IDX.get(m.group(mo_g).lower()[:3])
        if mo is None:
            return None, None
        try:
            return datetime.date(int(m.group(y_g)), mo, int(m.group(d_g))), "day"
        except ValueError:
            return None, None
    m = _DATE_YM_RE.search(ts)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return (datetime.date(y, mo, 1), "month") if 1 <= mo <= 12 else (None, None)
    m = YEAR_RE.search(ts)
    if m:
        return datetime.date(int(m.group(0)), 1, 1), "year"
    return None, None


def is_relative_ts(marker):
    """该标记是否为相对时间表达（yesterday / last week / Monday / 3 days ago ...）。"""
    if not marker:
        return False
    return bool(REL_TIMESTAMP_RE.fullmatch(" ".join(str(marker).split())))


def resolve_relative_ts(marker, anchor_ts):
    """相对时间 -> 确定日期字符串；无锚点或精度不足 -> None（只标记，不编造）。

    约定（确定性、可复现）：
      - yesterday/today/tomorrow、N days/weeks ago、裸星期名、last/this/next week
        需要【日】精度锚点；
      - N months ago、last/this/next month 需要日/月精度锚点，日精度时按"同日"取（月末夹取）；
      - N years ago、last/this/next year 允许年精度锚点；
      - last/this/next <season> 用季节代表日（春 3/20、夏 6/21、秋 9/22、冬 12/21），
        年份按 offset 平移——季节本身没有确切日期，这里给可复现代表日；
      - 裸星期名 -> 锚点当天或之前最近的那个星期几。
    输出格式跟随结果精度：day -> "YYYY-MM-DD"，month -> "YYYY-MM"，year -> "YYYY"。
    """
    if not marker:
        return None
    txt = " ".join(str(marker).lower().split())
    anchor, prec = parse_anchor_date(anchor_ts)
    if anchor is None or prec is None:
        return None

    if txt in _DAY_OFFSET:
        if prec != "day":
            return None
        return (anchor + datetime.timedelta(days=_DAY_OFFSET[txt])).isoformat()

    m = re.fullmatch(r"(\d+)\s+(day|week|month|year)s?\s+ago", txt)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if unit in ("day", "week"):
            if prec != "day":
                return None
            return (anchor - datetime.timedelta(days=n * (7 if unit == "week" else 1))).isoformat()
        if unit == "month":
            if prec == "year":
                return None
            y, mo = _shift_month(anchor.year, anchor.month, -n)
            return _fmt_ymd(y, mo, anchor.day) if prec == "day" else _fmt_ym(y, mo)
        return _shift_year(anchor, -n, prec)

    m = re.fullmatch(r"(last|this|next)\s+(week|month|year|summer|spring|autumn|fall|winter)", txt)
    if m:
        off = {"last": -1, "this": 0, "next": 1}[m.group(1)]
        unit = m.group(2)
        if unit == "week":
            if prec != "day":
                return None
            return (anchor + datetime.timedelta(days=7 * off)).isoformat()
        if unit == "month":
            if prec == "year":
                return None
            y, mo = _shift_month(anchor.year, anchor.month, off)
            return _fmt_ymd(y, mo, anchor.day) if prec == "day" else _fmt_ym(y, mo)
        if unit == "year":
            return _shift_year(anchor, off, prec)
        sm, sd = _SEASON_MD[unit]
        y = anchor.year + off
        return _fmt_ymd(y, sm, sd) if prec != "year" else f"{y:04d}"

    wd = _WEEKDAY_IDX.get(txt)
    if wd is not None:
        if prec != "day":
            return None
        return (anchor - datetime.timedelta(days=(anchor.weekday() - wd) % 7)).isoformat()
    return None


def resolve_time_marker(marker, anchor_ts):
    """时间标记 -> (ts, status)。

    status ∈ {"absolute", "relative", "unresolved"}：
      - "absolute"  ：绝对标记，ts 原样保留（历史口径不变）；
      - "relative"  ：相对标记 + 可用锚点（Session 头日期），ts 为确定日期；
      - "unresolved"：相对标记但无锚点/精度不足，ts=None —— 【只标记不解析，不编造日期】。
    """
    if not marker:
        return None, None
    if not is_relative_ts(marker):
        return marker, "absolute"
    ts = resolve_relative_ts(marker, anchor_ts)
    return (ts, "relative") if ts else (None, "unresolved")


# --------------------- LongMemEval 时间约束（first/last/before/after） ---------------------
# 只解析【约束型】表述。注意 "last week" / "last year" / "last night" 是时间点（相对时间），
# 不是"最近一次"的顺序约束 —— 必须排除，否则 "What did I do last week?" 会被当成 last 约束。
_TEMP_BEFORE_RE = re.compile(r"\b(?:before|prior to|earlier than|no later than)\s+(.{1,48})",
                             flags=re.IGNORECASE)
_TEMP_AFTER_RE = re.compile(r"\b(?:after|later than|since|no earlier than)\s+(.{1,48})",
                            flags=re.IGNORECASE)
_TEMP_ORDER_RE = {
    "last": re.compile(r"\b(?:last|latest|most recent(?:ly)?|finally|at the end)\b",
                       flags=re.IGNORECASE),
    "first": re.compile(r"\b(?:first|earliest|initially|at the beginning)\b",
                        flags=re.IGNORECASE),
}
# "last/first" 后面紧跟这些时间单位 -> 是时间点/时间范围，不是顺序约束
# （注意不含 "time"："last time" = 最近一次，仍是顺序约束）
_REL_UNIT_AFTER = re.compile(r"\s*(?:week|month|year|summer|spring|autumn|fall|winter|"
                             r"night|weekend|monday|tuesday|wednesday|thursday|"
                             r"friday|saturday|sunday)\b", flags=re.IGNORECASE)
_OPERAND_STOP_RE = re.compile(r"[,;?]|\b(?:and|or|but|when|while|that|which|who)\b",
                              flags=re.IGNORECASE)


def _trim_operand(text, max_words=6):
    """截出时间表达式：截到标点/从句连词为止，最多 max_words 个词。"""
    s = str(text or "").strip()
    m = _OPERAND_STOP_RE.search(s)
    if m:
        s = s[:m.start()].strip()
    return " ".join(s.split()[:max_words])


def _parse_order(query):
    """first/last 顺序约束；'last week' 之类相对时间点不算 last 约束。"""
    q = str(query or "")
    for kind in ("last", "first"):
        for m in _TEMP_ORDER_RE[kind].finditer(q):
            if _REL_UNIT_AFTER.match(q[m.end():]):
                continue          # "last week" / "first week" / "last Monday" -> 时间点
            return kind
    return None


def parse_temporal_constraint(query, query_ts=None):
    """query -> 时间约束 dict（kind ∈ {"none","before","after"}，order ∈ {None,"first","last"}）。

    - before/after 的操作数支持绝对日期（"before 2019-05-01"）与相对时间
      （"after yesterday"，以 query_ts 为锚；无锚则解析失败）；
    - first/last/most recent 只记顺序约束，不参与日期过滤；
    - **解析不出来一律 kind="none"**（调用方退回现状，结果不变，只记 temporal_constraint=none）。
    """
    q = str(query or "")
    out = {"kind": "none", "order": None, "operand": None, "raw": None,
           "date": None, "precision": None}
    for kind, rx in (("before", _TEMP_BEFORE_RE), ("after", _TEMP_AFTER_RE)):
        for m in rx.finditer(q):
            operand = _trim_operand(m.group(1))
            if not operand:
                continue
            date, prec = parse_anchor_date(operand)
            if date is None and is_relative_ts(operand):
                ts = resolve_relative_ts(operand, query_ts)
                date, prec = parse_anchor_date(ts) if ts else (None, None)
            if date is None:
                continue
            out.update(kind=kind, operand=operand,
                       raw=re.sub(r"[\s,;?.!]+$", "", m.group(0).strip()),
                       date=date.isoformat(), precision=prec)
            break
        if out["kind"] != "none":
            break
    out["order"] = _parse_order(q)
    return out


_PREC_RANK = {None: -1, "year": 0, "month": 1, "day": 2}


def unit_date(meta):
    """单元 -> (datetime.date | None, precision | None)。

    取 ts / event_time / anchor_ts 中【最精确】的那个（同精度时 ts 优先，因为句内显式日期
    比 Session 头更具体）。ts 的历史口径可能只到年（"2023/05/20" -> "2023"），
    此时用 Session 头日期（anchor_ts，整日精度）兜底，before/after 才有判别力。
    """
    best = (None, None)
    for key in ("ts", "event_time", "anchor_ts"):
        v = (meta or {}).get(key)
        if not v:
            continue
        d, p = parse_anchor_date(v)
        if d is None:
            continue
        if _PREC_RANK[p] > _PREC_RANK[best[1]]:
            best = (d, p)
    return best


def unit_date_span(meta):
    """单元 -> (最早可能日期, 最晚可能日期, precision)；无日期返回 (None, None, None)。

    before/after 过滤用【区间】语义，避免把粗粒度时间戳当成"当年 1 月 1 日"误杀：
      - day 精度   -> [d, d]
      - month 精度 -> [当月 1 日, 当月末日]
      - year 精度  -> [当年 1 月 1 日, 当年 12 月 31 日]
    即 "after X" 只要单元最晚可能日期 > X 就保留，"before X" 只要最早可能日期 < X 就保留。
    """
    d, p = unit_date(meta)
    if d is None:
        return None, None, None
    if p == "year":
        return datetime.date(d.year, 1, 1), datetime.date(d.year, 12, 31), p
    if p == "month":
        return (datetime.date(d.year, d.month, 1),
                datetime.date(d.year, d.month, _days_in_month(d.year, d.month)), p)
    return d, d, p


def unit_date_center(meta):
    """单元 -> 排序用的代表日期（时间区间中点）；无日期返回 None。

    day 精度 -> 当天；month/year 精度 -> 区间中点（不用年初，避免粗粒度单元在
    first/last 排序里被系统性地推到最前/最后）。
    """
    lo, hi, _p = unit_date_span(meta)
    if lo is None:
        return None
    return lo + datetime.timedelta(days=(hi - lo).days // 2)


YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
WORD_RE = re.compile(r"[A-Za-z0-9_\u00C0-\u024F'-]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Session / 日期头：LongMemEval 每个 session 前有一行日期；也覆盖 Session 1 / ## Session 等写法。
# 只在【行首】匹配，避免把句中的 "2023-05-20" 误判成新会话。
SESSION_RE = re.compile(
    r"^\s*(?:"
    r"\[?\s*(?:session|conversation|dialogue|chat)\s*#?\s*\d+"
    r"|#{1,6}\s*(?:session|conversation)"
    r"|={3,}\s*(?:session|conversation)"
    r"|\[?\s*(?:date|time)\s*[:：]"
    r"|\[?\s*(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}"
    # 行首月名日期头（"March 3, 2001" / "3 March 2001"），LongMemEval 之外的日期头写法
    rf"|\[?\s*{_MON}[a-z]*\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b"
    rf"|\[?\s*\d{{1,2}}\s+{_MON}[a-z]*\.?,?\s+\d{{4}}\b"
    r")",
    flags=re.IGNORECASE,
)


def split_sentences(text):
    """句切分（可被调用方覆盖：v9_slotpool.split_sentences = fn）。

    V9_SPLIT_ON_NEWLINE=1（默认）时先按换行切、再按句末标点切：
    聊天/对话数据常有多轮消息没有句末标点，只按 ". " 切会把多轮并成一个单元，
    再被 UNIT_MAX_WORDS 截断，检索单元与原文轮次对不上。
    V9_SPLIT_ON_NEWLINE=0 时退化为旧行为（整段按标点切），便于与历史结果对齐。
    """
    text = (text or "").strip()
    if not text:
        return []
    if not SPLIT_ON_NEWLINE:
        return [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.extend(p.strip() for p in _SENT_SPLIT_RE.split(line) if p.strip())
    return out


def split_sessions(context):
    """按显式 Session / 日期头把上下文切成段：[(session_id, header_ts, text)]。

    没有任何 Session 头时返回单段（session_id=0, header_ts=None）。
    V9_SESSION_RESET=0 时始终返回单段（等价旧行为）。
    设计不变量 I3：时间戳不得跨 Session 继承，因此必须先切出会话边界。

    header_ts 只取【绝对】日期（ABS_TIMESTAMP_RE），保持历史口径：Session 头里若只有
    相对表达（如 "Session 1 — yesterday"），header_ts 仍为 None，不会把相对词当日期继承。
    需要整日精度锚点时用 session_anchor_ts()（见下）。
    """
    text = (context or "").strip()
    if not text:
        return []
    if not SESSION_RESET:
        return [(0, None, text)]
    blocks, cur, cur_ts = [], [], None
    for line in text.splitlines():
        if SESSION_RE.match(line):
            if any(s.strip() for s in cur):
                blocks.append((len(blocks), cur_ts, "\n".join(cur)))
            cur = [line]
            tm = ABS_TIMESTAMP_RE.search(line)
            cur_ts = tm.group(0) if tm else None
        else:
            cur.append(line)
    if any(s.strip() for s in cur):
        blocks.append((len(blocks), cur_ts, "\n".join(cur)))
    return blocks or [(0, None, text)]


def session_anchor_ts(block_text):
    """Session 块首行若是 Session/日期头，返回其中最精确的日期文本（相对时间的锚点）。

    头里没有日期时返回 None —— 此时该 Session 内的相对时间【只标记不解析】。
    """
    lines = (block_text or "").splitlines()
    if not lines or not SESSION_RE.match(lines[0]):
        return None
    return extract_date_text(lines[0])


def extract_year(ts):
    """时间戳 -> 年份 int（'the 1990s' -> 1990；'12 January 2005' -> 2005）；无则 None。"""
    if not ts:
        return None
    m = YEAR_RE.search(str(ts))
    return int(m.group(0)) if m else None


def ts_prefix(ts):
    """时间戳前缀（只用于打分文本，不进入给 LLM 的原文）。"""
    return f"In {ts}:" if ts else ""


def build_index_text(ts, text):
    """打分文本 = 时间戳前缀 + 原文；让日期 token 参与逐 token MaxSim。"""
    pre = ts_prefix(ts)
    return f"{pre} {text}".strip() if pre else text


# ----------------------------- 非重叠单元抽取 -----------------------------
def extract_units(context, block_sents=None):
    """把上下文切成【非重叠】检索单元。

    block_sents=1 -> 单句索引；block_sents>1 -> 连续 b 句成一块（stride == b，无重叠）。
    每个句子恰好属于一个单元（不丢句、不重复），因此 Top-K / 位置排序 / 邻域扩展良定义。

    时间语义（不变量 I3）：
      - 显式时间戳（句子自带）> Session 头日期 > 本 Session 内前文最近一次显式时间戳；
      - **跨 Session 一律不继承**：新 Session 起点把 last_ts 重置为该 Session 的头日期
        （无头日期则为 None），避免 Session 2 的偏好事实继承 Session 1 的年份；
      - 相对时间（yesterday / last week / Monday / 3 days ago ...）以【本 Session 头日期】
        为锚解析成确定日期；头日期缺失或精度不足时【只标记不解析】（ts=None，不编造日期）；
      - `event_time` 只记"事件发生时间"（本句显式时间或 Session 头日期），
        `ingestion_order` 记"写入顺序"（句序号），二者分开维护。

    返回 list[dict]：
      text        原文（拼上下文用，不含人工前缀）
      index_text  打分文本（"In <ts>: <text>"，ts 可为显式或继承）
      start/end   句索引区间（闭区间）
      ts          该单元的时间戳（单元内首个显式标记，否则继承同 Session 起点之前最近一次）
      ts_explicit / ts_inherited
      ts_marker   单元内首个时间标记的原文（绝对或相对；无则 None）
      ts_relative 该标记是否为相对时间表达
      ts_unresolved 相对标记存在但无锚点/精度不足 -> ts 保持 None（只标记不解析）
      year        ts 的年份 int（时序重排用；无则 None）
      is_event    单元内是否含事件动词【或 Session 标记】
      session_id  所属 Session 序号（无 Session 头时恒为 0）
      event_time  事件发生时间（本句显式时间戳或 Session 头日期；无则 None）
      ingestion_order 单元在上下文中的写入顺序（== 首句句号）
    """
    sents, sid_at, hts_at, anchor_at, hdr_at = [], [], [], [], []
    for sid, hts, stext in split_sessions(context):
        anchor = session_anchor_ts(stext)
        for s in split_sentences(stext):
            sents.append(s)
            sid_at.append(sid)
            hts_at.append(hts)
            anchor_at.append(anchor)
            hdr_at.append(bool(SESSION_RE.match(s)))
    n = len(sents)
    b = max(1, int(BLOCK_SENTS if block_sents is None else block_sents))

    ts_at = [None] * n          # 每句显式时间戳（相对时间已解析成确定日期）
    marker_at = [None] * n      # 每句时间标记原文（含无法解析的相对标记）
    rel_at = [False] * n        # 标记是否为相对时间
    unres_at = [False] * n      # 相对标记但无锚点 -> 只标记不解析
    ev_time = [None] * n        # 每句事件时间（显式 or Session 头）
    resolved_ts = [None] * n    # 逐句"同 Session 内最近一次显式时间戳"（含自身），只读
    is_event = [False] * n
    last_ts = None
    last_sid = None
    for i, s in enumerate(sents):
        if sid_at[i] != last_sid:              # Session 边界：不继承上一段的时间
            last_ts, last_sid = hts_at[i], sid_at[i]
        m = TIMESTAMP_RE.search(s)
        if m:
            marker_at[i] = m.group(0)
            rel_at[i] = is_relative_ts(marker_at[i])
            # 锚点 = 本 Session 头日期（整日精度优先）；无锚 -> ts=None，只标记
            ts, status = resolve_time_marker(marker_at[i], anchor_at[i] or hts_at[i])
            unres_at[i] = status == "unresolved"
            if ts is not None:
                last_ts = ts
                ts_at[i] = ts
        resolved_ts[i] = last_ts
        ev_time[i] = ts_at[i] or hts_at[i]
        # Session 标记本身是时间/结构边界 -> 参与 ev 判定
        is_event[i] = bool(EVENT_HINT_RE.search(s)) or hdr_at[i]

    units = []
    for u0 in range(0, n, b):                 # 非重叠：起点每次前进 b
        u1 = min(u0 + b - 1, n - 1)
        explicit = next((ts_at[j] for j in range(u0, u1 + 1) if ts_at[j]), None)
        prev_ts = resolved_ts[u0 - 1] if u0 > 0 and sid_at[u0 - 1] == sid_at[u0] else None
        ts = explicit if explicit is not None else prev_ts
        text = " ".join(sents[u0:u1 + 1])
        units.append({
            "text": text,
            "index_text": build_index_text(ts, text),
            "start": u0,
            "end": u1,
            "ts": ts,
            "ts_explicit": explicit is not None,
            "ts_inherited": explicit is None and ts is not None,
            "ts_marker": next((marker_at[j] for j in range(u0, u1 + 1) if marker_at[j]), None),
            "ts_relative": any(rel_at[j] for j in range(u0, u1 + 1)),
            "ts_unresolved": any(unres_at[j] for j in range(u0, u1 + 1)),
            "year": extract_year(ts),
            "is_event": any(is_event[j] for j in range(u0, u1 + 1)),
            "session_id": sid_at[u0],
            "event_time": next((ev_time[j] for j in range(u0, u1 + 1) if ev_time[j]), None),
            # Session 头日期（整日精度）：ts 的历史口径可能只到年（"2023"），
            # 但 LongMemEval 的 turn 实际发生在头日期那天，before/after 需要这个精度
            "anchor_ts": anchor_at[u0],
            "ingestion_order": u0,
        })
    return units


# ----------------------------- 编码工具 -----------------------------
tok = None    # 调用方注入 tokenizer（CPU 测试可注入 stub）
EMB = None    # 调用方注入嵌入矩阵 [V, d]（同 device）


def _norm_word(w):
    return w.lower().translate(str.maketrans("", "", string.punctuation)).strip("'")


def _content_words(text):
    ws = [_norm_word(w) for w in WORD_RE.findall(text)]
    ws = [w for w in ws if w]
    kept = [w for w in ws if w not in STOPWORDS] or ws or [text.strip().lower()]
    return kept


def encode_words(words):
    """批量编码词列表 -> ([Σlw, d] 归一化逐 token fp32 向量, counts)。分批编码控制显存峰值。"""
    dim = EMB.shape[1] if EMB is not None else 1
    if not words:
        return torch.zeros((0, dim), dtype=torch.float32), []
    dev = EMB.device if EMB is not None else torch.device("cpu")
    out, counts = [], []
    for s in range(0, len(words), ENCODE_BS):
        chunk = list(words[s:s + ENCODE_BS])
        enc = tok(chunk, add_special_tokens=False, padding=True,
                  truncation=True, max_length=16, return_tensors="pt")
        ids = (enc["input_ids"] if isinstance(enc, dict) else enc.input_ids).to(dev)
        am = (enc["attention_mask"] if isinstance(enc, dict) else enc.attention_mask).to(dev)
        vecs = F.normalize(EMB[ids].float(), dim=-1)     # [W, Lw, d]
        for i in range(len(chunk)):
            L = max(int(am[i].sum()), 0)
            if L == 0:
                out.append(torch.zeros((0, dim), dtype=torch.float32))
                counts.append(0)
                continue
            out.append(vecs[i, :L].cpu())
            counts.append(L)
        del enc, ids, am, vecs
    flat = torch.cat([o for o in out], dim=0) if out else torch.zeros((0, dim), dtype=torch.float32)
    return flat, counts


# ----------------------------- 第 18 层隐状态接口 -----------------------------
L18_PROVIDER = None     # fn(texts: list[str], kind: "units"|"query") -> [N, d] | None
L18_STRICT = False      # True：provider 缺失/抛错/形状不符 -> 直接报错，禁止静默退化
L18_PRECOMPUTED = None  # [N, d]：调用方在【流式写入时】缓存好的单元侧隐状态（避免二次前向）
L18_PRECOMPUTED_TEXTS = None   # list[str] | None：与上面行一一对应的打分文本（用于按文本查表）
_L18_TEXT_INDEX = None  # dict[text -> row]：由 TEXTS 惰性建好的反查表
_L18_WARNED = False     # 一次性告警标记（"没给预计算矩阵，正在二次前向"）


def _l18_available():
    """稠密通道是否可用：有 provider，或有调用方缓存的预计算矩阵。"""
    return L18_PROVIDER is not None or L18_PRECOMPUTED is not None


def _warn_l18_once(msg):
    """只打印一次的告警（避免每个 batch 刷屏）。"""
    global _L18_WARNED
    if _L18_WARNED:
        return
    _L18_WARNED = True
    print(f"[v9/l18] WARN: {msg}", flush=True)


def set_layer18_provider(fn, strict=None, precomputed=None, precomputed_texts=None):
    """注入第 18 层隐状态提供者；fn=None 关闭稠密通道。

    fn 契约：
      - fn(texts, kind) 返回 [len(texts), d] 张量（CPU/GPU 均可，内部统一转 CPU fp32 并 L2 归一化）；
      - kind == "units" 用于建索引（一次批量），"query" 用于查询（一次一条）；
      - 返回 None / 抛异常 / 形状不符 -> 默认静默退化为纯词面打分（保证 CPU 单测与无模型路径可用）；
      - strict=True 时改为【严格模式】：上述任何一种失败都抛异常，不允许"看起来跑通了但稠密通道没生效"。
        真实稠密实验（AR_V9_L18=1）必须用 strict=True，否则故障会表现为指标骤降而非报错。
      - 单元与 query 必须来自同一层（V9_L18_LAYER，默认 18）、同一 pool 方式（mean-pool）。

    precomputed（复用流式前向中间态，避免二次前向）：
      - 传 [N, d] 矩阵 = 调用方【在流式写入/建索引时就已经算好】的单元侧第 18 层 mean-pool 向量，
        顺序与池内单元一致；给了它，建索引阶段直接切片使用，**不再调用 fn**（省掉一次完整前向）；
      - query 侧仍需一次前向（或用 readout(..., q_hidden=...) / attach_layer18 直接传）；
      - 没给 precomputed 时退回"按需前向"的旧行为，并在首次真正前向时打印一次告警，
        提示调用方应缓存流式写入的 hidden_states[V9_L18_LAYER]；
      - 也支持事后单独注入：set_layer18_precomputed(hidden) / clear_layer18_precomputed()。

    precomputed_texts（强烈建议给）：
      - 与 precomputed 行一一对应的打分文本（通常就是 epi.index_texts）。
        给了它就按**文本**查表，因此 epi 池 / state 池 / 任意顺序的重复文本都能命中，
        不会因为行号语义不同而静默取到别的向量；
      - 不给则退回"按 offset 盲切"的位置语义（仅当调用方保证 offset 就是矩阵行号时正确），
        并在首次命中时告警一次。
    """
    global L18_PROVIDER, L18_STRICT, L18_PRECOMPUTED, L18_PRECOMPUTED_TEXTS
    global _L18_TEXT_INDEX, _L18_WARNED
    L18_PROVIDER = fn
    _set_precomputed(precomputed, precomputed_texts)
    if strict is not None:
        L18_STRICT = bool(strict)


def _set_precomputed(hidden, texts=None):
    """内部：写入预计算矩阵（+ 可选文本索引），并清掉一次性告警。"""
    global L18_PRECOMPUTED, L18_PRECOMPUTED_TEXTS, _L18_TEXT_INDEX, _L18_WARNED
    L18_PRECOMPUTED = None if hidden is None else _as_l18_matrix(hidden)
    if texts is None or L18_PRECOMPUTED is None:
        L18_PRECOMPUTED_TEXTS = None
    else:
        texts = [str(t) for t in texts]
        if len(texts) != int(L18_PRECOMPUTED.shape[0]):
            raise ValueError(f"precomputed_texts 条数 {len(texts)} != 矩阵行数 "
                             f"{int(L18_PRECOMPUTED.shape[0])}（按文本查表需要一一对应）")
        L18_PRECOMPUTED_TEXTS = texts
    _L18_TEXT_INDEX = None
    _L18_WARNED = False


def _l18_text_index():
    """惰性构建 text -> row 反查表（同文本重复出现时取最早一行）。"""
    global _L18_TEXT_INDEX
    if L18_PRECOMPUTED_TEXTS is None:
        return None
    if _L18_TEXT_INDEX is None:
        idx = {}
        for row, t in enumerate(L18_PRECOMPUTED_TEXTS):
            if t not in idx:
                idx[t] = row
        _L18_TEXT_INDEX = idx
    return _L18_TEXT_INDEX


def _as_l18_matrix(hidden):
    """[N, d] 预计算矩阵 -> CPU fp32；1 维自动升维。"""
    h = torch.as_tensor(hidden, dtype=torch.float32)
    if h.dim() == 1:
        h = h.unsqueeze(0)
    if h.dim() != 2:
        raise ValueError(f"l18 预计算矩阵需要 [N, d]，实际 {tuple(h.shape)}")
    return h.to("cpu")


def set_layer18_precomputed(hidden, texts=None):
    """注入调用方缓存的单元侧隐状态矩阵（[N, d]）。

    texts 给定时按**文本**查表（推荐：epi/state 两池共享同一份矩阵也不会串行号）；
    不给则按 offset 位置语义使用。典型用法（run_ar_v7.py 流式写入）：
        v9.set_layer18_precomputed(vecs, texts=epi.index_texts)   # 建完 epi 池后
    这样后续 state 池里文本相同的槽位直接查表，不再二次前向。
    """
    _set_precomputed(hidden, texts)


def clear_layer18_precomputed():
    set_layer18_precomputed(None)


# ----------------------------- 池内 whitening -----------------------------
# 状态（2026-10-21，t10）：**已接线、已测试**。拟合后 dense_scores 用同一变换同时作用于
#    池内向量与 query（两处 l18_whiten_apply 调用点），池内相似度不再被共同方向分量主导。
#    口径变更：V9_L18_WHITEN=1 为默认；=0 与旧 L2-only **逐位一致**（有 torch.equal 回归断言）。
#    与 AR_V9_L18=1 的**历史跑分不可横比**，横比前必须显式设 V9_L18_WHITEN=0。
#    退化规则：N<=1 / d>=N（秩不足）/ eigh 失败 / 特征值非有限 -> center_only，并在 info.note 记录。
#    仍未验证：白化对真实指标（EventQA/LongMemEval）的影响只有合成池数值证据
#    （N=60,d=16：池内余弦 mean .8636/max .978 -> mean -.0166/max .7168），不得写成收益结论。
#    另：AR_V9_L18 默认 0，稠密通道本身默认关闭。
def l18_whiten_fit(hidden, eps=None):
    """拟合池内 whitening 变换：返回 (mean [d] | None, W [d,d] | None, info dict)。

    规则（确定性、可复现）：
      - V9_L18_WHITEN=0 -> (None, None, mode="l2_only")，等价旧的纯 L2 归一化行为；
      - N<=1 或 d>=N（协方差秩不足，白化会放大噪声）-> 退化为【只中心化】：
        (mean, None, mode="center_only")；
      - 否则 中心化 + 协方差白化：C = Hc^T Hc/(N-1) + eps·I，W = V·diag(1/sqrt(max(λ,eps)))·V^T，
        mode="whiten"。eps 同时是特征值下限，保证数值稳定（不会 1/0）。

    口径说明：池内向量在落盘前已按行 L2 归一化（_call_l18 / attach_layer18 的统一口径），
    因此这里的中心化与白化都作用在单位球面上的向量，白化后再归一化一次。
    这不改变"去掉共同方向 + 均衡各维方差"的目的，但数值上与对原始隐状态做 whitening 不同，
    跨实现对比时不要混用两种口径。
    """
    H = torch.as_tensor(hidden, dtype=torch.float32)
    if H.dim() == 1:
        H = H.unsqueeze(0)
    n, d = int(H.shape[0]), int(H.shape[1])
    e = L18_WHITEN_EPS if eps is None else float(eps)
    if not L18_WHITEN:
        return None, None, {"mode": "l2_only", "n": n, "d": d, "eps": None,
                            "note": "V9_L18_WHITEN=0（旧的纯 L2 行为）"}
    mean = H.mean(dim=0)
    if n <= 1:
        return mean, None, {"mode": "center_only", "n": n, "d": d, "eps": e,
                            "note": "N<=1，无法估计协方差"}
    if d >= n:
        return mean, None, {"mode": "center_only", "n": n, "d": d, "eps": e,
                            "note": "d>=N：协方差秩不足，退化为中心化"}
    Hc = H - mean
    C = (Hc.t() @ Hc) / float(max(n - 1, 1))
    C = C + e * torch.eye(d, dtype=torch.float32)
    try:
        evals, evecs = torch.linalg.eigh(C)
    except Exception as ex:                       # 数值异常 -> 安全退化
        return mean, None, {"mode": "center_only", "n": n, "d": d, "eps": e,
                            "note": f"eigh 失败（{ex!r}），退化为中心化"}
    if not torch.isfinite(evals).all():
        return mean, None, {"mode": "center_only", "n": n, "d": d, "eps": e,
                            "note": "特征值非有限，退化为中心化"}
    evals = evals.clamp_min(e)
    W = (evecs * evals.rsqrt()) @ evecs.t()        # 对称白化矩阵
    info = {"mode": "whiten", "n": n, "d": d, "eps": e,
            "cond": round(float(evals.max() / evals.min()), 4)}
    return mean, W, info


def l18_whiten_apply(hidden, mean=None, W=None):
    """按拟合出的变换作用到向量上，并按行 L2 归一化（mean=None 即纯 L2）。"""
    H = torch.as_tensor(hidden, dtype=torch.float32)
    if H.dim() == 1:
        H = H.unsqueeze(0)
    if mean is not None:
        H = H - mean
    if W is not None:
        H = H @ W
    return F.normalize(H, dim=-1)


def _l18_fail(msg):
    """稠密通道失败：严格模式抛错，否则返回 None 由调用方退化。"""
    if L18_STRICT:
        raise RuntimeError(f"[v9/l18] {msg}（L18_STRICT=1，禁止静默退化）")
    return None


def mean_pool_hidden(hidden, attention_mask=None):
    """把某一层 hidden_states 做 mask mean-pool -> [N, d]。

    hidden: [N, L, d]（批量）或 [L, d]（单条）；
    attention_mask: [N, L]（可选，0 位置不计入均值）。
    典型用法（run_ar_v7.py）：
        hs = model(..., output_hidden_states=True).hidden_states[V9_L18_LAYER]
        unit_vecs = v9.mean_pool_hidden(hs, attn_mask)     # [N, d]
        v9.set_layer18_provider(lambda texts, kind: ...)   # 或 epi.attach_layer18(unit_vecs)
    """
    h = torch.as_tensor(hidden, dtype=torch.float32)
    if h.dim() == 2:
        return h.mean(dim=0, keepdim=True)
    if h.dim() != 3:
        raise ValueError(f"hidden 需要 [N,L,d] 或 [L,d]，实际 {tuple(h.shape)}")
    if attention_mask is None:
        return h.mean(dim=1)
    m = torch.as_tensor(attention_mask, dtype=torch.float32)
    if m.dim() == 1:
        m = m.unsqueeze(0)
    m = m.unsqueeze(-1)
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)


def layer18_spec():
    """日志用：当前稠密通道配置。"""
    return {"layer": L18_LAYER, "alpha": L18_ALPHA, "enabled": _l18_available(),
            "whiten": bool(L18_WHITEN), "whiten_eps": L18_WHITEN_EPS,
            "precomputed": L18_PRECOMPUTED is not None,
            "precomputed_rows": 0 if L18_PRECOMPUTED is None else int(L18_PRECOMPUTED.shape[0]),
            "precomputed_texts": L18_PRECOMPUTED_TEXTS is not None,
            "precomputed_text_rows": 0 if L18_PRECOMPUTED_TEXTS is None
            else len(L18_PRECOMPUTED_TEXTS)}


def _call_l18(texts, kind, offset=None):
    """调用 provider 并规整为 [N, d] CPU fp32 L2 归一化向量；失败时按严格模式处理。

    统一落 CPU：单元侧向量常驻内存（262K 单元 × d），查询期只搬 query；
    设备一致性由 hybrid_scores 在最后一步对齐（dense 结果 .to(lexical.device)）。

    offset（仅 kind="units"）：本批单元在池内的起始行号。若调用方提供了预计算矩阵
    L18_PRECOMPUTED，优先按**文本**查表命中（texts 已给时），命中不了再看 offset：
      - 文本索引命中 -> 直接用（epi/state 池共享同一矩阵也不会串行号）；
      - 无文本索引 + offset 在界内 -> 按位置切片（调用方明确承诺位置语义时才对）；
      - 都不行 -> 告警一次并退回 provider（没有 provider 则退化为纯词面）。
    **完全不调用 provider** 的前提是查表/切片成功，这正是"避免二次前向"的收益点。
    """
    if not texts:
        return None
    if kind == "units" and L18_PRECOMPUTED is not None:
        idx = _l18_text_index()
        if idx is not None:
            rows = [idx.get(str(t)) for t in texts]
            if all(r is not None for r in rows):
                return F.normalize(L18_PRECOMPUTED[rows], dim=-1).to("cpu")
            # 文本索引存在但没全命中：只用位置语义兜底（且必须显式给了 offset）
            if offset is None:
                _warn_l18_once(f"预计算矩阵按文本查表未命中 "
                               f"{sum(r is None for r in rows)}/{len(rows)} 条，"
                               f"且未给 offset，本批退回按需前向")
        elif offset is not None:
            lo = int(offset)
            hi = lo + len(texts)
            if hi <= L18_PRECOMPUTED.shape[0]:
                _warn_l18_once("预计算矩阵未提供 precomputed_texts，正在按 offset 位置切片；"
                               "若调用方是多池共享同一矩阵，请补 precomputed_texts 以免串行号")
                return F.normalize(L18_PRECOMPUTED[lo:hi], dim=-1).to("cpu")
            _warn_l18_once(f"预计算矩阵行数不足（需要 {hi} 行，"
                           f"只有 {L18_PRECOMPUTED.shape[0]} 行），本批退回按需前向")
    if L18_PROVIDER is None:
        return None
    if kind == "units":
        _warn_l18_once("未提供预计算矩阵（set_layer18_provider(..., precomputed=...)），"
                       "正在对单元文本二次前向；建议在流式写入时缓存 "
                       "hidden_states[V9_L18_LAYER] 的 mean-pool 向量")
    try:
        h = L18_PROVIDER(list(texts), kind)
    except Exception as e:
        return _l18_fail(f"provider 抛错 kind={kind}: {e!r}")
    if h is None:
        return _l18_fail(f"provider 返回 None kind={kind} n={len(texts)}")
    h = torch.as_tensor(h, dtype=torch.float32)
    if h.dim() == 1:
        h = h.unsqueeze(0)
    if h.dim() != 2 or h.shape[0] != len(texts):
        return _l18_fail(f"provider 形状不符 kind={kind} got={tuple(h.shape)} "
                         f"want=({len(texts)}, d)")
    return F.normalize(h, dim=-1).to("cpu")


# ----------------------------- 打分底座 -----------------------------
class _ScoreStore:
    """单元存储 + 三路打分（词面 IDF-MaxSim / 第 18 层稠密 / 时序加成）。"""

    def __init__(self):
        self.texts = []         # 原文（输出用）
        self.index_texts = []   # 打分文本（时间戳前缀 + 原文）
        self.metas = []         # 元数据（start/end/ts/year/...）
        self.tokv = []          # list[cpu bf16/fp32 [L,d]] 逐 token 归一化向量
        self.words = []         # list[list[str]] 与 token 对齐的词（IDF 用）
        self.mean = []          # list[cpu fp32 [d]]
        self.years = []         # list[float]（nan = 无时间戳）
        self.word_df = Counter()
        self.n_dup = 0
        self.l18 = None         # [N, d_l18] cpu fp32 归一化（稠密通道，未 whitening 的原始向量）
        self._l18_cache = None  # 池内 whitening 变换缓存：(Z, mean, W, info)；None = 需重算
        self._idf = None
        self._mean_mat = None

    def __len__(self):
        return len(self.texts)

    # ---- 写入 ----
    def _store(self, texts, index_texts, metas):
        """编码并追加单元。词级截断先于 span 记账（v8 Bug2 修复后的顺序）。"""
        keep = [(t, it, m) for t, it, m in zip(texts, index_texts, metas)
                if t and t.strip()]
        if not keep:
            return
        texts = [k[0] for k in keep]
        index_texts = [k[1] for k in keep]
        metas = [k[2] for k in keep]

        n_before = len(self)
        all_words, spans = [], []
        for it in index_texts:
            ws = _content_words(it)[:UNIT_MAX_WORDS]
            spans.append((len(all_words), len(all_words) + len(ws)))
            all_words.extend(ws)
        vecs_flat, counts = encode_words(all_words)
        off = 0
        for k, (text, it, meta) in enumerate(zip(texts, index_texts, metas)):
            s, e = spans[k]
            w_list = all_words[s:e]
            c_counts = counts[s:e]
            L = sum(c_counts)
            if L <= 0:
                # 零 token 兜底：vecs_flat 无对应行 -> 不推进 off
                tokv = torch.zeros((1, vecs_flat.shape[1]), dtype=torch.float32)
                w_list = [w_list[0] if w_list else "<empty>"]
            else:
                tokv = vecs_flat[off:off + L]
                off += L
            if TOKV_BF16 and tokv.dtype != torch.bfloat16:
                tokv = tokv.to(torch.bfloat16)
            self.texts.append(text)
            self.index_texts.append(it)
            self.metas.append(meta)
            self.tokv.append(tokv)
            self.words.append(w_list)
            self.mean.append(F.normalize(tokv.float().mean(dim=0), dim=0))
            year = meta.get("year")
            if year is None:
                year = extract_year(meta.get("ts"))
            self.years.append(float(year) if year is not None else float("nan"))
            for w in set(w_list):
                self.word_df[w] += 1
        self._idf = None
        self._mean_mat = None
        # 稠密通道：provider/预计算矩阵可用时按本批编码并追加；形状/同步不符则跳过（不破坏词面路径）
        self._l18_cache = None            # 新单元进来 -> whitening 变换必须重算
        if _l18_available():
            h = _call_l18(index_texts, "units", offset=n_before)
            if h is not None and h.shape[0] == len(index_texts):
                if self.l18 is None and n_before == 0:
                    self.l18 = h
                elif self.l18 is not None and self.l18.shape[0] == n_before:
                    self.l18 = torch.cat([self.l18, h], dim=0)
                elif L18_STRICT:
                    raise RuntimeError(
                        f"[v9/l18] 稠密通道与单元数不同步：l18={None if self.l18 is None else self.l18.shape[0]} "
                        f"n_before={n_before} batch={h.shape[0]}（L18_STRICT=1）")

    def attach_layer18(self, hidden):
        """直接注入单元侧第 18 层隐状态 [N, d]（N 必须等于当前单元数）。

        统一存 CPU fp32：与 _call_l18 的落盘一致，打分时再由 hybrid_scores 对齐设备。
        """
        h = torch.as_tensor(hidden, dtype=torch.float32)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if h.dim() != 2 or h.shape[0] != len(self):
            raise ValueError(f"layer18 shape {tuple(h.shape)} != ({len(self)}, d)")
        self.l18 = F.normalize(h, dim=-1).to("cpu")
        self._l18_cache = None            # 换了向量 -> whitening 变换必须重算

    # ---- 词面打分 ----
    def idf_table(self):
        if self._idf is None:
            n = max(len(self), 1)
            self._idf = {w: math.log(1.0 + n / (1.0 + df)) for w, df in self.word_df.items()}
        return self._idf

    def _query_repr(self, query):
        """query -> (qe [Lq,d] dev fp32, q_idf [Lq] dev)。"""
        dev = EMB.device if EMB is not None else torch.device("cpu")
        ws = _content_words(query)[:64]
        qv, counts = encode_words(ws)
        if qv.shape[0] == 0:                       # 空 query 兜底（避免 mean 出 nan）
            qv = torch.zeros((1, EMB.shape[1] if EMB is not None else 1), dtype=torch.float32)
            counts = [1]
            ws = ["<q>"]
        idf_t = self.idf_table()
        max_idf = max(idf_t.values()) if idf_t else 1.0
        w_list = []
        for w, c in zip(ws, counts):
            w_list.extend([w] * c)
        if len(w_list) != qv.shape[0]:             # 截断保护
            w_list = (w_list + [ws[0]] * qv.shape[0])[:qv.shape[0]]
        qw = torch.tensor([idf_t.get(w, max_idf if IDF_UNSEEN == 0 else IDF_UNSEEN)
                           for w in w_list], dtype=torch.float32, device=dev)
        qe = F.normalize(qv.to(dev), dim=-1)
        return qe, qw

    def lexical_scores(self, query):
        """全池 IDF-MaxSim：score(u) = Σ_t idf(q_t)·max_j cos(q_t, u_j) / Σ_t idf(q_t)。"""
        dev = EMB.device if EMB is not None else torch.device("cpu")
        n = len(self)
        qe, qw = self._query_repr(query)
        denom = qw.sum().clamp(min=1e-6)
        scores = torch.zeros(n, dtype=torch.float32, device=dev)
        for s in range(0, n, BATCH):
            e = min(s + BATCH, n)
            Ls = [self.tokv[i].shape[0] for i in range(s, e)]
            Lm = max(Ls)
            buf = torch.zeros((e - s, Lm, qe.shape[1]), dtype=torch.float32, device=dev)
            for k, i in enumerate(range(s, e)):
                buf[k, :Ls[k]] = self.tokv[i].to(device=dev, dtype=torch.float32)
            sim = torch.einsum("qd,bld->bql", qe, buf)      # [B, Lq, Lm]
            mx = sim.max(dim=2).values                      # [B, Lq]
            scores[s:e] = (mx * qw).sum(dim=1) / denom
        return scores

    # ---- 稠密打分（第 18 层，含池内 whitening） ----
    def l18_transform(self):
        """池内 whitening 变换（缓存）：返回 (Z [N,d], mean [d]|None, W [d,d]|None, info)。

        V9_L18_WHITEN=1（默认）：中心化 + 协方差白化（d>=N 等秩不足情形自动退化为只中心化）；
        V9_L18_WHITEN=0：mean/W 为 None，等价旧的纯 L2 归一化行为。
        Z 已按行 L2 归一化；query 侧必须用同一个 (mean, W) 变换后再算余弦。
        """
        if self.l18 is None:
            return None
        if self._l18_cache is not None:
            return self._l18_cache
        mean, W, info = l18_whiten_fit(self.l18)
        Z = l18_whiten_apply(self.l18, mean, W)
        self._l18_cache = (Z, mean, W, info)
        return self._l18_cache

    def dense_scores(self, query=None, q_hidden=None):
        """单元-query 的第 18 层隐状态余弦（池内 whitening 后）；未接稠密通道时返回 None。

        返回 **CPU** fp32 [N]（l18 常驻 CPU，262K 单元也不会额外占显存）。
        设备对齐由 hybrid_scores 统一完成：绝不让"词面分在 GPU、稠密分在 CPU"直接相加。
        """
        if self.l18 is None:
            return None
        if self.l18.shape[0] != len(self):
            return _l18_fail(f"稠密向量与单元数不同步：l18={self.l18.shape[0]} n={len(self)}")
        if q_hidden is None:
            if query is None:
                return None
            h = _call_l18([query], "query")
            if h is None:
                return None
            q_hidden = h[0]
        tr = self.l18_transform()
        Z, mean, W, _info = tr
        qh = torch.as_tensor(q_hidden, dtype=torch.float32)
        if qh.dim() > 1:
            qh = qh.mean(dim=0)
        # query 与单元必须过【同一个】变换（中心化 + 白化），否则余弦不在同一空间
        qh = l18_whiten_apply(qh.to(device=self.l18.device, dtype=torch.float32),
                              mean, W)[0]
        return (Z @ qh).clamp(-1.0, 1.0).to("cpu")

    def hybrid_scores(self, query, q_hidden=None, alpha=None):
        """混合打分：(1-alpha)·词面 + alpha·((稠密cos+1)/2)；无稠密通道 -> 纯词面。

        词面分在 EMB.device（通常是 GPU），稠密分在 CPU -> 这里显式把稠密分搬到
        词面分所在设备再相加。这是 v9 第 18 层混合检索设备不一致的修复点。
        """
        s = self.lexical_scores(query)
        d = self.dense_scores(query, q_hidden)
        if d is None:
            return s
        if d.shape[0] != s.shape[0]:
            if L18_STRICT:
                _l18_fail(f"稠密分长度 {d.shape[0]} != 词面分长度 {s.shape[0]}")
            return s
        d = d.to(device=s.device, dtype=s.dtype)
        a = L18_ALPHA if alpha is None else float(alpha)
        a = min(max(a, 0.0), 1.0)
        return (1.0 - a) * s + a * ((d + 1.0) * 0.5)

    # ---- 时序重排 ----
    def temporal_rerank(self, scores, query_ts=None, query_year=None):
        """按 |year_u - year_q| 指数衰减给分加成；无年份单元不加分也不罚分。"""
        yq = query_year if query_year is not None else extract_year(query_ts)
        if yq is None or not self.years:
            return scores
        y = torch.tensor(self.years, dtype=torch.float32, device=scores.device)
        known = ~torch.isnan(y)
        aff = torch.zeros_like(y)
        if known.any():
            aff[known] = torch.exp(-(y[known] - float(yq)).abs() / max(TEMPORAL_TAU, 1e-6))
        return scores * (1.0 + TEMPORAL_BONUS * aff)

    def score(self, query, query_ts=None, q_hidden=None, alpha=None, temporal=True):
        """最终打分张量 [N]（混合 + 可选时序加成）。"""
        s = self.hybrid_scores(query, q_hidden=q_hidden, alpha=alpha)
        if temporal:
            yq = extract_year(query_ts) if query_ts is not None else extract_year(query)
            if yq is not None:
                s = self.temporal_rerank(s, query_year=yq)
        return s

    # ---- 检索 ----
    def retrieve(self, query, top_k=12, query_ts=None, q_hidden=None,
                 alpha=None, temporal=True):
        """【先按分数选 Top-K】。返回 [(idx, score)] 分数降序，无位置排序。"""
        if len(self) == 0:
            return []
        scores = self.score(query, query_ts=query_ts, q_hidden=q_hidden,
                            alpha=alpha, temporal=temporal)
        k = min(int(top_k), len(self))
        vals, idxs = scores.topk(k)
        return [(int(i), float(v)) for i, v in zip(idxs.tolist(), vals.tolist())]

    def retrieve_ordered(self, query, top_k=12, **kw):
        """先按分数选 Top-K，再把【已选中的 K 个】按原文位置排序（不是全池排序后截断）。"""
        hits = self.retrieve(query, top_k=top_k, **kw)
        return sorted((i for i, _ in hits), key=lambda i: self.metas[i].get("start", i))

    def best_unit_for(self, text, query_ts=None, **kw):
        """时序锚定：定位某段文本（已发生事件）最匹配的单元。返回 (idx, score)。"""
        hits = self.retrieve(text, top_k=1, query_ts=query_ts, **kw)
        return hits[0] if hits else (-1, 0.0)

    def mean_mat(self):
        if self._mean_mat is None and self.mean:
            self._mean_mat = torch.stack(self.mean)
        return self._mean_mat

    def stats(self):
        L = [t.shape[0] for t in self.tokv]
        out = {"n": len(self),
               "n_ts": sum(1 for y in self.years if not math.isnan(y)),
               "mean_tokens": round(sum(L) / len(L), 2) if L else 0.0,
               "has_l18": self.l18 is not None,
               "n_dup": self.n_dup}
        if hasattr(self, "n_attr_rejected"):        # StatePool 的诊断计数
            out["n_merge_rejected"] = self.n_merge_rejected
            out["n_attr_rejected"] = self.n_attr_rejected
        return out


# ----------------------------- 情景索引 -----------------------------
class EpisodicIndex(_ScoreStore):
    """非重叠情景索引：单元按原文位置一一对应，检索分数优先、按 anchor 做窗口扩展
    （默认前 16 后 4；窗口方向未定，见 EQA_BACK/EQA_AHEAD 与 expand_backward 的注释）。"""

    def add_units(self, units):
        """units: list[dict]（extract_units 的产物）。按位置全量保留，不做文本去重
        （去重会让同一句只留一个位置，破坏位置排序与邻域扩展）。"""
        if not units:
            return
        texts = [u["text"] for u in units]
        index_texts = [u.get("index_text") or build_index_text(u.get("ts"), u["text"])
                       for u in units]
        self._store(texts, index_texts, list(units))

    def expand_backward(self, anchor_idx, back=None, ahead=None):
        """EventQA 窗口扩展：以 anchor 为界，向前 back 个、向后 ahead 个单元。

        默认 back=V9_EQA_BACK=16、ahead=V9_EQA_AHEAD=4（沿用旧默认，前多后少）。
        ⚠️ 默认值未经显著验证：配对检验 Holm p=0.2314，MDE=9.7pp，
        见 experiments/results/paired_window_test.json。
        窗口方向（前宽后窄 vs 前窄后宽）目前【未定】：b0_a20/b4_a16 vs b16_a4 的 +8.00pp
        在 Holm 校正后不显著，且观测效应 < MDE（power<0.80）；改默认值前需把 n 提到
        ≥148（8pp 量级）或换更敏感的探针（合成 needle / 带 span 金标）。
        非重叠索引下单元下标即原文顺序，返回按位置升序的连续下标。
        """
        if len(self) == 0:
            return []
        b = EQA_BACK if back is None else max(0, int(back))
        a = EQA_AHEAD if ahead is None else max(0, int(ahead))
        anchor_idx = int(anchor_idx)
        if anchor_idx < 0:
            return []
        lo = max(0, anchor_idx - b)
        hi = min(len(self) - 1, anchor_idx + a)
        return list(range(lo, hi + 1))

    def anchor_from_events(self, events, cov_threshold=0.5, query_ts=None, **kw):
        """EventQA 时序锚定：取最后一个已发生事件；覆盖率 < cov_threshold 时
        拼接最后两个事件重锚（与 v7/v8 的 anchor_used=last/last2 口径一致）。

        返回 (idx, cov, used)，used ∈ {"last", "last2", "none"}。
        """
        events = [e for e in (events or []) if e]
        if not events or len(self) == 0:
            return -1, 0.0, "none"
        idx, cov = self.best_unit_for(events[-1], query_ts=query_ts, **kw)
        used = "last"
        if cov < cov_threshold and len(events) > 1:
            idx2, cov2 = self.best_unit_for(events[-2] + " " + events[-1],
                                            query_ts=query_ts, **kw)
            if cov2 > cov:
                idx, cov, used = idx2, cov2, "last2"
        return idx, cov, used

    # ---- LongMemEval：Session/Turn 扩展 + 跨 Session 配额 + 时间约束 ----
    def expand_session_turns(self, idxs, window=None):
        """命中单元在【本 Session 内】前后各扩 window 个 turn；跨 Session 一律不扩。

        window 默认 V9_LME_TURN_WINDOW=1。非重叠索引下单元下标即原文顺序，
        Session 边界由 metas[i]["session_id"] 判定；缺 session_id 的老数据视为同一
        Session（退回"整池一个 Session"的历史行为，不报错）。
        """
        w = LME_TURN_WINDOW if window is None else max(0, int(window))
        keep = sorted({int(i) for i in (idxs or []) if 0 <= int(i) < len(self)})
        if w <= 0 or not keep:
            return keep
        out = set()
        for i in keep:
            sid = self.metas[i].get("session_id", 0)
            lo = i
            while lo - 1 >= 0 and i - lo < w and self.metas[lo - 1].get("session_id", 0) == sid:
                lo -= 1
            hi = i
            while (hi + 1 < len(self) and hi - i < w
                   and self.metas[hi + 1].get("session_id", 0) == sid):
                hi += 1
            out.update(range(lo, hi + 1))
        return sorted(out)

    def select_with_session_quota(self, hits, top_k=None, max_per_session=None):
        """跨 Session 多样性：同一 Session 最多取 max_per_session 个命中单元。

        hits 为分数降序的 [(idx, score)]；超出配额的命中【跳过并顺延】给后续命中
        （通常是别的 Session），不因配额少给名额。max_per_session<=0 视为不限。
        返回 (picked, info)。
        """
        cap = MAX_PER_SESSION if max_per_session is None else int(max_per_session)
        k = len(hits) if top_k is None else max(0, int(top_k))
        n_sessions = len({self.metas[int(i)].get("session_id", 0) for i, _ in hits})
        fallback = None
        if cap > 0 and n_sessions <= 1:
            # 候选全在同一 Session：配额只减召回、无多样性收益 -> 退回不限
            cap, fallback = 0, "single_session"
        picked, per, dropped = [], {}, 0
        for idx, sc in hits:
            sid = self.metas[int(idx)].get("session_id", 0)
            if cap > 0 and per.get(sid, 0) >= cap:
                dropped += 1
                continue
            per[sid] = per.get(sid, 0) + 1
            picked.append((int(idx), float(sc)))
            if len(picked) >= k:
                break
        info = {"max_per_session": cap, "n_selected": len(picked),
                "n_dropped": dropped, "n_sessions": len(per),
                "fallback": fallback,
                "per_session": {str(s): c for s, c in sorted(per.items())}}
        return picked, info

    def apply_temporal_constraint(self, hits, constraint, keep_fallback=True):
        """按 first/last/before/after 【过滤或重排】候选（不是仅加分）。

        before/after：硬过滤——无日期或日期不满足的候选剔除；若过滤后为空则退回原候选
                      并记 fallback="empty_after_filter"（不因约束丢掉全部证据）。
        first/last  ：重排——极值日期组排最前，其余按与该日期的距离，同距离按原文位置
                      （first 正序 / last 倒序）；无日期单元排最后（不编造顺序）。
        返回 (hits, info)；info 直接进 dbg["temporal_constraint"]。
        """
        info = {"kind": "none", "order": None, "operand": None, "date": None,
                "applied": False, "n_in": len(hits), "n_out": len(hits), "fallback": None}
        if not hits:
            return hits, info
        c = dict(constraint or {})
        kind, order = c.get("kind", "none"), c.get("order")
        target = None
        if c.get("date"):
            try:
                target = datetime.date.fromisoformat(str(c["date"]))
            except (ValueError, TypeError):
                target = None
        info.update(kind=kind if kind in ("before", "after") else "none",
                    order=order, operand=c.get("operand"), date=c.get("date"))
        if kind in ("before", "after") and target is not None:
            kept = []
            for idx, sc in hits:
                lo, hi, _p = unit_date_span(self.metas[int(idx)])
                if lo is None:
                    continue                      # 无日期单元无法判定 before/after -> 剔除
                if (lo < target) if kind == "before" else (hi > target):
                    kept.append((idx, sc))
            if kept:
                info.update(applied=True, n_out=len(kept))
                hits = kept
            else:
                info.update(fallback="empty_after_filter" if keep_fallback else "dropped_all")
                if not keep_fallback:
                    return [], info
        if order in ("first", "last"):
            dated = [(unit_date_center(self.metas[int(idx)]), pos, int(idx), sc)
                     for pos, (idx, sc) in enumerate(hits)]
            if any(d is not None for d, *_ in dated):
                sign = 1 if order == "first" else -1

                def _key(t):
                    d, pos, _idx, _sc = t
                    if d is None:
                        return (1, 0, pos)          # 无日期排最后，不编造顺序
                    return (0, sign * d.toordinal(), pos if order == "first" else -pos)

                hits = [(idx, sc) for _d, _p, idx, sc in sorted(dated, key=_key)]
                info.update(applied=True, n_out=len(hits))
        return hits, info

    def readout(self, query, top_k=12, mode="topk", anchor_idx=None, anchor_text=None,
                anchor_events=None,
                query_ts=None, q_hidden=None, alpha=None, temporal=True,
                back=None, ahead=None,
                session_quota=None, max_per_session=None, turn_window=None,
                temporal_constraint=None):
        """读出：返回 (idxs, dbg)，idxs 一律按原文位置升序。

        mode="topk"    : 先按分数选 Top-K，再把选中的 K 个按原文位置排序；
        mode="eventqa" : 定位锚点（anchor_idx 优先；否则 anchor_events 走
                         anchor_from_events 的 last/last2 逻辑；否则 anchor_text/query
                         的 best_unit_for），再做窗口扩展（back/ahead，默认 16/4，方向未定）。
        mode="lme"     : LongMemEval 专用读出 —— 分数优先取候选池
                         -> 时间约束（first/last/before/after，过滤或重排）
                         -> 跨 Session 配额（同 Session 最多 max_per_session，默认
                            V9_MAX_PER_SESSION=2，超出的按分数顺延）
                         -> Session 内 turn 扩展（前后各 turn_window，默认
                            V9_LME_TURN_WINDOW=1，跨 Session 不扩）。
                         配额与时间约束的细节都在 dbg["session_quota"] /
                         dbg["temporal_constraint"] 里（topk 模式也会带这两个字段，
                         值为"未启用"）。
        """
        if len(self) == 0:
            return [], {"mode": "empty", "n_units": 0}
        yq = extract_year(query_ts) if query_ts is not None else extract_year(query)
        if mode == "eventqa":
            cov = 0.0
            used = "given" if anchor_idx is not None else "none"
            if anchor_idx is None and anchor_events:
                anchor_idx, cov, used = self.anchor_from_events(
                    anchor_events, query_ts=query_ts, q_hidden=q_hidden,
                    alpha=alpha, temporal=temporal)
            if anchor_idx is None:
                anchor_idx, cov = self.best_unit_for(anchor_text or query,
                                                     query_ts=query_ts,
                                                     q_hidden=q_hidden,
                                                     alpha=alpha, temporal=temporal)
                used = "text" if anchor_text else "query"
            if int(anchor_idx) < 0:
                idxs = self.retrieve_ordered(query, top_k=top_k, query_ts=query_ts,
                                             q_hidden=q_hidden, alpha=alpha,
                                             temporal=temporal)
                return idxs, {"mode": "eventqa_fallback_topk", "idxs": idxs}
            b = EQA_BACK if back is None else int(back)
            a = EQA_AHEAD if ahead is None else int(ahead)
            idxs = self.expand_backward(anchor_idx, back=b, ahead=a)
            return idxs, {"mode": "eventqa_backward", "anchor": int(anchor_idx),
                          "anchor_cov": round(float(cov), 4), "anchor_used": used,
                          "back": b, "ahead": a, "idxs": idxs, "query_year": yq}
        if mode == "lme":
            # LongMemEval：分数候选池 -> 时间约束 -> 跨 Session 配额 -> Session 内 turn 扩展
            tc = (temporal_constraint if temporal_constraint is not None
                  else parse_temporal_constraint(query, query_ts=query_ts))
            pool_k = min(len(self), max(int(top_k) * 4, int(top_k) + 8))
            hits = self.retrieve(query, top_k=pool_k, query_ts=query_ts,
                                 q_hidden=q_hidden, alpha=alpha, temporal=temporal)
            hits, tc_info = self.apply_temporal_constraint(hits, tc)
            quota_on = LME_QUOTA if session_quota is None else bool(session_quota)
            cap = (MAX_PER_SESSION if max_per_session is None else int(max_per_session))
            if not quota_on:
                cap = 0                                   # 0 -> 不限
            picked, q_info = self.select_with_session_quota(hits, top_k=top_k,
                                                            max_per_session=cap)
            w = LME_TURN_WINDOW if turn_window is None else int(turn_window)
            sel = [i for i, _ in picked]
            idxs = self.expand_session_turns(sel, window=w) if w > 0 else sorted(sel)
            idxs = sorted(idxs, key=lambda i: self.metas[i].get("start", i))
            q_info.update(turn_window=w, n_hits=len(sel), n_expanded=len(idxs))
            return idxs, {"mode": "lme",
                          "hits": [[i, round(s, 4)] for i, s in picked],
                          "idxs": idxs, "query_year": yq,
                          "session_quota": q_info, "temporal_constraint": tc_info}
        hits = self.retrieve(query, top_k=top_k, query_ts=query_ts, q_hidden=q_hidden,
                             alpha=alpha, temporal=temporal)
        idxs = sorted((i for i, _ in hits), key=lambda i: self.metas[i].get("start", i))
        return idxs, {"mode": "topk_score_first",
                      "hits": [[i, round(s, 4)] for i, s in hits],
                      "idxs": idxs, "query_year": yq,
                      "session_quota": {"max_per_session": 0, "enabled": False},
                      "temporal_constraint": {"kind": "none", "applied": False}}


# ----------------------------- CR 双阈值（v9 CR 用） -----------------------------
# 谓词短语：长谓词优先，避免 "is" 抢先匹配 "is located in"。
CR_REL_THETA = float(os.environ.get("V9_CR_REL_THETA", "0.80"))
CR_VP_RE = re.compile(
    r"\b(?:is associated with|is affiliated with|is located in the continent of|"
    r"is located in the country of|was educated at|is educated at|"
    r"was created in|is created in|was created by|is created by|"
    r"was founded by|is founded by|is located in|was located in|"
    r"is famous for|was famous for|was written in|is written in|"
    r"was written by|is written by|was directed by|is directed by|"
    r"was born in|was born on|is married to|was married to|"
    r"is a citizen of|was a citizen of|held the position of|holds the position of|"
    r"is the capital of|is the language of|worked at|works at|worked for|works for|"
    r"plays|played|founded|wrote|is|was|are|were)\b",
    flags=re.IGNORECASE,
)


# 系表结构的谓词键：裸系词 + 表语头词。'Alice is a doctor.' 与 'Alice is vegetarian.'
# 若只留裸系词 'is'，双阈值必然通过 -> 第二条覆写第一条，职业/饮食等属性事实丢失。
CR_COPULA_ONLY_RE = re.compile(r"^(?:is|was|are|were|be|been|being)$", re.IGNORECASE)
CR_AUX_PREFIX_RE = re.compile(
    r"^(?:has been|have been|had been|is|was|are|were|be|been|being)\b", re.IGNORECASE)
CR_DET_PREFIX_RE = re.compile(r"^(?:a|an|the|his|her|their|its|my|your|our)\b", re.IGNORECASE)
# NP 头词在第一个介词/连词之前：'a doctor at City Hospital' -> 'doctor'
CR_PRED_BOUNDARY = {"of", "in", "on", "at", "for", "with", "from", "by", "near",
                    "during", "after", "before", "to", "about", "as", "that",
                    "who", "which", "and", "or"}
# 程度/时间副词不改变属性类别：'a very good doctor' -> 'doctor'
CR_PRED_SKIP = {"very", "really", "quite", "also", "still", "now", "currently",
                "former", "already", "just"}


def _predicative_head(pred):
    """表语短语 -> (冠词, 头词)：'a doctor at City Hospital' -> ('a', 'doctor')。

    'vegetarian' / 'happy' 这类单词表语冠词为空、头词就是自身。取不到头词时
    返回 ("", "")，调用方退回裸系词（不因解析失败而丢事实）。
    """
    words = [_norm_word(w) for w in WORD_RE.findall((pred or "").lower())]
    words = [w for w in words if w]
    head_part = []
    for w in words:
        if w in CR_PRED_BOUNDARY and head_part:
            break
        head_part.append(w)
    det = head_part[0] if head_part and head_part[0] in {"a", "an", "the"} else ""
    kept = [w for w in head_part if w not in STOPWORDS and w not in CR_PRED_SKIP]
    return det, (kept[-1] if kept else "")


def _cr_verb_core(vp):
    """动词短语去系词/助动词/冠词后的核心：'was born in' -> 'born in'。

    时态变体（is/was located in）因此落在同一个属性类别里，仍可原位覆写。
    """
    v = CR_AUX_PREFIX_RE.sub("", (vp or "").strip().lower()).strip()
    v = CR_DET_PREFIX_RE.sub("", v).strip()
    return v


def _cr_parse_parts(text):
    """-> (subject, verb_phrase, 表语冠词, 表语头词)。后两项只在裸系词结构时非空。"""
    f = (text or "").strip().rstrip(".")
    m = CR_VP_RE.search(f)
    if not m:
        toks = f.split()
        return ((" ".join(toks[:-2]) if len(toks) > 3 else f),
                (toks[-1] if toks else ""), "", "")
    subj = f[:m.start()].strip()
    vp = m.group(0).strip()
    pred = f[m.end():].strip()
    det, head = ("", "")
    if CR_COPULA_ONLY_RE.match(vp) and pred:
        det, head = _predicative_head(pred)
    return subj, vp, det, head


def _cr_parse_fact(text):
    """轻量 (subject, relation) 切分，仅供 CR 双阈值使用。

    不做 v7 的 WHERE/PREFIX 全解析：同一模板的冲突事实切分一致即可
    （主语用于精确匹配，谓词短语用于余弦比较）。v9_cr 的图检索用 v7 全解析。

    系表结构**必须**把表语头词并入谓词键：'Alice is a doctor.' -> ('Alice','is a doctor')，
    'Alice is vegetarian.' -> ('Alice','is vegetarian')。否则两条不同属性都是
    ('Alice','is')，双阈值必然通过，第二条覆写第一条（属性事实丢失）。
    其余动词短语保持原样：宾语是"值"（Paris/Lyon），进谓词键会让覆写失效。
    """
    subj, vp, det, head = _cr_parse_parts(text)
    if not head:
        return subj, vp
    return subj, " ".join(x for x in (vp, det, head) if x)


def _cr_attr_key(text):
    """属性类别键（覆写保护）：动词核心 + 表语头词。

    'Alice is a doctor.'        -> '|doctor'     （裸系词：头词就是属性名）
    'Alice is vegetarian.'      -> '|vegetarian'
    'Alice was born in Paris.'  -> 'born in'     （宾语是值，不进类别键）
    'Alice was born on Monday.' -> 'born on'     （不同属性：出生地 vs 生日）
    同类别才允许原位覆写；不同类别一律新增槽（属性并存，绝不互相覆盖）。
    空串表示"类别不可判定"，调用方按通配处理（不因解析失败丢覆写能力）。
    """
    _subj, vp, _det, head = _cr_parse_parts(text)
    core = _cr_verb_core(vp)
    return f"{core}|{head}" if head else core


def _cr_norm(s):
    s = (s or "").lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(w for w in s.split() if w not in STOPWORDS) or s.strip()


def _cr_key_vec(text):
    """谓词短语 -> 归一化均值向量；未注入 tok/EMB 时返回 None。"""
    if tok is None or EMB is None:
        return None
    ws = _content_words(text)
    if not ws:
        return None
    v, _c = encode_words(ws)
    if v.shape[0] == 0:
        return None
    return F.normalize(v.float().mean(dim=0), dim=0)


# ----------------------------- 状态池（可选补充 / CR 扩展点） -----------------------------
class StatePool(_ScoreStore):
    """稳定事实单元池：同主语+同关系的**原位覆写**（CR 状态更新语义）。

    覆写判据（修 P0「CR 覆写选不到正确的旧槽」）：
      1. 用 `_subj_index`（主语归一化串 -> 槽下标）定位**所有同主语槽**；
      2. 在其中挑关系谓词余弦最大者，过 `V9_CR_REL_THETA`（默认 0.80）才覆写；
      3. 同主语但**属性类别**不同 -> 新增槽（不同属性，不是冲突）；
         属性类别 = 动词核心 + 表语头词（'is a doctor' vs 'is vegetarian' 不同类，
         谓词余弦再高也不覆写）；同类属性才允许原位更新（'is a doctor' -> 'is a doctor'）；
      4. 没有同主语槽时才退回 v8 的 mean-cos 单阈值（`V9_STATE_THETA`）。
    旧实现只检查全池 argmax 槽的主语：若另一条事实整体更像，同主语的旧值就覆写不了，
    旧值与新值同时存活 —— 直接违背 CR 的"新值覆盖旧值"语义。

    覆写是**原位更新**（`_replace`）：槽 id 稳定，不 drop+append，
    因此 `metas`/`tokv`/`years`/CR 键/图键始终与槽下标一一对应。
    建池复杂度：每条事实 O(同主语槽数)，不再是 O(N) 全池比较。
    """

    def __init__(self, theta=None):
        super().__init__()
        self.theta = STATE_THETA if theta is None else float(theta)
        self.cr_subj = []      # 主语归一化字符串（与 mean 槽一一对应）
        self.cr_relk = []      # 关系谓词均值向量 [d]
        self.cr_attr = []      # 属性类别键（动词核心 + 表语头词；覆写保护用）
        self._subj_index = {}  # 主语归一化串 -> [槽下标]（升序）
        self.n_merge_checked = 0
        self.n_merge_rejected = 0
        self.n_attr_rejected = 0   # 因"属性类别不同"被拒绝覆写的候选数

    def _store(self, texts, index_texts, metas):
        keep = [(t, it, m) for t, it, m in zip(texts, index_texts, metas)
                if t and t.strip()]
        n_before = len(self)
        super()._store(texts, index_texts, metas)
        for t, _it, _m in keep:
            subj, rel = _cr_parse_fact(t)
            self.cr_subj.append(_cr_norm(subj))
            self.cr_relk.append(_cr_key_vec(rel))
            self.cr_attr.append(_cr_attr_key(t))
        for j in range(n_before, len(self.cr_subj)):
            self._subj_index.setdefault(self.cr_subj[j], []).append(j)

    def _rebuild_subj_index(self):
        idx = {}
        for i, s in enumerate(self.cr_subj):
            idx.setdefault(s, []).append(i)
        self._subj_index = idx

    def _reindex_subject(self, old, j, new):
        """槽 j 的主语从 old 迁到 new（原位覆写时主语可能变）。"""
        lst = [i for i in self._subj_index.get(old, []) if i != j]
        if lst:
            self._subj_index[old] = lst
        else:
            self._subj_index.pop(old, None)
        if new:
            self._subj_index.setdefault(new, []).append(j)

    def _pending_item(self, text, meta, qm):
        """待写入项：(text, meta, mean_vec, 主语归一化串, 关系键向量, 属性类别键)。"""
        subj, rel = _cr_parse_fact(text)
        return (text, meta, qm, _cr_norm(subj), _cr_key_vec(rel), _cr_attr_key(text))

    def _mean_cos_target(self, qm, pending=None):
        """v8 兼容路径：全池 mean-cos 单阈值（只在主语键不可用时使用）。"""
        base = None
        if self.mean:
            base = torch.stack([m.cpu() for m in self.mean])
        if pending:
            pm = torch.stack([p[2] for p in pending])
            base = pm if base is None else torch.cat([base, pm])
        if base is None:
            return None
        sims = base @ qm
        j = int(sims.argmax())
        if float(sims[j]) <= self.theta:
            return None
        n = len(self.mean)
        return ("stored", j) if j < n else ("pending", j - n)

    def _merge_target(self, text, meta, qm, pending=None):
        """定位应被原位覆写的槽；返回 ("stored"|"pending", idx) 或 None。

        同主语槽里取关系谓词余弦最大者（谓词键缺失时退回该槽的 mean-cos），
        过阈值才覆写。**不再**因为"别的槽整体更相似"而放弃覆写。
        """
        self.n_merge_checked += 1
        subj, rel = _cr_parse_fact(text)
        s_norm = _cr_norm(subj)
        if not s_norm:                              # 主语不可用 -> v8 兼容路径
            return self._mean_cos_target(qm, pending)
        k_r = _cr_key_vec(rel)
        a_new = _cr_attr_key(text)                  # 属性类别：不同类不得互相覆写
        best = None                                 # (score, kind, idx, threshold)
        for j in self._subj_index.get(s_norm, ()):
            if not self._attr_compatible(a_new, j):
                continue
            if k_r is not None and self.cr_relk[j] is not None:
                sc, thr = float(self.cr_relk[j] @ k_r), CR_REL_THETA
            else:
                sc, thr = float(self.mean[j] @ qm), self.theta
            if best is None or sc > best[0]:
                best = (sc, "stored", j, thr)
        for k, p in enumerate(pending or ()):
            if p[3] != s_norm:
                continue
            if not self._attr_compatible(a_new, None, p[5] if len(p) > 5 else ""):
                continue
            if k_r is not None and p[4] is not None:
                sc, thr = float(p[4] @ k_r), CR_REL_THETA
            else:
                sc, thr = float(p[2] @ qm), self.theta
            if best is None or sc > best[0]:
                best = (sc, "pending", k, thr)
        if best is None:                            # 无同主语槽 -> 新状态
            return None
        sc, kind, idx, thr = best
        if sc > thr:
            return (kind, idx)
        self.n_merge_rejected += 1
        return None

    def _attr_compatible(self, a_new, j=None, a_old=None):
        """属性类别保护：不同类别的属性不得互相覆写（只允许同类原位更新）。

        'Alice is a doctor.' 与 'Alice is vegetarian.' 主语相同、谓词向量也可能相近，
        但属性类别不同（doctor vs vegetarian）——若允许覆写，后一条会吃掉前一条。
        类别不可判定（空串）时按通配处理，退回余弦判定（不因解析失败丢覆写能力）。

        取舍：粒度变体（'is located in' vs 'is located in the country of'）被视作
        不同属性类别（国家 vs 洲/地区确实是不同属性）；若真实数据需要合并，用
        `n_attr_rejected` 诊断后按需在 `_cr_verb_core` 里加归一。
        """
        if a_old is None:
            a_old = self.cr_attr[j] if j is not None and j < len(self.cr_attr) else ""
        if not a_new or not a_old or a_new == a_old:
            return True
        # 属性类别拒绝也是"覆写被拒"：n_merge_rejected 记候选拒绝总数，
        # n_attr_rejected 记其中因属性类别不同的子集（诊断可分辨原因）。
        self.n_attr_rejected += 1
        self.n_merge_rejected += 1
        return False

    def _should_merge(self, text, meta, qm, base):
        """兼容旧签名（v9 早期调用/测试）：是否应合并。`base` 参数已不再使用。

        真正的目标槽定位走 `_merge_target`；此处只做 bool 适配。
        """
        return self._merge_target(text, meta, qm) is not None

    def _drop(self, j):
        """回收第 j 个槽（df/tokv/words/mean/years 同步），仅供容量淘汰使用。

        注意：CR 覆写不再走这里（改用 `_replace` 原位更新），槽 id 因此保持稳定。
        """
        if j < len(self.cr_subj):
            self.cr_subj.pop(j)
            self.cr_relk.pop(j)
        if j < len(self.cr_attr):
            self.cr_attr.pop(j)
        for w in set(self.words[j]):
            self.word_df[w] -= 1
            if self.word_df[w] <= 0:
                self.word_df.pop(w, None)
        for lst in (self.texts, self.index_texts, self.metas, self.tokv,
                    self.words, self.mean, self.years):
            lst.pop(j)
        self._l18_cache = None            # 删槽后 whitening 变换必须重算
        if self.l18 is not None and self.l18.shape[0] == len(self):
            self.l18 = (torch.cat([self.l18[:j], self.l18[j + 1:]], dim=0)
                        if self.l18.shape[0] > 1 else None)
        self._rebuild_subj_index()
        self._idf = None
        self._mean_mat = None

    def _replace(self, j, text, meta):
        """原位覆写第 j 个槽（P1：CR 覆写保持槽 id 稳定，不 drop+append）。

        同步：原文/打分文本/元数据/token 向量/词表/均值/年份/df/CR 键/稠密向量。
        `metas[j]` 会被新事实的 meta 覆盖（serial 因此变成新值，槽位置不变）。
        """
        it = build_index_text(meta.get("ts"), text)
        ws = _content_words(it)[:UNIT_MAX_WORDS]
        vecs, counts = encode_words(ws)
        L = int(sum(counts))
        if L <= 0:
            dim = vecs.shape[1] if vecs.numel() else (EMB.shape[1] if EMB is not None else 1)
            tokv = torch.zeros((1, dim), dtype=torch.float32)
            ws = [ws[0] if ws else "<empty>"]
        else:
            tokv = vecs[:L].float()
        if TOKV_BF16 and tokv.dtype != torch.bfloat16:
            tokv = tokv.to(torch.bfloat16)

        for w in set(self.words[j]):              # df：先减旧词，再加新词
            self.word_df[w] -= 1
            if self.word_df[w] <= 0:
                self.word_df.pop(w, None)
        for w in set(ws):
            self.word_df[w] += 1

        self.texts[j] = text
        self.index_texts[j] = it
        self.metas[j] = meta
        self.tokv[j] = tokv
        self.words[j] = ws
        self.mean[j] = F.normalize(tokv.float().mean(dim=0), dim=0)
        year = meta.get("year")
        if year is None:
            year = extract_year(meta.get("ts"))
        self.years[j] = float(year) if year is not None else float("nan")

        subj, rel = _cr_parse_fact(text)
        s_new = _cr_norm(subj)
        s_old = self.cr_subj[j] if j < len(self.cr_subj) else None
        if j < len(self.cr_subj):
            self.cr_subj[j] = s_new
            self.cr_relk[j] = _cr_key_vec(rel)
            if j < len(self.cr_attr):
                self.cr_attr[j] = _cr_attr_key(text)
            if s_old != s_new:
                self._reindex_subject(s_old, j, s_new)
        self._idf = None
        self._mean_mat = None
        self._l18_cache = None            # 槽位向量变了 -> whitening 变换必须重算

        if self.l18 is not None and _l18_available():
            h = _call_l18([it], "units", offset=j)
            if h is not None and h.shape[0] == 1:
                self.l18[j] = h[0]

    def _mean_only(self, index_text):
        dev = EMB.device if EMB is not None else torch.device("cpu")
        ws = _content_words(index_text)[:UNIT_MAX_WORDS]
        qv, _c = encode_words(ws)
        if qv.shape[0] == 0:
            return torch.zeros((EMB.shape[1] if EMB is not None else 1,),
                               dtype=torch.float32, device=dev)
        qe = F.normalize(qv.to(dev), dim=-1)
        return F.normalize(qe.mean(dim=0), dim=0)

    def add_batch(self, units):
        """units: list[(text, meta)]。命中同主语+同关系（或 mean-cos 单阈值）则原位覆写。

        每条事实的候选定位是 O(同主语槽数)：不再为每条事实 stack 全池均值矩阵。
        """
        pending = []
        for text, meta in units:
            if not text or not text.strip():
                continue
            it = build_index_text(meta.get("ts"), text)
            qm = self._mean_only(it).cpu()
            target = self._merge_target(text, meta, qm, pending)
            if target is None:
                pending.append(self._pending_item(text, meta, qm))
                continue
            kind, idx = target
            if kind == "stored":
                self._replace(idx, text, meta)      # 原位覆写：槽 id 不变
            else:
                pending[idx] = self._pending_item(text, meta, qm)
            self.n_dup += 1
        if not pending:
            return
        texts = [p[0] for p in pending]
        metas = [p[1] for p in pending]
        index_texts = [build_index_text(m.get("ts"), t) for t, m in zip(texts, metas)]
        self._store(texts, index_texts, metas)


# ----------------------------- 组装入口 -----------------------------
def build_index(ctx, block_sents=None):
    """只建情景索引（v9 AR 主路径）。"""
    epi = EpisodicIndex()
    epi.add_units(extract_units(ctx, block_sents=block_sents))
    return epi


def build_pools_for_context(ctx, block_sents=None):
    """建 (情景索引, 状态池)。状态池只收无事件动词、无显式时间戳的稳定陈述句，
    供 AR 读出可选补充；CR 路径不依赖本函数。

    注意：v9 情景索引是非重叠全覆盖（每句恰好一个单元），因此状态单元与情景单元
    文本重叠。AR v9 主路径只用 epi；若拼入状态池补充，必须按文本去重
    （v8 的 build_block_v8 已有 seen 去重，可直接复用）。
    """
    units = extract_units(ctx, block_sents=block_sents)
    epi = EpisodicIndex()
    epi.add_units(units)
    # 复用（t10）：v9 情景索引是全覆盖的，状态单元文本必然与某些 epi 单元重叠，
    # 因此把 epi 已经算好的第 18 层向量按**文本**登记为预计算矩阵 —— 状态池建池时
    # 直接查表命中，不再对同一段文本二次前向；查不到的文本仍走 provider（正确性优先）。
    # V9_L18_REUSE_EPI=0 可关闭，用于"有没有省这次前向"的消融对照。
    if L18_REUSE_EPI and epi.l18 is not None and L18_PRECOMPUTED is None:
        set_layer18_precomputed(epi.l18, texts=epi.index_texts)
    st = StatePool()
    st.add_batch([(u["text"], u) for u in units
                  if (not u["is_event"]) and (not u["ts_explicit"])])
    return epi, st
