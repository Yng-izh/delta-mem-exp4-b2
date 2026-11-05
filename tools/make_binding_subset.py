#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''tools/make_binding_subset.py — 从 HotpotQA 验证集里划分「实体—关系—属性值绑定」子集。

为什么需要这个子集
------------------
下一阶段实验要回答：连续潜向量能不能比「原样保留 token」更好地保存
实体—关系—属性值绑定。这个比较不能只看总 EM —— 大量单实体桥接题靠语言先验
就能答对，会把绑定能力上的差异稀释掉。所以先对现有 HotpotQA 验证集做启发式
分层：哪些题必须把答案绑到「正确的那一个实体 / 那一条关系」上。

本脚本不新建数据集，只对现有 jsonl 打标，输出逐题标签（--out-labels）与可选的
原样子集（--out-subset）。三个风险标记互相独立，任一触发即算绑定题。

三个启发式的动机（细节见各函数注释）
------------------------------------
1. entity_swap_risk：问题里同时出现 >=2 个实体片段，且答案是实体。
   模型必须在多个实体之间选对那一个，这是「绑定」最直接的最小测试。
2. similar_entity_risk：上下文里存在同名或近似名的实体（非金标 anchor 与金标
   anchor 相同/互为子串，或两个金标 anchor 共享同一个词）。这类题词面匹配就能
   蒙对一部分，但真正要答对仍然要求把关系绑到具体实体上。
3. value_confusion_risk：答案是数值/日期，而上下文里有 >=2 句同类值。
   数值和日期本身没有语义锚点，只能靠绑定到正确的关系槽位来消歧。

答案类型一律复用 eval_latent_table.answer_type，不在这里另写一份：评测脚本与
本脚本必须用同一个口径，否则分层统计和总 EM 对不上。

用法
----
    python tools/make_binding_subset.py --data synth/hotpot_val500.jsonl \
        --out-labels results/binding_labels.json \
        --out-subset synth/hotpot_val500_binding.jsonl

    # 小样本调试 / 内置自测
    python tools/make_binding_subset.py --data synth/hotpot_val500.jsonl --limit 20 \
        --out-labels /tmp/lab.json
    python tools/make_binding_subset.py --self-test

除复用仓库内的 eval_latent_table 外只用标准库，CPU 秒级。
'''
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 复用仓库里的答案类型判定（返回 yesno/date/number/entity/empty）。
# eval_latent_table 会 import torch；本地没有 torch 时这里降级为 None，
# 纯启发式函数仍可单测，但真正跑数据会给出明确报错（服务器上一定有 torch）。
_EV_ERR = None
try:
    import eval_latent_table as EV
    answer_type = EV.answer_type
    item_hops = EV.item_hops
except (ImportError, ModuleNotFoundError) as _exc:  # pragma: no cover - 依赖环境
    EV = None
    answer_type = None
    item_hops = None
    _EV_ERR = _exc

KINDS = ('entity_swap_risk', 'similar_entity_risk', 'value_confusion_risk')


# =====================================================================
# 1. entity_swap_risk：问题里 >=2 个实体片段，且答案是实体
# =====================================================================
# 英文问句的句首词必然大写，如果只数「大写开头的词」，每个问题都会被判成
# 多实体（Which/What/Who 全算进去），标记就退化成恒真。所以这里做两件事：
#   (a) 把连续的大写词合并成一个「片段」（Arthur's Magazine 是一个实体，
#       不是两个词），标点/小写词处断开；
#   (b) 过滤掉疑问词、功能词、序数词等不可能独立充当实体的片段。
_TOKEN_RE = re.compile(r'''[A-Za-z][A-Za-z0-9'’.\-&]*''')

_NON_ENTITY_CAP = frozenset('''
which what who whom whose when where why how
is are was were be been being am
do does did done
has have had
can could will would shall should may might must
the a an this that these those there here
in on at of for to from by with without into over under between among
during after before since until while about across against along around
and or but if then than as so because
both also either neither
it its he she they we you i his her their our your my me him them us
first second third last next
many much more most few several some any all no not yes
'''.split())


def question_entity_spans(question):
    '''把问题切成「大写开头的实体片段」列表（已去重、已过滤功能词）。

    例：Which magazine was started first, Arthur's Magazine or First for Women?
        -> ['Arthur's Magazine', 'Women']（First 是序数词被过滤）
    '''
    q = str(question or '')
    spans = []
    cur = []
    prev_end = None
    for m in _TOKEN_RE.finditer(q):
        # 两个 token 之间只隔空白/连字符才属于同一个片段；夹逗号、括号等标点就断开。
        if prev_end is not None and not re.fullmatch(r'''[\s\-]+''', q[prev_end:m.start()]):
            if cur:
                spans.append(cur)
            cur = []
        tok = m.group(0)
        if tok[:1].isupper():
            cur.append(tok)
        else:
            if cur:
                spans.append(cur)
            cur = []
        prev_end = m.end()
    if cur:
        spans.append(cur)

    out = []
    seen = set()
    for words in spans:
        # 整个片段都是功能词（如 The / First）就不算实体；The Beatles 仍保留。
        if all(w.lower().strip('.,&') in _NON_ENTITY_CAP for w in words):
            continue
        # 单字母片段（I / A）不构成可区分的实体。
        if max(len(w) for w in words) < 2:
            continue
        key = ' '.join(w.lower() for w in words)
        if key in seen:
            continue
        seen.add(key)
        out.append(' '.join(words))
    return out


def entity_swap_risk(question, ans_type):
    '''问题同时点名 >=2 个实体、且答案本身是实体 -> 存在实体—关系交换风险。

    为什么要求答案是实体：问「哪本杂志先创刊」时答案若是 yes/no 或年份，
    多实体只是比较/定位条件，不构成「把属性值绑到哪个实体」的风险。
    '''
    if ans_type != 'entity':
        return False
    return len(question_entity_spans(question)) >= 2


# =====================================================================
# 2. similar_entity_risk：同名 / 近似名实体
# =====================================================================
# 为什么用「互为子串」而不是编辑距离：HotpotQA 的 anchor 是维基段落标题，
# 相似实体在标题上的表现几乎总是包含关系（Ed Wood / Ed Wood (film)、
# Wood / Ed Wood），子串是零成本且可解释的判据；编辑距离在短标题上会把
# 「Tim Burton / Tom Burton」这类不同实体也判相似。真正的别名（Ed Wood 与
# Edward Davis Wood Jr.）词面不包含，用「共享同一个词」补一条规则。
_ANCHOR_TOKEN_RE = re.compile(r'''[a-z0-9']+''')

# 共享词里排除纯功能词与人名后缀，否则 the/of/jr 会让任意两个标题「相似」。
_STOP_TOKENS = frozenset('''
the a an of and or for to in on at by with from as is are was were be been being
this that these those it its his her their our your my he she they we you i
jr sr ii iii iv st dr mr mrs ms sir
'''.split())


def _anchor_norm(x):
    return ' '.join(str(x or '').lower().split())


def _anchor_tokens(anchor):
    '''anchor -> 有区分度的词集合（去所有格、去功能词、去纯数字）。'''
    out = set()
    for t in _ANCHOR_TOKEN_RE.findall(anchor):
        t = t.strip('\'')
        if t.endswith('\'s'):
            t = t[:-2]
        if len(t) < 2:
            continue
        if not any(ch.isalpha() for ch in t):
            continue
        if t in _STOP_TOKENS:
            continue
        out.add(t)
    return out


def _unit_anchors(unit_meta):
    '''返回 (金标 anchor 列表, 非金标 anchor 列表)，均取规范化后的非空值。'''
    gold, other = [], []
    for m in unit_meta or []:
        if not isinstance(m, dict):
            continue
        a = _anchor_norm(m.get('anchor') or m.get('title') or '')
        if not a:
            continue
        if m.get('is_gold'):
            gold.append(a)
        else:
            other.append(a)
    return gold, other


def similar_entity_risk(unit_meta):
    '''上下文里存在同名/近似名实体时返回 True。

    规则 A（跨金标/干扰）：非金标 anchor 与某个金标 anchor 相同，或互为子串。
      同名 anchor 直接说明「同一个实体在上下文里出现了两次」，模型必须靠句子
      内容而不是名字来区分哪句是金标证据。
      子串匹配加 min(len) >= 2 的下限：单字符 anchor（如 A）会命中几乎所有
      标题，纯噪声；两字符（Ed）仍保留，因为 Ed / Ed Wood 是真实别名形态。
    规则 B（金标内部）：>=2 个不同的金标 anchor 共享同一个有区分度的词，
      例如 Ed Wood 与 Edward Davis Wood Jr. 共享 wood —— 词面不同但指向同一
      实体的不同写法，答案必须绑到具体那一段。
    '''
    gold, other = _unit_anchors(unit_meta)

    # 规则 A：干扰单元与金标单元同名/近似。
    for d in other:
        for g in gold:
            if d == g:
                return True
            if min(len(d), len(g)) >= 2 and (d in g or g in d):
                return True

    # 规则 B：金标 anchor 之间共享有区分度的词。
    uniq = sorted(set(gold))
    toks = {a: _anchor_tokens(a) for a in uniq}
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            if toks[uniq[i]] & toks[uniq[j]]:
                return True
    return False


# =====================================================================
# 3. value_confusion_risk：数值/日期答案 + 上下文多个同类值
# =====================================================================
_YEAR_RE = re.compile(r'''(?<![\d.])(?:1[0-9]{3}|20[0-9]{2})(?![\d])''')
_NUM_RE = re.compile(r'''(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])''')
# 月份只认首字母大写形式：小写 may 在英文里是情态动词，会被大量误判成日期。
_MONTH_RE = re.compile(
    r'''\b(?:January|February|March|April|May|June|July|August|September|October|'''
    r'''November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b''')
_ORD_DATE_RE = re.compile(r'''\b\d{1,2}(?:st|nd|rd|th)\b''', re.I)
_SLASH_DATE_RE = re.compile(r'''\b\d{1,2}/\d{1,2}/\d{2,4}\b''')


def _has_number(text):
    '''句子里是否有「数值」候选。四位年份（1000~2099）不算数值：
    答案类型是 number 时（如 3 / 1.5 / 3,000），上下文里的 1844 是日期候选，
    不是同类值；把它算进来会让几乎所有带年份的句子都变成数值干扰。'''
    for m in _NUM_RE.finditer(text):
        tok = m.group(0).replace(',', '')
        if re.fullmatch(r'''\d{4}''', tok) and 1000 <= int(tok) <= 2099:
            continue
        return True
    return False


def _has_date(text):
    return bool(_YEAR_RE.search(text) or _MONTH_RE.search(text)
                or _ORD_DATE_RE.search(text) or _SLASH_DATE_RE.search(text))


def value_confusion_risk(units, ans_type):
    '''答案是数值/日期，且 units 里有 >=2 句包含同类值时返回 True。

    为什么要按「句子数」而不是「不同值的个数」：HotpotQA 的干扰段落经常整段
    复述同一组年份/数字，不同值可能只有 1 个，但模型仍要在多句同类表述里选出
    与问题关系对应的那一句，这才是属性值混淆的来源。
    '''
    if ans_type == 'date':
        hit = _has_date
    elif ans_type == 'number':
        hit = _has_number
    else:
        return False
    n = 0
    for u in units or []:
        if hit(str(u or '')):
            n += 1
            if n >= 2:
                return True
    return False


# =====================================================================
# 单题判定与数据读写
# =====================================================================
def _ans_type(item):
    if answer_type is None:
        raise RuntimeError(
            'answer_type 不可用：需要仓库根目录可导入 eval_latent_table（依赖 torch）。'
            '原始错误：{0}'.format(_EV_ERR))
    return answer_type(item.get('answer'))


def count_gold(item):
    '''金标句数：优先数 unit_meta.is_gold；数据没有该字段时退回 evidence 长度。'''
    um = item.get('unit_meta') or []
    if any(isinstance(m, dict) and 'is_gold' in m for m in um):
        return sum(1 for m in um if isinstance(m, dict) and m.get('is_gold'))
    return len(item.get('evidence') or [])


def _hops(item):
    if item_hops is not None:
        return int(item_hops(item))
    try:
        return max(1, int(item.get('hops') or 1))
    except (TypeError, ValueError):
        return 1


def judge(item):
    '''对一条样本给出 {binding, kinds, ans_type, hops, n_gold}。'''
    at = _ans_type(item)
    flags = {
        'entity_swap_risk': entity_swap_risk(item.get('question'), at),
        'similar_entity_risk': similar_entity_risk(item.get('unit_meta')),
        'value_confusion_risk': value_confusion_risk(item.get('units'), at),
    }
    kinds = [k for k in KINDS if flags[k]]
    return {
        'binding': bool(kinds),
        'kinds': kinds,
        'ans_type': at,
        'hops': _hops(item),
        'n_gold': count_gold(item),
    }


def read_jsonl(path, limit=0):
    '''读 jsonl，返回 [(物理行号, 原始行, 解析后的 dict)]。

    行号用文件里的 0 基物理行号（跳过空行但不重置计数），与 --out-labels 的
    idx 定义一致，这样即使数据里有空行也能和原文件对得上。
    '''
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f):
            raw = line.rstrip('\n')
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit('第 {0} 行不是合法 JSON：{1}'.format(lineno, exc))
            rows.append((lineno, raw, item))
            if limit and len(rows) >= limit:
                break
    return rows


def _ensure_parent(path):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d)


def _write_json(path, obj):
    _ensure_parent(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def _write_lines(path, lines):
    _ensure_parent(path)
    with open(path, 'w', encoding='utf-8') as f:
        for ln in lines:
            f.write(ln)
            f.write('\n')


def _print_table(data_path, n, n_binding, kind_counts, cross_n, cross_b):
    def pct(c, d):
        return (100.0 * c / d) if d else 0.0

    print('=' * 72)
    print('关系绑定子集：{0}'.format(data_path))
    print('总题数 {0}   绑定题数 {1} ({2:.1f}%)'.format(n, n_binding, pct(n_binding, n)))
    for k in KINDS:
        c = kind_counts.get(k, 0)
        print('  {0:<22s}{1:6d}  ({2:5.1f}%)'.format(k, c, pct(c, n)))
    print('  注：一条题可同时触发多种风险，因此三项之和 >= 绑定题数')
    print('-' * 72)
    print('按答案类型 × 跳数：')
    print('  {0:<10s}{1:>5s}{2:>7s}{3:>9s}{4:>8s}'.format(
        'ans_type', 'hops', 'n', 'binding', 'ratio'))
    order = {t: i for i, t in enumerate(('entity', 'number', 'date', 'yesno', 'empty'))}
    for key in sorted(cross_n, key=lambda k: (order.get(k[0], 99), str(k[0]), k[1])):
        nb = cross_b.get(key, 0)
        print('  {0:<10s}{1:>5d}{2:>7d}{3:>9d}{4:>7.1f}%'.format(
            key[0], key[1], cross_n[key], nb, pct(nb, cross_n[key])))
    print('=' * 72)


def run(data_path, limit=0, out_labels=None, out_subset=None, min_ratio=0.0, seed=0):
    '''主流程：打标 -> 写 labels/子集 -> 打印小表 -> 可选的比例下限检查。'''
    # seed 目前不参与判定：判定是确定性的，子集不抽样。保留参数只为和仓库里
    # 其它划分脚本的 CLI 保持一致，将来若改成分层抽样可直接接上。
    _ = seed
    rows = read_jsonl(data_path, limit=limit)
    items = {}
    subset_lines = []
    kind_counts = Counter()
    cross_n = Counter()
    cross_b = Counter()
    n_binding = 0
    for idx, raw, item in rows:
        rec = judge(item)
        items[str(idx)] = rec
        if rec['binding']:
            n_binding += 1
            subset_lines.append(raw)
        for k in rec['kinds']:
            kind_counts[k] += 1
        key = (rec['ans_type'], rec['hops'])
        cross_n[key] += 1
        if rec['binding']:
            cross_b[key] += 1

    n = len(rows)
    labels = {
        'n': n,
        'n_binding': n_binding,
        'n_by_kind': {k: int(kind_counts.get(k, 0)) for k in KINDS},
        'items': items,
    }
    if out_labels:
        _write_json(out_labels, labels)
    if out_subset:
        _write_lines(out_subset, subset_lines)

    _print_table(data_path, n, n_binding, kind_counts, cross_n, cross_b)
    if out_labels:
        print('labels -> {0}'.format(out_labels))
    if out_subset:
        print('subset -> {0}（{1} 条，原样复制）'.format(out_subset, len(subset_lines)))

    ratio = (n_binding / n) if n else 0.0
    if min_ratio > 0 and ratio < min_ratio:
        print('[WARN] 绑定比例 {0:.1f}% 低于 --min-ratio {1:.1f}%：'
              '判定规则或数据可能有问题（输出已写出，便于排查）'.format(
                  100.0 * ratio, 100.0 * min_ratio), file=sys.stderr)
        return 2
    return 0


# =====================================================================
# 内置自测：3 条合成样本（2 条应判为 binding）
# =====================================================================
def _sample(question, answer, rows, hops=2, qtype='bridge'):
    '''rows: [(句子, 是否金标, anchor)]。'''
    units = [r[0] for r in rows]
    unit_meta = [{'anchor': r[2], 'title': r[2], 'is_gold': bool(r[1]), 'fact': r[0]}
                 for r in rows]
    evidence = [r[0] for r in rows if r[1]]
    return {
        'units': units,
        'unit_meta': unit_meta,
        'evidence': evidence,
        'question': question,
        'answer': answer,
        'meta': {'type': qtype, 'qid': 'self-test'},
        'hops': hops,
    }


_SELF_TEST_CASES = [
    ('entity_swap', _sample(
        'Which magazine was started first, Arthur\'s Magazine or First for Women?',
        'Arthur\'s Magazine',
        [('Arthur\'s Magazine was an American literary periodical.', True,
          'Arthur\'s Magazine'),
         ('First for Women is a women\'s magazine.', True, 'First for Women'),
         ('Arthur\'s Magazine was started in 1844 in Philadelphia.', True,
          'Arthur\'s Magazine'),
         ('Vogue is a fashion magazine founded in New York.', False, 'Vogue'),
         ('Elle was founded in 1945 in France.', False, 'Elle')]),
     ['entity_swap_risk']),
    ('value_confusion', _sample(
        'When was Arthur\'s Magazine started?',
        '1844',
        [('Arthur\'s Magazine was an American literary periodical.', True,
          'Arthur\'s Magazine'),
         ('Arthur\'s Magazine was started in 1844 in Philadelphia.', True,
          'Arthur\'s Magazine'),
         ('Vogue was founded in 1892 in New York.', False, 'Vogue'),
         ('Elle was founded in 1945 in France.', False, 'Elle'),
         ('The magazine industry grew rapidly in the 20th century.', False,
          'Magazine industry')]),
     ['value_confusion_risk']),
    ('negative', _sample(
        'Who wrote the novel Ed Wood?',
        'Rudolph Grey',
        [('Rudolph Grey wrote the novel Ed Wood.', True, 'Rudolph Grey'),
         ('Ed Wood is a 1994 American biographical film.', False, 'Ed Wood'),
         ('Tim Burton directed the film.', False, 'Tim Burton')]),
     []),
]


def self_test():
    '''跑内置样本并断言判定结果；返回 0 表示通过，非 0 表示失败。'''
    if answer_type is None:
        print('[FAIL] answer_type 不可用（{0}）：自测需要仓库内的 eval_latent_table'
              .format(_EV_ERR))
        return 1
    fails = 0
    n_binding = 0
    for name, item, expect in _SELF_TEST_CASES:
        rec = judge(item)
        got = list(rec['kinds'])
        good = (got == list(expect)) and (rec['binding'] == bool(expect))
        if rec['binding']:
            n_binding += 1
        print('[self-test] {0:<14s} ans={1:<7s} binding={2!s:<5s} kinds={3}'
              .format(name, rec['ans_type'], rec['binding'], got))
        if not good:
            fails += 1
            print('            ^ 期望 kinds={0} binding={1}'.format(
                list(expect), bool(expect)))
    if n_binding < 1:
        fails += 1
        print('[FAIL] 内置样本里至少要有 1 条判为 binding，实际 0 条')
    if fails:
        print('self-test FAILED：{0} 条不符合预期'.format(fails))
        return 1
    print('self-test PASS（{0} 条样本，{1} 条 binding）'.format(
        len(_SELF_TEST_CASES), n_binding))
    return 0


# =====================================================================
# CLI
# =====================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description='从 HotpotQA jsonl 划分关系绑定子集（启发式打标，不建新数据集）')
    ap.add_argument('--data', default=None, help='输入 jsonl（每行一个样本）')
    ap.add_argument('--limit', type=int, default=0,
                    help='只读前 N 条（0 表示全部；用于小样本调试）')
    ap.add_argument('--seed', type=int, default=0,
                    help='保留参数：判定是确定性的，当前不抽样')
    ap.add_argument('--out-labels', default=None, help='逐题标签 json 输出路径')
    ap.add_argument('--out-subset', default=None,
                    help='可选：把 binding=True 的原始样本行原样写成 jsonl')
    ap.add_argument('--min-ratio', type=float, default=0.0,
                    help='绑定比例下限（0 表示不检查；低于该值退出码 2，防止规则失效）')
    ap.add_argument('--self-test', action='store_true',
                    help='用内置 3 条合成样本跑判定并断言，失败退出码非 0')
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.data:
        ap.error('需要 --data（或用 --self-test 跑内置样本）')
    if not args.out_labels:
        ap.error('需要 --out-labels（逐题标签是下游分层统计的输入）')
    return run(args.data, limit=args.limit, out_labels=args.out_labels,
               out_subset=args.out_subset, min_ratio=args.min_ratio, seed=args.seed)


if __name__ == '__main__':
    raise SystemExit(main())
