#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''tests/test_binding_subset.py — 关系绑定子集划分器（tools/make_binding_subset.py）的单测。

为什么值得单独测：三个启发式都是「写错了不会报错、只会让分层统计静默失真」的东西
  - 实体片段切分：句首疑问词必须过滤，否则每个英文问句都成了多实体题，
    entity_swap_risk 恒真，绑定子集退化成全集。
  - 相似实体：同名/子串/共享词三条路径各自都要能触发，也都不能误触发。
  - 属性值混淆：年份不能被当成数值候选（否则 number 题几乎全触发）；
    日期只看首字母大写的月份（小写 may 是情态动词）。
  - answer_type 必须复用 eval_latent_table 的实现（同一个函数对象），不能另写一份。
  - labels json 的结构与 idx 口径（文件物理行号）是下游分层统计的输入。

用法：python tests/test_binding_subset.py
'''
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

# 本地可能没有 torch；没有就跳过依赖 answer_type 的断言，不要报错。
try:
    import eval_latent_table as EV
except (ImportError, ModuleNotFoundError) as _exc:
    EV = None
    print('[SKIP] 无法导入 eval_latent_table（{0}），依赖它的断言会跳过'.format(_exc))

import make_binding_subset as B

HAS_EV = B.answer_type is not None
N_OK = 0
FAILS = []


def ok(name, cond, extra=''):
    global N_OK
    if cond:
        N_OK += 1
    else:
        FAILS.append(name)
        print('[FAIL] {0} {1}'.format(name, extra))
    return bool(cond)


# =====================================================================
# 合成样本（与 HotpotQA 字段一致：units / unit_meta / evidence / question / answer）
# =====================================================================
def sample(question, answer, rows, hops=2, qtype='bridge'):
    units = [r[0] for r in rows]
    unit_meta = [{'anchor': r[2], 'title': r[2], 'is_gold': bool(r[1]), 'fact': r[0]}
                 for r in rows]
    return {
        'units': units,
        'unit_meta': unit_meta,
        'evidence': [r[0] for r in rows if r[1]],
        'question': question,
        'answer': answer,
        'meta': {'type': qtype, 'qid': 't'},
        'hops': hops,
    }


def _um(rows):
    return [{'anchor': a, 'title': a, 'is_gold': g} for _, g, a in rows]


S_ENTITY = sample(
    'Which magazine was started first, Arthur\'s Magazine or First for Women?',
    'Arthur\'s Magazine',
    [('Arthur\'s Magazine was an American literary periodical.', True,
      'Arthur\'s Magazine'),
     ('First for Women is a women\'s magazine.', True, 'First for Women'),
     ('Arthur\'s Magazine was started in 1844 in Philadelphia.', True,
      'Arthur\'s Magazine'),
     ('Vogue is a fashion magazine founded in New York.', False, 'Vogue'),
     ('Elle was founded in 1945 in France.', False, 'Elle')])

S_VALUE = sample(
    'When was Arthur\'s Magazine started?',
    '1844',
    [('Arthur\'s Magazine was an American literary periodical.', True,
      'Arthur\'s Magazine'),
     ('Arthur\'s Magazine was started in 1844 in Philadelphia.', True,
      'Arthur\'s Magazine'),
     ('Vogue was founded in 1892 in New York.', False, 'Vogue'),
     ('Elle was founded in 1945 in France.', False, 'Elle')])

S_NEG = sample(
    'Who wrote the novel Ed Wood?',
    'Rudolph Grey',
    [('Rudolph Grey wrote the novel Ed Wood.', True, 'Rudolph Grey'),
     ('Ed Wood is a 1994 American biographical film.', False, 'Ed Wood'),
     ('Tim Burton directed the film.', False, 'Tim Burton')])


# =====================================================================
# 1. entity_swap_risk
# =====================================================================
def test_entity_swap():
    q_multi = 'Which magazine was started first, Arthur\'s Magazine or First for Women?'
    q_single = 'Who directed the film Ed Wood?'
    ok('E1 两个实体片段 + 实体答案 -> 触发',
       B.entity_swap_risk(q_multi, 'entity') is True,
       extra=str(B.question_entity_spans(q_multi)))
    ok('E2 单个实体片段 -> 不触发',
       B.entity_swap_risk(q_single, 'entity') is False,
       extra=str(B.question_entity_spans(q_single)))
    ok('E3 两个实体但答案是 yes/no -> 不触发（比较题不是实体交换）',
       B.entity_swap_risk(
           'Were Scott Derrickson and Ed Wood of the same nationality?',
           'yesno') is False)
    ok('E4 两个实体但答案是数值 -> 不触发（问的是量，不是实体）',
       B.entity_swap_risk(q_multi, 'number') is False)
    ok('E5 句首疑问词不单独算实体',
       B.question_entity_spans('Which city is the largest?') == [],
       extra=str(B.question_entity_spans('Which city is the largest?')))
    ok('E6 片段切分：标点处断开，Arthur\'s Magazine 是一个片段',
       B.question_entity_spans(q_multi) == ['Arthur\'s Magazine', 'Women'],
       extra=str(B.question_entity_spans(q_multi)))


# =====================================================================
# 2. similar_entity_risk
# =====================================================================
def test_similar_entity():
    ok('S1 干扰 anchor 与金标 anchor 同名 -> 触发',
       B.similar_entity_risk(_um([('x', True, 'Ed Wood'),
                                   ('y', False, 'Ed Wood')])) is True)
    ok('S2 干扰 anchor 是金标 anchor 的子串 -> 触发',
       B.similar_entity_risk(_um([('x', True, 'Ed Wood'),
                                   ('y', False, 'Wood')])) is True)
    ok('S3 两个金标 anchor 共享同一个词 -> 触发（Ed Wood / Edward Davis Wood Jr.）',
       B.similar_entity_risk(_um([('x', True, 'Ed Wood'),
                                   ('y', True, 'Edward Davis Wood Jr.')])) is True)
    ok('S4 无同名/子串/共享词 -> 不触发',
       B.similar_entity_risk(_um([('x', True, 'Ed Wood'),
                                   ('y', True, 'Tim Burton'),
                                   ('z', False, 'Vogue')])) is False)
    ok('S5 单字符 anchor 不做子串匹配（否则命中一切标题）',
       B.similar_entity_risk(_um([('x', True, 'Arthur'),
                                   ('y', False, 'A')])) is False)
    ok('S6 空 unit_meta -> 不触发',
       B.similar_entity_risk([]) is False)


# =====================================================================
# 3. value_confusion_risk
# =====================================================================
def test_value_confusion():
    ok('V1 date：两句含年份 -> 触发',
       B.value_confusion_risk(['It began in 1844.', 'It ended in 1892.'], 'date') is True)
    ok('V2 date：只有一句含年份 -> 不触发',
       B.value_confusion_risk(['It began in 1844.', 'It is a magazine.',
                               'It is published monthly.'], 'date') is False)
    ok('V3 date：首字母大写月份也算日期',
       B.value_confusion_risk(['It began in May.', 'It ended in June.'], 'date') is True)
    ok('V4 number：两句含非年份数值 -> 触发',
       B.value_confusion_risk(['He had 3 children.', 'The film has 2 sequels.'],
                              'number') is True)
    ok('V5 number：四位年份不算数值候选 -> 不触发',
       B.value_confusion_risk(['It began in 1844.', 'It ended in 1900.'],
                              'number') is False)
    ok('V6 答案类型不是数值/日期 -> 恒不触发',
       B.value_confusion_risk(['1844 and 1900 and 3 and 2.'], 'entity') is False)


# =====================================================================
# 4. judge：binding 的或逻辑 + 三种风险各自独立触发
# =====================================================================
def test_judge_or_logic():
    if not HAS_EV:
        print('[SKIP] 没有 answer_type，跳过 judge / binding 或逻辑断言')
        return
    # 只触发 similar：单实体问题 + 两个金标 anchor 共享 wood。
    s_sim = sample(
        'Who wrote the novel Ed Wood?', 'Rudolph Grey',
        [('Rudolph Grey wrote the novel Ed Wood.', True, 'Ed Wood'),
         ('Edward Davis Wood Jr. was born in 1924.', True,
          'Edward Davis Wood Jr.')])
    r_entity = B.judge(S_ENTITY)
    r_value = B.judge(S_VALUE)
    r_neg = B.judge(S_NEG)
    r_sim = B.judge(s_sim)

    ok('J1 只触发 entity_swap -> binding=True',
       r_entity['binding'] is True and r_entity['kinds'] == ['entity_swap_risk'],
       extra=str(r_entity))
    ok('J2 只触发 value_confusion -> binding=True',
       r_value['binding'] is True and r_value['kinds'] == ['value_confusion_risk'],
       extra=str(r_value))
    ok('J3 只触发 similar_entity -> binding=True',
       r_sim['binding'] is True and r_sim['kinds'] == ['similar_entity_risk'],
       extra=str(r_sim))
    ok('J4 三种都不触发 -> binding=False, kinds=[]',
       r_neg['binding'] is False and r_neg['kinds'] == [], extra=str(r_neg))
    for tag, rec in (('J5 entity', r_entity), ('J6 value', r_value),
                     ('J7 similar', r_sim), ('J8 neg', r_neg)):
        expect = [k for k in B.KINDS if k in rec['kinds']]
        ok('{0} binding == bool(kinds) 且 kinds 按固定顺序'.format(tag),
           rec['binding'] == bool(rec['kinds']) and rec['kinds'] == expect,
           extra=str(rec))
    ok('J9 标签字段齐全且类型正确',
       set(r_neg) == {'binding', 'kinds', 'ans_type', 'hops', 'n_gold'}
       and isinstance(r_neg['binding'], bool)
       and isinstance(r_neg['kinds'], list)
       and isinstance(r_neg['ans_type'], str)
       and isinstance(r_neg['hops'], int)
       and isinstance(r_neg['n_gold'], int),
       extra=str(r_neg))
    ok('J10 n_gold 数的是 unit_meta.is_gold（S_NEG 有 1 句金标）',
       r_neg['n_gold'] == 1, extra=str(r_neg['n_gold']))
    ok('J11 hops 取自样本（2 跳）', r_neg['hops'] == 2, extra=str(r_neg['hops']))


# =====================================================================
# 5. answer_type 复用（同一函数对象 + judge 确实调用模块级函数）
# =====================================================================
def test_answer_type_reuse():
    if EV is None or B.answer_type is None:
        print('[SKIP] 本地没有 torch / eval_latent_table，跳过 answer_type 同一性断言')
    else:
        ok('A1 answer_type 就是 EV.answer_type（同一个函数对象，没有另写一份）',
           B.answer_type is EV.answer_type)

    # 注入一个假实现：judge 若自己另写了一份，就不会走这里。
    orig = B.answer_type
    seen = []

    def _fake(a):
        seen.append(a)
        return 'entity'

    B.answer_type = _fake
    try:
        rec = B.judge(S_NEG)
    finally:
        B.answer_type = orig
    ok('A2 judge 调用的是模块级 answer_type（可注入，说明没有内联实现）',
       seen == ['Rudolph Grey'] and rec['ans_type'] == 'entity',
       extra=str(seen))


# =====================================================================
# 6. labels json 结构 / idx 口径 / 子集原样复制
# =====================================================================
def _write_jsonl(path, objs, blank_after=()):
    with open(path, 'w', encoding='utf-8') as f:
        for i, o in enumerate(objs):
            f.write(json.dumps(o, ensure_ascii=False))
            f.write('\n')
            if i in blank_after:
                f.write('\n')


def _run_quiet(argv):
    buf = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        ret = B.main(argv)
    return ret, buf.getvalue()


def test_labels_json():
    if not HAS_EV:
        print('[SKIP] 没有 answer_type，跳过 labels json 断言')
        return
    tmp = tempfile.mkdtemp(prefix='binding_subset_')
    try:
        data = os.path.join(tmp, 'data.jsonl')
        # 第 1 行后插一个空行：验证 idx 用的是文件物理行号（0 和 2），不是样本序号。
        _write_jsonl(data, [S_ENTITY, S_VALUE, S_NEG], blank_after=(0,))
        lab = os.path.join(tmp, 'lab.json')
        sub = os.path.join(tmp, 'sub.jsonl')
        ret, out = _run_quiet(['--data', data, '--limit', '2',
                               '--out-labels', lab, '--out-subset', sub])
        ok('L1 main 返回 0', ret == 0, extra=str(ret))
        with open(lab, encoding='utf-8') as f:
            labels = json.load(f)
        ok('L2 顶层字段恰为 n/n_binding/n_by_kind/items',
           set(labels) == {'n', 'n_binding', 'n_by_kind', 'items'},
           extra=str(sorted(labels)))
        ok('L3 n=2（--limit 2）且 n_binding=2',
           labels['n'] == 2 and labels['n_binding'] == 2,
           extra='{0}/{1}'.format(labels.get('n'), labels.get('n_binding')))
        ok('L4 n_by_kind 三种风险齐全',
           set(labels['n_by_kind']) == set(B.KINDS), extra=str(labels['n_by_kind']))
        ok('L5 items 的 key 是文件物理行号（0 和 2，跳过空行）',
           set(labels['items']) == {'0', '2'}, extra=str(sorted(labels['items'])))
        it0 = labels['items']['0']
        ok('L6 单题字段齐全且类型正确',
           set(it0) == {'binding', 'kinds', 'ans_type', 'hops', 'n_gold'}
           and it0['binding'] is True
           and it0['kinds'] == ['entity_swap_risk']
           and it0['ans_type'] == 'entity'
           and it0['hops'] == 2
           and it0['n_gold'] == 3,
           extra=str(it0))
        ok('L7 第二条是日期题（1844）且判为 value_confusion',
           labels['items']['2']['ans_type'] == 'date'
           and labels['items']['2']['kinds'] == ['value_confusion_risk'],
           extra=str(labels['items']['2']))
        with open(sub, encoding='utf-8') as f:
            sub_lines = f.read().splitlines()
        with open(data, encoding='utf-8') as f:
            data_lines = f.read().splitlines()
        ok('L8 --out-subset 只写 binding 题、且原样复制原始行',
           sub_lines == [data_lines[0], data_lines[2]],
           extra='subset={0}'.format(len(sub_lines)))

        # --limit 1：只读第一条样本。
        lab1 = os.path.join(tmp, 'lab1.json')
        ret1, _ = _run_quiet(['--data', data, '--limit', '1', '--out-labels', lab1])
        with open(lab1, encoding='utf-8') as f:
            labels1 = json.load(f)
        ok('L9 --limit 1 只保留 idx=0',
           ret1 == 0 and labels1['n'] == 1 and set(labels1['items']) == {'0'},
           extra=str(labels1['items'].keys()))

        # --min-ratio 守卫：3 条里 2 条 binding（0.667）< 0.9 -> 退出码 2。
        lab2 = os.path.join(tmp, 'lab2.json')
        ret2, _ = _run_quiet(['--data', data, '--min-ratio', '0.9',
                              '--out-labels', lab2])
        ok('L10 绑定比例低于 --min-ratio 时退出码 2', ret2 == 2, extra=str(ret2))
        ok('L11 即使触发 min-ratio 守卫，labels 仍写出（便于排查）',
           os.path.isfile(lab2))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =====================================================================
# 7. CLI 行为：--self-test / 缺参数
# =====================================================================
def test_self_test_cli():
    if not HAS_EV:
        print('[SKIP] 没有 answer_type，跳过 --self-test 断言')
        return
    script = os.path.join(ROOT, 'tools', 'make_binding_subset.py')
    p = subprocess.run([sys.executable, script, '--self-test'],
                       capture_output=True, text=True, timeout=300)
    ok('C1 --self-test 退出码 0', p.returncode == 0,
       extra=(p.stderr or '')[-400:])
    ok('C2 --self-test 打印 PASS 且 3 条样本里 >=1 条 binding',
       'self-test PASS' in p.stdout and '2 条 binding' in p.stdout,
       extra=(p.stdout or '')[-400:])


def test_cli_required_args():
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            B.main([])
    except SystemExit as exc:
        code = exc.code
    else:
        code = None
    ok('C3 缺 --data 时 argparse 以非 0 退出', code not in (None, 0), extra=str(code))


def main():
    for fn in (test_entity_swap, test_similar_entity, test_value_confusion,
               test_judge_or_logic, test_answer_type_reuse, test_labels_json,
               test_self_test_cli, test_cli_required_args):
        print('--- {0}'.format(fn.__name__))
        fn()
    print('=' * 70)
    if FAILS:
        print('FAILED {0} / {1}：{2}'.format(len(FAILS), N_OK + len(FAILS), FAILS))
        return 1
    print('ALL PASS ({0} checks)'.format(N_OK))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
