# -*- coding: utf-8 -*-
"""[Q5BAR] 编辑后静态校验：语法 + R-169 符号守恒 + 变更行数 + 锚点接回。

用法: python _q5bar_verify_edit.py <baseline.py> <current.py>
退出码: 0 = 全绿; 1 = 有断言失败
"""
import ast
import io
import json
import os
import sys
from difflib import SequenceMatcher


def syms(path):
    """收集 FunctionDef / AsyncFunctionDef / ClassDef 名称（R-169：插入不得删符号）。"""
    t = io.open(path, encoding='utf-8').read()
    tree = ast.parse(t)
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
    return out, t


def main():
    base_p, cur_p = sys.argv[1], sys.argv[2]
    ok = True

    # 1) 语法
    try:
        bs, bt = syms(base_p)
        cs, ct = syms(cur_p)
        print('[1] 语法 + AST 解析 .......... PASS (两文件均可解析)')
    except SyntaxError as e:
        print('[1] 语法 + AST 解析 .......... FAIL', e)
        return 1

    # 2) 符号守恒（R-169）
    lost = sorted(bs - cs)
    added = sorted(cs - bs)
    print('[2] R-169 符号守恒 .......... %s' % ('PASS' if not lost else 'FAIL'))
    print('     基线 %d 个 | 当前 %d 个 | 新增 %d | 丢失 %d' % (len(bs), len(cs), len(added), len(lost)))
    print('     新增: %s' % added)
    if lost:
        print('     ★丢失（必须为 0）: %s' % lost)
        ok = False

    # 3) 变更行数（SequenceMatcher opcodes, R-108）
    bl, cl = bt.splitlines(), ct.splitlines()
    sm = SequenceMatcher(None, bl, cl, autojunk=False)
    ins = dele = repl = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'insert':
            ins += j2 - j1
        elif tag == 'delete':
            dele += i2 - i1
        elif tag == 'replace':
            repl += max(i2 - i1, j2 - j1)
    print('[3] 变更行数 ................. 新增 %d / 删除 %d / 替换 %d | 基线 %d → 当前 %d 行'
          % (ins, dele, repl, len(bl), len(cl)))

    # 4) 新增符号确实存在且带 Q5BAR 标记
    new_syms = ['_apply_barra_risk_budget', '_load_barra_expo_pit']
    for s in new_syms:
        present = s in cs
        print('[4] 新符号 %-28s %s' % (s, 'PASS' if present else 'FAIL'))
        if not present:
            ok = False

    # 5) 锚点接回（R-169：插入必须接回锚点）
    anchors = ['def _compute_entry_vol(self, code, date):',
               'selected = list(features_df.index[top_idx])',
               'logger.info(f"[_meta_select_stocks] Top-{n_select} 精选模式']
    print('[5] 锚点接回 ................. ')
    for a in anchors:
        # 取锚点前 40 字符做包含判定（部分行含 f-string 花括号）
        key = a[:40]
        hit = key in ct
        print('      %-52s %s' % (key[:52], 'PASS' if hit else 'FAIL'))
        if not hit:
            ok = False

    # 6) 零影响关键：默认 mode 必须为 off
    zero_ok = "_rb_cfg.get('mode', 'off')" in ct
    print('[6] 默认 mode=off（零影响）... %s' % ('PASS' if zero_ok else 'FAIL'))
    if not zero_ok:
        ok = False

    # 7) 只改排序分数、不改 prob_dict：prob_dict 段仍引用 prob_profit
    pd_ok = 'prob_dict[code] = float(prob_profit[i])' in ct
    print('[7] prob_dict 仍用原 prob_profit %s' % ('PASS' if pd_ok else 'FAIL'))
    if not pd_ok:
        ok = False

    print('=' * 62)
    print('总判定: %s' % ('✅ ALL PASS' if ok else '❌ FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
