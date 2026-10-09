# -*- coding: utf-8 -*-
"""逐期对齐 SmallAlpha191 修复前后 HP-Search 选中候选, 定位收益下降根因"""
import re
import sys

A = 'results/SmallAlpha191_20220801_20260522_20260819123016/backtest.log'   # 修复前
B = 'results/SmallAlpha191_20220801_20260522_20260819223909/backtest.log'   # 修复后

PAT = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?\[HP-Search\] 选中候选 '
                 r'(\w+): AUC=([0-9.]+) \(共4组: (.*?)\)')


def parse(log):
    rows = []
    for line in open(log, encoding='utf-8', errors='ignore'):
        m = PAT.search(line)
        if m:
            rows.append({'ts': m.group(1), 'cand': m.group(2),
                         'auc': float(m.group(3)), 'detail': m.group(4)})
    return rows


def main():
    a = parse(A)
    b = parse(B)
    print('重训次数: 修复前=%d 修复后=%d' % (len(a), len(b)))
    print()
    hdr = '%-3s %-19s %-8s %-8s %-8s %-8s %s' % ('#', '时间(修复后)', '修复前', '修复后', '前AUC', '后AUC', '变化')
    print(hdr)
    print('-' * len(hdr))
    diff_n = 0
    for i, (ra, rb) in enumerate(zip(a, b)):
        changed = ra['cand'] != rb['cand']
        if changed:
            diff_n += 1
        print('%-3d %-19s %-8s %-8s %-8.4f %-8.4f %s' % (
            i + 1, rb['ts'], ra['cand'], rb['cand'], ra['auc'], rb['auc'],
            '★CHANGE' if changed else ''))
    print()
    print('选中候选变化: %d/%d 期' % (diff_n, len(a)))
    # 汇总变化方向
    from collections import Counter
    moves = Counter()
    for ra, rb in zip(a, b):
        if ra['cand'] != rb['cand']:
            moves['%s→%s' % (ra['cand'], rb['cand'])] += 1
    print('变化方向:', dict(moves))


if __name__ == '__main__':
    main()
