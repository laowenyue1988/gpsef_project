# -*- coding: utf-8 -*-
"""一次性同步: 把 hs300_alpha191 中的 index_stocks_cache 来源说明/生成函数/缺口告警
同步到其余策略副本 (data_utils.py / strategy.py)。幂等, 可重复执行。"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRATS = os.path.join(ROOT, 'strategies')
SRC = os.path.join(STRATS, 'hs300_alpha191')

# 目标策略 (全部指数策略 + small/big/alpha191 源策略)
TARGETS = ['hs300_alpha191', 'hs300_raw', 'zz500_alpha191', 'zz500_raw',
           'zz1000_alpha191', 'zz1000_raw', 'cyb_alpha191', 'cyb_raw',
           'kcb_alpha191', 'kcb_raw', 'small_alpha191', 'small_raw',
           'big_alpha191', 'big_raw', 'alpha191']
# strategy.py 缺口告警只适用于 index_component 模式策略
STRATEGY_TARGETS = ['hs300_alpha191', 'hs300_raw', 'zz500_alpha191', 'zz500_raw',
                    'zz1000_alpha191', 'zz1000_raw']


def read(p):
    with open(p, encoding='utf-8') as f:
        return f.read()


def write(p, s):
    with open(p, 'w', encoding='utf-8') as f:
        f.write(s)


def sync_data_utils(src_text):
    # 1) 来源说明注释块 (插在 '# ---------- 指数成分股 ----------' 之后)
    src_anchor = '# ---------- 指数成分股 ----------\n'
    head = src_text[src_text.index(src_anchor):]
    head = head[:head.index('GM_AVAILABLE = False')]
    # head 现在是: '# ---------- 指数成分股 ----------\n' + 注释行 + '\n'
    new_block = head.rstrip() + '\nGM_AVAILABLE = False\n'

    # 2) 生成函数 fetch_index_constituents_by_gm (含 docstring 注释)
    m = re.search(r'def fetch_index_constituents_by_gm.*?(?=\ndef get_multi_index_stocks_by_akshare)', src_text, re.S)
    fetch_fn = m.group(0).rstrip() + '\n\n'

    # 3) get_multi_index_stocks_by_akshare 的新 docstring 首行
    m2 = re.search(r'def get_multi_index_stocks_by_akshare.*?"""[^\n]*', src_text, re.S)
    new_def_head = m2.group(0)

    return new_block, fetch_fn, new_def_head


def apply_data_utils(path, new_block, fetch_fn, new_def_head):
    s = read(path)
    orig = s
    # 幂等: 已同步则跳过
    if '数据来源(2026-08-04 审计确认)' in s:
        return False
    old_anchor = '# ---------- 指数成分股 ----------\nGM_AVAILABLE = False\n'
    assert old_anchor in s, f'{path}: 缺少旧注释锚点'
    s = s.replace(old_anchor, new_block, 1)
    old_fn = re.search(r'def get_multi_index_stocks_by_akshare.*?"""[^\n]*', s, re.S).group(0)
    # 插入新函数 + 替换 docstring 首行
    s = s.replace(old_fn, fetch_fn + new_def_head, 1)
    write(path, s)
    print(f'  data_utils.py 已同步: {os.path.basename(os.path.dirname(path))}')
    return True


def apply_strategy(path):
    s = read(path)
    if '指数成分股缓存缺口' in s:
        return False
    # 从 hs300_alpha191 读取缺口告警段 (保留完整缩进)
    src = read(os.path.join(SRC, 'strategy.py'))
    m = re.search(r'(\n                else:\n                    # 缺口告警:.*?break\n)', src, re.S)
    warn_block = m.group(1).lstrip('\n')
    old_anchor = 'if _n_dates == 0:\n                    logger.warning(f"[Pool] 指数 {self._pool_index} 无成分股数据")\n'
    assert old_anchor in s, f'{path}: 缺少 _n_dates 锚点'
    s = s.replace(old_anchor, old_anchor + warn_block, 1)
    write(path, s)
    print(f'  strategy.py 已同步: {os.path.basename(os.path.dirname(path))}')
    return True


def main():
    src_du = read(os.path.join(SRC, 'data_utils.py'))
    new_block, fetch_fn, new_def_head = sync_data_utils(src_du)
    for name in TARGETS:
        if name == 'hs300_alpha191':
            continue
        du = os.path.join(STRATS, name, 'data_utils.py')
        if os.path.exists(du):
            apply_data_utils(du, new_block, fetch_fn, new_def_head)
    for name in STRATEGY_TARGETS:
        if name == 'hs300_alpha191':
            continue
        sp = os.path.join(STRATS, name, 'strategy.py')
        if os.path.exists(sp):
            apply_strategy(sp)
    print('同步完成')


if __name__ == '__main__':
    main()
