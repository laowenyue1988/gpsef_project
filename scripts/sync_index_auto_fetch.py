# -*- coding: utf-8 -*-
"""同步指数成分股自动补充 (index_auto_fetch) 到 6 个 index_component 策略。
从 hs300_alpha191 提取改动段, 应用到 hs300_raw/zz500_alpha191/zz500_raw/zz1000_alpha191/zz1000_raw。
幂等。"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRATS = os.path.join(ROOT, 'strategies')
SRC = os.path.join(STRATS, 'hs300_alpha191')
TARGETS = ['hs300_raw', 'zz500_alpha191', 'zz500_raw', 'zz1000_alpha191', 'zz1000_raw']


def read(p):
    with open(p, encoding='utf-8') as f:
        return f.read()


def write(p, s):
    with open(p, 'w', encoding='utf-8') as f:
        f.write(s)


def extract_config(src):
    """提取 config data 段中自动补充两行 (smallcap_pool_size 行后)。"""
    m = re.search(r'  smallcap_pool_size: [^\n]*\n  index_auto_fetch: [^\n]*\n  index_auto_fetch_gap_days: [^\n]*\n', src)
    return m.group(0)


def extract_strategy(src):
    """提取 strategy.py 各改动段。"""
    # 1) import 行
    m1 = re.search(r'from \.data_utils import \(.*?build_industry_map_from_snapshots\)', src, re.S)
    import_block = m1.group(0)
    # 2) __init__ config 读取 (8 空格缩进)
    m2 = re.search(r'        self\._smallcap_pool_size = [^\n]*\n'
                   r'        self\._index_auto_fetch[^\n]*\n'
                   r'        self\._index_auto_fetch_gap_days[^\n]*\n'
                   r'        self\._index_cache_file[^\n]*\n', src)
    init_cfg = m2.group(0)
    # 3) _index_fetch_pending 初始化 (8 空格)
    m3 = re.search(r'        self\._index_fetch_pending = [^\n]*\n', src)
    pending_line = m3.group(0)
    # 4) _index_cache_file 赋值 (16 空格)
    m4 = re.search(r'                self\._index_cache_file = _ic_path\n', src)
    cache_file_line = m4.group(0)
    # 5) 方法块 (新 _get_index_components + _auto_fetch_index_components + _persist_index_fetch)
    m5 = re.search(r'    def _get_index_components\(self, date\).*?logger\.warning\(f"\[Pool\] 成分股缓存落盘失败 \(\{e\}\)[^\n]*\)\n', src, re.S)
    methods_block = m5.group(0)
    return import_block, init_cfg, pending_line, cache_file_line, methods_block


def apply_config(path, block):
    s = read(path)
    if 'index_auto_fetch:' in s:
        return False
    old = '  smallcap_pool_size: 300'
    # 各策略 smallcap_pool_size 值可能不同 (均 300)
    m = re.search(r'  smallcap_pool_size: \d+[^\n]*\n', s)
    assert m, f'{path}: 无 smallcap_pool_size'
    s = s.replace(m.group(0), m.group(0) + block, 1)
    write(path, s)
    return True


def apply_strategy(path, import_block, init_cfg, pending_line, cache_file_line, methods_block):
    s = read(path)
    if '_auto_fetch_index_components' in s:
        return False
    # 1) import
    old_imp = re.search(r'from \.data_utils import \(.*?build_industry_map_from_snapshots\)', s, re.S).group(0)
    s = s.replace(old_imp, import_block, 1)
    # 2) __init__ config (匹配完整 8 空格缩进, 避免误替换行内子串)
    old_cfg = re.search(r'        self\._smallcap_pool_size = [^\n]*\n', s)
    assert old_cfg, f'{path}: 无 _smallcap_pool_size'
    s = s.replace(old_cfg.group(0), init_cfg, 1)
    # 3) pending 初始化 (插在 _index_dates 初始化后)
    old_pending = '        self._index_dates = {}        # {index: sorted(date_str)}\n'
    assert old_pending in s, f'{path}: 无 _index_dates 初始化'
    s = s.replace(old_pending, old_pending + pending_line, 1)
    # 4) cache_file 赋值
    old_cf = '                _ic_path = os.path.join(self.data_provider.base_path, \'index_stocks_cache.parquet\')\n'
    assert old_cf in s, f'{path}: 无 _ic_path'
    s = s.replace(old_cf, old_cf + cache_file_line, 1)
    # 5) 方法块 (替换到 _cache_smallcap_result 之前, 保留其 def 行)
    old_m = re.search(r'    def _get_index_components\(self, date\).*?\n    def _cache_smallcap_result', s, re.S)
    assert old_m, f'{path}: 无旧 _get_index_components'
    head = old_m.group(0)[:old_m.group(0).rfind('    def _cache_smallcap_result')]
    s = s.replace(head, methods_block.rstrip() + '\n\n', 1)
    write(path, s)
    return True


def main():
    src_cfg = read(os.path.join(SRC, 'config.yaml'))
    src_sp = read(os.path.join(SRC, 'strategy.py'))
    cfg_block = extract_config(src_cfg)
    imp, init_cfg, pending, cf_line, methods = extract_strategy(src_sp)
    for name in TARGETS:
        base = os.path.join(STRATS, name)
        if apply_config(os.path.join(base, 'config.yaml'), cfg_block):
            print(f'  {name}: config 已同步')
        if apply_strategy(os.path.join(base, 'strategy.py'), imp, init_cfg, pending, cf_line, methods):
            print(f'  {name}: strategy.py 已同步')
    print('同步完成')


if __name__ == '__main__':
    main()
