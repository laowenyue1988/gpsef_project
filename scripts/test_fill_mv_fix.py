# -*- coding: utf-8 -*-
"""验证 _fill_total_mv_per_date 列冲突修复 (hs300 全 0.5 bug 根因)。

测试点:
  T1 修复后: 空分支(cache 缺股票)行 trade_date 保留正常值, 无 NaN
  T2 修复后: merge 分支行 trade_date 正常, total_mv 来自 cache
  T3 修复后: 缺失市值行 total_mv = 全局中位数填充
  T4 行序/行数保持: 输出行数 = 输入行数, 样本可追溯
  T5 其他目录同源检查: 是否也存在相同列冲突模式
"""
import os, sys, ast, types, logging, importlib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger('test_fill_mv_fix')
ok = True


def check(name, cond, extra=''):
    global ok
    mark = 'PASS' if cond else 'FAIL'
    if not cond:
        ok = False
    print(f'[{mark}] {name} {extra}')
    return cond


def extract_method(path, cls, meth):
    src = open(path, encoding='utf-8').read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == meth:
                    return ast.get_source_segment(src, item)
    raise RuntimeError(f'{cls}.{meth} not found in {path}')


def make_mock_self(cache_df):
    """mock self: 只提供 _load_total_mv_cache 和 data_provider + _realtime_total_mv"""
    self_mock = types.SimpleNamespace()
    self_mock._load_total_mv_cache = lambda: cache_df.copy()
    self_mock.data_provider = types.SimpleNamespace(
        raw_daily_path='data/raw_daily', base_path='data')
    # 从源文件提取 _realtime_total_mv 绑定 (cg.empty 分支实时重算需要)
    src_all = open('strategies/hs300_alpha191/strategy.py', encoding='utf-8').read()
    tree_all = ast.parse(src_all)
    for node in ast.walk(tree_all):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == '_realtime_total_mv':
                    rt_src = ast.get_source_segment(src_all, item)
                    rt_ns = {'pd': pd, 'np': np, 'logging': logging}
                    exec(compile(rt_src, '_realtime_total_mv', 'exec'), rt_ns)
                    self_mock._realtime_total_mv = types.MethodType(
                        rt_ns['_realtime_total_mv'], self_mock)
                    break
    return self_mock


def make_input_df():
    """构造训练形状数据: 2 只股票 × 3 天, trade_date=T, snapshot_date=T-1"""
    rows = []
    dates = ['2022-01-04', '2022-01-05', '2022-01-06']
    snaps = ['2022-01-03', '2022-01-04', '2022-01-05']
    for code in ['sh.600000', 'sz.000001']:   # 600000 在 cache, 000001 不在
        for d, s in zip(dates, snaps):
            rows.append({'ts_code': code, 'trade_date': pd.Timestamp(d),
                         'snapshot_date': pd.Timestamp(s), 'alpha_001': 1.0})
    df = pd.DataFrame(rows)
    return df


def make_cache_df():
    """市值缓存: 只含 sh.600000 (→ sz.000001 走 cg.empty 分支)"""
    rows = []
    for d in ['2022-01-01', '2022-01-04', '2022-01-06']:
        rows.append({'ts_code': 'sh.600000', 'trade_date': pd.Timestamp(d), 'total_mv': 5e10})
    return pd.DataFrame(rows)


def main():
    global ok
    src = extract_method('strategies/hs300_alpha191/strategy.py',
                         'Hs300Alpha191Strategy', '_fill_total_mv_per_date')
    _ns = {'pd': pd, 'np': np, 'logging': logging,
           'logger': logger}
    exec(compile(src, 'fill_total_mv_per_date', 'exec'), _ns)
    fill = _ns['_fill_total_mv_per_date']

    df = make_input_df()
    cache = make_cache_df()
    self_mock = make_mock_self(cache)

    out = fill(self_mock, df)

    # T1: trade_date 无 NaN (修复前: 空分支 3 行 NaN)
    n_nan = int(out['trade_date'].isna().sum())
    ok &= check('T1 修复后 trade_date 无 NaN', n_nan == 0, f'(NaN={n_nan}/{(out.shape[0])})')

    # T2: merge 分支市值正确 (sh.600000 3 行)
    mv_600000 = out.loc[out['ts_code'] == 'sh.600000', 'total_mv']
    ok &= check('T2 有市值分支 total_mv 正确', mv_600000.notna().all() and (mv_600000 == 5e10).all(),
                f'({mv_600000.tolist()})')

    # T3: 空分支 total_mv = 实时重算 (非中位数) — [2026-08-19 增强]
    mv_000001 = out.loc[out['ts_code'] == 'sz.000001', 'total_mv']
    ok &= check('T3 缺市值分支实时重算非NaN', mv_000001.notna().all(),
                f'({mv_000001.tolist()})')
    # 与直接调 _realtime_total_mv 一致 (真实 raw_daily)
    rt_mv = self_mock._realtime_total_mv(
        'sz.000001', df.loc[df['ts_code'] == 'sz.000001', 'snapshot_date'])
    consistent = np.allclose(mv_000001.to_numpy(), rt_mv.to_numpy(),
                             rtol=1e-9, equal_nan=True)
    ok &= check('T3b 空分支=实时重算值一致', consistent,
                f'(diff={np.abs(mv_000001.to_numpy() - rt_mv.to_numpy()).max():.2e})')

    # T4: 行数保持 + trade_date 值正确 (每行 trade_date == 原 T)
    ok &= check('T4 行数保持', len(out) == len(df), f'({len(out)} vs {len(df)})')
    t_match = (out['trade_date'].values == df['trade_date'].values).all()
    ok &= check('T4b trade_date 与输入逐行一致', t_match)

    # T4c: snapshot_date 也保留
    s_match = (out['snapshot_date'].values == df['snapshot_date'].values).all()
    ok &= check('T4c snapshot_date 逐行一致', s_match)

    # T5: 其他目录同源检查 — 是否同样存在 part 层未处理 trade_date_x 的模式
    print('\n=== T5 其他目录 _fill_total_mv_per_date 同源检查 ===')
    DIRS = ['alpha191', 'alpha191_feat_test', 'big_alpha191', 'cyb_alpha191',
            'kcb_alpha191', 'small_alpha191', 'small_alpha191_test',
            'zz1000_alpha191', 'zz500_alpha191', 'wyckoff20']
    for d in DIRS:
        p = f'strategies/{d}/strategy.py'
        if not os.path.exists(p):
            # V10 不同名
            alt = 'strategies/wyckoff20/gpsef_strategy_v10.py'
            p = alt if d == 'wyckoff20' else p
        if not os.path.exists(p):
            continue
        s = open(p, encoding='utf-8').read()
        if '_fill_total_mv_per_date' not in s:
            print(f'  [{d}] 无该方法')
            continue
        # 检查: part 层是否在 append 前处理 trade_date_x
        m = s[s.find('def _fill_total_mv_per_date'):]
        part_handled = ("if 'trade_date_x' in m.columns" in m) and \
                       ('parts.append(m)' in m) and \
                       (m.find("if 'trade_date_x' in m.columns") < m.find('parts.append(m)'))
        old_style = ("if 'trade_date_x' in merged.columns" in m) and not part_handled
        has_rt = '_realtime_total_mv' in s and "2026-08-19 增强] 缓存缺失的股票" in s
        print(f'  [{d}] part层处理={part_handled} | 旧模式(concat后处理)={old_style} | 实时重算={has_rt}')

    # ---- T6: 实时重算端到端 (真实缺市值股票) ----
    # 用真实数据: 找到缓存缺失但 raw_daily 存在的 hs300 股票, 验证 _fill_total_mv_per_date 实时补全
    try:
        cache_r = pd.read_parquet('data/total_mv_cache.parquet')
        have_r = set(cache_r['ts_code'].unique())
        cache_idx = pd.read_parquet('data/index_stocks_cache.parquet')
        ks_r = sorted([k for k in cache_idx['cache_key']
                       if k.startswith('hs300_') and k < 'hs300_20220201'])
        comp_r = set(cache_idx[cache_idx['cache_key'] == ks_r[-1]]['stock_list'].iloc[0])
        missing_r = [x for x in comp_r if x not in have_r]
        ok &= check('T6 找到缺市值股票', len(missing_r) > 0, f'({len(missing_r)} 只)')
        # 每只缺市值股票走实时重算: 直接调 _realtime_total_mv, 与手算 amount*100/turn 对比
        code_r = missing_r[0]
        safe = code_r.replace('.', '_')
        raw_r = pd.read_parquet(f'data/raw_daily/{safe}.parquet')
        raw_r['total_mv_manual'] = raw_r['amount'] * 100.0 / raw_r['turn']
        raw_r = raw_r.dropna(subset=['total_mv_manual'])
        if len(raw_r):
            probe_date = pd.Timestamp(raw_r['trade_date'].iloc[-1])
            rt_self = make_mock_self(None)
            mv_rt = rt_self._realtime_total_mv(
                code_r, pd.Series([probe_date]))
            manual_val = raw_r['total_mv_manual'].iloc[-1]
            rel_err = abs(mv_rt.iloc[0] - manual_val) / manual_val if manual_val else -1
            ok &= check('T6b 实时重算与手算一致', pd.notna(mv_rt.iloc[0]) and rel_err < 1e-6,
                        f'({code_r}: rel_err={rel_err:.2e})')
        else:
            ok &= check('T6b 实时重算与手算一致', False, '(raw 无有效数据)')
    except Exception as e:
        ok &= check('T6 实时重算端到端', False, str(e))

    # ---- T7: 10 目录 _realtime_total_mv 存在性 ----
    dirs7 = ['hs300_alpha191'] + ['alpha191', 'alpha191_feat_test', 'big_alpha191',
                                  'cyb_alpha191', 'kcb_alpha191', 'small_alpha191',
                                  'small_alpha191_test', 'zz1000_alpha191', 'zz500_alpha191']
    for d in dirs7:
        s = open(f'strategies/{d}/strategy.py', encoding='utf-8').read()
        has = '_realtime_total_mv' in s and '2026-08-19 增强] 缓存缺失的股票' in s
        ok &= check(f'[{d}] T7 实时重算已同步', has)

    print('\n==== 市值填充修复验证: %s ====' % ('ALL PASS' if ok else '有失败!'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
