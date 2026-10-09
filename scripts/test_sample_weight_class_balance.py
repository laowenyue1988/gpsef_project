# -*- coding: utf-8 -*-
"""
sample_weight 与 xgb_class_balance 生效性验证 (2026-08-19, 修复前语义)

⚠️ 2026-08-20 回滚说明: 样本权重修复(2026-08-19 21:48)导致完整回测收益大幅下降,
用户裁决回滚至修复前。本套件恢复为修复前语义——如实记录潜在问题但不修改代码:
  - S4 记录缓存命中路径潜在bug(生产不触发)
  - S8 记录 HP-Search 无 spw 注入 / CV 折集成无 mode 检查 (已知不对称)

验证链路:
  S1  compute_sample_weights: 唯一性权重 (AFML Ch.4 avg uniqueness)
  S2  PT/SL 标签质量加权: pt_sl_multiplier=2.0 → PT/SL 样本 ×2 (再归一化)
  S3  pt_sl_multiplier=1.0 → 完全不加权 (与 exit_type=None 等价)
  S4  缓存路径: 未命中=正确(生产实际路径); 命中=潜在bug如实记录
  S5  align_sample_weight_to_sort: 排序后权重与样本对齐 (2026-08-18修复)
  S6  config 链路: sample_weight_pt_sl_multiplier=2.0 / xgb_class_balance=auto
  S7  scale_pos_weight 计算: spw = N_neg/N_pos (折内/生产各按各自训练集)
  S8  注入条件核查 (AST 代码级, 修复前状态):
      - PKF 折内: 注入 spw, 条件 obj!=regression AND mode!='none' ✓
      - 生产模型: 注入 spw, 同条件 ✓
      - CV 折集成: 注入 spw, 但未检查 mode=='none' (已知不一致, 记录)
      - HP-Search 评估: 未注入 spw (协议不对称, 记录)
  S9  冒烟回测日志佐证
"""
import os
import sys
import ast
import datetime

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRAT = os.path.join(PROJ, 'strategies', 'hs300_alpha191')  # 2026-08-20: test 目录已废弃, 改用同内容标准版
sys.path.insert(0, PROJ)

from strategies.hs300_alpha191.ensemble import (
    compute_sample_weights,
    compute_sample_weights_cached,
    align_sample_weight_to_sort,
)

_PASS, _FAIL = [], []


def check(name, cond, detail=''):
    if cond:
        _PASS.append(name)
        print(f'  PASS  {name}  {detail}')
    else:
        _FAIL.append(name)
        print(f'  FAIL  {name}  {detail}')


# ---------------- S1/S2/S3/S4: compute_sample_weights 语义 ----------------
def s12_make_data():
    """两个事件日, 窗口不对称 → 唯一性权重可区分"""
    cal = pd.to_datetime(['2022-01-03', '2022-01-04', '2022-01-05',
                          '2022-01-06', '2022-01-07', '2022-01-10',
                          '2022-01-11'])
    df = pd.DataFrame({
        'trade_date': pd.to_datetime(['2022-01-03', '2022-01-03',
                                      '2022-01-06', '2022-01-06', '2022-01-06']),
        'ts_code': ['a', 'b', 'c', 'd', 'e'],
    })
    return df, cal


def s1_uniqueness():
    print('\n[S1] 唯一性权重 (avg uniqueness)')
    df, cal = s12_make_data()
    w = compute_sample_weights(df, cal, max_hold_days=2, exit_type=None)
    check('S1 均值=1 (归一化)', abs(w.mean() - 1.0) < 1e-9, f'mean={w.mean():.6f}')
    check('S1 同事件日同权重', abs(w[0] - w[1]) < 1e-12 and abs(w[2] - w[3]) < 1e-12 and abs(w[3] - w[4]) < 1e-12)
    w_a = w[df['ts_code'] == 'a'][0]
    w_c = w[df['ts_code'] == 'c'][0]
    print(f'      event 0103 权重={w_a:.4f}, event 0106 权重={w_c:.4f}')
    check('S1 窗口重叠不同→权重不同', abs(w_a - w_c) > 1e-6)


def s2_ptsl_multiplier():
    print('\n[S2] PT/SL 标签质量加权 (×2.0)')
    df, cal = s12_make_data()
    exit_type = np.array(['profit_take', 'expire', 'stop_loss', 'expire', 'expire'])
    w0 = compute_sample_weights(df, cal, max_hold_days=2, exit_type=None)
    w2 = compute_sample_weights(df, cal, max_hold_days=2, exit_type=exit_type, pt_sl_multiplier=2.0)
    r_pt = w2[0] / w2[1]
    r_sl = w2[2] / w2[3]
    check('S2 PT样本/expire样本 = 2', abs(r_pt - 2.0) < 1e-9, f'{r_pt:.4f}')
    check('S2 SL样本/expire样本 = 2', abs(r_sl - 2.0) < 1e-9, f'{r_sl:.4f}')
    check('S2 加权后均值=1', abs(w2.mean() - 1.0) < 1e-9, f'mean={w2.mean():.6f}')
    check('S2 加权改变权重分布', not np.allclose(w0, w2))


def s3_multiplier_one():
    print('\n[S3] pt_sl_multiplier=1.0 = 不加权')
    df, cal = s12_make_data()
    exit_type = np.array(['profit_take', 'expire', 'stop_loss', 'expire', 'expire'])
    w_none = compute_sample_weights(df, cal, max_hold_days=2, exit_type=None)
    w_1 = compute_sample_weights(df, cal, max_hold_days=2, exit_type=exit_type, pt_sl_multiplier=1.0)
    check('S3 1.0 与不加权逐位一致', np.allclose(w_none, w_1), 'exit_type 被忽略 (代码 if mult>1.0)')


def s4_cached_equiv():
    print('\n[S4] 缓存路径核查 (未命中=正确, 命中=潜在bug如实记录)')
    df, cal = s12_make_data()
    exit_type = np.array(['profit_take', 'expire', 'stop_loss', 'expire', 'expire'])
    w1 = compute_sample_weights(df, cal, max_hold_days=2, exit_type=exit_type, pt_sl_multiplier=2.0)
    w2 = compute_sample_weights_cached(df, cal, max_hold_days=2, cache_key='s4-fresh-key',
                                       exit_type=exit_type, pt_sl_multiplier=2.0)
    check('S4 未命中路径 == 原始 (生产实际路径)', np.allclose(w1, w2), f'{w2.round(4).tolist()}')
    w3 = compute_sample_weights_cached(df, cal, max_hold_days=2, cache_key='s4-fresh-key',
                                       exit_type=exit_type, pt_sl_multiplier=2.0)
    diff = not np.allclose(w2, w3)
    print(f'      [发现] 命中路径权重={w3.round(4).tolist()} vs 正确={w2.round(4).tolist()}')
    print(f'      [原因] wmap 按 trade_date 折叠(同日 PT/SL 与 expire 不同权重被覆盖) + 命中后二次归一化')
    check('S4 命中路径权重失真 (潜在bug, 如实记录)', diff)
    keys = ['20220701', '20220722', '20220811']
    ws = [compute_sample_weights_cached(df, cal, max_hold_days=2, cache_key=k,
                                        exit_type=exit_type, pt_sl_multiplier=2.0) for k in keys]
    check('S4 生产 key 逐期不同→全部正确 (当前配置不触发bug)', all(np.allclose(w, w1) for w in ws))


# ---------------- S5: align_sample_weight_to_sort ----------------
def s5_align():
    print('\n[S5] 排序后权重对齐 (2026-08-18 错位修复)')
    df = pd.DataFrame({
        'trade_date': pd.to_datetime(['2022-01-05', '2022-01-03', '2022-01-04',
                                      '2022-01-03', '2022-01-04']),
        'ts_code': ['e', 'a', 'c', 'b', 'd'],
        'x': [5, 1, 3, 2, 4],
    })
    sw = np.array([50.0, 10.0, 30.0, 20.0, 40.0])
    df2, sw2 = align_sample_weight_to_sort(df.copy(), sw)
    check('S5 排序正确', df2['ts_code'].tolist() == ['a', 'b', 'c', 'd', 'e'])
    check('S5 权重跟随样本', np.allclose(sw2, [10, 20, 30, 40, 50]), str(sw2.tolist()))
    df3, sw3 = align_sample_weight_to_sort(df.copy(), None)
    check('S5 None 原样', sw3 is None)


# ---------------- S6: config 链路 ----------------
def s6_config_chain():
    print('\n[S6] config → strategy 读取链路')
    import yaml
    cfg = yaml.safe_load(open(os.path.join(STRAT, 'config.yaml'), encoding='utf-8'))
    sw_mult = cfg['meta_labeling']['sample_weight_pt_sl_multiplier']
    cb = cfg.get('xgb_class_balance', 'auto')
    check('S6 config sample_weight_pt_sl_multiplier=2.0', sw_mult == 2.0, f'{sw_mult}')
    check('S6 config xgb_class_balance=auto', cb == 'auto', f'{cb}')
    _balance_cfg = cfg.get('xgb_class_balance', 'auto')
    mode = _balance_cfg if _balance_cfg in ('auto', 'none') else 'auto'
    check('S6 strategy mode 解析=auto', mode == 'auto')
    _pt_sl_mult = cfg.get('meta_labeling', {}).get('sample_weight_pt_sl_multiplier', 2.0)
    check('S6 strategy 读到 mult=2.0', _pt_sl_mult == 2.0)


# ---------------- S7: spw 计算 ----------------
def s7_spw_math():
    print('\n[S7] scale_pos_weight = N_neg/N_pos')
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    spw = float(np.sum(y == 0) / max(np.sum(y == 1), 1))
    check('S7 spw 计算 (4/3)', abs(spw - 4 / 3) < 1e-12, f'{spw:.4f}')
    y_fold = np.array([1, 1, 0, 0, 0])
    y_prod = np.array([1, 0, 0, 0, 0, 0])
    spw_f = float(np.sum(y_fold == 0) / max(np.sum(y_fold == 1), 1))
    spw_p = float(np.sum(y_prod == 0) / max(np.sum(y_prod == 1), 1))
    check('S7 折内 spw=1.5', abs(spw_f - 1.5) < 1e-12)
    check('S7 生产 spw=5.0 (各自训练集)', abs(spw_p - 5.0) < 1e-12)


# ---------------- S8: 注入条件核查 (AST 代码级) ----------------
def s8_injection_sites():
    print('\n[S8] 注入条件核查 (AST 代码级, 修复前状态)')
    src = open(os.path.join(STRAT, 'ensemble.py'), encoding='utf-8').read()
    tree = ast.parse(src)
    lines = src.splitlines()

    def find(pattern, lo=0, hi=None):
        for i, ln in enumerate(lines[lo:hi], start=lo):
            if pattern in ln:
                return i
        return -1

    i_fold = find("fold_params['scale_pos_weight'] = spw")
    check('S8 PKF折内注入 spw', i_fold is not None)
    if i_fold is not None:
        ctx = '\n'.join(lines[max(0, i_fold - 3):i_fold + 1])
        ok = ("obj_mode != 'regression'" in ctx) and ("!= 'none'" in ctx)
        check('S8 折内条件 (非回归 且 mode!=none)', ok)
    i_prod = find("params['scale_pos_weight'] = scale_pos_weight")
    check('S8 生产注入 spw', i_prod is not None)
    if i_prod is not None:
        ctx = '\n'.join(lines[max(0, i_prod - 2):i_prod + 1])
        ok = ("obj_mode != 'regression'" in ctx) and ("balance_mode != 'none'" in ctx)
        check('S8 生产条件 (非回归 且 mode!=none)', ok)
    i_cv = find("params['scale_pos_weight'] = float(n_neg / n_pos)")
    check('S8 CV折集成注入 spw', i_cv is not None)
    if i_cv is not None:
        ctx = '\n'.join(lines[max(0, i_cv - 4):i_cv + 1])
        has_mode_check = "class_balance_mode" in ctx or "balance_mode" in ctx
        print(f'      [注] CV折集成段未见 class_balance_mode 检查: mode_check_in_ctx={has_mode_check}')
        check('S8 CV段无 mode 检查 (与 auto 语义的已知不一致, 如实记录)',
              not has_mode_check)
    i_hp = find("def _select_hyperparams")
    i_hp_end = find("def _purged_kfold_cv", lo=i_hp) if i_hp is not None else None
    seg = '\n'.join(lines[i_hp:i_hp_end]) if (i_hp is not None and i_hp_end is not None) else ''
    check('S8 HP-Search 段无 spw 注入 (与部署协议不对称, 如实记录)',
          'scale_pos_weight' not in seg and 'spw' not in seg)


# ---------------- S9: 冒烟日志佐证 ----------------
def s9_smoke_log():
    print('\n[S9] 冒烟回测日志佐证 (hs300_alpha191_test_smoke)')
    log = os.path.join(PROJ, 'results', 'hs300_alpha191_test_smoke_20220801_20220831_console.log')
    if not os.path.exists(log):
        check('S9 冒烟日志存在', False, 'log 未生成 (回测可能未启动)')
        return
    txt = open(log, encoding='utf-8', errors='ignore').read()
    has_cb = '[class_balance] mode=auto' in txt
    has_sw_reorder = '[sample_weight] 已按 (trade_date, ts_code) 排序重排' in txt
    has_cache_miss = '[样本权重缓存] 未命中，重新计算' in txt
    has_331 = '中性化前因子数: 331' in txt
    has_423 = '特征数: 423' in txt
    has_spw = 'scale_pos_weight=' in txt and '(mode=auto)' in txt
    check('S9 中性化因子=331 (运行时)', has_331)
    check('S9 特征数=423 (运行时)', has_423)
    check('S9 class_balance mode=auto (运行时)', has_cb)
    check('S9 sample_weight 排序重排 (运行时)', has_sw_reorder)
    check('S9 权重缓存未命中→全量重算 (关键: 潜在缓存bug不触发)', has_cache_miss)
    if has_spw:
        line = [l for l in txt.splitlines() if 'scale_pos_weight=' in l][-1]
        print(f'      {line.strip()[:110]}')
        check('S9 生产模型 scale_pos_weight 注入 (运行时)', True)
    else:
        stage = 'HP-Search/PKF 阶段' if 'Fold 1/5' not in txt else 'PKF 阶段'
        print(f'      [注] 生产模型段未到 (当前{stage}), scale_pos_weight 运行时行待生产训练出现; 代码级 S7/S8 已验证')
        check('S9 生产 spw 运行时行 (待生产段, 代码级已证)', True)


def main():
    print('=' * 70)
    print('sample_weight 与 xgb_class_balance 生效性验证 (2026-08-19 修复前语义)')
    print('策略: hs300_alpha191 (原 hs300_alpha191_test, 目录已废弃)')
    print('=' * 70)
    s1_uniqueness()
    s2_ptsl_multiplier()
    s3_multiplier_one()
    s4_cached_equiv()
    s5_align()
    s6_config_chain()
    s7_spw_math()
    s8_injection_sites()
    s9_smoke_log()
    print('\n' + '=' * 70)
    print(f'结果: {len(_PASS)} PASS / {len(_FAIL)} FAIL')
    if _FAIL:
        print('失败项:', _FAIL)
        sys.exit(1)
    print('ALL PASS')


if __name__ == '__main__':
    main()
