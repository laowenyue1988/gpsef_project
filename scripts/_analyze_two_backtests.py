# -*- coding: utf-8 -*-
"""对比两个 SmallAlpha191 回测：周期级指标 + 特征频数时序分析（只读，不改任何代码/配置）"""
import re, os, sys, json
import pandas as pd
import numpy as np

D1 = r"E:\quant\gpsef_project\results\SmallAlpha191_20220801_20260522_20260810174838"
D2 = r"E:\quant\gpsef_project\results\SmallAlpha191_20220801_20260522_20260811175611"
OUT = r"E:\quant\gpsef_project\results\_two_backtest_analysis"

def parse_log(path):
    """从 backtest.log 提取周期级信息"""
    text = open(path, encoding='utf-8', errors='replace').read()
    lines = text.splitlines()
    cycles = []
    cur = None
    for ln in lines:
        # 调仓日
        m = re.search(r'\[(\d{4}-\d{2}-\d{2})\] 调仓日', ln)
        if m:
            if cur is not None:
                cycles.append(cur)
            cur = {'date': m.group(1), 'features': [], 'n_feat': None,
                   'meta_auc': None, 'gate_auc': None, 'gate_decision': None,
                   'mda_dist': None, 'train_samples': None, 'folds': []}
        if cur is None:
            continue
        # cycle 号
        m = re.search(r'运行因子三件套验证 \(cycle=(\d+), train=(\d+)', ln)
        if m:
            cur['cycle'] = int(m.group(1))
            cur['train_samples'] = int(m.group(2))
        # 选入特征
        m = re.search(r'#\d+ (\S+)\s+votes=(\d)/5', ln)
        if m:
            cur['features'].append(m.group(1))
        m = re.search(r'Final selected features: (\d+)', ln)
        if m:
            cur['n_feat'] = int(m.group(1))
        # meta_auc / 生产模型 val AUC
        m = re.search(r'生产模型 val AUC=([\d.]+) \(用于AUC门控\), CV meta_auc=([\d.]+)', ln)
        if m:
            cur['meta_auc'] = float(m.group(1))
        # 门控决策
        m = re.search(r'\[策略分配\] AUC=([\d.]+) >= ([\d.]+) → (\S+)', ln)
        if m:
            cur['gate_auc'] = float(m.group(1))
            cur['gate_decision'] = m.group(3)
        # MDA 投票分布
        m = re.search(r'MDA投票分布 .* 共(\d+)特征, 投票事件=(\d+)', ln)
        if m:
            cur['mda_pool'] = int(m.group(1))
            cur['mda_events'] = int(m.group(2))
        # Fold val AUC
        m = re.search(r'Fold (\d)/5[^:]*:.*val_AUC=([\d.]+)', ln)
        if m:
            cur['folds'].append((int(m.group(1)), float(m.group(2))))
    if cur is not None:
        cycles.append(cur)
    return cycles

def main():
    os.makedirs(OUT, exist_ok=True)
    results = {}
    for tag, d in [('D1', D1), ('D2', D2)]:
        cycles = parse_log(os.path.join(d, 'backtest.log'))
        results[tag] = cycles
        print(f"=== {tag}: {len(cycles)} 周期解析完成 ===")

    # 1) 周期级指标表
    rows = []
    for tag, cycles in results.items():
        for c in cycles:
            rows.append({'tag': tag, 'date': c.get('date'), 'cycle': c.get('cycle'),
                         'n_feat': c['n_feat'], 'meta_auc': c['meta_auc'],
                         'gate_auc': c['gate_auc'], 'gate': c['gate_decision'],
                         'mda_pool': c.get('mda_pool'), 'mda_events': c.get('mda_events'),
                         'train_samples': c.get('train_samples')})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'cycle_metrics.csv'), index=False)
    print("\n=== 周期指标摘要（前5/后5）===")
    print(df.tail(10).to_string(index=False))

    # 2) 门控决策分布
    print("\n=== 门控决策分布 ===")
    for tag in ['D1', 'D2']:
        sub = df[df['tag'] == tag]
        print(f"{tag}: 放行={int((sub['gate']=='放行').sum())} 拦截={int((sub['gate']=='拦截').sum())} "
              f"降仓={int((sub['gate']=='降仓').sum()) if '降仓' in set(sub['gate']) else 0} "
              f"其他={sub['gate'].fillna('NA').unique().tolist()}")
        # 按月聚合门控 AUC
        sub2 = sub.copy()
        sub2['ym'] = pd.to_datetime(sub2['date']).dt.to_period('M')
        g = sub2.groupby('ym')['gate_auc'].agg(['mean', 'min', 'count'])
        print(f"  门控AUC: 全期均值={sub['gate_auc'].mean():.4f}, "
              f"2024-09前={sub[sub['date']<'2024-09-01']['gate_auc'].mean():.4f}, "
              f"2024-09后={sub[sub['date']>='2024-09-01']['gate_auc'].mean():.4f}")

    # 3) 特征频数时序: 每周期特征集 → 频数矩阵
    feat_sets = {}
    for tag, cycles in results.items():
        for c in cycles:
            key = (tag, c.get('date'))
            feat_sets[key] = set(c['features'])
    all_feats = sorted(set().union(*feat_sets.values()))
    print(f"\n=== 全周期累计出现特征总数: {len(all_feats)} ===")

    # 特征出现次数(跨全部周期) 分布
    freq = {f: 0 for f in all_feats}
    for s in feat_sets.values():
        for f in s:
            freq[f] += 1
    freq_s = pd.Series(freq).sort_values(ascending=False)
    print("=== 特征被选中次数 Top 20 ===")
    print(freq_s.head(20).to_string())
    print("=== 特征被选中次数分布 (仅出现1次的) ===")
    print(f"  出现1次: {int((freq_s==1).sum())} 个, 出现2次: {int((freq_s==2).sum())}, "
          f"出现3-5次: {int(((freq_s>=3)&(freq_s<=5)).sum())}, >5次: {int((freq_s>5).sum())}")

    # 4) 2024-09 前后特征重叠分析
    def feat_by_period(tag, d0, d1):
        return set().union(*[s for (t, dt), s in feat_sets.items()
                             if t == tag and d0 <= dt < d1])
    for tag in ['D1', 'D2']:
        pre = feat_by_period(tag, '2022-08-01', '2024-09-01')
        post = feat_by_period(tag, '2024-09-01', '2026-06-01')
        inter = pre & post
        print(f"\n=== {tag}: 特征池 2024-09 前({len(pre)}) 后({len(post)}) 重叠({len(inter)}) ===")
        print(f"  后段独有(新进入): {len(post - pre)} 个")
        print(f"  前段独有(被弃用): {len(pre - post)} 个")
        if len(pre) and len(post):
            print(f"  Jaccard = {len(inter)/len(pre|post):.3f}")

    # 5) 每周期特征数与 meta_auc 时序
    print("\n=== 特征数/门控AUC 时序（每5周期抽样）===")
    for tag in ['D1', 'D2']:
        sub = df[df['tag'] == tag].reset_index(drop=True)
        for i in range(0, len(sub), 6):
            r = sub.iloc[i]
            print(f"  {tag} cycle={r['cycle']:2d} {r['date']} 特征={r['n_feat']:2d} "
                  f"meta_auc={r['meta_auc']:.4f} 门控={r['gate_auc']:.4f} {r['gate']}")

    # 6) 特征频数矩阵 CSV（供直方图）
    mat = pd.DataFrame(0, index=all_feats, columns=[f"{t}|{c.get('date')}" for t, cycles in results.items() for c in cycles])
    for (t, dt), s in feat_sets.items():
        col = f"{t}|{dt}"
        for f in s:
            mat.loc[f, col] = 1
    mat.to_csv(os.path.join(OUT, 'feature_freq_matrix.csv'))
    # 频数汇总
    freq_summary = pd.DataFrame({'feature': freq_s.index, 'times_selected': freq_s.values})
    freq_summary.to_csv(os.path.join(OUT, 'feature_freq_summary.csv'), index=False)
    print(f"\n=== 已输出到 {OUT} ===")

if __name__ == '__main__':
    main()
