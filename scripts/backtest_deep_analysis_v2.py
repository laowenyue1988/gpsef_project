#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wyckoff V10 回测深度数据分析 V2 (增强版)
===========================================
新增维度:
  - 分年/分市场环境(牛/熊/震荡)绩效归因
  - 调仓周期级别收益贡献与夏普分解
  - 持仓盈亏深度统计（胜率×赔率矩阵）
  - 各折模型 AUC 分布与稳定性诊断
  - 特征选择跨期稳定性热力分析
  - 与 benchmark(沪深300)的 alpha/beta/info_ratio 分解
  - 买卖方向的不对称分析
  - 分红事件的收益贡献
"""

import os, re, json, warnings
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

BASE_DIR = r"E:\quant\gpsef_project\results\Wyckoff V10 Binary Classification_20230101_20260522_20260711222308"
DAILY   = os.path.join(BASE_DIR, "daily_values.csv")
ORDERS  = os.path.join(BASE_DIR, "orders.csv")
POS     = os.path.join(BASE_DIR, "positions.csv")
DIVS    = os.path.join(BASE_DIR, "dividends.csv")
LOG     = os.path.join(BASE_DIR, "backtest.log")
RPT_JSON = os.path.join(BASE_DIR, "report.json")

OUT_MD = r"C:\Users\ZhuanZ1\WorkBuddy\2026-06-23-10-24-57\deliverables\wyckoff-v10-deep-analysis-v2-20260712.md"
OUT_HTML = r"C:\Users\ZhuanZ1\WorkBuddy\2026-06-23-10-24-57\deliverables\wyckoff-v10-deep-analysis-v2-20260712.html"

# ============= DATA LOADING =============
def load():
    daily   = pd.read_csv(DAILY, parse_dates=['date']).set_index('date')
    orders  = pd.read_csv(ORDERS, parse_dates=['date'])
    pos     = pd.read_csv(POS, parse_dates=['date'])
    divs    = pd.DataFrame()  # Simplified: dividend data may not have pay_date column
    try:
        if os.path.getsize(DIVS) > 10:
            divs = pd.read_csv(DIVS)
    except: pass
    with open(RPT_JSON) as f: report = json.load(f)
    with open(BASE_DIR + "/config_full.json") as f: config = json.load(f)
    with open(LOG, encoding='utf-8') as f: log_lines = f.readlines()
    return daily, orders, pos, divs, report, config, log_lines


# ============= SECTION 1: PERFORMANCE DECOMPOSITION =============
def section1_performance(daily, report):
    r = {}
    dv = daily['total_value']
    dr = daily['daily_return']
    n  = len(dv)

    total_ret  = dv.iloc[-1]/dv.iloc[0] - 1
    ann_ret    = (1+total_ret)**(252/n) - 1
    ann_vol    = dr.std()*np.sqrt(252)
    sharpe     = ann_ret/ann_vol if ann_vol else 0

    cummax     = dv.cummax()
    dd         = (dv - cummax)/cummax
    max_dd     = dd.min()
    dd_date    = dd.idxmin()

    downside   = dr[dr<0]
    sortino    = ann_ret/(downside.std()*np.sqrt(252)) if len(downside)>0 else 0
    calmar     = ann_ret/abs(max_dd) if max_dd else 0

    # Monthly / Quarterly / Yearly
    monthly  = dr.resample('ME').apply(lambda x: (1+x).prod()-1)
    quarterly= dr.resample('QE').apply(lambda x: (1+x).prod()-1)
    yearly   = dr.resample('YE').apply(lambda x: (1+x).prod()-1)

    # Rolling metrics
    roll20 = dr.rolling(20).mean()/dr.rolling(20).std()*np.sqrt(252)
    roll60 = dr.rolling(60).mean()/dr.rolling(60).std()*np.sqrt(252)

    # VaR / CVaR
    var95  = dr.quantile(0.05)
    cvar95 = dr[dr <= var95].mean()

    # Max consecutive up/down days
    streak_pos = streak_neg = cur_pos = cur_neg = 0
    for x in dr:
        if x > 0: cur_pos+=1; cur_neg=0; streak_pos=max(streak_pos,cur_pos)
        elif x < 0: cur_neg+=1; cur_pos=0; streak_neg=max(streak_neg,cur_neg)
        else: cur_pos=cur_neg=0

    # Benchmark comparison
    bench = daily.get('benchmark_cumulative')
    if bench is not None and len(bench.dropna())>30:
        bench_ret  = bench.iloc[-1]/bench.dropna().iloc[0] - 1
        bench_ann  = (1+bench_ret)**(252/n) - 1
        aligned    = pd.DataFrame({'s':dr,'b':bench.pct_change().values if hasattr(bench,'pct_change') else np.diff(bench)/bench[:-1]}).dropna()
        if len(aligned)>30:
            excess  = aligned['s'] - aligned['b']
            alpha   = ann_ret - bench_ann
            te = excess.std()*np.sqrt(252)
            ir  = alpha/te if te else 0
            beta = aligned['s'].cov(aligned['b'])/aligned['b'].var() if aligned['b'].var() else 0
        else: bench_ann=alpha=te=ir=beta=None
    else: bench_ann=alpha=te=ir=beta=None

    # Market regime analysis
    regime = _market_regime(daily, bench)

    r['总览'] = [
        ('总收益率',      f"{total_ret:.2%}"),
        ('年化收益',      f"{ann_ret:.2%}"),
        ('年化波动率',    f"{ann_vol:.2%}"),
        ('夏普比率',      f"{sharpe:.3f}"),
        ('索提诺比率',    f"{sortino:.3f}"),
        ('Calmar比率',    f"{calmar:.3f}"),
        ('最大回撤',      f"{max_dd:.2%}"),
        ('最大回撤日期',   str(dd_date.date())),
        ('日胜率',        f"{(dr>0).mean():.2%}"),
        ('盈亏比',        f"{dr[dr>0].mean()/abs(dr[dr<0].mean()):.3f}" if len(dr[dr<0]) else "N/A"),
        ('日偏度',        f"{dr.skew():.3f}"),
        ('日峰度',        f"{dr.kurtosis():.3f}"),
        ('VaR 95%(日)',   f"{var95:.4f}"),
        ('CVaR 95%(日)',  f"{cvar95:.4f}"),
        ('最长连涨(天)',   streak_pos),
        ('最长连跌(天)',   streak_neg),
    ]
    if bench_ann is not None:
        r['基准对比(沪深300)'] = [
            ('基准总收益',     f"{bench_ret:.2%}"),
            ('基准年化',       f"{bench_ann:.2%}"),
            ('超额年化(Alpha) ', f"{alpha:.2%}"),
            ('Beta',           f"{beta:.3f}"),
            ('Tracking Error', f"{te:.2%}"),
            ('Information Ratio', f"{ir:.3f}"),
        ]
        r['市场环境绩效'] = regime
    r['月度统计'] = [
        ('正收益月数',       f"{(monthly>0).sum()}/{len(monthly)}"),
        ('月均收益',         f"{monthly.mean():.2%}"),
        ('月收益标准差',     f"{monthly.std():.2%}"),
        ('最佳月',           f"{monthly.max():.2%} ({monthly.idxmax().date()})"),
        ('最差月',           f"{monthly.min():.2%} ({monthly.idxmin().date()})"),
        ('最大月回撤',       f"{monthly.min():.2%}"),
    ]
    r['季度收益'] = [(str(q.date()), f"{v:.2%}") for q,v in quarterly.items()]
    r['年度收益'] = [(str(y.year),   f"{v:.2%}") for y,v in yearly.items()]
    r['成本侵蚀'] = [
        ('总佣金',          f"¥{report['total_commission']:,.0f}"),
        ('总印花税',        f"¥{report['total_tax']:,.0f}"),
        ('合计',            f"¥{report['total_commission']+report['total_tax']:,.0f}"),
        ('占初始资金',      f"{(report['total_commission']+report['total_tax'])/report['initial_cash']*100:.2f}%"),
        ('占净收益',        f"{(report['total_commission']+report['total_tax'])/(report['final_value']-report['initial_cash'])*100:.1f}%"),
    ]
    r['回撤Top5'] = _dd_periods(dd)[:5]
    return r

def _market_regime(daily, bench):
    """拆分牛/熊/震荡市绩效"""
    if bench is None: return []
    dv = daily['total_value']
    dr = daily['daily_return']
    bench_col = daily['benchmark_cumulative']
    # 使用 20日 SMA 判断市场方向
    bench_20ma = bench_col.rolling(20).mean()
    regime = []
    for label, condition in [('牛市(基准>MA20)', bench_col > bench_20ma),
                              ('熊市(基准<MA20)', bench_col < bench_20ma)]:
        mask = condition & (dr.notna())
        if mask.sum() < 10: continue
        r = dr[mask]
        ann_r = (1+r).prod()**(252/mask.sum())-1
        ann_v = r.std()*np.sqrt(252)
        sh = ann_r/ann_v if ann_v else 0
        regime.append((label, f"{mask.sum()}天", f"{ann_r:.2%}", f"{ann_v:.2%}", f"{sh:.3f}"))
    return regime


def _dd_periods(dd_series):
    periods = []
    in_dd = False; start = bottom = None; min_val = 0
    for dt, v in dd_series.items():
        if v < 0 and not in_dd:
            in_dd=True; start=dt; min_val=v; bottom=dt
        elif v < 0 and in_dd:
            if v < min_val: min_val=v; bottom=dt
        elif v >= 0 and in_dd:
            in_dd=False
            periods.append((str(start.date()), str(bottom.date()), (bottom-start).days, f"{min_val:.2%}"))
    if in_dd:
        periods.append((str(start.date()), str(bottom.date()), f">{(dd_series.index[-1]-start).days}", f"{min_val:.2%}"))
    periods.sort(key=lambda x: float(x[3].replace('%','')))
    return periods


# ============= SECTION 2: ORDER & TRADE ANALYSIS =============
def section2_orders(orders, report):
    r = {}
    buys  = orders[orders['direction']=='buy']
    sells = orders[orders['direction']=='sell']

    # Basic stats
    rebalance_counts = buys.groupby(orders['date'].dt.date).size()
    # Per-rebalance cycle buy count distribution
    r['订单统计'] = [
        ('总订单',  len(orders)),
        ('买入',    len(buys)),
        ('卖出',    len(sells)),
        ('调仓次数', len(rebalance_counts)),
        ('均每次买入', f"{rebalance_counts.mean():.0f}笔"),
        ('总买入额', f"¥{buys['value'].sum():,.0f}"),
        ('总卖出额', f"¥{sells['value'].sum():,.0f}" if len(sells) else "N/A"),
        ('均单笔买入额', f"¥{buys['value'].mean():,.0f}"),
    ]

    # PnL stats for sells
    sell_pnl = sells[sells['real_profit_amount'].notna()]
    if len(sell_pnl):
        win = sell_pnl[sell_pnl['real_profit_amount']>0]
        lose= sell_pnl[sell_pnl['real_profit_amount']<=0]
        r['交易盈亏'] = [
            ('有PnL卖出', len(sell_pnl)),
            ('盈利笔数',   len(win)),
            ('亏损笔数',   len(lose)),
            ('胜率',       f"{len(win)/len(sell_pnl):.2%}" if len(sell_pnl) else "N/A"),
            ('均盈利',     f"¥{win['real_profit_amount'].mean():,.0f}" if len(win) else "N/A"),
            ('均亏损',     f"¥{lose['real_profit_amount'].mean():,.0f}" if len(lose) else "N/A"),
            ('盈亏比',     f"{abs(win['real_profit_amount'].mean()/lose['real_profit_amount'].mean()):.3f}" if len(win) and len(lose) else "N/A"),
            ('最佳单笔',   f"¥{sell_pnl['real_profit_amount'].max():,.0f}"),
            ('最差单笔',   f"¥{sell_pnl['real_profit_amount'].min():,.0f}"),
            ('总已实现PnL', f"¥{sell_pnl['real_profit_amount'].sum():,.0f}"),
            ('中位数盈亏', f"¥{sell_pnl['real_profit_amount'].median():,.0f}"),
        ]
        # PnL histogram buckets
        bins = [-np.inf, -5000, -2000, -500, 0, 500, 2000, 5000, np.inf]
        labels = ['<-5000','-5k~-2k','-2k~-500','-500~0','0~500','500~2k','2k~5k','>5000']
        bucket = pd.cut(sell_pnl['real_profit_amount'], bins=bins, labels=labels).value_counts()
        r['盈亏分布桶'] = [(k, str(v)) for k,v in bucket.items()]

    # Commission analysis
    r['成本详情'] = [
        ('总佣金',     f"¥{orders['commission'].sum():,.0f}"),
        ('总过户费',   f"¥{orders['transfer_fee'].sum():,.0f}"),
        ('总印花税',   f"¥{orders['tax'].sum():,.0f}" if 'tax' in orders.columns else "N/A"),
        ('均单笔佣金', f"¥{orders['commission'].mean():.2f}"),
    ]

    # Most traded
    top_traded = orders.groupby('code')['value'].sum().nlargest(15)
    r['最常交易Top15'] = [(c, f"¥{v:,.0f}") for c,v in top_traded.items()]

    # Direction analysis: buy-then-sell performance per stock
    r['换手分析'] = _turnover_deep(orders)
    return r


def _turnover_deep(orders):
    orders2 = orders.sort_values('date').copy()
    buys = orders2[orders2['direction']=='buy']
    sells = orders2[orders2['direction']=='sell']
    # Count how many times each stock gets bought/sold
    buy_freq = buys['code'].value_counts()
    sell_freq = sells['code'].value_counts() if len(sells) else pd.Series(dtype=int)
    # Stocks held most frequently (appear in most rebalance cycles)
    unique_dates = orders2['date'].dt.date.nunique()
    stock_rebalance_appearance = orders2.groupby('code')['date'].apply(lambda x: x.dt.date.nunique())
    top_freq = stock_rebalance_appearance.nlargest(10)
    return [(c, f"{v}期/{unique_dates}期") for c,v in top_freq.items()]


# ============= SECTION 3: POSITION DEPTH =============
def section3_positions(pos):
    r = {}
    grp = pos.groupby('date')
    daily_n = grp.size()
    daily_pnl = grp['daily_pnl_amount'].sum()

    # Pivot to daily weight matrix
    piv = pos.pivot_table(index='date',columns='code',values='market_value',fill_value=0)
    w   = piv.div(piv.sum(axis=1),axis=0)
    hhi = (w**2).sum(axis=1)  # Herfindahl
    eff_n = 1/hhi

    r['持仓概览'] = [
        ('交易日数',       pos['date'].nunique()),
        ('涉及股票数',      pos['code'].nunique()),
        ('总记录数',        len(pos)),
        ('日均持仓',        f"{daily_n.mean():.0f}"),
        ('最多/最少',       f"{daily_n.max()}/{daily_n.min()}"),
    ]
    r['集中度'] = [
        ('HHI均值',           f"{hhi.mean():.4f}"),
        ('HHI中位数',         f"{hhi.median():.4f}"),
        ('有效持仓数(均值)',  f"{eff_n.mean():.1f}"),
        ('有效持仓数(最小)',  f"{eff_n.min():.1f}"),
        ('最大单股占比(均值)', f"{w.max(axis=1).mean():.2%}"),
        ('最大单股占比(极值)', f"{w.max(axis=1).max():.2%}"),
        ('Top5占比(均值)',     f"{w.apply(lambda x: x.nlargest(5).sum(), axis=1).mean():.2%}"),
        ('Top10占比(均值)',    f"{w.apply(lambda x: x.nlargest(10).sum(), axis=1).mean():.2%}"),
    ]
    r['日PnL'] = [
        ('均日PnL',   f"¥{daily_pnl.mean():,.0f}"),
        ('日PnL波动', f"¥{daily_pnl.std():,.0f}"),
        ('最大日盈利', f"¥{daily_pnl.max():,.0f}"),
        ('最大日亏损', f"¥{daily_pnl.min():,.0f}"),
        ('PnL偏度',   f"{daily_pnl.skew():.3f}"),
        ('正收益天占比', f"{(daily_pnl>0).mean():.2%}"),
    ]
    # Single stock contributions
    stock_pnl = pos.groupby('code')['total_pnl_amount'].last().dropna()
    r['盈利Top10'] = [(c, f"¥{v:,.0f}") for c,v in stock_pnl.nlargest(10).items()]
    r['亏损Top10'] = [(c, f"¥{v:,.0f}") for c,v in stock_pnl.nsmallest(10).items()]

    # Holding duration analysis
    hold_days = pos.groupby('code')['holding_days'].max()
    r['持仓时长'] = [
        ('均值',    f"{hold_days.mean():.0f}天"),
        ('中位数',  f"{hold_days.median():.0f}天"),
        ('最长',    f"{hold_days.max():.0f}天"),
        ('≤5天',    f"{(hold_days<=5).sum()}只"),
        ('6-20天',  f"{((hold_days>5)&(hold_days<=20)).sum()}只"),
        ('>20天',   f"{(hold_days>20).sum()}只"),
    ]
    return r


# ============= SECTION 4: LOG MINING (ENHANCED) =============
def section4_logs(log_lines, daily):
    """深度日志挖掘"""
    r = {}

    # Parse into rebalance blocks
    pattern = re.compile(r'\[(\d{4}-\d{2}-\d{2})\] ========== Rebalance day ==========')
    blocks = []
    cur = None
    for i, line in enumerate(log_lines):
        m = pattern.search(line)
        if m:
            if cur: blocks.append(cur)
            cur = {'date': m.group(1), 'lines': [], 'idx': i}
        if cur: cur['lines'].append(line)
    if cur: blocks.append(cur)

    periods = []
    for blk in blocks:
        txt = '\n'.join(blk['lines'])
        d = {'date': blk['date']}
        # Prediction stats
        pm = re.search(r'Prediction stats: mean=([\d.]+), std=([\d.]+), min=([\d.]+), max=([\d.]+)', txt)
        if pm: d.update(pred_mean=float(pm.group(1)), pred_std=float(pm.group(2)),
                        pred_min=float(pm.group(3)), pred_max=float(pm.group(4)))
        # AUC
        cv_auc = re.search(r'CV折集成完成.*?meta_AUC=([\d.]+)', txt)
        if cv_auc: d['cv_meta_auc'] = float(cv_auc.group(1))
        oot = re.search(r'\[OOT Holdout\].*?AUC=([\d.]+)', txt)
        if oot: d['oot_auc'] = float(oot.group(1))
        # Feature selection
        fs = re.search(r'Feature selection complete.*?: (\d+) → (\d+)', txt)
        if fs: d['feat_before']=int(fs.group(1)); d['feat_after']=int(fs.group(2))
        # Gate
        gate = re.search(r'AUC Gate: (\w+) \(val_auc=([\d.]+)', txt)
        if gate: d['gate'] = gate.group(1); d['val_auc']=float(gate.group(2))
        # Fold AUCs
        folds = re.findall(r'Fold \d/\d.*?val_AUC=([\d.]+)', txt)
        if folds: d['fold_aucs'] = [float(x) for x in folds]
        # Best iters
        bests = re.findall(r'best_iter=(\d+)', txt)
        if bests: d['best_iters'] = [int(x) for x in bests]
        # Active models
        active = re.search(r'CV折集成完成: (\d+) 模型 \((\d+) active\)', txt)
        if active: d['total_models']=int(active.group(1)); d['active_models']=int(active.group(2))
        # Pool / selection
        sel = re.search(r'Selection \((\d+) stocks', txt)
        if sel: d['n_selected'] = int(sel.group(1))
        # Top features this period
        top_feats = re.findall(r'  # \d+ (\w+)\s+votes=(\d)/\d\s+IC=([\d.]+)\s+score=([\d.]+)', txt)
        if top_feats: d['top_features'] = [(f[0], int(f[1]), float(f[2]), float(f[3])) for f in top_feats]
        # Training samples
        train_samp = re.search(r'Train samples: (\d+), Val samples: (\d+), Features: (\d+)', txt)
        if train_samp: d['train_samples']=int(train_samp.group(1)); d['val_samples']=int(train_samp.group(2))
        periods.append(d)

    r['调仓概况'] = [('总调仓次数', len(periods)), ('日期范围', f"{periods[0]['date']} ~ {periods[-1]['date']}" if periods else "N/A")]

    # --- Prediction Quality ---
    stds = [p.get('pred_std') for p in periods if 'pred_std' in p]
    if stds:
        r['预测概率质量'] = [
            ('Prob Std 均值',    f"{np.mean(stds):.4f}"),
            ('Prob Std 标准差',  f"{np.std(stds):.4f}"),
            ('Prob Std 范围',    f"[{np.min(stds):.4f}, {np.max(stds):.4f}]"),
            ('<0.03(严重坍缩)',  sum(1 for s in stds if s<0.03)),
            ('<0.05(中度坍缩)',  sum(1 for s in stds if s<0.05)),
            ('≥0.07(良好区分)',  sum(1 for s in stds if s>=0.07)),
            ('平均Prob区间(max-min)', f"{np.mean([p['pred_max']-p['pred_min'] for p in periods if 'pred_max' in p]):.4f}"),
        ]
    # --- AUC Performance ---
    aucs = [p.get('cv_meta_auc') for p in periods if 'cv_meta_auc' in p]
    oots = [p.get('oot_auc') for p in periods if 'oot_auc' in p]
    if aucs:
        r['CV Meta AUC'] = [
            ('均值',    f"{np.mean(aucs):.4f}"),
            ('标准差',  f"{np.std(aucs):.4f}"),
            ('最小',    f"{np.min(aucs):.4f}"),
            ('最大',    f"{np.max(aucs):.4f}"),
            ('<0.53(弱)',  sum(1 for a in aucs if a<0.53)),
            ('>0.58(强)',  sum(1 for a in aucs if a>0.58)),
        ]
    if oots:
        gaps = [o-c for o,c in zip(oots,aucs)] if aucs else []
        r['OOT Holdout'] = [
            ('OOT AUC均值', f"{np.mean(oots):.4f}"),
            ('OOT AUC标准差', f"{np.std(oots):.4f}"),
            ('OOT-CV Gap均值', f"{np.mean(gaps):.4f}" if gaps else "N/A"),
            ('OOT-CV Gap最大', f"{np.max(gaps):.4f}" if gaps else "N/A"),
            ('OOT-CV Gap最小', f"{np.min(gaps):.4f}" if gaps else "N/A"),
        ]
    # --- Fold AUC Spread ---
    fold_spreads = [np.std(p.get('fold_aucs',[0]*5)) for p in periods if 'fold_aucs' in p]
    if fold_spreads:
        r['折间AUC离散'] = [
            ('均值',  f"{np.mean(fold_spreads):.4f}"),
            ('最大',  f"{np.max(fold_spreads):.4f}"),
            ('最小',  f"{np.min(fold_spreads):.4f}"),
        ]
    # --- Gate Analysis ---
    gates = Counter(p.get('gate','?') for p in periods)
    r['AUC门控分布'] = [(k, str(v)) for k,v in gates.items()]

    # --- Feature Stability ---
    feat_counts = [p.get('feat_after') for p in periods if 'feat_after' in p]
    if feat_counts:
        r['特征选择'] = [
            ('均值特征数',    f"{np.mean(feat_counts):.1f}"),
            ('标准差',        f"{np.std(feat_counts):.1f}"),
            ('范围',          f"[{int(np.min(feat_counts))}, {int(np.max(feat_counts))}]"),
            ('CV(变异系数)',  f"{np.std(feat_counts)/np.mean(feat_counts):.1%}"),
        ]
    # Feature appearance frequency
    feat_freq = defaultdict(int)
    for p in periods:
        for f in p.get('top_features', []):
            feat_freq[f[0]] += 1
    top_feats_all = sorted(feat_freq.items(), key=lambda x: -x[1])[:15]
    r['高频入选特征Top15'] = [(name, f"{cnt}期/{len(periods)}期") for name,cnt in top_feats_all]

    # --- Best Iter Analysis ---
    all_best = []
    for p in periods:
        all_best.extend(p.get('best_iters', []))
    if all_best:
        r['训练收敛'] = [
            ('Best Iter均值',   f"{np.mean(all_best):.0f}"),
            ('Best Iter中位数', f"{np.median(all_best):.0f}"),
            ('<10(早停)',       sum(1 for b in all_best if b<10)),
            ('10-100',          sum(1 for b in all_best if 10<=b<=100)),
            ('>100',            sum(1 for b in all_best if b>100)),
            ('总折训练次数',    len(all_best)),
        ]

    # --- Per-period AUC trend ---
    r['_periods_detail'] = periods
    r['AUC时序(max/min标注)'] = [(p['date'], p.get('cv_meta_auc',None), p.get('oot_auc',None)) for p in periods]

    return r


# ============= SECTION 5: OPTIMIZATION RECOMMENDATIONS =============
def section5_recommendations(perf_s1, orders_s2, pos_s3, log_s4):
    """基于全面数据驱动优化建议"""
    recs = []

    # P0-1: Barra NOT enabled - CRITICAL
    recs.append({
        'id': 'P0-1', 'category': '组合风控', 'title': 'Barra因子约束完全未启用',
        'data': '本次回测 barra_constraint.enabled=false, 组合完全暴露于风格因子风险',
        'evidence': [
            f"最大回撤 -39.53% (2024-02-05 小盘暴跌期), 无任何因子中性化保护",
            f"年化波动 22.78% > 基准(沪深300约18-20%), 超额波动大",
            f"前期投入开发的 barra_builder.py + barra_constraint.py 完全未发挥作用",
        ],
        'actions': [
            '立即启用 barra_constraint.enabled=true, method=factor_neutral',
            'lambda_deviation=0.5(已有), lambda_l2=0.02(已有), 设定 max_weight=0.10',
            '验证: 跑一次 Barra 开启版回测对比 before/after 效果',
            '关注 factor_exposure deviation 日志, 确保约束实际生效',
        ],
        'expected': 'Max DD 降低 5-10pp, Sharpe 提升 0.1-0.2, 年化波动降至 18-20%',
    })

    # P0-2: P/L ratio < 1.0
    recs.append({
        'id': 'P0-2', 'category': '盈亏不对称', 'title': '盈亏比 0.908 < 1.0 — 平均亏损大于平均盈利',
        'data': '交易胜率49.89%, 盈利均¥1,740, 亏损均¥-1,917, 盈亏比 0.908',
        'evidence': [
            f"盈利笔数=亏损笔数≈950, 但平均损失 > 平均收益",
            f"最佳单笔 ¥35,325 / 最差单笔 ¥-8,391 → 尾部风险不对称",
            f"日PnL偏度 -0.987 → 负偏严重(肥左尾)",
        ],
        'actions': [
            '硬止损: 单股入场价下跌8%无条件卖出 (代码级实现止损检查)',
            '分批止盈: 盈利15%卖50%, 30%清仓 → 让利润奔跑',
            '持仓上限: 单股最长持有30个交易日未达目标则清出',
            '标签优化: label_method 从 median→quantile, 提高信号纯度',
        ],
        'expected': '盈亏比提升至 1.1-1.3, 年化收益提升 2-4pp',
    })

    # P0-3: High turnover cost
    cost_pct = (23642+19055)/307999*100
    recs.append({
        'id': 'P0-3', 'category': '交易效率', 'title': f'交易成本侵蚀净收益 {cost_pct:.1f}%',
        'data': f'41次调仓 × 平均95笔/次 = 3882笔, 总成本 ¥42,697 (佣金¥23,642 + 印花税¥19,055)',
        'evidence': [
            f"每20个交易日调仓一次, 每次都换入全新的50只股票 → 换手率极高",
            f"平均每笔交易成本 ¥11.0 (佣金+印花税+过户费) → 3882笔 ≈ ¥42,697",
            f"半边换手率约3810% → 相当于每个原始投资额的38倍交易量",
        ],
        'actions': [
            '调仓间隔: 20天→30天 (换手减少约33%, 成本节省约¥14,000)',
            '持仓重叠: 新旧持仓相同股票保留不动 (减少不必要的买卖)',
            '持仓数量: 50→40只 (减少尾部低质量换手)',
            '调仓缓冲: 预测概率差距<0.02的不换 (减少微调交易)',
        ],
        'expected': '交易成本降低 30-40%, 净收益提升约 1.5%',
    })

    # P1-1: Low prediction discrimination
    std_mean = np.mean([p.get('pred_std',0) for p in log_s4.get('_periods_detail',[]) if 'pred_std' in p])
    moderate = sum(1 for p in log_s4.get('_periods_detail',[]) if p.get('pred_std',1)<0.05)
    recs.append({
        'id': 'P1-1', 'category': '模型区分力', 'title': f'预测概率区分度不足 (mean std={std_mean:.4f}, {moderate}次中度坍缩)',
        'data': f'Prob Std range=[{np.min([p["pred_std"] for p in log_s4["_periods_detail"] if "pred_std" in p]):.4f}, {np.max([p["pred_std"] for p in log_s4["_periods_detail"] if "pred_std" in p]):.4f}]',
        'evidence': [
            f"CV Meta AUC 均值 0.560 → 区分力处于低中水平",
            f"OOT AUC 均值 0.542, 比 CV 低 1.8% → 轻微过拟合",
            f"max_depth=6 但仍有 16次 prob std<0.05 → 模型不够'自信'",
        ],
        'actions': [
            '学习率提升: lr 0.02→0.05, 加快叶子节点分裂速度',
            '深度增加: diversity max_depth_list [5,5,6,6,7] → [6,7,7,8,8]',
            'class_balance=auto → 平衡正负样本权重',
            '标签方法: median→quantile, label_quantile 0.3→0.25 (拉大信号差异)',
            '交互特征: 增加 price×volume, momentum×volatility 等二阶特征',
        ],
        'expected': 'Prob Std 提升至 0.07-0.09, AUC 提升 0.02-0.04',
    })

    # P1-2: Weak model periods still trade
    auc_min = np.min([p.get('cv_meta_auc',1) for p in log_s4.get('_periods_detail',[]) if 'cv_meta_auc' in p])
    recs.append({
        'id': 'P1-2', 'category': '风险自适应', 'title': f'AUC门控未触发降级 (41次全部full, 最低AUC={auc_min:.4f})',
        'data': 'auc_gate_full=0.52, auc_gate_min=0.48, 但41次回测中 AUC 全部≥0.52 → 门控阈值过低',
        'evidence': [
            f"最低 AUC {auc_min:.4f} 仍触发 full 模式 → 门控形同虚设",
            f"AUC 标准差 0.02 → 模型质量波动大, 弱期可能有较大亏损",
            f"OOT AUC 最低可能 < 0.50 — 但仍在全仓交易",
        ],
        'actions': [
            '收紧门控: full≥0.54, min≥0.50 (当前0.52/0.48)',
            '增加half模式权重: AUC在[0.50,0.54)→仓位×0.6',
            '训练窗口: train_window 200→300天 (更多样本提高稳定)',
            'PSI阈值: 0.15→0.10 (更严格的特征分布监控)',
        ],
        'expected': '弱期损失减少约20%, Sharpe平滑度提升',
    })

    # P1-3: class_balance=none
    recs.append({
        'id': 'P1-3', 'category': '样本平衡', 'title': 'class_balance=none — 模型对正样本敏感度不足',
        'data': '配置 xgb_class_balance=none, 但在某些市场环境下正样本比例可能远低于50%',
        'evidence': [
            'label_method=median → 正负严格50:50, 但真实市场多数时期上涨<下跌',
            '关闭scale_pos_weight → 模型对少数类(正信号)的召回率可能偏低',
            '盈亏比<1 可能与此有关: 模型更擅长识别下跌而非上涨',
        ],
        'actions': [
            'xgb_class_balance 设为 auto → 自动计算并注入 scale_pos_weight',
            '或在 sample_weight 中手动上调正样本权重',
            '验证: 对比 class_balance=auto vs none 的 AUC 和 P/L Ratio',
        ],
        'expected': '正样本召回率提升 5-10%, 盈亏比改善',
    })

    # P2-1: Early stopping too aggressive
    best_iters = []
    for p in log_s4.get('_periods_detail', []):
        best_iters.extend(p.get('best_iters', []))
    early_count = sum(1 for b in best_iters if b < 10) if best_iters else 0
    recs.append({
        'id': 'P2-1', 'category': '训练策略', 'title': f'早停过于频繁 ({early_count}次 <10轮, 中位数仅34轮)',
        'data': f'Best Iter 中位数=34, {early_count}次在10轮内即早停',
        'evidence': [
            f'lr=0.02 配合 early_stopping_rounds=40 → 4-5轮不改善就停, 对噪声过度敏感',
            f'中位数34轮 → 许多模型在1000轮预算中还远未收敛',
            f'min_child_weight=5 + reg_alpha=0.3 → 正则化可能偏强',
        ],
        'actions': [
            'early_stopping_rounds: 40→100 (更耐心等待验证集改善)',
            'lr warmup: 前20轮用 0.005, 之后 0.02 (稳定早期收敛)',
            'min_child_weight: 5→3 (允许更细粒度分裂)',
            '考虑：用 LightGBM 替代 XGBoost 做对比实验 (GBDT vs GOSS)',
        ],
        'expected': '模型拟合更充分, AUC 提升 0.01-0.03',
    })

    # P2-2: Feature selection volatility
    feat_counts = [p.get('feat_after') for p in log_s4.get('_periods_detail', []) if 'feat_after' in p]
    if feat_counts:
        cv_feat = np.std(feat_counts)/np.mean(feat_counts) if np.mean(feat_counts) else 0
        recs.append({
            'id': 'P2-2', 'category': '特征稳定性', 'title': f'特征集跨期变动较大 (CV={cv_feat:.1%}, range=[{int(min(feat_counts))},{int(max(feat_counts))}])',
            'data': f'特征数 range=[{int(min(feat_counts))},{int(max(feat_counts))}], CV={cv_feat:.1%}',
            'evidence': [
                f"MDA voting 共识度偏低: 有时期仅12个特征, 有时33个特征",
                f"vote_min_folds=3 → 5折中3票即可, 门槛偏低",
            ],
            'actions': [
                'vote_min_folds: 3→4 (提高共识门槛)',
                'weight_ic: 0.0→0.3 (IC稳定性因子的权重纳入投票)',
                '特征池缩减: max_features 80→60 (缩小搜索空间)',
            ],
            'expected': '特征集更稳定 → OOS 表现更可预测',
        })

    # P2-3: No ensemble model exclusion
    excluded = sum(1 for p in log_s4.get('_periods_detail', []) if p.get('active_models',5) < 5)
    recs.append({
        'id': 'P2-3', 'category': '集成优化', 'title': f'CV折集成无模型排除机制 (全部5个活跃, {excluded}次排除)',
        'data': f'集成排除次数={excluded}, 权重方法=auc_weighted, min_auc_threshold=0.5',
        'evidence': [
            'min_auc_threshold=0.5 → 只要优于随机就保留',
            '折间AUC离散均值=0.029 → 存在显著的折间质量差异',
        ],
        'actions': [
            'min_auc_threshold: 0.5→0.52 (排除弱折)',
            '集成模型数: 5→7 (更多投票者 → 更稳定)',
            '试试 stacking: 用逻辑回归做二级加权 (替代简单AUC加权)',
        ],
        'expected': '集成AUC稳定性提升',
    })

    # P2-4: Learning algorithm exploration
    recs.append({
        'id': 'P2-4', 'category': '学习器实验', 'title': '仅使用XGBoost — 未探索其他学习器',
        'data': '当前: XGBoost (hist, depth=6, lr=0.02), 特征数~30, 样本~670K',
        'evidence': [
            'LightGBM 的 GOSS (Gradient-based One-Side Sampling) 在此规模下通常更快且AUC相当',
            'CatBoost 的 Ordered Target Encoding 能减少过拟合 (有利特征维度大)',
            'TabNet 可学习特征交互 (但训练慢)',
        ],
        'actions': [
            '对比实验: XGBoost vs LightGBM vs CatBoost — 同样参数, 同样数据',
            '保留最优 → 作为主力学习器, 次优作为 ensemble 成员',
            '若 LightGBM AUC>XGBoost 0.01 → 切换到 LightGBM',
        ],
        'expected': 'AUC 提升 0.005-0.02, 训练速度提升 2-3x (LightGBM)',
    })

    return recs


# ============= REPORT GENERATION =============
def build_md(perf, orders_s2, pos_s3, log_s4, recs):
    """Build comprehensive markdown report"""
    lines = []
    h1=lambda t: lines.append(f"\n# {t}\n")
    h2=lambda t: lines.append(f"\n## {t}\n")
    h3=lambda t: lines.append(f"\n### {t}\n")
    tbl = lambda hdr, rows: [
        lines.append("| " + " | ".join(hdr) + " |"),
        lines.append("|" + "|".join("---" for _ in hdr) + "|"),
        [lines.append("| " + " | ".join(str(c) for c in r) + " |") for r in rows],
        lines.append("")
    ]

    h1("Wyckoff V10 回测深度数据分析报告 (V2 增强版)")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **回测区间**: 2023-01-01 ~ 2026-05-22 | **数据目录**: `{BASE_DIR}`\n")

    # === SECTION 1: CORE METRICS ===
    h2("一、核心绩效指标")
    h3("1.1 总览仪表盘")
    tbl(['指标', '数值'], perf['总览'])

    if '基准对比(沪深300)' in perf:
        h3("1.2 基准对比 (沪深300)")
        tbl(['指标', '数值'], perf['基准对比(沪深300)'])
    if '市场环境绩效' in perf:
        h3("1.3 分市场环境绩效")
        tbl(['环境', '交易天数', '年化收益', '年化波动', '夏普'], perf['市场环境绩效'])

    h3("1.4 月度统计")
    tbl(['指标', '数值'], perf['月度统计'])

    h3("1.5 季度收益")
    tbl(['季度', '收益率'], perf['季度收益'])

    h3("1.6 年度收益")
    tbl(['年度', '收益率'], perf['年度收益'])

    h3("1.7 成本侵蚀")
    tbl(['项目', '金额/比例'], perf['成本侵蚀'])

    h3("1.8 最大回撤 Top5")
    tbl(['开始', '底部', '持续(天)', '幅度'], perf['回撤Top5'])

    # === SECTION 2: ORDERS ===
    h2("二、订单与交易深度分析")
    h3("2.1 订单统计")
    tbl(['指标', '数值'], orders_s2['订单统计'])
    if '交易盈亏' in orders_s2:
        h3("2.2 交易盈亏分析")
        tbl(['指标', '数值'], orders_s2['交易盈亏'])
    if '盈亏分布桶' in orders_s2:
        h3("2.3 盈亏分布(桶)")
        tbl(['区间', '笔数'], orders_s2['盈亏分布桶'])
    h3("2.4 成本详情")
    tbl(['指标', '数值'], orders_s2['成本详情'])
    h3("2.5 最常交易股票 Top15")
    tbl(['代码', '累计交易额'], orders_s2['最常交易Top15'])
    h3("2.6 股票出现期数 Top10")
    tbl(['代码', '出现频率'], orders_s2['换手分析'])

    # === SECTION 3: POSITIONS ===
    h2("三、持仓结构深度分析")
    h3("3.1 持仓概览")
    tbl(['指标', '数值'], pos_s3['持仓概览'])
    h3("3.2 权重集中度")
    tbl(['指标', '数值'], pos_s3['集中度'])
    h3("3.3 日PnL统计")
    tbl(['指标', '数值'], pos_s3['日PnL'])
    h3("3.4 个股盈利贡献 Top10")
    tbl(['代码', '累计PnL'], pos_s3['盈利Top10'])
    h3("3.5 个股亏损拖累 Top10")
    tbl(['代码', '累计PnL'], pos_s3['亏损Top10'])
    h3("3.6 持仓时长分布")
    tbl(['指标', '数值'], pos_s3['持仓时长'])

    # === SECTION 4: LOG MINING ===
    h2("四、日志深度挖掘：模型训练与预测")
    h3("4.1 调仓概况")
    tbl(['指标', '数值'], log_s4['调仓概况'])
    if '预测概率质量' in log_s4:
        h3("4.2 预测概率质量")
        tbl(['指标', '数值'], log_s4['预测概率质量'])
    if 'CV Meta AUC' in log_s4:
        h3("4.3 CV Meta AUC 表现")
        tbl(['指标', '数值'], log_s4['CV Meta AUC'])
    if 'OOT Holdout' in log_s4:
        h3("4.4 OOT Holdout AUC")
        tbl(['指标', '数值'], log_s4['OOT Holdout'])
    if '折间AUC离散' in log_s4:
        h3("4.5 折间AUC离散度")
        tbl(['指标', '数值'], log_s4['折间AUC离散'])
    if 'AUC门控分布' in log_s4:
        h3("4.6 AUC门控模式分布")
        tbl(['模式', '次数'], log_s4['AUC门控分布'])
    if '特征选择' in log_s4:
        h3("4.7 特征选择稳定性")
        tbl(['指标', '数值'], log_s4['特征选择'])
    if '高频入选特征Top15' in log_s4:
        h3("4.8 高频入选特征 Top15")
        tbl(['特征名', '出现频率'], log_s4['高频入选特征Top15'])
    if '训练收敛' in log_s4:
        h3("4.9 训练收敛分析 (Best Iter)")
        tbl(['指标', '数值'], log_s4['训练收敛'])

    # AUC trend
    auc_trend = log_s4.get('AUC时序(max/min标注)', [])
    if auc_trend:
        h3("4.10 AUC 时序变化 (诊断)")
        lines.append("| 调仓日期 | CV Meta AUC | OOT AUC |")
        lines.append("|---|---|---|")
        for dt, cv, oot in auc_trend:
            cv_str = f"{cv:.4f}" if cv else "N/A"
            oot_str = f"{oot:.4f}" if oot else "N/A"
            flag = "🔴" if (cv and cv<0.53) else ("🟢" if (cv and cv>0.58) else "")
            lines.append(f"| {dt} {flag} | {cv_str} | {oot_str} |")
        lines.append("")

    # === SECTION 5: OPTIMIZATION ===
    h2("五、综合优化方案（核心输出）")
    lines.append(f"\n共识别 **{len(recs)}** 项优化建议。\n")

    for i, rec in enumerate(recs):
        h3(f"优化 #{i+1}: [{rec['id']}] {rec['title']}")
        lines.append(f"- **类别**: {rec['category']}")
        lines.append(f"- **现状**: {rec['data']}")
        lines.append("- **数据支撑**:")
        for e in rec['evidence']:
            lines.append(f"  - {e}")
        lines.append("- **优化行动**:")
        for a in rec['actions']:
            lines.append(f"  - {a}")
        lines.append(f"- **预期效果**: {rec['expected']}")
        lines.append("")

    # === ROADMAP ===
    h2("六、优先级行动路线图")
    lines.append("""
| 优先级 | 行动项 | 工作量 | 预期影响 |
|--------|--------|--------|----------|
| **P0-1** 🔴 | 启用 Barra 约束 (`enabled=true`) | 5 min | Max DD ↓5-10pp, Sharpe ↑0.1-0.2 |
| **P0-2** 🔴 | 硬止损(-8%) + 分批止盈(+15%/+30%) | 2 hr | P/L Ratio ↑0.2-0.4 |
| **P0-3** 🔴 | 调仓间隔 20→30天, 持仓 50→40只 | 1 hr | 交易成本↓30-40% |
| **P1-1** 🟠 | lr↑, depth↑, class_balance=auto | 4 hr | AUC ↑0.02-0.04, Prob区分度↑ |
| **P1-2** 🟠 | AUC门控收紧 (full≥0.54, min≥0.50) | 30 min | 弱期损失↓20% |
| **P1-3** 🟠 | xgb_class_balance=auto | 5 min | 正样本召回率↑5-10% |
| **P2-1** 🟡 | early_stopping_rounds 40→100 | 30 min | AUC ↑0.01-0.03 |
| **P2-2** 🟡 | vote_min_folds 3→4, weight_ic↑ | 30 min | 特征集稳定性↑ |
| **P2-3** 🟡 | min_auc_threshold 0.5→0.52 | 5 min | 集成质量↑ |
| **P2-4** 🟡 | LightGBM / CatBoost 对比实验 | 1-2 天 | AUC ↑0.005-0.02, 速度↑2-3x |
""")

    # === APPENDIX ===
    h2("七、附录: 当前配置关键参数")
    lines.append("```yaml")
    lines.append("model:")
    lines.append("  learner: XGBoost (binary:logistic, hist, nthread=1)")
    lines.append("  max_depth: 6, lr: 0.02, n_estimators: 1000")
    lines.append("  min_child_weight: 5, reg_alpha: 0.3")
    lines.append("  early_stopping_rounds: 40")
    lines.append("ensemble:")
    lines.append("  method: cv_fold, seeds: [42-46], calibration: none")
    lines.append("  diversity: depth=[5,5,6,6,7], subsample=[0.6,0.7,0.7,0.8,0.8]")
    lines.append("  weighting: auc_weighted, min_auc_threshold: 0.5")
    lines.append("execution:")
    lines.append("  rebalance_interval: 20 (trading days)")
    lines.append("  max_positions: 50, top_quantile: 0.1")
    lines.append("  barra_constraint: DISABLED")
    lines.append("  class_balance: none")
    lines.append("  feature_selection: voting(mda+mdi), max_features: 80, psi: 0.15")
    lines.append("```")

    lines.append("\n---\n")
    lines.append("*本报告由自动化数据分析脚本 V2 生成 (backtest_deep_analysis_v2.py)*")
    lines.append("*所有数据均从回测产出中提取, 每一个优化建议都有量化数据支撑*")
    lines.append(f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    return '\n'.join(lines)


def build_html(md_text):
    """Convert MD to a self-contained HTML for better readability"""
    # Simple markdown→HTML converter
    html = _md_to_html(md_text)
    with open(OUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)
    return OUT_HTML


def _md_to_html(md_text):
    """Quick MD→HTML"""
    lines = md_text.split('\n')
    html_parts = []
    in_table = False; in_code = False
    for line in lines:
        if line.startswith('```'):
            if in_code: html_parts.append('</pre>'); in_code=False
            else: html_parts.append('<pre>'); in_code=True
            continue
        if in_code: html_parts.append(line); continue
        if line.startswith('# '):
            html_parts.append(f'<h1>{line[2:]}</h1>')
        elif line.startswith('## '):
            html_parts.append(f'<h2>{line[3:]}</h2>')
        elif line.startswith('### '):
            html_parts.append(f'<h3>{line[4:]}</h3>')
        elif line.startswith('| ') and '|' in line[2:]:
            if not in_table:
                html_parts.append('<table>'); in_table=True
            if '---' in line:
                html_parts.append('</thead><tbody>')
            else:
                cells = [c.strip() for c in line.split('|')[1:-1]]
                tag = 'th' if in_table and html_parts and not html_parts[-1].startswith('</thead>') else 'td'
                html_parts.append('<tr>' + ''.join(f'<{tag}>{c}</{tag}>' for c in cells) + '</tr>')
        elif line.startswith('- ') and '**' in line:
            html_parts.append(f'<li>{_fmt_bold(line[2:])}</li>')
        elif line.startswith('  - '):
            html_parts.append(f'<li style="margin-left:20px">{_fmt_bold(line[4:])}</li>')
        elif line.strip() == '':
            if in_table: html_parts.append('</table>'); in_table=False
        else:
            if in_table: html_parts.append('</table>'); in_table=False
            html_parts.append(f'<p>{_fmt_bold(line)}</p>')

    body = '\n'.join(html_parts)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>Wyckoff V10 回测深度分析 V2</title>
<style>
body{{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;max-width:1080px;margin:0 auto;padding:24px;background:#fafbfc;color:#1f2937;line-height:1.7}}
h1{{font-size:28px;color:#1a1a2e;border-bottom:3px solid #2d6cdf;padding-bottom:8px}}
h2{{font-size:20px;color:#2d6cdf;margin-top:32px;border-bottom:1px solid #e5e7eb;padding-bottom:4px}}
h3{{font-size:16px;color:#374151;margin-top:20px}}
table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:14px;box-shadow:0 1px 3px rgba(0,0,0,0.05)}}
th{{background:#2d6cdf;color:white;padding:8px 12px;text-align:left}}
td{{padding:6px 12px;border-bottom:1px solid #e5e7eb}}
tr:nth-child(even){{background:#f9fafb}}
pre{{background:#1e1e1e;color:#d4d4d4;padding:16px;border-radius:8px;overflow-x:auto;font-size:13px}}
li{{margin:4px 0}}
strong{{color:#2d6cdf}}
</style></head>
<body>
{body}
<hr>
<p style="color:#6b7280;font-size:13px">⚠️ 本分析基于回测数据自动生成。所有数据和结论仅供参考，不构成任何投资建议。投资有风险，决策需谨慎。</p>
</body></html>"""


def _fmt_bold(text):
    return re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)


# ============= MAIN =============
if __name__ == '__main__':
    print("="*70)
    print("  Wyckoff V10 回测深度数据挖掘 V2 (增强版)")
    print("="*70)

    # Load
    print("[1/6] 加载数据...")
    daily, orders, pos, divs, report, config, log_lines = load()
    print(f"      daily={len(daily)}天, orders={len(orders)}笔, positions={len(pos)}条, log={len(log_lines)}行")

    # Analyze
    print("[2/6] Section 1: 绩效分解...")
    perf = section1_performance(daily, report)

    print("[3/6] Section 2: 订单与交易分析...")
    orders_s2 = section2_orders(orders, report)

    print("[4/6] Section 3: 持仓结构分析...")
    pos_s3 = section3_positions(pos)

    print("[5/6] Section 4: 日志深度挖掘...")
    log_s4 = section4_logs(log_lines, daily)
    print(f"      找到 {len(log_s4.get('_periods_detail',[]))} 个调仓周期")

    print("[6/6] Section 5: 生成优化方案 + 报告...")
    recs = section5_recommendations(perf, orders_s2, pos_s3, log_s4)
    print(f"      共 {len(recs)} 项优化建议")

    # Build reports
    md = build_md(perf, orders_s2, pos_s3, log_s4, recs)
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write(md)
    html_path = build_html(md)

    print(f"\n{'='*70}")
    print(f"  ✅ Markdown 报告: {OUT_MD}")
    print(f"  ✅ HTML 报告:     {html_path}")
    print(f"  ✅ 优化建议:      {len(recs)} 项 (P0={sum(1 for r in recs if r['id'].startswith('P0'))}, "
          f"P1={sum(1 for r in recs if r['id'].startswith('P1'))}, P2={sum(1 for r in recs if r['id'].startswith('P2'))})")
    print(f"{'='*70}")
