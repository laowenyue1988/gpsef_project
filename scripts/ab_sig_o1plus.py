# -*- coding: utf-8 -*-
"""O1+ (small_alpha191_test) vs 基准 (small_alpha191) 绩效显著性检验

方法 (沿用 ab_sig_test.py 方法学):
  1) 日收益配对 t 检验 + Welch t 检验
  2) 夏普比率差: Memmel(2003) 解析检验 + 配对 Bootstrap (10万次)
  3) 最大回撤差: 移动分块 Bootstrap (block=21日)
  4) 月度收益: 独立样本 t 检验
  5) 关键绩效指标对比表
"""
import numpy as np
import pandas as pd
from scipy import stats

A = 'results/SmallAlpha191_20220801_20260522_20260819123016'       # 基准 small_alpha191
B = 'results/SmallAlpha191Test_20220801_20260522_20260819143844'   # O1+ small_alpha191_test

rets, monthly = {}, {}
for k, d in {'A': A, 'B': B}.items():
    df = pd.read_csv(f'{d}/daily_values.csv', parse_dates=['date'])
    r = df['daily_return'].values.astype(float)
    r = r[~np.isnan(r)][1:]
    rets[k] = r
    s = pd.Series(df['daily_return'].values.astype(float), index=df['date']).dropna()[1:]
    m = (1 + s).resample('ME').prod() - 1
    monthly[k] = m.dropna().values

T = min(len(rets['A']), len(rets['B']))
ra, rb = rets['A'][:T], rets['B'][:T]
print(f'交易日: {T}  (基准末 {len(rets["A"])} / O1+ {len(rets["B"])})')
print()

ANN = 252
def sr(r, rf=0.0):
    return (r.mean() * ANN - rf) / (r.std(ddof=1) * np.sqrt(ANN))
def ann(r):
    return r.mean() * ANN
def mdd(r):
    nav = np.cumprod(1 + r)
    return float(np.min(nav / np.maximum.accumulate(nav) - 1))

print('=' * 72)
print('① 日收益差: 配对 t + Welch')
print('=' * 72)
d = rb - ra
t_pair = stats.ttest_rel(rb, ra)
t_welch = stats.ttest_ind(rb, ra, equal_var=False)
print(f'  均值差(日): {d.mean():+.6f}  (年化 {d.mean()*ANN:+.2%})')
print(f'  配对 t:    t={t_pair.statistic:.3f}  p={t_pair.pvalue:.4f}')
print(f'  Welch t:   t={t_welch.statistic:.3f}  p={t_welch.pvalue:.4f}')

print()
print('=' * 72)
print('② 夏普比率差: Memmel 解析 + 配对 Bootstrap')
print('=' * 72)
srA, srB = sr(ra), sr(rb)
print(f'  Sharpe: 基准 {srA:.3f}  O1+ {srB:.3f}  差 {srB-srA:+.3f}')
corr = np.corrcoef(ra, rb)[0, 1]
var_diff = (2 * (1 - corr)) / T  # 日口径 Sharpe 差的渐近方差 (Memmel 简化, 日尺度)
se = np.sqrt(var_diff)
z = (srB - srA) / se if se > 0 else 0
p_memmel = 2 * (1 - stats.norm.cdf(abs(z)))
print(f'  Memmel:  z={z:.3f}  p={p_memmel:.4f}  (corr={corr:.3f})')
rng = np.random.default_rng(42)
n_boot = 100000
boot_diff = np.empty(n_boot)
for i in range(n_boot):
    idx = rng.integers(0, T, T)
    boot_diff[i] = sr(rb[idx]) - sr(ra[idx])
p_boot = float((boot_diff <= 0).mean())
ci = np.percentile(boot_diff, [2.5, 97.5])
print(f'  Bootstrap(10万): P(Sharpe差<=0)={p_boot:.4f}  95%CI=[{ci[0]:+.3f}, {ci[1]:+.3f}]')

print()
print('=' * 72)
print('③ 最大回撤差: 分块 Bootstrap (block=21)')
print('=' * 72)
mddA, mddB = mdd(ra), mdd(rb)
print(f'  回撤: 基准 {mddA:.2%}  O1+ {mddB:.2%}  差 {mddB-mddA:+.2%}')
n_blk = int(np.ceil(T / 21))
blk_diff = np.empty(10000)
for i in range(10000):
    idx = np.concatenate([rng.integers(0, T - 21, n_blk) + j for j in range(21)])
    idx = idx[:T]
    blk_diff[i] = mdd(rb[idx]) - mdd(ra[idx])
p_mdd = float((blk_diff >= 0).mean())  # O1+ 回撤更小(差>=0) 的概率
print(f'  Bootstrap(1万): P(O1+回撤更小)={p_mdd:.4f}')

print()
print('=' * 72)
print('④ 月度收益: 独立样本 t')
print('=' * 72)
mA, mB = monthly['A'], monthly['B']
t_m = stats.ttest_ind(mB, mA, equal_var=False)
print(f'  月均: 基准 {mA.mean():+.3%}  O1+ {mB.mean():+.3%}  差 {mB.mean()-mA.mean():+.3%}')
print(f'  检验: t={t_m.statistic:.3f}  p={t_m.pvalue:.4f}  (月数 {len(mA)}/{len(mB)})')

print()
print('=' * 72)
print('⑤ 关键指标总表')
print('=' * 72)
rep = {}
for k, d in {'基准': A, 'O1+': B}.items():
    txt = open(f'{d}/report.txt', encoding='utf-8', errors='ignore').read()
    def g(key):
        for line in txt.splitlines():
            if line.startswith(key):
                return line.split(':', 1)[1].strip()
        return ''
    rep[k] = {
        '总收益': g('总收益率'), '年化': g('年化收益率'), '波动率': g('年化波动率'),
        'Sharpe': g('夏普比率'), '回撤': g('最大回撤'), '胜率': g('日胜率'),
        '盈亏比': g('盈亏比'), '订单': g('总订单数'),
    }
print(f'{"指标":<8}{"基准(small)":<22}{"O1+(test)":<22}')
for key in ['总收益', '年化', '波动率', 'Sharpe', '回撤', '胜率', '盈亏比', '订单']:
    print(f'{key:<8}{rep["基准"][key]:<22}{rep["O1+"][key]:<22}')
