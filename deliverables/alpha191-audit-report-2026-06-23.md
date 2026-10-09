# Alpha191 策略完整审计报告（修订版）

**审计日期**: 2026-06-23  
**审计版本**: mda_adjusted（t检验 + XGBoost 优化版本）  
**审计范围**: config.yaml (155行), strategy.py (2667行), ensemble.py (1546行), labels.py (568行), barriers.py (61行), data_utils.py (394行)  
**审计方法**: 逐文件全量读取本地最新代码，逐行审查  
**修订说明**: 本版纠正了旧版审计报告中"AUC门控禁用"的错误判断（实际为启用状态），并基于全量代码重新评估所有审计发现

---

## 目录

1. [架构总览](#1-架构总览)
2. [配置文件审计 (config.yaml)](#2-配置文件审计-configyaml)
3. [核心模块审计](#3-核心模块审计)
4. [问题清单（按严重程度排序）](#4-问题清单按严重程度排序)
5. [未来数据泄露检查](#5-未来数据泄露检查)
6. [业界最佳实践对齐检查](#6-业界最佳实践对齐检查)
7. [修复建议](#7-修复建议)
8. [附录：完整文件清单与读取确认](#8-附录完整文件清单与读取确认)

---

## 1. 架构总览

Alpha191 是一个 XGBoost 元标注选股策略，集成 Wyckoff20 V10 基本面主模型 + Alpha191 因子 + 因子中性化 + Triple Barrier 元标签。

**数据流**:
```
V10 Wyckoff 主模型 → 候选池(300只)
    ↓
T-1 因子快照 → Winsorize+Z-score+OLS中性化 → alpha_*_neu 特征
    ↓
V10 pred_rank + 5个上下文特征(ctx_*) → 完整特征集
    ↓
Purged K-Fold CV(5折, embargo=21) → IC+PSI+MDA多折投票 → Top-40特征
    ↓
生产模型(5种子XGBoost集成, early stopping) → Isotonic概率校准
    ↓
AUC门控(3档) → 概率加权仓位 → Triple Barrier PT/SL
```

**关键参数** (config.yaml):
- `pt_mult: 100`, `sl_mult: 100`, `max_hold_days: 20`
- `auc_gate_enabled: true`, `auc_gate_min: 0.50`, `auc_gate_full: 0.55`
- `cv_folds: 5`, `cv_embargo_days: 21`, `cv_feature_consensus: 1`
- `mda.n_repeats: 20`, `mda.use_std_threshold: true`, `mda.t_alpha: 0.05`
- `vote_min_folds: 2`, `top_k_features: 40`
- `xgb_n_estimators: 300`, `xgb_max_depth: 4`, `xgb_learning_rate: 0.02`
- `ensemble_seeds: [42, 43, 44, 45, 46]`
- `label_mode: relative_barrier`, `calibration_method: isotonic`
- `vol_method: ewma`, `vol_ewma_lambda: 0.94`
- `neutralization.enabled: true`, `neutralization.use_market_cap: true`

---

## 2. 配置文件审计 (config.yaml)

### 2.1 ✅ AUC 门控配置（已纠正）

**旧版审计报告错误**: 称"AUC门控禁用"。

**实际代码确认** (config.yaml L109-111):
```yaml
auc_gate_enabled: true    # ← 已启用，非禁用
auc_gate_min: 0.50        # AUC < 0.50 → 空仓
auc_gate_full: 0.55       # AUC ≥ 0.55 → 全仓
                           # 0.50 ≤ AUC < 0.55 → 半仓
```

**代码实现确认** (strategy.py L2296-2328):
```python
auc_gate_enabled = meta_cfg.get('auc_gate_enabled', False)
if auc_gate_enabled:
    meta_auc_val = getattr(self.ensemble_manager, 'meta_auc', None)
    if meta_auc_val is None:
        meta_auc_val = getattr(self, 'meta_auc', 0.50)
    auc_min = meta_cfg.get('auc_gate_min', 0.50)
    auc_full = meta_cfg.get('auc_gate_full', 0.55)
    if meta_auc_val < auc_min:
        gate_mult = 0.0  # 空仓
    elif meta_auc_val >= auc_full:
        gate_mult = 1.0  # 全仓
    else:
        gate_mult = 0.5  # 半仓
else:
    gate_mult = 1.0
```

**AUC 来源** (ensemble.py L1374-1378):
```python
# Fix#7: AUC 门控使用生产模型验证集 AUC
self.meta_auc = prod_val_auc
self.xgb_val_auc = prod_val_auc
```

**结论**: AUC 门控已正确启用，三档仓位控制逻辑正确实现，AUC 来自生产模型验证集 OOS 估计。旧版审计报告判断错误，本次予以纠正。

### 2.2 其他配置项审计

| 配置项 | 值 | 审计结论 |
|--------|------|----------|
| `pt_mult: 100` | 100 | ⚠️ 极大值，PT/SL 实际不触发（详见 P2-1） |
| `sl_mult: 100` | 100 | ⚠️ 同上 |
| `max_hold_days: 20` | 20 | ✅ 合理 |
| `rebalance_interval: 20` | 20 | ✅ 与 max_hold_days 对齐 |
| `train_window_days: 240` | 240 | ✅ 约1年训练窗口 |
| `cv_folds: 5` | 5 | ✅ 标准 |
| `cv_embargo_days: 21` | 21 | ✅ 大于 max_hold_days，防止标签泄露 |
| `cv_feature_consensus: 1` | 1 | ⚠️ 共识阈值=1意味着只要在1折中出现即入选，较为宽松 |
| `mda.n_repeats: 20` | 20 | ✅ 足够支持 t 检验 |
| `mda.t_alpha: 0.05` | 0.05 | ✅ 标准显著性水平 |
| `vote_min_folds: 2` | 2 | ✅ 5折中需2折通过 |
| `top_k_features: 40` | 40 | ✅ 合理 |
| `ensemble_seeds` | [42-46] | ✅ 5种子集成 |
| `vol_method: ewma` | ewma | ⚠️ 配置为 EWMA，但标签端未实现（详见 P0-1） |
| `vol_ewma_seed_days: 60` | 60 | ⚠️ 配置存在但代码未使用（详见 P2-2） |

---

## 3. 核心模块审计

### 3.1 ensemble.py — 集成学习模块

#### ✅ Purged K-Fold CV (L951-1161)

标准 Prado AFML §7.4 实现:
- K=5 折，embargo=21 天
- 每折验证块前后各 embargo 天被 purge
- 训练集包含验证块之后的数据（利用完整样本）
- 折内 IC+PSI 特征选择（严格时间序列拆分）
- Early stopping with validation set

**审计结论**: 实现正确，无泄露。

#### ✅ 特征选择 v2.0 (L559-800)

三层递进:
1. IC 共识预筛选（≥cv_feature_consensus 折）
2. MDA 投票准入（≥vote_min_folds 折通过 t 检验）
3. 综合评分 = w_ic*avg_ic_rank + w_mdi*avg_mdi_rank + w_mda*avg_mda_rank
4. Top-K 截断

t 检验实现正确 (L602-606):
```python
t_crit = float(sp_stats.t.ppf(1 - mda_t_alpha, mda_n_repeats - 1))
```

降级保护: 特征不足时自动放宽投票阈值 → 降级至纯 IC 排序。

**审计结论**: 实现正确，有合理的降级保护。

#### ✅ 生产模型训练 (L1254-1422)

- 时序末尾 train/val 分割（val_size = min(max_hold, n_dates//5)）
- 多种子 XGBoost 集成（5种子，early stopping）
- 验证集 OOS 概率用于 Isotonic 校准
- 过拟合检测（train AUC - val AUC > 0.1 警告）
- CV vs 生产模型 val 一致性检查

**审计结论**: 实现正确。

#### ✅ 概率校准 (L387-434)

- IsotonicRegression(y_min=0.05, y_max=0.95, out_of_bounds='clip')
- 在验证集 OOS 概率上拟合（非 in-sample）
- 退化检测（校准概率范围 < 0.01 警告）
- 预测时裁剪至 [0.05, 0.95]

**审计结论**: 实现正确，Prado AFML Ch.10 Step 2 对齐。

#### ✅ 样本权重 (L52-178)

基于平均唯一性（Prado AFML Ch.7），按 trade_date 分组优化，数学等价性已证明。带跨周期缓存。

**审计结论**: 实现正确。

### 3.2 labels.py — Triple Barrier 标签生成

#### ✅ 批量优化标签生成 (L381-567)

- 日线预筛 + 分钟线确认两阶段
- 入场价对齐：`event_date_close`（默认）或 `next_day_open`
- Back-adjustment 一致性处理
- FIX-3: NaN 入场价跳过

#### ❌ 波动率计算 (L242-265) — 见 P0-1

```python
# _batch_daily_volatility
window_c = adj_close[mask][-self.daily_vol_window:]
rets = np.diff(window_c) / window_c[:-1]
v = np.std(rets, ddof=1)  # ← simple rolling std, 不尊重 config 的 vol_method: ewma
```

#### ✅ 标签端 Barrier 计算 (L489-494)

```python
pt_ths, sl_ths = compute_barriers(
    entry_prices, vols,
    pt_mult=self.pt_mult, sl_mult=self.sl_mult,
    vol_mult=self.vol_multiplier, hold_days=1,  # ← hold_days=1
    vectorized=True
)
```

与 barriers.py 公式一致，与建仓端 hold_days=1 一致。

### 3.3 barriers.py — 统一屏障计算

```python
hold_vol = daily_vol * np.sqrt(hold_days)
pt = entry_price * (1 + pt_mult * hold_vol * vol_mult)
sl = entry_price * (1 - sl_mult * hold_vol * vol_mult)
```

**审计结论**: 公式正确，支持标量和向量化。设计原则符合 Prado Ch.3.5（屏障 = f(σ)，非 f(signal)）。

### 3.4 strategy.py — 策略主类

#### ✅ 因子中性化 (L594-645)

Winsorize(1/99) → Z-score → OLS 残差回归（行业哑变量 + 对数市值）。
- fit/transform 分离：训练期 `neutralize_fit` 捕获系数，预测期 `neutralize_apply` 复用
- 训练期强制重新拟合（`self._train_ols_betas = None`）
- 截面按日分组 Winsorize+Z-score（`groupby('trade_date')`）

**审计结论**: 实现正确，无泄露。

#### ✅ V10 pred_rank (L1827-1867)

- 从 V7 预生成快照读取 pred 值
- 截面 rank percentile（替代 z-score）
- PSI 豁免但不跳过 IC 筛选

**审计结论**: 实现正确。

#### ❌ 上下文特征 (L1062-1178) — 见 P0-3

训练时多日数据 rolling(20) 正常工作，预测时单日数据 rolling 退化为 NaN→0。

#### ❌ 持仓 PT/SL 检查 (L2240-2246) — 见 P0-2

```python
hold_days = self.meta_labeler.max_hold_days  # = 20
hold_vol = daily_vol * np.sqrt(hold_days)
pt_threshold = entry_price * (1 + pt_mult * hold_vol * vol_mult)
sl_threshold = entry_price * (1 - sl_mult * hold_vol * vol_mult)
```

与标签端 hold_days=1 不一致，且不使用存储的 PT/SL 值。

#### ❌ 新开仓波动率 (L2493-2500) — 见 P0-1

```python
dv = adj.pct_change().dropna().std()  # ← simple std, 不尊重 config 的 vol_method: ewma
```

#### ✅ AUC 门控 (L2296-2328)

三档仓位控制正确实现，详见 §2.1。

#### ✅ 概率加权仓位 (L2441-2460)

Softmax(T=0.3) + 等权混合(α=0.7) + MAX_POSITION_PCT 裁剪。

#### ✅ 到期退出 (L1983-1999)

```python
if held_trade_days >= self.meta_labeler.max_hold_days:
    self._expiry_sells_today.append(code)
```

以开盘价退出，与标签端 close 入场的差异为日内波动。

#### ✅ 分红处理 (L2534-2583)

除权事件正确追踪，PT/SL 基于前复权价不受影响，price_raw 保留原始买入价。

#### ✅ 分钟线 PT/SL 检查 (L2587-2667)

向量化实现，先检查 PT 后检查 SL（与 compute_labels 一致），支持日线回退。

### 3.5 data_utils.py — 数据工具

- 交易日历: Baostock + 缓存 ✅
- 指数成分股: 掘金 + 缓存 ✅
- 基本面快照: Parquet 加载 ✅
- 行业映射: 从快照提取 ✅

**审计结论**: 无问题。

---

## 4. 问题清单（按严重程度排序）

### P0-1: 波动率计算方法不一致（严重）

**位置**:
| 代码位置 | 计算方法 | 尊重 config? |
|----------|----------|:---:|
| labels.py L242-265 `_batch_daily_volatility` | simple rolling std | ❌ |
| strategy.py L2226-2232 持仓检查 | EWMA (λ=0.94) | ✅ |
| strategy.py L2368-2383 续期建仓 | EWMA (λ=0.94) | ✅ |
| strategy.py L2493-2500 新开仓 | simple rolling std | ❌ |

**config**: `vol_method: ewma`, `vol_ewma_lambda: 0.94`

**影响**:
- 标签端用 rolling std，交易端（持仓检查+续期）用 EWMA → PT/SL 阈值计算基础不同
- 新开仓用 rolling std，续期建仓用 EWMA → 同一股票在不同调仓周期的 PT/SL 阈值不一致
- EWMA 对近期波动更敏感，rolling std 对窗口内波动一视同仁，两者数值可能有 10-30% 差异

**严重程度**: P0 — 训练-交易不一致，影响标签正确性和 PT/SL 触发准确性

**修复建议**:
1. labels.py `_batch_daily_volatility` 增加 EWMA 分支
2. strategy.py L2493-2500 新开仓波动率增加 EWMA 分支
3. 统一四处波动率计算的回退默认值

---

### P0-2: 持仓 PT/SL hold_days 与标签端不一致（严重）

**位置**:
| 代码位置 | hold_days | 公式 |
|----------|:---:|------|
| labels.py L489-494 (标签端) | 1 | `pt = entry * (1 + pt_mult * daily_vol * sqrt(1))` |
| strategy.py L2502-2504 (新开仓) | 1 | 同上 ✅ |
| strategy.py L2388-2390 (续期建仓) | 1 | 同上 ✅ |
| strategy.py L2240-2246 (持仓检查) | 20 | `pt = entry * (1 + pt_mult * daily_vol * sqrt(20))` |

**影响**:
- 持仓检查的 PT/SL 阈值比标签端宽 `sqrt(20) ≈ 4.47` 倍
- `self.holdings[code]['pt']` 和 `self.holdings[code]['sl']` 存储的值（hold_days=1）仅用于日志，实际检查使用不同的阈值（hold_days=20）
- 日志中显示的 PT/SL 与实际检查的 PT/SL 不一致，误导调试

**当前参数下的实际影响**:
- `pt_mult=100, sl_mult=100` 使得 PT/SL 阈值极大（详见 P2-1），实际不触发
- 但如果未来调小 pt_mult/sl_mult，此不一致将导致严重的训练-交易偏差

**严重程度**: P0 — 结构性 bug，当前因参数极端而影响有限

**修复建议**:
- 方案 A（推荐）: 持仓检查 L2240-2246 改用 `hold_days=1`，与标签端和建仓端一致
- 方案 B: 持仓检查直接使用 `self.holdings[code]['pt']` 和 `self.holdings[code]['sl']` 存储的值

---

### P0-3: 上下文特征训练-预测分布不一致（严重）

**位置**: strategy.py L1062-1178 (`_compute_context_features`), L1244-1249 (预测调用), L1872 (训练调用)

**训练时** (L1872):
```python
train_data = self._compute_context_features(train_data, 'ts_code', 'trade_date')
# train_data 包含 240 交易日 × 多只股票
# rolling(20, min_periods=5) 正常工作
# ctx_market_vol = 每日截面中位数波动率（有意义）
# ctx_relative_vol, ctx_turnover_ratio, ctx_max_dd_20d, ctx_rank_x_vol 均有实际值
```

**预测时** (L1247-1249):
```python
features_df['trade_date'] = t_minus_1  # 单日
features_df = self._compute_context_features(features_df, 'ts_code', 'trade_date')
# features_df 只包含 T-1 单日数据
# rolling(20, min_periods=5) 在单日数据上 → NaN
# groupby(date_col)['_stk_vol'].transform('median') → NaN（只有1天的截面中位数）
# 所有 5 个 ctx 特征 → NaN → nan_to_num → 0.0
```

**影响**:
- 训练时 ctx 特征有实际分布（经 Winsorize+Z-score 标准化后的值）
- 预测时 ctx 特征全部为 0
- 模型在训练时学到的 ctx 特征权重在预测时指向恒定值 0，相当于丢失 5 个特征的信息
- 如果 ctx 特征在训练时有显著重要性，预测精度会下降

**严重程度**: P0 — 特征分布严重不一致，影响模型预测有效性

**修复建议**:
- 方案 A（推荐）: 预测时传入多日历史数据（至少 20 个交易日），使 rolling 正常工作
- 方案 B: 预测时单独计算 ctx 特征（不依赖 rolling），如用预缓存的市场波动率
- 方案 C: 如果 ctx 特征实际贡献不大，考虑移除以简化管线

---

### P1-1: trade_date 重标签导致上下文特征截面计算偏差

**位置**: strategy.py L1739

```python
snap['trade_date'] = sample_date  # T-1 快照重标签为 T
```

**分析**:
- 特征数据来自 T-1 close（正确，无未来信息泄露）
- 但 trade_date 被标记为 T
- 训练时 `groupby('trade_date')` 按 T 分组计算截面中位数
- 预测时 `groupby('trade_date')` 按 t_minus_1（即 T-1）分组
- 日期标签不一致，但由于预测时只有单日数据，此问题被 P0-3 掩盖

**严重程度**: P1 — 当前被 P0-3 掩盖，修复 P0-3 后需同步修复

**修复建议**: 统一训练和预测的 trade_date 语义

---

### P1-2: 新开仓波动率不尊重 EWMA 配置

**位置**: strategy.py L2493-2500

```python
# 新开仓
if 'raw_close' in h.columns and 'backAdjustFactor' in h.columns:
    adj = h['raw_close'] * h['backAdjustFactor']
    dv = adj.pct_change().dropna().std()  # ← simple std
else:
    dv = h['close'].pct_change().dropna().std()
```

对比续期建仓 L2368-2383:
```python
# 续期建仓
vol_method = self.config.get('meta_labeling', {}).get('vol_method', 'rolling')
if vol_method == 'ewma':
    lambd = self.config.get('meta_labeling', {}).get('vol_ewma_lambda', 0.94)
    ewma_var_series = daily_ret.ewm(alpha=1-lambd, adjust=False).var()
    ewma_var = float(ewma_var_series.iloc[-1])
    dv = float(np.sqrt(ewma_var)) if (pd.notna(ewma_var) and ewma_var > 0) else 0.025
else:
    dv = daily_ret.std()
```

**影响**: 新开仓和续期建仓对同一股票可能计算不同的 daily_vol，导致 PT/SL 阈值不一致。

**严重程度**: P1 — 与 P0-1 相关，应一并修复

---

### P1-3: 波动率回退默认值不一致

**位置**:
| 代码位置 | 回退默认值 |
|----------|:---:|
| labels.py L258, L263 | 0.01 |
| strategy.py L2234 (持仓检查) | 0.02 |
| strategy.py L2385 (续期建仓) | 0.025 |
| strategy.py L2500 (新开仓) | 0.025 |

**影响**: 数据不足时，不同代码路径使用不同的回退波动率，导致 PT/SL 阈值差异。

**严重程度**: P1 — 极端情况下的不一致

**修复建议**: 统一为 0.02（或从 config 读取）

---

### P2-1: pt_mult=100/sl_mult=100 效果等同于禁用 PT/SL

**分析**:
- `hold_days=1, daily_vol≈0.02` 时:
  - PT = entry × (1 + 100 × 0.02) = entry × 3.0（200% 涨幅才触发）
  - SL = entry × (1 - 100 × 0.02) = entry × (-1.0)（负价，永不触发）
- 标签几乎全部为 "expire"（持有到期）
- 策略实质上是纯方向性预测：label = 1 if ret > 0 else 0

**影响**: 
- 这可能是 `relative_barrier` 模式的设计意图（用极端参数禁用 PT/SL，只看到期方向）
- 但 P0-2 中的 hold_days=20 使得持仓检查的阈值更极端（PT = entry × 9.94），进一步确保不触发
- 实际效果：策略不依赖 PT/SL 退出，仅依赖 max_hold_days=20 到期退出

**严重程度**: P2 — 可能是设计意图，但应在文档中明确说明

**建议**: 如果是有意为之，在 config 中添加注释说明；如果非有意，考虑调整为更合理的值（如 pt_mult=2.0, sl_mult=1.5）

---

### P2-2: vol_ewma_seed_days 配置未实现

**位置**: config.yaml L105 `vol_ewma_seed_days: 60`

**分析**: 代码中搜索 `vol_ewma_seed_days` 无任何引用。EWMA 在 lookback 窗口（`daily_vol_window * 2 = 40` 日历天 ≈ 28 交易日）内计算，可能数据量不足导致 EWMA 不稳定。

**严重程度**: P2 — 配置项存在但未实现

**修复建议**: 要么实现 vol_ewma_seed_days（扩展 lookback 到至少 60 交易日），要么从 config 中移除

---

### P2-3: cv_feature_consensus=1 共识阈值过于宽松

**位置**: config.yaml L39 `cv_feature_consensus: 1`

**分析**: 共识阈值=1 意味着特征只要在 5 折中的 1 折通过 IC+PSI 筛选即入选共识池。Prado AFML 建议至少 3 折共识。但后续有 MDA 投票（≥2 折）作为第二层筛选，部分缓解。

**严重程度**: P2 — 依赖 MDA 投票兜底，但仍可能引入不稳定特征

---

### P2-4: ProbabilityCalibrator 裁剪范围可能过紧

**位置**: ensemble.py L403, L423

```python
IsotonicRegression(y_min=0.05, y_max=0.95, out_of_bounds='clip')
# predict: np.clip(calibrated, 0.05, 0.95)
```

**分析**: 对于 AUC ≈ 0.55-0.58 的弱模型，概率范围 [0.05, 0.95] 可能过宽（模型无法支撑 0.95 的确定性）。但这也是一种保守设计，防止极端概率导致的仓位过度集中。

**严重程度**: P2 — 设计选择，非 bug

---

## 5. 未来数据泄露检查

### ✅ 确认无泄露

| 检查项 | 位置 | 结论 |
|--------|------|------|
| Purged K-Fold CV embargo | ensemble.py L1021-1025 | ✅ 验证块前后各 21 天 purge |
| 生产模型时序分割 | ensemble.py L1262-1271 | ✅ 末尾 val_size 天作为验证集 |
| 因子中性化 fit/transform | strategy.py L632-643 | ✅ 训练拟合 OLS，预测复用系数 |
| 样本权重 | ensemble.py L52-118 | ✅ 基于 trade_date 和 max_hold_days，无未来信息 |
| 特征选择 IC/PSI | ensemble.py L1046-1065 | ✅ 折内训练数据计算 |
| MDA | ensemble.py L1119-1131 | ✅ 验证集 OOS 计算 |
| Isotonic 校准 | ensemble.py L1382-1386 | ✅ 验证集 OOS 概率拟合 |
| V10 pred_rank | strategy.py L1828-1839 | ✅ 从 V7 预生成快照读取 |
| 因子快照 | strategy.py L1727, L1196 | ✅ 使用 T-1 数据 |
| 市值填充 | strategy.py L1815 | ✅ 按 trade_date 精确匹配 |

### ⚠️ 潜在风险

| 检查项 | 位置 | 风险描述 | 严重程度 |
|--------|------|----------|----------|
| trade_date 重标签 | strategy.py L1739 | T-1 快照标记为 T，特征本身无泄露，但可能影响下游时序计算 | 低 |
| 上下文特征退化 | strategy.py L1247-1249 | 预测时 ctx 特征全为 0，非泄露但导致预测偏差 | 高（P0-3） |

---

## 6. 业界最佳实践对齐检查

### ✅ 符合 Prado AFML

| 实践 | 来源 | 实现位置 | 状态 |
|------|------|----------|:---:|
| Purged K-Fold CV | AFML §7.4 | ensemble.py L951-1161 | ✅ |
| 样本权重(平均唯一性) | AFML Ch.7 | ensemble.py L52-118 | ✅ |
| Triple Barrier Labeling | AFML Ch.3 | labels.py L381-567 | ✅ |
| MDI + MDA 特征重要性 | AFML Ch.8 | ensemble.py L437-555, L1108-1131 | ✅ |
| 概率校准 | AFML Ch.10 Step 2 | ensemble.py L387-434 | ✅ |
| 屏障 = f(σ) | AFML Ch.3.5 | barriers.py | ✅ |
| Meta-labeling | AFML Ch.3 | strategy.py L1882-1888 | ✅ |

### ✅ 符合 ML 工程最佳实践

| 实践 | 实现位置 | 状态 |
|------|----------|:---:|
| Early stopping | ensemble.py L1094, L1346 | ✅ |
| Multi-seed ensemble | ensemble.py L1326-1349 | ✅ |
| 过拟合检测 | ensemble.py L1399-1401 | ✅ |
| CV 一致性检查 | ensemble.py L1406-1408 | ✅ |
| 特征稳定性(PSI) | ensemble.py L181-203 | ✅ |
| 降级保护 | ensemble.py L708-720 | ✅ |
| NaN 原生处理(XGBoost) | ensemble.py L376-384 | ✅ |
| Back-adjustment(分红复权) | labels.py L14-36, strategy.py L2218-2223 | ✅ |

### ❌ 不符合

| 实践 | 问题描述 | 严重程度 |
|------|----------|----------|
| 训练-交易一致性 | 波动率计算方法不一致(P0-1) | P0 |
| 训练-交易一致性 | PT/SL hold_days 不一致(P0-2) | P0 |
| 训练-预测分布一致 | 上下文特征退化(P0-3) | P0 |

---

## 7. 修复建议

### 优先级 P0（建议立即修复）

#### 修复 1: 统一波动率计算 (P0-1)

**labels.py** `_batch_daily_volatility` 增加 EWMA 分支:
```python
def _batch_daily_volatility(self, daily_df, event_dates, ref_date=None, vol_method='rolling', ewma_lambda=0.94):
    # ... existing code ...
    for i, ed in enumerate(event_dates):
        # ... existing window extraction ...
        rets = np.diff(window_c) / window_c[:-1]
        if vol_method == 'ewma' and len(rets) > 1:
            ewma_var = pd.Series(rets).ewm(alpha=1-ewma_lambda, adjust=False).var().iloc[-1]
            v = np.sqrt(ewma_var) if not (np.isnan(ewma_var) or ewma_var <= 0) else np.std(rets, ddof=1)
        else:
            v = np.std(rets, ddof=1)
        vols[i] = v if not (np.isnan(v) or v == 0) else 0.02  # 统一回退值
```

**strategy.py** L2493-2500 新开仓增加 EWMA 分支:
```python
vol_method = self.config.get('meta_labeling', {}).get('vol_method', 'rolling')
if vol_method == 'ewma' and len(daily_ret) > 1:
    lambd = self.config.get('meta_labeling', {}).get('vol_ewma_lambda', 0.94)
    ewma_var = daily_ret.ewm(alpha=1-lambd, adjust=False).var().iloc[-1]
    dv = float(np.sqrt(ewma_var)) if pd.notna(ewma_var) and ewma_var > 0 else 0.02
else:
    dv = daily_ret.std() if len(daily_ret) > 0 else 0.02
```

#### 修复 2: 统一 PT/SL hold_days (P0-2)

**strategy.py** L2239-2246 改为:
```python
# 与标签端 compute_barriers(hold_days=1) 对齐
hold_days = 1  # 原: self.meta_labeler.max_hold_days
hold_vol = daily_vol * np.sqrt(hold_days)
vol_mult = self.meta_labeler.vol_multiplier if self.meta_labeler else 1.0
pt_mult = self.meta_labeler.pt_mult
sl_mult = self.meta_labeler.sl_mult
pt_threshold = entry_price * (1 + pt_mult * hold_vol * vol_mult)
sl_threshold = entry_price * (1 - sl_mult * hold_vol * vol_mult)
```

或更优方案，直接使用存储的 PT/SL:
```python
if 'pt' in self.holdings.get(code, {}) and 'sl' in self.holdings.get(code, {}):
    pt_threshold = self.holdings[code]['pt']
    sl_threshold = self.holdings[code]['sl']
else:
    # fallback: 与标签端一致的公式
    hold_vol = daily_vol * np.sqrt(1)
    pt_threshold = entry_price * (1 + pt_mult * hold_vol * vol_mult)
    sl_threshold = entry_price * (1 - sl_mult * hold_vol * vol_mult)
```

#### 修复 3: 上下文特征预测时退化 (P0-3)

**strategy.py** `_meta_select_stocks` 中，预测前加载多日历史数据:
```python
# 预测时传入近 20 个交易日的数据，使 rolling 正常工作
lookback_dates = self._get_recent_trading_days(t_minus_1, n=20)
multi_day_features = []
for d in lookback_dates:
    snap_d = context.data_provider.get_factor_snapshot(d)
    if snap_d is not None and not snap_d.empty:
        snap_d = snap_d[snap_d['ts_code'].isin(recommended)].copy()
        snap_d['trade_date'] = d
        multi_day_features.append(snap_d)

if multi_day_features:
    multi_day_df = pd.concat(multi_day_features, ignore_index=True)
    multi_day_df = self._compute_context_features(multi_day_df, 'ts_code', 'trade_date')
    # 提取 T-1 的 ctx 特征值
    t_minus_1_ctx = multi_day_df[multi_day_df['trade_date'] == t_minus_1][ctx_cols]
    # 合并到预测特征
    for col in ctx_cols:
        features_df[col] = features_df['ts_code'].map(
            t_minus_1_ctx.set_index('ts_code')[col]
        ).fillna(0.0)
```

### 优先级 P1（建议近期修复）

#### 修复 4: 统一回退默认值 (P1-3)

将所有波动率回退默认值统一为 0.02:
- labels.py L258, L263: `0.01` → `0.02`
- strategy.py L2234: 已为 `0.02` ✅
- strategy.py L2385: `0.025` → `0.02`
- strategy.py L2500: `0.025` → `0.02`

### 优先级 P2（建议评估后处理）

- 评估 pt_mult/sl_mult=100 是否为有意设计，添加文档注释
- 实现 vol_ewma_seed_days 或从 config 移除
- 评估 cv_feature_consensus=1 是否需要提高至 2 或 3

---

## 8. 附录：完整文件清单与读取确认

| 文件 | 行数 | 读取状态 | 关键确认 |
|------|------|:---:|----------|
| config.yaml | 155 | ✅ 全量 | AUC 门控启用 (L109=true) |
| strategy.py | 2667 | ✅ 全量 | AUC 门控逻辑(L2296-2328), PT/SL(L2240-2246), 上下文特征(L1062-1178) |
| ensemble.py | 1546 | ✅ 全量 | Purged K-Fold(L951-1161), 生产模型(L1254-1422), 校准(L387-434) |
| labels.py | 568 | ✅ 全量 | 标签生成(L381-567), 波动率(L242-265), Barrier(L489-494) |
| barriers.py | 61 | ✅ 全量 | 统一公式(L32-60) |
| data_utils.py | 394 | ✅ 全量 | 交易日历, 指数成分股, 基本面快照 |

---

## 审计总结

### 纠正项
1. ✅ **AUC 门控状态**: 旧版报告称"禁用"，实际为**启用**（config L109 `auc_gate_enabled: true`）。门控逻辑正确实现三档仓位控制（空仓/半仓/全仓），AUC 来自生产模型验证集 OOS 估计。

### 发现的问题汇总

| 编号 | 严重程度 | 问题 | 位置 |
|------|:---:|------|------|
| P0-1 | 🔴 | 波动率计算方法不一致（标签 rolling std vs 交易 EWMA） | labels.py L242, strategy.py L2493 |
| P0-2 | 🔴 | 持仓 PT/SL hold_days=20 与标签 hold_days=1 不一致 | strategy.py L2240 |
| P0-3 | 🔴 | 上下文特征预测时退化为 0 | strategy.py L1247 |
| P1-1 | 🟡 | trade_date 重标签影响截面计算 | strategy.py L1739 |
| P1-2 | 🟡 | 新开仓波动率不尊重 EWMA 配置 | strategy.py L2493 |
| P1-3 | 🟡 | 波动率回退默认值不一致（0.01/0.02/0.025） | 4处 |
| P2-1 | 🟢 | pt_mult=100/sl_mult=100 效果等同禁用 PT/SL | config.yaml L97-98 |
| P2-2 | 🟢 | vol_ewma_seed_days 配置未实现 | config.yaml L105 |
| P2-3 | 🟢 | cv_feature_consensus=1 共识阈值宽松 | config.yaml L39 |
| P2-4 | 🟢 | 概率裁剪范围 [0.05, 0.95] 可能过紧 | ensemble.py L403 |

### 整体评价

Alpha191 策略在**特征选择（IC+PSI+MDA多折投票）、交叉验证（Purged K-Fold+embargo）、模型训练（多种子集成+early stopping）、概率校准（Isotonic OOS）、AUC门控**等核心环节实现了业界最佳实践，框架设计扎实。

主要风险集中在**训练-交易一致性**：波动率计算方法、PT/SL hold_days、上下文特征分布三处不一致（P0级），需要在实盘部署前修复。当前因 pt_mult/sl_mult=100 的极端参数使得 PT/SL 实际不触发，P0-1 和 P0-2 的实际影响有限，但这是结构性 bug，一旦参数调整将立即暴露。

---

*本审计报告基于 2026-06-23 本地最新代码全量读取生成。所有文件均已逐行审查，所有代码引用均可在对应文件中定位。*
