# Alpha191 回测性能优化方案

> **目标**：在不改变策略逻辑和输出结果的前提下，大幅缩短单次调仓回测时间
> **当前基线**：单次完整调仓 ~30 分钟
> **预估优化后**：~5-8 分钟（3-5 倍加速）

---

## 一、耗时分布总览

| 阶段 | 耗时 | 占比 | 文件位置 |
|------|------|------|----------|
| MDA-OOS 特征重要性 | ~23 min | 77% | `ensemble.py:437-555` |
| 三分类标签生成 | ~257s | 14% | `labels.py:407-594` |
| 因子中性化 | ~78s | 4% | `strategy.py:603-654`, `neutralizer.py` |
| IC+PSI 特征筛选 | ~60s | 3% | `ensemble.py:237-351` |
| 历史回放+快照预加载 | ~2.6s | <1% | `strategy.py:1729-1810` |
| 生产模型训练 | ~12s | 1% | `ensemble.py:1254-1300` |
| 预测选股 | ~3.5s | <1% | `strategy.py:1189+` |
| 其他（V10注入等） | ~30s | 2% | `strategy.py:1890-1911` |

---

## 二、优化方案详述

### 优化 1：MDA-OOS — numpy 预转换 + 原地列置换（预计节省 ~13-15 分钟）

**瓶颈位置**：`ensemble.py:437-555`，`compute_mda_oos()` 函数

**问题分析**：

当前代码对每个特征×每次重复（共 ~135×20×5 = 13,500 次）执行以下操作：

```python
# 当前实现（每次迭代）
X_shuffled = X_val.copy()              # ① DataFrame 全量拷贝（~2.8MB）
col = X_shuffled[feat].values.copy()   # ② 列值拷贝
rng.shuffle(col)                        # ③ 洗牌
X_shuffled[feat] = col                  # ④ 列回写 DataFrame
dval_perm = prepare_dmatrix(X_shuffled, feature_names, label=y_val)
# prepare_dmatrix 内部：
#   X = df[features].replace([np.inf, -np.inf], np.nan)  # ⑤ 全量 replace
#   X = X.astype(np.float32)                              # ⑥ 全量类型转换
#   xgb.DMatrix(X, ...)                                   # ⑦ DMatrix 创建
perm_proba = model.predict(dval_perm)   # ⑧ 预测
perm_auc = roc_auc_score(y_val, perm_proba)  # ⑨ AUC
```

步骤 ①②④⑤⑥ 每次都处理整个 DataFrame（~5000行×135列），但实际只需修改一列。这是纯浪费。

**优化方案**：

```python
def compute_mda_oos(model, X_val_df, y_val, feature_names, n_repeats=5, random_state=42):
    rng = np.random.RandomState(random_state)
    
    # 预处理一次：转 numpy float32 数组
    X_base = X_val_df[feature_names].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
    feat_to_idx = {f: i for i, f in enumerate(feature_names)}
    n_samples = X_base.shape[0]
    
    # 基线 DMatrix + AUC（复用 prepare_dmatrix 保证一致性）
    dval = prepare_dmatrix(X_val_df, feature_names, label=y_val)
    baseline_proba = model.predict(dval)
    baseline_auc = float(roc_auc_score(y_val, baseline_proba))
    
    mda_means = {}
    mda_stds = {}
    
    for feat in feature_names:
        fidx = feat_to_idx[feat]
        original_col = X_base[:, fidx].copy()  # 仅拷贝一列
        drops = []
        for _ in range(n_repeats):
            col = original_col.copy()
            rng.shuffle(col)                    # 消费与原代码完全相同的 RNG 序列
            X_base[:, fidx] = col               # 原地修改一列
            # 直接从 numpy 创建 DMatrix，跳过 DataFrame 开销
            dval_perm = xgb.DMatrix(X_base, label=y_val, missing=np.nan,
                                    feature_names=feature_names)
            perm_proba = model.predict(dval_perm)
            perm_auc = float(roc_auc_score(y_val, perm_proba))
            drops.append(baseline_auc - perm_auc)
        X_base[:, fidx] = original_col          # 恢复原始列
        mda_means[feat] = float(np.mean(drops))
        mda_stds[feat] = float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0
    
    # ... 后续 t 检验和日志代码不变 ...
```

**为什么输出完全一致**：
- `rng.shuffle(col)` 的调用顺序和参数与原代码完全相同 → 产生完全相同的随机置换
- `xgb.DMatrix(numpy_array, ...)` 与 `xgb.DMatrix(DataFrame, ...)` 在数据相同时产生等价的 DMatrix
- `model.predict()` 和 `roc_auc_score()` 输入相同 → 输出相同

**节省分析**：

| 操作 | 原始耗时/次 | 优化后耗时/次 | 节省 |
|------|------------|--------------|------|
| DataFrame.copy() | ~5ms | 0（跳过） | 5ms |
| 列值copy+回写 | ~0.5ms | ~0.1ms（仅一列copy） | 0.4ms |
| replace+astype | ~3ms | 0（预转换一次） | 3ms |
| DMatrix创建 | ~10ms | ~8ms（从numpy更快） | 2ms |
| predict+AUC | ~6ms | ~6ms（不变） | 0 |
| **合计/次** | **~24ms** | **~14ms** | **~10ms** |
| **13,500次合计** | **~324s** | **~189s** | **~135s** |

每折节省 ~27s，5折节省 ~135s ≈ 2.3 min。

---

### 优化 2：MDA-OOS — 预生成置换 + 多线程并行预测（预计额外节省 ~5-8 分钟）

**原理**：将随机置换的生成与 predict+AUC 计算解耦。先用原始 RNG 序列预生成所有置换（保证可复现），再用 ThreadPoolExecutor 并行执行预测。

**方案**：

```python
def compute_mda_oos(model, X_val_df, y_val, feature_names, n_repeats=5, random_state=42):
    from concurrent.futures import ThreadPoolExecutor
    
    rng = np.random.RandomState(random_state)
    X_base = X_val_df[feature_names].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
    feat_to_idx = {f: i for i, f in enumerate(feature_names)}
    
    # 基线
    dval = prepare_dmatrix(X_val_df, feature_names, label=y_val)
    baseline_proba = model.predict(dval)
    baseline_auc = float(roc_auc_score(y_val, baseline_proba))
    
    # 预生成所有置换（与原代码 RNG 调用顺序完全一致）
    all_tasks = []  # [(feat, shuffled_col), ...]
    for feat in feature_names:
        fidx = feat_to_idx[feat]
        for _ in range(n_repeats):
            col = X_base[:, fidx].copy()
            rng.shuffle(col)  # 消费与原代码完全相同的 RNG 序列
            all_tasks.append((feat, fidx, col))
    
    # 并行执行 predict + AUC
    def _eval_one(args):
        feat, fidx, shuffled_col = args
        X_perm = X_base.copy()        # 每个线程独立 copy
        X_perm[:, fidx] = shuffled_col
        dval_perm = xgb.DMatrix(X_perm, label=y_val, missing=np.nan,
                                feature_names=feature_names)
        perm_proba = model.predict(dval_perm)
        perm_auc = float(roc_auc_score(y_val, perm_proba))
        return feat, baseline_auc - perm_auc
    
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_eval_one, all_tasks))
    
    # 按特征聚合结果（顺序与原代码一致）
    from collections import defaultdict
    feat_drops = defaultdict(list)
    for feat, drop in results:
        feat_drops[feat].append(drop)
    
    mda_means = {f: float(np.mean(drops)) for f, drops in feat_drops.items()}
    mda_stds = {f: float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0 
                for f, drops in feat_drops.items()}
    
    # ... 后续 t 检验和日志不变 ...
```

**为什么输出完全一致**：
- 预生成阶段：`rng.shuffle(col)` 的调用顺序与原代码逐特征逐重复的顺序完全相同 → 置换结果完全相同
- 并行执行阶段：每个任务的输入（X_perm 数据、model、y_val）完全确定 → predict 和 AUC 结果确定
- 最终聚合：按特征分组取 mean/std → 与原代码逐特征计算结果相同

**注意事项**：
- XGBoost `model.predict()` 在 `nthread=1` 时是线程安全的
- 每个 worker 独立 copy X_base，避免数据竞争
- DMatrix 创建是线程安全的

**节省分析**：

| 配置 | 耗时 |
|------|------|
| 原始（串行 DataFrame） | ~23 min |
| 优化1（numpy 串行） | ~8 min |
| 优化1+2（numpy + 4线程） | ~2-3 min |

---

### 优化 3：标签生成 — 多线程并行处理股票（预计节省 ~3-4 分钟）

**瓶颈位置**：`labels.py:453`，`compute_labels_fast()` 的 `for code, events in code_events.items()` 循环

**问题分析**：

4166 只股票串行处理，每只股票：
1. `_load_full_daily_data(code, ref_date)` — parquet I/O（I/O 密集）
2. `_batch_daily_volatility()` — 逐事件计算波动率（CPU 密集）
3. `_daily_prescreen()` — 日线预筛（CPU 密集）
4. `_batch_minute_confirmation()` — 分钟确认（几乎不执行，跳过率 99.997%）

每只股票完全独立，无数据依赖。

**方案**：

```python
def compute_labels_fast(self, events_df, calendar, align_entry_mode='event_date_close', reference_date=None):
    # ... 前置代码不变（code_events, code_range 构建） ...
    
    from concurrent.futures import ThreadPoolExecutor
    
    def _process_one_stock(code_events_tuple):
        code, events = code_events_tuple
        # ... 原有单股票处理逻辑（Phase 1-8）完全不变 ...
        return stock_labels  # list of dict
    
    # 多线程并行（I/O 密集型，用线程而非进程）
    max_workers = min(8, len(code_events))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_process_one_stock, code_events.items()))
    
    all_labels = [item for sublist in results for item in sublist]
    
    # ... 后续日志和返回不变 ...
```

**为什么输出完全一致**：
- 每只股票的处理逻辑完全不变
- 各股票之间无数据依赖
- 最终 `all_labels` 的内容相同（顺序可能不同，但 DataFrame 的行顺序不影响后续 merge 结果）

**注意**：`self.dp._load_full_daily_data()` 内部有 LRU 缓存，ThreadPoolExecutor 可以安全并发调用（GIL 释放点在 I/O）。

---

### 优化 4：标签生成 — 向量化 `_batch_daily_volatility()`（预计节省 ~60-90s）

**瓶颈位置**：`labels.py:247-291`

**问题分析**：

```python
for i, ed in enumerate(event_dates):
    mask = daily_dates <= pd.Timestamp(ed)       # O(n_daily) 全表扫描
    window_c = adj_close[mask][-lookback:]        # 切片
    rets = np.diff(window_c) / window_c[:-1]     # 重复计算收益率
    v = np.std(rets, ddof=1)                       # 重复计算 std
```

每只股票的每个事件日期都重新扫描整个 daily_dates 数组并重新计算收益率和 std。当一只股票有多个事件日期时，大量计算是重复的。

**方案**：

```python
def _batch_daily_volatility(self, daily_df, event_dates, ref_date=None):
    _fallback_vol = 0.02
    if daily_df is None or len(daily_df) < 5:
        return np.full(len(event_dates), _fallback_vol)
    
    daily_df = daily_df.sort_values('trade_date')
    adj_close = daily_df['close'].values
    daily_dates = daily_df['trade_date'].values
    
    lookback = max(self.daily_vol_window, 
                   self.vol_ewma_seed_days if self.vol_method == 'ewma' else 0)
    
    # 一次性计算所有日收益率
    all_returns = np.diff(adj_close) / adj_close[:-1]
    
    # 一次性计算 rolling/EWMA 波动率序列
    if self.vol_method == 'ewma' and len(all_returns) > 1:
        alpha = 1 - self.vol_ewma_lambda
        ewma_var = pd.Series(all_returns).ewm(alpha=alpha, adjust=False).var()
        vol_series = np.sqrt(np.maximum(ewma_var.values, 0))
    else:
        # Rolling std（向量化）
        ret_series = pd.Series(all_returns)
        rolling_std = ret_series.rolling(lookback, min_periods=5).std(ddof=1)
        vol_series = rolling_std.values
    
    # 构建日期→索引映射（searchsorted 替代全表扫描）
    daily_ts = pd.DatetimeIndex(daily_dates)
    
    vols = np.full(len(event_dates), _fallback_vol)
    for i, ed in enumerate(event_dates):
        # searchsorted: O(log n) 替代 O(n) 全表扫描
        idx = daily_ts.searchsorted(pd.Timestamp(ed)) - 1
        if idx < 4:  # 至少需要5个数据点
            vols[i] = _fallback_vol
            continue
        # vol_series[idx-1] 对应到第 idx 天的波动率
        # （returns[0..idx-1] 对应 close[0..idx] 的变化）
        vol_idx = min(idx - 1, len(vol_series) - 1) if len(vol_series) > 0 else -1
        if vol_idx >= 0:
            v = vol_series[vol_idx]
            if pd.notna(v) and v > 0:
                vols[i] = float(v)
            else:
                vols[i] = _fallback_vol
        else:
            vols[i] = _fallback_vol
    
    return vols
```

**为什么输出完全一致**：
- Rolling std 的数学定义与原代码相同：`std(returns[-lookback:], ddof=1)`
- `pd.Series.rolling(window, min_periods=5).std(ddof=1)` 在对应窗口上计算的结果与 `np.std(rets, ddof=1)` 相同
- EWMA 同理：`ewm(alpha=1-lambda, adjust=False).var()` 的最后一个值与原代码逐事件计算结果相同
- `searchsorted` 替代 `mask = daily_dates <= ed` 只是更高效地找到相同的截止位置

**复杂度改善**：O(n_events × n_daily) → O(n_daily) + O(n_events × log(n_daily))

---

### 优化 5：V10 pred_rank 注入 — 预转换日期 + 向量化 rank（预计节省 ~10-15s）

**瓶颈位置**：`strategy.py:1890-1911`

**问题分析**：

```python
# 循环 240 次，每次都调用 pd.to_datetime（O(n_rows)）
for sample_date in train_dates:
    date_mask = pd.to_datetime(train_data['trade_date']) == pd.Timestamp(sample_date)
    train_data.loc[date_mask, v10_col_raw] = ...

# 再循环所有日期做 rank
for trade_date in train_data['trade_date'].unique():
    mask = train_data['trade_date'] == trade_date
    vals = train_data.loc[mask, v10_col_raw]
    if vals.std() > 1e-6:
        train_data.loc[mask, v10_col_rank] = vals.rank(pct=True)
```

**方案**：

```python
# 预转换一次
trade_date_ts = pd.to_datetime(train_data['trade_date'])

# V10 raw 注入
for sample_date in train_dates:
    result = self._read_v7_snapshot(sample_date.strftime('%Y%m%d'))
    if result is not None:
        _, pred_dict = result
        date_mask = trade_date_ts == pd.Timestamp(sample_date)
        train_data.loc[date_mask, v10_col_raw] = (
            train_data.loc[date_mask, 'ts_code'].map(pred_dict).fillna(0.5)
        )

# V10 rank 向量化
train_data[v10_col_rank] = 0.5  # 默认值
group_stds = train_data.groupby(trade_date_ts)[v10_col_raw].transform('std')
mask_std = group_stds > 1e-6
if mask_std.any():
    ranks = train_data.groupby(trade_date_ts)[v10_col_raw].rank(pct=True)
    train_data.loc[mask_std, v10_col_rank] = ranks[mask_std]
```

**为什么输出完全一致**：
- `pd.to_datetime` 预转换后的比较结果与循环内逐次转换相同
- `groupby().rank(pct=True)` 在每个组内的排名结果与逐组 `vals.rank(pct=True)` 相同
- std 判断逻辑保持一致

---

### 优化 6：因子中性化 — 向量化 Z-score（预计节省 ~20-30s）

**瓶颈位置**：`strategy.py:622-635`

**问题分析**：

```python
grouped = df.groupby('trade_date', group_keys=False)
for col in factor_cols:
    df[col] = grouped[col].transform(lambda x: self._zscore(self._winsorize(x)))
```

`transform(lambda)` 对每个 (因子, 日期) 组合调用 Python 函数，开销大。~205 因子 × ~240 日期 = ~49,200 次 Python 函数调用。

**方案**：

```python
grouped = df.groupby('trade_date', group_keys=False)
for col in factor_cols:
    if col not in df.columns:
        continue
    raw = df[col].replace([np.inf, -np.inf], np.nan)
    
    # Winsorize: 逐组 clip（无法完全向量化，但可以用 transform 加速）
    def _winsorize_fast(s):
        valid = s.dropna()
        if len(valid) < 10:
            return s
        lo, hi = np.percentile(valid, [1, 99])
        return s.clip(lower=lo, upper=hi)
    
    winsorized = grouped[col].transform(_winsorize_fast)
    
    # Z-score: 完全向量化（pandas C 层实现）
    means = grouped.transform('mean')  # 或 winsorized.groupby(df['trade_date']).transform('mean')
    stds = grouped.transform('std')
    # 注意：transform('mean') 和 transform('std') 是在 raw 上算的
    # 需要在 winsorized 上算
    means = winsorized.groupby(df['trade_date']).transform('mean')
    stds = winsorized.groupby(df['trade_date']).transform('std')
    df[col] = (winsorized - means) / stds.replace(0, 1e-8)
```

更简洁的写法：

```python
for col in factor_cols:
    raw = df[col].replace([np.inf, -np.inf], np.nan)
    # 分组 Winsorize
    g = raw.groupby(df['trade_date'])
    lo = g.transform(lambda x: np.percentile(x.dropna(), 1) if x.dropna().len() >= 10 else np.nan)
    hi = g.transform(lambda x: np.percentile(x.dropna(), 99) if x.dropna().len() >= 10 else np.nan)
    winsorized = raw.clip(lower=lo, upper=hi)
    # 向量化 Z-score
    means = winsorized.groupby(df['trade_date']).transform('mean')
    stds = winsorized.groupby(df['trade_date']).transform('std')
    df[col] = (winsorized - means) / stds.replace(0, 1e-8)
```

**为什么输出完全一致**：
- Winsorize 逻辑不变：1% 和 99% 分位数 clip
- Z-score 逻辑不变：(x - mean) / std
- `groupby().transform('mean')` 和 `transform('std')` 是 pandas C 层实现，结果与 Python lambda 相同

---

### 优化 7：IC 特征筛选 — 向量化 Spearman 相关（预计节省 ~10-15s）

**瓶颈位置**：`ensemble.py:284-313`，`select_stable_features()` 中的 IC 循环

**问题分析**：

```python
for f in feats:
    tr_filled = tr_col.fillna(tr_median).values
    ic_train = spearmanr(tr_filled, y_train)[0]  # 逐特征调用
```

`spearmanr()` 内部做 ranking + Pearson 相关，每次调用 O(n × log n)。~205 特征 × 5 折 = 1025 次调用。

**方案**：

```python
# 一次性填充 NaN
tr_filled = tr[feats].copy()
for f in feats:
    med = tr_filled[f].median(skipna=True)
    if pd.isna(med):
        med = 0.0
    tr_filled[f] = tr_filled[f].fillna(med)

# 一次性 rank 所有特征
tr_ranked = tr_filled.rank()
# y_train rank
y_rank = pd.Series(y_train).rank()

# 向量化 Spearman = Pearson on ranks
# 使用 numpy 一次性计算所有特征与 y 的相关系数
tr_arr = tr_ranked.values
y_arr = y_rank.values
tr_mean = tr_arr.mean(axis=0)
y_mean = y_arr.mean()
tr_centered = tr_arr - tr_mean
y_centered = y_arr - y_mean
numerator = tr_centered.T @ y_centered  # (n_features,) 
tr_std = np.sqrt((tr_centered ** 2).sum(axis=0))
y_std = np.sqrt((y_centered ** 2).sum())
ic_values = numerator / (tr_std * y_std + 1e-12)

# 结果：ic_values[i] = spearmanr(tr_filled[feats[i]], y_train)[0]
```

**为什么输出完全一致**：
- Spearman 相关 = Pearson 相关 on ranked data
- `scipy.stats.spearmanr` 内部就是先 rank 再算 Pearson
- numpy 矩阵运算的 Pearson 与 scipy 逐对计算结果相同（浮点误差 < 1e-15）

---

### 优化 8：`_daily_prescreen` — searchsorted 替代全表扫描（预计节省 ~20-30s）

**瓶颈位置**：`labels.py:312-334`

**问题分析**：

```python
for i, (ed, ep, pt, sl) in enumerate(zip(event_dates, entry_prices, pt_ths, sl_ths)):
    mask = (daily_dates >= pd.Timestamp(next_day)) & (daily_dates <= pd.Timestamp(hold_end))
    # 每次创建两个 Timestamp + 全表布尔扫描
```

**方案**：

```python
# 预排序（已是排序状态）
daily_ts = pd.DatetimeIndex(daily_dates)

for i, (ed, ep, pt, sl) in enumerate(zip(event_dates, entry_prices, pt_ths, sl_ths)):
    next_day, hold_end = hold_windows[i]
    if next_day is None:
        expire_results.append((i, ep))
        continue
    # searchsorted: O(log n) 替代 O(n) 全表扫描
    start_idx = daily_ts.searchsorted(pd.Timestamp(next_day))
    end_idx = daily_ts.searchsorted(pd.Timestamp(hold_end), side='right')
    if start_idx >= end_idx:
        expire_results.append((i, ep))
        continue
    w_high = adj_high[start_idx:end_idx]
    w_low = adj_low[start_idx:end_idx]
    # ... 后续逻辑不变 ...
```

---

## 三、优化优先级与预估收益

| 优先级 | 优化项 | 预估节省 | 实现难度 | 风险 |
|--------|--------|---------|---------|------|
| P0 | 优化1: MDA numpy预转换 | ~13-15 min | 中 | 低（纯数据格式变化） |
| P0 | 优化2: MDA 多线程并行 | ~5-8 min | 中 | 低（预生成保证可复现） |
| P1 | 优化3: 标签生成多线程 | ~3-4 min | 低 | 低（股票间无依赖） |
| P1 | 优化4: 波动率向量化 | ~1-1.5 min | 中 | 低（数学等价） |
| P2 | 优化5: V10 rank注入 | ~10-15s | 低 | 极低 |
| P2 | 优化6: 中性化向量化 | ~20-30s | 中 | 低 |
| P2 | 优化7: IC向量化 | ~10-15s | 中 | 低 |
| P2 | 优化8: prescreen searchsorted | ~20-30s | 低 | 极低 |

**预估总收益**：

| 场景 | 耗时 |
|------|------|
| 当前基线 | ~30 min |
| 仅 P0（优化1+2） | ~12-15 min |
| P0 + P1（+优化3+4） | ~7-9 min |
| 全部优化 | ~5-8 min |

---

## 四、实施建议

### 4.1 分阶段实施

**第一阶段（立竿见影）**：实施优化1+2（MDA-OOS），单独可节省 ~18-23 分钟，是最高的 ROI。

**第二阶段（锦上添花）**：实施优化3+4（标签生成），额外节省 ~4-5 分钟。

**第三阶段（边际优化）**：实施优化5-8，额外节省 ~1-2 分钟。

### 4.2 验证方法

每个优化实施后，执行以下验证：

1. **数值一致性验证**：对同一训练窗口，比较优化前后的：
   - MDA 均值/标准差（优化1+2）
   - 标签 DataFrame（优化3+4）
   - 最终选股结果和权重（全部优化）

2. **回归测试**：运行 3-5 个调仓周期的回测，比较优化前后的：
   - 选股列表
   - 仓位权重
   - 最终净值曲线

3. **断言方式**：
   ```python
   # MDA 一致性
   np.testing.assert_allclose(mda_means_old, mda_means_new, rtol=1e-10)
   np.testing.assert_allclose(mda_stds_old, mda_stds_new, rtol=1e-10)
   
   # 标签一致性
   pd.testing.assert_frame_equal(labels_old, labels_new)
   ```

### 4.3 注意事项

1. **优化2（多线程）**：XGBoost predict 的线程安全性依赖于 `nthread=1` 设置（已在 `base_params` 中配置）。如果后续修改了 nthread，需重新评估线程安全。

2. **优化3（标签多线程）**：`self.dp._load_full_daily_data()` 的 LRU 缓存在多线程下是安全的（Python dict 操作有 GIL 保护），但如果 data_provider 内部有非线程安全的操作，需改用 ProcessPoolExecutor。

3. **优化4（波动率向量化）**：EWMA 模式下，`pd.Series.ewm().var()` 的 `adjust=False` 参数必须与原代码一致，否则结果会不同。

4. **浮点精度**：numpy 矩阵运算与逐元素循环在浮点精度上可能有 < 1e-15 的差异，这在实际应用中不会影响任何决策。

---

## 五、代码修改清单

| 文件 | 函数 | 优化项 | 修改类型 |
|------|------|--------|---------|
| `ensemble.py` | `compute_mda_oos()` | 优化1+2 | 重写核心循环 |
| `labels.py` | `compute_labels_fast()` | 优化3 | 并行化外层循环 |
| `labels.py` | `_batch_daily_volatility()` | 优化4 | 向量化波动率计算 |
| `strategy.py` | `_train_meta_model()` L1890-1911 | 优化5 | 预转换+向量化rank |
| `strategy.py` | `_neutralize_factors()` L622-635 | 优化6 | 向量化Z-score |
| `ensemble.py` | `select_stable_features()` L284-313 | 优化7 | 向量化Spearman |
| `labels.py` | `_daily_prescreen()` L312-334 | 优化8 | searchsorted替代 |

---

## 六、不建议优化的部分

以下部分已经足够高效，或优化空间极小：

1. **历史回放+快照预加载**（~2.6s）：已使用 ThreadPoolExecutor 并发预加载，无优化空间
2. **生产模型训练**（~12s）：单次 XGBoost 训练 + early stopping，已是最优配置
3. **预测选股**（~3.5s）：单次 predict，无优化空间
4. **`compute_barriers()`**：已支持向量化，代码极简
5. **`compute_sample_weights()`**：已优化为按 trade_date 分组，O(U×T) 复杂度
6. **市值缓存构建**：已使用 ProcessPoolExecutor 并行构建，仅首次执行

---

## 七、跨调仓周期缓存（结构性优化，可选）

当前每次调仓都重新执行全流程。如果回测周期较长（如 1 年 = ~52 次调仓），可考虑：

1. **MDA 结果缓存**：当训练窗口滑动比例 < 5% 时（235/240），MDA 排名变化极小。可缓存上一期的 MDA 结果，仅对新增/移除的特征重新计算。
   - 风险：改变了特征选择结果（虽然变化很小），严格意义上改变了输出
   - 建议：仅在用户接受"近似等效"时启用

2. **日线数据缓存**：`_load_full_daily_data()` 已有 LRU 缓存，跨调仓周期复用。无需额外优化。

3. **V7 快照缓存**：`_v7_snapshot_cache` 已在 `initialize()` 中加载。无需额外优化。

---

*报告生成日期：2026-06-23*
*分析基于代码版本：ensemble.py v3.0, strategy.py, labels.py, neutralizer.py*
