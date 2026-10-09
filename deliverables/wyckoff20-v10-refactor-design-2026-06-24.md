# Wyckoff V10 重构系统设计文档

> **日期**: 2026-06-24
> **目标**: 将 V5 完整合并进 V10，修复审查报告中全部 P0/P1/P2/P3 问题，AUC 门控替换 IC 门控，配置参数适配 120 特征/200 交易日，对齐 Alpha191 效率优化思路
> **前置文档**: `wyckoff20-v10-audit-report-2026-06-24.md`（审查报告）

---

## 一、总体架构设计

### 1.1 V10 独立化（脱离 V5 继承）

**当前状态**: `WyckoffV10Strategy` 继承 `Wyckoff20GPSEFStrategyV5`，大量核心逻辑在父类中。

**重构后**: V10 成为完全独立的类，直接继承 `IStrategy`，不再依赖 `gpsef_strategy.py`。V5 文件合并后删除。

**合并内容清单**（从 V5 迁移到 V10 内部）:

| V5 组件 | 行号范围 | 迁移方式 |
|---------|---------|---------|
| BLAS 环境变量设置 | L1-26 | 原样迁移到 V10 文件头 |
| `FactorNeutralizer` 类 | L61-185 | 原样迁移 |
| `XGBRegressorWrapper` | L192-209 | 迁移（V10 内部特征选择仍需用） |
| `XGBClassifierWrapper` | L212-247 | 原样迁移 |
| `compute_sample_weights` | L254-336 | 原样迁移 |
| `ALL_FACTORS` 列表 | L362-376 | 原样迁移 |
| `__init__` | L378-403 | 合并 V10 自有属性 |
| `initialize` | L409-497 | 合并 V10 覆盖逻辑 |
| `_preload_daily_data` | L503-531 | 原样迁移 |
| `_get_trading_day_offset` | L533-548 | 原样迁移 |
| `_get_previous_snapshot_date` | L550-574 | 原样迁移 |
| `_load_snapshot` | L576-585 | 原样迁移 |
| `_get_close_prices` | L587-595 | 原样迁移 |
| `_get_label_components` | L597-610 | 原样迁移 |
| `_get_snapshot_dates` | L612-627 | 原样迁移 |
| `_build_neutral_X` | L633-661 | 原样迁移 |
| `_neutralize` | L663-718 | 原样迁移（fit/transform 范式保留） |
| `_apply_score_neutralization` | L720-739 | 向量化优化（见 §8.2） |
| `_calc_psi` | L745-769 | 原样迁移 |
| `_sanitize_features` | L771-806 | 原样迁移 |
| `_select_features` | L808-983 | **重写**（见 §3） |
| `_build_training_dataset` (V5版) | L989-1260 | V10 已有覆盖版，保留 V10 版并改造 |
| `_prepare_features` | L1266-1308 | 原样迁移 |
| `_build_feature_matrix` | L1314-1366 | 原样迁移 |
| `_log_feature_importance` | L1408-1433 | 原样迁移 |

### 1.2 新增模块

| 模块 | 功能 | 来源 |
|------|------|------|
| `_purged_kfold_cv` | Purged K-Fold 交叉验证 + 跨折共识投票 | 新写，参考 alpha191 ensemble.py L1020-1140 |
| `_train_ensemble` | 5 种子模型集成训练 | 新写 |
| `_calibrate_proba` | Isotonic 概率校准 | 新写，参考 alpha191 `ProbabilityCalibrator` |
| `_compute_mda_oos` | OOS MDA + t 检验（优化版） | 移植 alpha191 ensemble.py L474-621 |
| `_compute_ic_vectorized` | 向量化 IC 计算 | 移植 alpha191 ensemble.py L312-336 |
| `_auc_gate` | AUC 门控逻辑 | 新写，替换原 IC 门控 |

### 1.3 重构后文件结构

```
strategies/wyckoff20/
├── __init__.py              # 保留
├── factors_v5.py            # 保留（40 因子引擎）
├── build_snapshots.py       # 保留（快照生成）
├── config_v10.yaml          # 更新（见 §9）
├── gpsef_strategy_v10.py    # 重写（V5合并 + 全部修复）
├── README.md                # 保留
├── data/                    # 保留
└── results/                 # 保留
```

**删除清单**: `gpsef_strategy.py`, `gpsef_strategy_v5_backup.py`, `gpsef_strategy_v6.py`, `gpsef_strategy_v7.py`, `gpsef_strategy_v8.py`, `gpsef_strategy_v9.py`, `gpsef_strategy_v11.py`, `config.yaml`, `config_v6.yaml`, `config_v7.yaml`, `config_v8.yaml`, `config_v9.yaml`, `config_v11.yaml`, `_tmp_patched.py`, `ml_scorer.py`, `volume_profile.py`, `wyckoff_engine.py`, `market_state_guard.py`, `__pycache__/`

---

## 二、Purged K-Fold 交叉验证（P0-1 修复）

### 2.1 设计方案

**参数**:
- K = 5 折
- embargo = 21 天（消除 20 日前向收益标签的重叠泄露）
- 跨折共识阈值：特征出现在 ≥ 3 折中 → 生产特征集

**数据划分逻辑**:

```
200 个训练快照日 → 按时间排序 → 分成 5 个连续块（每块 ~40 天）

Fold i (i=0..4):
  验证块 = block[i]
  训练块 = 所有日期 - [block[i] 前后各 21 天]
  
  每折内：
    1. 训练块再按时间分前后半：前半算 IC，后半做 PSI 对比
    2. 执行 IC → PSI → MDI → MDA(t检验) 特征选择
    3. 记录该折选出的特征集

跨折共识：
  feature_count[f] = 出现在多少折中
  production_features = [f for f if feature_count[f] >= 3]
```

### 2.2 关键实现细节

1. **快照日作为分割单位**（非交易日）：V10 的训练样本以快照日为锚点，前向 20 日标签窗口重叠。用快照日分割可精确控制 embargo。

2. **embargo 计算**：验证块前后各 purge 21 个快照日（而非 21 个交易日）。由于快照日本身就是交易日且间隔约 1 天，21 个快照日 ≈ 21 个交易日，恰好覆盖 20 日标签窗口 + 1 天缓冲。

3. **生产模型训练**：在全部 200 天训练数据上，用共识特征集训练 5 种子集成模型（见 §4）。

4. **fallback 机制**：如果共识特征数 < `min_features`(10)，则取跨折出现频次最高的前 10 个特征；如果完全无共识，回退到 IC 排名前 15。

### 2.3 与原单次划分的对比

| 维度 | 原方案（80/20 单次） | 新方案（Purged K-Fold） |
|------|---------------------|----------------------|
| 特征选择稳定性 | 依赖单一分割 | 跨 5 折投票，稳定性 √5 |
| 标签泄露 | 无 embargo | 21 天 embargo 隔离 |
| 数据利用率 | 80% 训练 | 每折 ~80% 训练（purge 后）|
| 计算量 | 1× | 5×（但折内特征选择更快）|

### 2.4 性能优化

折内特征选择只需处理 ~160 天数据（vs 原来 200 天），且每折独立。5 折顺序执行，但由于折内 IC 计算已向量化（见 §8.1），总耗时预计与原单次划分持平或略增 20-30%。

---

## 三、特征选择管线重写（P0-1 + P1-1 + P1-2 + P1-6 + P3-4）

### 3.1 新管线流程

```
输入: train_data, val_data, feats (候选特征)

Step 0: 低方差过滤（VarianceThreshold threshold=0，仅移除常量列）

Step 1: 缺失率过滤 + 高相关去重（_sanitize_features）

Step 2: PSI 过滤（新增，P1-2 修复）
  for each feat:
    psi = calc_psi(train[feat], val[feat])
    if psi > 0.15: drop
  
Step 3: 向量化 IC 过滤（优化，P1-1 相关）
  一次性 rank 所有通过特征 → Pearson on ranks 矩阵运算
  |IC| >= 0.02 → 通过
  
Step 4: MDI 过滤（FI 模型加 Early Stopping，P1-6 修复）
  fi_model = xgb.train(..., early_stopping_rounds=20, evals=[(dtrain,'train'),(dval,'val')])
  MDI = gain importance > 0
  
Step 5: MDA t 检验（重写，P1-1 修复）
  compute_mda_oos(fi_model, X_val, y_val, n_repeats=20)
  t_crit = t.ppf(0.95, df=19) = 1.729
  通过条件: mean > t_crit * (std / sqrt(20))
  
Step 6: min/max_features 约束（新增，P3-4 修复）
  if len(final) < min_features(10): 放宽阈值，取 IC 排名前 10
  if len(final) > max_features(80): 取 composite_score 前 80
```

### 3.2 IC 向量化实现（移植 alpha191 优化）

```python
# 原实现：逐特征 spearmanr，O(N×F) Python 循环
for f in feats:
    ic = spearmanr(train[f].values, y_train)[0]  # 慢

# 新实现：矩阵运算，O(1) Python 调用
tr_ranked = train[feats].fillna(median).rank()  # 一次性 rank
y_rank = pd.Series(y_train).rank()
tr_arr = tr_ranked.values  # [N, F]
y_arr = y_rank.values      # [N]
tr_centered = tr_arr - tr_arr.mean(axis=0)
y_centered = y_arr - y_arr.mean()
numerator = tr_centered.T @ y_centered      # [F]
tr_ss = np.sqrt((tr_centered**2).sum(axis=0))
y_ss = np.sqrt((y_centered**2).sum())
ic_vector = numerator / (tr_ss * y_ss + 1e-12)  # [F]
```

**预期加速**: 120 特征 × 80 万行，原 ~30s → 新 ~3s（10x）

### 3.3 MDA 优化版实现（移植 alpha191 优化）

```python
def _compute_mda_oos(self, model, X_val_df, y_val, feature_names, n_repeats=20):
    # 1. 预转换为 numpy float32（避免每次迭代 DataFrame copy）
    X_base = X_val_df[feature_names].replace([inf, -inf], nan).values.astype(float32)
    
    # 2. 基线 AUC
    baseline_auc = roc_auc_score(y_val, model.predict(dval))
    
    # 3. 预生成所有置换列（串行，RNG 顺序确定）
    all_tasks = []
    for feat in feature_names:
        for _ in range(n_repeats):
            col = X_base[:, idx].copy()
            rng.shuffle(col)
            all_tasks.append((feat, idx, col))
    
    # 4. 并行 predict + AUC（ThreadPoolExecutor, max_workers=4）
    #    XGBoost predict 释放 GIL，threading 高效
    
    # 5. t 检验
    t_crit = t.ppf(0.95, df=n_repeats-1)  # 1.729 for n_repeats=20
    pass_condition: mean > t_crit * (std / sqrt(n_repeats))
```

**预期加速**: 120 特征 × 20 repeats = 2400 次 predict，ThreadPoolExecutor 并行 → ~4x 加速

### 3.4 FI 模型 Early Stopping（P1-6 修复）

```python
# 原实现：num_boost_round=500 全量训练
fi_model = xgb.train(fi_params, dtrain_fi, num_boost_round=500, verbose_eval=False)

# 新实现：加验证集 + early stopping
fi_model = xgb.train(
    fi_params, dtrain_fi,
    num_boost_round=500,
    evals=[(dtrain_fi, 'train'), (dval_fi, 'val')],
    early_stopping_rounds=20,
    verbose_eval=False
)
```

---

## 四、5 种子模型集成（P0-3 修复）

### 4.1 设计方案

```python
SEEDS = [42, 43, 44, 45, 46]
self.models = []  # List[xgb.Booster]

for seed in SEEDS:
    params = self.xgb_params.copy()
    params['seed'] = seed
    model = xgb.train(
        params, dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=20,
        verbose_eval=False
    )
    self.models.append(model)
```

### 4.2 预测集成

```python
# 5 个模型概率取平均
all_preds = np.column_stack([m.predict(dpred) for m in self.models])
preds = all_preds.mean(axis=1)
```

### 4.3 性能考虑

- 5 个模型串行训练，每个 ~10-20s，总计 ~50-100s
- 预测时 5 次 DMatrix.predict，每次 ~0.1s，总计 ~0.5s（可忽略）
- 可选优化：ThreadPoolExecutor 并行训练（XGBoost 释放 GIL），但需确保 nthread=1 时线程安全

### 4.4 AUC 评估

集成模型的 AUC 用平均概率计算：
```python
val_auc = roc_auc_score(y_val, mean_preds)
```

---

## 五、Isotonic 概率校准（P0-4 修复）

### 5.1 设计方案

```python
from sklearn.isotonic import IsotonicRegression

# 训练阶段：在验证集预测上拟合校准器
val_preds = self.models[0].predict(dval)  # 用第一个模型的验证集预测
# 或用集成平均
val_preds_ensemble = np.mean([m.predict(dval) for m in self.models], axis=0)

self.calibrator = IsotonicRegression(y_min=0.05, y_max=0.95, out_of_bounds='clip')
self.calibrator.fit(val_preds_ensemble, y_val)

# 预测阶段：先过校准器
raw_preds = np.mean([m.predict(dpred) for m in self.models], axis=1)
calibrated_preds = self.calibrator.predict(raw_preds)
calibrated_preds = np.clip(calibrated_preds, 0.05, 0.95)  # P3-2 修复
```

### 5.2 校准器退化检测

```python
# 检测校准后概率范围
prob_range = calibrated_preds.max() - calibrated_preds.min()
if prob_range < 0.01:
    logger.warning("概率校准退化：模型可能无区分力")
```

---

## 六、AUC 门控替换 IC 门控（P0-2 + P3-3 修复）

### 6.1 门控逻辑

```python
# 计算集成模型验证集 AUC
val_auc = roc_auc_score(y_val, val_preds_ensemble)

# AUC 门控三档
if val_auc >= 0.55:
    confidence_level = "normal"
    position_multiplier = 1.0
elif val_auc >= 0.50:
    confidence_level = "reduced"
    position_multiplier = 0.5
else:
    confidence_level = "defensive"
    position_multiplier = 0.25
```

### 6.2 配置参数变更

**删除**（IC 门控冗余参数）:
```yaml
# 删除以下三行
val_ic_trade_threshold: -10
val_ic_reduce_threshold: -10
val_ic_defend_threshold: -10
```

**新增**:
```yaml
# AUC 门控
auc_gate_normal: 0.55    # AUC >= 0.55 → 正常模式，仓位 1.0x
auc_gate_reduced: 0.50   # 0.50 <= AUC < 0.55 → 减仓模式，仓位 0.5x
                         # AUC < 0.50 → 防御模式，仓位 0.25x
```

### 6.3 代码清理

1. **V5 `initialize` 中的 IC 门控参数读取**（L437-440）→ 删除
2. **V10 `before_market_open` 中的 IC 门控逻辑**（L641-654）→ 替换为 AUC 门控
3. **V10 `_recomd_daily_select` 中的 IC 门控逻辑**（L986-998）→ 替换为 AUC 门控
4. **`self.val_ic` 属性** → 改为 `self.val_auc`
5. **`_calc_model_performance`** → 保留 AUC 计算，移除 IC 用于门控的代码路径
6. **recomd 模式门控阈值统一**（P2-7 修复）→ 回测和 recomd 模式使用相同 AUC 阈值

---

## 七、其他 P2/P3 修复

### 7.1 执行参数补齐（P2-1）

```yaml
execution:
  commission_rate: 0.0003
  min_commission: 5.0          # 新增
  transfer_fee_rate: 1.0e-05   # 新增
  tax_rate: 0.0005
  slippage: 0.001
  max_positions: 50
  max_position_pct: 0.10
```

### 7.2 price_min 提高（P2-2）

```yaml
# 修改
- "price_min:3"   # 原 price_min:2
```

### 7.3 XGBoost 参数统一（P2-3）

删除 `v10:` 块中的冗余 XGBoost 参数定义，统一在顶层定义一次。`v10:` 块只保留 V10 特有参数（`label_threshold`）。

### 7.4 tree_method=hist（P2-4）

```python
self.xgb_params = {
    ...
    'tree_method': 'hist',   # 新增
    'device': 'cpu',          # 新增
    'nthread': 1,
}
```

### 7.5 概率裁剪（P3-2）

```python
preds = np.clip(preds, 0.05, 0.95)
```

已在 §5 Isotonic 校准后执行。

### 7.6 min/max_features 约束（P3-4）

```python
MIN_FEATURES = 10
MAX_FEATURES = 80  # 120 候选特征 × 2/3 上限

if len(final_feats) < MIN_FEATURES:
    # 放宽 MDA 阈值，取 IC 排名前 10
    final_feats = sorted_by_ic[:MIN_FEATURES]
elif len(final_feats) > MAX_FEATURES:
    # 取 composite_score 前 80
    final_feats = sorted_by_composite[:MAX_FEATURES]
```

### 7.7 局部 RNG（P3-1）

```python
# 原实现
np.random.seed(42)

# 新实现
rng = np.random.RandomState(42)
```

在 `_build_training_dataset` 和 `recommend` 中使用局部 RNG。

### 7.8 softmax 温度（P1-7）

```yaml
pred_weight_temperature: 0.3   # 原 0.1
```

### 7.9 min_child_weight（P1-5）

```yaml
xgb_min_child_weight: 10   # 原 50
```

### 7.10 IV 减仓冷却期（P2-6）

调仓日计算目标权重时，如果当前持仓在 IV 减仓列表中且减仓时间 < 5 个交易日，跳过该股票的买入（避免反向操作）。

---

## 八、性能优化（对齐 Alpha191）

### 8.1 向量化 IC 计算

见 §3.2，预期 10x 加速。

### 8.2 向量化中性化（score_neutralization）

```python
# 原实现：逐列循环
for col in score_cols:
    series = df[col].replace([inf, -inf], nan)
    lower, upper = np.percentile(series.dropna(), (1, 99))
    series = series.clip(lower=lower, upper=upper)
    mean, std = series.mean(), series.std()
    df[col] = (series - mean) / std

# 新实现：向量化 Winsorize + Z-score
# 利用 groupby.transform 替代逐列循环
```

由于 score/conf 列的处理是独立的逐列操作，向量化空间有限。但可以用 `df.apply` 替代显式循环，减少 Python 开销。

### 8.3 MDA 并行化

见 §3.3，ThreadPoolExecutor 并行 predict，预期 4x 加速。

### 8.4 特征选择缓存策略

Purged K-Fold 下，每次调仓的 200 天窗口与上次重叠 ~180 天（移动 20 天）。但跨折投票已提供稳定性保证，不再需要额外缓存。

可选优化：如果连续两次调仓的快照日列表完全一致（recomd 模式可能发生），直接复用上次的共识特征集。

### 8.5 DMatrix 构建优化

```python
# 原实现
X_train = train_data[feats].values.astype(np.float32)

# 新实现：直接用 DataFrame 构建 DMatrix（XGBoost 原生支持）
# 避免不必要的 .values 拷贝
dtrain = xgb.DMatrix(
    train_data[feats].replace([np.inf, -np.inf], np.nan).astype(np.float32),
    label=y_train,
    weight=sample_weights,
    missing=np.nan,
    feature_names=feats
)
```

### 8.6 预测时全市场中性化（P1-4 修复）

```python
# 原实现：仅对候选池做中性化
snap = self._neutralize(snap, raw_cols)  # 候选池 ~2000 只

# 新实现：对全市场做中性化，再提取候选池
full_snap = self._load_snapshot(prev)  # 全市场 ~4500 只
full_snap = self._neutralize(full_snap, raw_cols)  # 全市场中性化
snap = full_snap[full_snap['code'].isin(cand_digits)]  # 提取候选池
```

这样训练（全市场截面）和预测（全市场截面）的 Winsorize 百分位和 Z-score 统计量一致。

---

## 九、配置参数适配（120 特征 / 200 交易日）

### 9.1 完整 config_v10.yaml 变更表

| 参数 | 原值 | 新值 | 理由 |
|------|------|------|------|
| `xgb_min_child_weight` | 50 | 10 | P1-5：对齐 Alpha191，logistic loss 下等效 ~40 样本可分裂 |
| `xgb_tree_method` | (缺) | hist | P2-4：跨机器确定性 |
| `xgb_device` | (缺) | cpu | P2-4：显式指定 |
| `pred_weight_temperature` | 0.1 | 0.3 | P1-7：降低个股集中度，权重差异从 7.4x 降到 2x |
| `feat_sel_psi_threshold` | 0.25 | 0.15 | P1-2：Alpha191 标准 PSI<0.15 |
| `feat_sel_mda_n_repeats` | 5 | 20 | P1-1：支持 t 检验，假阳性率 50%→5% |
| `feat_sel_min_ic` | 0.02 | 0.02 | 保持不变（二分类 IC 较低） |
| `feat_sel_max_corr` | 0.95 | 0.90 | 120 特征下更严格去重 |
| `price_min` | 2 | 3 | P2-2：过滤低价操纵股 |
| `val_ic_*_threshold` | -10 ×3 | 删除 | P0-2：AUC 门控替代 |
| `auc_gate_normal` | (缺) | 0.55 | 新增 |
| `auc_gate_reduced` | (缺) | 0.50 | 新增 |
| `cv_folds` | (缺) | 5 | 新增：Purged K-Fold |
| `cv_embargo_days` | (缺) | 21 | 新增：标签窗口隔离 |
| `cv_feature_consensus` | (缺) | 3 | 新增：≥3 折共识 |
| `ensemble_seeds` | (缺) | [42,43,44,45,46] | 新增：5 种子集成 |
| `min_features` | (缺) | 10 | 新增：P3-4 |
| `max_features` | (缺) | 80 | 新增：P3-4，120×2/3 |
| `min_commission` | (缺) | 5.0 | 新增：P2-1 |
| `transfer_fee_rate` | (缺) | 1.0e-05 | 新增：P2-1 |
| `prob_clip_min` | (缺) | 0.05 | 新增：P3-2 |
| `prob_clip_max` | (缺) | 0.95 | 新增：P3-2 |

### 9.2 参数适配理由

**120 特征的影响**:
- 候选特征 120 个（40 因子 × 3 列），加上 3 个派生特征 = 123 个
- `max_features=80`：保留 2/3，允许模型有足够特征选择空间
- `min_features=10`：即使特征选择严格，也保证最低 10 个特征
- `max_corr=0.90`：120 特征中相关性较高，需要更严格的去重

**200 交易日训练窗口的影响**:
- 200 天 × ~4500 股票 = ~90 万行训练数据
- 5 折 CV：每折 ~40 天验证 + ~120 天训练（purge 后）
- `min_child_weight=10`：90 万行 / 10 = 9 万个叶子节点上限，足够拟合
- `n_repeats=20`：200 天数据足够支撑 20 次置换的统计效力
- `embargo=21`：恰好覆盖 20 日前向标签窗口

---

## 十、测试计划

### 10.1 测试模块结构

```
tests/
├── test_v10_purged_kfold.py      # Purged K-Fold CV 测试
├── test_v10_feature_selection.py  # 特征选择管线测试（IC向量化 + PSI + MDA t检验）
├── test_v10_ensemble.py           # 5种子集成测试
├── test_v10_calibration.py        # Isotonic 校准测试
├── test_v10_auc_gate.py           # AUC 门控测试
├── test_v10_neutralization.py     # 中性化一致性测试
├── test_v10_merge.py              # V5合并完整性测试
└── test_v10_e2e.py                # 端到端冒烟测试
```

### 10.2 测试用例设计

#### test_v10_purged_kfold.py
- `test_kfold_splits`: 验证 5 折划分正确，每折验证块大小 ≈ 40 天
- `test_embargo_isolation`: 验证 embargo 区间内无训练样本
- `test_consensus_voting`: 验证特征出现在 ≥3 折才进入生产特征集
- `test_fallback_min_features`: 验证共识特征 < 10 时 fallback 逻辑
- `test_determinism`: 同一输入两次运行结果完全一致

#### test_v10_feature_selection.py
- `test_ic_vectorized_matches_loop`: 向量化 IC 与逐特征 spearmanr 结果一致
- `test_psi_filter`: PSI > 0.15 的特征被过滤
- `test_mda_ttest`: t 检验通过条件正确（mean > t_crit * SE）
- `test_mda_n_repeats`: n_repeats=20 时的统计效力
- `test_fi_early_stopping`: FI 模型使用 early stopping
- `test_min_max_features`: 特征数约束生效

#### test_v10_ensemble.py
- `test_five_seeds`: 5 个模型使用不同种子
- `test_prediction_average`: 预测取 5 模型平均
- `test_ensemble_variance`: 集成预测方差 < 单模型方差

#### test_v10_calibration.py
- `test_isotonic_fit_predict`: 校准器正确拟合和预测
- `test_prob_clip`: 概率裁剪到 [0.05, 0.95]
- `test_degradation_detection`: 退化检测逻辑

#### test_v10_auc_gate.py
- `test_normal_mode`: AUC >= 0.55 → position_multiplier=1.0
- `test_reduced_mode`: 0.50 <= AUC < 0.55 → 0.5
- `test_defensive_mode`: AUC < 0.50 → 0.25
- `test_gate_consistency`: 回测和 recomd 模式使用相同阈值

#### test_v10_neutralization.py
- `test_full_market_neutralize`: 预测时全市场中性化
- `test_fit_transform_consistency`: fit/transform 范式一致性
- `test_score_neutralization`: score 列 Winsorize+Z-score 正确

#### test_v10_merge.py
- `test_no_v5_import`: V10 文件不 import gpsef_strategy
- `test_all_factors_present`: ALL_FACTORS 列表完整（40 个）
- `test_all_methods_present`: 所有关键方法存在
- `test_compute_sample_weights`: 序贯权重计算正确

#### test_v10_e2e.py
- `test_smoke_train_predict`: 小数据集端到端训练→预测→选股
- `test_recomd_mode`: recomd 模式端到端流程

### 10.3 测试数据

使用合成数据（小规模，~20 天 × ~100 股票）进行单元测试，确保快速运行。端到端测试使用真实快照数据（取最近 30 天）。

---

## 十一、实施顺序

```
Step 1: 编写系统设计文档（本文档）→ 用户审核
  ↓
Step 2: 更新 config_v10.yaml（配置先行，代码依赖配置）
  ↓
Step 3: 重写 gpsef_strategy_v10.py
  3a: V5 代码合并（不加新功能，确保合并后行为不变）
  3b: 逐项修复 P0-P3（每项修复后编写对应测试）
  3c: 性能优化移植
  ↓
Step 4: 编写测试模块（与 Step 3 同步进行）
  ↓
Step 5: 运行全部测试，确保通过
  ↓
Step 6: 删除无用文件
  ↓
Step 7: 更新工作日志和记忆
```

### 11.1 代码重写顺序（Step 3 内部）

```
3a. 文件骨架：BLAS 设置 → imports → FactorNeutralizer → XGB Wrappers → compute_sample_weights
3b. WyckoffV10Strategy 类定义：ALL_FACTORS + __init__
3c. initialize（合并 V5 + V10，AUC 门控参数，tree_method=hist）
3d. 数据加载工具方法（原样迁移）
3e. 中性化方法（原样迁移 + 全市场优化）
3f. _compute_ic_vectorized（新增，向量化 IC）
3g. _compute_mda_oos（新增，优化 MDA + t 检验）
3h. _select_features（重写：PSI + 向量化IC + MDA t检验 + min/max_features）
3i. _purged_kfold_cv（新增）
3j. _train_ensemble（新增，5 种子集成）
3k. _calibrate_proba（新增，Isotonic 校准）
3l. _build_training_dataset（改造：二分类标签 + Purged K-Fold）
3m. _calc_model_performance（保留 AUC，移除 IC 门控）
3n. _auc_gate（新增，AUC 门控）
3o. before_market_open（改造：集成训练 + 校准 + AUC 门控 + 概率裁剪）
3p. handle_bar（原样保留）
3q. recommend（改造：集成 + 校准）
3r. _recomd_daily_select（改造：AUC 门控统一）
3s. 趋势过滤方法（原样保留）
3t. 其他工具方法（原样保留）
```

---

## 十二、风险控制

### 12.1 高风险修改

1. **Purged K-Fold CV**（P0-1）：改变特征选择核心逻辑
   - 缓解：保留 fallback 机制；先测试 CV 划分正确性再测特征选择
   - 回退方案：如果 CV 导致策略表现显著下降，可配置 `cv_folds=1` 回退到单次划分

2. **V5 合并**：大量代码迁移，可能遗漏
   - 缓解：编写 `test_v10_merge.py` 验证所有方法存在且可调用
   - 回退方案：git 历史可恢复 V5 文件

3. **全市场中性化**（P1-4）：改变预测路径
   - 缓解：先在 recomd 模式验证中性化结果一致性

### 12.2 低风险修改

- AUC 门控替换 IC 门控（P0-2）：配置驱动，逻辑简单
- 5 种子集成（P0-3）：增加训练，不改逻辑
- Isotonic 校准（P0-4）：后处理步骤
- MDA t 检验（P1-1）：特征选择更严格，不影响已有逻辑
- PSI 过滤（P1-2）：新增过滤步骤
- 配置参数调整（P1-5, P1-7, P2-1, P2-2, P2-4）：配置驱动

---

## 十三、预期收益总结

| 修复项 | 预期收益 |
|--------|---------|
| Purged K-Fold CV | 特征稳定性提升 ~√5 倍，回测过拟合风险降低 |
| 5 种子集成 | 预测方差降低 ~2 倍，选股稳定性提升 |
| Isotonic 校准 | softmax 权重更准确，个股集中度降低 |
| AUC 门控 | 模型失效时自动减仓，尾部风险保护 |
| MDA t 检验 | 假阳性率从 50% 降到 5% |
| PSI 过滤 | 过滤分布不稳定特征，OOS 表现退化降低 |
| 向量化 IC | 特征选择速度提升 10x |
| MDA 并行 | MDA 计算速度提升 4x |
| 全市场中性化 | 训练-预测分布一致，预测失准降低 |

---

*本设计文档供用户审核，审核通过后开始执行 Step 2-7。*
