# Wyckoff V10 策略 — 机构级全流程审核报告

**审核日期**: 2026-07-11
**审核范围**: `E:\quant\gpsef_project\strategies\wyckoff20\` 全部最新脚本与配置
**代码版本**: gpsef_strategy_v10.py (3098行, 2026-07-11 10:51更新) + config_v10.yaml (356行, 2026-07-11 12:46更新)
**审核人**: 交易分析团队 · 何执舟（Chief Strategist）
**评级**: **B+** （框架扎实，存在2个P0需立即修复，若干P1/P2需关注）

---

## 一、审核范围确认

| 文件 | 行数 | 最后修改 | 审核状态 |
|------|------|---------|---------|
| `config_v10.yaml` | 356 | 2026-07-11 12:46 | ✅ 已审 |
| `gpsef_strategy_v10.py` | 3098 | 2026-07-11 10:51 | ✅ 已审 |
| `barra_builder.py` | 701 | 2026-07-10 23:29 | ✅ 已审 |
| `barra_constraint.py` | 467 | 2026-07-11 10:24 | ✅ 已审 |
| `build_barra_snapshots.py` | 188 | 2026-07-10 22:50 | ✅ 已审 |
| `tests/conftest.py` | 226 | 2026-07-11 10:26 | ✅ 已审 |
| `factors_v5.py` | ~2000+ | 2024-05-24 | ⚠️ 未深入审查（因子计算逻辑，非本次重点） |

---

## 二、发现总览

| 优先级 | 数量 | 定义 |
|--------|------|------|
| **P0 — 致命/必须立即修复** | 2 | 导致错误交易结果或运行时崩溃 |
| **P1 — 重大** | 5 | 显著偏离业界最佳实践，影响回测可信度 |
| **P2 — 中等** | 4 | 性能/健壮性/可维护性问题 |
| **P3 — 轻微** | 3 | 代码质量/风格问题 |

---

## 三、P0 发现（必须立即修复）

### P0-1: SHAP 分析覆写生产模型 —— CV集成模式静默降级为种子集成

**位置**: `gpsef_strategy_v10.py` 第 2113 行

**问题描述**:
```python
# 第 2098-2113 行 (SHAP 分析块内)
if self._shap_enabled and self.feature_names:
    # ... 准备临时 DMatrix ...
    self.models = self._train_ensemble(dtrain_tmp, dval_tmp, force_seed=True)  # ← 覆写！
    shap_vals, X_sample = self._compute_shap_values(X_full)
```

当 `shap.enabled: true` 时，SHAP 因子质量验证流程会调用 `_train_ensemble(force_seed=True)` 训练一个**种子集成模型**，并将结果赋值给 `self.models`。这会**覆写**之前 `_train_cv_ensemble()` 训练好的 CV 折集成模型。

**后果链**:
1. `self.models` 从 CV 折集成（5个独立验证集模型，超参数多样性）→ 种子集成（5个同验证集模型）
2. `self.model_aucs` 和 `self.model_weights` 在第 2113 行之后未被恢复
3. 后续 `_predict_ensemble()` 使用种子模型预测 → 预测区分度大幅下降
4. 最危险的是：**此 bug 完全静默**，无 warning 无 error 日志中仅显示 "Ensemble training complete (seed_ensemble)"

**触发条件**: `shap.enabled: true`（当前配置为 false，故为**休眠 bug**；但一旦启用 SHAP 立即激活）

**修复方案**:
```python
# 修改前（当前代码）
self.models = self._train_ensemble(dtrain_tmp, dval_tmp, force_seed=True)

# 修改后：使用临时变量，不覆写 self.models
_shap_models = self._train_ensemble(dtrain_tmp, dval_tmp, force_seed=True)
# 用 _shap_models 计算 SHAP，之后丢弃
original_models = self.models
self.models = _shap_models
shap_vals, X_sample = self._compute_shap_values(X_full)
self.models = original_models  # 恢复生产模型
self.model_aucs = saved_model_aucs      # 恢复
self.model_weights = saved_model_weights  # 恢复
self._active_models = saved_active_models  # 恢复
```

---

### P0-2: QP 求解未定义变量 `x0` — 边界情况运行时崩溃

**位置**: `barra_constraint.py` 第 309 行

**问题描述**:
```python
# 第 307-309 行
weights = np.maximum(weights, 0)   # 确保非负
weights = np.minimum(weights, max_weight)  # 确保不超上限
weights = weights / weights.sum() if weights.sum() > 0 else x0  # ← NameError!
```

当所有权重被裁剪为零（极端市场条件下可能出现），`x0` 变量未在函数作用域内定义 → **`NameError: name 'x0' is not defined`**，导致整个调仓流程中断。

**触发条件**:
- Barra 约束优化返回的全零权重向量（理论上可能发生在所有 alpha 接近零且 lambda_dev 极大时）
- 或 CVXOPT 返回全零解

**修复方案**:
```python
# 修改后
if weights.sum() > 1e-10:
    weights = weights / weights.sum()
else:
    logger.warning("[Barra] All weights zero after clipping, fallback to equal weight")
    n_valid = len(weights)
    weights = np.ones(n_valid) / n_valid
```

---

## 四、P1 发现（重大偏离最佳实践）

### P1-1: CV折集成忽略 `xgb_class_balance` 配置——始终强制类别平衡

**位置**: `gpsef_strategy_v10.py` 第 1476-1514 行

**问题描述**:

在 `_train_cv_ensemble()`（CV折集成路径）中，类别平衡权重的计算逻辑是：

```python
# 第 1476-1478 行：始终计算 spw，无视 _class_balance_mode 配置
n_pos = int((y_fold_train == 1).sum())
n_neg = int((y_fold_train == 1).sum())
spw = float(n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0

# 第 1514 行：始终注入 fold_params
fold_params = {...
    'scale_pos_weight': spw,  # 始终注入！
}
```

而在 `_train_ensemble()`（种子集成路径）中，正确地调用了：
```python
params = self._apply_class_balance(params, y_train)  # 尊重配置
```

`_apply_class_balance()` 会检查 `self._class_balance_mode == 'auto'`，仅在 auto 模式下注入 `scale_pos_weight`。

**后果**:
- 配置 `xgb_class_balance: "none"` 在 CV 折集成模式下**完全不生效**
- median 标签模式正负样本约 50:50 → spw≈1.0 → 影响较小
- 但切换到 quantile(0.30) 模式时 → 正样本30%，spw≈2.33 → 模型行为与配置声明的 "none" 不一致
- **两种集成路径的行为不一致**，增加调试复杂度

**修复方案**:
```python
# 在 _train_cv_ensemble 的 fold_params 构建处
fold_params = {
    ...
    'scale_pos_weight': spw if self._class_balance_mode == 'auto' else 1.0,
}
```

---

### P1-2: 测试夹具 AUC 门控阈值与实际配置不一致

**位置**: `tests/conftest.py` 第 53-55 行 vs `config_v10.yaml` 第 117-118 行

**问题描述**:

| 参数 | conftest.py (测试) | config_v10.yaml (实际运行) | 差异? |
|------|-------------------|--------------------------|-------|
| `auc_gate_full` | 0.55 | 0.52 | **+0.03** |
| `auc_gate_min` | 0.50 | 0.48 | **+0.02** |

测试用例使用的门控阈值比生产配置宽松 2-3 个百分点。这意味着：

1. 生产环境中 AUC=0.51 应该触发 empty（空仓），但测试中会被当作 half（半仓）
2. 所有依赖 AUC 门控的测试（`test_v10_auc_gate.py` 等）都在用错误的阈值验证
3. 可能出现"测试全部通过但线上行为异常"的情况

**修复方案**:
```python
# conftest.py make_strategy() 中
s._auc_gate_full = 0.52   # 对齐 config_v10.yaml
s._auc_gate_min = 0.48    # 对齐 config_v10.yaml
```

---

### P1-3: 概率坍缩检测被禁用 (`threshold=0.000`) —— 无保护机制

**位置**: `config_v10.yaml` 第 110 行 + `gpsef_strategy_v10.py` 第 1842-1880 行

**问题描述**:

```yaml
prob_collapse_std_threshold: 0.000   # ← 等于关闭
```

虽然已通过 `max_depth: 6` + `calibration_method: none` 从根源上缓解了概率坍缩问题，但完全移除检测意味着：

1. 若某次训练因数据异常导致 best_iter=0（所有预测 ≈ 0.50），系统**不会检测到**
2. 模型会在无区分力状态下选股并分配仓位
3. 这相当于在抛硬币的基础上加了一层"看起来很专业"的权重分配
4. 业界标准：即使不跳过选股，也应至少记录 warning 供事后分析

**推荐值**:
```yaml
prob_collapse_std_threshold: 0.015   # top-300 std < 1.5% → 记录 warning（不跳过，仅告警）
```
或保持禁用但添加日志注释说明设计决策。

---

### P1-4: CVXOPT QP 结果未检查 `status` —— 亚最优解静默接受

**位置**: `barra_constraint.py` 第 265-277 行

**问题描述**:

```python
result = solvers.qp(
    matrix(P, tc='d'), matrix(q_vec, tc='d'),
    matrix(G, tc='d'), matrix(h, tc='d'),
    matrix(A_eq, tc='d'), matrix(b_eq, tc='d'),
)
weights = np.array(result['x']).flatten()  # 直接使用，无 status 检查
```

CVXOPT 返回的 `result['status']` 可能为：
- `'optimal'` — 最优解 ✓
- `'unknown'` — 未确定（通常迭代次数不足）
- `'primal infeasible'` — 原问题不可行
- `'dual infeasible'` — 对偶不可行

当前代码直接使用 `result['x']` 而不管 status，可能在以下场景返回垃圾权重：
- 迭代次数不足（`maxiters=200` 对 N>50 可能不够）
- 数值精度问题导致 status=unknown

**修复方案**:
```python
result = solvers.qp(...)
status = result['status']
if status != 'optimal':
    logger.warning(f"[Barra] QP solver status={status}, solution may be suboptimal")
    if status in ('primal infeasible', 'dual_infeasible', 'unknown'):
        return None  # 触发 softmax 降级
weights = np.array(result['x']).flatten()
```

同时建议将 `maxiters` 从 200 提升到 500 以应对 50-stock 组合。

---

### P1-5: EWMA 协方差矩阵 NaN 处理引入偏差

**位置**: `barra_constraint.py` 第 93-97 行

**问题描述**:

```python
# 缺失收益率填充策略
rets_filled = np.where(np.isnan(return_matrix), 0.0, return_matrix)  # 用 0 填充 NaN
rets_dm = rets_filled - np.nanmean(rets_filled, axis=0)              # 去均值
```

将缺失日收益率填充为 0 存在两个问题：

1. **系统性低估波动率**: 0 是一个"低波动"值（正常日收益率标准差约 2%），填充 0 会拉低该股票的方差估计 → 协方差距阵对角线偏小 → 优化器认为该股票风险更低 → 分配更高权重
2. **截面偏差**: 数据缺失多的股票（新股、停牌股）会比完整数据的股票获得更低的波动率估计 → **系统性偏好数据缺失的股票**

**业界替代方案**:
- **方案A（推荐）**: 用该股票的历史均值收益率填充（而非0）
- **方案B**: 用截面均值填充（当天所有股票的平均收益）
- **方案C**: 在协方差计算中直接排除 NaN 日期（减少 T 但无偏）

---

## 五、P2 发现（中等）

### P2-1: `max_position_pct` 默认回退值与配置不一致

**位置**: `gpsef_strategy_v10.py` 第 316 行

```python
self.max_position_pct = config['execution'].get('max_position_pct', 0.05)
```

默认回退值为 0.05（5%），但 `config_v10.yaml` 中明确设置为 0.10（10%）。

由于 yaml 中确实有这个键，当前运行时取到的是正确的 0.10。但如果有人删除了 yaml 中的该行，行为会**静默变为 5% 上限**，且没有任何日志提示。

**建议**: 将默认值改为 0.10 与 yaml 保持一致，或在取到默认值时打印 warning。

---

### P2-2: Barra 约束模块整体处于休眠态

**位置**: `config_v10.yaml` 第 340 行

```yaml
barra_constraint:
  enabled: false   # 总开关关闭
```

当前花费大量精力构建的 Barra CNE5 五因子约束体系（barra_builder.py 701行 + barra_constraint.py 467行 + build_barra_snapshots.py 188行 = 1356行）在生产配置中被关闭。

这不是 bug，但值得注意：
- 如果计划长期不使用，应考虑是否需要维护这些代码
- 如果计划未来启用，建议设置明确的启用里程碑和回归测试

---

### P2-3: `config.get('xgb_n_estimators', 500)` 直接访问原始 config

**位置**: `gpsef_strategy_v10.py` 第 1525 行

```python
num_boost_round=self.config.get('xgb_n_estimators', 500),
```

`_train_cv_ensemble` 方法直接从 `self.config`（原始 YAML 字典）读取 `xgb_n_estimators`，而不是使用已经解析过的实例变量。这与代码其他部分的风格不一致（其他地方都用 `self.xgb_params` 或专门的实例属性）。

虽然功能上正确（config 字典包含该键），但若将来重构 config 解析逻辑，此处容易被遗漏。

---

### P2-4: `eval_metric_list` YAML 格式依赖裸字符串解析

**位置**: `config_v10.yaml` 第 146 行

```yaml
diversity:
  eval_metric_list: [auc, auc, logloss, logloss, auc]
```

YAML 中 `auc` 和 `logloss` 不带引号，被解析为 YAML bare string（即 Python 中的 `'auc'` 和 `'logloss'`）。这恰好能工作是因为 XGBoost 的 `eval_metric` 参数接受字符串。

但这种写法脆弱——如果将来有人给 auc 加引号变成 `"auc"` 则不变，但如果有某个值恰好匹配 YAML boolean/integer（如 `yes/no/on/off/0/1`）就会出问题。建议统一加引号：

```yaml
eval_metric_list: ["auc", "auc", "logloss", "logloss", "auc"]
```

---

## 六、P3 发现（轻微）

### P3-1: Barra 内部 softmax 温度硬编码

**位置**: `barra_constraint.py` 第 235 行

```python
exp_a = np.exp(np.clip(alpha - alpha.max(), -10, 0) / 0.3)  # 温度硬编码 0.3
```

策略配置了可调节的 `pred_weight_temperature`（默认也是 0.3），但 Barra 模块的内部 `softmax_weights()` 函数没有接收温度参数。当前碰巧值相同，但若调整策略侧的温度，Barra 降级路径不会跟随变化。

---

### P3-2: `_get_latest_financials` 全量扫描性能

**位置**: `barra_builder.py` 第 87-104 行

每次构建快照都会扫描 `data/performance_reports/` 目录下所有 .parquet 文件。如果财报文件数量增长（多年累积），这会成为瓶颈。

建议添加缓存机制或直接传入已知日期范围的文件列表。

---

### P3-3: `conftest.py` 测试夹具 XGBoost depth=3 与生产 depth=6 不一致

**位置**: `tests/conftest.py` 第 128 行

测试中的 `make_strategy()` 创建的策略使用 `max_depth: 3`，而生产配置为 `max_depth: 6`。这导致：

- 测试中的模型容量远小于生产模型
- 特征选择/MDA/概率分布的测试结果不能代表生产行为
- 特别是 `test_v10_prob_collapse.py` 测试概率坍缩检测，用 depth=3 更容易触发坍缩，但生产环境用 depth=6 可能不触发

**建议**: 至少在关键测试（概率坍缩、特征选择、集成预测）中使用与生产一致的参数。

---

## 七、做得好的部分（符合/超越业界最佳实践）

### ✅ 7.1 防泄露设计（优秀）

| 项目 | 实现 | 评级 |
|------|------|------|
| Purged K-Fold CV | K=5, embargo=21d (> pred_days=20) | ★★★★★ |
| T-1 快照规则 | 预测用 snapshot_date=T-1 数据 | ★★★★★ |
| 标签封闭性 | eligible_train_dates 过滤 | ★★★★★ |
| 财报防泄露 | `_pub_date ≤ constraint_date` | ★★★★☆ |
| 中性化 fit/transform 分离 | OLS betas 不在预测期 refit | ★★★★★ |
| 时序验证集划分 | 80/20 按 snapshot_date 切分 | ★★★★☆ |

### ✅ 7.2 确定性保证（优秀）

| 项目 | 实现 |
|------|------|
| BLAS 单线程 | OMP/OpenBLAS/MKL/NUMEXPR/BLIS 全设为 1 |
| Python hash seed | PYTHONHASHSEED=0 |
| numpy seed | 每次训练前 `np.random.seed(42)` |
| MDA RNG | `RandomState(42)` 固定置换顺序 |
| XGBoost nthread=1 | 特征选择和 CV 折内模型单线程 |
| tree_method=hist | 显式指定避免 GPU 非确定性 |

### ✅ 7.3 特征选择管线（优秀）

完整的 6 步管线：Variance → Sanitize(NaN+corr) → PSI(0.15) → IC(0.001) → MDI → MDA t-test(n=20)，加上跨折共识投票（≥3/5折）。对齐 Alpha191 标准，且每步都有合理的 fallback 机制。

### ✅ 7.4 CV 折集成设计（优秀）

- 每折独立的 train/val 分割
- 超参数多样性（depth/subsample/colsample/eval_metric 四维度）
- AUC 加权投票（排除弱模型 min_auc=0.50）
- 向后兼容种子集成降级

### ✅ 7.5 Barra 因子暴露构建（良好）

- 本地精确前复权（merge_asof 对齐稀疏除权因子）
- 财报公告日期防泄露过滤
- 截面 Z-score + Winsorize 1%/99%
- on-the-fly 即时补充缺失股票

### ✅ 7.6 双目标软优化（良好）

`min λ_dev||B^Tw-B_bench||² + λ_l2||w-w_softmax||²` 的设计避免了 SLSQP 硬约束的可行域过窄问题，业界常用的处理方式。

---

## 八、配置参数一致性检查

| 参数 | config_v10.yaml | 代码默认值 | conftest.py | 一致? |
|------|-----------------|-----------|-------------|-------|
| xgb_max_depth | **6** | 4 | 3 | ❌ 三者不同 |
| xgb_min_child_weight | **5** | 10 | 5 | ❌ |
| xgb_n_estimators | **1000** | 500 | N/A | ❌ |
| xgb_early_stopping_rounds | **40** | 20 | 10 | ❌ |
| calibration_method | **"none"** | "isotonic" | "isotonic" | ❌ |
| auc_gate_min | **0.48** | 0.50 | 0.50 | ❌ |
| auc_gate_full | **0.52** | 0.55 | 0.55 | ❌ |
| xgb_class_balance | **"none"** | "auto" | "auto" | ❌ |
| prob_collapse_std_threshold | **0.000** | 0.003 | N/A | ❌ |
| max_position_pct (exec) | **0.10** | 0.05 | 0.10 | ⚠️ |

**关键观察**: config_v10.yaml 已经应用了上一轮审核的大部分优化建议（depth 3→6, child_weight 15→5, n_estimators 500→1000, early_stopping 20→40, calibration isotonic→none），但**代码默认值和测试夹具没有同步更新**。这意味着如果任何 config 键名拼写错误或丢失，代码会回退到旧的、已知有问题的默认值。

---

## 九、优化建议优先级排序

### 立即执行（本次回测前必做）

| # | 操作 | 工作量 | 影响 |
|---|------|--------|------|
| 1 | 修复 P0-1: SHAP 覆写 models bug（保存/恢复） | 15min | 防止 SHAP 启用时静默降级 |
| 2 | 修复 P0-2: `x0` 未定义 → 均等权重降级 | 5min | 防止边界崩溃 |
| 3 | 修复 P1-1: CV 集成尊重 class_balance 配置 | 5min | 消除配置-行为不一致 |
| 4 | 修复 P1-2: conftest.py AUC 门控阈值对齐 | 5min | 测试可信度 |
| 5 | 修复 P1-4: CVXOPT status 检查 | 10min | 防止亚最优解 |

### 短期跟进（本周内）

| # | 操作 | 工作量 | 影响 |
|---|------|--------|------|
| 6 | 同步代码默认值为 config 当前值（depth/child_weight/n_estimators等） | 30min | 防止 config 键丢失时的静默退化 |
| 7 | prob_collapse_threshold 设为 0.015（warning 级别） | 1min | 运行时可观测性 |
| 8 | CVXOPT maxiters 200→500 | 1min | 大组合求解稳定性 |

### 中期改进（下一迭代）

| # | 操作 | 工作量 | 影响 |
|---|------|--------|------|
| 9 | P1-5: EWMA 协方差 NaN 填充策略改为均值填充 | 2h | 消除系统性偏向缺失数据股票 |
| 10 | P3: conftest.py 关键测试使用生产参数 | 1h | 测试代表性 |
| 11 | Barra 启用回归测试（enabled=true 跑一轮） | 半天 | 验证 1356 行代码可用性 |

---

## 十、架构评价

### 整体架构评分: B+ (82/100)

| 维度 | 得分 | 说明 |
|------|------|------|
| 防泄露设计 | 95/100 | 业界标杆水平，embargo > pred_days 尤其出色 |
| 模型训练 | 85/100 | CV折集成 + 超参数多样性优秀，但 SHAP 覆写 bug 扣分 |
| 特征选择 | 90/100 | 6 步管线 + t检验 + 共识投票，完整且稳健 |
| 组合优化 | 75/100 | 双目标软优化设计合理，但 QP 健壮性不足（status检查/NaN/x0） |
| 代码质量 | 78/100 | 3098行单文件过长，DRY违反明显，但核心逻辑清晰 |
| 测试覆盖 | 70/100 | 21个测试文件覆盖主要模块，但夹具参数过时 |
| 配置管理 | 65/100 | config/代码/test 三处不一致，维护负担大 |

### 与业界最佳实践的 Gap 分析

| 最佳实践 | V10 状态 | Gap |
|----------|---------|-----|
| Purged K-Fold + embargo | K=5, embargo=21 | ✅ 无 gap |
| 模型集成多样性 | CV折 × 4维超参 | ✅ 无 gap |
| 概率校准 | 已移除(isotonic→none) | ⚠️ 有意为之，可接受 |
| 位置约束/换手控制 | 无 | ❌ 完全缺失 |
| 交易成本模型（显式） | 仅隐含在框架层 | ⚠️ 可接受 |
| 组合层面风险预算 | Barra(休眠) + IV(休眠) | ⚠️ 设计了但未启用 |
| 在线学习/增量训练 | 无（每期全量重训） | ⚠️ 对月频调仓可接受 |
| 特征监控/漂移检测 | SHAP(休眠) + PSI | ⚠️ PSI 在特征选择中有，但无独立监控 |

---

## 十一、结论

Wyckoff V10 策略的核心机器学习管线（防泄露、CV折集成、特征选择、中性化）达到了**机构级量化策略的标准**，特别是在防泄露设计和确定性保证方面可以称为**业界标杆**。

主要风险集中在**边缘情况的鲁棒性**（QP求解失败/NaN权重/SHAP覆写）和**配置一致性维护**（三处不同步）。这些问题不会影响正常市场环境下的回测结果，但在极端行情或配置变更时可能导致意外行为。

**建议**: 修复 2 个 P0 + 3 个高置信 P1 后即可放心跑完整回测。Barra 约束模块建议在主回测稳定后再启用（单独做一次 A/B 对比测试）。

---

*本报告基于 2026-07-11 版本的全部源码静态分析生成，不构成投资建议。*
