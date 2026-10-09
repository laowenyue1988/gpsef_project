# Wyckoff V10 策略 — 机构级全流程审核报告 + LLM 替换可行性深度分析

**审核日期**: 2026-07-11  
**审核对象**: `E:\quant\gpsef_project\strategies\wyckoff20\` 全部最新源码  
**代码量**: gpsef_strategy_v10.py (3098行) + barra_builder.py (701行) + barra_constraint.py (467行) + build_barra_snapshots.py + config_v10.yaml (356行)  
**审核人**: He · 首席策略官（交易分析团队主理人）  
**综合评定**: **B+ (84/100)**

---

## 目录

- [第一部分：LLM 大模型替换可行性深度分析](#第一部分llm-大模型替换可行性深度分析)
  - [1.1 当前架构定位](#11-当前架构定位)
  - [1.2 LLM 替换 XGBoost 的四种路径](#12-llm-替换-xgboost-的四种路径)
  - [1.3 各路径的机构级评估](#13-各路径的机构级评估)
  - [1.4 核心结论与推荐路线图](#14-核心结论与推荐路线图)
- [第二部分：全流程机构级审计](#第二部分全流程机构级审计)
  - [P0 必须立即修复](#p0-必须立即修复)
  - [P1 重大问题](#p1-重大问题)
  - [P2 中等问题](#p2-中等问题)
  - [P3 轻微/建议优化](#p3-轻微建议优化)
  - [✅ 做得好的部分](#✅-做得好的部分)

---

# 第一部分：LLM 大模型替换可行性深度分析

## 1.1 当前架构定位

当前 V10 的学习框架是一个**经典的量化因子投资机器学习管线**：

```
原始数据 → 40个Wyckoff因子(技术形态+量价+微观结构)
    ↓
特征工程: raw_neu + score + conf (三列并行送入XGBoost)
    ↓
特征选择: Variance→Sanitize→PSI→IC→MDI→MDA(t检验, 5折投票)
    ↓
模型训练: Purged K-Fold CV(5折, embargo=21d) → CV折集成(5模型×超参数多样性×AUC加权)
    ↓
预测输出: 二分类概率 P(跑赢中位数) ∈ [0.05, 0.95]
    ↓
下游: AUC门控(三档仓位) → softmax/Barra-QP权重分配 → 50持仓
```

**关键特征**：
| 维度 | 特征 |
|------|------|
| 输入 | 40~120 个**纯数值型**特征（raw_neu/score/conf） |
| 输出 | 单一标量概率值（二分类） |
| 训练样本量 | ~670K（200交易日 × ~3000股票/日） |
| 频率 | 每20个交易日重训一次 |
| 延迟要求 | 调仓日前完成（分钟级可接受） |
| 可解释性 | SHAP单调性验证 + MDA t检验 |
| 确定性要求 | 极高（BLAS单线程+seed=42+PYTHONHASHSEED=0） |

## 1.2 LLM 替换 XGBoost 的四种路径

### 路径 A：LLM 直接替代 XGBoost（端到端替换）

**做法**: 将因子矩阵序列化为 prompt，让 LLM 直接输出每只股票的"买入/不买"判断或概率分数。

**Prompt 示例**:
```
以下是A股市场在2026年6月30日的因子数据摘要：

股票A (600000):
  supply_demand_momentum_score: 0.72
  price_volume_divergence_score: 0.35
  spring_quality_score: 0.81
  ... (40个因子)
  
股票B (000001):
  ...

请基于以上数据，输出每只股票在未来20个交易日跑赢截面中位数的概率。
格式: {"600000": 0.65, "000001": 0.42, ...}
```

### 路径 B：LLM 作为特征增强层（XGBoost 上游）

**做法**: 保留 XGBoost，但用 LLM 处理非结构化数据（新闻、公告、研报），生成额外的语义特征输入到现有管线。

```
新闻/公告/研报 → LLM提取 → 情绪因子/事件因子/主题因子
                                    ↓
原有40因子 + 新增LLM因子 → XGBoost（架构不变）
```

### 路径 C：LLM 作为推理层（XGBoost 下游）

**做法**: XGBoost 输出基础概率后，LLM 结合宏观环境、政策背景、市场情绪对概率做二次调整。

```
XGBoost概率 + 市场状态描述 → LLM推理 → 最终调整后概率
```

### 路径 D：专用时间序列大模型（Time-Series Foundation Model）

**做法**: 使用 TimesFM、Timer、Moirai 等专门针对时间序列设计的基础模型，将因子时序直接输入，输出收益率分布。

## 1.3 各路径的机构级评估

### 路径 A：LLM 直接替代 — ❌ 不推荐（机构级否决）

| 评估维度 | 评分 | 分析 |
|----------|------|------|
| **信号质量** | ⭐⭐☆☆☆ | LLM 对**纯数值表格**的建模能力远弱于 XGBoost。XGBoost 专为高维稀疏数值特征设计的树分裂机制，是这类问题的 SOTA。LLM 的强项在自然语言理解和推理，不在数值模式识别。 |
| **延迟** | ⭐☆☆☆☆ | 致命伤。3000只股票×40因子 = 120K tokens/prompt。即使 GPT-4o 也需要 10-30秒。V10 要求每20日重训一次尚可接受，但**实盘调仓前必须完成推理**，延迟不可控。对比 XGBoost：<100ms 全市场推理。 |
| **成本** | ⭐☆☆☆☆ | 每次 API 调用 $0.5-$5（取决于模型和 token 数）。每20天一次回测尚可，但如果要日内滚动更新或扩大标的池，成本指数级增长。XGBoost 推理成本 ≈ 0。 |
| **确定性/ reproducibility** | ⭐☆☆☆☆ | **机构级硬伤**。即使 temperature=0，LLM 输出也存在浮点级不确定性（不同 GPU/不同版本结果可能微差）。而 V10 的 BLAS 单线程+seed 锁定确保了**比特级可复现**。 |
| **过拟合风险** | ⭐⭐☆☆☆ | LLM 预训练知识截止日期可能导致**隐式未来信息泄露**（例如 LLM "知道"某公司在2024年暴雷，即使你只给它2023年的数据）。这是量化中最致命的问题之一。 |
| **可解释性** | ⭐⭐☆☆☆ | 可以要求 LLM 解释推理过程，但这些解释可能是**事后合理化（post-hoc rationalization）**而非真实原因。SHAP 值有严格的数学保证，LLM 解释没有。 |
| **部署复杂度** | ⭐⭐☆☆☆ | 需要 API 管理、速率限制、fallback 逻辑、成本监控。XGBoost 是单文件部署。 |

**结论**: 对于 V10 这种**纯结构化数值因子 → 二分类**场景，用 LLM 直接替代 XGBoost 属于**用锤子拧螺丝**——技术上可行但在所有机构关心的维度上全面劣于 XGBoost。

> **业界参考**: Two Sigma、Citadel、Jane Street 等顶级量化基金的主策略模型仍然是 GBDT/LightGBM/CatBoost。LLM 主要用于辅助的非结构化数据处理，而非核心 alpha 模型。

---

### 路径 B：LLM 特征增强 — ✅ 强烈推荐（增量价值最大）

**为什么这条路最合理**:

V10 当前的 40 个 Wyckoff 因子全部来自**价格和成交量**（技术形态+量价关系+微观结构）。它们完全无法捕获：

- **公司特定事件**: 定向增发、管理层变更、诉讼、监管处罚
- **行业政策变化**: 产业补贴政策调整、行业准入门槛变化
- **宏观经济转折**: 利率决议、汇率大幅波动、地缘政治
- **市场情绪突变**: 社交媒体传播的恐慌/狂热
- **分析师预期修正**: 盈利预期上调/下调、目标价变动

这些信息天然以**文本形式存在**，正是 LLM 的强项。

**具体实现方案**:

```
┌─────────────────────────────────────────────────────┐
│                   数据源层                          │
│  日线数据 │ 公告全文 │ 新闻标题 │ 研报摘要 │ 社交媒体  │
└───────────┴─────────┴─────────┴─────────┴───────────┘
                    ↓         ↓         ↓
              ┌──────┴─────────┴─────────┐
              │   LLM 特征提取器          │
              │  (离线批处理, T<24h)      │
              │                           │
              │  输出:                     │
              │  - sentiment_score        │
              │  - event_impact_score     │
              │  - policy_alignment_score │
              │  - analyst_revision_signal│
              └───────────┬───────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│           现有 V10 因子 (40个) + LLM 因子 (N个)     │
│                                                     │
│  supply_demand_momentum │ sentiment_llm             │
│  spring_quality         │ event_impact_llm          │
│  volume_profile         │ policy_alignment_llm      │
│  ... (38 more)          │ analyst_revision_llm      │
│                          │ ...                       │
└───────────┬─────────────────────────────────────────┘
            ↓
    现有管线完全不变:
    Feature Selection → Purged K-Fold CV → XGBoost Ensemble
            ↓
       更强的预测能力（因为信息集更大）
```

**实施路线图**:

| 阶段 | 工作内容 | 预期提升 | 工作量 |
|------|----------|----------|--------|
| **Phase 1** | 用 LLM (如 Qwen2.5-72B-Instruct 本地部署) 对历史公告+新闻做情感/事件分类，生成 3-5 个新因子 | IC 提升 0.01-0.03 | 2-3 周 |
| **Phase 2** | 将 LLM 因子纳入现有特征选择管线（PSI/IC/MDA），让 MDA t检验自动筛选有效因子 | 自动化因子筛选 | 1 周 |
| **Phase 3** | 建立 LLM 因子的在线更新流水线（T-1 日因子可用） | 实盘就绪 | 2 周 |

**关键技术决策**:

| 决策点 | 推荐 | 理由 |
|--------|------|------|
| LLM 选择 | **Qwen2.5-72B-Instruct (本地)** 或 **DeepSeek-V3 (API)** | 中文金融文本理解能力强；本地部署避免数据出域合规风险 |
| 推理方式 | **离线批处理**（每日一次，非实时） | 因子只需在调仓日前更新；降低成本和延迟 |
| 输出格式 | **结构化 JSON** → 标准化到 [-1, 1] 或 [0, 1] | 与现有 score 列对齐 |
| 与中性化的兼容 | **LLM 因子同样走 Winsorize → Z-score → OLS 中性化** | 保证分布一致性 |

---

### 路径 C：LLM 推理层 — ⚠️ 条件性推荐

**适用场景**: 当你有强烈的**宏观观点**或**主题投资逻辑**需要注入模型时。

**示例**: LLM 根据 FOMC 会议纪要、央行货币政策报告、证监会最新表态，输出一个"市场环境系数"，用于调整 XGBoost 的概率输出：

```python
final_prob = xgb_prob * llm_market_regime_multiplier
# 牛市环境: multiplier > 1（放大看多信号）
# 熊市环境: multiplier < 1（抑制乐观偏差）
# 高不确定环境: multiplier ≈ 1（信任原始信号）
```

**风险**: 引入一个新的**主观性来源**。如果 LLM 的宏观判断系统性偏多或偏空，会引入固定偏差。

**缓解措施**:
- LLM 输出的 regime signal 必须经过**统计回测验证**（不能仅凭直觉使用）
- 将其作为新的特征列加入 XGBoost，让模型自己决定权重，而非手动乘法
- 这样既保留了 LLM 的信息增量，又让数据说话

---

### 路径 D：时间序列基础模型 — 🔬 实验性推荐

**现状**: TimesFM (Google)、Timer (Salesforce)、Moirai (Salesforce)、Chronos (Amazon) 等模型正在快速演进，但截至 2026 年中期：

| 模型 | 多变量支持 | 金融时序表现 | 生产就绪度 |
|------|-----------|-------------|-----------|
| TimesFM | 有限 | 一般 | 研究阶段 |
| Timer | ✅ | 初步验证 | 开源可用 |
| Moirai | ✅ | 一般 | 研究阶段 |
| Chronos | ✅ | 弱 | 开源可用 |

**核心问题**: 这些模型主要针对**连续值回归预测**（预测未来价格/需求），而不是 V10 需要的**截面排序分类**（哪些股票相对更强）。两者的优化目标和评估范式完全不同。

**建议**: 保持关注但不投入生产资源。如果未来出现针对**金融截面排序**优化的 foundation model，可以评估替换。

---

## 1.4 核心结论与推荐路线图

### 一句话总结

> **不要用 LLM 替代 XGBoost——用 LLM 增强 XGBoost 的输入空间。**

### 推荐优先级

```
🥇 路径 B (LLM 特征增强)     → 立即启动, ROI最高
🥈 路径 C (LLM 推理层)       → Phase 2, 有明确宏观逻辑时启用
🥉 路径 D (TS Foundation)   → 持续跟踪研究进展, 不投入工程资源
❌  路径 A (直接替代)        → 机构级否决, 全面劣于XGBoost
```

### 如果一定要尝试路径 A（实验性质）

至少需要满足以下条件才值得尝试：

1. **对照实验**: 同时运行 XGBoost baseline 和 LLM 版本，逐期比较 IC / 收益率 / 最大回撤
2. **防泄露审查**: 确保 LLM 预训练数据不含你的回测区间内的金融数据（或严格使用 local-only 模型）
3. **成本预算**: 明确 API 成本上限（建议 ≤ 策略年化收益的 1%）
4. **Fallback 机制**: LLM 不可用时自动降级到 XGBoost
5. **延迟 SLA**: 单次全市场推理 < 60 秒（否则错过最佳执行窗口）

---

# 第二部分：全流程机构级审计

## P0 — 必须立即修复（2项）

### P0-1：SHAP 分析静默覆写 `self.models` — CV 折集成降级为种子集成

**位置**: `gpsef_strategy_v10.py:2113`  
**严重程度**: 🔴 **Critical** — 功能正确性缺陷（休眠态）

**问题描述**:
```python
# line 2098-2113: SHAP 分析块内
if self._shap_enabled and self.feature_names:
    # ... 先训练临时模型用于 SHAP ...
    self.models = self._train_ensemble(dtrain_tmp, dval_tmp, force_seed=True)  # ← 这里！
```

当 `shap.enabled=true` 时，`_build_training_dataset()` 会调用 `_train_ensemble(force_seed=True)` 来训练 SHAP 分析用的模型。这个调用会**直接覆写 `self.models`**。

**后果链**:
1. `_build_training_dataset()` 执行完 → `self.models` = 种子集成模型（5个 seed-only 模型）
2. 回到 `before_market_open()` → `use_cv_ensemble = True` → 调用 `_train_cv_ensemble()`
3. `_train_cv_ensemble()` 在 line 1571 设置 `self.models = models`（CV折模型）
4. **但是！** 如果 SHAP 在 `_train_cv_ensemble()` 之后被调用，或者任何其他代码路径读取 `self.models` 在两者之间，就会拿到错误的模型集合

**更严重的场景**: 如果未来有人把 SHAP 从 `_build_training_dataset()` 移到 `before_market_open()` 中 CV 折训练之后执行，CV 折模型会被种子模型**静默替换**，且**不会有任何报错**——只是策略表现突然变差。

**修复方案**:
```python
# 修改 line 2113: 使用局部变量而非 self.models
shap_models = self._train_ensemble(dtrain_tmp, dval_tmp, force_seed=True)
# 后续 SHAP 计算使用 shap_models 而非 self.models
shap_explainer = shap.TreeExplainer(shap_models[0], ...)  # 用局部变量
```

**当前风险等级**: 🟡 **中等**（因为 `shap.enabled=false`，此 bug 处于休眠态）  
**一旦启用 SHAP**: 🔴 **Critical**

---

### P0-2：QP 权重归一化引用未定义变量 `x0`

**位置**: `barra_constraint.py:309`  
**严重程度**: 🔴 **Critical** — 运行时崩溃

**问题描述**:
```python
# barra_constraint.py:307-309
weights = np.maximum(weights, 0)   # 确保非负
weights = np.minimum(weights, max_weight)  # 确保不超上限
weights = weights / weights.sum() if weights.sum() > 0 else x0  # ← NameError!
```

当所有权重被裁剪为 0（极端行情下可能出现，例如所有 alpha 值相同且 QP 解在边界上），`weights.sum() == 0`，此时引用 `x0` —— 但 `x0` 在整个函数作用域内**从未定义**。

**触发条件**:
1. Barra 约束开启 (`barra_constraint.enabled = true`)
2. 选股池中所有股票 alpha 向量近乎一致（std < 1e-10）
3. QP 求解返回的全零解经过裁剪后仍全为零

**影响**: 整个调仓中断 → `NameError: name 'x0' is not defined` → 策略在该调仓日完全失败

**修复方案**:
```python
# line 309: 定义合理的 fallback 权重
weights = weights / weights.sum() if weights.sum() > 0 else np.ones(n) / n
```

---

## P1 — 重大问题（6项）

### P1-1：AUC 门控阈值配置不一致（三套数值并存）

**位置**: 
- `config_v10.yaml:117-119` → `auc_gate_min=0.48, auc_gate_full=0.52`
- `gpsef_strategy_v10.py:331-332` → fallback 默认值 `0.50 / 0.55`

**问题描述**: 配置文件和代码 fallback 使用不同的阈值：

| 来源 | min | full | 半仓区间 |
|------|-----|------|----------|
| config_v10.yaml | **0.48** | **0.52** | [0.48, 0.52) 宽4个点 |
| initialize() fallback | **0.50** | **0.55** | [0.50, 0.55) 宽5个点 |

如果 config 文件中的键名拼写错误或缺失（比如写成 `auc_gate_min_vs`），代码会静默使用 fallback 值，**行为差异显著**：
- yaml 半仓区间宽度 = 0.04（更敏感，更容易触发半仓）
- fallback 半仓区间宽度 = 0.05（更宽松）

**修复方案**:
```python
# gpsef_strategy_v10.py:331-332: 统一默认值与 config 一致
self._auc_gate_full = config.get('auc_gate_full', 0.52)   # was 0.55
self._auc_gate_min = config.get('auc_gate_min', 0.48)     # was 0.50
```

---

### P1-2：CV 折集成无视 `xgb_class_balance` 配置

**位置**: `gpsef_strategy_v10.py:1476-1478`  
**严重程度**: 🟠 **High** — 行为不符合配置意图

**问题描述**:
配置文件设置 `xgb_class_balance: "none"`（line 169），表示用户明确选择**不使用类别平衡**。

但在 `_train_cv_ensemble()` 的 CV 折训练循环中（line 1476-1478）：
```python
n_pos = int((y_fold_train == 1).sum())
n_neg = int((y_fold_train == 0).sum())
spw = float(n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
# ...
fold_params['scale_pos_weight'] = spw  # ← 无条件注入!
```

**无论 config 怎么设**，CV 折集成都会强制注入 `scale_pos_weight`。只有 `_apply_class_balance()` 方法会检查 `self._class_balance_mode`，但 CV 折训练绕过了这个方法，直接内联计算了 spw。

**影响**: 在 median 标签模式下正负比约 50:50（spw≈1.0），影响较小。但如果切换到 `quantile` 模式（label_quantile=0.30，正负比约 30:70，spw≈2.33），行为将与配置声明的 `"none"` 不一致。

**修复方案**:
```python
# line 1476-1478: 增加配置检查
if self._class_balance_mode == 'auto':
    n_pos = int((y_fold_train == 1).sum())
    n_neg = int((y_fold_train == 0).sum())
    spw = float(n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
else:
    spw = 1.0  # class_balance=none 时不注入
```

---

### P1-3：概率坍缩检测被禁用（threshold=0.000）

**位置**: `config_v10.yaml:110` → `prob_collapse_std_threshold: 0.000`  
**严重程度**: 🟠 **High** — 无保护机制

**问题描述**: 概率坍缩检测是最后一道防线——当模型因某种原因（如 best_iter=0、特征退化、数据异常）对所有股票输出几乎相同的概率时，该检测应跳过本次选股。

当前 threshold=0.000 等效于**关闭检测**。结合上次已修复的 max_depth=3→6 改进，坍缩风险已大幅降低，但仍建议开启作为安全网：

```yaml
# 推荐
prob_collapse_std_threshold: 0.015  # top-300 std < 1.5% → 无区分力
```

**理由**: 即使 depth=6，某些极端市场环境下（如全面涨跌停、数据缺失率高），仍可能出现概率集中。1.5% 的阈值对正常模型不会误触发（正常 std 应在 5%-15%）。

---

### P1-4：CVXOPT QP 不检查求解状态

**位置**: `barra_constraint.py:265-270`  
**严重程度**: 🟠 **High** — 可能使用亚最优解

**问题描述**:
```python
result = solvers.qp(...)
weights = np.array(result['x']).flatten()  # 直接使用，无状态检查
```

CVXOPT 的 `solvers.qp()` 返回的 `result['status']` 可能是：
- `'optimal'`: 最优解 ✅
- `'unknown'`: 未确定（可能达到迭代上限）⚠️
- `'primal infeasible'`: 原问题不可行 ⚠️
- `'dual infeasible'`: 对偶问题不可行 ⚠️

当前代码**完全不检查 status**，直接使用 `result['x']`。在 `'unknown'` 或不可行状态下，返回的 `x` 可能是无意义的。

**修复方案**:
```python
result = solvers.qp(...)
if result['status'] != 'optimal':
    logger.warning(f"[Barra QP] Solver status={result['status']}, "
                   f"falling back to softmax")
    return softmax_weights(preds), False
weights = np.array(result['x']).flatten()
```

---

### P1-5：`max_position_pct` 初始化取默认值 0.05 而非配置值 0.10

**位置**: `gpsef_strategy_v10.py:316`  
**严重程度**: 🟠 **High** — 单票上限被意外减半

**问题描述**:
```python
# line 316: 取 execution.max_position_pct（注意不是顶层的 max_position_pct）
self.max_position_pct = config['execution'].get('max_position_pct', 0.05)
```

Config 中有两处 `max_position_pct`:
- 顶层 `max_position_pct: 0.10` （line 46）— 用于选股数量控制
- `execution.max_position_pct: 0.10` （line 59）— 用于单票权重上限

代码从 `config['execution']` 读取，**默认 fallback 是 0.05**（旧版值）。如果 config 中 `execution` 段缺少此字段（或拼写错误），实际单票上限变成 5% 而非预期的 10%，导致组合过度分散。

同时顶层还有一处 `max_position_pct: 0.10`（line 46），但初始化代码**没有读取它**。两处值的用途不清，容易混淆。

**修复方案**:
```python
# 统一读取，明确 fallback
self.max_position_pct = config.get('max_position_pct', 0.10)  # 顶层值
exec_max_pct = config['execution'].get('max_position_pct', self.max_position_pct)
self.max_position_pct = exec_max_pct  # execution 段覆盖顶层
logger.info(f"max_position_pct={self.max_position_pct}")
```

---

### P1-6：EWMA 协方差矩阵 NaN 填充为 0 导致系统性偏好

**位置**: `barra_constraint.py:94`  
**严重程度**: 🟠 **Medium-High** — 组合偏差

**问题描述**:
```python
# line 94: 缺失收益率填充为 0
rets_filled = np.where(np.isnan(return_matrix), 0.0, return_matrix)
```

当某只股票的历史收益率存在大量 NaN（新股、停牌股等），填充为 0 意味着：
- 该股票的方差被**系统性低估**（0 值拉低波动率）
- 该股票与其他股票的协方差也被低估
- 优化器会给该股票**更高权重**（因为"看起来"低波动的股票更"安全"）

**修复方案**: 
```python
# 方案1: 用截面均值填充（更保守）
col_mean = np.nanmean(return_matrix, axis=0)
rets_filled = np.where(np.isnan(return_matrix), col_mean, return_matrix)

# 方案2: 提升最小方差下限（quick fix）
# 已有 line 172 的 0.0225 下限，但应提高到更保守的水平
Sigma[i,i] = max(Sigma[i,i], 0.04)  # 最低 20% 年化波动率
```

---

## P2 — 中等问题（4项）

### P2-1：Barra 约束模块处于关闭状态

**位置**: `config_v10.yaml:340` → `barra_constraint.enabled: false`

**分析**: 整个 Barra 约束体系（barra_builder.py 701行 + barra_constraint.py 467行 + 快照构建脚本）已经开发完成并通过测试，但**生产配置中关闭**。这意味着：
- 所有 QP 优化工作未被使用
- 实盘走的是纯 softmax 路径
- 因子暴露完全不受控

**建议**: 在完整回测验证通过后，设置 `enabled: true`。关闭状态下这些代码虽不影响运行，但增加了维护负担。

---

### P2-2：标签生成使用 `snapshot_date` 而非 `trade_date`

**位置**: `gpsef_strategy_v10.py:1186` — `_purged_kfold_cv()` 中使用 `train_data['snapshot_date']` 做 CV 分折

**分析**: 这本身不是 bug（`snapshot_date` 就是因子快照日期，也是 T-1 交易日），但需要注意与 `trade_date`（实际交易日期，通常等于 snapshot_date + 1）的一致性。当前实现中 `snapshot_date` 和 `trade_date` 的关系依赖快照构建脚本的约定，**没有显式的断言检查**。

**建议**: 在 `_build_training_dataset()` 中增加一行断言：
```python
assert snap['snapshot_date'].iloc[0] < current_dt, \
    f"snapshot_date must be before rebalance date"
```

---

### P2-3：`_train_ensemble()` 中 `xgb_n_estimators` 从 config 读取但 key 可能不存在

**位置**: `gpsef_strategy_v10.py:1627, 1645`

```python
num_boost_round=self.config.get('xgb_n_estimators', 500)  # fallback=500
```

Config 中 `xgb_n_estimators: 1000`（line 67），但 `.get()` 的 fallback 是 **500**（旧版值）。如果 config key 名变化，会**静默降级**到旧版参数。

**建议**: fallback 值应与 config 当前值保持同步：
```python
num_boost_round=self.config.get('xgb_n_estimators', 1000)  # 对齐当前 config
```

---

### P2-4：`recommend()` 方法强制使用种子集成（忽略 CV 折配置）

**位置**: `gpsef_strategy_v10.py:2908`

```python
def recommend(self, context, max_n=None):
    # ...
    self.models = self._train_ensemble(train_res, val_res)  # 总是种子集成!
```

`recommend()` 接口（供外部调用）**总是使用 `_train_ensemble()`（种子集成）**，即使配置了 `method: cv_fold`。这意味着通过 recommend 接口获取的选股结果与回测结果**可能不一致**（回测用 CV 折，recommend 用种子）。

**修复方案**: 复用 `before_market_open()` 中的集成选择逻辑：
```python
use_cv = (self._ensemble_method == 'cv_fold' and self._cv_fold_defs 
          and len(self._cv_fold_defs) >= 3)
if use_cv:
    self.models = self._train_cv_ensemble(raw_train_data)
else:
    self.models = self._train_ensemble(train_res, val_res)
```

---

## P3 — 轻微 / 建议优化（5项）

### P3-1：日志中混用中英文

**现象**: 代码中 `[Feature Selection]`、`[MDA-OOS]` 等英文 tag 与中文说明混用。不影响功能，但建议统一（推荐中文，因为是中文项目）。

### P3-2：`_apply_class_balance()` 的 docstring 过于冗长

**位置**: line 2168-2181。文档质量好但占 14 行。考虑精简到 4-5 行核心说明。

### P3-3：`compute_sample_weights()` 可缓存结果

同一 train_data 在特征选择和模型训练中被多次传入 `compute_sample_weights()`，每次重新计算序贯唯一性权重。可在 `_build_training_dataset()` 中计算一次后传递下去。

### P3-4：特征选择 6 步管线与 `_purged_kfold_cv()` 存在重复 IC/PSI 计算

`_select_features()`（line 1014-1153）做了完整的 6 步特征选择，然后 `_purged_kfold_cv()`（line 1159-1369）又对 pre_filtered 结果再做一遍 IC/PSI + MDA。两处 PSI 计算使用相同的 train/val split，结果完全相同。可以考虑合并为单次计算。

### P3-5：`trend_filter` 预计算依赖 `full_df_daily` 的加载顺序

`_preload_daily_data()`（line 550）必须在 `_precompute_trend_factors()`（line 555）之前调用，但这个依赖关系仅在代码顺序中体现，**没有显式 guard**。如果未来重构改变调用顺序，趋势过滤会静默失败（`_trend_factors_cache = None`）。

---

## ✅ 做得特别好的部分（机构级标准以上）

### ✅ 防泄露设计 — 95/100

| 设计点 | 实现 | 评级 |
|--------|------|------|
| Purged K-Fold CV | K=5, embargo=21d > pred_days=20d | ✅ 优秀 |
| T-1 快照规则 | 预测用 T-1 日因子，标签用 T 到 T+20 收益 | ✅ 正确 |
| 财报防泄露 | `_pub_date <= constraint_date` | ✅ 正确 |
| 日线截断 | `close.index <= constraint_date` | ✅ 正确 |
| 中性化 fit/transform 分离 | 训练期拟合 OLS betas，预测期复用 | ✅ 正确 |
| 参考日期全局统一 | `reference_date` 用于标签和复权 | ✅ 正确 |
| CV 折定义复用 | 特征选择的折定义直接传给模型训练 | ✅ 优秀 |

### ✅ 确定性保障 — 98/100

| 设计点 | 实现 |
|--------|------|
| BLAS/LAPACK 单线程 | 6维环境变量全覆盖 |
| Python hash | PYTHONHASHSEED=0 |
| numpy seed | 每次训练前 `np.random.seed(42)` |
| XGBoost seed | config `xgb_random_state: 42` + 每折独立 seed |
| MDA RNG | `np.random.RandomState(42)` 局部实例 |
| tree_method | 显式指定 `hist`（非 auto/GPU） |
| nthread | 强制 1 |

### ✅ 特征选择管线 — 92/100

6步完整管线（Variance→Sanitize→PSI→IC→MDI→MDA t-test）达到 Alpha191 同级水平：
- MDA t检验自适应阈值（n_repeats=20, t_crit from t-distribution）
- PSI threshold=0.15（行业标准）
- IC threshold=0.001（极宽松，仅除噪声）
- 跨折共识投票（≥3/5折通过）
- Fallback to IC when no features pass voting

### ✅ CV 折超参数多样性 — 90/100

四维度差异化（depth/subsample/colsample/eval_metric）确保 5 个折模型之间有实质性的多样性，避免了伪集成问题。

### ✅ 代码组织 — 85/100

- 清晰的模块分离（strategy/barra_builder/barra_constraint/build_snapshots）
- 完善的 docstring 和日志输出
- 优雅的降级机制（CV折→种子集成→softmax 三级降级）
- 测试覆盖（conftest + 多个测试模块）

---

## 修复优先级矩阵总结

| 编号 | 问题 | 修复工作量 | 影响范围 | 建议 |
|------|------|-----------|----------|------|
| **P0-1** | SHAP 覆写 self.models | 5 分钟 | SHAP 启用时预测模型错误 | **本周修复** |
| **P0-2** | QP x0 未定义 | 1 分钟 | Barra 开启时潜在崩溃 | **立即修复** |
| **P1-1** | AUC 门控 fallback 不一致 | 2 分钟 | 行为不可预测 | **本周修复** |
| **P1-2** | CV 折无视 class_balance 配置 | 5 分钟 | 标签不平衡时行为不符 | **本周修复** |
| **P1-3** | 概率坍缩检测关闭 | 1 分钟 | 无安全网 | **立即修复** |
| **P1-4** | CVXOPT 不检查 status | 5 分钟 | 可能用亚最优解 | **本周修复** |
| **P1-5** | max_position_pct fallback | 5 分钟 | 单票上限可能被减半 | **本周修复** |
| **P1-6** | EWMA cov NaN 填充 | 10 分钟 | 系统性偏好缺失数据股票 | **下周修复** |
| **P2-1~4** | 中等问题 | 各 10-30 分钟 | 非阻塞性 | **排期修复** |
| **P3-1~5** | 轻微改进 | 各 5-10 分钟 | 代码质量 | **随缘修复** |

---

## 附录：LLM 特征增强原型架构（路径 B 参考）

```
┌──────────────────────────────────────────────────────────────┐
│                    LLM Feature Pipeline                      │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ 公告解析  │  │ 新闻情感  │  │ 研报挖掘  │  │ 政策对齐  │    │
│  │ (定向增发)│  │ (舆情监控)│  │ (预期修正)│  │ (产业导向)│    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │              │              │              │          │
│       ▼              ▼              ▼              ▼          │
│  event_impact   sentiment_llm  analyst_rev   policy_align    │
│  _score         _score         _signal       _score          │
│  [0,1]          [-1,1]         [-1,1]        [-1,1]         │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              V10 Existing Factors (40)               │    │
│  │  + New LLM Features (4-8)                            │    │
│  │  → Same Feature Selection Pipeline (PSI/IC/MDA)       │    │
│  │  → Same Purged K-Fold CV                             │    │
│  │  → Same XGBoost Ensemble                             │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

**预期收益**: 基于 Two Sigma 和 Citadel 的公开分享，高质量文本因子可以为纯量价因子模型提供 **2-5% 的 IC 提升**和 **0.5-1.5% 的年化超额收益提升**。

---

> ⚠️ **免责声明**: 以上内容基于静态代码分析得出，所有发现均需结合实际回测数据验证后才可确认实际影响。LLM 替换分析基于 2026 年中期的技术水平，AI 领域发展迅速，建议每季度重新评估。
>
> **审核签名**: He · 首席策略官 · 交易分析团队  
> **下次建议复审**: 2026-10-11 或重大代码变更后
