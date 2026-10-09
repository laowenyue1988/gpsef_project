# Wyckoff V10 策略审查报告 & 优化方案

> **审查日期**: 2026-06-24
> **审查标准**: Alpha191 通用量化架构知识（已存入项目长期记忆）
> **审查范围**: `strategies/wyckoff20/gpsef_strategy_v10.py`（1084行）+ `config_v10.yaml`（201行）+ V5 父类 `gpsef_strategy.py`（继承链）
> **审查模式**: 只读审查，不修改任何文件

---

## 一、架构总览对比

| 维度 | Alpha191（标准） | Wyckoff V10（当前） | 差距评估 |
|------|------------------|---------------------|----------|
| **特征选择** | Purged K-Fold CV (5折, 21天embargo) + 跨折共识投票 + OOS MDA (t检验) | 单次 80/20 时序划分 + 顺序管线 (IC→MDI→MDA) | 🔴 重大差距 |
| **标签生成** | 三重屏障标签 (PT/SL/expiry, f(σ)) + 分钟级路径跟踪 + EWMA波动率 | 二分类（跑赢截面中位数）+ 简单20日前向收益 | 🟡 设计差异（可接受但需注意风险） |
| **模型训练** | 5种子集成 (42-46) + Early Stopping (AUC) + Isotonic概率校准 | 单模型 + Early Stopping (logloss) + 无校准 | 🔴 重大差距 |
| **置信度门控** | AUC 门控 3档 (0.50/0.55) → 仓位 0.25x/0.5x/1.0x | IC 门控 3档，但阈值全设 -10 → **实际禁用** | 🔴 严重配置问题 |
| **中性化** | Winsorize→Z-score→OLS残差（fit/transform范式，向量化） | 同上（继承V5） | ✅ 对齐 |
| **样本权重** | 序贯唯一性权重 (De Prado AFML Ch.7) | 同上（继承V5） | ✅ 对齐 |
| **防泄露** | T-1快照 + Purged K-Fold embargo + 标签封闭性 + reference_date统一 | T-1快照 + 时序划分（无embargo） | 🟡 部分对齐 |
| **确定性** | BLAS单线程 + hash seed=0 + nthread=1 + tree_method=hist | BLAS单线程 + hash seed=0（继承V5） | 🟡 部分对齐 |
| **执行参数** | min_commission=5.0 + transfer_fee_rate=1e-5 | 缺失 min_commission 和 transfer_fee_rate | 🟡 配置缺失 |
| **选股池** | price_min=3 | price_min=2 | 🟡 阈值偏低 |

---

## 二、详细发现

### P0 — 严重问题（必须修复）

#### P0-1: 无交叉验证 / 无 Purged K-Fold

**位置**: `gpsef_strategy_v10.py` L269-276, V5 `_select_features` L808-983

**现状**: V10 使用单次 80/20 时序划分进行特征选择和模型训练。训练数据按快照日排序，前80%为训练集，后20%为验证集。特征选择（IC→MDI→MDA）只在这一个划分上执行一次。

**问题**:
1. **特征选择脆弱**：单次划分导致特征选择高度依赖这个特定的 train/val 分割。不同划分可能选出完全不同的特征集，缺乏稳定性验证。
2. **无标签重叠隔离**：20日前向收益标签在连续快照日之间有19天重叠。训练集最后一个快照日的标签窗口延伸到验证集时间段，造成信息泄露。
3. **Alpha191 标准要求**：5折 Purged K-Fold + 21天 embargo（Prado AFML §7.4），每折独立做特征选择，跨折投票达成共识。

**影响**: 特征选择不稳定 → 模型泛化能力差 → 回测过拟合

**修复方案**: 
- 引入 Purged K-Fold CV（K=5, embargo=21天）
- 每折内独立执行 IC→MDI→MDA 特征选择
- 跨折投票：特征出现在 ≥2 折中 → 生产特征集
- 生产模型在全部训练数据上用共识特征训练

---

#### P0-2: 置信度门控实际禁用

**位置**: `config_v10.yaml` L79-81

**现状**:
```yaml
val_ic_trade_threshold: -10
val_ic_reduce_threshold: -10
val_ic_defend_threshold: -10
```

**问题**: Spearman IC 的正常范围是 [-1, 1]。阈值设为 -10 意味着任何模型表现都不会触发降档，策略永远以满仓 + 正常选股数量运行，即使模型完全失效。

**Alpha191 标准**: AUC 门控三档（normal ≥ 0.55 → reduced [0.50, 0.55) → defensive < 0.50），对应仓位 1.0x / 0.5x / 0.25x。

**影响**: 模型失效时无法自动减仓 → 全额亏损风险

**修复方案**:
- 将阈值改为合理值。对于二分类的 Spearman IC：
  - `val_ic_trade_threshold: 0.02`（低于此值减仓）
  - `val_ic_reduce_threshold: 0.01`（低于此值大幅减仓）
  - `val_ic_defend_threshold: 0.0`（负IC时防御模式）
- 或引入 AUC 门控（更符合二分类场景）：`auc_gate_min: 0.50, auc_gate_full: 0.55`

---

#### P0-3: 无模型集成（单模型预测）

**位置**: `gpsef_strategy_v10.py` L611-618

**现状**: 每次调仓训练单个 XGBoost 模型，预测也用这个单模型。

**Alpha191 标准**: 5个不同随机种子（42-46）分别训练，预测取平均。集成降低单模型方差约 √5 ≈ 2.24 倍。

**影响**: 单模型预测方差大 → 选股不稳定 → 换手率偏高

**修复方案**:
- 训练 5 个不同种子的 XGBoost 模型
- 预测时取 5 个模型概率的均值
- 可选：按验证集 AUC 加权

---

#### P0-4: 无概率校准

**位置**: `gpsef_strategy_v10.py` L692（`preds = self.model.predict(dpred)`）

**现状**: 直接使用 XGBoost `binary:logistic` 的原始输出概率。XGBoost 的 sigmoid 概率在尾部可能不够校准（过度自信或不够自信）。

**Alpha191 标准**: Isotonic 回归校准，在 OOS 验证预测上拟合校准器。

**影响**: softmax 权重分配基于未校准概率 → 权重可能偏离预期

**修复方案**:
- 在验证集预测上拟合 Isotonic 回归器
- 预测时先过校准器再进 softmax
- 或改用 Platt Scaling（sigmoid 校准，参数更少）

---

### P1 — 重要问题（建议修复）

#### P1-1: MDA 实现非最优（in-sample，无 t 检验）

**位置**: V5 `_select_features` L934-957

**现状**: 使用 sklearn `permutation_importance`，在验证集上做置换，`n_repeats=5`，`scoring=neg_log_loss`。判定标准是 `mda_series[f] > 0`（固定阈值0）。

**问题**:
1. **n_repeats=5 太少**：统计效力不足，纯噪声特征的假阳性率高（约50%）
2. **无 t 检验**：只用 `mean > 0` 判定，丢弃了 `std` 信息。Alpha191 使用 `mean > t_crit × SE`（t检验自适应阈值，df=19, t_crit=1.729）
3. **无 RNG 种子固定在 MDA 内部**：sklearn `permutation_importance` 用 `random_state=42`，但内部 shuffle 实现与 XGBoost 原生 DMatrix shuffle 不同

**Alpha191 标准**: OOS MDA（在 CV 验证折上置换），n_repeats=20，t检验自适应阈值。

**修复方案**:
- 将 `n_repeats` 提升到 20
- 引入 t 检验：`mean > t_crit(α=0.05, df=n_repeats-1) × (std/sqrt(n_repeats))`
- 或直接移植 alpha191 的 `compute_mda_oos` 函数（已在 ensemble.py 中优化实现）

---

#### P1-2: PSI 过滤未实现

**位置**: V5 `_select_features` L848-852, `config_v10.yaml` L89

**现状**: 配置文件有 `feat_sel_psi_threshold: 0.25`，V5 `__init__` 中也读取了 `self._feat_sel_psi_threshold`（L445），但 `_select_features` 方法中**没有使用** PSI 过滤。步骤2只做了 NaN 过滤和高相关去重。

**Alpha191 标准**: PSI < 0.15 才通过（train/val 分布稳定性检查）。

**影响**: 分布不稳定的特征可能进入模型 → OOS 表现退化

**修复方案**: 在步骤2和步骤3之间增加 PSI 过滤步骤。

---

#### P1-3: 无特征选择缓存

**位置**: V5 L452-454（缓存基础设施存在但 V10 未使用）

**现状**: V5 定义了 `_feat_sel_cache` 字典，但 V10 的 `_build_training_dataset` 每次调仓都重新执行完整特征选择管线。200天训练窗口 × 40+因子 × MDI 500轮 + MDA 5次重复，耗时可观。

**Alpha191 标准**: 无显式缓存，但 Purged K-Fold 本身通过跨折投票提供了特征稳定性验证，减少了单次选择的重要性。

**修复方案**: 
- 如果训练窗口重叠度高（每次调仓只移动20天），可缓存特征选择结果
- 或在 Purged K-Fold 框架下，特征稳定性由跨折投票保证，无需额外缓存

---

#### P1-4: 预测时特征分布潜在不匹配

**位置**: V5 `_build_feature_matrix` L1314-1366

**现状**: 训练数据由200天快照拼接而成（~200 × 4000 = 80万行），预测时只有单日快照（~4000行）。中性化中的 Winsorize 和 Z-score 是截面操作，在单日截面上计算。训练时每个截面有4000+股票，预测时如果候选池过滤后只剩几百只，Winsorize 的百分位和 Z-score 的均值/标准差会不同。

**Alpha191 标准**: 同样存在此问题，但 Alpha191 通过上下文特征（25日历史快照）缓解了 rolling 特征的退化。

**影响**: 训练-预测分布偏移 → 中性化残差有偏 → 预测失准

**修复方案**:
- 预测时对全市场做中性化，再从中提取候选池股票的中性化后因子值
- 或在候选池足够大时（>1000只）接受当前方案

---

#### P1-5: min_child_weight=50 过于保守

**位置**: `config_v10.yaml` L69

**现状**: `xgb_min_child_weight: 50`。对于二分类标签（0/1），一个叶子节点的 min_child_weight=50 意味着至少需要50个样本的二阶导之和。对于 logistic loss，二阶导 = p(1-p) ≤ 0.25，所以实际需要至少 200 个样本才能分裂。

**Alpha191 标准**: `min_child_weight=10`，等效约40个样本可分裂。

**影响**: 模型过于保守，可能无法学习有用的细粒度模式

**修复方案**: 降到 10-20，与 Alpha191 标准对齐。

---

#### P1-6: 特征选择的 FI 模型无 Early Stopping

**位置**: V5 `_select_features` L908-913

**现状**: FI 模型用 `num_boost_round=500` 全量训练，没有 early stopping。

**问题**: 可能过拟合训练数据 → MDI/MDA 偏差。Alpha191 的 CV 折内模型都有 early stopping（early_stopping_rounds=20）。

**修复方案**: 给 FI 模型加上验证集和 early_stopping_rounds。

---

#### P1-7: softmax 温度参数过激

**位置**: `config_v10.yaml` L100, `gpsef_strategy_v10.py` L719

**现状**: `pred_weight_temperature: 0.1`。对于二分类概率 [0.4, 0.6]，温度0.1使得 `exp((p-max)/0.1)` 范围从 1 到 `exp(-2)≈0.135`，权重差异可达 7.4 倍。

**Alpha191 标准**: `prob_temperature=0.3`，权重差异约 `exp(-0.67)≈0.51`，即最多2倍差异，更温和。

**影响**: 权重过度集中在最高概率股票 → 个股集中度风险

**修复方案**: 提高温度到 0.3-0.5，或参考 Alpha191 的 `prob_weight_alpha=0.7` 方案。

---

### P2 — 中等问题（可选修复）

#### P2-1: 缺少执行参数

**位置**: `config_v10.yaml` L51-56

**现状**:
```yaml
execution:
  commission_rate: 0.0003
  tax_rate: 0.0005
  slippage: 0.001
  max_positions: 50
  max_position_pct: 0.10
```

**缺失**:
- `min_commission: 5.0`（最低佣金，小额订单影响大）
- `transfer_fee_rate: 1.0e-05`（过户费，沪市股票必需）

**修复方案**: 补齐两个参数。

---

#### P2-2: 选股池 price_min=2 偏低

**位置**: `config_v10.yaml` L31

**现状**: `price_min:2`，Alpha191 用 `price_min:3`。

**问题**: 低价股（2-3元）更容易被操纵，波动更大，流动性可能不足。

**修复方案**: 提高到 3，与 Alpha191 对齐。

---

#### P2-3: XGBoost 参数冗余定义

**位置**: `config_v10.yaml` L62-73 vs L136-143

**现状**: XGBoost 参数在文件中被定义了两次——顶层 `xgb_*` 和 `v10:` 块内的 `xgb_*`。V10 的 `initialize` 方法先在 V5 中读取顶层参数构建 `self.xgb_params`，然后在 V10 中用 `v10:` 块覆盖 `objective`、`eval_metric`、`early_stopping_rounds`。

**问题**: 容易混淆哪个参数生效。特别是 `xgb_early_stopping_rounds` 在两处都定义为20，但修改时容易只改一处。

**修复方案**: 统一到一处定义，`v10:` 块只放 V10 特有的覆盖项。

---

#### P2-4: 缺少 tree_method 和 device 参数

**位置**: V5 `initialize` L470-482

**现状**: V5 的 `xgb_params` 没有设置 `tree_method` 和 `device`。XGBoost 默认 `tree_method=auto`，如果检测到 GPU 会使用 `gpu_hist`，引入跨机器非确定性。

**Alpha191 标准**: 显式 `tree_method: hist`, `device: cpu`。

**修复方案**: 在 `xgb_params` 中添加 `'tree_method': 'hist'`。

---

#### P2-5: trend_filter 查表使用 D-1 数据

**位置**: `gpsef_strategy_v10.py` L498

**现状**: `_apply_trend_filter` 使用 `prev_date = cal[idx - 1]`（D-1 交易日）的趋势因子。因子本身是5日回溯（lookback=5），所以 D-1 的因子使用了 D-2 到 D-6 的价格数据。

**分析**: 无未来数据泄露。但 trend_filter 的 back-adjusted price 使用 `raw_close * backAdjustFactor`（不含 `bf[ref]`），与 V5 的 `_get_label_components` 使用 `raw * bf[ref]` 方式不同。两者数学等价（L411注释已说明），但需确认 `backAdjustFactor` 的更新不依赖未来数据。

**风险**: 低。`backAdjustFactor` 是累积复权因子，到 D-1 为止的值不依赖未来。

---

#### P2-6: IV 控制非调仓日减仓逻辑

**位置**: `gpsef_strategy_v10.py` L570-585

**现状**: 非调仓日检测到 IV 升高 → 按比例减仓所有持仓 → 记录 `_iv_reduction_pending`。下一个调仓日重新计算目标权重时，不考虑已减仓状态。

**问题**: 
1. 减仓后到下次调仓之间可能产生不必要的现金拖累
2. 调仓日可能立即买回刚减仓的股票，产生无效换手

**修复方案**: 调仓日计算目标权重时，考虑当前持仓和 IV 状态，避免反向操作。或在 IV 减仓后设置冷却期。

---

#### P2-7: recomd 模式置信度门控阈值不一致

**位置**: `gpsef_strategy_v10.py` L986-998

**现状**: recomd 模式使用 `val_ic_defend_threshold` 和 `val_ic_trade_threshold` 两个阈值，但回测模式（L643-654）使用 `val_ic_defend_threshold` 和 `val_ic_reduce_threshold`。阈值名称和中间档位的语义不一致。

**修复方案**: 统一阈值名称和逻辑。

---

### P3 — 轻微问题（建议优化）

#### P3-1: `np.random.seed(42)` 位置

**位置**: `gpsef_strategy_v10.py` L172, L865

**现状**: `_build_training_dataset` 和 `recommend` 方法开头都调用了 `np.random.seed(42)`，重置全局随机状态。这确保了确定性，但如果上游有依赖随机状态的异步操作，可能被干扰。

**修复方案**: 使用局部 `rng = np.random.RandomState(42)` 替代全局 seed（低优先级）。

---

#### P3-2: 缺少概率裁剪

**位置**: `gpsef_strategy_v10.py` L692

**现状**: XGBoost 预测概率直接使用，无裁剪。极端概率（0.001 或 0.999）会导致 softmax 权重过度集中。

**Alpha191 标准**: 概率裁剪 [0.05, 0.95]。

**修复方案**: `preds = np.clip(preds, 0.05, 0.95)` 后再进 softmax。

---

#### P3-3: 模型性能指标缺少 AUC 门控

**位置**: `gpsef_strategy_v10.py` L759-803

**现状**: `_calc_model_performance` 计算了 AUC 但只用于日志输出，不用于仓位控制。仓位控制只看 Spearman IC。

**建议**: 对于二分类模型，AUC 是比 IC 更直接的模型质量指标。建议将 AUC 纳入门控逻辑。

---

#### P3-4: 特征选择无 `min_features` / `max_features` 约束

**位置**: V5 `_select_features`

**现状**: 特征选择管线可能返回任意数量的特征（0到全部）。没有最小/最大特征数约束。

**Alpha191 标准**: `min_features=10, max_features=150`。

**修复方案**: 增加 fallback 逻辑——如果通过所有过滤的特征少于 `min_features`，放宽阈值或返回 IC 排名前 N。

---

## 三、V10 设计亮点（应保留）

1. **二分类标签替代回归** — 合理决策。对于选股场景（排序问题），分类直接优化排序能力（IC），比回归（优化 R²）更适合。V10 文档中 R²=0.04 → 预测值坍缩的问题分析到位。

2. **趋势过滤层** — 创新设计。通过3维度（价格下降、成交量放量、MA20偏离）的 min 评分过滤弱势共振股票，且使用预计算缓存避免重复计算。设计合理，无泄露。

3. **IV 仓位控制** — 市场 regime 感知。通过隐含波动率分位数动态调整仓位，概念上优于固定仓位。需注意实现细节（P2-6）。

4. **移除日内止损止盈** — 简化决策。V9 回测显示筹码峰止损误判正常回调，拖累28.6%交易。V10 移除后依靠固定周期调仓 + 趋势过滤管理风险，逻辑自洽。

5. **继承 V5 中性化和样本权重** — 与 Alpha191 对齐的 Winsorize→Z-score→OLS 残差流程和 De Prado 序贯唯一性权重。

6. **固定复权基准日** — `_get_fuquan_reference_date` 统一复权基准，避免训练/预测因复权基准不同产生偏差。

---

## 四、优化方案优先级排序

### 第一批（P0，必须修复）

| # | 优化项 | 预估工作量 | 风险等级 | 预期收益 |
|---|--------|-----------|---------|---------|
| 1 | 引入 Purged K-Fold CV + 跨折共识 | 大（新增 ~200行） | 中 | 特征稳定性大幅提升，回测过拟合风险降低 |
| 2 | 修复置信度门控阈值 | 小（改配置） | 低 | 模型失效时自动减仓，尾部风险保护 |
| 3 | 5种子模型集成 | 中（修改 ~50行） | 低 | 预测方差降低 ~2x，选股稳定性提升 |
| 4 | 概率校准（Isotonic） | 中（新增 ~30行） | 低 | softmax 权重更准确 |

### 第二批（P1，建议修复）

| # | 优化项 | 预估工作量 | 风险等级 | 预期收益 |
|---|--------|-----------|---------|---------|
| 5 | MDA t 检验 + n_repeats=20 | 小（修改 ~20行） | 低 | 假阳性率从50%降到5% |
| 6 | 实现 PSI 过滤 | 小（新增 ~15行） | 低 | 过滤分布不稳定特征 |
| 7 | FI 模型加 Early Stopping | 小（修改 ~5行） | 低 | MDI/MDA 更准确 |
| 8 | min_child_weight 降到 10-20 | 极小（改配置） | 低 | 模型表达能力提升 |
| 9 | softmax 温度提高到 0.3 | 极小（改配置） | 低 | 降低个股集中度 |
| 10 | 预测时全市场中性化 | 中（修改 ~30行） | 中 | 训练-预测分布一致 |

### 第三批（P2/P3，可选优化）

| # | 优化项 | 预估工作量 | 风险等级 | 预期收益 |
|---|--------|-----------|---------|---------|
| 11 | 补齐执行参数 | 极小 | 低 | 交易成本更准确 |
| 12 | price_min 提高到 3 | 极小 | 低 | 过滤操纵股 |
| 13 | 添加 tree_method=hist | 极小 | 低 | 跨机器确定性 |
| 14 | 概率裁剪 [0.05, 0.95] | 极小 | 低 | 防止极端权重 |
| 15 | 统一 XGBoost 参数定义 | 极小 | 低 | 代码可维护性 |
| 16 | AUC 门控替代 IC 门控 | 小 | 低 | 更直接的模型质量评估 |
| 17 | min/max_features 约束 | 小 | 低 | 防止特征数失控 |

---

## 五、实施建议

### 5.1 推荐实施顺序

```
第一批（P0）→ 回测验证 → 第二批（P1）→ 回测验证 → 第三批（P2/P3）→ 回测验证
```

每批修改后必须回测对比，确认：
1. 策略收益不降低
2. 最大回撤不恶化
3. 换手率不显著上升
4. 特征选择结果更稳定（跨调仓日的特征重叠率提升）

### 5.2 风险控制

- **P0-1（Purged K-Fold）风险最高**：修改特征选择核心逻辑，可能改变策略行为。建议：
  1. 先实现 CV 框架但不改特征选择逻辑（只做 CV 评估）
  2. 确认 CV 评估结果与单次划分一致后，再切换到 CV + 跨折共识
  3. 保留原始单次划分代码作为 fallback

- **P0-3（模型集成）风险低但耗时增加**：5个模型训练时间 × 5。需评估回测耗时是否可接受。可考虑并行训练。

- **P1-10（全市场中性化）风险中**：改变预测路径的中性化范围，可能影响选股结果。建议先在 recomd 模式验证。

### 5.3 与 Alpha191 的关系

V10 策略目前被 Alpha191 用作"先验特征"（`v10_pred_rank`）。V10 的优化会直接影响 Alpha191 的输入质量。建议：
1. V10 优化后，重新生成 recomd 快照
2. Alpha191 端验证 `v10_pred_rank` 的 IC 是否提升
3. 如果 V10 特征选择更稳定，Alpha191 的 MDA 投票中 `v10_pred_rank` 的通过率应提升

---

## 六、总结

Wyckoff V10 策略在**因子设计**（40个威科夫因子 + 趋势过滤）和**标签设计**（二分类替代回归）方面有独到见解，但在**特征选择方法论**（无 CV、无共识投票）、**模型稳健性**（无集成、无校准）、**风控机制**（门控禁用）三个方面与 Alpha191 标准存在显著差距。

核心改进路径：**Purged K-Fold CV + 模型集成 + 门控修复**三项 P0 修复预计可将策略从"单次划分过拟合"提升到"CV 验证 + 集成稳健"的水平，与 Alpha191 形成统一的量化方法论体系。

---

*本报告仅供审查参考，未修改任何代码文件。请审核后决定下一步行动。*
