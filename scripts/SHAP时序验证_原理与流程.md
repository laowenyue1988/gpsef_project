# 因子验证三件套之 SHAP 时序验证 · 完整原理与流程讲解

> 对应代码：`strategies/hs300_alpha191/factor_validator.py`（以及 alpha191 系全部策略的同名文件）
> 阅读对象：需要理解"因子验证三件套中 SHAP 探针到底在干什么"的学习者
> 本文档风格：每个概念先讲**大白话**，再给**严格数学**，最后用**真实回测日志实例**串起来。
> 文中所有实例数字来自真实回测 `results/_hs300_fix_test/backtest.log`（hs300_alpha191，47 个调仓周期）。

---

## 目录

1. [一句话回答](#一一句话回答)
2. [SHAP 探针要解决什么问题](#二shap-探针要解决什么问题)
3. [前置知识：SHAP 值到底是什么（含数学）](#三前置知识shap-值到底是什么含数学)
4. [三件套全景：SHAP / ADV / CIC 的分工与投票](#四三件套全景shap--adv--cic-的分工与投票)
5. [SHAP 时序验证的完整流程（逐行对应代码）](#五shap-时序验证的完整流程逐行对应代码)
6. [两个版本：快照版 vs 回测版](#六两个版本快照版-vs-回测版)
7. [真实实例：cycle=0 完整走一遍](#七真实实例cycle0-完整走一遍)
8. [设计考量与已知局限（为什么它"从不触发"）](#八设计考量与已知局限为什么它从不触发)
9. [参数表与调参指南](#九参数表与调参指南)
10. [FAQ](#十faq)
11. [附录：代码速查图](#十一附录代码速查图)

---

## 一、一句话回答

**SHAP 时序验证（SHAP 探针）是因子验证三件套里的"体检医生"**：它每期用一小个轻量模型，计算每个因子对预测的贡献（SHAP 值），然后比较"训练期"与"近期"因子的贡献有没有**大幅下滑**——下滑超过阈值就判定该因子"正在失效"，标记剔除候选。**它的目的是在因子彻底没用之前提前发现"重要性衰减"**。

但**现实是**：在当前 alpha191 框架下，这个探针**结构性失效——47 个调仓周期它 0 次触发（SHAP=0 票）**。本文档不仅讲清原理，也会讲清它为什么失效（时间尺度错配），这对你判断"要不要修它"至关重要。

---

## 二、SHAP 探针要解决什么问题

**问题背景**：alpha191 有 200+ 个候选因子（`alpha_001` ~ `alpha_191` 及它们的 `_score`/`_neu` 变体）。这些因子**不是永远有效的**——某个因子可能在一段时间内能预测股价（比如"过去 20 天涨幅"在趋势市有效），换一个行情就失效了。

**如果因子失效了却继续用它训练模型**，模型会把垃圾当信号学，造成两种危害：
1. **过拟合**：模型在旧数据上学到的关系，未来不复存在；
2. **信号污染**：失效因子的噪声干扰其他好因子的学习。

**SHAP 探针的职责**：在每个调仓周期，检测"哪些因子的重要性在近期大幅下降"，把它们标记出来，交给三件套投票剔除。

---

## 三、前置知识：SHAP 值到底是什么（含数学）

### 3.1 大白话

SHAP（**SH**apley **A**dditive ex**P**lanations）回答一个问题：**"一次预测的结果，各个特征各自贡献了多少？"**

类比：三个同事（特征 A、B、C）一起完成一个项目（预测），最终产出 100 分（模型输出）。SHAP 值就是"公平地"把 100 分分给 A、B、C，使得每个人拿到的份额加起来恰好等于 100。

**公平**在这里有精确含义：一个人的贡献 = 他**在所有可能合作组合中的边际贡献的平均**。比如 A 单独干得 40 分、A+B 干得 70 分，则 A 在"B 在场"时的边际贡献是 70−40=30。把所有组合都试一遍再平均，就是 A 的 Shapley 值。

### 3.2 严格数学

对单个样本 $x$，特征 $j$ 的 SHAP 值（Shapley 值）定义为所有特征子集上的加权平均边际贡献：

$$\phi_j(x) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!} \left[ f_x(S \cup \{j\}) - f_x(S) \right]$$

其中 $F$ 是全部特征集合，$f_x(S)$ 是"只知道集合 $S$ 里特征的值、其余特征用基准值填充"时模型的预测值。

**三条公理**（Shapley 值的理论基础）：
1. **可加性**：$f(x) = \phi_0 + \sum_j \phi_j(x)$（所有 SHAP 值 + 基准值 = 预测值）
2. **对称性**：两个特征对任何子集的边际贡献相同 → 它们的 SHAP 值相同
3. **哑元性**：对任何子集都没有边际贡献的特征 → SHAP 值 = 0

**对于树模型**，XGBoost/LightGBM 采用 **TreeSHAP** 算法：它利用树的结构（叶子值、决策路径）把上面的组合枚举在 $O(T L D)$ 时间内算完（$T$=树数、$L$=叶子数、$D$=深度），而不是枚举 $2^{|F|}$ 个子集——这是它能在 200+ 特征上实用的原因。

### 3.3 为什么用 |SHAP| 的均值当"特征重要性"

- 单个样本的 $\phi_j(x)$ 有正有负（正=推动预测往 1 走，负=往 0 走）；
- 取**绝对值** $|\phi_j(x)|$ = 该特征对这个样本预测的"影响力大小"（不管方向）；
- 对所有样本求平均 $mean(|SHAP_j|)$ = 这个特征**平均有多重要**。
- 这就是经典的 **mean |SHAP| 重要性**，比 XGBoost 自带的 `gain` 重要性更稳健（gain 只看分裂点，mean|SHAP| 看对每个样本预测的贡献）。

> **直观理解**：如果删除一个特征对预测影响巨大，它的 |SHAP| 均值就大；如果它可有可无，均值≈0。

---

## 四、三件套全景：SHAP / ADV / CIC 的分工与投票

因子验证三件套（`FactorValidationPipeline.run_rebalance`）每期跑三个独立检查器，各自标记"可疑因子"，最后**投票**决定剔除谁。

| 检查器 | 检测什么 | 度量方式 | 标记语义 |
|---|---|---|---|
| **SHAP 时序** | 因子重要性是否**随时间衰减** | train vs val 的 mean\|SHAP\| 比值 | "这个因子正在失效" |
| **ADV 对抗验证** | 训练/验证分布是否**漂移** | 对抗分类器 AUC | "这个因子分布不稳定" |
| **CIC 条件 IC** | 因子 IC 是否随市场状态**翻转** | 各条件桶 RankIC 符号/范围 | "这个因子方向不稳定" |

**投票与剔除**（`factor_validator.py` L1124-1148）：

```python
vote_counts = {}                       # 每个特征被几个检查器标记
for result in [shap, adv, cic]:
    for feat in result.affected_features:
        vote_counts[feat] = vote_counts.get(feat, 0) + 1
report.features_to_remove = [f for f, v in vote_counts.items()
                             if v >= self.min_votes_to_remove]
```

即：**被 ≥ min_votes 个检查器标记的特征进入剔除名单**，随后从特征池中移除，不再参与模型训练。

**实测票数**（47 周期平均，来自回测日志）：

```
Votes: SHAP=0/302, ADV=5, CIC=100.7 → total≈105 (min_votes=1)
```

→ **SHAP 探针 0 票（从不触发）、ADV 恒定 5 票、CIC 是绝对主力（每期约 100 票）**。

---

## 五、SHAP 时序验证的完整流程（逐行对应代码）

回测路径调用的是 `FactorValidationPipeline._validate_shap_rebalance`（L1207-1339）。完整流程 7 步：

### 第 1 步：前置检查（L1218-1225）

```python
if not _SHAP_AVAILABLE or len(feature_names) < 3:
    return SKIP          # shap 包没装 / 特征太少
if n_train < 200:
    return SKIP          # 训练样本太少
```

**作用**：环境或数据不足时优雅跳过，不打乱主流程。

### 第 2 步：确定性采样（L1228-1252）

```python
rng = np.random.RandomState(42)              # 固定随机种子 → 结果可复现
max_train_samples = 50000
if n_train > 50000:
    sample_idx_tr = rng.choice(n_train, 50000, replace=False)   # 最多 5 万
```

**作用**：训练集可能 7 万+ 行，SHAP 计算昂贵，采样到 5 万（固定 seed=42，保证每次回测结果一致、可审计）。

**实例**（cycle=0 日志）：
```
[SHAP/Rebalance] Training on 50000 samples (sampled from 50402, reproducible seed=42)
```
即：50402 行采样 50000 行。

### 第 3 步：训练"探针模型"（L1254-1268）

```python
params = {'max_depth': 6, 'eta': 0.1, 'objective': 'binary:logistic',
          'eval_metric': 'logloss', 'tree_method': 'hist',
          'subsample': 0.8, 'nthread': 1, 'seed': 42}
model = xgboost.train(params, dtrain_shap, num_boost_round=100)
```

**关键点**：
- 这是**专用探针模型**（depth=6、100 轮、单线程），与主模型（更深的 XGB + LGB 集成）**不同**——它只用来算 SHAP，不参与交易决策；
- 参数可从 config 调整（`probe_max_depth=6`、`probe_num_rounds=100`）；
- **为什么参数演进**：历史上 depth=3 时探针只用 6% 的特征（其余特征 SHAP≈0），depth 提到 6 后覆盖率到 100%（config 注释原文）。

### 第 4 步：计算 SHAP 值（L1270-1281）

```python
explainer = shap.TreeExplainer(model, feature_names=feature_names)
shap_train = np.abs(explainer.shap_values(X_train[idx_tr]))   # train 采样 2000 行
shap_val   = np.abs(explainer.shap_values(X_val[idx_vl]))     # val 采样 1000 行
```

**作用**：用 TreeSHAP 快速算出每个样本上每个特征的贡献 $\phi_j$，取绝对值。

### 第 5 步：逐特征比较 train vs val 的重要性（L1283-1311）

```python
tr_mean = train 上 |SHAP_j| 的均值
vl_mean = val   上 |SHAP_j| 的均值
decay_ratio = vl_mean / max(tr_mean, 1e-8)

# 重要性下限 (probe_importance_floor='median')
imp_floor = median(所有特征 train 的 |SHAP| 均值)
if tr_mean < imp_floor:
    continue          # 探针本身就不重视的特征 → 不判衰减, 交下游 MDA 裁决
if decay_ratio < probe_decay_threshold(0.35):
    decay_features.append(fname)      # 标记"正在衰减"
```

**这是核心判定**：一个特征在 val（近期）上的平均贡献，如果掉到 train（整个窗口）的 **35% 以下**，且它本身在 train 上是"重要特征"（高于全体中位数），就判定**衰减**。

**实例**（cycle=0 日志）：
```
[SHAP/Rebalance] 重要性下限=0.008594, 豁免 202/302 个低重要性特征的衰减判定
```
即：302 个特征里，train 上 mean|SHAP| 低于中位数（0.008594）的 202 个被豁免（它们 SHAP≈0，比值不稳定，直接判衰减会造成大面积误报），**只有 100 个高重要性特征参与判定**。

### 第 6 步：汇总判定 PASS / WARNING / FAIL（L1313-1335）

```python
n_decayed = 衰减特征数
decay_frac = n_decayed / 特征总数
if n_decayed == 0:                    status = PASS
elif decay_frac < 0.15:               status = WARNING   # <15% 特征衰减
else:                                 status = FAIL      # ≥15% 特征严重衰减
```

返回 `affected_features = decay_features`（PASS 时为空列表 → 0 票）。

### 第 7 步：进入三件套投票（见第四节）

被标记的衰减特征进入 `vote_counts`，≥ min_votes 即剔除。

---

## 六、两个版本：快照版 vs 回测版

项目中 SHAP 时序验证有两个实现，**检测的时间跨度不同**：

| | 快照版 `ShapTemporalValidator`（L97） | 回测版 `_validate_shap_rebalance`（L1207） |
|---|---|---|
| 使用场景 | 离线全量扫描 `run()` | 每调仓周期 `run_rebalance()` |
| 时间分桶 | **12 个时间桶**（训练窗口内按时间均分） | **train vs val 两段**（整个窗口 vs 最近 20 天） |
| 比较对象 | 前 10 桶（早期）vs 后 2 桶（近期） | train 段 vs val 段 |
| 衰减定义 | 后 2 桶 mean\|SHAP\| / 前 N-2 桶 < 0.5 | val mean\|SHAP\| / train mean\|SHAP\| < 0.35 |
| 检测尺度 | 窗口内"早期 vs 近期"（几个月） | 窗口 vs 最近 20 天 |

**设计意图**：快照版检测"窗口内部的长期衰减"，回测版检测"近期突变"。

---

## 七、真实实例：cycle=0 完整走一遍

以 hs300 回测 cycle=0（2022-08-01 调仓）为例，日志时间线：

```
03:20:03 [SHAP/Rebalance] Training on 50000 samples (sampled from 50402, ...)   ← 第2步: 采样
03:20:35 [SHAP/Rebalance] 重要性下限=0.008594, 豁免 202/302 个低重要性特征的衰减判定
          └─ 302 特征中 202 个 mean|SHAP| < 0.008594 被豁免 → 仅 100 个参与判定
...（SHAP 计算与比较，无日志输出，内部完成）...
03:23:37 [FactorValidation] Votes: SHAP=0/302, ADV=5, CIC=58 → total=61 (min_votes=1)
          └─ SHAP 探针 0 个衰减特征 → affected_features=[] → 0 票 → PASS
```

**解读**：
1. 探针在 5 万样本上训练（31 秒）；
2. 302 个特征中只有 100 个"够格"参与判定；
3. 这 100 个特征在 val（最近 20 天）上的 mean|SHAP| 都 ≥ train 的 35% → **无衰减 → PASS → 0 票**；
4. 该周期由 CIC（58 票）主导剔除。

47 个周期全部如此：`SHAP=0/302`（平均），**探针从未投出一票**。

---

## 八、设计考量与已知局限（为什么它"从不触发"）

这是本文档最重要的部分。实测 SHAP 探针 47 周期 0 触发，不是偶然，是**结构性失效**，原因有四层：

### 原因 1：时间尺度错配（根本原因）

- **因子衰减是慢变量**：一个因子的有效性通常以 6~12 个月为单位逐渐消失（风格切换、市场参与者变化）；
- **探针对比的是"整个训练窗口" vs "最近 20 天"**：训练窗口约 240 个交易日（近 1 年），val 是其中最后 20 天。**两者都在同一"有效期"内**——窗口早期的因子有效，20 天前依然有效，结构相似 → ratio≈1，永不衰减；
- **要检测慢衰减，需要对比"更早 vs 现在"**（快照版的 12 桶设计就是这个思路），而回测版改成了 train vs val，等于把"长期体检"改成了"20 天快速复查"——复查当然查不出慢性病。

### 原因 2：探针模型与主模型不一致

- 探针用轻量模型（depth=6、100 轮、单线程）算 SHAP，主模型是深 XGB + LGB 集成；
- 探针眼中的"重要性"与主模型眼中的不一定相同——探针说"没衰减"不代表主模型里的真实贡献没衰减；反之亦然。

### 原因 3：阈值太宽松 + 豁免机制

- 衰减判定阈值 0.35（val 要跌到 train 的 35% 以下才算）——20 天内特征重要性崩 65% 是极端事件；
- `importance_floor=median` 豁免了 2/3 的特征（SHAP≈0 的不判），进一步缩小触发面。

### 原因 4：注入测试证实（2026-08-01 实测）

项目曾做过**注入测试**：人为构造"因子预测力衰减"的场景（如把后期数据替换为噪声），观察探针是否触发——结果是**结构性不敏感，永远无法触发**。说明这不是参数没调好，而是检测口径本身测不到目标事件。

### 结论与建议

- **探针不是"功能坏了"，而是"测量对象选错了"**：它检测的是 20 天短期重要性突变（几乎不发生），而真正需要检测的是"窗口 vs 更早历史"的长期衰减；
- 若想让它真正工作，应改为**对比"更早窗口（如 9 个月前）vs 当前窗口"的 mean|SHAP|**（对齐快照版 12 桶的思路），并配合更宽的阈值与模型对齐；
- 当前框架下，**因子筛选的真实担当是 IC/PSI 粗筛 + Purged K-Fold MDA 投票**（模型内度量）——SHAP 探针的 0 票不影响最终特征质量（它只是没帮忙，也没捣乱）。

---

## 九、参数表与调参指南

| 参数 | 位置 | 默认 | 含义 | 调参方向 |
|---|---|---|---|---|
| `probe_max_depth` | config `factor_validation.shap_temporal` | 6 | 探针树深 | 3→6 修复了特征覆盖率 6%→100% |
| `probe_num_rounds` | 同上 | 100 | 探针迭代轮数 | 40→100 对齐 depth=6 收敛 |
| `probe_decay_threshold` | 同上 | 0.35 | val/train 比值低于此值判衰减 | 0.5→0.35，只杀真正崩盘 |
| `probe_importance_floor` | 同上 | `median` | 低于该 train 重要性的特征豁免 | 防"SHAP≈0→ratio≈0→误报" |
| `min_votes_to_remove` | 同上 | 2（config 实际 1） | 至少几个检查器标记才剔除 | 越大越保守 |
| `n_time_bins`（快照版） | 代码参数 | 12 | 时间桶数 | 快照版用 |
| `decay_threshold`（快照版） | 代码参数 | 0.5 | 快照版衰减阈值 | 快照版用 |

**调参指南**：想让探针"更灵敏"→ 调大 `probe_decay_threshold`（如 0.5）、放宽 `probe_importance_floor`（如 0）；想让探针"更保守"→ 反之。但按第八节分析，**调参无法解决时间尺度错配**——治本需要改对比窗口。

---

## 十、FAQ

**Q1：SHAP 值等于特征重要性吗？**
A：不完全。|SHAP| 均值是重要性的一种度量（基于对每个样本预测的贡献），比 gain 更稳健。SHAP 值本身是"贡献值"（有正负），不是重要性。

**Q2：为什么探针训练要用和主模型不同的轻量模型？**
A：性能考虑——主模型训练要几分钟，SHAP 每期还要算几百特征，探针用轻量模型（depth6/100轮/单线程）30 秒内完成。代价是探针与主模型的"视角"不完全一致。

**Q3：探针 0 票是不是 bug？**
A：不是运行时 bug（流程正常走完、PASS），而是**检测口径与目标事件错配**（见第八节）。是"设计性失效"，不是"代码性故障"。

**Q4：SHAP 探针是不是完全没用？**
A：在当前形态下它对"剔除因子"无实际贡献（0 票），但保留了两点价值：①作为"重要性衰减监控"的骨架（改造后可用）；②探针的 mean|SHAP| 可作为因子诊断的辅助参考。

**Q5：为什么 202 个特征被豁免？**
A：302 个候选特征里，多数在探针模型中平均 |SHAP| 极低（<全体中位数 0.008594）——它们本来就是弱信号，SHAP≈0 时 ratio 数值不稳定（0.0001/0.0003 的噪声抖动会被误判成衰减），豁免它们避免大面积误报。

---

## 十一、附录：代码速查图

```
run_rebalance(train_data, val_data, feature_names, market_cache)
 │
 ├─ ① SHAP 时序验证 ── _validate_shap_rebalance(X_train, X_val, features, labels)
 │     采样(≤50K, seed=42) → 训探针XGB(depth6,100轮) → TreeExplainer
 │     → train|SHAP|均值 vs val|SHAP|均值 → ratio<0.35 且 高于median → 衰减特征
 │     → affected_features → 0票(PASS) / N票(WARNING/FAIL)
 │
 ├─ ② 对抗验证 ── adv_validator.validate(train_mask)  → 分布漂移特征
 │
 ├─ ③ 条件IC ── cic_analyzer.validate(quarter/volatility/trend 分桶 RankIC)
 │     → 符号翻转/范围过大特征
 │
 └─ 投票: vote_counts[feat] >= min_votes → features_to_remove → 剔除
         （另有跨周期 persistent 持久化剔除）
```

---

*文档版本：v1.0（2026-08-11）· 对应 factor_validator.py 三件套中的 SHAP 时序验证*
