# 威科夫 V10 对齐 small_alpha191 修复方案（2026-08-18，待审核）

> 对齐目标：small_alpha191 当前的超参筛选方式（HP-Search A+B+C）、K-Fold 用法（PKF + 折内 inner-OOS）、模型集成（折集成围绕选中超参扰动 + LGBM 权重降 0）。
> 范围：`strategies/wyckoff20/gpsef_strategy_v10.py` + `strategies/wyckoff20/config_v10.yaml`。
> 状态：**仅方案，未改任何代码**。审核通过后执行。

---

## 一、现状对比（代码实证）

| 维度 | small_alpha191（对齐目标） | V10（现状） |
|---|---|---|
| 特征数 | 301 原始 → 127 预处理 → MDA 后 15-19 | **120 原始** → prefilter → ~80 max（feat_sel_max_features） |
| 训练样本 | 66,000（240 天 × 300 只/日） | **1,260,245**（200 天 × 全市场，日志实证） |
| 超参选择 | **HP-Search A+B+C**：3 窗 expanding + 生产 val 零接触 + 每候选每窗早停 + LRWarmup + AUC<0.5 防护不写回 | **无**。xgb 参数固定：depth=6, lr=0.05, ss=0.7, cs=0.8, mcw=3, α=0.3, λ=1.0, γ=0.0, n_estimators=1000 |
| PKF | 折内再切 **inner-OOS**（15% 或 ≥20 天），**MDA 在 inner-OOS 上算**，验证块只评 AUC | MDA 在 **fold_val（验证块）** 上算 → 选择偏倚（Cawley-Talbot） |
| IC 准入 | **已移除**（consensus_features=None，MDA 投票池=基础清理全集） | `cv_feature_consensus: 3`（IC ≥3 折共识才入投票池，限制 MDA 裁决范围） |
| 折集成多样性 | 围绕**选中超参扰动**：depth=[d,d,d+1,d+1,d+2]、ss/cs=±0.05、metric 全 auc | 硬编码 `[6,7,7,8,8]/[0.6..0.8]/[0.6..0.8]/[auc,auc,logloss,logloss,auc]`（与超参选择结果无关） |
| 评估协议 | HP-Search 带 **LRWarmup**（与生产一致） | 无 HP-Search；生产侧已有 LRWarmup（lr_warmup_enabled: true） |
| LGBM | `enabled: false` / 权重 1.0/0.0（XGB-only） | `enabled: true` / 权重 **0.6/0.4**（LGB 参与融合） |
| 门控口径 | 折集成 meta_AUC 单口径 | 折集成 meta_AUC + 三档（full≥0.52 / half≥0.48 / empty<0.48）——**已是折集成口径，无需改** |
| 样本权重 | 曾修排序错位（align_sample_weight_to_sort） | 折内**重算**（compute_sample_weights(fold_train, ...)），无位置索引错位风险——**天然安全，无需改** |

V10 与 small 的规模差异是超参设计的关键约束：**特征少（~80 vs 301）、样本多（126 万 vs 6.6 万）**。

---

## 二、待修复点清单

### P1（用户明确要求的三项，必改）

**P1-1 移植 HP-Search 超参选择（small 的 A+B+C 全协议）**
- 插入点：`_prefilter_features` 之后、`_purged_kfold_cv` 之前（V10 L3053-3078 之间），与 small 的 `train()` 中 pre_feats 后一致。
- 移植内容（与 small 逐项对齐）：
  - [A] 选择-评估隔离：剔除与生产同规则的末 20 天（`inner_val_days`），选择只用其前段，生产 val 零接触；
  - [B] 多窗评估：3 窗 × 20 天 expanding walk-forward（训练严格早于评估），得分=各窗平均；
  - [C] 每候选每窗独立 early stopping + **LRWarmupCallback(target_lr=候选 lr)**（协议与生产一致）；
  - 评估特征：与 PKF 折内一致的 IC/PSI 筛选子集（复用 V10 自身的 `_prefilter_features` 逻辑）；
  - 防护：binary 全部候选 val AUC<0.5 → 不写回、保持原超参；
  - 写回机制：写回 `self.xgb_params`（max_depth/eta/subsample/colsample_bytree/alpha/lambda/min_child_weight/gamma）——V10 的折集成与生产模型均从 `self.xgb_params` 读取，天然传导。
- 超参候选集：**按 V10 规模重新设计**（见第三节），不能直接套 small 的 4 组。

**P1-2 PKF 折内 inner-OOS（P0-1 对齐）**
- `_purged_kfold_cv`（L1531）：每折 `fold_train` 尾部再切 inner-OOS（`inner_oos_frac: 0.15` 或 `inner_oos_min_days: 20`）；
- 特征选择（IC/PSI）与模型训练用 inner-train，**MDA（含 CFI 簇级分支）改在 inner_oos_set 上计算**，`fold_val` 只评 AUC；
- 配置：`feature_selection.mda` 段新增 `inner_oos_frac: 0.15` / `inner_oos_min_days: 20`；
- 收益：消除"验证块参与特征选择"的选择偏倚，折集成 meta_AUC（门控依据）回归可信。

**P1-3 LGBM 权重降 0（对齐 XGB-only）**
- `config_v10.yaml` → `ensemble.lightgbm`：`enabled: true → false`，`ensemble_weight_xgb: 0.6 → 1.0`，`ensemble_weight_lgb: 0.4 → 0.0`；
- 代码侧 `_lgb_enabled`（L465）读 enabled=false 后，训练/校准/预测融合分支全部跳过（与 small 相同机制，行为=纯 XGB 折集成）。

### P2（对齐 kfold/集成语义，建议一并改）

**P2-1 折集成多样性围绕选中超参扰动**
- 现状：`ensemble.diversity.max_depth_list: [6,7,7,8,8]` 等硬编码（L455-457 读入 `self._ensemble_depth_list`）；
- 改法：与 small 的 [A] 修复同构——`depth_list=[d,d,d+1,d+1,d+2]`（d=HP-Search 选中）、`ss/cs=选中±0.05`（clip 0.3-1.0）、`metric_list=['auc']*5`；
- 注意：`eval_metric_list` 从混合 auc/logloss 改全 auc（与 small 一致，避免早停目标不一致干扰权重排序）；
- config 的 `diversity` 段改为"相对扰动"语义（`perturb_depth: [0,0,1,1,2]`、`perturb_ss: [-0.05,0,0,+0.05,+0.05]` 或直接代码派生，**方案倾向代码派生 + config 保留开关**，减少配置面）。

**P2-2 移除 IC 准入（P1-1 对齐）——需用户裁决**
- V10 的 `cv_feature_consensus: 3` 是"IC ≥3 折共识才入投票池"，**与 small 原 `=1`（并集）语义不同**：small 的 =1 无过滤作用，V10 的 =3 有实际过滤；
- 对齐 small 的裁决（"线性 IC 不决定非线性模型投票池"）：改 `consensus_features=None`，投票池=prefilter 全集，MDA 投票（`vote_min_folds: 2`）裁决一切；
- **代价**：投票池扩大（~80 → 全部 prefilter 特征），MDA 计算量上升（126 万样本 × n_repeats=20 × 5 折，耗时增加，可调 n_repeats 或依赖重训间隔）；
- **决策点**：若担心 V10 计算成本，可保留 consensus=3（保守选项）；对齐则移除。**建议移除**（方法论一致性，与 small 完全对齐）。

### P3（范围外确认项，需用户指示）

**P3-1 V10 的 factor_validator 三件套**
- V10 仍运行自己的三件套（日志实证：`[FactorValidation] 自动剔除特征: 123 → 95`、SHAP 衰减、CIC）；
- small_alpha191 已完全移除三件套（用户 2026-08-18 决策）。**是否对 V10 一并移除？**（本次范围未含，单列询问；若移除，另出详细清单）

**P3-2 门控阈值与标签模式**
- V10 三档门控（0.52/0.48）vs small 单阈值（0.52）：策略行为差异，**不动**（用户未要求）；
- V10 标签为二分类（跑赢截面中位数），small 为 staged_binary 分阶段退出：**不动**（本次范围仅超参/kfold/集成）。

---

## 三、V10 超参候选设计（适配 120 特征 / 126 万样本）

### 3.1 规模差异的约束推导

| 维度 | small_alpha191 | V10 | 对超参的影响 |
|---|---|---|---|
| 特征数 | 301 → MDA 后 15-19 | 120 → prefilter ~80 | 特征少 → 每棵树能充分利用 → **colsample 可高（0.7-0.9）** |
| 训练样本 | 66,000 | 1,260,245 | 样本多 → 容量可大 → **depth 可更深（5-8）**、**mcw 可小（2-8）** |
| 单日截面 | 300 只 | 全市场 ~6,300 只/日 | V10 截面多样性高 → 树分裂方差小 → 正则可略松 |
| 信噪比 | 低（val AUC 0.52-0.57） | 中等（历史 meta_AUC ~0.52-0.56） | 均需 moderate 正则，但 V10 可承受更深模型 |

### 3.2 候选表（4 组，c2 = V10 现状基线）

| 候选 | depth | lr | n_estimators | subsample | colsample | α | λ | mcw | γ | 定位 |
|---|---|---|---|---|---|---|---|---|---|---|
| c1_balanced_shallow | 5 | 0.04 | 800 | 0.7 | 0.8 | 0.2 | 0.8 | 5 | 0.0 | 稳健防御（保守档） |
| **c2_v10_baseline** | **6** | **0.05** | **1000** | **0.7** | **0.8** | **0.3** | **1.0** | **3** | **0.0** | **V10 现状基线（对照锚点）** |
| c3_deeper | 7 | 0.03 | 1000 | 0.7 | 0.85 | 0.2 | 0.6 | 3 | 0.0 | 深容量（样本支撑） |
| c4_strong | 8 | 0.02 | 1200 | 0.75 | 0.85 | 0.1 | 0.5 | 2 | 0.0 | 最强表达（126 万样本） |

设计逻辑：
- **c2 必须保留为候选**——作为"现状锚点"，若 HP-Search 选出 c2，证明现有配置已近最优（回测可直接对照）；
- depth 从 5→8（small 是 3→6）：样本 19 倍 → 容量上限提高约 2 档；
- mcw 从 15-30（small）降到 2-5：mcw 是"叶节点最小样本和"，样本多时应减小（同样 30 在 126 万样本下形同虚设）；
- colsample 0.8-0.85（small 0.6-0.8）：特征少 → 每棵树用更多特征；
- lr 0.02-0.05（small 0.02-0.05 相同区间）：n_estimators 上限提至 800-1200（样本多 → 可训更多轮，早停兜底）。

### 3.3 评估协议参数（与 small 一致）

```
hyperparam_search:
  enabled: true
  inner_val_days: 20        # 选择-评估隔离：剔除生产 val 同规则 20 天
  eval_windows: 3           # expanding 多窗
  early_stopping_rounds: 20 # 每窗早停耐心
  eval_rounds: 1200         # 每窗轮数上限（≥ 最激进候选 n_estimators）
  use_early_stop: true      # 每窗独立早停（业界标准）
```

### 3.4 耗时预估

- HP-Search：3 窗 × 4 候选 = 12 次训练；V10 单次 80 特征 × 126 万样本 ×（早停 ~50-150 轮）≈ 30-60s → **约 6-12 min/重训期**；
- PKF inner-OOS：MDA 样本略减（-15%），耗时近似不变；
- 折集成 5 模型：不变；
- 总重训耗时增量 ~10 min/期，可接受（重训间隔建议与 small 一致 5 期一次——**V10 当前无 retrain_interval，本次是否一并加？P3 确认项**）。

---

## 四、实施顺序与验证

1. **P1-3（LGBM 关停）**：config 3 行改动，风险最低，先做；
2. **P1-1（HP-Search 移植）**：核心改动。实现 V10 版 `_select_hyperparams`（AST 从 small_alpha191 提取方法体 → 适配 V10 的 self.xgb_params/命名空间 → 插入主流程）→ 候选表按 3.2 写入 config；
3. **P2-1（折集成扰动）**：改 `_train_cv_fold_ensemble` 的 diversity 构造；
4. **P1-2（inner-OOS）**：改 `_purged_kfold_cv` 折内结构；
5. **P2-2（IC 准入移除）**：投票调用传 None（等用户裁决）。

测试（沿用项目 AST 提取约定，双倍严谨）：
- 专项测试 `scripts/test_v10_align.py`：
  - HP-Search：enabled=false 跳过 / 候选空跳过 / 选中写回 self.xgb_params 九参数一致 / AUC<0.5 不写回 / 多窗 detail 存在 / 隔离（生产 val 零接触）/ warmup 生效（无 "are not used" 警告）
  - inner-OOS：源码断言 MDA 输入改 inner_oos_set / val 零接触 / 端到端 train 成功
  - 折集成扰动：depth 派生自选中值、越界 clip、metric 全 auc
  - LGBM：enabled=false → lgb_models 空、predict_proba 纯 XGB
- 全量回归：V10 现有测试（tests/test_v10_*.py）+ small_alpha191 五套件不受影响；
- py_compile + config 解析。

---

## 五、需要用户裁决的 3 个决策点

1. **P2-2 IC 准入**：移除（完全对齐，推荐）还是保留 consensus=3（保计算）？
2. **P3-1 factor_validator 三件套**：V10 是否一并移除（small 已移）？
3. **retrain_interval**：V10 是否同步加"每 N 次调仓重训"（small 已有 retrain_interval: 5，注意当前 small 两目录 config 实际是 1）？
