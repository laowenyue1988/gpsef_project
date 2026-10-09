# alpha191 系策略全量对齐 small_alpha191 方案（2026-08-19，待审核）

> 对齐标准：strategies/small_alpha191（当前最新：HP-Search A+B+C + PKF inner-OOS + IC准入移除 + 折集成扰动 + 三件套移除 + retrain_interval + sample_weight 对齐 + LGBM 停用）。
> 对齐范围：**框架结构**（特征选择/kfold/集成/超参选择机制/验证层），**不对齐选股池/标签/风控等策略层差异**。
> 状态：**仅方案，未改任何代码**。

---

## 一、现状扫描（代码实证，2026-08-19）

| 目录 | 三件套 | HP-Search | inner-OOS | IC准入移除 | 重训间隔 | 折集成扰动 | sample_weight对齐 | ensemble 同源 |
|---|---|---|---|---|---|---|---|---|
| **small_alpha191（标准）** | 已删 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| small_alpha191_test | 已删 | ✅ | ✅ | ✅ | ✅ | ❌ 缺 | ✅ | DIFF（差扰动+门控） |
| alpha191 | 在 | ❌ | ❌ | 部分 | ❌ | ❌ | ❌ | DIFF |
| alpha191_feat_test | 在 | ❌ | ❌ | 部分 | ❌ | ❌ | ❌ | DIFF |
| big_alpha191 | 在 | ❌ | ❌ | 部分 | ❌ | ❌ | ❌ | DIFF |
| cyb_alpha191 | 在 | ❌ | ❌ | 部分 | ❌ | ❌ | ❌ | DIFF |
| hs300_alpha191 | 在 | ❌ | ❌ | 部分 | ❌ | ❌ | ❌ | DIFF |
| kcb_alpha191 | 在 | ❌ | ❌ | 部分 | ❌ | ❌ | ❌ | DIFF |
| zz1000_alpha191 | 在 | ❌ | ❌ | 部分 | ❌ | ❌ | ❌ | DIFF |
| zz500_alpha191 | 在 | ❌ | ❌ | 部分 | ❌ | ❌ | ❌ | DIFF |
| small_alpha191_test2 | 在 | ❌ | ❌ | 部分 | ❌ | ❌ | ❌ | DIFF（**CNN 实验目录**） |

关键事实：
- 9 个未对齐目录的 `ensemble.py` **md5 完全一致**（63b91c599a，08-07 CFI 同步版），且 **import 无目录耦合**（纯第三方库）→ 可直接用 small 版本整体替换；
- 9 目录的 `strategy.py` **md5 全异**（类名/池配置不同）→ 需逐目录移植；
- "IC准入移除"列显示"部分"：`consensus_features=None` 在文件中有出现（签名默认/CFI 路径），但需核实调用处是否完整（对齐后随 ensemble 替换自动解决）；
- `small_alpha191_test2` 是 **CNN 实验目录**（cnn_ensemble.py，08-12 创建），框架含 CNN 定制——单独决策。

## 二、各策略规模（超参适配依据，日志实证）

| 策略 | 选股池 | 训练样本（240天窗，实测） | 现状超参 depth/lr/n_est |
|---|---|---|---|
| alpha191 | 全市场→候选300 | ~7.2万 | 6 / 0.03 / 600 |
| alpha191_feat_test | 同 alpha191 | ~7.2万 | 6 / 0.03 / 600 |
| big_alpha191 | 大盘300 | **72,000**（08-10日志） | 6 / 0.03 / 600 |
| hs300_alpha191 | hs300成分300 | **70,941**（08-11日志） | 6 / 0.03 / 600 |
| zz1000_alpha191 | 候选池300 | **72,000**（08-05日志） | 7 / 0.03 / 800 |
| zz500_alpha191 | zz500成分500 | **111,648**（08-04日志） | 6 / 0.03 / 600 |
| cyb_alpha191 | 创业板指~100 | **23,952**（08-07日志） | 4 / 0.03 / 400 |
| kcb_alpha191 | 科创50~50 | **11,632**（08-05日志） | 4 / 0.03 / 400 |
| small_alpha191 | 小盘300 | 72,000 | 3 / 0.05 / 300（S档） |
| small_alpha191_test | 小盘300 | 72,000 | 3 / 0.05 / 300 |

注：zz1000 的 `candidate_pool_size: 300` 限制使实际样本=7.2万（非 1000×240），depth=7 明显偏深。

## 三、待对齐点清单（9 个未对齐目录）

### ① ensemble.py → small 版本整体替换（一劳永逸）
直接复制 `small_alpha191/ensemble.py` 到 9 目录。一次性带入：
- `_select_hyperparams`（HP-Search A+B+C：隔离/多窗/早停/LRWarmup/AUC<0.5防护）
- PKF 折内 inner-OOS（MDA 迁移，验证块只评 AUC）
- P1-1 `consensus_features=None`（MDA 投票池=基础清理全集）
- `align_sample_weight_to_sort`（排序重排样本权重）
- train AUC 修复（`df_prod_train`）
- 折集成围绕选中超参扰动（depth=[d,d,d+1,d+1,d+2]）
- 门控口径统一（prod_val_auc = 折集成 meta_AUC）

**风险**：ensemble.py 引用 `self.config.XGB_*` 等属性——需确认 9 目录 strategy 的 `_create_model_config` 是否注入相同属性集（HP-Search 需要 `cfg.hyperparam_search`，inner-OOS 需要 `cfg.mda_inner_oos_frac/min_days`，隔离需要 `cfg.MAX_HOLD_DAYS`）→ 对应 ②③ 的注入。

### ② strategy.py 逐目录移植（5 处）
| # | 移植项 | 内容 |
|---|---|---|
| 2.1 | 三件套移除 | import 块 / `__init__` 初始化段（_val_pipeline 等）/ 训练前 run_rebalance 调用段 删除（对齐 small 08-18 已删的 3 处） |
| 2.2 | retrain_interval | `_should_retrain_model()` + 调仓主流程包裹（计数/判断/沿用/重置）+ `__init__` 参数 |
| 2.3 | MAX_HOLD_DAYS 注入 | `_create_model_config` 加 `cfg.MAX_HOLD_DAYS = getattr(self.meta_labeler, 'max_hold_days', 20)` |
| 2.4 | hyperparam_search 注入 | `_create_model_config` 加 `cfg.hyperparam_search = self.config.get('hyperparam_search', {})` |
| 2.5 | LGBM 停用适配 | `cfg.lightgbm` 已注入的保持；`_lgb_enabled` 读 config 自动为 false（随 config 改） |

**风险**：9 目录 strategy.py 结构可能不完全一致（md5 全异）——每个目录需单独定位锚点，用"锚点断言"防错。

### ③ config.yaml 每目录（4 处）
| # | 项 | 内容 |
|---|---|---|
| 3.1 | 删 validation 段 | 三件套配置段删除（对齐 small） |
| 3.2 | 加 hyperparam_search 段 | 按第三节档位模板 + 各策略现状锚点 |
| 3.3 | 加 retrain_interval | 默认与 small 现状一致（=1，见决策点3） |
| 3.4 | lightgbm 停用 | `enabled: true→false`，权重 `→1.0/0.0` |

### ④ factor_validator.py 删除（9 目录）
备份 `.bak_20260819_remove` 后删除（对齐 small 08-18 做法）。

## 四、超参候选模板（按实际样本分档）

HP-Search 候选组 = 覆盖该策略样本量下的合理深度区间，**c2 = 该策略现状配置（对照锚点）**。

**T1 档（~7.2万样本）：alpha191 / feat_test / big / hs300 / zz1000 /（small 系已配）**
沿用 small 模板结构，覆盖 depth 3-6：
| 候选 | depth | lr | n_est | ss | cs | λ | mcw | γ |
|---|---|---|---|---|---|---|---|---|
| c1 | 3 | 0.05 | 300 | 0.6 | 0.6 | 3.0 | 30 | 0.1 |
| c2(现状锚点) | **6** | **0.03** | **600** | 0.6 | 0.7 | 2.0 | 25 | 0.0 |
| c3 | 5 | 0.03 | 500 | 0.7 | 0.7 | 1.5 | 20 | 0.05 |
| c4 | 6 | 0.02 | 600 | 0.7 | 0.8 | 1.0 | 15 | 0.0 |

（c2 覆盖 7 万级策略现状 depth=6；c1 覆盖 small 系 depth=3——两系现状都在候选覆盖内）

**T2 档（~2.4万样本）：cyb**
| c1 | 3 | 0.04 | 300 | 0.6 | 0.6 | 3.0 | 30 | 0.1 |
| c2(现状) | **4** | **0.03** | **400** | 0.6 | 0.7 | 2.5 | 25 | 0.0 |
| c3 | 4 | 0.02 | 500 | 0.7 | 0.7 | 2.0 | 20 | 0.0 |
| c4 | 5 | 0.02 | 400 | 0.7 | 0.8 | 1.5 | 15 | 0.0 |

**T3 档（~1.2万样本）：kcb**
| c1 | 2 | 0.05 | 200 | 0.6 | 0.6 | 4.0 | 50 | 0.2 |
| c2(现状) | **4** | **0.03** | **400** | 0.6 | 0.7 | 3.0 | 40 | 0.0 |
| c3 | 3 | 0.03 | 300 | 0.7 | 0.7 | 2.5 | 30 | 0.1 |
| c4 | 5 | 0.02 | 300 | 0.7 | 0.8 | 2.0 | 25 | 0.0 |

**T4 档（~11万样本）：zz500**
| c1 | 5 | 0.04 | 500 | 0.7 | 0.7 | 1.5 | 15 | 0.0 |
| c2(现状) | **6** | **0.03** | **600** | 0.7 | 0.8 | 1.5 | 20 | 0.0 |
| c3 | 7 | 0.03 | 800 | 0.7 | 0.8 | 1.0 | 10 | 0.0 |
| c4 | 8 | 0.02 | 1000 | 0.75 | 0.85 | 0.8 | 8 | 0.0 |

评估协议（各档统一）：`inner_val_days: 20 / eval_windows: 3 / early_stopping_rounds: 20 / eval_rounds: 1200 / use_early_stop: true`（与 small/V10 一致）。

## 五、实施顺序与验证

1. **① ensemble.py 替换**（9 目录复制 → md5 校验一致 → import 冒烟）
2. **② strategy.py 移植**（逐目录 5 处，锚点断言防错）
3. **③④ config + factor_validator**（段增删 + 备份删除）
4. **专项测试** `scripts/test_alpha191_align.py`（每目录断言：ensemble md5=small / 无三件套残留 / HP/inner-OOS/扰动/RI 存在 / config 段齐全 / import 正常）
5. **回归**：small 五套件 + V10 专项不受影响

## 六、决策点（4 个）

1. **small_alpha191_test 是否补 A+C+D（折集成扰动+门控口径）**：test 之前明确"留作 AB 对照"（不同步 A+C+D）；本次"所有 alpha191 系对齐"若含 test，则 AB 对照失效。**建议：test 同步对齐**（框架统一，AB 对照可改用"策略层 alpha_095_FFD 开关"承担），请裁决。
2. **small_alpha191_test2（CNN 实验目录）是否纳入**：框架含 CNN 定制（cnn_ensemble.py/CNNModelManager），强对齐会破坏实验。**建议：本次排除**，仅在其 ensemble 层与 small 同源时顺带说明。请裁决。
3. **retrain_interval 默认值**：small 现状=1（每期重训）；V10 刚配 5。**建议各目录=1**（与 small 现状严格对齐，用户可按需改 5）。请确认。
4. **超参候选**：**建议按 T1-T4 分档模板**（每策略覆盖其样本量区间 + 现状锚点）；也可每策略独立设计（工作量大）。请确认。
