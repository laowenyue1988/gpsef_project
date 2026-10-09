# small_alpha191_test2 — CNN 元标注策略完整文档

- 日期：2026-08-12
- 策略目录：`strategies/small_alpha191_test2/`
- 核心文件：`strategy.py`（回测集成）、`cnn_ensemble.py`（CNN 模型/训练/预测）、`config.yaml`

---

## 一、设计概要（用户要求 → 实现映射）

| 用户要求 | 实现 |
|---|---|
| 特征固定为 `results/20260811_time_seg_100pc/plots` 的 47 个特征 | `cnn_ensemble.FEATURES_47`（30 基础 + 17 score），**含 alpha_095_FFD**（用户确认不排除）；快照已验证 30 基础列全覆盖、17 score 基础列全覆盖 |
| 不做任何预筛选 | 策略运行期**无 IC/PSI/MDA/聚类/三件套验证**；特征集每期固定 47 个 |
| 47 特征中性化（市值+行业）+ 原始特征并集 | 复用 `_neutralize_factors`（行业哑变量 + 对数市值 OLS 残差）→ 47 个 `_neu` 列；**输入 = 47 原始 + 47 中性化 = 94 维**（固定顺序） |
| 每次调仓重训前 purged-k-fold（5 折/240 天）确定网络结构与超参 | `CNNModelManager.search_structure`：5 折 Purged（embargo=21），**嵌套**——每折训练部分内部再切 inner-val（尾部 15%/≥20 天），候选在 train_sub 训练、inner-val 早停评估，**验证块不参与选择**（Cawley-Talbot 嵌套原则） |
| 全量数据训练 | `CNNModelManager.train_full`：最优结构 + 全部 240 天数据重训（尾部 inner-val 早停），保存 `meta_models/cnn_model.pt` |
| 标签和退出机制保持当前设计 | 完全复用：`staged_binary` 三重屏障标签（`labels.py`）、分阶段移动止盈止损 S1~S4（`strategy.py`）、AUC 门控（0.52） |

## 二、特征流程（训练/预测同构，防泄漏）

```
快照 30 基础列 → 现算 17 score 列 (score = tanh(base))
  → 47 特征 → 市值+行业中性化 (OLS 残差, 训练期 ref_date=None 重新拟合)
  → 47 个 _neu 列
  → 输入 94 维 = FEATURES_47(47) + [_neu × 47]   ← 固定顺序
  → 标准化 (train_sub fit mean/std, 预测用同一参数; NaN→特征均值→0)
  → 1D CNN
```

**防泄漏要点**：
- 中性化 OLS 系数在训练期 fit（ref_date=None）、预测期 transform（ref_date=t_minus_1）；
- 标准化 mean/std 只在训练子集 fit，预测复用；
- 嵌套 purged-kfold：超参选择只用训练部分内部，验证块仅做最终评估（meta_auc 无选择偏倚）；
- score 由原始列现算（不依赖未来数据）。

## 三、CNN 架构（1D CNN 沿特征维）

```
输入 (B, 1, 94)
  → Conv1d(1→C1, k1, pad=k1//2) + BatchNorm + ReLU + Dropout
  → Conv1d(C1→C2, k2, pad=k2//2) + BatchNorm + ReLU + Dropout     (depth 层)
  → AdaptiveGlobalAvgPool1d(1)
  → Linear(C_last→16) + ReLU + Dropout
  → Linear(16→1) → logit → BCEWithLogitsLoss(pos_weight=负/正样本比)
优化器: Adam(lr, weight_decay=1e-4) | 早停: inner-val AUC patience=5
```

**候选网格（6 组，batch=1024 CPU 最优）**：
| # | depth | channels | kernels | dropout | lr |
|---|---|---|---|---|---|
| 1 | 2 | (32,16) | (5,3) | 0.3 | 1e-3 |
| 2 | 2 | (32,16) | (5,3) | 0.5 | 5e-4 |
| 3 | 2 | (64,32) | (5,3) | 0.3 | 1e-3 |
| 4 | 3 | (32,16,8) | (5,3,3) | 0.3 | 1e-3 |
| 5 | 3 | (32,16,8) | (5,3,3) | 0.5 | 5e-4 |
| 6 | 3 | (64,32,16) | (3,3,3) | 0.4 | 1e-3 |

设计依据：每期 ~5.7 万样本（240 天 × ~300 股），**小网络 + 强 dropout + 早停**防过拟合；CPU 训练（无 GPU）以 batch=1024 为最优（实测 6.9s/epoch@50K 样本）。

## 四、嵌套 Purged K-Fold 超参搜索（核心）

```
240 天训练窗口（一次历史回放+标签+中性化）
  → 5 折 Purged K-Fold (时间有序分块, embargo=21, Prado AFML §7.4)
  → 每折: 训练部分 → train_sub + inner-val(尾部 15%)
       候选在 train_sub 训练 (60% 子采样粗筛, inner-val 早停)
       → inner-val AUC（用于选择）
       → 验证块 AUC（仅记录, 不用于选择）
  → 最优候选 = 6 组中 5 折 inner-val AUC 均值最高者
  → meta_auc（门控）= 最优候选在 5 折验证块上的平均 AUC（真正 OOS）
  → 全量重训: 最优候选 + 全部训练数据 (30 epoch, inner-val 早停)
```

**计算量**（实测 CPU）：搜索 ~17 分钟 + 全量 ~3-4 分钟 ≈ **~21 分钟/期**，47 期 ≈ 16 小时（后台跑）。

## 五、配置文件（`config.yaml` 关键段）

```yaml
strategy:
  name: SmallAlpha191Test2
  class: strategies.small_alpha191_test2.strategy.SmallAlpha191Strategy

cnn:                              # [CNN] 1D CNN 元标注模型
  cv_folds: 5                      # 嵌套 Purged K-Fold 折数
  cv_embargo_days: 21              # 验证块前后 purge 天数 (max_hold=20)
  inner_frac: 0.15                 # 折内 inner-val 段占训练部分比例
  inner_min_days: 20               # inner-val 最少天数
  search_epochs: 8                 # 搜索阶段粗训 epoch (60% 子采样, 早停)
  full_epochs: 30                  # 全量重训最大 epoch (早停)
  candidates: 6                    # 结构候选数 (default_candidates)
```

其余（回测区间 20220801~20260522、标签 staged_binary、分阶段退出、AUC 门控 0.52、max_meta_selected=20 等）与生产 small_alpha191 一致，未改动。

## 六、测试总结（`test_cnn_test2.py`，36/36 通过）

| 测试 | 内容 | 结果 |
|---|---|---|
| T1 | 47 特征冻结清单（含 alpha_095_FFD / 30+17 / score 映射 / 与 plots 目录一致） | ✅ |
| T2 | score 现算正确（tanh 口径、NaN 填充） | ✅ |
| T3 | 94 维构造（47 原始+47 中性化、固定顺序、无重复） | ✅ |
| T4 | FactorCNN 1D 前向（depth 2/3 均正常） | ✅ |
| T5 | 嵌套 Purged K-Fold 搜索（返回最优结构、inner AUC 有效、验证块隔离） | ✅ |
| T6 | 全量重训 + 模型保存 + 预测闭环（形状/概率域/selected_features/meta_auc） | ✅ |
| T7 | 真实快照 47 特征全覆盖（30 基础 + 17 score 基础列） | ✅ |
| T8 | strategy.py 集成点（AST：CNN import/搜索/重训/预测存在，无 feature_cols_neu/v10/三件套残留） | ✅ |
| 附加 | 真实快照 → 94 维端到端构建（5198 股 × 94 维，NaN 2.15% 已处理） | ✅ |
| 附加 | CPU 训练测速（ch(32,16)+bs1024=6.9s/epoch；搜索 60% 子采样优化后 21 分钟/期） | ✅ |

**环境**：PyTorch 2.13.0+cpu 已安装（安装时 setuptools 被误升 78 破坏 gm，已降回 69.1.1 恢复 gm + torch 正常共存）。

## 七、运行

```bash
python run_backtest.py --config strategies/small_alpha191_test2/config.yaml --output results/SmallAlpha191Test2_20260812
```

预计耗时 ~16 小时（47 期 × ~21 分钟），建议后台运行。

## 八、已知边界（诚实说明）

1. **CNN 与 tabular 数据**：1D CNN 对无序因子特征的归纳偏置弱于 GBDT，这是用户明确选择的架构；嵌套搜索会选出相对最优的候选，但绝对水平依赖特征本身信息量；
2. **搜索子采样**：搜索阶段 60% 子采样用于快速排序候选（业界粗筛），最终全量重训不受影响；
3. **特征窗口二阶重叠**：与 P1 分析相同的 purged CV 通用边界（embargo 只防标签窗口，不防特征回看窗口）——与生产框架一致；
4. **torch 线程**：训练用 `torch.set_num_threads(os.cpu_count())`，与回测其他阶段共享 CPU；
5. **score 口径**：与 plots 分析一致（tanh(原始列)），若后续对比发现口径差异需同步调整。

## 九、文件清单

- `strategies/small_alpha191_test2/strategy.py`（CNN 集成改造，备份 `strategy.py.bak_20260812_cnn`）
- `strategies/small_alpha191_test2/cnn_ensemble.py`（新增：模型/搜索/重训/预测）
- `strategies/small_alpha191_test2/config.yaml`（test2 身份 + cnn 段）
- `strategies/small_alpha191_test2/features_47.json`（47 特征清单）
- `test_cnn_test2.py`（36 项测试）
