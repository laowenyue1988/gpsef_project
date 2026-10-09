# INCIDENT · `strategies/` 策略包文件缺失（2026-10-08 16:01）

> 发现时刻：**2026-10-08 16:18**（Q5BAR 烟测三臂同时秒退，报 `ModuleNotFoundError`）
> 严重级别：**高**（生产策略包不完整）｜当前状态：**运行时闭包已恢复，策略可运行**；非运行时文件仍缺

---

## 一、事实

`strategies/` 目录在 **16:01** 发生结构变动，现存仅 5 个子目录：

```
alpha191/  fundamental_ml/  grid_lowvol/  small_alpha191/  wyckoff20/
```

**消失**：
| 项 | 说明 | 备注 |
|---|---|---|
| `strategies/small_alpha191_test3/` | **整个目录** | 实验源包（D-342 的 B 臂） |
| `strategies/anchoreq_small191/` | **整个目录** | D-342 临时建的锚代际对照包（C 臂） |
| `strategies/small_alpha191.bak_preFinalize_20261008_1329/` | **整个目录**（40 文件） | D-342 落码前的全量备份 |
| `strategies/small_alpha191/` 内的全部 `.bak_*` | 若干 | 含 `strategy.py.bak_preOPT_v2_20260924_0538`（**锚代际**）、`ensemble.py.bak_preQ5TbC_20260927_1856`（**锚代际**） |

**`strategies/small_alpha191/` 现只剩 12 个 `.py`**（D-342 时为 **26 个**）：

| 现存 12 | 缺失 14 |
|---|---|
| `__init__` `barriers` `benchmark` `data_utils` `ensemble` `factors` `garch_position` `labels` `snapshot` `snapshot_fast` `snapshots` `strategy` | **`determinism_provenance`**★ · **`panel_engine`**▲ · **`target_transforms`**▲ · `e1_counterfactual` · 10×`test_*.py`（`test_audit_fixes`/`test_determinism`/`test_e3_changes`/`test_e4_revert_check`/`test_exdiv_scale_fix`/`test_garch_position_audit`/`test_intraday_event_driven`/`test_profit_lock_param`/`test_pt_shift_stale_cum`/`test_snapshot_fingerprint`） |

★ = **回测运行时必需**；▲ = 功能模块（条件/离线路径）；其余 = 测试与分析脚本

---

## 二、责任认定：**非本助手的归档操作**

| 证据 | 结论 |
|---|---|
| 归档 `MANIFEST.csv`（1,765 条）中 `strategies/small_alpha191_test3` 命中 **0**、`determinism_provenance` 命中 **0** | 归档脚本**从未搬运**这些文件 |
| 归档 `README.md` L137 明文：「`strategies/`、`gpsef/` **未动**：整理时正有 3 个等价性测试臂在运行…任何改动都可能破坏在跑的验证」 | 归档时已显式保护 |
| `strategies/` 目录 mtime = **16:01**，而归档操作在 **14:10** 完成 | **时间窗不符** |

⇒ **归档/清理操作与本次缺失无关。** 用户在 16:0x 明确表示「**我手动删除了这些脚本**」。

## 三、时间线

| 时刻 | 事件 |
|---|---|
| 13:29 | D-342 建 `small_alpha191.bak_preFinalize_20261008_1329`（40 文件） |
| 13:50~15:03 | D-342 三臂（A=`small_alpha191`／B=`small_alpha191_test3`／C=`anchoreq_small191`）**全部正常跑完** ⇒ 三包当时**完整** |
| **15:34** | `determinism_provenance.cpython-313.pyc` **最后一次编译** ⇒ 该模块当时**仍存在** |
| **16:01** | `strategies/` 目录 mtime 变动 ⇒ **缺失发生窗口 15:34~16:01** |
| 16:08 | 本助手备份并编辑 `strategy.py`（G19），改动 `small_alpha191/` mtime |
| 16:12 | Q5BAR 三臂起臂 |
| 16:18 | 三臂**同时**秒退：`strategy.py:1069` `from .determinism_provenance import files_fingerprint` ⇒ **`ModuleNotFoundError`**（**fail-fast，无静默降级** ⇒ 行为正确） |

## 四、影响评估

### 4.1 对 Q5BAR 实验（波 5）
- **阻塞 → 已解除**：`determinism_provenance.pyc` 已恢复，`strategy.py` 运行时依赖闭包 **完整**
- 运行时闭包（实测 import 全 OK）：`ensemble`／`labels`／`data_utils`／`garch_position`／`determinism_provenance`
- `panel_engine`／`target_transforms` **不在采纳配置的回测路径**：
  - `panel_engine` 仅被 `snapshot_fast.py`（**离线快照生成脚本**）顶层 import
  - `target_transforms` 仅在 `strategy.py` 的 `if self._target_transform != 'none'` 分支 import（采纳配置为 `none` ⇒ 不触发）

### 4.2 数据可复现性
- `strategies/small_alpha191_test3/` 消失 ⇒ **D-342 的 B 臂不可再复现**（其结论依赖当时的逐字节比对记录，记录本身完整保留在 `D-342` 档）
- `anchoreq_small191/` 消失 ⇒ **C 臂不可再复现**；其两枚锚代际文件（`dbd340bf`／`062f45a1`）的**源副本也已随之消失**
- ⇒ **锚代际字节码现已无任何副本**（`strategy.py.bak_preOPT_v2_*` 与 `ensemble.py.bak_preQ5TbC_*` 同时丢失）

### 4.3 功能完整性
- `panel_engine.py` / `target_transforms.py` 缺失 ⇒ 若将来切换 `target_transform` 或重新生成因子快照，**会失败**

## 五、恢复处置（已完成 / 可做）

| 项 | 来源 | 状态 |
|---|---|---|
| **`determinism_provenance`** | `__pycache__/determinism_provenance.cpython-313.pyc`（**源码 mtime 2026-09-10 16:39:07、19,315 B**，pyc 于 15:34 编译） | ✅ **已恢复**：复制为 `strategies/small_alpha191/determinism_provenance.pyc`（**sourceless import**，字节码与原版**逐位一致**）。实测 8 个 API 全在（`FINGERPRINT_VERSION=3`、`FileLockTimeout`、`file_lock`、`files_fingerprint`、`md5_file`、`merge_pool_cache`、`pool_cache_fingerprint`、`rank_pool_codes`） |
| `panel_engine.py` | 归档 `root_dirs/backups/backup_small_alpha191_test3_full_20260907_002201/strategies_small_alpha191_test3/`（**09-07 版，可能过旧**） | ⏸ 待用户决定 |
| `target_transforms.py` | **无任何来源** | ⏸ 待用户决定 |
| 10×`test_*.py`／`e1_counterfactual.py` | **无任何来源** | ⏸ 待用户决定（非必需） |
| 回收站 | 已验：**`$R` 实体 0 个**（不可恢复） | ✗ |
| `.idea` / `.workbuddy` / 两个老备份（`gpsef_project_backup_20260529_215159`、`_perf`） | 已验：**均无该文件** | ✗ |

**★sourceless 模块的等价性说明**：`.pyc` 为该模块**原始编译产物**（非重建），故**行为逐位等价**；但①**不可读/不可编辑**，②若 Python 主版本升级（3.13→3.14）将失效。**建议在用户确认原始文件确已丢失后，由本助手从字节码重建等价 `.py` 源码**（可用「编译回来比对字节码」做等价性证明）。

## 六、对结论的影响

- **Q5BAR 零影响验证**：**不受影响** —— 恢复的是**原字节码**；判据（Z1 臂 vs D-342 A 臂逐字节）仍然成立，将在重跑后给出
- **历史结论**：`D-337`/`D-338`/`D-342`/`D-344` 等**均不受影响**（其依赖的产物与记录完整）
- **待办**：①请求用户确认删除意图与是否需要重建缺失文件；②**启用冷备**（本助手此前已建议 `robocopy /E /COPYALL`，尚未执行）

## 七、教训（建议入库）

1. **★策略包完整性必须显式门控**：起臂前不仅校验配置与代际，还应校验**运行时依赖闭包**是否齐全（本次三臂是"起臂后 6 分钟才发现"，应在起臂**前** 5 秒的导入预检里覆盖 —— 我的预检只 import 了 `strategy`，而 `determinism_provenance` 是**函数内 import**，顶层预检漏掉）。
2. **函数内 import 是包完整性门控的盲区**（本次的直接原因）。
3. **删除外部文件后应立即有冷备**，否则锚代际字节码这类"一次性证据"会永久丢失。
