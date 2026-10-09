# SmallAlpha191Test2 方案A优化实施手册（修复说明 + 测试说明 + 回测实验建议）

**日期**: 2026-08-29
**状态**: 代码修改与测试全部完成（45/45 PASS），待用户手动实施回测
**原则**: 以回测绩效为唯一裁决标准；每模块独立 A/B，避免叠加后无法归因

---

## 1. 覆盖操作说明（test → test2）

已完成，操作记录如下：

| 步骤 | 命令 | 结果 |
|---|---|---|
| 备份 test2（CNN 旧版） | `mkdir -p backups/small_alpha191_test2.bak_20260829_cnn && cp -r strategies/small_alpha191_test2/. backups/small_alpha191_test2.bak_20260829_cnn/` | ✅ 完整备份 |
| 覆盖 test2 | `rm -rf strategies/small_alpha191_test2/ && mkdir -p strategies/small_alpha191_test2 && cp strategies/small_alpha191_test/*.py strategies/small_alpha191_test2/ && cp strategies/small_alpha191_test/config.yaml strategies/small_alpha191_test2/config.yaml` | ✅ 13 个 .py + config.yaml |
| 完整性校验 | `diff <(ls strategies/small_alpha191_test/) <(ls strategies/small_alpha191_test2/)` | ✅ 仅差 .bak 备份文件（有意未拷） |
| import 验证 | `from strategies.small_alpha191_test2.strategy import SmallAlpha191Strategy` | ✅ class OK |

**注意**: test 目录中的 `*.bak_*` 备份文件未拷贝（`cp *.py` 模式），不影响运行。test2 的 `__pycache__` 已重建。

---

## 2. 方案A四项优化修复说明

### 2.1 改动文件清单

| 文件 | 改动 |
|---|---|
| `strategies/small_alpha191_test2/strategy.py` | 新增 4 个方法 + 3 处调用注入 + 1 处 bug 修复（峰值重置） |
| `strategies/small_alpha191_test2/config.yaml` | 策略名/class 改 test2；AUC 门控 0.52→0.54；S1 2.0σ→2.5σ；新增 drawdown_breaker / rebalance_trend_gate 两段 |
| `scripts/test_small_alpha191_test2_riskmgr.py` | **新增**，45 项风控专项测试 |

### 2.2 方案A-1：组合级回撤熔断（drawdown_breaker）

**代码位置**: `strategy.py` L577-592（initialize 配置加载）、L3292-3298（盘前状态更新）、L3555-3557（日终跟踪）、L3561-3620（方法区）

**状态机**: `0=NORMAL(建仓x1.0) → 1=HALF(建仓x0.5) → 2=FLAT(建仓x0, 清空目标)`

```
触发（保守优先）:
  回撤 > 15% → FLAT（仅当不在 FLAT，防止重复锁死）
  回撤 > 10% → HALF（仅当 NORMAL，HALF 中加深仍升级 FLAT）
恢复:
  FLAT→HALF: 指数 ≥ MA20 连续 3 日确认 + 峰值重置（回撤基线归零重新起算）
  HALF→NORMAL: 回撤收窄至 <5%（相对重置后基线，半仓净值会动，可达）
```

**关键设计决策（含历史教训）**:
1. **只限建仓、不强清**——FLAT 只清空"本次调仓目标"（不建仓），不动已有持仓。grid_lowvol 教训：风控=限建仓非强清，强清会锁死恢复。
2. **恢复用市场信号，不用净值**——空仓时净值恒定，任何"回撤收窄/净值回升"型恢复条件都会死锁（grid_lowvol 2026-08-22 教训，已两次踩坑）。FLAT→HALF 必须靠指数趋势确认。
3. **峰值重置 = 直接赋值**（`_dd_peak = self._prev_nav`）——**本次实施中发现并修复的 bug**：初版用 `max(peak, prev)` 在 peak>prev 时不重置，FLAT→HALF 后 dd 残留 -16%，次日必回 FLAT（振荡死锁）。测试 `振荡回归` 用例抓出，已修为直接赋值（以恢复入场点净值为新基线）。
4. **触发守卫 + 计数清零**——FLAT 触发时 `_dd_above_days=0`（防清仓次日即恢复，grid_lowvol 2024-04-12 教训）；已在 FLAT 不重复触发。
5. **浮点 epsilon**——`dd <= -threshold + 1e-9`，防 `85/100-1 = -0.1499...` 边界误判（grid_lowvol 同款修复）。
6. **无未来**——回撤用 T-1 净值（日终更新，盘前计算）；指数用 `< date` 严格 T-1。

**参数**（config.yaml `drawdown_breaker` 段）:
| 参数 | 值 | 含义 |
|---|---|---|
| enabled | true | 总开关 |
| dd_half | 0.10 | 回撤 >10% → 建仓减半 |
| dd_flat | 0.15 | 回撤 >15% → 不建仓 |
| dd_recover_full | 0.05 | HALF→NORMAL 回撤收窄 <5% |
| trend_confirm_days | 3 | FLAT→HALF 指数确认天数 |
| index_code | sh000001 | 趋势确认指数（上证综指） |
| index_ma | 20 | 确认均线周期 |

### 2.3 方案A-2：调仓日趋势过滤（rebalance_trend_gate）

**代码位置**: `strategy.py` L599-606（配置）、L3500（handle_bar 调仓日）、L3622-3631（方法）

- 调仓日盘前判断：上证综指 T-1 收盘 < MA20 → 当日建仓规模 ×0.5
- 只影响当日建仓，不清仓不干预网格
- **实测依据**: 2024-01-22 调仓日满仓接盘 -6.4%（143844 回测日志实证）——调仓日遇指数破位时减半建仓

**参数**: enabled/index_code(sh000001)/ma(20)/half_position(true)

### 2.4 方案A-3：S1 阈值 2.0σ → 2.5σ

**代码位置**: config.yaml `staged_exit.stage1`，pt_sigma 2.0→2.5，sl_sigma -2.0→-2.5

- **理论依据**: 微盘股日波动 3-5%，20 天持有窗口 P(触及2σ)≈99.9%——2σ 是必然事件而非止盈信号（143844 实证：S1 触发 1746 次，占止损止盈 2935 次的 59%）。提至 2.5σ（≈5.5-6% at 中位波动）恢复选择性，减少过度止盈导致的被动低仓位（41.6%）。

### 2.5 方案A-4：AUC 门控 0.52 → 0.54

**代码位置**: config.yaml `auc_threshold`

- 低 AUC 期（预测能力弱）减少无效暴露。143844 门控口径存在"折集成加权 0.5606 vs CV 集成 0.5143"双口径不一致，本次收紧到 0.54 只收严不放松。

---

## 3. 测试说明

### 3.1 运行方式

```bash
cd E:\quant\gpsef_project
.venv/Scripts/python.exe scripts/test_small_alpha191_test2_riskmgr.py
```

### 3.2 测试覆盖矩阵（45 项）

**T-回撤熔断状态机（14 项）**——AST 提取 `_update_dd_breaker` 真实源码 + SimpleNamespace mock self（项目测试约定，不 import 重依赖）:
| 类别 | 用例 |
|---|---|
| 触发 | -16%→FLAT(scale 0)；-12%→HALF(scale 0.5)；-5%→NORMAL |
| 边界 epsilon | -15.0001%→FLAT；-14.9999%→HALF（不误判） |
| **死锁回归（grid_lowvol 教训）** | FLAT 中 dd 恒定-16% 指数确认未满→保持 FLAT；确认满 3 日→恢复 HALF + 峰值重置 + 计数清零 |
| **振荡回归（本次新发现的 bug）** | 峰值重置后净值-7%→保持 HALF 不回 FLAT（max 实现残留-16% 会振荡） |
| 分层恢复 | HALF 回撤收窄-3.5%→NORMAL；-6% 未收窄→保持（滞后带） |
| 升级 | HALF 回撤加深-20%→FLAT + 计数清零 |
| disabled | 强制 NORMAL scale 1.0 |

**T-趋势过滤（4 项）**: 指数<MA→x0.5；≥MA→x1.0；disabled→x1.0；half_position=false→x1.0

**T-build_scale 组合（4 项）**: 0.5×0.5=0.25；1.0×1.0=1.0；0.0×0.5=0.0（FLAT 一票否决）；0.5×1.0=0.5

**T-AST 源码断言（11 项）**: handle_bar 建仓应用 `target_w * build_scale`；调仓日先更新趋势门；日终回撤跟踪；盘前每日更新状态机；FLAT 清空目标；initialize 加载双配置；FLAT 计数清零；恢复用市场信号；峰值重置直接赋值（且无 max 残留）；指数 ≤T-1 无未来；指数加载路径

**T-config.yaml（6 项）**: 策略名/class 指 test2；auc 0.54；S1 2.5/-2.5；drawdown_breaker 段 6 参数；rebalance_trend_gate 段 4 参数

**T-指数数据可用性（3 项）**: sh000001.parquet 存在；date/close 列；数据量 >500

### 3.3 测试方法说明

- `load_method()`: AST 提取类方法真实源码 → exec 编译（globals 提供 logger/pd/np/os/datetime/timedelta/StrategyContext dummy）→ 返回函数 → `__get__(self)` 绑定调用。**测的是生产代码本身，不是复制品**。
- mock self 用 `types.SimpleNamespace`，只提供方法引用的属性（_dd_cfg/_dd_state/_dd_scale/_dd_above_days/_prev_nav/_dd_peak/_trend_gate_cfg/_trend_scale/_breaker_index_down）。

### 3.4 测试结果

```
结果: 45 PASS, 0 FAIL
```

---

## 4. 回测实验完整建议（12 组敏感性格栅）

### 4.1 实验基础

- **统一回测窗口**: **2022-08-01 ~ 2024-03-01（390 交易日，用户已手动修改 config.yaml 生效）**
- **⚠️ 与旧基线不可直接比**: 143844 窗口为 2022-08-01~2026-05-22（含 2025 年小盘牛市），新区间截止 2024-03-01，**缺失 2025 年数据**——143844 的 Sharpe 1.42/仓位 41.6%/回撤 -15.97% 只作参考不作对比基准；**组间对比以 R0 为唯一基准**。
- **新区间特性（实验设计重点利用）**: 覆盖 2022 熊市尾段 + 2023 震荡市 + **2024-01/02 微盘危机（在区间内！）**——熔断参数区分度最大的场景就在区间里，"2024-02 红绿灯"验证可直接执行；2025 年行情无法检验，结论外推性受限（外推到牛市需后续回测）。
- **新基线 R0**: 当前 test2 默认配置（方案A 四项全开）——所有格栅相对 R0 单因子扰动。

### 4.2 格栅顺序与理论依据

**阶段一：熔断阈值灵敏度（R1-R2）——影响最直接，先跑**
| 编号 | 修改 | 理论预期 |
|---|---|---|
| R0 | 无（四项全开基线） | 仓位 41.6%→50%+（S1 2.5σ 少卖）、回撤 -15.97%→-13%±、Sharpe 1.42±0.1 |
| R1 | dd_half 0.10→0.08, dd_flat 0.15→0.12 | 熔断更灵敏→回撤更浅，但降仓更频繁→仓位略降（探下限） |
| R2 | dd_half 0.10→0.12, dd_flat 0.15→0.18 | 熔断更宽松→仓位更高收益更高，回撤可能加深（探上限） |

**阶段二：恢复参数（R3-R4）——直接决定空仓期长短（用户核心诉求：减少踏空）**
| 编号 | 修改 | 理论预期 |
|---|---|---|
| R3 | trend_confirm_days 3→2 | FLAT→HALF 更快→空仓期缩短、踏空减少；代价：假突破过滤减弱 |
| R4 | dd_recover_full 0.05→0.03 | HALF→NORMAL 更谨慎（净值需涨更多才满仓）；与"缩短空仓期"方向相反，验证宽松度代价 |

**阶段三：趋势过滤 MA（R5-R6）——治理调仓日接刀**
| 编号 | 修改 | 理论预期 |
|---|---|---|
| R5 | index_ma 20→10 | 2024-01-22 类接盘日更多被过滤（破位 10 日即减半），但误杀反弹日 |
| R6 | index_ma 20→30 | 更钝，只过滤深度破位，触发次数少 |

**阶段四：S1 σ（R7-R8）——交易层核心参数**
| 编号 | 修改 | 理论预期 |
|---|---|---|
| R7 | pt_sigma 2.5→2.2, sl_sigma -2.5→-2.2 | 折中：止盈略早于 2.5σ，验证 2.5 是否最优 |
| R8 | pt_sigma 2.5→3.0, sl_sigma -2.5→-3.0 | 更宽松：长尾利润留到 S4(9σ)，仓位更高；风险：浮亏时间更长 |

**阶段五：AUC 门控（R9-R10）——低频参数**
| 编号 | 修改 | 理论预期 |
|---|---|---|
| R9 | auc_threshold 0.54→0.52 | 对照旧值，验证 A4 收紧是否有效（预期 R0 优于 R9） |
| R10 | auc_threshold 0.54→0.56 | 更严格→空仓更多，若 R0/R10 差异小说明门控不敏感 |

**阶段六：组合验证（R11-R12）——从单因子最优组合**
| 编号 | 修改 | 理论预期 |
|---|---|---|
| R11 | R1-R10 每维度最优值叠加 | 各维度最优组合，预期 Sharpe 最高（叠加可能非线性，需验证） |
| R12 | 稳定性检验：R11 的次优组合（+1 个参数取次优值） | 确认最优组合不是单点过拟合；两组合差异 <0.05 Sharpe 则稳健 |

### 4.2A 每组包含的实验步骤（分解明细）

**每组实验 = 5 个步骤**（运行 → 采集 → 验证 → 裁决 → 记录），全部自动化脚本执行：

1. **运行回测**：`python -u run_backtest.py --config strategies/small_alpha191_test2/config_R{n}.yaml > results/R{n}.log 2>&1`，后台执行；每组独立日志与结果目录（`results/SmallAlpha191Test2_20220801_20240301_<时间戳>/`），失败不影响后续组（断点续跑）。
2. **结果采集**（脚本读取 5 类文件）：
   - `report.txt` → 总收益/年化/Sharpe/最大回撤/胜率/换手率
   - `daily_values.csv` → 分年度收益（2022/2023/2024-01~02）、平均仓位、空仓天数、最大回撤区间定位
   - `orders.csv` → S1 触发数（对比 143844 的 1746 次）、预警清仓数、熔断订单（combo_flat/combo_half）、建仓笔数/次调仓
   - `attribution/` → Active 超额/特质 alpha/规模/价值/波动率暴露贡献
   - `backtest.log` → `[方案A 回撤熔断]` 触发次数（FLAT/HALF 分列）、趋势过滤触发次数
3. **验证点核对**（每组必须回答的问题）：
   - [ ] 熔断触发次数：FLAT 几次、HALF 几次（对比 R0）
   - [ ] 2024-02 微盘危机期间组合回撤（红绿灯：> -8% 标风险；**统一口径 = 2024-01-15~02-29 窗内相对峰值回撤，脚本测算 143844 同期约 -8.7%**）
   - [ ] 2024-01-22 调仓日是否触发趋势过滤（指数<MA20 → 建仓减半日志）
   - [ ] 空仓天数 vs R0（缩短空仓期目标）
   - [ ] S1 触发次数 vs R0（2.5σ 是否降频）
   - [ ] 平均仓位 vs R0（仓位利用率）
4. **裁决**（阶段内对比，仅相邻组）：
   - 主指标 Sharpe；次指标 收益/回撤比；显著检验用 ab_sig_test（390 日，P<0.05）
   - 阶段内三组（如 R0/R1/R2）对比后定方向，无效方向砍掉后续组
5. **记录**：写入 `results/GridLowVol_*` 同级的格栅总表 `results/small_alpha191_test2_grid_summary.csv`（每组一行：参数、收益、Sharpe、回撤、仓位、空仓天数、熔断次数），每组完成后追加。

**阶段规划与裁决点**（每批完成后汇报，用户确认后继续下一批）：
| 批次 | 组 | 裁决内容 |
|---|---|---|
| 批 0（试运行） | R0 + R1 | 验证执行链路 + 建新基线；确认分析质量 |
| 批 1 | R2 | 与 R0/R1 三组对比，定熔断方向（灵敏/宽松） |
| 批 2 | R3 + R4 | 恢复参数方向（缩短空仓期是否有效） |
| 批 3 | R5 + R6 | 趋势过滤 MA 灵敏度 |
| 批 4 | R7 + R8 | S1 σ 最优值 |
| 批 5 | R9 + R10 | 门控敏感性（低频，若 R0 差异小可合并） |
| 批 6 | R11 + R12 | 组合 + 稳定性 |

**耗时预估**：390 日区间约 5-6h/组（143844 为 920 日 14.5h 含 O1+，无 O1+ 按日线性折算），12 组约 60-72h；批 0 试运行约 10-12h。

### 4.3 操作指导（每个格栅完整命令行）

**⚠️ 本机为 PowerShell 环境（`(.venv) PS>`），以下命令均为 PS 可直接运行的写法**：
- 配置复制用 `Copy-Item`（PS 原生命令）；参数修改用 **python 文本替换**（`python -c`，字面量 replace，保留 UTF-8 中文注释与格式，比 sed 安全）；运行用 `python`（venv 已激活）。
- 每条命令单独一行、无换行，逐行复制执行；**命令中不含 `#` 注释行**（PS 中 `#` 后跟空格会误解析）。
- 结果目录由框架自动生成（`results/SmallAlpha191Test2_20220801_20260522_<时间戳>/`），log 输出到 `results/R{n}.log`。
- 若 `python` 提示找不到，改用 `./.venv/Scripts/python.exe`（PS 中 `.venv` 开头必须加 `./` 前缀）。

**通用模板**（每格栅 3 条命令：复制 → 替换 → 运行）：
```powershell
Copy-Item strategies/small_alpha191_test2/config.yaml strategies/small_alpha191_test2/config_R{n}.yaml
python -c "import io;p='strategies/small_alpha191_test2/config_R{n}.yaml';s=io.open(p,encoding='utf-8').read();s=s.replace('旧值','新值').replace('旧值','新值');io.open(p,'w',encoding='utf-8').write(s)"
python -u run_backtest.py --config strategies/small_alpha191_test2/config_R{n}.yaml > results/R{n}.log 2>&1
```

**替换映射表**（replace 的「旧值→新值」字符串对，已实测全部唯一命中、无残留、不误伤其他参数）：
| 格栅 | replace 字符串对 |
|---|---|
| R1 | `dd_half: 0.10`→`dd_half: 0.08`；`dd_flat: 0.15`→`dd_flat: 0.12` |
| R2 | `dd_half: 0.10`→`dd_half: 0.12`；`dd_flat: 0.15`→`dd_flat: 0.18` |
| R3 | `trend_confirm_days: 3`→`trend_confirm_days: 2` |
| R4 | `dd_recover_full: 0.05`→`dd_recover_full: 0.03` |
| R5 | `index_ma: 20`→`index_ma: 10`（仅 drawdown_breaker 段一处；rebalance_trend_gate 段字段是 `ma`，不会误改） |
| R6 | `index_ma: 20`→`index_ma: 30` |
| R7 | `pt_sigma: 2.5`→`pt_sigma: 2.2`；`sl_sigma: -2.5`→`sl_sigma: -2.2`（均仅 stage1 一处） |
| R8 | `pt_sigma: 2.5`→`pt_sigma: 3.0`；`sl_sigma: -2.5`→`sl_sigma: -3.0` |
| R9 | `auc_threshold: 0.54`→`auc_threshold: 0.52` |
| R10 | `auc_threshold: 0.54`→`auc_threshold: 0.56` |
| R11/R12 | 组合（见下，按 R1-R10 结果替换） |

> 注：上述替换对已用脚本逐格栅断言验证（目标串在 config.yaml 中恰好出现 1 次、替换后无残留、stage2/3/4 的 σ 参数与趋势过滤段 `ma` 行数不变）。

**R0（基线，四项全开默认）**
```powershell
Copy-Item strategies/small_alpha191_test2/config.yaml strategies/small_alpha191_test2/config_R0.yaml
python -u run_backtest.py --config strategies/small_alpha191_test2/config_R0.yaml > results/R0.log 2>&1
```

**R1（熔断更灵敏：10/15 → 8/12）**
```powershell
Copy-Item strategies/small_alpha191_test2/config.yaml strategies/small_alpha191_test2/config_R1.yaml
python -c "import io;p='strategies/small_alpha191_test2/config_R1.yaml';s=io.open(p,encoding='utf-8').read();s=s.replace('dd_half: 0.10','dd_half: 0.08').replace('dd_flat: 0.15','dd_flat: 0.12');io.open(p,'w',encoding='utf-8').write(s)"
python -u run_backtest.py --config strategies/small_alpha191_test2/config_R1.yaml > results/R1.log 2>&1
```

**R2（熔断更宽松：10/15 → 12/18）**
```powershell
Copy-Item strategies/small_alpha191_test2/config.yaml strategies/small_alpha191_test2/config_R2.yaml
python -c "import io;p='strategies/small_alpha191_test2/config_R2.yaml';s=io.open(p,encoding='utf-8').read();s=s.replace('dd_half: 0.10','dd_half: 0.12').replace('dd_flat: 0.15','dd_flat: 0.18');io.open(p,'w',encoding='utf-8').write(s)"
python -u run_backtest.py --config strategies/small_alpha191_test2/config_R2.yaml > results/R2.log 2>&1
```

**R3（恢复确认 3→2 日，缩短空仓期）**
```powershell
Copy-Item strategies/small_alpha191_test2/config.yaml strategies/small_alpha191_test2/config_R3.yaml
python -c "import io;p='strategies/small_alpha191_test2/config_R3.yaml';s=io.open(p,encoding='utf-8').read();s=s.replace('trend_confirm_days: 3','trend_confirm_days: 2');io.open(p,'w',encoding='utf-8').write(s)"
python -u run_backtest.py --config strategies/small_alpha191_test2/config_R3.yaml > results/R3.log 2>&1
```

**R4（HALF→NORMAL 恢复收窄 5%→3%）**
```powershell
Copy-Item strategies/small_alpha191_test2/config.yaml strategies/small_alpha191_test2/config_R4.yaml
python -c "import io;p='strategies/small_alpha191_test2/config_R4.yaml';s=io.open(p,encoding='utf-8').read();s=s.replace('dd_recover_full: 0.05','dd_recover_full: 0.03');io.open(p,'w',encoding='utf-8').write(s)"
python -u run_backtest.py --config strategies/small_alpha191_test2/config_R4.yaml > results/R4.log 2>&1
```

**R5（恢复确认均线 20→10）**
```powershell
Copy-Item strategies/small_alpha191_test2/config.yaml strategies/small_alpha191_test2/config_R5.yaml
python -c "import io;p='strategies/small_alpha191_test2/config_R5.yaml';s=io.open(p,encoding='utf-8').read();s=s.replace('index_ma: 20','index_ma: 10');io.open(p,'w',encoding='utf-8').write(s)"
python -u run_backtest.py --config strategies/small_alpha191_test2/config_R5.yaml > results/R5.log 2>&1
```

**R6（恢复确认均线 20→30）**
```powershell
Copy-Item strategies/small_alpha191_test2/config.yaml strategies/small_alpha191_test2/config_R6.yaml
python -c "import io;p='strategies/small_alpha191_test2/config_R6.yaml';s=io.open(p,encoding='utf-8').read();s=s.replace('index_ma: 20','index_ma: 30');io.open(p,'w',encoding='utf-8').write(s)"
python -u run_backtest.py --config strategies/small_alpha191_test2/config_R6.yaml > results/R6.log 2>&1
```

**R7（S1 σ 2.5→2.2）**
```powershell
Copy-Item strategies/small_alpha191_test2/config.yaml strategies/small_alpha191_test2/config_R7.yaml
python -c "import io;p='strategies/small_alpha191_test2/config_R7.yaml';s=io.open(p,encoding='utf-8').read();s=s.replace('pt_sigma: 2.5','pt_sigma: 2.2').replace('sl_sigma: -2.5','sl_sigma: -2.2');io.open(p,'w',encoding='utf-8').write(s)"
python -u run_backtest.py --config strategies/small_alpha191_test2/config_R7.yaml > results/R7.log 2>&1
```

**R8（S1 σ 2.5→3.0）**
```powershell
Copy-Item strategies/small_alpha191_test2/config.yaml strategies/small_alpha191_test2/config_R8.yaml
python -c "import io;p='strategies/small_alpha191_test2/config_R8.yaml';s=io.open(p,encoding='utf-8').read();s=s.replace('pt_sigma: 2.5','pt_sigma: 3.0').replace('sl_sigma: -2.5','sl_sigma: -3.0');io.open(p,'w',encoding='utf-8').write(s)"
python -u run_backtest.py --config strategies/small_alpha191_test2/config_R8.yaml > results/R8.log 2>&1
```

**R9（AUC 门控 0.54→0.52，对照旧值）**
```powershell
Copy-Item strategies/small_alpha191_test2/config.yaml strategies/small_alpha191_test2/config_R9.yaml
python -c "import io;p='strategies/small_alpha191_test2/config_R9.yaml';s=io.open(p,encoding='utf-8').read();s=s.replace('auc_threshold: 0.54','auc_threshold: 0.52');io.open(p,'w',encoding='utf-8').write(s)"
python -u run_backtest.py --config strategies/small_alpha191_test2/config_R9.yaml > results/R9.log 2>&1
```

**R10（AUC 门控 0.54→0.56）**
```powershell
Copy-Item strategies/small_alpha191_test2/config.yaml strategies/small_alpha191_test2/config_R10.yaml
python -c "import io;p='strategies/small_alpha191_test2/config_R10.yaml';s=io.open(p,encoding='utf-8').read();s=s.replace('auc_threshold: 0.54','auc_threshold: 0.56');io.open(p,'w',encoding='utf-8').write(s)"
python -u run_backtest.py --config strategies/small_alpha191_test2/config_R10.yaml > results/R10.log 2>&1
```

**R11（组合，需先跑完 R1-R10）**——示例取「熔断 8/12 + 确认 2 日 + 均线 10 + S1 3.0σ + 门控 0.56」，**按 R1-R10 实际最优值替换**：
```powershell
Copy-Item strategies/small_alpha191_test2/config.yaml strategies/small_alpha191_test2/config_R11.yaml
python -c "import io;p='strategies/small_alpha191_test2/config_R11.yaml';s=io.open(p,encoding='utf-8').read();s=s.replace('dd_half: 0.10','dd_half: 0.08').replace('dd_flat: 0.15','dd_flat: 0.12').replace('trend_confirm_days: 3','trend_confirm_days: 2').replace('index_ma: 20','index_ma: 10').replace('pt_sigma: 2.5','pt_sigma: 3.0').replace('sl_sigma: -2.5','sl_sigma: -3.0').replace('auc_threshold: 0.54','auc_threshold: 0.56');io.open(p,'w',encoding='utf-8').write(s)"
python -u run_backtest.py --config strategies/small_alpha191_test2/config_R11.yaml > results/R11.log 2>&1
```

**R12（稳定性检验：R11 同组合但 S1 取次优值 2.2）**
```powershell
Copy-Item strategies/small_alpha191_test2/config.yaml strategies/small_alpha191_test2/config_R12.yaml
python -c "import io;p='strategies/small_alpha191_test2/config_R12.yaml';s=io.open(p,encoding='utf-8').read();s=s.replace('dd_half: 0.10','dd_half: 0.08').replace('dd_flat: 0.15','dd_flat: 0.12').replace('trend_confirm_days: 3','trend_confirm_days: 2').replace('index_ma: 20','index_ma: 10').replace('pt_sigma: 2.5','pt_sigma: 2.2').replace('sl_sigma: -2.5','sl_sigma: -2.2').replace('auc_threshold: 0.54','auc_threshold: 0.56');io.open(p,'w',encoding='utf-8').write(s)"
python -u run_backtest.py --config strategies/small_alpha191_test2/config_R12.yaml > results/R12.log 2>&1
```

**每次运行前自检**：`Select-String -Path strategies/small_alpha191_test2/config_R{n}.yaml -Pattern "dd_half|dd_flat|auc_threshold"` 应看到预期值；`strategy.class` 必须指向 `strategies.small_alpha191_test2.strategy.SmallAlpha191Strategy`（勿误用 test）。

### 4.4 结果记录模板（每个格栅回测完成后填写）

```markdown
## R{n} 结果 (日期)
- 配置变更: xxx
- 结果目录: results/...
- 总收益: __% | 年化: __% | Sharpe: __ | 最大回撤: __%
- 平均仓位: __% | 空仓天数: __ | 订单数: __ (预警 __, combo __)
- 分年度: 2022 __% / 2023 __% / 2024 __% / 2025 __% / 2026 __%
- 归因: Active __% | 特质 alpha __% | 规模 __% | 价值 __% | 波动率 __%

### 验证点 (重点核对)
- [ ] 日志出现 "[方案A 回撤熔断]" 触发 N 次 (FLAT __ 次, HALF __ 次)
- [ ] 2024-01-22 调仓日: 趋势过滤是否触发 (指数<MA20 → 建仓减半日志)
- [ ] 2024-02 微盘危机: 回撤是否触发熔断, 组合回撤 vs 143844 的 -0.4%
- [ ] 最大回撤区间: 熔断后仓位变化
- [ ] S1 触发次数 vs 143844 (1746 次) 是否下降
- [ ] 空仓天数 vs 143844
```

### 4.5 裁决规则

1. **主指标**: Sharpe（优先）；次指标: 收益/回撤比
2. **统计检验**: 最优 vs 次优组合用 `scripts/ab_sig_test.py`（920 日, Holm 校正），P<0.05 才算显著改进
3. **稳健性**: 最优组合必须在 R12 稳定性检验中保持（差异 <0.05 Sharpe）
4. **纪律**: 单因子格栅阶段发现某参数方向无效，立即跳过该方向后续格栅（例如 R1 劣于 R0 则 R2 不再需要反向验证的必要性可放宽，直接进阶段二）
5. **红绿灯**: 任一格栅若 2024-02 微盘危机期间回撤 > -8%（143844 为 -0.4%），该方向立即标记风险（被动低仓位是现有保护，不能主动破坏）

---

## 5. 已知限制与风险

1. **上证综指代理中证2000**: 中证2000 指数数据缺失，趋势闸门用上证综指（sh000001）代理——综指是权重指数，微盘行情下可能失真（2024-02 微盘危机时综指相对抗跌，趋势过滤可能不触发）。若后续拿到中证2000 数据可替换（`index_code` 参数已支持）。
2. **网格化参数未动**: 本次只实施风控四项，网格/选股/标签等模块未改，与 143844 完全一致。
3. **峰值重置语义**: FLAT→HALF 以恢复入场点净值为新基线，意味着"空仓期间净值损失清零"——这是设计意图（减少踏空），但 HALF 阶段回撤容忍变宽，需 R1-R4 格栅验证。
4. **cp 覆盖不拷 .bak**: test2 无 O1+ 备份文件，如需要可随时从 test 目录拷贝。
