# small_alpha191_test2 实盘对接方案
## gpsef（离线选股）→ 信号文件 → Eureka（实盘执行/实时止盈止损）

**文档版本**: v2.0（2026-08-31）
**方案状态**: ✅ 接口已全部打通并验证（实盘数据未接入，链路用本地数据+回测撮合验证）
**架构确认**: 2026-08-31 用户明确 —— gpsef 是离线框架（训练/选股），Eureka 是实盘框架（实时监控/交易），**实盘侧不做训练**

---

## 1. 总体架构

```
┌───────────────────────────┐        ┌───────────────────────────────┐
│   gpsef（离线框架）          │        │   Eureka（实盘框架）             │
│   E:\quant\gpsef_project   │        │   E:\quant\Eureka              │
│                           │        │                               │
│  ┌─────────────────────┐  │  信号   │  ┌─────────────────────────┐  │
│  │ 回测引擎(训练/选股)    │  │  JSON  │  │ LiveEngine (实时时钟/bar) │  │
│  │ 调仓日选股 target     │──┼───────▶│  │  Strategy(信号驱动)       │  │
│  │                     │  │ 文件    │  │  ├─ 建仓/调仓(按信号)      │  │
│  └─────────────────────┘  │        │  │  └─ 实时止盈止损(分钟级)   │  │
│        │                  │        │  └─────────┬───────────────┘  │
│  export_signals.py        │        │            │ broker.order     │
│  (导出调仓信号)             │        │  ┌─────────▼───────────────┐  │
│                           │        │  │ Broker: 回测撮合/xuntou实盘│  │
│  signals/*.json ◀─────────┼────────┼──┤ DataCenter: GpsefDataCenter│ │
│  (每调仓日一个文件)         │        │  │ (桥接 gpsef 本地数据/待接行情)│ │
└───────────────────────────┘        └───────────────────────────────┘
```

### 职责边界（关键约束）

| 环节 | 归属 | 说明 |
|---|---|---|
| 训练/回测/选股 | **gpsef（离线）** | 元标注模型、因子、调仓日选股全部离线完成 |
| 信号导出 | gpsef → 文件 | 每个调仓日写一个 JSON（code+weight+ref_price） |
| 信号消费 | Eureka | 读信号文件 → 目标持仓 |
| 建仓/调仓执行 | Eureka | 按目标权重下单（卖旧买新） |
| 实时止盈止损 | Eureka | S1-S4 分阶段规则引擎（分钟级检查，**纯规则无训练**） |
| 交易撮合/券商对接 | Eureka | 回测撮合 broker / xuntou 实盘 broker |
| 行情数据 | Eureka 数据层 | 未接入期用 GpsefDataCenter 桥本地数据 |

**为什么这样划分**：训练/选股需要全量历史+因子计算，放离线侧可复用 gpsef 已充分验证的回测框架；实盘侧只保留**确定性规则**（止盈止损/调仓执行），无训练依赖 → 实盘延迟低、可审计、故障面小。

---

## 2. 信号文件协议（两框架的唯一接口）

### 2.1 文件组织

```
E:\quant\Eureka\signals\
  ├── 20220801.json      # 每个调仓日一个文件, 文件名 = 调仓日 YYYYMMDD
  ├── 20220829.json
  ├── ...
  └── 20260622.json      # (当前为 gpsef R12 回测导出的 43 个真实信号)
```

### 2.2 文件内容（JSON）

```json
{
  "signal_date": "2022-08-01",
  "strategy": "small_alpha191_test2",
  "target_count": 20,
  "targets": [
    { "code": "sh.601279", "weight": 0.05454, "ref_price": 6.6366 },
    { "code": "sz.002494", "weight": 0.053689, "ref_price": 4.5846 },
    ...
  ],
  "note": "由 gpsef R12 回测 orders.csv 导出 (离线选股结果样例)"
}
```

- `code`: 交易所代码（sh./sz. 前缀，两框架一致）
- `weight`: 目标权重（可不等权；Eureka 侧归一化）
- `ref_price`: 选股时的参考价（用于下单股数估算）

### 2.3 信号导出（gpsef 侧）

```bash
cd E:\quant\gpsef_project
.venv\Scripts\python.exe scripts/export_signals.py \
    --orders results/<回测目录>/orders.csv \
    --out E:/quant/Eureka/signals
```

**正式流程**：gpsef 策略在调仓日选股完成后，将 `target_stocks/target_weights` 序列化写入该目录（在策略 `handle_bar` 调仓分支加一个导出钩子即可，或由离线批处理在每个调仓日收盘后导出）。

---

## 3. Eureka 侧实现（strategies/small_alpha191_test2_live/）

### 3.1 文件清单

| 文件 | 职责 |
|---|---|
| `strategy.py` | 信号驱动 IStrategy 适配器（v2，无训练） |
| `staged_exit.py` | S1-S4 分阶段止盈止损规则引擎（从 gpsef StageManager 移植，纯逻辑） |
| `signal_loader.py` | 信号文件加载/按日期查询 |
| `datacenter.py` | GpsefDataCenter —— IDataCenter 桥接 gpsef 本地数据 |
| `config.yaml` | 信号目录/止盈止损参数/区间 |

### 3.2 策略执行语义（strategy.py）

```
每日盘前 before_market_open:
  查当日是否有信号文件 → 有则设置 _pending_targets（目标持仓）

盘中 run_period（实盘 3s 轮询, 分钟粒度）:
  ① 调仓日: _execute_rebalance()
     - 卖出不在目标中的持仓
     - 按权重买入新目标（预算 = 现金×98%）
     - 注册止盈止损 tracker
  ② 每个 bar: _check_exits()
     - 对每个持仓: 取当日 high/low → 规则引擎检查 S1-S4
     - 触发 → 按阶段比例卖出（基于建仓基准股数）

收盘 after_market_close:
  估值 + keeper 快照（cash/positions/trackers）
```

### 3.3 止盈止损规则引擎（staged_exit.py）

从 gpsef `StageManager/StageTracker/_check_staged_minute` 移植的**纯函数版本**：

| 阶段 | 持有期 | PT(σ) | SL(σ) | 卖出比例 |
|---|---|---|---|---|
| S1 | ≤5 天 | 3.0 | -3.0 | 25% |
| S2 | ≤10 天 | 3.0 | -3.0 | 35% |
| S3 | ≤15 天 | 4.0 | -4.0 | 20% |
| S4 | ≤20 天 | 9.0 | -4.0 | 20% |
| 硬止损 | 任意 | — | -6.0σ | 全清 |

动态调整（与 gpsef 一致）：
- PT 触发 → SL 上移（S1→breakeven, S2→+1σ, S3→+2σ 利润锁定）
- SL 触发 → PT 下移（S2→1.5σ, S3→2.0σ）
- 参数完全由 config.yaml `staged_exit` 段驱动（当前=R12 最终裁决值）

**说明**：本引擎只包含 σ 阈值判定与卖出比例等**确定性规则**，不涉及任何模型训练——严格满足"实盘不训练"约束。

### 3.4 数据层（datacenter.py）

- `get_current_data(code)`: 按 Eureka 时钟返回对应交易日行情（未接入期用本地 parquet）
- `get_price_data(code, start, end, fields, count)`: 兼容 broker 撮合契约
- `get_trade_calendar/get_stock_list/get_adjust_factor/get_dividend_info`: 全接口桥接

**接入真实行情后的替换点**：仅需替换 `get_current_data`/`get_price_data` 内部实现（如接掘金/券商行情），策略与 broker 零改动。

---

## 4. 测试与验证结果

### 4.1 单元验证（规则引擎）

| 用例 | 结果 |
|---|---|
| 上涨 +5%（≈1.67σ < 3σ） | ✅ 无触发 |
| 下跌 -10%（≈3.33σ > 3σ） | ✅ 触发 S1 止损 (stage=1, is_pt=False) |

### 4.2 链路测试（3 天：首日建仓 + 止盈止损）

```
E:/quant/gpsef_project/.venv/Scripts/python.exe testscripts/test_signal_live.py 3
```

| 验证点 | 结果 |
|---|---|
| 信号加载（2022-08-01, 20 只） | ✅ |
| 首日建仓下单 | ✅ 20 单 |
| 撮合成交 | ✅ 18 只成交（仓位 57.4%）|
| 止盈止损触发 | ✅ sh.688182 被 S1(900股)+S2(1260股) 精确按比例卖出 |
| 期末持仓/总资产 | ✅ 18 只 / ¥969,215 |
| 跨日状态保持（trackers） | ✅ 19 个 |

### 4.3 二次调仓测试（30 天：覆盖第二个调仓日）

```
E:/quant/gpsef_project/.venv/Scripts/python.exe testscripts/test_signal_live.py 30
```

| 验证点 | 结果 |
|---|---|
| 08-29 第二信号日识别 | ✅ last_signal_date=2022-08-29 |
| 调仓卖出（不在新目标） | ✅ 16 笔全部成交 |
| 调仓买入（新目标） | ✅ 19 单（1 只已在持仓跳过） |
| S1-S4 全生命周期 | ✅ sz.300509 四阶段 25/35/20/20% 精确清仓 |
| 期末 | ✅ 17 只 / ¥982,273 / 仓位 50.0% |

### 4.4 已验证的撮合语义（真实市场行为）

- 涨停股买入被拒（sz.002494 8-01 涨停 close=4.99=high → 拒单）——与真实市场一致 ✅
- 现金不足拒单（超出预算的订单被 broker 拒绝，不产生负现金）✅

### 4.5 测试结论

```
[RESULT] ✅ 信号驱动链路全部打通
```

完整链路已验证：**gpsef 导出信号 → Eureka 读信号 → 建仓 → 撮合成交 → 实时止盈止损（S1-S4）→ 二次调仓（卖旧买新）→ 收盘估值**，接口层全部打通。

---

## 5. 实盘阶段规划（数据/券商接通后）

### 阶段 1：接真实行情（当前状态 → 下一步）
- 替换 `datacenter.py` 的 `get_current_data`：接掘金 SDK / 券商行情推送
- `run_period` 已支持分钟粒度（实盘模式 3s 轮询 + 分钟过滤），无需改动策略逻辑
- 验证：同一信号文件在"回测撮合"与"真实行情"下止盈止损行为一致

### 阶段 2：接 xuntou 实盘券商
- Eureka 已内置 `impl/broker/xuntou/`（华鑫 QMT 模拟）
- config.yaml `broker.module` 切换为 xuntou broker
- 需配置：券商终端路径/账号（config/default.yaml 已有字段）
- 验证：模拟盘下单/成交回报

### 阶段 3：信号自动导出（gpsef 侧）
- 在 gpsef 策略调仓分支加导出钩子（调仓日自动写信号文件）
- 或用定时任务：每个调仓日收盘后运行导出脚本
- 加入信号文件校验（格式/完整性），避免脏信号进入实盘

### 阶段 4：上线前清单
- [ ] 信号文件签名/校验（防篡改）
- [ ] 调仓幂等（重复信号不重复下单 —— 已通过 _done_dates 防重）
- [ ] 异常兜底（行情断流/下单失败告警）
- [ ] 手动确认机制（调仓日人工确认后再执行）
- [ ] 与 gpsef 回测做一次"信号回放"对比（同一信号 → 实盘路径 vs 回测路径持仓一致）

---

## 6. 风险与注意事项

1. **实盘不训练**：Eureka 侧绝不 import gpsef 策略类的训练路径（本方案已彻底解耦，信号文件是唯一输入）
2. **止盈止损参数同步**：`staged_exit` 参数在 gpsef config.yaml 与 Eureka config.yaml 各有一份，需保持同步（建议以 gpsef 为准，Eureka 侧加启动校验）
3. **信号延迟**：调仓信号由离线生成，实盘执行存在 T+N 延迟（当前回测验证为 T+0 开盘执行）——正式实盘需明确信号时间戳语义
4. **价格口径**：当前用前复权价（reference_date=20260804），与 gpsef 一致；实盘下单需换算为真实价（gpsef 双轨价格体系：复权价触发、真实价执行）
5. **涨停/停牌**：撮合 broker 已正确拒绝涨停买入；实盘需处理停牌（get_current_data.paused）
6. **现金预算**：调仓预算 = 现金×98% 预留滑点/税费；实际买入可能因涨停/停牌部分落空，需再平衡机制（下一阶段）

---

## 7. 快速上手

```bash
# 1) 导出信号（gpsef 侧, 从回测结果）
cd E:\quant\gpsef_project
.venv\Scripts\python.exe scripts/export_signals.py --orders results/<回测>/orders.csv --out E:/quant/Eureka/signals

# 2) 链路测试（Eureka 侧, 3 天）
cd E:\quant\Eureka
E:/quant/gpsef_project/.venv/Scripts/python.exe testscripts/test_signal_live.py 3

# 3) 二次调仓测试（30 天）
E:/quant/gpsef_project/.venv/Scripts/python.exe testscripts/test_signal_live.py 30
```

---

## 附录 A：与旧方案（v1）的差异

| 维度 | v1（已废弃） | v2（本方案） |
|---|---|---|
| Eureka 侧职责 | 完整复用 gpsef 策略（含训练） | 只消费信号 + 规则止盈止损 |
| 训练位置 | Eureka（违反用户约束） | gpsef 离线 |
| 数据流 | 策略内部直接调用 | 信号文件（解耦） |
| 测试 | 需完整训练 10-15 分钟 | 纯规则+数据桥, 秒级 |

## 附录 B：文件位置

```
gpsef 侧:
  scripts/export_signals.py           # 信号导出
  results/.../orders.csv              # 回测订单（信号来源）
Eureka 侧:
  strategies/small_alpha191_test2_live/strategy.py      # 信号驱动适配器
  strategies/small_alpha191_test2_live/staged_exit.py   # 止盈止损规则引擎
  strategies/small_alpha191_test2_live/signal_loader.py # 信号加载
  strategies/small_alpha191_test2_live/datacenter.py    # 数据桥
  strategies/small_alpha191_test2_live/config.yaml      # 配置
  testscripts/test_signal_live.py    # 链路测试（3/30 天）
  signals/                           # 信号文件（43 个真实信号样例）
```
