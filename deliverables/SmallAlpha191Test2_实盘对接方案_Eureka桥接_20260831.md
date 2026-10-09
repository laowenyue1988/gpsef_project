# small_alpha191_test2 实盘对接方案（Eureka × gpsef 框架桥接）

**日期**: 2026-08-31
**状态**: 阶段 1（接口打通）实现完成 + 测试中，供审核

---

## 1. 背景与目标

gpsef 回测框架下已完成 small_alpha191_test2 策略（R12 最终裁决配置）的全量回测验证
（全窗口 Sharpe 1.68 / 回撤 -7.6%）。本方案将该策略**完整迁移到 Eureka 实盘框架**，
实现"回测 → 实盘"的同一套决策逻辑复用。当前阶段目标：**接口打通**（Eureka 引擎驱动 +
gpsef 决策复用 + 撮合/持仓链路贯通），真实行情接入为后续阶段。

## 2. Eureka 框架全解

### 2.1 架构（框架/实现分离）

```
Eureka/
├── run.py                     # 统一入口: config 合并 → 动态装配 → 启动
├── config/default.yaml        # 全局默认（broker/datacenter/output/模式）
├── framework/                 # 框架核心（抽象契约 + 引擎）
│   ├── interface/             # IStrategy / IBroker / IDataCenter / Portfolio / Keeper
│   ├── engine/                # BacktestEngine（回测） / LiveEngine（实盘） / Analyzer / Reporter
│   └── common/                # Clock（回测/实盘统一时钟） / Log / OneShot
├── impl/                      # 实现层
│   ├── broker/                # backtest（SimulatedBroker 撮合）/ xuntou（迅投 QMT 实盘）
│   └── datacenter/            # baostock / tdx / akshare / mock（合成数据）
├── strategies/                # 策略目录（config.yaml + strategy.py）
└── testscripts/               # 测试脚本
```

### 2.2 核心接口

| 接口 | 职责 | 关键方法 |
|---|---|---|
| **IStrategy** | 策略业务逻辑 | `initialize()` → `before_market_open()` → `run_period()`（~3s 轮询）→ `after_market_close()` |
| **IBroker** | 交易 | `order(code, volume, price)`（正买负卖）/ `cancel` / `get_orders` / `get_trades` / `get_positions` / `get_available_cash` / `get_total_value` |
| **IDataCenter** | 行情数据 | `get_current_data(code)`（快照）/ `get_price_data(code, start, end)`（K线）/ `get_trade_calendar` |
| **Portfolio** | 账户只读代理 | `get_position` / `has_position` / 现金/总资产 |
| **Keeper** | 状态持久化 | 策略任意属性（重启恢复） |
| **Clock** | 统一时钟 | 回测=引擎驱动模拟时间；实盘=真实时间（3s 轮询） |

### 2.3 双引擎驱动

- **BacktestEngine**：遍历交易日历 → 每日 `before_market_open` → `run_period`（1d=每日一次，1m/5m=每 bar）→ `after_market_close` → Analyzer 收集
- **LiveEngine**：真实时间 → 盘前钩子 → 盘中 ~3s 轮询 `run_period`（含午休/收市判断）→ 盘后钩子 → Keeper 落盘（`state.pkl`）

### 2.4 订单模型

`OrderStatus`: PENDING → PARTIAL → FILLED / REJECTED / CANCELLED；策略必须处理所有状态（不假设下单即成交）。
撮合模式：NEXT（下一 bar 开盘价，防未来）/ CURRENT（当前价即时成交）。

## 3. gpsef 侧接口分析（small_alpha191_test2）

### 3.1 策略结构

```
class SmallAlpha191Strategy(gpsef IStrategy):
    initialize(context)          # 读 config/装配 data_provider/stock_pool/参数
    before_market_open(context)  # 每日: 到期标记/IPBS检查/调仓判断/V10门控/选股/权重/风控(熔断/趋势过滤)
    handle_bar(context)          # 每日/每bar: 到期卖出/PT-SL分钟检查/调仓执行(直通式改portfolio)
```

### 3.2 依赖组件（回测引擎装配）

| 组件 | 类 | 说明 |
|---|---|---|
| 数据 | `LocalDataProvider` | 前复权日线/分钟线/复权因子/交易日历（本地 parquet） |
| 候选池 | `DynamicStockPool` | 向量化截面过滤（ST/次新/价格/可交易） |
| 交易 | `StrategyContext.portfolio` | **直通式**：`add_position/remove_position` 直接改持仓（无订单层） |
| 模型 | `EnsembleManager` | Purged K-Fold CV + MDA 特征选择 + 概率校准（训练/预测） |
| 风控 | 熔断状态机 + 趋势过滤 + 分阶段退出（S1-S4 + hard_sl） | 组合级 + 持仓级 |

### 3.3 关键差异（对接难点）

| 维度 | gpsef 回测 | Eureka 实盘 |
|---|---|---|
| 时间驱动 | 引擎逐日/逐bar | Clock 3s 轮询 |
| 交易 | 直通式（改 portfolio） | 订单式（IBroker.order） |
| 状态 | 进程内自维护（holdings/stage_mgr） | Keeper 落盘/恢复 |
| 上下文 | StrategyContext（current_dt/portfolio/calendar） | self.broker/dcenter/portfolio |

## 4. 对接架构设计

### 4.1 总体架构

```
┌──────────────────────── Eureka 实盘框架 ────────────────────────┐
│  LiveEngine / BacktestEngine                                    │
│      │ 驱动                                                      │
│      ▼                                                          │
│  SmallAlpha191Test2LiveStrategy (适配器, 新增)                   │
│      │                                                          │
│      ├─▶ gpsef SmallAlpha191Strategy ◀── 决策层 100% 复用        │
│      │       ├─ LocalDataProvider  (真实本地数据)                │
│      │       ├─ DynamicStockPool   (候选池)                      │
│      │       └─ EurekaBrokerPortfolio ◀── 直通交易→Eureka订单    │
│      │                                                            │
│  broker: backtest(撮合) / xuntou(QMT实盘) ◀── 订单/成交           │
│  dcenter: GpsefDataCenter (IDataCenter 桥接) ◀── 行情/日历         │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 接口映射表

| Eureka | gpsef | 桥接方式 |
|---|---|---|
| `initialize()` | `SmallAlpha191Strategy.initialize(ctx)` | 适配器装配 gpsef context（config/data_provider/stock_pool/calendar） |
| `before_market_open()` | `before_market_open(ctx)` | 适配器检测新交易日 → 驱动（`ctx.current_dt` 同步 Clock） |
| `run_period()`（3s） | `handle_bar(ctx)` | 回测=bar 粒度直接驱动；实盘=分钟粒度 + 交易时段过滤（防 trade_day_seq 被 3s 轮询破坏） |
| `after_market_close()` | — | 收盘估值（dcenter 行情 → update_value）+ Keeper 持仓快照 |
| `IBroker.order()` | portfolio.add/remove_position | **EurekaBrokerPortfolio**：拦截直通交易 → 立即转 Eureka 订单 → 撮合成交后同步持仓 |
| `IDataCenter` | `LocalDataProvider` | **GpsefDataCenter**：get_current_data=最新历史日收盘（未接实时行情期）；get_price_data=前复权日线/分钟线 |
| `Keeper` | 策略内存状态 | 阶段 2：holdings/stage_mgr 序列化落盘 |

### 4.3 跨日状态保持（关键设计）

Eureka BacktestEngine **每个交易日重建策略实例**——而 gpsef 策略是有状态的
（训练模型/持仓/stage_mgr）。适配器用**模块级缓存**复用同一 gpsef 策略实例：
- 引擎重建 adapter → `initialize()` 检查缓存 → 命中则复用 gpsef 策略 + context
- 桥接 Portfolio 每日重建（从 broker 同步持仓，保持与真实账户一致）

## 5. 实现清单（本阶段交付）

| 文件 | 职责 |
|---|---|
| `Eureka/strategies/small_alpha191_test2_live/strategy.py` | 适配器（IStrategy）：gpsef 装配/驱动/状态保持 |
| `.../datacenter.py` | GpsefDataCenter（IDataCenter 桥接真实本地数据） |
| `.../portfolio_bridge.py` | EurekaBrokerPortfolio（直通交易→Eureka 订单桥） |
| `.../config.yaml` | Eureka 侧配置（broker/datacenter/参数） |
| `Eureka/testscripts/test_live_bridge.py` | 接口打通测试 |

## 6. 测试方案与结果

### 6.1 测试链路
BacktestEngine（Eureka 回测驱动）→ 适配器 → gpsef 策略（真实本地数据 + 真实训练）→
EurekaBrokerPortfolio → backtest broker（CURRENT 撮合）→ 成交/持仓同步。

### 6.2 验证点
1. gpsef 组件装配（数据/日历/股票池/策略 initialize）
2. 选股信号产生（训练 + 预测 → target_stocks 非空）
3. 桥接下单（目标权重 → Eureka 订单）
4. 成交与持仓同步（FILLED + broker positions + bridge portfolio 一致）
5. 跨日状态保持（gpsef 策略缓存复用，不重复训练）
6. 收盘估值 + Keeper 快照

### 6.3 测试结果
（测试运行中，完成后回填：订单/成交/持仓/一致性验证输出）

## 7. 风险与限制（阶段 1 已知）

| 项 | 说明 | 阶段 2 对策 |
|---|---|---|
| 成交回报对齐 | gpsef 策略自维护 holdings 用请求价；Eureka 成交价含滑点 | 成交回报(Trade)回调 → 校准 holdings/avg_cost |
| 状态持久化 | Keeper 仅存持仓快照，holdings/stage_mgr 未完整序列化 | 完整 pickle + 恢复校验 |
| 实时行情 | GpsefDataCenter 返回"最新历史日"模拟盘中 | 接入 xuntou/第三方行情源替换 `get_current_data` |
| 分钟级 PT/SL | 实盘按分钟粒度驱动 handle_bar（依赖 dcenter 分钟线） | 实盘接实时分钟行情 |
| Eureka venv 损坏 | `.venv` 指向不存在的 Python310 | 重建 venv 或统一用 gpsef venv（已验证兼容） |
| 重启恢复 | 断线/重启后 gpsef 策略状态需恢复 | Keeper 全量状态 + LiveEngine 断线重连 |

## 8. 阶段规划

| 阶段 | 内容 | 验收 |
|---|---|---|
| **1（本次）** | 接口打通：适配器 + 数据桥 + 订单桥 + 回测驱动测试 | 链路跑通（订单/成交/持仓一致） |
| 2 | 状态完整持久化（Keeper）+ 成交回报校准 + LiveEngine 实盘驱动 | 重启恢复 + 真实订单状态 |
| 3 | 接入真实行情（xuntou 行情/第三方）→ get_current_data 实时化 | 盘中信号实时产生 |
| 4 | 风控落地（熔断/趋势过滤实盘语义）+ 灰度/模拟盘 | 模拟盘全流程 |

## 9. 运行指南

```powershell
# 接口打通测试（3 天回测驱动, 真实本地数据 + 回测撮合）
cd E:\quant\Eureka
E:\quant\gpsef_project\.venv\Scripts\python.exe testscripts\test_live_bridge.py 3

# 完整回测驱动（修改 config.yaml 区间后）
E:\quant\gpsef_project\.venv\Scripts\python.exe run.py strategies\small_alpha191_test2_live
```
