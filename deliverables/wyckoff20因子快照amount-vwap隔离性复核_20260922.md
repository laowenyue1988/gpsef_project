# `strategies/wyckoff20` 因子快照 · 是否受 `amount`/`vwap` 复权方向缺陷（D-281）影响

> **复核对象**：`strategies/wyckoff20/`（`build_snapshots.py` / `factors_v5.py` / `gpsef_strategy_v10.py` / `barra_builder.py`）及其产物 `data/wyckoff20_v5_snapshots/`、`data/wyckoff20_v10_recomd_snapshots/`
> **复核动机**：用户 2026-09-22 17:40 质询「wyckoff20 策略的因子快照是否存在类似问题」
> **关联**：D-281（`amount`/`vwap` 复权方向相反）、R-147（数据管线经济不变量断言）、`Step0_amount-vwap缺陷决策影响评估_20260922.md`
> **结论（先行）**：**❌ 不存在类似问题 —— wyckoff20 快照链路对 D-281 完整免疫**，四条独立证据闭合。**但发现两个必须报告的边界事实**（见 §5）。

---

## §0 一页结论

| 环节 | 是否消费被污染量 | 判定 | 证据 |
|---|---|---|---|
| **① `build_snapshots.py` 的价格/量过滤** | ❌ 用 `raw_close`（provider 直供不复权列）+ `volume`（未复权） | **免疫** | §2.1 |
| **② `build_snapshots.py` 的 `total_mv`** | ⚠️ 形式上是 `amount/(turn/100)` —— **但只取末行**，而末行 `f(T)=1` | **免疫** | §2.2（逐位一致 40/40） |
| **③ `factors_v5.py` 的 40 个因子** | ❌ `amount` **0 次**；`vwap` 40 次**全为自算**（`(h+l+c)/3` 加权） | **免疫** | §3（40 因子 × 5 股 **零差异** + 反自证 PASS） |
| **④ `wyckoff20_v10_recomd_snapshots`** | ❌ 仅 `code/pred/rank_pct` 3 列 | **免疫** | §4.1 |
| **⑤ `barra_builder.py`（V10 用）** | ❌ `_load_daily_data` **直读 `raw_daily`**；D4 provider 路径**只取 `close`** | **免疫** | §4.2 |
| **⑥ `gpsef_strategy_v10.py`（实盘/回测主体）** | ❌ `amount` 0 次；`vwap` 2 次**均为因子名字符串** | **免疫** | §4.3 |

**⇒ 与 alpha191 的关键差别**：alpha191 的 39 个因子**直接读取 provider 的 `vwap` 列**（被 ×`1/f`）；wyckoff20 **从不读该列**，它的"VWAP"是**自己用 OHLC 算的典型价加权均价**（`(high+low+close)/3 × volume`），而 OHLC 全部乘同一个 `f` ⇒ **比值与排序不受影响**。

---

## §1 复核方法（三层，非单一静态检查）

1. **静态层**：逐文件 grep `amount` / `vwap` / `['vwap']` / `raw_amount`，定位所有读取点；
2. **数据层**：实测 `f(T)=1` 恒等式、`total_mv` 与 `raw_daily` 直接口径对账；
3. **★行为层（决定性）**：构造 **BASE（现状，含缺陷）vs FIXED（`amount×f`、`vwap=amount/volume`）** 两份输入（**其余列逐位相同**），跑**同一个** `WyckoffFactorEngineV5`，逐因子比 `raw/score/conf`；并设**反自证**（`close×1.37` 必须引起大面积变化，否则说明比较方法失效）。

工具：`_audit_wyckoff_snapshot.py`（可复跑）。

---

## §2 `build_snapshots.py`（快照生成器）

### 2.1 数据来源与过滤条件

```python
# 第 290-295 行：同样是 LocalDataProvider ⇒ 形式上继承 D-281
df = data_provider.get_daily_data(code, lookback_start, prev_trade_date,
                                 reference_date=prev_trade_date)
# 第 308-310 行：仅做 pd.to_numeric 类型转换（不是复权）
for col in ['open','high','low','close','volume','amount','turn']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
# 第 325-334 行：★过滤器用 raw_close（不复权）与 volume（未复权）
if 'raw_close' in df.columns: last_close = float(df['raw_close'].iloc[-1])
...
if last_close < 2.0 or last_close > 1000.0: continue
if last_volume < 500000: continue
```

**判定**：价格过滤器已由 **D3 修复（2026-09-15）** 改用 `raw_close`（provider 直供的**不复权价**，`local_provider.py:217`），**对基准日免疫**；量过滤器用 `volume`——**`volume` 全程未被复权调整**（D-281 只影响 `amount`/`vwap`）。
⇒ **两个过滤器均不受影响。**

### 2.2 `total_mv`（形式上唯一触碰 `amount` 的地方）

```python
# 第 354-357 行
last_amount = float(df['amount'].iloc[-1])   # ← 第 221 行：amount 被 ×1/f
last_turn   = float(df['turn'].iloc[-1])
row['total_mv'] = last_amount / (last_turn / 100.0)
```

**关键**：`df` 的**末行日期 = `prev_trade_date` = `reference_date`** ⇒ 按复权定义 `f(T)=smooth_bf(T)/smooth_bf(T)=1`。

**① 恒等式实测（`reference_date = 数据末日 T`）**

| 股票 | T | `amount_adj(T)` | `raw_amount(T)` | 比值 | `vwap_adj(T)` | `raw_vwap(T)` | 比值 |
|---|---|---|---|---|---|---|---|
| sh.600000 | 2024-08-02 | 266,363,128.80 | 266,363,128.80 | **1.00000000** | 8.493308 | 8.493308 | **1.00000000** |
| sh.600006 | 2024-08-02 | 952,424,519.68 | 952,424,519.68 | **1.00000000** | 7.541632 | 7.541632 | **1.00000000** |
| sh.600016 | 2024-08-02 | 495,595,095.65 | 495,595,095.65 | **1.00000000** | 3.537694 | 3.537694 | **1.00000000** |

**② 快照产物对账**（`factors_20260804.parquet`，构建时数据末日 T = 2026-08-03）

| code | snapshot `total_mv` | raw 口径 `amount/(turn/100)` | provider 口径 | snap/raw | snap/prov |
|---|---|---|---|---|---|
| 600000 | 318,482,057,468.50 | 318,482,057,468.50 | 318,482,057,468.50 | **1.00000000** | **1.00000000** |
| 600004 | 18,880,891,969.78 | 18,880,891,969.78 | 18,880,891,969.78 | **1.00000000** | **1.00000000** |
| …（共 40 只） | | | | … | … |

**与 raw 口径逐位一致 40/40，不一致 0；`snap/provider` 比值 min=中位=max=1.0000000000。**
⇒ **`total_mv` 免疫**（末行 `f=1`，且与 `raw_daily` 直接口径完全一致）。

> ⚠️ **方法学留痕**：初版复核曾用"快照日 T 当天"的 raw 数据做除数，得到 0.969~1.030 的**伪散布**（60/60"不一致"）。真因 = 快照日 ≠ 数据末日（数据末行为 `prev_trade_date`）。改用正确口径后比值恒为 1。
> **这正是 R-145 的同型情形：数值"该是常数却出现散布"时，先怀疑自己的中间量。**

---

## §3 ★`factors_v5.py`（40 个威科夫因子）—— 决定性行为级实证

### 3.1 静态层

| 项 | 计数 | 说明 |
|---|---|---|
| `amount` 出现 | **0 次** | 完全不使用成交额 |
| `vwap` 出现 | 40 次 | **全部为自算** |
| 读取 provider 的 `['vwap']` 列 | **0 处** | 全仓 grep（含 `.bak`/`backup` 外全文件）为空 |
| `volume` 出现 | 113 次 | `volume` **未复权** ⇒ 无影响 |

**自算 VWAP 的两处实现**（均只用 OHLC）：

```python
# _factor_vwap（V-08 vwap_anchor）与 _factor_vwap_deviation（V-06）
def _calc_vwap(self, df):
    tp = (df['high'] + df['low'] + df['close']) / 3      # 典型价
    return np.sum(tp * df['volume']) / np.sum(df['volume'])
# _factor_big_flow 亦然：vwap = (highs[i] + lows[i] + closes[i]) / 3
```

**为什么这能免疫**：D-281 只影响 `amount` 与 `vwap` 两列；OHLC 全部乘**同一个** `f(t)`。
⇒ `_calc_vwap ≡ raw_calc_vwap × f`，与 `close`（= `raw_close × f`）**同量纲** ⇒ **比值、排序、阈值判定全部不变**。

### 3.2 ★行为层（决定性）

构造 BASE（现状）/ FIXED（`amount×f`、`vwap=amount/volume`），**其余列逐位相同**，跑同一因子引擎：

**① 输入层（确认"修复确实生效"——即 FIXED 变体真的改了东西）**

| 股票 | n | `f≠1` 天数 | \|Δamount\|/amt 中位 | \|Δvwap\|/vwap 中位 |
|---|---|---|---|---|
| sh.600000 | 202 | **190** | 0.0354 | 0.0354 |
| sh.600004 | 202 | 0（该窗无除权） | 0.0000 | 0.0000 |
| sh.600006 | 202 | **187** | 0.0044 | 0.0044 |
| sh.600009 | 202 | **191** | 0.0034 | 0.0034 |
| sh.600016 | 202 | **186** | 0.0566 | 0.0566 |

⇒ **输入确实被改动**（最高 5.66%），不是"没改到"。

**② 因子层（40 个因子 × 5 只股票）**

| 股票 | 因子数 | **有差异的因子数** |
|---|---|---|
| sh.600000 | 40 | **0** |
| sh.600004 | 40 | **0** |
| sh.600006 | 40 | **0** |
| sh.600009 | 40 | **0** |
| sh.600016 | 40 | **0** |

⇒ **`raw` / `score` / `conf` 全部零差异。**

**③ ★反自证（判别力）**：把 `close` 乘 1.37（明显扰动）⇒ **12/40 个因子发生变化 ⇒ PASS**。
⇒ 证明"零差异"**不是因为比较方法失灵**，而是因为引擎**真的不读那两列**。

---

## §4 其余消费链路

### 4.1 `data/wyckoff20_v10_recomd_snapshots/`（V10 推荐快照，1235 个）

列结构与样本：

| code | pred | rank_pct |
|---|---|---|
| … | … | … |

**仅 3 列 ⇒ 不含 `amount`/`vwap` ⇒ 免疫。**

### 4.2 `barra_builder.py`（V10 的 BARRA 暴露，含 D4 修复）

| 路径 | 事实 | 判定 |
|---|---|---|
| `_load_daily_data(code)`（第 243 行） | **直接 `pd.read_parquet(data/raw_daily/*.parquet)`** —— 原始**不复权**数据 | ⇒ `size = amount/(turn/100)`（第 1120-1124 行）用的是 **raw `amount`** ⇒ **免疫** |
| D4 统一复权路径（第 320-340 行） | 调用 provider，但**只取 `close` 列**：`_s = _dfp.set_index(...)['close']` | ⇒ **不碰** `amount`/`vwap` ⇒ **免疫** |
| 全文件 `vwap` 出现 | **仅 1 处且为 `close`**（第 336 行） | ⇒ 免疫 |

### 4.3 `gpsef_strategy_v10.py`（V10 主体）

| 项 | 计数 | 判定 |
|---|---|---|
| `amount` | **0 次** | 免疫 |
| `vwap` | **2 次**，均为**因子名字符串**：`'vwap_anchor'`（第 333 行）、`'vwap_deviation'`（第 340 行） | **非数据访问** ⇒ 免疫 |

### 4.4 消费方（完整）

| 消费方 | 用法 |
|---|---|
| `strategies/wyckoff20/gpsef_strategy_v10.py:664` | `factor_path` 读 `data/wyckoff20_v5_snapshots` |
| `results/hs300_alpha191_test_*` 系（3 个） | `snapshot_dir: ./data/wyckoff20_v5_snapshots` 作为**扩展因子池** |
| `results/Wyckoff V10 Binary Classification_*` 系（多个） | `factor_path` 同上 |
| **本轮微盘股策略（test3）** | `recomd_snapshot_dir` 已标注"**不再使用（微盘池不依赖 V10 快照）**" ⇒ **不消费** |

⇒ 无论哪个消费方，读到的都是**已验证免疫的 40 个因子列**（+ `total_mv`）。

---

## §5 ★两个必须报告的边界事实

### 5.1 缺陷至少自 **2026-06-10** 就已存在（长期潜伏，非新引入）

| 版本 | `amount` 缩放行 |
|---|---|
| `gpsef/backups/20260610_preMajorSurgery/data/local_provider.py:230` | `df_merged['amount'] = df_merged['amount'] * (back_factor_T / df_merged['backAdjustFactor'])` |
| `gpsef/data/local_provider.py.bak_DET_20260910_142628` | 同 |
| `gpsef/data/local_provider.py.bak_preSnapshotFp_20260915_201927` | 同 |
| `gpsef/data/local_provider.py`（现行） | 同（第 259-263 行） |

**⇒ 这是**长期潜伏**的实现错误（≥3.5 个月），**不是 D2 引入、也不是近期引入**。**
⇒ 副作用：wyckoff v5 快照（写于 2026-08-10~08-17）与 `factor_snapshots` 族（08-06~09-02）**在同一缺陷期间生成**——但因本档已证 wyckoff 链路免疫，**不影响**。

### 5.2 ★快照自带的 `volume`/`amount`/`vwap` 三列：当前无害，但是隐患

| 快照族 | 列数 | 是否含这三列 | 写入方 |
|---|---|---|---|
| `data/factor_snapshots_test`（**test3 现行**） | **221** = 190 alpha + 8 FFD + `ts_code`/`trade_date` + **`volume`/`amount`/`vwap`** | ✅ **含（且被污染）** | `strategies/small_alpha191_test3/snapshot_fast.py` |
| `data/factor_snapshots`（legacy，1616 个） | **219** | ✅ **含（且被污染）** | `strategies/alpha191/snapshots.py` 等 |
| `data/wyckoff20_v5_snapshots` | 124（因子名 + `code`/`industry`/`total_mv`） | ❌ **不含** | `wyckoff20/build_snapshots.py` |
| `data/wyckoff20_v10_recomd_snapshots` | 3 | ❌ 不含 | V10 策略 |

**核查（当前安全性）**：grep `snapshot['amount']` / `snap['vwap']` / `['raw_amount']` 于 `strategies/small_alpha191_test3/` 与 `strategies/wyckoff20/` ⇒ **0 处消费** ⇒ **目前无害**。

**隐患**：这两列是**被 ×`1/f` 污染的值**，且**未带任何口径标注**；下游若将来有人把它们当特征或当市值代理用，会**静默引入缺陷**。
⇒ **处置**：随 D-281 修复一并处理；并加 **R-147 扩展断言**（见 §6）。

---

## §6 对 D-281 修复的增量含义（并入修复方案）

| 增量项 | 内容 |
|---|---|
| **D-281 修复范围确认** | 修复**只需动 `local_provider.py` 两行**（`amount × 1/f → × f`）。**wyckoff20 侧无需改动**（已证免疫，改反而引入不可比） |
| **快照重算范围** | 只需重算 **`factor_snapshots_test`（1315 个）**；`wyckoff20_v5_snapshots` **不必重算**（因子不变）；`wyckoff20_v10_recomd_snapshots` 不必重算 |
| **额外断言（R-147 扩展）** | ①`low ≤ vwap ≤ high` ②除权日 `close/vwap` 比值连续性 ≈ raw 口径 ③**快照自带的原始列必须带口径标注**（或干脆**不再落盘**，避免下游误用） |
| **新增登记** | **D-282**：wyckoff20 快照链路对 D-281 免疫（本条）+ **R-148**：跨口径不可直接比较（见路线 v13 §6） |

---

## §7 小白版四问（D-269）

**① 这段在解决什么问题？**
之前发现一个数据小毛病：**"成交额"这一列被按相反方向调了**（价格调高，成交额却调低）。alpha191 有 39 个因子会读"成交均价（VWAP）"这一列，所以它们受影响。你问的是：**威科夫那条线有没有同样的问题？**

**② 用了什么办法？**
三层查：**读代码**（谁读了这两列）、**算数**（拿真实数据验算）、**做实验**（把数据改对，重跑威科夫全部 40 个因子，看结果变不变）。关键区别是：alpha191 是**直接拿现成的"成交均价"列**，而威科夫是**自己用"高、低、收"三个价算均价**——而这三个价是**一起**被调整的，所以它们之间的关系不变。

**③ 结果是什么？**
- 威科夫 40 个因子 × 5 只股票：**一个都没变**（差异 0）。
- 为了确认不是"没比较出来"，我故意把价格整体乘 1.37，结果 40 个里有 12 个立刻变了 ⇒ **证明方法灵敏**。
- 威科夫快照里的"市值"字段与原始数据**逐位一致**（40/40，比值 1.0000000000）。
- 顺带查了 BARRA 那条线：它**直接读原始数据**，也只有一处用价格，**同样没问题**。

**④ 这对你意味着什么？**
- **威科夫那条线不用改、也不用重算**——修复只需要动 alpha191 侧的两行。
- **否定的是"这条缺陷波及面很广"这个担心**，不是否定缺陷本身：缺陷确实存在（且**至少从 6 月 10 日就在了**），只是**威科夫恰好绕开了它**（因为它自己算均价）。
- **两个要注意的**：①alpha191 的快照文件里**夹带了被污染的 `amount`/`vwap` 两列**，现在没人用，但**没标注口径**，以后有人拿去用就会踩雷 ⇒ 修复时一并处理；②这也说明一件事：**"自己不读那一列"比"那一列是对的"更可靠**——威科夫的做法值得保留。
