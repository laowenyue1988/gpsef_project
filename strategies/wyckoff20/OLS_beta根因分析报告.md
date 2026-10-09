# OLS Beta 跨进程发散根因分析报告

## 结论摘要

经过对完整代码链路的逐行分析和控制实验，定位到 **三个层叠的根因**，按可能性排序：

---

## 🔴 根因 #1: `beta_hash` 对浮点精度过度敏感（确认 - 代码逻辑缺陷）

**位置**: `gpsef_strategy_v10.py` line ~2783-2787

```python
beta_hash = 0
for k, v in self._train_ols_betas.get('betas', {}).items():
    if isinstance(v, np.ndarray):
        beta_hash ^= hash(tuple(float(x) for x in v[:3]))
```

**问题**: `hash(float)` 基于浮点数的 **精确位表示**，即使差异在 1e-17 级别（float64 机器精度 ~2.2e-16），也会产生完全不同的 hash 值。

**实验证据** (cloud控制实验):
```
beta_a[0] = 1.23456789012344997358e-01
beta_b[0] = 1.23456789012345011236e-01  (差异 1.39e-17)
hash(float(beta_a[0])) = 284671973884085696
hash(float(beta_b[0])) = 284671973884085728  ← 完全不同!
```

模拟真实场景（8因子，33维beta向量，1e-17扰动）:
```
beta_hash_a = 5590383361877032103
beta_hash_b = 1656757653299325230  ← 完全不同!
```

**`neu_sum = -0.00 vs 0.00` 的解释**:
```python
# IEEE 754: -0.0 和 +0.0 数值相等但位表示不同
np.float64(-0.0).hex()  # '-0x0.0p+0'  
np.float64(+0.0).hex()  # '0x0.0p+0'
f"{-0.0:.2f}"  # '-0.00'
f"{+0.0:.2f}"  # '0.00'
```
OLS残差理论sum应为0，实际在1e-12级别。不同进程残差sum在0两侧微小波动，产生负零/正零。

---

## 🔴 根因 #2: `np.linalg.lstsq` 的浮点累加顺序敏感性（确认）

**位置**: `gpsef/utils/neutralizer.py` line ~228-230

```python
Betas, _residuals_ls, _rank, _s = np.linalg.lstsq(X_v, Y_v, rcond=None)
```

**控制实验** (cloud, 703613行×33列，模拟真实数据规模):
```
同一数据，不同行序 → lstsq 结果差异:
  max_diff = 1.17e-17 (机器精度级别)
  但 hash 完全不同!
  beta_hash(行序1) = 1633779384581688444
  beta_hash(行序2) = 6860546727556995936  ← 完全不同!
```

**机制**: `np.linalg.lstsq` 调用 LAPACK `xGELSD`，内部矩阵乘法 `X'X` 和 `X'Y` 的浮点累加顺序依赖行序。不同行序 → 不同累加顺序 → 1e-17级差异。

**即使 BLAS 已设为单线程**（策略代码 line 24-31 已设置），同进程内同数据同行序的 lstsq 是确定性的，但 **跨进程的行序差异** 仍会导致 beta 差异。

---

## 🟡 根因 #3: `_row_idx` 排序 tiebreaker 的潜在失效场景

**位置**: `gpsef_strategy_v10.py` line ~2685-2687

```python
data['_row_idx'] = np.arange(len(data))
data = data.sort_values(['snapshot_date', 'code', '_row_idx'], kind='mergesort')
```

**分析**: `_row_idx` 是 `pd.concat` 后的位置索引。如果两个进程以不同顺序加载 parquet 文件行：
- `_row_idx` 对同一行数据赋值不同
- 排序键变为 `(snapshot_date, code, 不同_row_idx)`
- **如果 `(snapshot_date, code)` 完全唯一**：排序结果仍然相同（`_row_idx` 仅作 tiebreaker）
- **如果存在重复 `(snapshot_date, code)`**：排序结果不同 → 行数据不同 → lstsq 输入不同

**控制实验**:
```
无重复code: ✓ 排序后行序完全相同
有重复code: ✗ 排序后行序不同! 差异行数=10
```

**快照数据是否可能有重复 code**: 取决于 `barra_builder.py` 的快照构建逻辑和 parquet 文件内容。

---

## 已排除的假设

| 假设 | 排除原因 | 代码位置 |
|------|---------|---------|
| BLAS 多线程非确定性 | line 24-31 已设 `OPENBLAS_NUM_THREADS=1` 等，line 82-84 用 `threadpool_limits(limits=1)` | gpsef_strategy_v10.py:24-31, 82-84 |
| `_build_neutral_X` 列序不确定 | `sorted(df['industry'].unique())` 显式排序 categories；`pd.get_dummies` 按 sorted categories 生成列 | gpsef_strategy_v10.py:1031-1037 |
| `ALL_FACTORS` 迭代序不确定 | 是 list 字面量，不是 set/dict | gpsef_strategy_v10.py:355-364 |
| `feats = sorted(set(...))` | sorted() 保证确定性 | gpsef_strategy_v10.py:2823 |
| `groups` dict 迭代序不确定 | Python 3.7+ dict 保插入序；bytes key 不受 PYTHONHASHSEED 影响 | neutralizer.py:192-195 |
| PYTHONHASHSEED 直接影响 | line 35-37 已检测并警告；且关键路径无 str-key set/dict 迭代 | gpsef_strategy_v10.py:35-37 |

---

## 关键证据链

```
raw_data_sum 相同 (5858185398069.04)
  ↓ 但 sum 是聚合统计，无法检测行序差异
SortCheck 前5行相同
  ↓ 仅验证头部，703613行中可能尾部不同
beta_hash 不同 (2731324588465921574 vs 2187575417533370020)
  ↓ hash(float) 对 1e-17 差异极度敏感
neu_sum: -0.00 vs 0.00
  ↓ IEEE 754 负零/正零，实际值 ~1e-12
  ↓ 证明残差在0附近有微小差异
```

**核心逻辑**: 即使数据内容相同，只要 **行序有微小差异** → lstsq 浮点累加顺序不同 → beta 差异 ~1e-17 → hash 完全不同。

---

## 诊断脚本

已创建诊断脚本 `diag_ols_root_cause_v2.py`，在 Lao 桌面执行:
```
python E:\quant\gpsef_project\strategies\wyckoff20\diag_ols_root_cause_v2.py
```

脚本会检测:
1. **快照中重复 code 数量** → 验证根因 #3
2. **`_build_neutral_X` 两次构建的 X 矩阵对比** → 验证 X 确定性
3. **lstsq 同进程 10 次运行的确定性** → 验证 BLAS 确定性
4. **行序对 lstsq 的影响** → 量化行序敏感性
5. **加载顺序对排序后行序的影响** → 验证 `_row_idx` 有效性

---

## 修复方案

### 方案 A: 修复 `beta_hash` 比较（推荐 - 解决症状）

将 `beta_hash` 改为容忍浮点精度差异的比较方式:

```python
# 替换 beta_hash 计算 (line ~2783)
import hashlib
beta_bytes = b''
for k, v in sorted(self._train_ols_betas.get('betas', {}).items()):
    if isinstance(v, np.ndarray):
        beta_bytes += np.round(v[:3], decimals=10).tobytes()
beta_hash_str = hashlib.md5(beta_bytes).hexdigest()[:16]
logger.info(f"[PostNeutralize] ... beta_hash={beta_hash_str} ...")
```

**原理**: `np.round(v, 10)` 消除 1e-17 级差异，`md5(bytes)` 产生稳定的 hex 字符串。

### 方案 B: 消除行序敏感性（根治 - 需验证重复 code）

如果诊断发现重复 code:
```python
# 在数据拼接后立即去重
data = data.drop_duplicates(subset=['snapshot_date', 'code'], keep='first')
data['_row_idx'] = np.arange(len(data))  # 去重后重新编号
```

### 方案 C: 全量行序验证（最严谨）

在 SortCheck 中比较所有行而非仅前5行:
```python
# 替换 SortCheck (line ~2755)
import hashlib
full_fingerprint = hashlib.md5(
    train_data[['snapshot_date', 'code']].to_csv(index=False).encode()
).hexdigest()
logger.info(f"[SortCheck] fingerprint={full_fingerprint}")
```

### 方案 D: PYTHONHASHSEED 启动修复

确保命令行启动前设置:
```powershell
# Windows
$env:PYTHONHASHSEED="0"; python your_script.py

# 或在 .bat 文件中
set PYTHONHASHSEED=0
python your_script.py
```

---

## 总结

| 根因 | 确认度 | 影响 | 修复难度 |
|------|--------|------|---------|
| `hash(float)` 对 1e-17 差异过度敏感 | ✅ 100% 确认 | beta_hash 误报差异 | 低 (改 hash 逻辑) |
| `lstsq` 浮点累加顺序依赖行序 | ✅ 实验确认 | beta 值 1e-17 级差异 | 中 (需确保行序一致) |
| 重复 code 导致 `_row_idx` tiebreaker 失效 | ⚠️ 待诊断验证 | 行序真正不同 | 低 (去重) |

**优先级**: 先运行诊断脚本确认是否有重复 code → 再决定采用方案 A/B/C。

方案 A 可立即实施（仅改诊断输出方式），方案 B/C 需先确认根因。
