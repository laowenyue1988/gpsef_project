# -*- coding: utf-8 -*-
"""[Q5BAR] ★起臂前「运行时依赖闭包」预检。

新纪律（源自 2026-10-08 INCIDENT）：
    **仅 import 顶层模块不够** —— 本次事故的直接原因是 `determinism_provenance` 被
    `strategy.py` 在**函数内** import（4 处），顶层预检**看不到**它，导致三臂起臂后
    6 分钟才崩。本脚本用 AST 递归扫描**全部 import 语句（含函数内、条件分支内）**，
    对包内模块逐个尝试真实 import。

用法: python _q5bar_preimport_check.py [包名] [种子模块]
     默认 包=strategies.small_alpha191  种子=strategy
退出码: 0 = 全绿; 1 = 有缺失
"""
import ast
import importlib
import io
import os
import sys

os.chdir(r'E:\quant\gpsef_project')
if os.path.abspath('.') not in sys.path:
    sys.path.insert(0, os.path.abspath('.'))
PKG = sys.argv[1] if len(sys.argv) > 1 else 'strategies.small_alpha191'
SEED = sys.argv[2] if len(sys.argv) > 2 else 'strategy'


def local_path(mod):
    """返回包内模块的本地文件路径（.py 优先，其次 .pyc（sourceless））。"""
    rel = os.path.join(*mod.split('.'))
    for c in (rel + '.py', rel + '.pyc', os.path.join(rel, '__init__.py')):
        if os.path.exists(c):
            return c
    return None


def resolve_rel(mod, level, cur):
    """把相对导入解析成绝对模块名。cur 如 'strategies.small_alpha191.strategy'。"""
    if level == 0:
        return mod
    parts = cur.split('.')
    base = parts[:max(0, len(parts) - level)]
    return '.'.join(base + ([mod] if mod else []))


def scan(path, cur):
    """AST 扫描一个文件里的**全部** import（含嵌套在函数/条件里的）。

    返回 [(dep, kind)]：kind='mod' 表示**模块**；kind='sym' 表示 `from X import a` 展开出的
    **符号**（不是模块）。★跨包存在性检查**只认 'mod'**，否则会把函数名误判成缺失模块。
    """
    t = io.open(path, encoding='utf-8', errors='ignore').read()
    tree = ast.parse(t)
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            m = resolve_rel(n.module, n.level, cur)
            if m:
                out.append((m, 'mod'))
                for a in n.names:                       # from X import a,b -> 也试 X.a（符号）
                    if a.name != '*':
                        out.append((m + '.' + a.name, 'sym'))
        elif isinstance(n, ast.Import):
            for a in n.names:
                out.append((a.name, 'mod'))
    return out


print('=' * 96)
print('[Q5BAR] 运行时依赖闭包预检  包=%s  种子=%s' % (PKG, SEED))
print('=' * 96)

seen, order, failed, external = set(), [], [], set()
LOCAL_ROOTS = ('strategies', 'gpsef')          # ★同项目内的本地包前缀
cross = []                                     # ★(src, dep) 跨包引用（同项目、非本包）
stack = [PKG + '.' + SEED]
while stack:
    mod = stack.pop()
    if mod in seen:
        continue
    seen.add(mod)
    p = local_path(mod)
    if p is None:
        continue                                        # 非本包模块，另测
    order.append((mod, p))
    if not p.endswith('.py'):
        # ★sourceless（.pyc）模块：无法 AST 扫描其依赖，但仍登记并做 import 测试
        continue
    for dep, kind in scan(p, mod):
        if dep == PKG or dep.startswith(PKG + '.'):   # ★须带**点边界**：否则 'a.b_c' 会被 'a.b' 误吞
            if local_path(dep) is not None:
                if dep not in seen:
                    stack.append(dep)
        else:
            external.add(dep.split('.')[0])
            # ★跨包检查只认「模块」（kind='mod'）：`from X import a` 里的 a 是**符号**不是模块
            if kind == 'mod' and dep.split('.')[0] in LOCAL_ROOTS:
                cross.append((mod, dep))       # ★同项目、非本包 ⇒ 须核目标模块是否存在

print('[1] 闭包内模块 %d 个（%s）' % (len(order), PKG))
for m, p in sorted(order):
    print('     %-58s %s' % (m.replace(PKG + '.', ''), os.path.basename(p)))

print('')
print('[2] 逐个真实 import（★含函数内/条件分支内的依赖）')
for m, p in sorted(order):
    try:
        mod = importlib.import_module(m)
        tag = 'OK  ' if getattr(mod, '__file__', None) else 'OK? '
        print('     %s %-52s file=%s' % (tag, m.replace(PKG + '.', ''),
                                         os.path.basename(str(getattr(mod, '__file__', '')))))
    except Exception as e:                              # noqa: BLE001
        failed.append((m, type(e).__name__, str(e)[:110]))
        print('     ★FAIL %-50s %s: %s' % (m.replace(PKG + '.', ''), type(e).__name__, str(e)[:90]))

print('')
cross_uniq = sorted(set(cross))
hard_cross = []
# ★豁免机制：默认 fail-closed；若某跨包缺失引用已被**书面登记**为「分支不可达」，
#    在 crosspkg_allowlist.txt 中以 `模块名|理由` 逐行登记 ⇒ 降级为「登记豁免」。
_AL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'crosspkg_allowlist.txt')
_allow = {}
if os.path.exists(_AL):
    for _l in io.open(_AL, encoding='utf-8', errors='ignore').read().splitlines():
        _l = _l.strip()
        if _l and not _l.startswith('#') and '|' in _l:
            _k, _v = _l.split('|', 1)
            _allow[_k.strip()] = _v.strip()
print('[2b] ★跨包引用扫描（同项目内、非本包 ⇒ 策略包迁移/替换后的定时炸弹）')
if not cross_uniq:
    print('     （无）')
for src, dep in cross_uniq:
    _ok = local_path(dep) is not None
    if _ok:
        _tag = 'OK  '
    elif dep in _allow:
        _tag = '豁免'
    else:
        _tag = '★缺 '
    print('     %s %-56s ← %s' % (_tag, dep, src.replace(PKG + '.', '')))
    if _tag == '豁免':
        print('          理由: %s' % _allow[dep])
    if not _ok and dep not in _allow:
        hard_cross.append((src, dep))
print('')
print('[3] 外部顶层依赖（信息）：%s' % ', '.join(sorted(external)))

print('')
print('=' * 96)
# ★【门控信号改为「标志文件」】—— 与 R-202 同源：判"是否通过"**只认产物**，禁用返回码。
#   原因：本环境下该脚本的进程退出码与判定结果出现过不一致；标志文件是跨进程可靠信号。
_SENT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tmp', 'gate0.result')


def _sentinel(v):
    try:
        os.makedirs(os.path.dirname(_SENT), exist_ok=True)
        io.open(_SENT, 'w', encoding='utf-8', newline='\n').write(v + '\n')
    except Exception:
        pass


if failed or hard_cross:
    if failed:
        print('❌ 预检 FAIL —— %d 个包内模块无法 import（**禁止起臂**）' % len(failed))
        for m, k, e in failed:
            print('     %s -> %s: %s' % (m, k, e))
    if hard_cross:
        print('❌ 预检 FAIL —— %d 处★跨包硬编码引用指向**不存在的模块**（**禁止起臂**）' % len(hard_cross))
        for src, dep in hard_cross:
            print('     %s  →  %s' % (src, dep))
        print('     ⇒ 处置：改为**包内导入**（或补 fallback 分支）；否则一旦执行到该分支必崩')
    _sentinel('FAIL')
    sys.exit(1)
print('✅ 预检 PASS —— 运行时依赖闭包完整（%d 模块全可 import）' % len(order))
_sentinel('PASS')
sys.exit(0)
