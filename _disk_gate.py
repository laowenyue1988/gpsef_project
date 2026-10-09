# -*- coding: utf-8 -*-
"""[R-183] 磁盘门控（起臂前 fail-closed 检查）

动机：C: 使用率 88%（可用 19GB）；实验产物在 E:，但**进程临时文件默认落 C:**。
      若 C: 写满 ⇒ 系统/Python/WorkBuddy 都可能失败 ⇒ 实验崩。
纪律（与 R-150 内存门同族）：**起臂前必查；不达阈值即拒绝起臂并告警**（fail-closed）。

阈值（可按需调整；改动须登记）：
  C: 可用 ≥ 8000 MB   （告警线 12000 MB）
  E: 可用 ≥ 50000 MB  （告警线 80000 MB）

用法：
  python _disk_gate.py            # 人类可读
  python _disk_gate.py --gate     # 仅门控：exit 0=通过 / 3=C不足 / 4=E不足 / 5=两者
"""
import ctypes
import sys


class _MS(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def free_mb(drive):
    """drive 形如 'C:' / 'E:'"""
    free = ctypes.c_ulonglong(0)
    total = ctypes.c_ulonglong(0)
    ok = ctypes.windll.kernel32.GetDiskFreeSpaceExW(
        ctypes.c_wchar_p(drive + "\\"), None, ctypes.byref(total), ctypes.byref(free))
    if not ok:
        return None, None
    MB = 1048576
    return free.value / MB, total.value / MB


TH = {"C:": (8000.0, 12000.0), "E:": (50000.0, 80000.0)}


def main():
    gate = "--gate" in sys.argv
    bad = 0
    lines = []
    for drv in ("C:", "E:"):
        f, t = free_mb(drv)
        if f is None:
            lines.append(f"  {drv} 读取失败 ★")
            bad |= 1 if drv == "C:" else 2
            continue
        g, w = TH[drv]
        tag = "OK" if f >= g else "★不足"
        warn = "（告警线以下）" if f < w else ""
        lines.append(f"  {drv} 可用 {f:8.0f} MB / 总 {t:8.0f} MB  门控线 {g:6.0f}  告警线 {w:6.0f}  {tag}{warn}")
        if f < g:
            bad |= 1 if drv == "C:" else 2
    if not gate:
        print("磁盘门控（R-183）：")
        for l in lines:
            print(l)
        print("  结论：", "通过 ✓" if bad == 0 else f"★未通过（code={bad}）")
    return bad


if __name__ == "__main__":
    sys.exit(main())
