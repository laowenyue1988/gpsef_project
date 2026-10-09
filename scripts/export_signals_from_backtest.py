# -*- coding: utf-8 -*-
"""
export_signals_from_backtest.py — 从任意 gpsef 回测结果目录导出 Eureka 实盘信号

一行命令完成: 给定回测结果目录 → 自动定位 orders.csv + backtest.log →
调用 export_signals.py 标准导出逻辑(协议v3: weight+sigma+scale) →
输出信号 JSON 到 Eureka signals 目录。

config 选择优先级:
  1. 命令行 --config 显式指定
  2. 结果目录内 config.yaml (回测时保存的副本, sigma/scale 口径与该回测一致)
  3. 策略默认 config.yaml (strategies/small_alpha191_test2/config.yaml)

用法 (PyCharm Terminal, PowerShell 语法):
  .venv\\Scripts\\python.exe scripts\\export_signals_from_backtest.py ^
      --result-dir results\\SmallAlpha191Test2_20260522_20260831_20260831171030

  指定输出目录 / 指定 config:
  .venv\\Scripts\\python.exe scripts\\export_signals_from_backtest.py ^
      --result-dir results\\SmallAlpha191Test2_20260522_20260831_20260831171030 ^
      --config strategies\\small_alpha191_test2\\config.yaml ^
      --out E:\\quant\\Eureka\\signals
"""
import argparse
import os
import sys

# 让 import export_signals 可用 (scripts/ 同级)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def resolve_result_dir(project_root: str, result_dir: str) -> str:
    """相对路径按 results/ 解析, 绝对路径原样使用;
    兼容带或不带 results\\ 前缀的写法"""
    if os.path.isabs(result_dir):
        return os.path.normpath(result_dir)
    if result_dir.replace("\\", "/").startswith("results/"):
        return os.path.normpath(os.path.join(project_root, result_dir))
    return os.path.normpath(os.path.join(project_root, "results", result_dir))


def main():
    ap = argparse.ArgumentParser(
        description="从 gpsef 回测结果目录导出 Eureka 实盘信号 (协议v3: weight+sigma+scale)")
    ap.add_argument("--result-dir", required=True,
                    help="回测结果目录: 相对 results/ 的子目录名, 或绝对路径")
    ap.add_argument("--config", default="",
                    help="gpsef 策略 config.yaml 路径 (默认: 结果目录内 config.yaml, 其次策略默认)")
    ap.add_argument("--out", default=r"E:\quant\Eureka\signals",
                    help="信号输出目录 (默认 E:\\quant\\Eureka\\signals)")
    ap.add_argument("--cash", type=float, default=1_000_000.0,
                    help="初始资金(仅参考, 默认 1000000)")
    args = ap.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rdir = resolve_result_dir(project_root, args.result_dir)

    # ── 校验结果目录与关键文件 ──
    if not os.path.isdir(rdir):
        print(f"[ERR] 回测结果目录不存在: {rdir}")
        sys.exit(1)
    orders = os.path.join(rdir, "orders.csv")
    log = os.path.join(rdir, "backtest.log")
    for p, tag in ((orders, "orders.csv"), (log, "backtest.log")):
        if not os.path.exists(p):
            print(f"[ERR] 结果目录缺少 {tag}: {p}")
            sys.exit(1)

    # ── config 选择: 显式 > 结果目录 config.yaml > 策略默认 ──
    if args.config:
        cfg = args.config
        print(f"[i] 使用命令行指定 config: {cfg}")
    elif os.path.exists(os.path.join(rdir, "config.yaml")):
        cfg = os.path.join(rdir, "config.yaml")
        print(f"[i] 使用结果目录内 config.yaml: {cfg}")
    else:
        cfg = os.path.join(project_root, "strategies", "small_alpha191_test2", "config.yaml")
        print(f"[i] 使用策略默认 config.yaml: {cfg}")
    if not os.path.exists(cfg):
        print(f"[ERR] config 不存在: {cfg}")
        sys.exit(1)

    # ── 调用 export_signals.py 标准导出 (同一套协议v3逻辑, 不重复实现) ──
    import export_signals
    sys.argv = ["export_signals.py",
                "--orders", orders,
                "--log", log,
                "--config", cfg,
                "--out", args.out,
                "--cash", str(args.cash)]
    export_signals.main()


if __name__ == "__main__":
    main()
