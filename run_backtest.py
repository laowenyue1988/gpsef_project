#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用回测/选股入口。

模式:
  backtest (默认) — 完整 ML 训练→选股→交易→回测
  recomd           — 每日选股快照模式（不交易，结果存入 recomd 快照目录）

使用:
  python run_backtest.py --config strategies/wyckoff20/config.yaml
  python run_backtest.py --config strategies/wyckoff20/config_v7.yaml --mode recomd
"""
# Ensure deterministic execution: env vars MUST be set before Python starts.
# If not pre-set, set them and re-exec this process.
import os as _os_env
import sys as _sys_env
_DETERMINISTIC_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}
if _os_env.environ.get("PYTHONHASHSEED") != "0":
    for _k, _v in _DETERMINISTIC_ENV.items():
        _os_env.environ[_k] = _v
    _os_env.execvp(_sys_env.executable, [_sys_env.executable] + _sys_env.argv)
del _os_env, _sys_env, _DETERMINISTIC_ENV


import argparse
import os
import yaml
import importlib
import logging
from gpsef.backtest import BacktestEngine
from gpsef.utils.logger import setup_logger
import logging
logging.basicConfig(level=logging.DEBUG)

def main():
    parser = argparse.ArgumentParser(description="GPSEF 回测 / 选股入口")
    parser.add_argument('--config', required=True, help='策略配置文件路径')
    parser.add_argument('--output', type=str, default=None, help='可选：手动指定输出目录')
    parser.add_argument(
        '--mode', type=str, default=None,
        choices=['backtest', 'recomd'],
        help='运行模式: backtest (默认，完整回测) | recomd (每日选股快照，不交易)'
    )
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)

    # ---- 日志级别控制 ----
    # 屏蔽 numba JIT 编译内部 DEBUG 日志 (SSA/IR 重写管线刷屏, 与回测逻辑无关)。
    # 可用 config 的 logging.numba_level 覆盖 (如 DEBUG/INFO/WARNING/ERROR), 默认 WARNING。
    _nb_level = config.get('logging', {}).get('numba_level', 'WARNING')
    logging.getLogger('numba').setLevel(
        getattr(logging, str(_nb_level).upper(), logging.WARNING)
    )

    # 保存源文件路径（engine 用于复制 YAML 原文到 results 目录）
    config['_source_file'] = os.path.abspath(args.config)

    # ---- 模式注入 ----
    # CLI --mode 优先级高于配置文件 mode 字段
    if args.mode is not None:
        config['mode'] = args.mode
    elif 'mode' not in config:
        config['mode'] = 'backtest'

    logger = logging.getLogger(__name__)
    logger.info(f"运行模式: {config['mode']}")

    strategy_path = config['strategy']['class']
    module_name, class_name = strategy_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    strategy_cls = getattr(module, class_name)

    if config['mode'] == 'recomd':
        recomd_dir = config.get('recomd', {}).get(
            'snapshot_dir', './data/wyckoff20_v7_recomd_snapshots'
        )
        logger.info(f"recomd 快照输出目录: {recomd_dir}")

    engine = BacktestEngine(config, output_dir=args.output)
    strategy = strategy_cls()

    # ---- [FIX-DET-2026-09-10] 可复现性溯源: 运行前采集输入摘要 ----
    # 目的: 让"两次运行结果为何不同"可被机器判定, 而不是靠人工回忆改了哪些文件。
    # (事故背景: 2026-09-09 选股池缓存于两次运行之间被改写, 造成 +6.90% vs +6.27%)
    _prov = None
    _prov_pre = None
    try:
        from strategies.small_alpha191_test3 import determinism_provenance as _prov
        _prov_pre = _prov.snapshot_inputs(config, stage='pre')
    except Exception as _e:
        logging.getLogger(__name__).warning(f"[FIX-DET] 溯源模块加载失败(不影响回测): {_e}")

    report = engine.run(strategy)

    # 回测结束: 刷新策略落盘缓存 (small_alpha191 的 flush_smallcap_cache / big_alpha191 的 flush_bigcap_cache)
    for _flush_name in ('flush_smallcap_cache', 'flush_bigcap_cache'):
        if hasattr(strategy, _flush_name):
            try:
                getattr(strategy, _flush_name)()
            except Exception as _e:
                logging.getLogger(__name__).warning(f"刷新策略落盘缓存失败 ({_flush_name}): {_e}")

    # ---- [FIX-DET-2026-09-10] 可复现性溯源: 运行后复核输入 + 落盘 ----
    if _prov is not None:
        try:
            _prov_post = _prov.snapshot_inputs(config, stage='post')
            _runtime = {
                'pool_cache_status': getattr(strategy, '_pool_cache_status', None),
                'pool_cache_loaded_dates': getattr(strategy, '_pool_cache_loaded_dates', None),
                'pool_cache_recomputed': getattr(strategy, '_pool_cache_recomputed', None),
                'pool_cache_conflicts': getattr(strategy, '_pool_cache_conflicts', None),
                'pool_cache_stale_dates': getattr(strategy, '_pool_cache_stale_dates', None),
                'barra_snap_used': sorted(getattr(strategy, '_barra_snap_used', set()) or []),
                'barra_rebuilt_during_run': getattr(strategy, '_barra_rebuilt', None),
                'mode': config.get('mode'),
                'backtest_start': (config.get('backtest') or {}).get('start_date'),
                'backtest_end': (config.get('backtest') or {}).get('end_date'),
            }
            _path = _prov.write_provenance(engine.output_dir, config,
                                           pre=_prov_pre, post=_prov_post, runtime=_runtime)
            if _path:
                _mut = (_prov.load_provenance(engine.output_dir) or {}).get('input_mutation_detected')
                if _mut:
                    logging.getLogger(__name__).warning(
                        f"[FIX-DET] ⚠ 回测运行期间共享输入被改写: {_mut} → "
                        f"本次结果与输入未变的运行**不可直接对比** (已记入 {_path})")
                else:
                    logging.getLogger(__name__).info(f"[FIX-DET] 溯源已落盘: {_path}")
        except Exception as _e:
            logging.getLogger(__name__).warning(f"[FIX-DET] 溯源落盘失败(不影响回测): {_e}")

    if config['mode'] == 'recomd':
        recomd_dir = config.get('recomd', {}).get(
            'snapshot_dir', './data/wyckoff20_v7_recomd_snapshots'
        )
        if os.path.isdir(recomd_dir):
            files = [f for f in os.listdir(recomd_dir) if f.endswith('.parquet')]
            print(f"recomd 完成，共生成 {len(files)} 个快照，保存在: {recomd_dir}")
        else:
            print(f"recomd 完成，结果保存在: {engine.output_dir}")
    else:
        print(f"回测完成，结果保存在: {engine.output_dir}")

if __name__ == '__main__':
    main()
