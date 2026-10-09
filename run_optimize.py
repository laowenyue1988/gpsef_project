#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPSEF 通用超参数优化入口（完全解耦版 + 通用详细日志）

用法:
    python run_optimize.py --config strategies/alpha191/config.yaml --auto
    python run_optimize.py --config strategies/alpha191/config.yaml --stage 1 --trials 30 --jobs 1

优化结果将自动保存为 JSON 文件，命名规则：
    {策略名}_{回测开始日期}_{回测结束日期}_{时间戳}_best_params.json
"""

import argparse
import yaml
import importlib
import tempfile
import logging
import sys
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

from gpsef.optimization import MultiStageHyperOptimizer, OptimizationProblem
from gpsef.backtest import BacktestEngine
from gpsef.utils.logger import setup_logger

logger = setup_logger('optimize', level=logging.INFO)


class GenericOptimizationProblem(OptimizationProblem):
    """
    通用优化问题类，完全依赖策略包提供的接口。
    """

    def __init__(self, base_config: Dict[str, Any], strategy_package):
        """
        初始化通用优化问题。

        输入：
            base_config (Dict[str, Any]): 基础配置字典（从 config.yaml 读取）。
            strategy_package (module): 策略包模块对象。

        输出：无
        """
        super().__init__(base_config)
        self.strategy_package = strategy_package
        # 从策略包获取执行参数键列表（若未定义则为空）
        self.exec_keys: List[str] = getattr(strategy_package, 'EXEC_PARAM_KEYS', [])
        # 获取需要调试的 logger 名称列表（若未定义则使用默认值）
        self.debug_loggers: List[str] = getattr(
            strategy_package, 'DEBUG_LOGGERS',
            ['gpsef.backtest', 'gpsef.execution', strategy_package.__name__]
        )
        # 必须存在 create_param_space 函数
        if not hasattr(strategy_package, 'create_param_space'):
            raise AttributeError(
                f"策略包 {strategy_package.__name__} 必须定义 create_param_space 函数"
            )
        self.create_param_space_func = strategy_package.create_param_space

    def _setup_detailed_logging(self):
        """
        重置日志配置，确保子进程中输出详细日志到控制台。

        输入：无
        输出：无
        调用方：evaluate 方法，每次试验前调用。
        """
        root_logger = logging.getLogger()
        # 清除所有已有 handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        for name in self.debug_loggers:
            lg = logging.getLogger(name)
            lg.setLevel(logging.DEBUG)
            lg.addHandler(console_handler)
            lg.propagate = False

        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(console_handler)

    def create_param_space(self, trial, stage: int, fixed_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建当前阶段的超参数搜索空间。

        输入：
            trial: Optuna 试验对象。
            stage (int): 当前阶段编号。
            fixed_params (Dict[str, Any]): 前序阶段已确定的最优参数。

        输出：
            Dict[str, Any]: 本次试验的完整参数字典。

        调用方：MultiStageHyperOptimizer 在每次试验时调用。
        """
        params = self.create_param_space_func(trial, stage, fixed_params)
        params.update(fixed_params)
        return params

    def evaluate(self, params: Dict[str, Any]) -> float:
        """
        评估一组参数的性能（运行一次回测）。

        输入：
            params (Dict[str, Any]): 本次试验的参数组合。

        输出：
            float: 优化目标值（当前为夏普比率）。

        调用方：Optuna 在每次试验时调用。
        """
        self._setup_detailed_logging()

        cfg = self.base_config.copy()

        # 将执行参数移入 execution 子配置
        if self.exec_keys:
            if 'execution' not in cfg:
                cfg['execution'] = {}
            for k in self.exec_keys:
                if k in params:
                    cfg['execution'][k] = params.pop(k)

        cfg.update(params)

        # 动态加载策略类
        strategy_path = cfg['strategy']['class']
        module_name, class_name = strategy_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        strategy_cls = getattr(module, class_name)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = BacktestEngine(cfg, output_dir=tmpdir)
            strategy = strategy_cls()
            report = engine.run(strategy)
            score = report.get('sharpe_ratio', -1e9)
            logger.info(f"Trial 完成，夏普: {score:.4f}")
            return score


def main():
    parser = argparse.ArgumentParser(description='GPSEF 通用超参数优化')
    parser.add_argument('--config', required=True, help='策略配置文件路径')
    parser.add_argument('--stage', type=int, help='运行指定的单个阶段')
    parser.add_argument('--auto', action='store_true', help='自动运行配置文件中定义的所有阶段')
    parser.add_argument('--trials', type=int, default=50, help='试验次数（单阶段模式）')
    parser.add_argument('--jobs', type=int, default=1, help='并行进程数（单阶段模式）')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna.db', help='Optuna 存储 URL')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)

    # 动态加载策略包
    strategy_path = config['strategy']['class']
    parts = strategy_path.split('.')
    if len(parts) < 3:
        logger.error(f"策略类路径格式错误，期望至少三段: {strategy_path}")
        sys.exit(1)

    package_name = '.'.join(parts[:-2])
    strategy_package = importlib.import_module(package_name)

    problem = GenericOptimizationProblem(config, strategy_package)
    study_name = config['strategy']['name']
    optimizer = MultiStageHyperOptimizer(problem, study_name=study_name, storage=args.storage)

    final_params = None
    if args.auto:
        stage_configs = config.get('AUTO_STAGE_CONFIG', {})
        if not stage_configs:
            logger.error("配置文件中未定义 AUTO_STAGE_CONFIG，无法自动运行")
            sys.exit(1)
        final_params = optimizer.run_all_stages(stage_configs)
    elif args.stage:
        final_params = optimizer.run_stage(args.stage, args.trials, args.jobs)
    else:
        logger.error("必须指定 --stage 或 --auto")
        sys.exit(1)

    # 保存最优参数到规范命名的 JSON 文件
    if final_params:
        strategy_name = config['strategy']['name']
        backtest_cfg = config['backtest']
        start_date = backtest_cfg['start_date']
        end_date = backtest_cfg['end_date']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{strategy_name}_{start_date}_{end_date}_{timestamp}_best_params.json"
        output_path = os.path.join(os.getcwd(), filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_params, f, indent=4, default=str)

        print("\n最终累积最优参数:")
        for k, v in final_params.items():
            print(f"  {k}: {v}")
        print(f"\n最优参数已保存至: {output_path}")


if __name__ == '__main__':
    main()