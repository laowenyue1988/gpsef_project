#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPSEF 通用选股模式入口

用法:
    python run_select.py --config strategies/alpha191/config.yaml --date 20260418

如果不指定 --date，则默认使用当前日期（需确保是交易日）。
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
import yaml
import importlib
import logging
from datetime import datetime, timedelta
import sys
import os
from gpsef.data import LocalDataProvider
from gpsef.pool import DynamicStockPool
from gpsef.core import StrategyContext
from gpsef.utils.logger import setup_logger

logger = setup_logger('select', level=logging.INFO)
os.environ['GM_TOKEN'] = '177c16ceb2ea308dca384cda74a9386b1afb8376'
# 在 main() 之前或 main() 内最开始添加：
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('select_debug.log', encoding='utf-8')
    ]
)
# 同时降低特定模块日志等级，确保看到所有信息
logging.getLogger('strategies.alpha191').setLevel(logging.DEBUG)
logging.getLogger('gpsef').setLevel(logging.DEBUG)


def build_select_context(config: dict, target_date: datetime):
    """
    根据策略配置和目标日期，构造一个完整的选股用 StrategyContext。
    该函数完全模拟 BacktestEngine 的初始化过程，但不创建 Broker/Portfolio。
    """
    data_cfg = config['data']
    backtest_cfg = config.get('backtest', {})

    # 1. 数据提供者
    data_provider = LocalDataProvider(data_cfg['base_path'])

    # 2. 获取完整日历（向前回溯足够天数，与回测引擎一致）
    #    回测引擎默认向前扩展 500 天，这里也照做
    extended_start = target_date - timedelta(days=500)
    full_calendar = data_provider.get_trade_calendar(extended_start, target_date)
    full_calendar = [d for d in full_calendar if d <= target_date]
    if not full_calendar:
        raise ValueError("无法获取交易日历")

    # 若目标日不是交易日，自动向前取最近交易日
    if full_calendar[-1] != target_date:
        logger.warning(f"目标日期 {target_date.date()} 非交易日，使用最近交易日 {full_calendar[-1].date()}")
        target_date = full_calendar[-1]

    # 3. 股票池（根据配置创建，并调用 initialize）
    pool_cfg = data_cfg.get('stock_pool', {})
    stock_pool = DynamicStockPool(pool_cfg)

    # 如果股票池有 initialize 方法，调用之（与回测引擎一致）
    if hasattr(stock_pool, 'initialize'):
        logger.info("正在初始化股票池...")
        # min_date 取 full_calendar 最早日期再往前推 100 天，留足缓冲
        min_pool_date = full_calendar[0] - timedelta(days=100)
        stock_pool.initialize(data_provider, min_date=min_pool_date)

    # 4. 构造上下文对象
    context = StrategyContext()
    context.config = config
    context.data_provider = data_provider
    context.stock_pool = stock_pool
    context.calendar = full_calendar           # 策略可能遍历此列表
    context.full_calendar = full_calendar      # 策略预加载使用
    context.current_dt = target_date

    # 注意：不设置 broker 和 portfolio，选股模式不需要交易
    context.broker = None
    context.portfolio = None

    return context, target_date


def main():
    parser = argparse.ArgumentParser(description='GPSEF 通用选股')
    parser.add_argument('--config', required=True, help='策略配置文件路径')
    parser.add_argument('--date', type=str, help='选股日期 (YYYYMMDD)，默认今天')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)

    # 动态加载策略类
    strategy_path = config['strategy']['class']
    module_name, class_name = strategy_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    strategy_cls = getattr(module, class_name)

    # 确定目标日期
    if args.date:
        target_date = datetime.strptime(args.date, '%Y%m%d')
    else:
        target_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # 构建选股上下文（通用逻辑）
    context, actual_date = build_select_context(config, target_date)

    # 创建策略并初始化
    strategy = strategy_cls()
    strategy.initialize(context)

    # 强制触发调仓日选股
    # 将 trade_day_seq 和 rebalance_interval 设为合适值，保证首次调用就执行选股
    strategy.trade_day_seq = 0
    strategy.rebalance_interval = 2  # 使得 1 % 2 == 1，满足调仓条件

    # 执行盘前选股
    strategy.before_market_open(context)

    # 输出结果
    print("\n" + "=" * 60)
    print(f"选股日期: {actual_date.strftime('%Y-%m-%d')}")
    print("=" * 60)
    if hasattr(strategy, 'target_stocks') and strategy.target_stocks:
        print(f"选中股票数量: {len(strategy.target_stocks)}")
        print("股票列表:")
        for i, code in enumerate(strategy.target_stocks, 1):
            print(f"  {i:2d}. {code}")
    else:
        print("未选中任何股票")
    print("=" * 60)


if __name__ == '__main__':
    main()
