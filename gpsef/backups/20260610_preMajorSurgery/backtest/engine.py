# -*- coding: utf-8 -*-

"""
GPSEF 事件驱动回测引擎

功能:
    - 事件驱动回测循环
    - 支持多种策略类型
    - 自动生成绩效报告
    - 支持基准对比分析

核心概念:
    - 事件驱动: 每日调用策略的before_market_open和handle_bar
    - 前复权: 所有价格数据统一复权到 canonical reference_date（数据最大日期）
    - 股票池: 动态过滤ST、次新股等
    - 模拟经纪商: 模拟真实交易环境（佣金、滑点、涨跌停等）

复权基准日设计:
    Canonical reference: 始终使用数据中可用最大日期（如2026-12-31）作为基准日。
    adjusted_price = raw × bf(t) / bf(canonical_ref)
    虽然 bf(ref) 在个股收益率中抵消（ret独立于ref），但 quantity = target_value / price
    依赖于绝对价格水平。不同 ref → 不同价格 → 不同股数 → 不同持仓 → 净值累积偏离。
    统一使用 canonical reference 确保不同 end_date 的回测结果完全可比较。

使用方法:
    from gpsef.backtest import BacktestEngine

    engine = BacktestEngine(config)
    report = engine.run(strategy)
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging

from gpsef.core import IStrategy, StrategyContext, Portfolio
from gpsef.data import LocalDataProvider
from gpsef.pool import DynamicStockPool
from gpsef.execution import SimulatedBroker

# 抑制matplotlib的INFO日志
logging.getLogger('matplotlib').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _to_scalar(val):
    """
    将Series或数组转换为标量值。

    输入:
        val: pandas.Series, numpy.ndarray 或其他类型
    输出:
        标量值或np.nan
    作用:
        确保后续运算使用标量而非数组
    调用方:
        BacktestEngine._update_portfolio, _calculate_benchmark_returns
    """
    if hasattr(val, 'iloc'):
        if len(val) > 0:
            val = val.iloc[0]
        else:
            return np.nan
    if hasattr(val, 'item'):
        val = val.item()
    return float(val)


class BacktestEngine:
    """
    事件驱动回测引擎。

    职责:
        - 管理回测生命周期
        - 驱动策略执行
        - 记录每日净值和持仓
        - 生成绩效报告

    组成:
        - 数据提供者: 提供历史价格数据
        - 股票池: 过滤候选股票
        - 经纪商: 模拟订单执行
        - 策略: 实现交易逻辑

    复权基准日:
        使用数据中可用最大日期作为 canonical reference。
        adjusted_price = raw × bf(t) / bf(canonical_ref)
        虽然个股收益率中 bf(ref) 抵消，但 quantity = target_value / price
        依赖于绝对价格水平。统一使用 canonical reference 确保不同 end_date
        的回测结果完全可比较。

    调用方:
        - run_backtest.py: 入口脚本
        - run_optimize.py: 优化器内部
    """

    def __init__(self, config: Dict[str, Any], output_dir: Optional[str] = None):
        """
        初始化回测引擎。

        输入:
            config: 策略配置字典，包含backtest、data、execution等配置
            output_dir: 输出目录路径，若为None则自动生成以策略名_回测区间_时间戳命名的目录
        输出:
            无
        作用:
            初始化数据提供者、股票池、经纪商、策略上下文等
        调用方:
            - run_backtest.py main函数
            - GenericOptimizationProblem.evaluate
        """
        self.config = config
        backtest_cfg = config['backtest']

        # 解析回测区间
        self.start_date = datetime.strptime(backtest_cfg['start_date'], '%Y%m%d')
        self.end_date = datetime.strptime(backtest_cfg['end_date'], '%Y%m%d')
        self.initial_cash = backtest_cfg['initial_cash']

        strategy_name = config['strategy']['name']

        # 自动生成输出目录: 策略名_开始日期_结束日期_时间戳
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            output_dir = f"./results/{strategy_name}_{backtest_cfg['start_date']}_{backtest_cfg['end_date']}_{timestamp}"

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"回测输出目录: {self.output_dir}")

        # 初始化数据提供者
        data_cfg = config['data']

        # 复权基准日：从配置文件读取，默认为回测结束日
        # 所有买卖和估值都使用此固定基准日，收益率自然包含分红信息
        ref_date_str = data_cfg.get('reference_date', backtest_cfg['end_date'])
        self.trade_reference_date = datetime.strptime(ref_date_str, '%Y%m%d')
        logger.info(f"复权基准日(reference_date): {self.trade_reference_date.date()}")

        self.data_provider = LocalDataProvider(data_cfg['base_path'])

        # 获取完整日历（扩展500天，确保策略能获取充足历史数据）
        extended_start = self.start_date - timedelta(days=500)
        full_calendar = self.data_provider.get_trade_calendar(extended_start, self.end_date)
        self.calendar = [d for d in full_calendar if d >= self.start_date]
        self.full_calendar = full_calendar

        logger.info(f"回测区间: {self.start_date.date()} 至 {self.end_date.date()}")
        logger.info(f"交易日数量: {len(self.calendar)}")

        # 初始化股票池
        pool_cfg = data_cfg.get('stock_pool', {})
        self.stock_pool = DynamicStockPool(pool_cfg)

        # 初始化股票池数据
        if hasattr(self.stock_pool, 'initialize'):
            logger.info("正在初始化股票池...")
            min_data_date = self.full_calendar[0] - timedelta(days=100)
            self.stock_pool.initialize(self.data_provider, min_date=min_data_date)

        # 初始化投资组合
        self.portfolio = Portfolio(self.initial_cash)

        # 初始化经纪商（传递固定的trade_reference_date）
        exec_cfg = config.get('execution', {})
        self.broker = SimulatedBroker(
            self.portfolio, self.data_provider, exec_cfg, self.trade_reference_date
        )

        # 加载基准数据
        self.benchmark_data = None
        benchmark_code = backtest_cfg.get('benchmark_code')
        if benchmark_code:
            self.benchmark_data = self._load_benchmark_data(benchmark_code)

        # 构造策略上下文
        self.context = StrategyContext()
        self.context.config = config
        self.context.data_provider = self.data_provider
        self.context.stock_pool = self.stock_pool
        self.context.broker = self.broker
        self.context.portfolio = self.portfolio
        self.context.calendar = self.calendar
        self.context.full_calendar = self.full_calendar

        # 回测记录
        self.daily_records = []
        self.daily_positions = []

        # ---------- 回测日志写入本地文件 ----------
        self._log_file_handler = logging.FileHandler(
            os.path.join(self.output_dir, 'backtest.log'),
            encoding='utf-8'
        )
        self._log_file_handler.setLevel(logging.DEBUG)
        self._log_file_handler.setFormatter(
            logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
        )
        for logger_name in ['strategies', 'gpsef']:
            _log = logging.getLogger(logger_name)
            _log.addHandler(self._log_file_handler)
        logging.getLogger(__name__).addHandler(self._log_file_handler)

    def _update_portfolio(self, date: datetime):
        """
        更新持仓市值。

        输入:
            date: 当前交易日期
        输出:
            无
        作用:
            获取持仓股票的最新收盘价，计算最新市值。
            使用固定的trade_reference_date作为复权基准日。
        调用方:
            BacktestEngine.run

        注意:
            reference_date=trade_reference_date（固定值），
            确保估值价格与买卖价格在同一个前复权体系下。
        """
        prices = {}
        for code in list(self.portfolio.positions.keys()):
            df = self.data_provider.get_daily_data(
                code, date, date, reference_date=self.trade_reference_date
            )
            if not df.empty:
                price = _to_scalar(df.iloc[0]['close'])
                prices[code] = price
        self.portfolio.update_value(prices)

    def _load_benchmark_data(self, benchmark_code: str) -> Optional[pd.DataFrame]:
        """
        加载基准指数数据。

        输入:
            benchmark_code: 基准指数代码，如'sh.000300'
        输出:
            基准指数日线DataFrame或None
        作用:
            从baostock获取基准指数数据用于对比分析
        调用方:
            BacktestEngine.__init__
        """
        try:
            import baostock as bs
            from strategies.alpha191.data_utils import get_baostock_session

            session = get_baostock_session()
            session.ensure_logged_in()

            start_fmt = self.start_date.strftime('%Y-%m-%d')
            end_fmt = self.end_date.strftime('%Y-%m-%d')

            rs = bs.query_history_k_data_plus(
                benchmark_code,
                "date,code,open,high,low,close,preclose,volume,amount,pctChg",
                start_date=start_fmt,
                end_date=end_fmt,
                frequency="d"
            )

            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())

            if data_list:
                df = pd.DataFrame(data_list, columns=rs.fields)
                df['date'] = pd.to_datetime(df['date'])

                numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'pctChg']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                df = df.rename(columns={'date': 'trade_date'})
                logger.info(f"基准数据获取成功: {len(df)}条")
                return df
            else:
                logger.warning("基准数据获取返回空列表")
                return None

        except Exception as e:
            logger.warning(f"获取基准数据失败: {e}")
            return None

    def _calculate_benchmark_returns(self) -> Optional[pd.DataFrame]:
        """
        计算基准累计收益率。

        输入:
            无
        输出:
            包含trade_date和benchmark_cumulative列的DataFrame
        作用:
            计算基准指数从起始到当前的累计收益率
        调用方:
            BacktestEngine._generate_report
        """
        if self.benchmark_data is None:
            return None

        benchmark = self.benchmark_data[
            (self.benchmark_data['trade_date'] >= self.start_date) &
            (self.benchmark_data['trade_date'] <= self.end_date)
            ].copy()

        if benchmark.empty:
            return None

        benchmark = benchmark.sort_values('trade_date').reset_index(drop=True)
        benchmark['benchmark_return'] = benchmark['close'].pct_change()
        benchmark['benchmark_cumulative'] = (benchmark['close'] / benchmark['close'].iloc[0] - 1) * 100

        return benchmark[['trade_date', 'benchmark_cumulative']].rename(columns={'trade_date': 'date'})

    def run(self, strategy: IStrategy) -> Dict[str, Any]:
        """
        执行回测。

        输入:
            strategy: 策略实例（必须实现IStrategy接口）
        输出:
            绩效指标字典
        作用:
            - 初始化策略
            - 逐日执行事件驱动循环
            - 记录每日数据
            - 生成绩效报告
        调用方:
            - run_backtest.py
            - GenericOptimizationProblem.evaluate
        """
        logger.info(f"回测开始: {self.start_date.date()} -> {self.end_date.date()}")
        logger.info(f"初始资金: {self.initial_cash:,.2f}")
        logger.info(f"复权基准日: {self.trade_reference_date.date()}")

        # 初始化策略
        strategy.initialize(self.context)

        # 事件驱动循环
        for i, date in enumerate(self.calendar):
            self.context.current_dt = date
            self.broker.set_current_date(date)

            # 盘前更新持仓估值
            self._update_portfolio(date)

            # 记录每日开盘前状态
            daily = {
                'date': date,
                'cash': self.portfolio.cash,
                'position_value': self.portfolio.total_value - self.portfolio.cash,
                'total_value': self.portfolio.total_value,
                'daily_return': 0.0
            }

            # 计算日收益率
            if i > 0:
                prev = self.daily_records[-1]['total_value']
                daily['daily_return'] = (self.portfolio.total_value / prev) - 1

            self.daily_records.append(daily)

            # 调用策略钩子
            strategy.before_market_open(self.context)
            strategy.handle_bar(self.context)

            # 盘后更新持仓估值
            self._update_portfolio(date)
            self.daily_records[-1]['total_value_eod'] = self.portfolio.total_value

            # 记录持仓快照
            positions_snapshot = {
                'date': date,
                'positions': {code: self.portfolio.positions[code] for code in self.portfolio.positions}
            }
            self.daily_positions.append(positions_snapshot)

            # 进度日志
            if (i + 1) % 20 == 0 or i == 0:
                logger.info(f"回测进度: {i + 1}/{len(self.calendar)} ({100 * (i + 1) / len(self.calendar):.1f}%)")

        logger.info("回测完成，正在生成报告...")

        # 生成绩效报告
        return self._generate_report()

    def _generate_report(self) -> Dict[str, Any]:
        """
        生成绩效报告。

        输入:
            无
        输出:
            绩效指标字典
        作用:
            - 计算各类绩效指标
            - 保存CSV和JSON报告
            - 生成可视化图表
        调用方:
            BacktestEngine.run
        """
        df = pd.DataFrame(self.daily_records)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # 合并基准收益
        if self.benchmark_data is not None:
            benchmark_df = self._calculate_benchmark_returns()
            if benchmark_df is not None:
                benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
                benchmark_df.set_index('date', inplace=True)
                df = df.merge(benchmark_df[['benchmark_cumulative']], left_index=True, right_index=True, how='left')

        if 'benchmark_cumulative' not in df.columns:
            df['benchmark_cumulative'] = np.nan

        # ===== 计算绩效指标 =====

        # 基础收益指标
        total_return = (df['total_value_eod'].iloc[-1] / self.initial_cash) - 1
        daily_returns = df['daily_return'].dropna()
        n_days = len(df)
        years = n_days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        annual_vol = daily_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # 最大回撤
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        max_dd_date = drawdown.idxmin() if not drawdown.empty else None

        # 胜率与盈亏比
        win_rate = (daily_returns > 0).mean()
        avg_gain = daily_returns[daily_returns > 0].mean()
        avg_loss = abs(daily_returns[daily_returns < 0].mean())
        profit_loss_ratio = avg_gain / avg_loss if avg_loss > 0 else 0

        # 组装指标
        metrics = {
            'strategy': self.config['strategy']['name'],
            'backtest_period': f"{self.start_date.date()} to {self.end_date.date()}",
            'initial_cash': self.initial_cash,
            'final_value': df['total_value_eod'].iloc[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'max_drawdown_date': str(max_dd_date.date()) if max_dd_date else None,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'total_commission': self.broker.total_commission,
            'total_tax': self.broker.total_tax,
            'total_orders': len(self.broker.orders),
        }

        # ===== 保存报告文件 =====

        # 每日净值
        df.to_csv(os.path.join(self.output_dir, 'daily_values.csv'))

        # 订单记录
        if self.broker.orders:
            pd.DataFrame(self.broker.orders).to_csv(
                os.path.join(self.output_dir, 'orders.csv'), index=False
            )

        # 持仓记录
        if self.daily_positions:
            pos_records = []
            for rec in self.daily_positions:
                date = rec['date']
                for code, pos in rec['positions'].items():
                    pos_records.append({
                        'date': date,
                        'code': code,
                        'quantity': pos.quantity,
                        'avg_cost': pos.avg_cost,
                        'market_value': pos.market_value,
                    })
            if pos_records:
                pd.DataFrame(pos_records).to_csv(
                    os.path.join(self.output_dir, 'positions.csv'), index=False
                )

        # JSON报告
        with open(os.path.join(self.output_dir, 'report.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        # 文本报告
        with open(os.path.join(self.output_dir, 'report.txt'), 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"策略回测报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"策略名称: {metrics['strategy']}\n")
            f.write(f"回测区间: {metrics['backtest_period']}\n")
            f.write(f"复权基准日: {self.trade_reference_date.date()}\n")
            f.write(f"初始资金: {metrics['initial_cash']:,.0f}\n")
            f.write(f"最终资金: {metrics['final_value']:,.0f}\n")
            f.write(f"总收益率: {metrics['total_return']:.2%}\n")
            f.write(f"年化收益率: {metrics['annual_return']:.2%}\n")
            f.write(f"年化波动率: {metrics['annual_volatility']:.2%}\n")
            f.write(f"夏普比率: {metrics['sharpe_ratio']:.2f}\n")
            f.write(f"最大回撤: {metrics['max_drawdown']:.2%} (日期: {metrics['max_drawdown_date']})\n")
            f.write(f"日胜率: {metrics['win_rate']:.2%}\n")
            f.write(f"盈亏比: {metrics['profit_loss_ratio']:.2f}\n")
            f.write(f"总佣金: {metrics['total_commission']:.2f}\n")
            f.write(f"总印花税: {metrics['total_tax']:.2f}\n")
            f.write(f"总订单数: {metrics['total_orders']}\n")
            f.write("=" * 60 + "\n")

        # 绘制图表
        self._plot_performance(df, metrics, drawdown)

        logger.info(f"绩效报告已保存: {self.output_dir}")

        return metrics

    def _plot_performance(self, df: pd.DataFrame, metrics: Dict, drawdown: pd.Series):
        """
        绘制绩效图表。

        输入:
            df: 每日净值DataFrame
            metrics: 绩效指标字典
            drawdown: 回撤序列
        输出:
            无
        作用:
            生成净值曲线和回撤曲线图表
        调用方:
            BacktestEngine._generate_report
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            # 净值曲线
            ax1 = axes[0]
            nav = df['total_value_eod'] / self.initial_cash
            ax1.plot(nav.index, nav, label='Strategy NAV', linewidth=2, color='blue')

            if 'benchmark_cumulative' in df.columns:
                benchmark_nav = 1 + df['benchmark_cumulative'] / 100
                ax1.plot(benchmark_nav.index, benchmark_nav, label='Benchmark',
                         linewidth=2, color='red', linestyle='--')

            ax1.set_title('Strategy vs Benchmark')
            ax1.set_ylabel('Net Asset Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

            # 回撤曲线
            ax2 = axes[1]
            valid_idx = drawdown.index
            ax2.fill_between(valid_idx, drawdown * 100, 0, color='red', alpha=0.3)
            ax2.plot(valid_idx, drawdown * 100, color='red', linewidth=1)
            ax2.set_title(f'Drawdown (Max: {metrics["max_drawdown"] * 100:.2f}%)')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'performance.png'), dpi=150)
            plt.close()

        except Exception as e:
            logger.warning(f"绘图失败: {e}")
