# -*- coding: utf-8 -*-
"""
gpsef/backtest/analyzers.py
通用绩效分析器 - 适用于任何策略的回测结果
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    通用绩效分析器

    用法:
        analyzer = PerformanceAnalyzer(result_dir='./results/backtest_xxx')
        metrics = analyzer.run()
        analyzer.plot(save_path='./results/performance.png')
    """

    def __init__(self, result_dir: str, benchmark_df: Optional[pd.DataFrame] = None):
        """
        Parameters
        ----------
        result_dir : str
            回测输出目录，应包含 daily_values.csv 和 orders.csv（可选）
        benchmark_df : pd.DataFrame, optional
            基准数据，需包含 'date' 和 'close' 列
        """
        self.result_dir = result_dir
        self.benchmark_df = benchmark_df

        self.daily_df: Optional[pd.DataFrame] = None
        self.orders_df: Optional[pd.DataFrame] = None
        self.metrics: Dict = {}

        self._load_data()

    def _load_data(self):
        """加载回测输出文件"""
        daily_path = os.path.join(self.result_dir, 'daily_values.csv')
        if os.path.exists(daily_path):
            self.daily_df = pd.read_csv(daily_path)
            self.daily_df['date'] = pd.to_datetime(self.daily_df['date'])
            self.daily_df.set_index('date', inplace=True)
            logger.info(f"加载每日净值数据: {len(self.daily_df)} 条")
        else:
            logger.warning(f"未找到 daily_values.csv: {daily_path}")

        orders_path = os.path.join(self.result_dir, 'orders.csv')
        if os.path.exists(orders_path):
            self.orders_df = pd.read_csv(orders_path)
            self.orders_df['date'] = pd.to_datetime(self.orders_df['date'])
            logger.info(f"加载订单数据: {len(self.orders_df)} 条")
        else:
            logger.info("未找到订单文件，跳过交易分析")

    def run(self) -> Dict:
        """执行所有分析，返回指标字典"""
        if self.daily_df is None or self.daily_df.empty:
            logger.error("无每日净值数据，无法分析")
            return {}

        self.metrics = {}
        self._calc_basic_metrics()
        self._calc_risk_metrics()
        self._calc_trade_metrics()

        if self.benchmark_df is not None:
            self._calc_benchmark_metrics()

        self._save_report()
        return self.metrics

    def _calc_basic_metrics(self):
        """基础收益指标"""
        df = self.daily_df
        initial_value = df['total_value_eod'].iloc[0] if 'total_value_eod' in df.columns else df['total_value'].iloc[0]
        final_value = df['total_value_eod'].iloc[-1] if 'total_value_eod' in df.columns else df['total_value'].iloc[-1]

        total_return = (final_value / initial_value) - 1
        n_days = len(df)
        n_years = n_days / 252

        if n_years > 0:
            annual_return = (1 + total_return) ** (1 / n_years) - 1
        else:
            annual_return = 0.0

        self.metrics.update({
            'start_date': df.index[0].strftime('%Y-%m-%d'),
            'end_date': df.index[-1].strftime('%Y-%m-%d'),
            'trading_days': n_days,
            'years': n_years,
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
        })

    def _calc_risk_metrics(self):
        """风险与调整收益指标"""
        df = self.daily_df
        returns = df['daily_return'].dropna()

        if len(returns) == 0:
            self.metrics.update({'volatility': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0})
            return

        annual_vol = returns.std() * np.sqrt(252)
        risk_free = 0.03
        sharpe = (self.metrics['annual_return'] - risk_free) / annual_vol if annual_vol > 0 else 0.0

        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin() if not drawdown.empty else None

        # Calmar
        calmar = self.metrics['annual_return'] / abs(max_dd) if max_dd != 0 else 0.0

        # 下行波动率 & 索提诺
        downside = returns[returns < 0]
        downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.0
        sortino = (self.metrics['annual_return'] - risk_free) / downside_std if downside_std > 0 else 0.0

        # 胜率 & 盈亏比
        win_rate = (returns > 0).mean()
        avg_gain = returns[returns > 0].mean() if (returns > 0).any() else 0.0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1.0
        profit_loss_ratio = avg_gain / avg_loss if avg_loss != 0 else 0.0

        self.metrics.update({
            'volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'max_drawdown_date': max_dd_date.strftime('%Y-%m-%d') if max_dd_date else None,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'avg_gain': avg_gain,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
        })

    def _calc_trade_metrics(self):
        """交易相关指标"""
        if self.orders_df is None or self.orders_df.empty:
            self.metrics.update({'total_orders': 0, 'total_commission': 0.0, 'total_tax': 0.0})
            return

        orders = self.orders_df
        buys = orders[orders['direction'] == 'buy']
        sells = orders[orders['direction'] == 'sell']

        total_commission = orders['commission'].sum() if 'commission' in orders else 0.0
        total_tax = orders['tax'].sum() if 'tax' in orders else 0.0

        # 交易盈亏分析（FIFO简化）
        trade_pnl = self._calc_trade_pnl()

        win_trades = [p for p in trade_pnl if p > 0]
        loss_trades = [p for p in trade_pnl if p < 0]
        trade_win_rate = len(win_trades) / len(trade_pnl) if trade_pnl else 0.0
        avg_win = np.mean(win_trades) if win_trades else 0.0
        avg_loss = abs(np.mean(loss_trades)) if loss_trades else 0.0
        profit_factor = sum(win_trades) / abs(sum(loss_trades)) if loss_trades else float('inf')

        self.metrics.update({
            'total_orders': len(orders),
            'buy_orders': len(buys),
            'sell_orders': len(sells),
            'total_commission': total_commission,
            'total_tax': total_tax,
            'total_trades': len(trade_pnl),
            'win_trades': len(win_trades),
            'loss_trades': len(loss_trades),
            'trade_win_rate': trade_win_rate,
            'avg_trade_pnl': np.mean(trade_pnl) if trade_pnl else 0.0,
            'avg_win_trade': avg_win,
            'avg_loss_trade': avg_loss,
            'profit_factor': profit_factor,
        })

    def _calc_trade_pnl(self) -> List[float]:
        """基于订单记录计算每笔完整交易的盈亏（简化版）"""
        if self.orders_df is None or self.orders_df.empty:
            return []

        orders = self.orders_df.sort_values(['code', 'date'])
        pnl_list = []
        positions = {}  # code -> {'quantity': int, 'cost': float}

        for _, row in orders.iterrows():
            code = row['code']
            direction = row['direction']
            quantity = row['quantity']
            price = row['price']
            commission = row.get('commission', 0.0)
            tax = row.get('tax', 0.0)

            if direction == 'buy':
                if code not in positions:
                    positions[code] = {'quantity': 0, 'cost': 0.0}
                positions[code]['quantity'] += quantity
                positions[code]['cost'] += quantity * price + commission
            else:  # sell
                if code not in positions or positions[code]['quantity'] == 0:
                    continue
                pos = positions[code]
                sell_qty = min(quantity, pos['quantity'])
                if sell_qty == 0:
                    continue
                avg_cost = pos['cost'] / pos['quantity']
                pnl = sell_qty * (price - avg_cost) - commission - tax
                pnl_list.append(pnl)
                pos['quantity'] -= sell_qty
                pos['cost'] -= sell_qty * avg_cost
                if pos['quantity'] == 0:
                    del positions[code]

        return pnl_list

    def _calc_benchmark_metrics(self):
        """与基准对比的指标"""
        if self.benchmark_df is None or self.benchmark_df.empty:
            return

        bench = self.benchmark_df.copy()
        bench['date'] = pd.to_datetime(bench['date'])
        bench = bench[(bench['date'] >= self.daily_df.index[0]) & (bench['date'] <= self.daily_df.index[-1])]
        bench = bench.sort_values('date').set_index('date')

        if 'close' not in bench.columns:
            logger.warning("基准数据缺少 close 列")
            return

        bench_return = bench['close'].pct_change().dropna()
        bench_total = (bench['close'].iloc[-1] / bench['close'].iloc[0]) - 1
        bench_annual = (1 + bench_total) ** (1 / self.metrics['years']) - 1 if self.metrics['years'] > 0 else 0.0

        # 对齐日期计算 Beta / Alpha
        merged = pd.DataFrame({
            'strategy': self.daily_df['daily_return'],
            'benchmark': bench_return
        }).dropna()

        if len(merged) > 1:
            cov = merged['strategy'].cov(merged['benchmark'])
            var = merged['benchmark'].var()
            beta = cov / var if var != 0 else 0.0
            alpha = self.metrics['annual_return'] - 0.03 - beta * (bench_annual - 0.03)
            tracking_error = (merged['strategy'] - merged['benchmark']).std() * np.sqrt(252)
            information_ratio = (self.metrics[
                                     'annual_return'] - bench_annual) / tracking_error if tracking_error > 0 else 0.0
        else:
            beta = alpha = information_ratio = 0.0

        self.metrics.update({
            'benchmark_return': bench_total,
            'benchmark_annual': bench_annual,
            'excess_return': self.metrics['total_return'] - bench_total,
            'beta': beta,
            'alpha': alpha,
            'information_ratio': information_ratio,
        })

    def _save_report(self):
        """保存 JSON 报告"""
        report_path = os.path.join(self.result_dir, 'performance_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        logger.info(f"绩效报告已保存: {report_path}")

        # 同时生成文本报告
        txt_path = os.path.join(self.result_dir, 'performance_report.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(self._format_text_report())
        logger.info(f"文本报告已保存: {txt_path}")

    def _format_text_report(self) -> str:
        """格式化文本报告"""
        m = self.metrics
        lines = []
        lines.append("=" * 70)
        lines.append("策略绩效报告")
        lines.append("=" * 70)
        lines.append(f"回测区间: {m.get('start_date', 'N/A')} 至 {m.get('end_date', 'N/A')}")
        lines.append(f"交易日数: {m.get('trading_days', 0)} ({m.get('years', 0):.2f}年)")
        lines.append(f"初始资金: {m.get('initial_value', 0):,.2f}")
        lines.append(f"最终资金: {m.get('final_value', 0):,.2f}")
        lines.append("")
        lines.append("收益率指标:")
        lines.append(f"  总收益率:        {m.get('total_return', 0) * 100:.2f}%")
        lines.append(f"  年化收益率:      {m.get('annual_return', 0) * 100:.2f}%")
        lines.append(f"  年化波动率:      {m.get('volatility', 0) * 100:.2f}%")
        lines.append(f"  夏普比率:        {m.get('sharpe_ratio', 0):.4f}")
        lines.append(f"  索提诺比率:      {m.get('sortino_ratio', 0):.4f}")
        lines.append(f"  最大回撤:        {m.get('max_drawdown', 0) * 100:.2f}%")
        lines.append(f"  最大回撤日期:    {m.get('max_drawdown_date', 'N/A')}")
        lines.append(f"  Calmar比率:      {m.get('calmar_ratio', 0):.4f}")
        lines.append("")
        lines.append("交易统计:")
        lines.append(f"  总订单数:        {m.get('total_orders', 0)}")
        lines.append(f"  总交易次数:      {m.get('total_trades', 0)}")
        lines.append(f"  胜率:            {m.get('trade_win_rate', 0) * 100:.2f}%")
        lines.append(f"  平均盈亏:        {m.get('avg_trade_pnl', 0):,.2f}")
        lines.append(f"  平均盈利:        {m.get('avg_win_trade', 0):,.2f}")
        lines.append(f"  平均亏损:        {m.get('avg_loss_trade', 0):,.2f}")
        lines.append(f"  盈亏比:          {m.get('profit_loss_ratio', 0):.4f}")
        lines.append(f"  总佣金:          {m.get('total_commission', 0):,.2f}")
        lines.append(f"  总印花税:        {m.get('total_tax', 0):,.2f}")
        if 'benchmark_return' in m:
            lines.append("")
            lines.append("基准对比:")
            lines.append(f"  基准收益率:      {m.get('benchmark_return', 0) * 100:.2f}%")
            lines.append(f"  超额收益:        {m.get('excess_return', 0) * 100:.2f}%")
            lines.append(f"  Beta:            {m.get('beta', 0):.4f}")
            lines.append(f"  Alpha:           {m.get('alpha', 0) * 100:.4f}%")
            lines.append(f"  信息比率:        {m.get('information_ratio', 0):.4f}")
        lines.append("=" * 70)
        return "\n".join(lines)

    def plot(self, save_path: Optional[str] = None, show: bool = False):
        """
        生成可视化图表
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import seaborn as sns
        except ImportError:
            logger.error("matplotlib/seaborn 未安装，无法绘图")
            return

        if self.daily_df is None or self.daily_df.empty:
            return

        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        df = self.daily_df.copy()
        df['nav'] = df['total_value_eod'] / df['total_value_eod'].iloc[0] if 'total_value_eod' in df.columns else df[
                                                                                                                      'total_value'] / \
                                                                                                                  df[
                                                                                                                      'total_value'].iloc[
                                                                                                                      0]
        df['cum_return'] = (df['nav'] - 1) * 100

        returns = df['daily_return'].dropna()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100

        fig = plt.figure(figsize=(16, 12))

        # 1. 净值曲线
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(df.index, df['nav'], label='策略净值', color='blue', linewidth=1.5)
        if self.benchmark_df is not None:
            bench = self.benchmark_df.copy()
            bench['date'] = pd.to_datetime(bench['date'])
            bench = bench[(bench['date'] >= df.index[0]) & (bench['date'] <= df.index[-1])]
            bench = bench.sort_values('date')
            bench_nav = bench['close'] / bench['close'].iloc[0]
            ax1.plot(bench['date'], bench_nav, label='基准净值', color='red', linestyle='--', linewidth=1.5)
        ax1.set_title('净值曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 累计收益率
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(df.index, df['cum_return'], color='green', linewidth=1.5)
        ax2.fill_between(df.index, 0, df['cum_return'], alpha=0.3, color='green')
        ax2.set_title(f"累计收益率 ({df['cum_return'].iloc[-1]:.2f}%)")
        ax2.grid(True, alpha=0.3)

        # 3. 回撤曲线
        ax3 = plt.subplot(3, 2, 3)
        ax3.fill_between(df.index[1:], 0, drawdown, color='red', alpha=0.4)
        ax3.plot(df.index[1:], drawdown, color='darkred', linewidth=0.8)
        ax3.set_title(f"回撤 (最大: {self.metrics.get('max_drawdown', 0) * 100:.2f}%)")
        ax3.grid(True, alpha=0.3)

        # 4. 日收益分布
        ax4 = plt.subplot(3, 2, 4)
        ax4.hist(returns * 100, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax4.axvline(returns.mean() * 100, color='red', linestyle='--', label=f"均值: {returns.mean() * 100:.2f}%")
        ax4.set_title('日收益率分布')
        ax4.set_xlabel('日收益率 (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. 月度收益热力图
        ax5 = plt.subplot(3, 2, 5)
        monthly = df['daily_return'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        monthly.index = monthly.index.to_period('M')
        if len(monthly) > 0:
            monthly_df = monthly.reset_index()
            monthly_df['year'] = monthly_df['date'].dt.year
            monthly_df['month'] = monthly_df['date'].dt.month
            pivot = monthly_df.pivot(index='year', columns='month', values='daily_return')
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax5,
                        cbar_kws={'label': '收益率 %'})
            ax5.set_title('月度收益率热力图 (%)')

        # 6. 年度收益柱状图
        ax6 = plt.subplot(3, 2, 6)
        yearly = df['daily_return'].resample('Y').apply(lambda x: (1 + x).prod() - 1) * 100
        if len(yearly) > 0:
            years = yearly.index.year.tolist()
            colors = ['green' if v > 0 else 'red' for v in yearly.values]
            ax6.bar(years, yearly.values, color=colors, alpha=0.7)
            ax6.set_title('年度收益率 (%)')
            ax6.set_xticks(years)
            ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"图表已保存: {save_path}")
        if show:
            plt.show()
        else:
            plt.close()


def analyze_backtest_result(result_dir: str, benchmark_df: Optional[pd.DataFrame] = None) -> Dict:
    """
    快捷函数：分析指定回测结果目录并生成报告
    """
    analyzer = PerformanceAnalyzer(result_dir, benchmark_df)
    metrics = analyzer.run()
    analyzer.plot(save_path=os.path.join(result_dir, 'performance_chart.png'))
    return metrics