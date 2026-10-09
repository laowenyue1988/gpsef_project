# -*- coding: utf-8 -*-
"""
Barra 因子暴露快照预构建脚本

在首次运行 Barra 约束回测前执行，预计算所有约束日的因子暴露快照。
避免回测中逐日调用 baostock 导致超时/限流。

用法：
  python strategies/wyckoff20/build_barra_snapshots.py \
      --start 20230101 --end 20260522

  或从配置文件自动读取日期：
  python strategies/wyckoff20/build_barra_snapshots.py \
      --config strategies/wyckoff20/config_v10.yaml

原理：
  - 约束日 = rebalance_day - 1 交易日
  - rebalance 每 20 个交易日一次
  - 构建约束日前一个交易日的快照即可覆盖所有调仓日
"""
import os
import sys
import argparse
import logging
import time
from datetime import datetime, timedelta

import yaml
import pandas as pd

# 确保项目根目录在 path 中
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from strategies.wyckoff20.barra_builder import build_barra_snapshot

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('build_barra_snapshots')


def get_trade_calendar(start_date: datetime, end_date: datetime):
    """获取交易日列表（仅真实交易日，不含周末/假期）。"""
    cal_path = os.path.join(_PROJECT_ROOT, 'data', 'trade_datas.csv')
    if not os.path.exists(cal_path):
        logger.error(f"Trade calendar not found: {cal_path}")
        return []

    cal_df = pd.read_csv(cal_path)
    cal_df['calendar_date'] = pd.to_datetime(cal_df['calendar_date'])

    # 仅保留真实交易日
    if 'is_trading_day' in cal_df.columns:
        cal_df = cal_df[cal_df['is_trading_day'] == 1]

    dates = sorted([pd.Timestamp(d).to_pydatetime() for d in cal_df['calendar_date'].tolist()])
    dates = [d for d in dates if start_date <= d <= end_date]
    return dates


def get_constraint_dates(start_date: datetime, end_date: datetime, rebalance_interval: int = 20):
    """
    获取所有需要构建快照的约束日。

    调仓逻辑（对齐 V10 策略 gpsef_strategy_v10.py）:
      trade_day_seq 每交易日 +1
      is_rebalance = ((trade_day_seq - 1) % rebalance_interval == 0)
      → 第 1, 21, 41, 61, ... 个交易日调仓

    约束日 = 调仓日的前一交易日。
    """
    calendar = get_trade_calendar(start_date - timedelta(days=30), end_date + timedelta(days=5))
    if not calendar:
        logger.error("无法获取交易日历")
        return []

    # 找到 start_date 之后第一个交易日对应的索引
    first_idx = None
    for i, d in enumerate(calendar):
        if d.date() >= start_date.date():
            first_idx = i
            break
    if first_idx is None:
        first_idx = 0

    # 按交易日间隔取调仓日: trade_day_seq = 1, 1+20, 1+40, ...
    rebalance_dates = []
    i = first_idx
    while i < len(calendar) and calendar[i].date() <= end_date.date():
        rebalance_dates.append(calendar[i])
        i += rebalance_interval  # ← 交易日间隔，不是自然日

    logger.info(f"回测期间: {start_date.date()} ~ {end_date.date()}")
    logger.info(f"交易日历: {len(calendar)} 个交易日")
    logger.info(f"首个调仓日: {rebalance_dates[0].date() if rebalance_dates else 'N/A'} "
                f"(calendar idx={first_idx})")
    logger.info(f"调仓间隔: {rebalance_interval} 个交易日")
    logger.info(f"调仓日数量: {len(rebalance_dates)}")

    # 对每个调仓日，取前一交易日作为约束日
    constraint_dates = []
    for rd in rebalance_dates:
        try:
            idx = calendar.index(rd)
            if idx > 0:
                constraint_dates.append(calendar[idx - 1])
            else:
                constraint_dates.append(rd)
        except ValueError:
            continue

    # 去重排序
    constraint_dates = sorted(set(constraint_dates))
    logger.info(f"约束日数量: {len(constraint_dates)} (去重后)")
    return constraint_dates


def main():
    parser = argparse.ArgumentParser(description='预构建 Barra 因子暴露快照')
    parser.add_argument('--config', type=str, help='策略配置文件路径')
    parser.add_argument('--start', type=str, help='开始日期 YYYYMMDD')
    parser.add_argument('--end', type=str, help='结束日期 YYYYMMDD')
    parser.add_argument('--rebalance-interval', type=int, default=20, help='调仓周期 (默认20)')
    args = parser.parse_args()

    # 从 config 或参数获取起止日期
    if args.config:
        abs_path = os.path.abspath(args.config)
        with open(abs_path, 'r', encoding='utf-8-sig') as f:
            cfg = yaml.safe_load(f)
        bt_cfg = cfg.get('backtest', {})
        start_str = bt_cfg.get('start_date', '20230101')
        end_str = bt_cfg.get('end_date', '20260522')
        start_date = datetime.strptime(start_str, '%Y%m%d')
        end_date = datetime.strptime(end_str, '%Y%m%d')
    else:
        if not args.start or not args.end:
            parser.error("需要 --config 或 --start + --end")
        start_date = datetime.strptime(args.start, '%Y%m%d')
        end_date = datetime.strptime(args.end, '%Y%m%d')

    constraint_dates = get_constraint_dates(start_date, end_date, args.rebalance_interval)

    if not constraint_dates:
        logger.error("无约束日可用")
        return

    logger.info("=" * 60)
    logger.info(f"开始构建 {len(constraint_dates)} 个约束日的 Barra 快照")
    logger.info("=" * 60)

    success = 0
    skipped = 0
    failed = 0

    for i, dt in enumerate(constraint_dates):
        date_str = dt.strftime('%Y%m%d')
        out_path = os.path.join(_PROJECT_ROOT, 'data', 'barra_exposures', f'barra_expo_{date_str}.parquet')

        if os.path.exists(out_path):
            skipped += 1
            if i % 20 == 0:
                logger.info(f"[{i+1}/{len(constraint_dates)}] {date_str} — 已存在，跳过")
            continue

        logger.info(f"[{i+1}/{len(constraint_dates)}] 构建 {date_str} ...")
        t0 = time.time()
        try:
            df = build_barra_snapshot(dt, force_rebuild=True)
            if df is not None and len(df) > 100:
                elapsed = time.time() - t0
                success += 1
                logger.info(f"  完成: {len(df)} stocks, {elapsed:.1f}s")
            else:
                failed += 1
                logger.warning(f"  失败: 数据不足 (len={len(df) if df is not None else 0})")
        except Exception as e:
            failed += 1
            logger.error(f"  异常: {e}")

    logger.info("=" * 60)
    logger.info(f"构建完成: {success} 成功, {skipped} 跳过(已存在), {failed} 失败")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
