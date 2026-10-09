#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
from gpsef.data.downloader import DataDownloader
from gpsef.utils.logger import setup_logger

# ========== 强制设置日志级别 ==========
# 设置根 logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# 显式设置 downloader 模块的 logger 级别
logging.getLogger('gpsef.data.downloader').setLevel(logging.INFO)
logging.getLogger('gpsef.data').setLevel(logging.INFO)
logging.getLogger('gpsef').setLevel(logging.INFO)

logger = logging.getLogger('download')
# ========================================


class DownloadConfig:
    """数据下载配置类，方便在IDE中直接修改参数"""

    # ========== 在这里直接修改参数 ==========
    MODE = 'full'  # 可选: raw_daily, back_adjust, dividend, minute, full
    BASE_PATH = './data'  # 数据存储根目录
    START_DATE = '20260801'  # 日线起始日期 YYYYMMDD（设为 None 则自动从缓存末尾开始）
    MINUTE_START_DATE = '20200103'  # 分钟线起始日期（baostock分钟线最早2020-01-03）
    END_DATE = '20260902'  # 目标结束日期 YYYYMMDD
    MINUTE_FREQ = '5'  # 分钟线频率: 5, 15, 30, 60 (仅 minute/full 模式有效)

    # =====================================

    @classmethod
    def to_args(cls):
        """转换为 argparse 的 Namespace 对象"""
        args = argparse.Namespace()
        args.mode = cls.MODE
        args.base_path = cls.BASE_PATH
        args.start = cls.START_DATE
        args.end = cls.END_DATE
        args.freq = cls.MINUTE_FREQ
        return args


def main(config=None):
    if config is None:
        parser = argparse.ArgumentParser(description='GPSEF 数据下载工具')
        parser.add_argument('--mode',
                            choices=['raw_daily', 'back_adjust', 'dividend', 'minute', 'full'],
                            default='raw_daily',
                            help='下载模式')
        parser.add_argument('--base-path', default='./data', help='数据存储根目录')
        parser.add_argument('--start', default='20260805', help='开始日期 YYYYMMDD')
        parser.add_argument('--end', default='20260831', help='目标结束日期 YYYYMMDD')
        parser.add_argument('--freq', choices=['5', '15', '30', '60'], default='5',
                            help='分钟线频率')
        args = parser.parse_args()
    else:
        args = config

    d = DataDownloader(args.base_path)

    # 下载交易日历和股票列表
    d.download_trade_calendar()
    d.download_stock_list()

    # 格式化目标日期
    from datetime import datetime
    target_end = args.end
    if len(target_end) == 8:
        target_end = f"{target_end[:4]}-{target_end[4:6]}-{target_end[6:]}"

    target_start = None
    if hasattr(args, 'start') and args.start is not None:
        target_start = args.start
        if len(target_start) == 8:
            target_start = f"{target_start[:4]}-{target_start[4:6]}-{target_start[6:]}"

    if args.mode in ('raw_daily', 'full'):
        d.update_all_raw_daily(target_end, target_start_date=target_start)

    if args.mode in ('back_adjust', 'full'):
        d.update_all_back_adjust()

    if args.mode in ('minute', 'full'):
        logger.info(f"开始下载 {args.freq} 分钟线数据...")
        # 分钟线起始日期：使用MINUTE_START_DATE或target_start中较晚者
        # baostock分钟线最早数据为2020-01-03，早于此日期的请求只会返回空数据
        from datetime import datetime as _dt
        minute_start = target_start
        minute_default = config.MINUTE_START_DATE if hasattr(config, 'MINUTE_START_DATE') else '20200103'
        if minute_start is not None:
            minute_default_dt = _dt.strptime(minute_default, '%Y%m%d')
            minute_start_dt = _dt.strptime(minute_start.replace('-', '')[:8], '%Y%m%d')
            if minute_start_dt < minute_default_dt:
                logger.info(f"分钟线起始日期 {minute_start} 早于baostock分钟线数据范围(2020-01-03)，"
                           f"自动调整为 {minute_default[:4]}-{minute_default[4:6]}-{minute_default[6:]}")
                minute_start = f"{minute_default[:4]}-{minute_default[4:6]}-{minute_default[6:]}"
        else:
            minute_start = f"{minute_default[:4]}-{minute_default[4:6]}-{minute_default[6:]}"
        d.update_all_raw_minute(
            freq=args.freq,
            target_start_date=minute_start,
            target_end_date=target_end
        )
        logger.info(f"{args.freq} 分钟线数据下载完成")

    if args.mode in ('dividend', 'full'):
        logger.info("开始下载分红数据...")
        # [2026-09-02 修复] 原实现无参调用 → CLI --start/--end 被忽略,
        #   end_year 恒=今年且按年粒度跳过 → 年内新增分红永不补拉。
        #   现传入 CLI 日期区间 (日期粒度检查, 见 update_all_dividend)
        d.update_all_dividend(start_date=target_start, end_date=target_end)
        logger.info("分红数据下载完成")

    logger.info("数据下载/更新完成")
    print(f"\n{'='*60}")
    print(f"下载完成！模式: {args.mode}")
    if args.mode in ('minute', 'full'):
        print(f"分钟线频率: {args.freq}分钟")
        if target_start:
            print(f"开始日期: {target_start}")
    print(f"结束日期: {target_end}")
    print(f"{'='*60}")


if __name__ == '__main__':
    # [2026-08-05 修复] 原实现 main(config=DownloadConfig.to_args()) 用类常量
    # MODE='full' 覆盖命令行 --mode 参数, 导致 --mode raw_daily 实际跑了 full
    # (连分钟线/分红一起下载, 数小时)。修复为使用命令行参数。
    main()