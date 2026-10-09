# -*- coding: utf-8 -*-
"""Alpha191 策略依赖的数据工具函数（交易日历、指数成分股等）"""
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

VALID_STOCK_PREFIXES = ('sh.600', 'sh.601', 'sh.603', 'sh.605', 'sh.688', 'sh.689',
                        'sz.000', 'sz.001', 'sz.002', 'sz.003', 'sz.300', 'sz.301', 'sz.302')

# ---------- Baostock 会话（复用） ----------
class BaostockSession:
    _instance = None
    _logged_in = False
    _last_activity_time = None
    _connection_timeout = 300  # 5分钟无活动则检测连接

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BaostockSession, cls).__new__(cls)
        return cls._instance

    def _is_connection_alive(self) -> bool:
        try:
            import baostock as bs
            rs = bs.query_stock_basic(code="sh.600000")
            return rs is not None and rs.error_code == '0'
        except:
            return False

    def login(self, force=False):
        if not self._logged_in or force:
            if self._logged_in and force:
                try:
                    import baostock as bs
                    bs.logout()
                except:
                    pass
                self._logged_in = False
            import baostock as bs
            lg = bs.login()
            if lg.error_code != '0':
                raise ValueError(f'登录失败: {lg.error_msg}')
            self._logged_in = True
            self._last_activity_time = time.time()
            logger.info("baostock登录成功")
        return self._logged_in

    def logout(self):
        if self._logged_in:
            import baostock as bs
            bs.logout()
            self._logged_in = False
            self._last_activity_time = None
            logger.info("baostock登出")

    def ensure_logged_in(self):
        if not self._logged_in:
            self.login()
            return
        # 检查空闲超时或连接失效
        if self._last_activity_time and (time.time() - self._last_activity_time) > self._connection_timeout:
            if not self._is_connection_alive():
                logger.warning("检测到连接断开，正在自动重连...")
                self.login(force=True)
        self._last_activity_time = time.time()

    def update_activity(self):
        """更新最后活动时间，供外部调用"""
        self._last_activity_time = time.time()

_baostock_session = BaostockSession()
def get_baostock_session():
    return _baostock_session

# ---------- 交易日历 ----------
_TRADE_CALENDAR_CACHE = None
_TRADE_CALENDAR_FILE = './data/trade_datas.csv'

def _fetch_trade_calendar_from_baostock(start_date, end_date):
    import baostock as bs
    session = get_baostock_session()
    session.ensure_logged_in()
    rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
    if rs.error_code != '0':
        raise ValueError(f"获取交易日历失败: {rs.error_msg}")
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    df = pd.DataFrame(data_list, columns=rs.fields)
    df['calendar_date'] = pd.to_datetime(df['calendar_date'])
    df['is_trading_day'] = pd.to_numeric(df['is_trading_day'])
    return df

def _load_trade_calendar():
    global _TRADE_CALENDAR_CACHE
    if _TRADE_CALENDAR_CACHE is not None:
        return _TRADE_CALENDAR_CACHE
    if not os.path.exists(_TRADE_CALENDAR_FILE):
        os.makedirs(os.path.dirname(_TRADE_CALENDAR_FILE), exist_ok=True)
        start_str = '2018-01-01'
        end_str = (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')
        df = _fetch_trade_calendar_from_baostock(start_str, end_str)
        df.to_csv(_TRADE_CALENDAR_FILE, index=False, encoding='utf-8-sig')
    else:
        df = pd.read_csv(_TRADE_CALENDAR_FILE, encoding='utf-8')
        df['calendar_date'] = pd.to_datetime(df['calendar_date'])
        df['is_trading_day'] = pd.to_numeric(df['is_trading_day'])
    _TRADE_CALENDAR_CACHE = df[df['is_trading_day'] == 1].copy().reset_index(drop=True)
    _TRADE_CALENDAR_CACHE = _TRADE_CALENDAR_CACHE.sort_values('calendar_date').reset_index(drop=True)
    return _TRADE_CALENDAR_CACHE

def get_trade_calendar(start_date, end_date):
    def _to_datetime(d):
        if isinstance(d, str):
            if len(d) == 8:
                return datetime.strptime(d, '%Y%m%d')
            else:
                return datetime.strptime(d, '%Y-%m-%d')
        return d
    start = _to_datetime(start_date)
    end = _to_datetime(end_date)
    df_cal = _load_trade_calendar()
    mask = (df_cal['calendar_date'] >= start) & (df_cal['calendar_date'] <= end)
    return df_cal.loc[mask, ['calendar_date']].copy().reset_index(drop=True)

# ---------- 指数成分股 ----------
# 数据来源(2026-08-04 审计确认):
#   data/index_stocks_cache.parquet 由旧框架 E:\quant\quant\alpha191\modules\data_manager.py
#   的 get_multi_index_stocks_by_akshare() 通过掘金量化 SDK 生成(2026-05-08), 核心调用:
#     stk_get_index_constituents(index='SHSE.000300'/'SHSE.000905'/'SHSE.000852',
#                                trade_date=查询日期)
#   返回该交易日收盘后的指数成分股, 过滤 VALID_STOCK_PREFIXES 后转 'sh.xxx'/'sz.xxx' 格式。
#   查询语义: D 日盘前取 cache_key <= D 的最近一条(见 strategy.py _get_index_components),
#   与真实指数调仓公布节奏一致, 严格无未来数据。
#   已知历史问题(已修复): gpsef 版缓存曾缺 2021-04~2023-04(含2022全年), 已于 2026-08-04
#   合并旧框架副本补齐 (scripts/manage_index_cache.py merge; 原文件备份 .bak_20260804)。
#   缓存管理: python scripts/manage_index_cache.py {merge|verify|rebuild}
GM_AVAILABLE = False
try:
    from gm.api import *
    GM_AVAILABLE = True
except ImportError:
    pass

_INDEX_CACHE = {}
_CACHE_FILE = './data/index_stocks_cache.parquet'
_GM_TOKEN_SET = False  # 掘金 set_token 进程级标志 (fetch_index_constituents_by_gm 使用)

def fetch_index_constituents_by_gm(index: str, trade_date: str) -> List[str]:
    """[生成器核心片段] 通过掘金量化拉取指定指数在指定交易日的成分股。

    与缓存 index_stocks_cache.parquet 的原生成方式完全同源:
      stk_get_index_constituents(index, trade_date=trade_date) 返回该日收盘后成分股名单。

    Args:
        index: 指数名, 'hs300'/'zz500'/'zz1000' (映射 SHSE.000300/000905/000852)
        trade_date: 'YYYY-MM-DD' 或 'YYYYMMDD', 掘金按该交易日返回成分股
    Returns:
        ['sh.600000', 'sz.000001', ...] 已过滤 VALID_STOCK_PREFIXES 并去重;
        失败返回 [] 并打 warning。
    """
    if not GM_AVAILABLE:
        logger.warning("掘金不可用, 无法获取指数成分股")
        return []
    # 掘金 SDK token 为进程级设置; 自动补充路径(策略内)需自行 set_token (幂等, 模块级标志防重复)
    global _GM_TOKEN_SET
    if not _GM_TOKEN_SET:
        token = os.environ.get('GM_TOKEN')
        if token:
            try:
                set_token(token)
                _GM_TOKEN_SET = True
            except Exception as e:
                logger.warning(f"掘金 set_token 失败: {e}")
        else:
            logger.warning("未设置 GM_TOKEN 环境变量, 掘金自动补充将不可用")
    symbol = {'hs300': 'SHSE.000300', 'zz500': 'SHSE.000905', 'zz1000': 'SHSE.000852'}.get(index)
    if symbol is None:
        logger.warning(f"未知指数: {index}")
        return []
    if '-' not in trade_date:
        try:
            trade_date = pd.to_datetime(trade_date).strftime('%Y-%m-%d')
        except Exception:
            pass
    try:
        df = stk_get_index_constituents(index=symbol, trade_date=trade_date)
        if df is None or df.empty:
            return []
        col = 'symbol' if 'symbol' in df.columns else df.columns[0]
        codes = []
        for full_code in df[col].tolist():
            if str(full_code).startswith('SHSE.'):
                codes.append('sh.' + str(full_code)[5:])
            elif str(full_code).startswith('SZSE.'):
                codes.append('sz.' + str(full_code)[5:])
        codes = [c for c in codes if c.startswith(VALID_STOCK_PREFIXES)]
        return list(set(codes))
    except Exception as e:
        logger.warning(f"掘金获取指数 {index} @{trade_date} 失败: {e}")
        return []

def get_multi_index_stocks_by_akshare(df_daily: pd.DataFrame, date: Optional[str] = None) -> Dict[str, List[str]]:
    """获取多个指数成分股 (历史成分股, 无未来数据)。
    加载指定日期的基本面快照（parquet 格式）。
    快照文件名格式：factors_YYYYMMDD.parquet
    返回的 DataFrame 至少包含：code, close, pe, pb, ..., industry, total_mv
    """
    import os
    fname = f'factors_{date.strftime("%Y%m%d")}.parquet'
    path = os.path.join(snapshot_dir, fname)
    if not os.path.exists(path):
        logger.warning(f"基本面快照不存在: {path}")
        return None
    df = pd.read_parquet(path)
    # 确保 code 列为6位字符串
    if 'code' in df.columns:
        df['code'] = df['code'].astype(str).str.zfill(6)
    # [v3.0] Deterministic sort + tiebreaker
    df = df.sort_values('code', kind='mergesort').reset_index(drop=True)
    df['_row_idx'] = range(len(df))
    logger.info(f"加载基本面快照: {path}, {len(df)} 条")
    return df


def build_industry_map_from_snapshots(snapshot_dir: str, as_of_date=None) -> Dict[str, str]:
    """
    从基本面快照中提取所有股票的行业信息，构建映射表。
    
    Args:
        snapshot_dir: 快照目录路径
        as_of_date: 可选 datetime/str，仅使用不晚于此日期的快照（防前视偏差）
    Returns:
        {code_digits: industry_name}
    """
    import os
    from datetime import datetime as dt
    industry_map = {}
    if not os.path.exists(snapshot_dir):
        return industry_map

    # 解析 as_of_date
    if as_of_date is not None:
        if isinstance(as_of_date, str):
            as_of_date = dt.strptime(as_of_date.replace('-', ''), '%Y%m%d')
        elif hasattr(as_of_date, 'strftime'):
            as_of_date = dt.strptime(as_of_date.strftime('%Y%m%d'), '%Y%m%d')

    files = sorted([f for f in os.listdir(snapshot_dir) if f.endswith('.parquet')])
    for fname in files:
        # 从文件名提取日期（格式: factors_YYYYMMDD.parquet 或类似）
        date_str = fname.replace('factors_', '').replace('.parquet', '')
        try:
            file_date = dt.strptime(date_str, '%Y%m%d')
        except ValueError:
            pass  # 无法解析日期，保留所有（保守处理无日期的文件）
        else:
            if as_of_date is not None and file_date > as_of_date:
                continue  # 跳过未来的快照，防止前视偏差 [v3.2 audit fix]
        
        try:
            df = pd.read_parquet(os.path.join(snapshot_dir, fname))
            if 'code' not in df.columns or 'industry' not in df.columns:
                continue
            df['code'] = df['code'].astype(str).str.zfill(6)
            for _, row in df.iterrows():
                code = row['code']
                if code not in industry_map:
                    industry_map[code] = row.get('industry', 'OTHER')
        except Exception:
            pass
    logger.info(f"从基本面快照构建行业映射: {len(industry_map)} 只股票"
                f"{', 截至于=' + as_of_date.strftime('%Y%m%d') if as_of_date else ''}")
    return industry_map


def map_code_to_ts_code(code_digits: str) -> str:
    """
    将6位数字代码转换为 'sh.600xxx' 或 'sz.000xxx' 格式。
    """
    code = str(code_digits).zfill(6)
    if code.startswith(('6', '9')):
        return 'sh.' + code
    else:
        return 'sz.' + code


def map_ts_code_to_digits(ts_code: str) -> str:
    """
    将 'sh.600xxx' 转换为 '600xxx'。
    """
    return ts_code.replace('sh.', '').replace('sz.', '')