# gpsef/utils/logger.py
"""
日志工具模块
提供统一的日志配置，支持通过环境变量 GPSEF_LOG_LEVEL 控制日志详细程度。
用法：
    from gpsef.utils.logger import setup_logger
    logger = setup_logger(__name__)          # 自动从环境变量读取等级
    logger = setup_logger(__name__, level=logging.DEBUG)  # 手动指定
"""

import logging
import os
import sys

# 默认日志等级（若未设置环境变量，则为 INFO）
DEFAULT_LOG_LEVEL = logging.INFO

# 从环境变量读取日志等级，支持常见的等级名称
_ENV_LOG_LEVEL = os.environ.get('GPSEF_LOG_LEVEL', '').upper()
_LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}
if _ENV_LOG_LEVEL in _LEVEL_MAP:
    DEFAULT_LOG_LEVEL = _LEVEL_MAP[_ENV_LOG_LEVEL]


def setup_logger(name: str, level: int = None) -> logging.Logger:
    """
    创建并配置一个日志记录器。

    输入：
        name (str): 日志记录器的名称（通常使用 __name__）。
        level (int, optional): 日志等级。若为 None，则自动使用环境变量 GPSEF_LOG_LEVEL
                               或默认的 INFO 等级。

    输出：
        logging.Logger: 配置完成的日志记录器实例。

    调用方：
        框架内所有需要日志的模块及入口脚本。
    """
    if level is None:
        level = DEFAULT_LOG_LEVEL

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 handler
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def set_global_log_level(level: int) -> None:
    """
    设置全局默认日志等级（覆盖环境变量设置）。

    输入：
        level (int): 日志等级（如 logging.DEBUG）。

    输出：
        无

    调用方：
        入口脚本中需要临时调整日志详细程度时。
    """
    global DEFAULT_LOG_LEVEL
    DEFAULT_LOG_LEVEL = level