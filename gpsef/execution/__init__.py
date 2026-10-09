# -*- coding: utf-8 -*-
"""
GPSEF 执行层模块
提供统一的订单执行接口
"""
from .broker import SimulatedBroker

__all__ = ['SimulatedBroker']