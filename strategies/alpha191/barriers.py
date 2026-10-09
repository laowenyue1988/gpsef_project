# -*- coding: utf-8 -*-
"""
Alpha191 止盈止损屏障计算模块

基于 Prado Triple-Barrier 理论（Advances in FML, Ch.3）的统一波动率归一化屏障。
供 strategy.py 和 labels.py 共享使用，确保交易端和标签端公式完全一致。

设计原则：
1. 屏障 = f(σ_持有期)，不是 f(signal)
   - Prado Ch.3.5: "barriers are a function of volatility, not meta-model probability"
2. 信号控制选股，波动率控制风险
   - V7 Wyckoff pred → 候选池
   - Alpha191 XGBoost → 最终选股
   - 统一 PT/SL → 风险边界（不用信号做第三次决策）
3. 个股差异化由 daily_vol 自然提供（波动大的股票屏障更宽）
4. 波动率直接使用日波动率（hold_days=1，不进行持有期缩放）

公式：
    hold_vol = daily_vol * sqrt(hold_days)
    pt_threshold = entry_price * (1 + pt_mult * hold_vol * vol_mult)
    sl_threshold = entry_price * (1 - sl_mult * hold_vol * vol_mult)

默认参数（日波动直接做屏障）：
    hold_days = 1  →  σ_hold = σ_daily（不缩放）
    pt_mult = 2.0  →  ~4.0% PT at 2% daily vol
    sl_mult = 1.5  →  ~3.0% SL at 2% daily vol
"""

import numpy as np


def compute_barriers(entry_price, daily_vol, pt_mult=2.0, sl_mult=1.5, vol_mult=1.0, hold_days=1, vectorized=False):
    """
    统一波动率归一化止盈止损阈值（支持标量和向量化计算）。

    Args:
        entry_price: 入场价 (scalar or np.array)
        daily_vol: 日波动率 (scalar or np.array)
        pt_mult: PT 倍数
        sl_mult: SL 倍数
        vol_mult: 波动率调整系数
        hold_days: 持有期（交易日数）
        vectorized: 是否返回 numpy 数组（批量模式）

    Returns:
        (pt_threshold, sl_threshold): 止盈价、止损价
        当 vectorized=True 时返回 (np.array, np.array)
    """
    if vectorized:
        entry_price = np.asarray(entry_price, dtype=float)
        daily_vol = np.asarray(daily_vol, dtype=float)
        hold_vol = daily_vol * np.sqrt(hold_days)
        pt = entry_price * (1 + pt_mult * hold_vol * vol_mult)
        sl = entry_price * (1 - sl_mult * hold_vol * vol_mult)
        return pt, sl

    hold_vol = daily_vol * np.sqrt(hold_days)
    pt = entry_price * (1 + pt_mult * hold_vol * vol_mult)
    sl = entry_price * (1 - sl_mult * hold_vol * vol_mult)
    return pt, sl
