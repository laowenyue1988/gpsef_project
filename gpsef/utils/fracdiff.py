# gpsef/utils/fracdiff.py
import numpy as np
import pandas as pd


def getWeights_FFD(d, thres=1e-5):
    """生成固定窗口的FFD权重，权重绝对值小于thres的被截断"""
    w = [1.]
    while True:
        w_ = -w[-1] * (d - len(w) + 1) / len(w)
        if abs(w_) < thres:
            break
        w.append(w_)
    return np.array(w[:-1])


def fracDiff_FFD(series, d, thres=1e-5):
    """
    对序列执行固定窗口FFD。
    返回：
        pd.Series（输入为Series时）或 np.array（输入为array时）
        长度与输入相同
    """

    w = getWeights_FFD(d, thres)
    width = len(w) - 1


    # 提取数值并强制一维
    if isinstance(series, pd.Series):
        name = series.name
        idx = series.index
        vals = series.values.ravel()
    else:
        vals = np.asarray(series).ravel()
        idx = None

    n = len(vals)


    out = np.full(n, np.nan)   # 关键：用户数据长度


    for i in range(width, n):
        out[i] = np.dot(w, vals[i - width:i + 1][::-1])

    # 返回前打印最终长度
    if idx is not None:
        return pd.Series(out, index=idx, name=name)
    else:
        return out