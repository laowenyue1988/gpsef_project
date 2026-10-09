# gpsef/utils/sequential_bootstrap.py
import numpy as np
import pandas as pd

def getIndMatrix(barIx, t1):
    """构建指示矩阵，barIx为有序时间索引，t1: pd.Series(index=样本开始, values=样本结束)"""
    indM = pd.DataFrame(0, index=barIx, columns=range(t1.shape[0]))
    for i, (t0, t1_val) in enumerate(t1.items()):   # 改用 items()
        if pd.notna(t1_val):
            indM.loc[t0:t1_val, i] = 1
        else:
            indM.loc[t0:, i] = 1
    return indM

def getAvgUniqueness(indM):
    """计算每个样本的平均唯一性"""
    c = indM.sum(axis=1)           # 每时间点的并发数
    u = indM.div(c, axis=0)       # 每个时间点的唯一性
    avgU = u[u > 0].mean()        # 仅考虑活跃的时间点
    return avgU

def seqBootstrap(indM, sLength=None, random_state=None):
    """序贯Bootstrap采样，返回样本索引列表"""
    if sLength is None:
        sLength = indM.shape[1]
    rng = np.random.RandomState(random_state)
    phi = []
    while len(phi) < sLength:
        avgU = pd.Series(index=indM.columns)
        for i in indM.columns:
            temp_indM = indM[phi + [i]]
            avgU[i] = getAvgUniqueness(temp_indM).iloc[-1]
        prob = avgU / avgU.sum()
        choice = rng.choice(indM.columns, p=prob.values)
        phi.append(choice)
    return phi