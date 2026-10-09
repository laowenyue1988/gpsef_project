# gpsef/utils/feature_importance.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import weightedtau

def compute_MDI(X, y, feature_names, n_estimators=100, max_depth=5, random_state=42):
    """计算MDI重要性"""
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                 random_state=random_state)
    clf.fit(X, y)
    imp = clf.feature_importances_
    return pd.DataFrame({'feature': feature_names, 'mdi_importance': imp}).sort_values(
        'mdi_importance', ascending=False)

def compute_MDA(X, y, feature_names, cv, scoring='neg_log_loss',
                sample_weight=None, n_repeats=5, random_state=42):
    """
    计算MDA（置换重要性）。
    cv: 交叉验证生成器（如PurgedKFold），需产生训练/测试索引。
    返回平均重要性及标准差。
    """
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import make_scorer, log_loss
    # 这里实现简化版：在整体数据上，用交叉验证得到permutation importance
    # 实际应使用清洗后的cv
    importances = []
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf = RandomForestClassifier(n_estimators=50, random_state=random_state)
        clf.fit(X_train, y_train)
        perm_imp = permutation_importance(clf, X_test, y_test, n_repeats=n_repeats,
                                          random_state=random_state, n_jobs=-1,
                                          scoring=scoring)
        importances.append(perm_imp.importances_mean)
    avg_imp = np.mean(importances, axis=0)
    std_imp = np.std(importances, axis=0)
    return pd.DataFrame({'feature': feature_names,
                         'mda_importance': avg_imp,
                         'mda_std': std_imp})

def pca_weighted_kendall_tau(X, feature_names, feat_imp_series, n_components=0.95):
    """
    PCA一致性检验。
    feat_imp_series: pd.Series, index为feature, value为重要性(如MDI)
    返回加权Kendall's Tau和p-value。
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    # 主成分特征值
    ev = pca.explained_variance_
    # 计算每个主成分的重要性：用特征向量的加权平均
    comp_imp = []
    for i in range(len(ev)):
        loadings = pca.components_[i]
        # weighted sum of feature importance
        imp_val = np.sum(np.abs(loadings) * feat_imp_series.reindex(feature_names).fillna(0).values)
        comp_imp.append(imp_val)
    # PCA排名（按特征值降序）
    pc_rank = pd.Series(np.arange(1, len(ev)+1))
    # 重要性排名
    imp_rank = pd.Series(comp_imp).rank(ascending=False)
    # 加权Kendall's Tau (使用倒数排名，使得高重要性对应低排名数字)
    # weightedtau 期望x按降序的权重，我们使用1/排名
    tau, p_value = weightedtau(imp_rank.values, 1/pc_rank.values)
    return tau, p_value