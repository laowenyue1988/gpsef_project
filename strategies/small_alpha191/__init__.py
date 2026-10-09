# strategies/alpha191/__init__.py
from optuna.trial import Trial
from typing import Dict, Any

# 定义哪些参数属于执行参数，需放入 execution 节
EXEC_PARAM_KEYS = ['max_positions', 'max_position_pct']


def create_param_space(trial: Trial, stage: int, fixed_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    定义 Alpha191 策略的多阶段超参数空间。

    Parameters
    ----------
    trial : Trial
        Optuna 试验对象
    stage : int
        当前阶段（1~5）
    fixed_params : Dict[str, Any]
        前序阶段已确定的最优参数，当前阶段必须直接使用这些值，不得重新搜索。

    Returns
    -------
    Dict[str, Any]
        完整的参数字典
    """
    params = {}
    # 首先合并所有固定参数
    params.update(fixed_params)

    # 各阶段的默认值（如果未在前序阶段优化，则使用默认值）
    defaults = {
        'xgb_max_depth': 6,
        'xgb_min_child_weight': 5,
        'xgb_gamma': 0.1,
        'xgb_subsample': 0.8,
        'xgb_col_sample_by_tree': 0.6,
        'xgb_reg_alpha': 0.0,
        'xgb_reg_lambda': 1.0,
        'xgb_learning_rate': 0.1,
        'xgb_n_estimators': 2000,
        'xgb_early_stopping_rounds': 50,
        'psi_threshold': 0.15,
        'min_feat_ic': 0.03,
        'nan_threshold': 0.3,
        'max_corr': 0.85,
        'min_features': 20,
        'max_features': 80,
        'max_positions': 20,
        'max_position_pct': 0.10,
    }

    # 阶段 1：树结构参数
    if stage == 1:
        if 'xgb_max_depth' not in fixed_params:
            params['xgb_max_depth'] = trial.suggest_int('xgb_max_depth', 3, 8)
        if 'xgb_min_child_weight' not in fixed_params:
            params['xgb_min_child_weight'] = trial.suggest_int('xgb_min_child_weight', 1, 15)
        if 'xgb_gamma' not in fixed_params:
            params['xgb_gamma'] = trial.suggest_float('xgb_gamma', 0.0, 0.5)

    # 阶段 2：采样与正则化
    elif stage == 2:
        if 'xgb_subsample' not in fixed_params:
            params['xgb_subsample'] = trial.suggest_float('xgb_subsample', 0.5, 0.9)
        if 'xgb_col_sample_by_tree' not in fixed_params:
            params['xgb_col_sample_by_tree'] = trial.suggest_float('xgb_col_sample_by_tree', 0.3, 0.8)
        if 'xgb_reg_alpha' not in fixed_params:
            params['xgb_reg_alpha'] = trial.suggest_float('xgb_reg_alpha', 0.0, 1.0)
        if 'xgb_reg_lambda' not in fixed_params:
            params['xgb_reg_lambda'] = trial.suggest_float('xgb_reg_lambda', 0.5, 10.0, log=True)

    # 阶段 3：学习收敛
    elif stage == 3:
        if 'xgb_learning_rate' not in fixed_params:
            params['xgb_learning_rate'] = trial.suggest_float('xgb_learning_rate', 0.01, 0.1, log=True)
        if 'xgb_n_estimators' not in fixed_params:
            params['xgb_n_estimators'] = trial.suggest_int('xgb_n_estimators', 800, 2000, step=200)
        if 'xgb_early_stopping_rounds' not in fixed_params:
            params['xgb_early_stopping_rounds'] = trial.suggest_int('xgb_early_stopping_rounds', 20, 80, step=10)

    # 阶段 4：特征筛选
    elif stage == 4:
        if 'psi_threshold' not in fixed_params:
            params['psi_threshold'] = trial.suggest_float('psi_threshold', 0.05, 0.25)
        if 'min_feat_ic' not in fixed_params:
            params['min_feat_ic'] = trial.suggest_float('min_feat_ic', 0.01, 0.08)
        if 'nan_threshold' not in fixed_params:
            params['nan_threshold'] = trial.suggest_float('nan_threshold', 0.1, 0.4)
        if 'max_corr' not in fixed_params:
            params['max_corr'] = trial.suggest_float('max_corr', 0.7, 0.95)
        if 'min_features' not in fixed_params:
            min_feat = trial.suggest_int('min_features', 10, 30)
            params['min_features'] = min_feat
        if 'max_features' not in fixed_params:
            # 确保 max_features > min_features
            current_min = params.get('min_features', defaults['min_features'])
            params['max_features'] = trial.suggest_int('max_features', current_min + 20, 120)

    # 阶段 5：仓位管理
    elif stage == 5:
        if 'max_positions' not in fixed_params:
            params['max_positions'] = trial.suggest_int('max_positions', 10, 30)
        if 'max_position_pct' not in fixed_params:
            params['max_position_pct'] = trial.suggest_float('max_position_pct', 0.05, 0.20)

    # 为所有未设置的参数填充默认值
    for k, v in defaults.items():
        if k not in params:
            params[k] = v

    return params