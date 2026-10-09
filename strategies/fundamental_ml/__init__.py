from optuna.trial import Trial
from typing import Dict, Any

EXEC_PARAM_KEYS = ['max_positions', 'max_position_pct']

def create_param_space(trial: Trial, stage: int, fixed_params: Dict[str, Any]) -> Dict[str, Any]:
    params = {}
    params.update(fixed_params)
    defaults = {
        'xgb_max_depth': 5,
        'xgb_learning_rate': 0.01,
        'xgb_n_estimators': 300,
        'xgb_subsample': 0.8,
        'xgb_colsample_bytree': 0.6,
        'xgb_reg_alpha': 0.1,
        'xgb_reg_lambda': 1.0,
        'max_positions': 20,
        'max_position_pct': 0.10,
        'train_window_days': 120,
        'top_n': 20,
    }
    if stage == 1:
        if 'xgb_max_depth' not in fixed_params:
            params['xgb_max_depth'] = trial.suggest_int('xgb_max_depth', 3, 8)
        if 'xgb_learning_rate' not in fixed_params:
            params['xgb_learning_rate'] = trial.suggest_float('xgb_learning_rate', 0.005, 0.05, log=True)
        if 'xgb_n_estimators' not in fixed_params:
            params['xgb_n_estimators'] = trial.suggest_int('xgb_n_estimators', 100, 500, step=50)
    elif stage == 2:
        if 'max_positions' not in fixed_params:
            params['max_positions'] = trial.suggest_int('max_positions', 10, 30)
        if 'max_position_pct' not in fixed_params:
            params['max_position_pct'] = trial.suggest_float('max_position_pct', 0.05, 0.20)
        if 'train_window_days' not in fixed_params:
            params['train_window_days'] = trial.suggest_int('train_window_days', 60, 200, step=20)
    for k, v in defaults.items():
        if k not in params:
            params[k] = v
    return params