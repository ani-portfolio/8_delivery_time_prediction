import pandas as pd
import numpy as np

from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

import lightgbm as lgb
import optuna

import logging

logger = logging.getLogger(__name__)
def hyperparameter_tuning(X, y, n_splits: int, n_trials: int, params: dict, model) -> dict:
    """
    Performs hyperparameter tuning using Optuna with K-fold cross-validation.
    
    Args:
        n_splits: Number of K-fold splits
        n_trials: Number of Optuna trials
        params: Dictionary with 'search_space' and 'fixed_params'
        model: Model type ('lgb' supported)
        X: Training features
        y: Training target
        
    Returns:
        dict: Best parameters found
    """
    
    def objective(trial):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Build params from search space
        trial_params = params.get('fixed_params', {}).copy()
        for param_name, param_config in params['search_space'].items():
            if param_config['type'] == 'int':
                trial_params[param_name] = trial.suggest_int(
                    param_name, param_config['low'], param_config['high']
                )
            elif param_config['type'] == 'float':
                trial_params[param_name] = trial.suggest_float(
                    param_name, param_config['low'], param_config['high']
                )
        
        rmse_scores = []
        
        for train_idx, valid_idx in kf.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]
            
            lgb_train = lgb.Dataset(X_tr, y_tr)
            lgb_valid = lgb.Dataset(X_val, y_val, reference=lgb_train)
            
            trained_model = lgb.train(
                trial_params,
                lgb_train,
                valid_sets=[lgb_valid],
                callbacks=[lgb.early_stopping(10, verbose=False)],
            )
            
            y_pred = trained_model.predict(X_val, num_iteration=trained_model.best_iteration)
            rmse = root_mean_squared_error(y_val, y_pred)
            rmse_scores.append(rmse)
        
        return np.mean(rmse_scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_params.update(params.get('fixed_params', {}))

    logger.info('Hyper-parameter tuning complete')
    
    return best_params