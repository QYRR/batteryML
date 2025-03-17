from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import numpy as np

__all__ = ['lgb_optimize', 'lgb_optimize_objmse']

def lgb_optimize(trial, x, y, vx, vy):
    """
    Args:
        x: input of tarining data
        y: output of training data
        vx: input of valid data
        vy: output of valid data

    Returns: mean abs error

    """
    lgbm_params = {
        'objective': 'regression',
        # 'objective': 'huber',
        # 'alpha': 0.8,  # (0.8, 0.95)
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.3),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'num_leaves': trial.suggest_int('num_leaves', 3, 35),
        'min_child_samples': trial.suggest_int('min_child_samples',2, 200),
        'subsample': trial.suggest_float('subsample', 0.0, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.001, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'n_estimators': trial.suggest_int('n_estimators', 30, 500),
        'random_state': 42,
        'verbose' : -1
        # Add more hyperparameters to optimize
    }

    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(x, y)
 
    valid_pred = model.predict(vx)
    mae = mean_absolute_error(vy, valid_pred)
 
    return mae




def lgb_optimize_objmse(trial, x, y, vx, vy):
    """
    Args:
        x: input of training data
        y: output of training data
        vx: input of valid data
        vy: output of valid data

    Returns: mean abs error
    """
    lgbm_params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.3),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'num_leaves': trial.suggest_int('num_leaves', 3, 35),
        'min_child_samples': trial.suggest_int('min_child_samples', 2, 200),
        'subsample': trial.suggest_float('subsample', 0.0, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.001, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'n_estimators': trial.suggest_int('n_estimators', 30, 500),
        'verbose': -1,
        'random_state': 42,
    }

    # 回调函数：动态计算预测分布并加入正则化
    def custom_callback(env):
        """计算方差并输出到日志"""
        y_pred = env.model.predict(env.validation_data[0][0])
        variance = np.var(y_pred)
        print(f"Validation Variance: {variance:.4f}")

    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(
        x, y,
        eval_set=[(vx, vy)],
        eval_metric='mae',
        callbacks=[custom_callback]
    )

    valid_pred = model.predict(vx)
    mae = mean_absolute_error(vy, valid_pred)

    return mae
