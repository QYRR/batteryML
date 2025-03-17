# select the most important features
import lightgbm as lgb
import optuna
import numpy as np
import pandas as pd
import psutil
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
np.random_state = 42
pd.random_state = 42

__all__ = ['select_features']
 
# Columns you mentioned
columns_soh = [
    'voltage',       # Voltage feature
    'temperature',
    'power',
    'current',
    'discharge_power_rate',  # Discharge power feature
    'discharge_current_rate',
    'discharge_voltage_rate',
    'sum_relativeTime',
    'range_voltage',
    'range_current',
    'range_temperature',
    'step_length',
    'duration',
    'delta_current',
    'delta_voltage',
    'delta_temperature',
    # 'energy',
    # 'discharge_energy_rate',
]

# Objective function for Optuna
def objective(trial, xtrain, ytrain, xvalid, yvalid):
    # Suggest hyperparameters for LightGBM
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.7),
        'num_leaves': trial.suggest_int('num_leaves', 2, 50),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0000, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0000, 10.0),
        'n_estimators' : trial.suggest_int('n_estimators', 5, 150),
        'verbose': -1
    }
    
    # Create dataset for LightGBM, focusing on selected features
    dtrain = lgb.Dataset(xtrain, label=ytrain, free_raw_data=False)
    dvalid = lgb.Dataset(xvalid, label=yvalid, free_raw_data=False)
    
    # Measure start time, memory, and CPU usage
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # in MB
    start_cpu = psutil.cpu_percent(interval=None)
    
    # Train the model
    #gbm = lgb.train(param, dtrain, num_boost_round= trial.suggest_int('num_boost_round', 1, 100), keep_training_booster=True)
    gbm = lgb.LGBMRegressor(**param).fit(dtrain.data, dtrain.label)
    # Measure end time, memory, and CPU usage
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # in MB
    end_cpu = psutil.cpu_percent(interval=None)
    
    # Predict on validation set
    y_pred = gbm.predict(xvalid)#, num_iteration=gbm.best_iteration)
    rmse = np.sqrt(mean_absolute_error(yvalid, y_pred))
    
    # Calculate resource consumption metrics
    time_elapsed = end_time - start_time
    #print("INFO ------> # of trees: ", gbm.num_trees())
    memory_used = end_memory - start_memory
    cpu_used = gbm.booster_.num_trees() #end_cpu - start_cpu
    
    # Multi-objective optimization: minimize both RMSE and resource consumption
    return rmse, memory_used, time_elapsed, cpu_used
 
# Function to select important features using permutation importance
def select_important_features(xtrain, ytrain, n_features):
    # Train an initial LightGBM model with default parameters
    param = {'objective':'regression', 'n_estimators' :80}
    model = lgb.LGBMRegressor(**param)
    #dtrain = lgb.Dataset(data=xtrain, label=ytrain)
    model.fit(xtrain, ytrain)
 
    # Get permutation importance from sklearn
    result = permutation_importance(model, xtrain, ytrain, n_repeats=25, random_state=42, scoring='neg_mean_squared_error')
    
    # Get the top n_features based on importance
    importance_df = pd.DataFrame({'feature': xtrain.columns, 'importance': result.importances_mean})
    importance_df = importance_df.sort_values(by='importance', ascending=False)
 
    # Select the top n features
    important_features = importance_df['feature'].head(n_features).tolist()
 
    print(f"INFO -----> Selected Important Features: {important_features}")
    return important_features
 
# Select the most important features based on permutation importance
def feature_selection_and_optimization(xtrain, ytrain, xvalid, yvalid, n_features):
    # Select top important features using permutation importance
    #print(xtrain.columns)
    selected_features = select_important_features(xtrain[columns_soh], ytrain, n_features)
    
    # Optimize the model using Optuna and selected features
    study = optuna.create_study(directions=["minimize", "minimize", "minimize", "minimize"])
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    study.optimize(lambda trial: objective(trial, xtrain[selected_features], ytrain, xvalid[selected_features], yvalid), n_trials=400)
    
    return study, selected_features
 
 
def check_shape(data):
    try:
        data.shape[1]
        if data.shape[1] > 1:
            data = data.iloc[:,0]
    except:
        pass
    return data
 
def check_soh(data):
    try:
        data.SOH
        data = data.drop(columns = ['SOH'])
    except:
        pass
    return data
 
def final_plot(model, data, columns):
    x = data[0]
    y = check_shape(data[1])
    pred = model.predict(x[columns])
    mae = mean_absolute_error(y, pred)
    print(mae)
    plt.scatter(np.arange(len(x)), y)
    plt.scatter(np.arange(len(x)), pred)
    plt.show()
 
 
# Retrain and evaluate the best model on the test set
def evaluate_best_pareto_model(study, xtrain, ytrain, xvalid, yvalid, xtest, ytest, selected_features):
    # Retrieve the list of best trials (Pareto front)
    best_trials = study.best_trials
    
    # You can select the best trial based on RMSE (or other criteria)
    # Here we pick the trial with the lowest RMSE (first objective)
    best_trial = min(best_trials, key=lambda trial: trial.values[0])  # values[0] is the RMSE
 
    best_params = best_trial.params
    #num_boost_round = best_params.pop('num_boost_round')
    # Train the model on the selected features using the best hyperparameters
    dtrain = lgb.Dataset(xtrain[selected_features], label=ytrain, free_raw_data=False)
    dvalid = lgb.Dataset(xvalid[selected_features], label=yvalid, free_raw_data=False)
    
    # Measure start time, memory, and CPU usage for test evaluation
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # in MB
    start_cpu = psutil.cpu_percent(interval=None)
    
    # Retrain the model on the full training set with selected features
    #print("INFO -----> # trees = ", num_boost_round)
    #print(ytrain.shape, yvalid.shape)
    gbm = lgb.LGBMRegressor(**best_params).fit(dtrain.data, dtrain.label)
    #gbm = lgb.train(best_params, dtrain, num_boost_round= num_boost_round, valid_sets=dvalid, keep_training_booster=True)
    
    # Measure end time, memory, and CPU usage
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # in MB
    end_cpu = psutil.cpu_percent(interval=None)
    
    # Predict on the test set
    y_pred_val = gbm.predict(dvalid.data)
    y_pred_test = gbm.predict(xtest[selected_features])#, num_iteration=gbm.best_iteration)
 
    # Calculate RMSE on the valid set
    rmse_test = np.sqrt(mean_squared_error(dvalid.label, y_pred_val))
    
    # Calculate RMSE on the test set
    rmse_test = np.sqrt(mean_squared_error(ytest, y_pred_test))
    
    # Calculate resource consumption metrics for the test set
    time_elapsed_test = end_time - start_time
    memory_used_test = end_memory - start_memory
    cpu_used_test = gbm.booster_.num_trees() #end_cpu - start_cpu
    
    # Print the test set results
    print(f"     Test MAE: {rmse_test}")
    print(f"     Memory used (MB): {memory_used_test}")
    print(f"     Time elapsed (s): {time_elapsed_test}")
    print(f"     CPU usage (%): {cpu_used_test}")
    print()
 
    return rmse_test, memory_used_test, time_elapsed_test, cpu_used_test, gbm
 
 
 
def select_features(train_x, train_y, valid_x, valid_y, test_x, test_y):
    train_y = check_shape(train_y)
    test_y = check_shape(test_y)
    valid_y = check_shape(valid_y)
    train_x = check_soh(train_x)
    valid_x = check_soh(valid_x)
    test_x = check_soh(test_x)
    # Perform feature selection and optimization with a specified number of features
    n_features = 10  # Specify the number of features to select
    best_mae = 100
    selected_models = []
    # for n_features in range(2, len(columns_soh)):
    for n_features in range(2, 6):
        study, selected_feature = feature_selection_and_optimization(train_x, train_y, valid_x, valid_y, n_features)
        best_trials = study.best_trials
        best_trial = min(best_trials, key=lambda trial: trial.values[0])  # values[0] is the RMSE
        best_params = best_trial.params
        if best_trial.values[0]< best_mae:
            best_mae = best_trial.values[0]
        print("RESU -----> MAE found = ", best_trial.values[0])
    
        # Evaluate the best model on the test set using the Pareto front
        rmse_test, memory_used_test, time_elapsed_test, cpu_used_test, gbm = evaluate_best_pareto_model(
            study, train_x, train_y, valid_x, valid_y, test_x, test_y, selected_feature
        )
        selected_models.append([gbm, best_mae, selected_feature])
        # Plot the Pareto front: Performance vs Resource Usage
        trials = study.best_trials
        rmse_vals = [trial.values[0] for trial in trials]
        memory_vals = [trial.values[1] for trial in trials]
        time_vals = [trial.values[2] for trial in trials]
        cpu_vals = [trial.values[3] for trial in trials]
        # final_plot(gbm, final_data, selected_feature)
    return selected_models, selected_feature