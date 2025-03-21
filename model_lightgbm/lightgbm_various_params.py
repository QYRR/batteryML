import os
import sys
import sys
import os
import psutil
import numpy as np
import time

def set_cores(n_cores):
    """
    Set the CPU cores to use for the current process.

    Parameters
    ----------
    n_cores : int
        n_cores: Number of cores to use.
    """

    # Get the least active cores
    least_active_cores = get_least_active_cores(n_cores)

    # Set the least active cores to use
    limit_cpu_cores(least_active_cores)


def limit_cpu_cores(cores_to_use):

    num_cores = len(cores_to_use)

    pid = os.getpid()  # the current process

    available_cores = list(range(psutil.cpu_count()))
    # selected_cores = available_cores[:num_cores]
    selected_cores = []
    for ii in cores_to_use:
        if ii in available_cores:
            selected_cores.append(ii)

    os.sched_setaffinity(pid, selected_cores)

    # Limit the number of threads used by different libraries
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    os.environ["MKL_NUM_THREADS"] = str(num_cores)


def get_least_active_cores(num_cores, num_readings=10):

    # Get CPU usage for each core for multiple readings
    cpu_usage_readings = []
    for ii in range(num_readings):
        cpu_usage_readings.append(psutil.cpu_percent(percpu=True))
        time.sleep(0.05)

    # Calculate the average CPU usage for each core
    avg_cpu_usage = [sum(usage) / num_readings for usage in zip(*cpu_usage_readings)]

    # Create a list of tuples (core_index, avg_cpu_usage)
    core_usage_tuples = list(enumerate(avg_cpu_usage))

    # Sort the list based on average CPU usage
    sorted_cores = sorted(core_usage_tuples, key=lambda x: x[1])

    # Get the first 'num_cores' indices (least active cores)
    least_active_cores = [index for index, _ in sorted_cores[:num_cores]]

    return least_active_cores

SEED = 0
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
np.random.seed(SEED)
set_cores(8)

import csv
import math
import time
import pickle
import json
import psutil
from pathlib import Path
from types import SimpleNamespace
from functools import partial
from typing import Callable, List

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

import lightgbm as lgb
import optuna

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import yaml

optuna.logging.set_verbosity(optuna.logging.WARNING)



# ==================== FUNCTION ====================
def load_parameters(params_file_path:str) -> SimpleNamespace:
    with open(params_file_path, 'r') as file:
        para_dict = yaml.safe_load(file)
    for key, value in para_dict.items():
        print(f'{key}: {value}')

    # create the namespace for parameters, use params.features to access
    params = SimpleNamespace(**para_dict)
    return params


def compute_features(group: pd.DataFrame, features: list, labels: list) -> pd.DataFrame:
    """
    Args:
        group: dataframe
        features: contains all input features
        labels: contains all output labels
    
    Returns:
        result_df: dataframe after feature extraction
    """
    group = group.copy()

    # compute the column
    group['relativeTime'] = (group.relativeTime - group.relativeTime.iloc[0]).astype(np.float32)
    # group['discharge_rate'] = (group.SOC.diff()/group.relativeTime.diff()).astype("float")
    group['power'] = (group.current * group.voltage).astype(np.float32)
    group['discharge_power_rate'] = (group.power.diff()/group.relativeTime.diff()).astype(np.float32)
    group['discharge_voltage_rate'] = (group.voltage.diff()/group.relativeTime.diff()).astype(np.float32)
    group['discharge_current_rate'] = (group.current.diff()/group.relativeTime.diff()).astype(np.float32)
    # group['discharge_energy_rate'] = (group.energy.diff()/group.relativeTime.diff()).astype(np.float32)

    group['duration'] = (group['relativeTime'].iloc[-1] - group['relativeTime'].iloc[0]).astype(np.float32)
    group['step_length'] = len(group)
    group['sum_relativeTime'] = (group['relativeTime'].sum()).astype(np.float32)

    for col_name in ['current', 'voltage', 'temperature']:
        group[f'range_{col_name}'] = (group[col_name].max() - group[col_name].min()).astype(np.float32)
        group[f'delta_{col_name}'] = (np.concatenate([[0], cumulative_trapezoid(group[col_name].values, x=group['relativeTime'].values)]) / 3600.0).astype(np.float32)

    # calculate the mean value of all columns in this group
    base_cols = ['cycle']
    # output_list = output_list.append('SOC')
    output_list = list({*base_cols, *features, *labels})
    # output_list = pd.Series(output_list).drop_duplicates().tolist()
    result_df = group[output_list].mean().to_frame().T

    return result_df


def split_helper(data: pd.DataFrame, data_groupby: list, features: list, labels: list,
                 split_group_func: Callable[..., List[pd.DataFrame]]=None) -> pd.DataFrame:
    """
    Args:
        data: dataframe that includes all training/validation/test data
        split_group_func: a callable function that processes the grouped data

    Returns:
        t: dataframe after processing

    Examples:
        from functools import partial

        def process_func(data, features, labels) -> pd.DataFrame:
            return processed_data

        # pass the partial arguments to 'process_func' by name
        partial_train = partial(process_func, features=features, labels=labels)
        train = split_helper(train, features, labels, partial_train)
    """
    all_results = []

    # select the useful columns
    base_cols = ['cycle', 'voltage', 'current', 'relativeTime', 'temperature']
    # base_cols.append('energy')
    # base_cols.append('SOH')
    col_list = list({*base_cols, *labels})  # drop the repeated elements

    for idx, group in data.groupby(data_groupby):
        if group.empty:
            print("Empty data!!!")
            continue

        group = group[col_list].copy()       
        # group = group.sort_values(by='relativeTime', ascending=True)

        if split_group_func is None:
            # directly apply the feature extraction if not split this group
            result = compute_features(group, features, labels)
            if not result.empty:
                all_results.append(result)
        else:
            # call a function to split the group data in this time window
            # this function will return a <list> contains the splited group
            splits_list = split_group_func(group)

            # do the feature extraction for each part in this group            
            for df in splits_list:
                result = compute_features(df, features, labels)
                if not result.empty:
                    all_results.append(result)

    return pd.concat(all_results, ignore_index=True)


def split_without_overlap(group: pd.DataFrame, split_size: int) -> List[pd.DataFrame]:
    length = len(group)
    num_splits = math.floor(length / split_size)  # Calculate number of splits

    splits = []
    start = 0
    for i in range(num_splits):
        end = min(start + split_size, length)
        splits.append(group[start:end])
        start += split_size     # update the start position

    return splits


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
    'metric': 'mae',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1, log=True),
    'max_depth': trial.suggest_int('max_depth', 3, 20),
    'num_leaves': trial.suggest_int('num_leaves', 10, 200),
    'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 1.0, log=True),
    'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 1.0, log=True),
    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
    'verbose': -1,
    'seed': SEED
    }


    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(x, y)
 
    valid_pred = model.predict(vx)
    mae = mean_absolute_error(vy, valid_pred)
 
    return mae


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
        'learning_rate': trial.suggest_float('learning_rate', 0.00001, 0.7),
        # 'num_leaves': trial.suggest_int('num_leaves', 2, 50),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0000, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0000, 10.0),
        'n_estimators' : trial.suggest_int('n_estimators', 50, 700),
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
# ==================== FUNCTION ====================


def main(args):
    """Main function to process arguments"""
    print(f"Received {len(args)} argument(s): {args}")


    # load parameters ====================
    params = load_parameters('model_lightgbm/parameters.yaml')
    # params.multi_split_size = args[1]
    # print(f'params.sequence_length = {params.multi_split_size}')


    # load the data ====================
    # data read path
    processed_data_folder = Path(f'model_lightgbm/data_{params.dataset}')/f'{len(params.features)}features_{len(params.multi_split_size)}splits'
    print(f'load data from {processed_data_folder}')

    dfs = {}
    for prefix in ['train', 'valid', 'test']:
        csv_files = processed_data_folder.glob(f'{prefix}*.csv')
        dfs[prefix] = pd.concat(
            (pd.read_csv(f) for f in csv_files),
            ignore_index=True
        )
    train, valid, test = dfs['train'], dfs['valid'], dfs['test']


    # generate the train, valid, test samples ====================
    x = train[params.features]
    y = train[params.labels]
    vx = valid[params.features]
    vy = valid[params.labels]
    test_x = test[params.features]
    test_y = test[params.labels]


    # select the most important features ====================
    selected_features =  [
        'discharge_voltage_rate',
        'voltage', 
        'temperature', 
        'power'
    ]
    # model_list, selected_feature = select_features(x, y, vx, vy, test_x, test_y)
    # selected_features =  selected_feature[:4]
    # update the samples
    x = train[selected_features]
    vx = valid[selected_features]
    test_x = test[selected_features]


    # get the best parameters ====================
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study_nasa = optuna.create_study(direction="minimize", sampler=sampler)
    study_nasa.optimize(lambda trial: lgb_optimize(trial, x, y, vx, vy), n_trials=params.num_trials, n_jobs=1, show_progress_bar=True)
    best_hyperparameters = study_nasa.best_params
    print(f"Best hyperparameters: {best_hyperparameters}")
    lgbm_params = {
    'objective': 'regression',
    'metric': 'mae',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'verbose': -1,
    'seed': SEED
    }
    best_hyperparameters.update(lgbm_params)


    # fit the model ====================
    best_model = lgb.LGBMRegressor(**best_hyperparameters)
    best_model.fit(
        x,
        y,
        eval_set=[(x, y), (vx, vy)],
        eval_names=['train', 'valid'],
        eval_metric=['mae'],
    )
    # record the evalution result
    evals_result = best_model.evals_result_


    # get the result and print ====================
    final_train_loss = evals_result['train']['l1'][-1]
    final_train_mae = evals_result['train']['l1'][-1]
    final_valid_loss = evals_result['valid']['l1'][-1]
    final_valid_mae = evals_result['valid']['l1'][-1]
    pred = best_model.predict(test_x, verbose=0)
    test_loss = mean_absolute_error(test_y, pred)
    test_mae = mean_absolute_error(test_y, pred)
    r2 = r2_score(test_y, pred)

    print(f"Final Training Loss (MAE): {final_train_loss}, MAE: {final_train_mae}")
    print(f"Final Validation Loss (MAE): {final_valid_loss}, MAE: {final_valid_mae}")
    print(f"Final Test Loss (MAE): {test_loss}, MAE: {test_mae}")
    booster = best_model.booster_
    model_dump = booster.dump_model()

    def count_nodes(tree):
        if 'left_child' in tree and 'right_child' in tree:
            return 1 + count_nodes(tree['left_child']) + count_nodes(tree['right_child'])
        else:
            return 1

    # Calculate the total number of nodes
    num_nodes = sum(count_nodes(tree['tree_structure']) for tree in model_dump['tree_info'])


    
    print("Number of nodes: ", num_nodes)
    print("Estimated memory usage: ", (num_nodes * 4 / 1024), "KB")


    # save the final best model ====================
    # model save path
    model_save_folder = Path(f'model_lightgbm/model_{params.dataset}')
    model_save_folder.mkdir(parents=True, exist_ok=True)
    # generate model name based on test MAE
    mae_decimal_part = f"{test_mae:.8f}".split('.')[1]
    model_name = f'{params.dataset}_{len(selected_features)}features_{int(test_mae)}_{mae_decimal_part}.pkl'
    
    with open(model_save_folder/model_name, 'wb') as f:
        pickle.dump(best_model, f)
    print(f'already saved into {model_save_folder/model_name}')


    # save the result ====================
    # check the csv file exists or not
    filename = 'result_new.csv'
    if not os.path.exists(f'model_lightgbm/{filename}'):
        columns = [
            'Date', 
            'dataset', 
            'model_name', 
            'sequence_length',
            'n_features', 
            'features',
            'train_loss',
            'train_mae',
            'val_loss',
            'val_mae',
            'test_loss',
            'test_mae',
            'r2',
            'learning_rate',
            'max_depth', 
            'reg_alpha', 
            'reg_lambda', 
            'n_estimators'
        ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(f'model_lightgbm/{filename}', index=False)
        print(f"'model_lightgbm/{filename}' does not exist. A new empty file has been created.")
    else:
        print(f"'model_lightgbm/{filename}' already exists.")


    # generate the noew record for this training
    new_record = [
        np.datetime64("now", "m"),      # date
        params.dataset,                 # dataset
        model_name,                     # model_name
        json.dumps(params.multi_split_size),         # seq_len
        f'{len(selected_features)}',    # n_features
        json.dumps(selected_features),  # features
        final_train_loss,
        final_train_mae,
        final_valid_loss,
        final_valid_mae,
        test_loss,                      # test_loss
        test_mae,                       # test_mae
        r2,
        best_hyperparameters['learning_rate'],
        best_hyperparameters['max_depth'],
        best_hyperparameters['reg_alpha'],
        best_hyperparameters['reg_lambda'],
        best_hyperparameters['n_estimators'],
    ]

    # write the new record into csv file
    result_path = Path(f'model_lightgbm/{filename}')
    with open(result_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(new_record)
    print(f'file updated, with mae={test_mae}, on {np.datetime64("now", "m")}')

    return 0


if __name__ == "__main__":
    # The first argument is the script name (bilstm.py), followed by the actual arguments
    input_args = sys.argv[1:]  
    
    # Argument validation
    if not input_args:
        print("Error: At least one argument is required", file=sys.stderr)
        sys.exit(1)
        
    # Call the main logic
    exit_code = main(input_args)
    sys.exit(exit_code)





