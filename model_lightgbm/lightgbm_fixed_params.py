import os
import sys
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
import tensorflow as tf
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

tf.keras.utils.set_random_seed(42)


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
    # update the samples
    x = train[selected_features]
    vx = valid[selected_features]
    test_x = test[selected_features]


    # fit the model ====================
    best_hyperparameters = {
        'learning_rate': 0.13300274898597073,
        'max_depth': 9, 
        'reg_alpha': 0.02635230779686289, 
        'reg_lambda': 3.6334680800784684, 
        'n_estimators': 700
    }
    tf.keras.utils.set_random_seed(42)
    best_model = lgb.LGBMRegressor(**best_hyperparameters)
    best_model.fit(
        x,
        y,
        eval_set=[(x, y), (vx, vy)],
        eval_names=['train', 'valid'],
        eval_metric=['l1','rmse'],
    )
    # record the evalution result
    evals_result = best_model.evals_result_


    # get the result and print ====================
    final_train_loss = evals_result['train']['rmse'][-1]
    final_train_mae = evals_result['train']['l1'][-1]
    final_valid_loss = evals_result['valid']['rmse'][-1]
    final_valid_mae = evals_result['valid']['l1'][-1]
    pred = best_model.predict(test_x, verbose=0)
    test_loss = mean_squared_error(test_y, pred)
    test_mae = mean_absolute_error(test_y, pred)
    r2 = r2_score(test_y, pred)

    print(f"Final Training Loss (RMSE): {final_train_loss}, MAE: {final_train_mae}")
    print(f"Final Validation Loss (RMSE): {final_valid_loss}, MAE: {final_valid_mae}")
    print(f"Final Test Loss (RMSE): {test_loss}, MAE: {test_mae}")


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





