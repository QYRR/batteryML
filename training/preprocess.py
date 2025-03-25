""" 
Script to preprocess the nasa dataset
"""
import glob
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from types import SimpleNamespace
import yaml
import numpy as np

def load_parameters(params_file_path:str) -> SimpleNamespace:
    with open(params_file_path, 'r') as file:
        para_dict = yaml.safe_load(file)
    for key, value in para_dict.items():
        print(f'{key}: {value}')

    # create the namespace for parameters, use params.features to access
    params = SimpleNamespace(**para_dict)
    return params

def preprocess_and_window(data_path, window_length=20, overlap=0, normalize=False, features = None, labels = None):
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    valid_df = pd.read_csv(os.path.join(data_path, "valid.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))

    # Normalize the data with MinMaxScaler
    if normalize:
        scaler = MinMaxScaler()
        train_df[features] = scaler.fit_transform(train_df[features])
        valid_df[features] = scaler.transform(valid_df[features])
        test_df[features] = scaler.transform(test_df[features])
    
    # Group by the "cycle" column and window the data
    train_samples = []
    valid_samples = []
    test_samples = []

    train_targets = []
    valid_targets = []
    test_targets = []

    for df, xlist, ylist in zip(
        [train_df, valid_df, test_df],
        [train_samples, valid_samples, test_samples],
        [train_targets, valid_targets, test_targets]
    ):
        # For each cycle
        for cycle in df['cycle'].unique():
            group_features = df[df['cycle'] == cycle][features].to_numpy()
            group_labels = df[df['cycle'] == cycle][labels].to_numpy()
            n = len(group_features)

            # Create the windows
            for i in range(0, n - window_length + 1, window_length - overlap):
                xlist.append(group_features[i:i + window_length])
                ylist.append(group_labels[i + window_length - 1])
    
    xtrain = np.array(train_samples)
    ytrain = np.array(train_targets)
    xvalid = np.array(valid_samples)
    yvalid = np.array(valid_targets)
    xtest = np.array(test_samples)
    ytest = np.array(test_targets)


    return xtrain, ytrain, xvalid, yvalid, xtest, ytest




