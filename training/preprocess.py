"""
Script to preprocess battery data (e.g. st, randomized).
It can handle reading train/valid/test CSVs, optional normalization,
and windowing by arbitrary group columns.
"""

import os
import yaml
import numpy as np
import pandas as pd
from types import SimpleNamespace
from sklearn.preprocessing import MinMaxScaler

def load_parameters(config_path, dataset_override=None):
    """
    Loads parameters from a YAML file. If the YAML contains a top-level
    'dataset:' key plus a 'datasets:' sub-dict, merges the relevant sub-dict
    with the top-level keys. If dataset_override is provided, that overrides
    the 'dataset:' key in the file.

    Returns a SimpleNamespace with the merged config.
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    # If --dataset was passed, override the YAML's 'dataset' key
    if dataset_override is not None:
        raw["dataset"] = dataset_override

    # If there's a "datasets" block, pick the correct sub-block
    ds_key = raw.get("dataset", None)
    if "datasets" in raw and ds_key in raw["datasets"]:
        dataset_block = raw["datasets"][ds_key]
        # Merge top-level keys with dataset block
        merged = {**raw, **dataset_block}
        # Remove the entire 'datasets' dict to avoid confusion
        #merged.pop("datasets", None)
        # Remove the 'dataset' key as it's no longer needed
        #merged.pop("dataset", None)
        return SimpleNamespace(**merged)
    else:
        # If no sub-block or none found, just convert entire YAML
        return SimpleNamespace(**raw)


def _group_and_window(df, features, labels, window_length, overlap, group_cols):
    """
    Helper to group a single dataframe by `group_cols` and produce
    sliding windows of `window_length` with `overlap`.
    """
    xlist, ylist = [], []

    # Group the dataframe by the specified columns
    grouped = df.groupby(group_cols)

    overlap_win = int(window_length * overlap) if overlap > 0 else 0
    # For each group, create sliding windows
    for _, group in grouped:
        group_features = group[features].to_numpy()
        group_labels = group[labels].to_numpy()
        n = len(group_features)

        # Slide with step = (window_length - overlap)
        step = window_length - overlap_win if window_length > overlap_win else 1

        for i in range(0, n - window_length + 1, step):
            # Features: [i : i+window_length]
            x_win = group_features[i : i + window_length]

            # If labels has 1 column: take the last row's label
            # If multiple label columns: take the entire last row
            if group_labels.ndim == 1:
                y_win = group_labels[i + window_length - 1]
            else:
                y_win = group_labels[i + window_length - 1, :]

            xlist.append(x_win)
            ylist.append(y_win)

    return xlist, ylist


def preprocess_and_window(
    data_path,
    sequence_length=20,
    overlap=0,
    normalize=False,
    features=None,
    labels=None,
    data_groupby=None,
):
    """
    Reads train/valid/test CSV files from `data_path`, applies optional
    MinMax normalization on `features`, groups by `data_groupby`, then
    creates sliding windows of length `sequence_length` with overlap.

    Args:
        data_path (str): Path to the folder containing train.csv, valid.csv, test.csv
        sequence_length (int): Window length
        overlap (int): Overlap size for sliding windows (0 = no overlap)
        normalize (bool): Whether to apply MinMaxScaler to `features`
        features (list[str]): List of feature column names
        labels (list[str]): List of label column names
        data_groupby (list[str]): Columns by which to group data (e.g. ["cycle"], ["filename", "Date"], etc.)

    Returns:
        (xtrain, ytrain, xvalid, yvalid, xtest, ytest) as NumPy arrays
    """
    # Default groupby if none provided
    if not data_groupby:
        data_groupby = ["cycle"]

    # Read the CSVs
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    valid_df = pd.read_csv(os.path.join(data_path, "valid.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))

    # Optionally normalize the feature columns
    if normalize and features is not None:
        scaler = MinMaxScaler()
        train_df[features] = scaler.fit_transform(train_df[features])
        valid_df[features] = scaler.transform(valid_df[features])
        test_df[features] = scaler.transform(test_df[features])

    # Group & Window each split
    train_samples, train_targets = _group_and_window(
        train_df, features, labels, sequence_length, overlap, data_groupby
    )
    # NOTE: Validation has no overlap
    valid_samples, valid_targets = _group_and_window(
        valid_df, features, labels, sequence_length, 0, data_groupby
    )
    # NOTE: Test set has no overlap
    test_samples, test_targets = _group_and_window(
        test_df, features, labels, sequence_length, 0, data_groupby
    )

    # Convert lists to NumPy arrays
    xtrain = np.array(train_samples, dtype=np.float32)
    ytrain = np.array(train_targets, dtype=np.float32)
    xvalid = np.array(valid_samples, dtype=np.float32)
    yvalid = np.array(valid_targets, dtype=np.float32)
    xtest = np.array(test_samples, dtype=np.float32)
    ytest = np.array(test_targets, dtype=np.float32)

    return xtrain, ytrain, xvalid, yvalid, xtest, ytest


def get_first_window_test_set(data_path, sequence_length=20, features=None, labels=None, data_groupby=None):
    """
    Reads the test CSV file from `data_path`, groups by `data_groupby`,
    and extracts only the first sliding window of length `sequence_length`
    from each group.

    Args:
        data_path (str): Path to the folder containing test.csv
        sequence_length (int): Window length
        features (list[str]): List of feature column names
        labels (list[str]): List of label column names
        data_groupby (list[str]): Columns by which to group data (e.g. ["cycle"], ["filename", "Date"], etc.)

    Returns:
        (xtest, ytest) as NumPy arrays containing only the first window from each group.
    """
    # Default groupby if none provided
    if not data_groupby:
        data_groupby = ["cycle"]

    # Read the test CSV
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))

    xlist, ylist = [], []

    # Group the dataframe by the specified columns
    grouped = test_df.groupby(data_groupby)

    # For each group, extract the first window
    for _, group in grouped:
        group_features = group[features].to_numpy()
        group_labels = group[labels].to_numpy()

        if len(group_features) >= sequence_length:
            # Features: first `sequence_length` rows
            x_win = group_features[:sequence_length]

            # If labels has 1 column: take the last row's label
            # If multiple label columns: take the entire last row
            if group_labels.ndim == 1:
                y_win = group_labels[sequence_length - 1]
            else:
                y_win = group_labels[sequence_length - 1, :]

            xlist.append(x_win)
            ylist.append(y_win)

    # Convert lists to NumPy arrays
    xtest = np.array(xlist, dtype=np.float32)
    ytest = np.array(ylist, dtype=np.float32)

    return xtest, ytest


def _group_and_window_with_dates(df, features, labels, date_col, window_length, overlap, group_cols):
        xlist, ylist, datelist = [], [], []
        grouped = df.groupby(group_cols)
        overlap_win = int(window_length * overlap) if overlap > 0 else 0
        step = window_length - overlap_win if window_length > overlap_win else 1

        for _, group in grouped:
            group_features = group[features].to_numpy()
            group_labels = group[labels].to_numpy()
            group_dates = group[date_col].to_numpy() if date_col else None
            n = len(group_features)

            for i in range(0, n - window_length + 1, step):
                x_win = group_features[i : i + window_length]
                if group_labels.ndim == 1:
                    y_win = group_labels[i + window_length - 1]
                    date_win = group_dates[i + window_length -1]
                else:
                    y_win = group_labels[i + window_length - 1, :]
                    date_win = group_dates[i + window_length -1]

                xlist.append(x_win)
                ylist.append(y_win)
                if date_win is not None:
                    datelist.append(date_win)

        return xlist, ylist, datelist

def preprocess_and_window_with_dates(
    data_path,
    sequence_length=20,
    overlap=0,
    normalize=False,
    features=None,
    labels=None,
    date_col=None,
    data_groupby=None,
):
    """
    Similar to `preprocess_and_window`, but also includes the date column
    for each window to help with plotting later.

    Args:
        data_path (str): Path to the folder containing train.csv, valid.csv, test.csv
        sequence_length (int): Window length
        overlap (int): Overlap size for sliding windows (0 = no overlap)
        normalize (bool): Whether to apply MinMaxScaler to `features`
        features (list[str]): List of feature column names
        labels (list[str]): List of label column names
        date_col (str): Name of the date column to include in the output
        data_groupby (list[str]): Columns by which to group data (e.g. ["cycle"], ["filename", "Date"], etc.)

    Returns:
        (xtrain, ytrain, xvalid, yvalid, xtest, ytest, train_dates, valid_dates, test_dates)
        as NumPy arrays, where `*_dates` contains the corresponding date windows.
    """
    if not data_groupby:
        data_groupby = ["cycle"]

    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    valid_df = pd.read_csv(os.path.join(data_path, "valid.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))

    if normalize and features is not None:
        scaler = MinMaxScaler()
        train_df[features] = scaler.fit_transform(train_df[features])
        valid_df[features] = scaler.transform(valid_df[features])
        test_df[features] = scaler.transform(test_df[features])

    train_samples, train_targets, train_dates = _group_and_window_with_dates(
        train_df, features, labels, date_col, sequence_length, overlap, data_groupby
    )
    valid_samples, valid_targets, valid_dates = _group_and_window_with_dates(
        valid_df, features, labels, date_col, sequence_length, 0, data_groupby
    )
    test_samples, test_targets, test_dates = _group_and_window_with_dates(
        test_df, features, labels, date_col, sequence_length, 0, data_groupby
    )

    xtrain = np.array(train_samples, dtype=np.float32)
    ytrain = np.array(train_targets, dtype=np.float32)
    xvalid = np.array(valid_samples, dtype=np.float32)
    yvalid = np.array(valid_targets, dtype=np.float32)
    xtest = np.array(test_samples, dtype=np.float32)
    ytest = np.array(test_targets, dtype=np.float32)
    train_dates = np.array(train_dates, dtype=object) if train_dates else None
    valid_dates = np.array(valid_dates, dtype=object) if valid_dates else None
    test_dates = np.array(test_dates, dtype=object) if test_dates else None

    return xtrain, ytrain, xvalid, yvalid, xtest, ytest, train_dates, valid_dates, test_dates