from typing import Callable, List

import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid


__all__ = ['split_helper']



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
    group['relativeTime'] = group.relativeTime - group.relativeTime.iloc[0]
    group['discharge_rate'] = (group.SOC.diff()/group.relativeTime.diff()).astype("float")
    group['power'] = group.current * group.voltage
    group['discharge_power_rate'] = (group.power.diff()/group.relativeTime.diff()).astype("float")
    group['discharge_voltage_rate'] = (group.voltage.diff()/group.relativeTime.diff()).astype("float")
    group['discharge_current_rate'] = (group.current.diff()/group.relativeTime.diff()).astype("float")

    group['duration'] = group['relativeTime'].iloc[-1] - group['relativeTime'].iloc[0]
    group['step_length'] = len(group)
    group['sum_relativeTime'] = group['relativeTime'].sum()
    
    # group['range_voltage'] = group['voltage'].max() - group['voltage'].min()
    # group['range_current'] = group['current'].max() - group['current'].min()
    # group['range_temperature'] = group['temperature'].max() - group['temperature'].min()

    # group['delta_current'] = np.concatenate([[0], cumulative_trapezoid(group['current'].values, x=group['relativeTime'].values)]) / 3600.0
    # group['delta_voltage'] = np.concatenate([[0], cumulative_trapezoid(group['voltage'].values, x=group['relativeTime'].values)]) / 3600.0
    # group['delta_temperature'] = np.concatenate([[0], cumulative_trapezoid(group['temperature'].values, x=group['relativeTime'].values)]) / 3600.0

    for col_name in ['current', 'voltage', 'temperature']:
        group[f'range_{col_name}'] = group[col_name].max() - group[col_name].min()
        group[f'delta_{col_name}'] = np.concatenate([[0], cumulative_trapezoid(group[col_name].values, x=group['relativeTime'].values)]) / 3600.0

    # calculate the mean value of all columns in this group
    output_list = ['cycle', 'SOC']+features+labels
    output_list = pd.Series(output_list).drop_duplicates().tolist()
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
    data = data.copy()
    t = pd.DataFrame()

    for idx, group in data.groupby(data_groupby):
        if group.empty:
            print("Empty data!!!")
            continue

        # select the useful columns
        col_list = ['cycle','SOH','voltage','current','SOC','relativeTime','temperature']+labels
        col_list = pd.Series(col_list).drop_duplicates().tolist()
        group = group[col_list]
        
        group = group.sort_values(by='relativeTime', ascending=True)

        result_df = pd.DataFrame()
        if split_group_func is None:
            # directly apply the feature extraction if not split this group
            result_df = compute_features(group, features, labels)
        else:
            # call a function to split the group data in this time window
            # this function will return a <list> contains the splited group
            splits_list = split_group_func(group)

            # do the feature extraction for each part in this group            
            for df in splits_list:
                processed_df = compute_features(df, features, labels)
                result_df = pd.concat([result_df, processed_df], ignore_index=True)

        # concatenate the process result of each group
        t = pd.concat([t, result_df], ignore_index=True)
        t = t.sort_values(by='cycle', ascending=True)

    return t



# def split_with_multiple_fixed_windows(group: pd.DataFrame, 
#                                  features: list, labels: list, 
#                                  num_samples: int=20) -> pd.DataFrame:
#     """
#     Args:
#         group: a time window grouped by 'cycle/date' and 'soh'
#         features: contains all input features
#         labels: contains all output labels
#         num_samples: the interval to split the group
    
#     Returns:
#         result_df: the result after splitting and feature extraction
#     """
#     group = group.copy()

#     # if the length can't be divided by 'num_samples', remove the first excess data
#     length = len(group)  
#     excess = length % num_samples
#     if excess != 0:
#         group = group.iloc[excess:]  # remove the first excess rows

#     # add a new subgroup label, creating a new group for every <num_samples> rows
#     group = group.reset_index(drop=True)
#     group['subgroup'] = (group.index // num_samples).astype(int)

#     # compute features for each subgroup
#     result_df = pd.DataFrame()
#     for idx, subgroup in group.groupby('subgroup'):
#         new_df = compute_features(subgroup, features, labels)
#         result_df = pd.concat([result_df, new_df], ignore_index=True)  # concatenate results

#     return result_df

