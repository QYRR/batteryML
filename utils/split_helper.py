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
    group['relativeTime'] = (group.relativeTime - group.relativeTime.iloc[0])#.astype(np.float32)
    # group['discharge_rate'] = (group.SOC.diff()/group.relativeTime.diff()).astype("float")
    group['power'] = (group.current * group.voltage)#.astype(np.float32)
    group['discharge_power_rate'] = (group.power.diff()/group.relativeTime.diff())#.astype(np.float32)
    group['discharge_voltage_rate'] = (group.voltage.diff()/group.relativeTime.diff())#.astype(np.float32)
    group['discharge_current_rate'] = (group.current.diff()/group.relativeTime.diff())#.astype(np.float32)
    # group['discharge_energy_rate'] = (group.energy.diff()/group.relativeTime.diff()).astype(np.float32)

    group['duration'] = (group['relativeTime'].iloc[-1] - group['relativeTime'].iloc[0])#.astype(np.float32)
    group['step_length'] = len(group)
    group['sum_relativeTime'] = (group['relativeTime'].sum())#.astype(np.float32)
    group['resistance'] = (group.voltage / (group.current + 1e-9))#.astype(np.float32)
    group['discharge_temperature_rate'] = (group.temperature.diff() / group.relativeTime.diff())#.astype(np.float32)
    group['delta_power'] = (np.concatenate([[0], cumulative_trapezoid(group['power'].values, x=group['relativeTime'].values)]) / 3600.0)#.astype(np.float32)
    group['delta_voltage'] = (np.concatenate([[0], cumulative_trapezoid(group['voltage'].values, x=group['relativeTime'].values)]) / 3600.0)#.astype(np.float32)

    # Compute total energy and total charge
    group['total_energy'] = np.sum(group['power'][1:] * np.diff(group['relativeTime']), axis=0)
    group['total_charge'] = np.sum(group['current'][1:] * np.diff(group['relativeTime']), axis=0)
    group['delta_current'] = (np.concatenate([[0], cumulative_trapezoid(group['current'].values, x=group['relativeTime'].values)]) / 3600.0)#.astype(np.float32)
    group['discharge_soc_rate'] = (group['delta_current'].diff() * group['delta_voltage'].diff() / group['relativeTime'].diff())#.astype(np.float32)
    group['approx_C_rate'] = (group['current'] / group['delta_current'].abs())#.astype(np.float32)
    group['approx_SOC'] = (group['delta_current'].cumsum() / group['delta_current'].abs())#.astype(np.float32)
    group['discharge_approx_soc_rate'] = (group['approx_SOC'].diff() / group['relativeTime'].diff())#.astype(np.float32)
    group['dV_dQ'] = (group['voltage'].diff() / group['delta_current'].diff())#.astype(np.float32)
    group['dVt_dQt'] = (group['delta_voltage'] / group['delta_current'])#.astype(np.float32)
    
    for col_name in group.columns:
        group[f'min_{col_name}'] = group[col_name].min()#.astype(np.float32)
        group[f'max_{col_name}'] = group[col_name].max()#.astype(np.float32)
        group[f'range_{col_name}'] = (group[col_name].max() - group[col_name].min())#.astype(np.float32)
        group[f'mean_{col_name}'] = group[col_name].mean()#.astype(np.float32)
        group[f'std_{col_name}'] = group[col_name].std()#.astype(np.float32)
        #group[f'mean_range_{col_name}'] = (group[col_name].mean() - group[col_name].min()).astype(np.float32)
        #group[f'delta_{col_name}'] = (cumulative_trapezoid(group[col_name].values, x=group['relativeTime'].values) / 3600.0).astype(np.float32)
    '''for col_name in ['current', 'voltage', 'temperature']:
        group[f'range_{col_name}'] = (group[col_name].max() - group[col_name].min()).astype(np.float32)
        group[f'delta_{col_name}'] = (np.concatenate([[0], cumulative_trapezoid(group[col_name].values, x=group['relativeTime'].values)]) / 3600.0).astype(np.float32)'''
    #print(group.columns)
    # calculate the mean value of all columns in this group
    #base_cols = ['cycle']
    # output_list = output_list.append('SOC')
    #print(group.columns)
   # output_list = list({*base_cols, *features, *labels})
    # output_list = pd.Series(output_list).drop_duplicates().tolist()
    #result_df = group[output_list].mean().to_frame().T

    return group



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
        #print(group)
        #if split_group_func is None:
        #print(f"Group {idx} has {len(group)} rows")
        # directly apply the feature extraction if not split this group
        result = compute_features(group, features, labels)
        print(result)
        if not result.empty:
            all_results.append(result)
        else:
            print("Empty result!!!")
        '''else:
            #print(group)
            # call a function to split the group data in this time window
            # this function will return a <list> contains the splited group
            splits_list = split_group_func(group)

            # do the feature extraction for each part in this group            
            for df in splits_list:
                result = compute_features(df, features, labels)
                if not result.empty:
                    all_results.append(result)'''

    return pd.concat(all_results, ignore_index=True)



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

