import math
from typing import List

import pandas as pd
import numpy as np

# all functions here will return a list contains the splited group
__all__ = ['restrict_group_samples',
           'split_with_overlap_last',
           'split_with_overlap_all',
           'split_without_overlap',
           'split_cycle_data_no_missing',
           'multiple_split']



def restrict_group_samples(group: pd.DataFrame, num_restricted: int) -> List[pd.DataFrame]:
    splits = []
    num_rows = min(num_restricted, len(group))
    splits.append(group.head(num_rows))

    return splits



def split_with_overlap_last(group: pd.DataFrame, split_size: int) -> List[pd.DataFrame]:
    length = len(group)
    num_splits = math.ceil(length / split_size)  # Calculate number of splits

    splits = []
    start = 0
    for i in range(num_splits-1):
        end = min(start + split_size, length)
        splits.append(group[start:end])
        start += split_size     # update the start position
    splits.append(group[-split_size:])

    return splits



def split_with_overlap_all(group: pd.DataFrame, split_size: int, overlap_percentage: float) -> List[pd.DataFrame]:
    assert (overlap_percentage>0 and overlap_percentage<100), "'overlap_percentage' must between 0 and 100 (not include)."
    overlap = round(split_size * overlap_percentage / 100)   
    
    length = len(group)

    splits = []
    start = 0
    end = start + split_size
    while(end<length):        
        splits.append(group[start:end])
        start = end - overlap    # update the start position
        end = start + split_size
    # splits.append(group[-split_size:])
    splits.append(group[start:])

    return splits



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



def split_cycle_data_no_missing(data, threshold):
    """
    Split cycle data into subgroups with dynamically calculated overlap to avoid missing data.
    Parameters:
        data (pd.DataFrame): The cycle data to split.
        threshold (int): The desired length of each subgroup.
    Returns:
        list of pd.DataFrame: Subgroups split from the input cycle data.
    """
    data_length = len(data)
    if threshold >= data_length:
        return [data]
    
    # Dynamically calculate overlap to ensure no values are missed
    num_subgroups = int(np.ceil((data_length - threshold) / (threshold * 0.7)) + 1)
    step = (data_length - threshold) // (num_subgroups - 1)  # Adjust step dynamically
    subgroups = []
    start = 0
    while start < data_length:
        end = start + threshold
        # Ensure the last subgroup fits perfectly within the data length
        if end > data_length:
            start = max(0, data_length - threshold)  # Move start back to fit the last group
            end = data_length
        subgroups.append(data.iloc[start:end])
        # Break the loop if we're at the last group
        if end == data_length:
            break
        # Increment start by the step size
        start += step
    return subgroups



def multiple_split(group: pd.DataFrame, 
                   multiple_split_steps: list, 
                   overlap_mode: str,
                   overlap_percentage: float=30) -> List[pd.DataFrame]: 
    group = group.copy()
    
    splits = []
    for i in multiple_split_steps:
        if overlap_mode == "last":
            splits.extend(split_with_overlap_last(group, i))
        elif overlap_mode == "all":
            # splits.extend(split_with_overlap_all(group, i, overlap_percentage))
            splits.extend(split_cycle_data_no_missing(group,i))
        elif overlap_mode == "no":
            splits.extend(split_without_overlap(group, i))
        else:
            raise ValueError("The value of 'overlap_mode' should be 'last', 'all' or 'no'.")
    
    return splits