import pandas as pd
import yaml
from types import SimpleNamespace

__all__ = ['rename_and_filter_columns', 'load_parameters']

# FUNCTION --------------------------------------------------------------
def rename_and_filter_columns(df: pd.DataFrame, col_dict: dict):
    """
    Rename columns in a DataFrame based on a dictionary and retain only the renamed columns.

    Parameters:
    - df: pandas DataFrame, input data.
    - col_dict: Dictionary where the key is the original column name and the value is the new column name.

    Returns:
    - A new DataFrame with renamed columns, retaining only the specified columns.
    """
    # Check if the columns in the dictionary exist in the DataFrame
    df = df.copy()
    existing_columns = {key: value for key, value in col_dict.items() if key in df.columns}

    if not existing_columns:
        print("No matching column names found in the dictionary. Please check the input.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Rename and retain only the specified columns
    renamed_df = df[existing_columns.keys()].rename(columns=existing_columns)

    return renamed_df
# FUNCTION ------------------------------------------------------------------------------------



def load_parameters(params_file_path:str) -> SimpleNamespace:
    with open(params_file_path, 'r') as file:
        para_dict = yaml.safe_load(file)
    for key, value in para_dict.items():
        print(f'{key}: {value}')

    # create the namespace for parameters, use params.features to access
    params = SimpleNamespace(**para_dict)
    return params
