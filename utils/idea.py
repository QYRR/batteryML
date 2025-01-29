import pandas as pd
import numpy as np

__all__ = ['quadratic_fit', 'remove_outliers']

def quadratic_fit(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fit data grouped by 'cycle' using a quadratic polynomial (ax^2 + bx + c).
    
    Parameters:
    - data: pd.DataFrame, must contain 'cycle', 'voltage', and 'discharge_power_rate' columns.
    
    Returns:
    - pd.DataFrame with columns ['cycle', 'a', 'b', 'c', 'R^2'].
    """
    # Ensure the data contains necessary columns
    required_columns = {'cycle', 'voltage', 'discharge_power_rate'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"Input data must contain columns: {required_columns}")

    # Group data by 'cycle'
    grouped = data.groupby('cycle')

    # Prepare to store results
    fit_results = []

    # Fit each group with a quadratic polynomial
    for cycle, group in grouped:
        # Extract x and y for the current group
        x = group['voltage'].values
        y = group['discharge_power_rate'].values

        # Fit a quadratic polynomial
        coefficients = np.polyfit(x, y, 2)  # Fit to ax^2 + bx + c

        # Store the results
        fit_results.append({
            'cycle': cycle, 
            'a': coefficients[0], 
            'b': coefficients[1], 
            'c': coefficients[2], 
            'capacity': group['capacity'].mean()
        })

    # Convert results to DataFrame
    return pd.DataFrame(fit_results)



def remove_outliers(df, threshold=3):
    """
    使用 3 个标准差规则去除 DataFrame 中每列的异常值。
    
    参数:
        df (pd.DataFrame): 输入 DataFrame。
        threshold (float): 标准差阈值，默认为 3。
    
    返回:
        pd.DataFrame: 去除异常值后的 DataFrame。
    """
    # 计算每列的均值和标准差
    mean = df.mean()
    std = df.std()
    
    # 判断每个值是否在均值 ± (threshold * 标准差) 范围内
    mask = ((df - mean).abs() <= threshold * std)
    
    # 按行保留所有列都满足条件的样本
    return df[mask.all(axis=1)]

# # remove the outlier
# train = remove_outliers(train)
# valid = remove_outliers(valid)
# test = remove_outliers(test)

# # fit quadratic
# train = quadratic_fit(train)
# valid = quadratic_fit(valid)
# test = quadratic_fit(test)
# features = ['a','b','c']

