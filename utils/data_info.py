import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


__all__ = ['get_unique_cycle_soh', 'prepare_scatter_data', 'plot_column_distribution']


def get_unique_cycle_soh(data:pd.DataFrame) -> dict:
    cycle_soh_dict = (
        data.groupby("cycle")["SOH"]  # Group the data by 'cycle_index' column
        .apply(lambda x: sorted(set(x)))  # Remove duplicates, sort the SOH values
        .to_dict()
    )
    return cycle_soh_dict

# Function to prepare scatter plot data
def prepare_scatter_data(info:dict):
    x = []  # Stores the cycle indices
    y = []  # Stores the corresponding SOH values
    for cycle, soh_list in info.items():
        x.extend([cycle] * len(soh_list))  # Repeat the cycle index for each associated SOH value
        y.extend(soh_list)  # Add the SOH values to the list
    return x, y 



def plot_column_distribution(dataframe, column_name, bins=30, kde=True, figsize=(5, 3), basic_info=True, range_limits=None):
    """
    查看 DataFrame 某一列的分布情况并绘图。
    
    Args:
        dataframe (pd.DataFrame): 数据表。
        column_name (str): 要查看分布的列名。
        bins (int): 直方图的箱数（默认为 30）。
        kde (bool): 是否绘制核密度估计曲线（默认 True）。
        figsize (tuple): 图像的大小（默认为 (5,3)）。
        basic_info (bool): 是否打印基本统计信息（默认 True）。
        range_limits (tuple): 指定分布图的范围 (min, max)（默认 None，表示自动范围）。
    
    Returns:
        None
    """
    # 检查列名是否存在
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    # 提取目标列数据，排除 NaN 值
    data = dataframe[column_name].dropna()

    # 如果指定了范围，过滤数据
    if range_limits:
        min_val, max_val = range_limits
        data = data[(data >= min_val) & (data <= max_val)]
    
    # 打印基本统计信息
    if basic_info:
        print(f"Distribution of '{column_name}':")
        print(data.describe())
    
    # 绘制分布图
    plt.figure(figsize=figsize)
    sns.histplot(data, bins=bins, kde=kde, color="blue", stat="density", alpha=0.6, edgecolor="black")
    plt.title(f"Distribution of '{column_name}'", fontsize=12)
    plt.xlabel(column_name, fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

