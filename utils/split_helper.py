from typing import Callable, List

import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.stats import entropy


__all__ = ['split_helper']


def compute_features(
    group: pd.DataFrame,
    features: List[str],
    labels: List[str],
    return_all: bool = True
) -> pd.DataFrame:
    """
    Compute a single-row feature vector from a single DataFrame 'group' that has 
    battery data across time. This function replicates the same computations
    as the multi-window 'extract_features' approach, ensuring consistent outputs.

    Args:
        group: DataFrame containing at least the columns:
            [ 'relativeTime', 'current', 'voltage', 'temperature' ]
            (plus any labels or optional columns like 'cycle').
        features: List of feature names to extract. If you supply ['all'], it returns
                  all computed features. Otherwise, it will only return columns
                  that match items in `features` or in `labels`.
        labels:  List of target columns you also want to preserve in the final output.
        return_all: If True, compute and return all possible features. If False, 
                    return only those in `features` + `labels`.

    Returns:
        A single-row DataFrame (shape = (1, num_features_extracted)).
    
    Notes on Complexity:
        - Time Complexity: O(N), where N = number of rows in 'group'.
        - Space Complexity: O(N) to store intermediate signals/arrays.
    """
    # Defensive copy; sort by relativeTime if not already sorted
    group = group.copy()
    group = group.sort_values('relativeTime', ascending=True).reset_index(drop=True)
    
    # ---------------------------------------------------------------------
    # 1) Basic raw columns
    # ---------------------------------------------------------------------
    # Make sure these columns exist
    required_cols = ['relativeTime', 'current', 'voltage', 'temperature']
    for col in required_cols:
        if col not in group.columns:
            raise ValueError(f"Column '{col}' is missing from the input DataFrame.")

    # Shorthand references
    t = group['relativeTime'].values
    i = group['current'].values
    v = group['voltage'].values
    temp = group['temperature'].values

    # Shift time so that it starts at 0
    t0 = t[0]
    t = t - t0

    # ---------------------------------------------------------------------
    # 2) Compute signals (like the multi-window version)
    # ---------------------------------------------------------------------
    signals = {}

    # Basic signals
    signals["relativeTime"] = t  # shape (N,)
    signals["power"] = i * v     # shape (N,)
    signals["resistance"] = v / (i + 1e-9)

    # Discharge rates (one element shorter if we do diff)
    dt = np.diff(t)  # shape (N-1,)
    signals["discharge_power_rate"] = np.diff(signals["power"]) / dt
    signals["discharge_voltage_rate"] = np.diff(v) / dt
    signals["discharge_current_rate"] = np.diff(i) / dt
    signals["discharge_temperature_rate"] = np.diff(temp) / dt

    # Cumulative trapezoidal integrals (delta_x)
    # dividing by 3600 to convert from (seconds to hours) if desired
    signals["delta_power"] = cumulative_trapezoid(signals["power"], x=t, initial=0.0) / 3600.0
    signals["delta_voltage"] = cumulative_trapezoid(v, x=t, initial=0.0) / 3600.0
    signals["delta_current"] = cumulative_trapezoid(i, x=t, initial=0.0) / 3600.0

    # discharge_soc_rate - replicate the same logic: np.diff(delta_current * delta_voltage)/np.diff(t)
    # This is tricky because everything is shorter by 1 or 2. We'll keep it consistent:
    soc_product = signals["delta_current"] * signals["delta_voltage"]
    signals["discharge_soc_rate"] = np.diff(soc_product) / np.diff(t) if len(soc_product) > 1 else np.array([])

    # approx_C_rate
    # The multi-window version uses data[:, 1:] / abs(delta_current). We replicate:
    # Because we have shape N, the "approx_C_rate" is only definable for indexes [1..N-1].
    # So let's do i[1:] / abs(delta_current[:-1]) for shape consistency.
    if len(i) > 1:
        signals["approx_C_rate"] = i[1:] / (np.abs(signals["delta_current"][:-1]) + 1e-9)
    else:
        signals["approx_C_rate"] = np.array([])

    # approx_SOC -> cumsum(delta_current)/abs(delta_current)
    # The second function does cumsum along axis=1. We do it in 1D.
    # But note that delta_current is already the cumulative integral. Summation again 
    # might be extraneous, so we replicate the exact logic from multi-window approach:
    #   signals['approx_SOC'] = np.cumsum(signals['delta_current'], axis=1)/ np.abs(signals['delta_current'])
    # For 1D data, cumsum of an already "delta_current" is somewhat double integral. 
    # But let's replicate strictly:
    dc = signals["delta_current"]
    if len(dc) > 0:
        cumsum_dc = np.cumsum(dc)  # shape (N,)
        denominator = np.abs(dc) + 1e-9  # avoid division by zero
        signals["approx_SOC"] = cumsum_dc / denominator
    else:
        signals["approx_SOC"] = np.array([])

    # discharge_approx_soc_rate = diff(approx_SOC)/diff(t)
    if len(signals["approx_SOC"]) > 1:
        signals["discharge_approx_soc_rate"] = np.diff(signals["approx_SOC"]) / np.diff(t)
    else:
        signals["discharge_approx_soc_rate"] = np.array([])

    # dV_dQ and dVt_dQt
    #  - dV_dQ = diff(voltage)[1:] / diff(delta_current)
    #  - dVt_dQt = delta_voltage / delta_current
    if len(v) > 2 and len(dc) > 2:
        dv = np.diff(v)
        ddc = np.diff(dc)
        signals["dV_dQ"] = dv / (ddc + 1e-9)  # shift by 1 to match second code
    else:
        signals["dV_dQ"] = np.array([])

    with np.errstate(divide='ignore', invalid='ignore'):
        signals["dVt_dQt"] = np.where(dc != 0, signals["delta_voltage"] / dc, 0)

    # ---------------------------------------------------------------------
    # 3) Additional battery-aging features 
    #    (like the second function 'extract_features')
    # ---------------------------------------------------------------------
    feat_dict = {}  # store final scalar features here

    # coulombic_efficiency = delta_current[-1] / abs(delta_current[0])
    # handle edge cases
    if len(dc) >= 2 and np.abs(dc[0]) > 1e-9:
        feat_dict["coulombic_efficiency"] = dc[-1] / np.abs(dc[0])
    else:
        feat_dict["coulombic_efficiency"] = np.nan

    # energy_efficiency = delta_power[-1] / abs(delta_power[0])
    dp = signals["delta_power"]
    if len(dp) >= 2 and np.abs(dp[0]) > 1e-9:
        feat_dict["energy_efficiency"] = dp[-1] / np.abs(dp[0])
    else:
        feat_dict["energy_efficiency"] = np.nan

    # capacity_fade = delta_current[-1]
    feat_dict["capacity_fade"] = dc[-1] if len(dc) > 0 else np.nan

    # resistance_growth = resistance[-1] - resistance[0]
    r = signals["resistance"]
    if len(r) >= 2:
        feat_dict["resistance_growth"] = r[-1] - r[0]
    else:
        feat_dict["resistance_growth"] = np.nan

    # voltage_hysteresis = max(voltage) - min(voltage)
    feat_dict["voltage_hysteresis"] = v.max() - v.min() if len(v) > 0 else np.nan

    # temperature_rise = temp[-1] - temp[0]
    feat_dict["temperature_rise"] = temp[-1] - temp[0] if len(temp) > 0 else np.nan

    # duration = t[-1] - t[0]
    feat_dict["duration"] = t[-1] - t[0] if len(t) > 0 else 0.0

    # step_length = number of samples
    feat_dict["wlen"] = len(group)

    # total_energy = sum(power[1:] * dt)
    # (the second function uses np.sum(signals["power"][:,1:] * np.diff(...), axis=1))
    # For 1D:
    if len(signals["power"]) > 1:
        feat_dict["total_energy"] = np.sum(signals["power"][1:] * dt)
    else:
        feat_dict["total_energy"] = 0.0

    # total_charge = sum(current[1:] * dt)
    if len(i) > 1:
        feat_dict["total_charge"] = np.sum(i[1:] * dt)
    else:
        feat_dict["total_charge"] = 0.0

    # RMS voltage and RMS current
    feat_dict["rms_voltage"] = np.sqrt(np.mean(v ** 2)) if len(v) > 0 else np.nan
    feat_dict["rms_current"] = np.sqrt(np.mean(i ** 2)) if len(i) > 0 else np.nan

    # Entropy of voltage/current
    # If all v are the same, entropy is 0. If len(v) is small, handle gracefully
    try:
        feat_dict["voltage_entropy"] = entropy(v) if (len(v) > 1 and np.ptp(v) > 0) else 0.0
    except ValueError:
        feat_dict["voltage_entropy"] = 0.0
    try:
        feat_dict["current_entropy"] = entropy(i) if (len(i) > 1 and np.ptp(i) > 0) else 0.0
    except ValueError:
        feat_dict["current_entropy"] = 0.0

    # Peak-to-peak
    feat_dict["voltage_peak_to_peak"] = np.ptp(v) if len(v) > 0 else 0.0
    feat_dict["current_peak_to_peak"] = np.ptp(i) if len(i) > 0 else 0.0

    # Skewness and kurtosis (simple manual computations)
    def _skewness(x):
        m = x.mean()
        s = x.std() + 1e-9
        return np.mean((x - m)**3) / (s**3)

    def _kurtosis(x):
        m = x.mean()
        s = x.std() + 1e-9
        return np.mean((x - m)**4) / (s**4)

    feat_dict["voltage_skewness"] = _skewness(v) if len(v) > 1 else 0.0
    feat_dict["current_skewness"] = _skewness(i) if len(i) > 1 else 0.0
    feat_dict["voltage_kurtosis"] = _kurtosis(v) if len(v) > 1 else 0.0
    feat_dict["current_kurtosis"] = _kurtosis(i) if len(i) > 1 else 0.0

    # Variance
    feat_dict["voltage_variance"] = np.var(v) if len(v) > 1 else 0.0
    feat_dict["current_variance"] = np.var(i) if len(i) > 1 else 0.0

    # SNR
    feat_dict["voltage_snr"] = (v.mean() / (v.std() + 1e-9)) if len(v) > 1 else 0.0
    feat_dict["current_snr"] = (i.mean() / (i.std() + 1e-9)) if len(i) > 1 else 0.0

    # Mean absolute deviation (MAD)
    if len(v) > 0:
        feat_dict["voltage_mad"] = np.mean(np.abs(v - v.mean()))
    else:
        feat_dict["voltage_mad"] = 0.0

    # Cumulative energy / charge (same as total_energy / total_charge above, but
    # we replicate the second function's approach with np.cumsum):
    # for the final value:
    if len(signals["power"]) > 1:
        cume = np.cumsum(signals["power"][1:] * dt)
        feat_dict["cumulative_energy"] = cume[-1]
    else:
        feat_dict["cumulative_energy"] = 0.0

    if len(i) > 1:
        cumq = np.cumsum(i[1:] * dt)
        feat_dict["cumulative_charge"] = cumq[-1]
    else:
        feat_dict["cumulative_charge"] = 0.0

    # ---------------------------------------------------------------------
    # 4) Range, mean, std, etc. for each raw column
    # ---------------------------------------------------------------------
    raw_cols = ['relativeTime', 'current', 'voltage', 'temperature']
    for c in raw_cols:
        arr = group[c].values
        if len(arr) > 0:
            feat_dict[f"range_{c}"] = arr.max() - arr.min()
            feat_dict[f"mean_{c}"] = arr.mean()
            feat_dict[f"std_{c}"] = arr.std()
            feat_dict[f"max_{c}"] = arr.max()
            feat_dict[f"min_{c}"] = arr.min()
            feat_dict[f"diff_{c}"] = arr[-1] - arr[0]
        else:
            feat_dict[f"range_{c}"] = 0.0
            feat_dict[f"mean_{c}"] = 0.0
            feat_dict[f"std_{c}"] = 0.0
            feat_dict[f"max_{c}"] = 0.0
            feat_dict[f"min_{c}"] = 0.0
            feat_dict[f"diff_{c}"] = 0.0

    # ---------------------------------------------------------------------
    # 5) Range, mean, std, etc. for each computed signal
    # ---------------------------------------------------------------------
    for sname, svals in signals.items():
        if len(svals) == 0:
            # fill with 0 or NaN if you prefer
            feat_dict[f"range_{sname}"] = 0.0
            feat_dict[f"mean_{sname}"] = 0.0
            feat_dict[f"std_{sname}"] = 0.0
            feat_dict[f"max_{sname}"] = 0.0
            feat_dict[f"min_{sname}"] = 0.0
            feat_dict[f"diff_{sname}"] = 0.0
            continue

        feat_dict[f"range_{sname}"] = svals.max() - svals.min()
        feat_dict[f"mean_{sname}"] = svals.mean()
        feat_dict[f"std_{sname}"] = svals.std()
        feat_dict[f"max_{sname}"] = svals.max()
        feat_dict[f"min_{sname}"] = svals.min()
        feat_dict[f"diff_{sname}"] = svals[-1] - svals[0]

    # ---------------------------------------------------------------------
    # 6) Also preserve the label columns (and possibly 'cycle')
    #    We'll store their mean or the first value.
    # ---------------------------------------------------------------------
    # base_cols to preserve in final output if present
    base_cols = ['cycle']  # add more if needed (e.g. 'SOH', etc.)
    # We'll store them by taking the first value (or mean, as you prefer)
    for col in base_cols + labels:
        if col in group.columns:
            # store a single representative value; here we take the first
            feat_dict[col] = group[col].iloc[0]
        else:
            feat_dict[col] = np.nan

    # ---------------------------------------------------------------------
    # 7) Convert to DataFrame, filter if not return_all
    # ---------------------------------------------------------------------
    feature_df = pd.DataFrame(feat_dict, index=[0])

    if (not return_all) and (features != ["all"]):
        # Return only those features that are in `features + labels + base_cols`
        # But make sure we don't lose the newly computed ones you actually want
        keep_cols = set(features + labels + base_cols)
        # Keep any columns that match the user request, else drop
        # This means if the user asked specifically for a raw signal, we keep that array-based aggregator.
        # Or you can implement a more direct approach. For simplicity:
        final_cols = []
        for c in feature_df.columns:
            if c in keep_cols:
                final_cols.append(c)
            # If c is a known aggregator of a requested feature, you might keep it, etc.
            # For now, keep it if 'features' == ["all"] or if c is in `keep_cols`.
        if len(final_cols) == 0:
            # fallback: keep everything
            final_cols = list(feature_df.columns)
        feature_df = feature_df[final_cols]

    return feature_df


def split_helper(
    data: pd.DataFrame,
    data_groupby: List[str],
    features: List[str],
    labels: List[str],
    split_group_func: Callable[..., List[pd.DataFrame]] = None
) -> pd.DataFrame:
    """
    Group the DataFrame 'data' by 'data_groupby', then for each group
    call 'compute_features' (or a splitting function that subdivides the group).
    
    Args:
        data: DataFrame that includes all training/validation/test data.
        data_groupby: columns on which to group (e.g. ['cycle']).
        features: list of input features (if you specify ['all'], it returns everything).
        labels: list of label columns that you want to preserve in the result.
        split_group_func: if provided, it should be a function that takes a group
                          and returns a list of sub-DataFrames. We then compute
                          features for each sub-DataFrame individually. If None,
                          we directly compute features on the entire group.

    Returns:
        A concatenated DataFrame containing one row (feature vector) per group 
        or sub-group, depending on whether 'split_group_func' is used.
    """
    all_results = []

    # For safety, define the minimal columns you need
    # (relativeTime, current, voltage, temperature, plus labels, etc.)
    base_cols = ['cycle', 'relativeTime', 'current', 'voltage', 'temperature']
    col_list = list({*base_cols, *labels})

    for idx, group in data.groupby(data_groupby):
        if group.empty:
            print(f"Empty data for group {idx}!")
            continue

        # Restrict to the minimal needed columns
        group = group[col_list].copy()

        if split_group_func is None:
            # Directly compute features
            result = compute_features(group, features=features, labels=labels, return_all=True)
            if not result.empty:
                all_results.append(result)
            else:
                print(f"Empty feature result for group {idx}!")
        else:
            # Use a custom splitting function that returns multiple sub-groups
            subgroup_list = split_group_func(group)
            if not subgroup_list:
                print(f"No sub-groups returned for group {idx}!")
                continue
            for subdf in subgroup_list:
                result = compute_features(subdf, features=features, labels=labels, return_all=True)
                if not result.empty:
                    all_results.append(result)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
