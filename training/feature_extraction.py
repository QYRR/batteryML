import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.stats import entropy


def extract_features(data, raw_features, feature_list, return_names = False):
    RELATIVETIME = raw_features.index("relativeTime")
    CURRENT = raw_features.index("current")
    VOLTAGE = raw_features.index("voltage")

    signals = {}  # Features with dimensionality that depends on the window size
    features = {}

    # Fix the time to start from 0, relative to the window (n_windows, wlen, n_features)
    signals["relativeTime"] = (
        data[:, :, RELATIVETIME] - data[:, 0, RELATIVETIME][:, np.newaxis]
    )
    signals["power"] = data[:, :, CURRENT] * data[:, :, VOLTAGE]
    signals["resistance"] = data[:, :, VOLTAGE] / (data[:, :, CURRENT] + 1e-9)
    signals["discharge_power_rate"] = np.diff(signals["power"], axis=1) / np.diff(
        signals["relativeTime"], axis=1
    )
    signals["discharge_voltage_rate"] = np.diff(data[:, :, VOLTAGE], axis=1) / np.diff(
        signals["relativeTime"], axis=1
    )
    signals["discharge_current_rate"] = np.diff(data[:, :, CURRENT], axis=1) / np.diff(
        signals["relativeTime"], axis=1
    )

    signals["discharge_temperature_rate"] = np.diff(
        data[:, :, raw_features.index("temperature")], axis=1
    ) / np.diff(signals["relativeTime"], axis=1)

    signals["delta_power"] = (
        cumulative_trapezoid(signals["power"], x=signals["relativeTime"], axis=1)
        / 3600.0
    )
    signals["delta_voltage"] = (cumulative_trapezoid(data[:, :, VOLTAGE], x=signals["relativeTime"], axis=1)
        / 3600.0
    )
    signals["delta_current"] = (
        cumulative_trapezoid(data[:, :, CURRENT], x=signals["relativeTime"], axis=1)
        / 3600.0
    )#[:,np.newaxis]
    #signals['delta_capacity'] = -1.0 * np.concatenate([[0], cumulative_trapezoid(group['current'].values, x=group['relativeTime'].values/3600.0)])
    signals["discharge_soc_rate"] = np.diff(signals["delta_current"]*signals['delta_voltage'], axis=1) / np.diff(
        signals["relativeTime"][:,1:], axis=1
    )
    # Calculate the pseudo state of charge (SOC) and discharge rate
    signals['approx_C_rate'] = data[:, :, CURRENT][:, 1:]/np.abs(signals['delta_current'])
    signals['approx_SOC'] = (
        np.cumsum(signals['delta_current'], axis=1) / np.abs(signals['delta_current'])
    )
    
    signals["discharge_approx_soc_rate"] = np.diff(
        signals["approx_SOC"], axis=1
    ) / np.diff(signals["relativeTime"][:,1:], axis=1)
    
    # Extract features related to battery aging
    # Coulombic efficiency: ratio of charge extracted to charge input
    features["coulombic_efficiency"] = (
        signals["delta_current"][:, -1] / np.abs(signals["delta_current"][:, 0])
    )

    # Energy efficiency: ratio of energy output to energy input
    features["energy_efficiency"] = (
        signals["delta_power"][:, -1] / np.abs(signals["delta_power"][:, 0])
    )

    # Capacity fade: change in capacity over time
    features["capacity_fade"] = signals["delta_current"][:, -1]

    # Internal resistance growth: change in resistance over time
    features["resistance_growth"] = signals["resistance"][:, -1] - signals["resistance"][:, 0]

    # Voltage hysteresis: difference between charge and discharge voltages
    features["voltage_hysteresis"] = (
        np.max(data[:, :, VOLTAGE], axis=1) - np.min(data[:, :, VOLTAGE], axis=1)
    )

    # Temperature rise: change in temperature during discharge
    features["temperature_rise"] = (
        data[:, :, raw_features.index("temperature")][:, -1]
        - data[:, :, raw_features.index("temperature")][:, 0]
    )
    # Error dimension issues, signal has [:,:9], data has [:,:10]
    dQ = np.gradient(-1*signals['delta_current'], axis=1)
    dV = np.gradient(data[:, :, VOLTAGE], axis=1)[:, 1:]  # Match shapes
    dV_dQ = np.divide(dV, dQ, out=np.zeros_like(dV), where=dQ != 0)
    
    signals['dV_dQ'] = dV_dQ
    signals['dVt_dQt'] = signals['delta_voltage'] / signals['delta_current']
    features["duration"] = (
        signals["relativeTime"][:, -1] - signals["relativeTime"][:, 0]
    )
    features["wlen"] = np.ones(data.shape[0]) * data.shape[1]
    # Extract additional features for SOH estimation

    # Total energy delivered during discharge
    features["total_energy"] = np.sum(signals["power"][:,1:] * np.diff(signals["relativeTime"], axis=1), axis=1)

    # Total charge delivered during discharge
    features["total_charge"] = np.sum(data[:, :, CURRENT][:,1:] * np.diff(signals["relativeTime"], axis=1), axis=1)

    # Root mean square (RMS) of voltage
    features["rms_voltage"] = np.sqrt(np.mean(data[:, :, VOLTAGE] ** 2, axis=1))

    # Root mean square (RMS) of current
    features["rms_current"] = np.sqrt(np.mean(data[:, :, CURRENT] ** 2, axis=1))

    # Entropy of voltage signal
    features["voltage_entropy"] = np.apply_along_axis(entropy, axis=1, arr=data[:, :, VOLTAGE])

    # Entropy of current signal
    features["current_entropy"] = np.apply_along_axis(entropy, axis=1, arr=data[:, :, CURRENT])
    # Peak-to-peak amplitude of voltage
    features["voltage_peak_to_peak"] = np.ptp(data[:, :, VOLTAGE], axis=1)

    # Peak-to-peak amplitude of current
    features["current_peak_to_peak"] = np.ptp(data[:, :, CURRENT], axis=1)

    # Skewness of voltage
    features["voltage_skewness"] = np.apply_along_axis(
        lambda x: np.mean((x - np.mean(x))**3) / (np.std(x)**3 + 1e-9),
        axis=1,
        arr=data[:, :, VOLTAGE]
    )

    # Skewness of current
    features["current_skewness"] = np.apply_along_axis(
        lambda x: np.mean((x - np.mean(x))**3) / (np.std(x)**3 + 1e-9),
        axis=1,
        arr=data[:, :, CURRENT]
    )

    # Kurtosis of voltage
    features["voltage_kurtosis"] = np.apply_along_axis(
        lambda x: np.mean((x - np.mean(x))**4) / (np.std(x)**4 + 1e-9),
        axis=1,
        arr=data[:, :, VOLTAGE]
    )

    # Kurtosis of current
    features["current_kurtosis"] = np.apply_along_axis(
        lambda x: np.mean((x - np.mean(x))**4) / (np.std(x)**4 + 1e-9),
        axis=1,
        arr=data[:, :, CURRENT]
    )
    # Add new features for better characterization

    # Variance of voltage
    features["voltage_variance"] = np.var(data[:, :, VOLTAGE], axis=1)

    # Variance of current
    features["current_variance"] = np.var(data[:, :, CURRENT], axis=1)

    # Signal-to-noise ratio (SNR) of voltage
    features["voltage_snr"] = np.mean(data[:, :, VOLTAGE], axis=1) / (np.std(data[:, :, VOLTAGE], axis=1) + 1e-9)

    # Signal-to-noise ratio (SNR) of current
    features["current_snr"] = np.mean(data[:, :, CURRENT], axis=1) / (np.std(data[:, :, CURRENT], axis=1) + 1e-9)

    # Mean absolute deviation (MAD) of voltage
    features["voltage_mad"] = np.mean(np.abs(data[:, :, VOLTAGE] - np.mean(data[:, :, VOLTAGE], axis=1, keepdims=True)), axis=1)

    # Cumulative energy delivered during discharge
    features["cumulative_energy"] = np.cumsum(signals["power"][:, 1:] * np.diff(signals["relativeTime"], axis=1), axis=1)[:, -1]

    # Cumulative charge delivered during discharge
    features["cumulative_charge"] = np.cumsum(data[:, :, CURRENT][:, 1:] * np.diff(signals["relativeTime"], axis=1), axis=1)[:, -1]
    for idx, col_name in enumerate(raw_features):
        features[f"range_{col_name}"] = data[:, :, idx].max(axis=1) - data[
            :, :, idx
        ].min(axis=1)
        features[f"mean_{col_name}"] = np.mean(data[:, :, idx], axis=1)
        features[f"std_{col_name}"] = np.std(data[:, :, idx], axis=1)
        features[f"max_{col_name}"] = np.max(data[:, :, idx], axis=-1)
        features[f"min_{col_name}"] = np.min(data[:, :, idx], axis=-1)
        #features[f"median_{col_name}"] = np.min(data[:, :, idx], axis=-1)
        # Bad idea, it depends on wlen
        #features[f"sum_{col_name}"] = np.sum(data[:, :, idx], axis=1)
        features[f"diff_{col_name}"] = data[:, :, idx][:, -1] - data[:, :, idx][:, 0]

    for idx, col_name in enumerate(signals):
        features[f"range_{col_name}"] = signals[col_name].max(axis=-1) - signals[
            col_name
        ].min(axis=-1)
        features[f"mean_{col_name}"] = np.mean(signals[col_name], axis=-1)
        features[f"max_{col_name}"] = np.max(signals[col_name], axis=-1)
        features[f"min_{col_name}"] = np.min(signals[col_name], axis=-1)
        features[f"std_{col_name}"] = np.std(signals[col_name], axis=-1)
        #features[f"sum_{col_name}"] = np.sum(signals[col_name], axis=-1)
        features[f"diff_{col_name}"] = (
            signals[col_name][:, -1] - signals[col_name][:, 0]
        )

    feature_out = []
    for feature in feature_list:
        if feature_list == ["all"]:
            for f in features:
                fval = features[f].reshape(data.shape[0], -1)
                feature_out.append(fval)
            break

        if feature not in features and feature not in raw_features:
            raise ValueError(f"Feature {feature} not found in the extracted features")
        if feature in features:
            feature_val = features[feature]
        else:
            feature_val = data[:, :, raw_features.index(feature)]

        feature_out.append(feature_val.reshape(data.shape[0], -1))

    features_out = np.concatenate(feature_out, axis=1)
    if return_names:
        return features_out, list(features.keys())
    else:
        return features_out
