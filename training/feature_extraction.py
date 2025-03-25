import numpy as np
from scipy.integrate import cumulative_trapezoid


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
    signals["delta_voltage"] = (
        cumulative_trapezoid(data[:, :, VOLTAGE], x=signals["relativeTime"], axis=1)
        / 3600.0
    )
    signals["delta_current"] = (
        cumulative_trapezoid(data[:, :, CURRENT], x=signals["relativeTime"], axis=1)
        / 3600.0
    )
    # Error dimension issues, signal has [:,:9], data has [:,:10]
    '''dQ = np.gradient(signals['delta_current'], axis=1)
    dV = np.gradient(data[:,:,VOLTAGE], axis=1)
    dV_dQ = np.divide(dV, dQ, out=np.zeros_like(dV), where=dQ != 0)
    signals['dV_dQ'] = dV_dQ'''

    features["duration"] = (
        signals["relativeTime"][:, -1] - signals["relativeTime"][:, 0]
    )
    features["wlen"] = np.ones(data.shape[0]) * data.shape[1]

    for idx, col_name in enumerate(raw_features):
        features[f"range_{col_name}"] = data[:, :, idx].max(axis=1) - data[
            :, :, idx
        ].min(axis=1)
        features[f"mean_{col_name}"] = np.mean(data[:, :, idx], axis=1)
        features[f"std_{col_name}"] = np.std(data[:, :, idx], axis=1)
        # Bad idea, it depends on wlen
        #features[f"sum_{col_name}"] = np.sum(data[:, :, idx], axis=1)
        features[f"diff_{col_name}"] = data[:, :, idx][:, -1] - data[:, :, idx][:, 0]

    for idx, col_name in enumerate(signals):
        features[f"range_{col_name}"] = signals[col_name].max(axis=-1) - signals[
            col_name
        ].min(axis=-1)
        features[f"mean_{col_name}"] = np.mean(signals[col_name], axis=-1)
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
