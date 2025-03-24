import numpy as np

def extract_features(data, idx_to_name_map, feature_list):
    RELATIVETIME = idx_to_name_map['relativeTime']
    CURRENT = idx_to_name_map['current']
    VOLTAGE = idx_to_name_map['voltage']

    features_windows = []

    # Fix the time to start from 0, relative to the window (n_windows, wlen, n_features)
    time = data[:, :, RELATIVETIME] - data[:, 0, RELATIVETIME][:, np.newaxis]
    power = data[:, :, CURRENT] * data[:, :, VOLTAGE]
    discharge_power_rate = np.diff(power, axis=1) / np.diff(time, axis=1)
    discharge_voltage_rate = np.diff(data[:, :, VOLTAGE], axis=1) / np.diff(time, axis=1)
    discharge_current_rate = np.diff(data[:, :, CURRENT], axis=1) / np.diff(time, axis=1)

    duration = time[:, -1][:, np.newaxis]

    



        

