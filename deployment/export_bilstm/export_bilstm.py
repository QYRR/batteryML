"""
A simple script to export the untrained models to a file, just 
for latency/memory profiling purposes on device
- Requires tensorflow 2.4.0 
CARE: If tensorflow is too new CubeAI seems not to load correctly the model.
CARE N.2: tensorflow 2.4.0 is compatible with python 3.8, not later versions!
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM


def create_bilstm_model(hidden_size: int, learning_rate: float, input_shape: tuple):
    """
    Build a BiLSTM Keras model with the provided hidden_size (units)
    and learning_rate (for the Adam optimizer).
    """
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(hidden_size, return_sequences=False)),
        Dense(1),  # Single output
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mae",
        metrics=["mae"]
    )
    return model


def export_bilstms():
    hidden_size = 14
    learning_rate=0.01
    
    for input_len in [20, 30, 40, 50, 60, None]:
        input_shape = (input_len, 4)
        model = create_bilstm_model(hidden_size, learning_rate, input_shape)

        model.summary()
        model.save(f"deployment/export_bilstm/bilstm_wlen_{str(input_len).lower()}.h5")


if __name__ == "__main__":
    export_bilstms()