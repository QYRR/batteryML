import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

__all__ = ['data_reshape','create_bilstm_model','bilstm_optmize','data_reshape_groupby']

def data_reshape(data:pd.DataFrame, sequence_length:int, features:list, labels:list) -> tuple[np.array, np.array]:
    data_samples = data[features]
    data_targets = data[labels]

    # # Normalize the features using MinMaxScaler
    # scaler = MinMaxScaler()
    # scaled_samples = scaler.fit_transform(data_samples)

    # Reshape the data to be three-dimensional: samples, timesteps, features
    samples = []    # the input data
    targets = []    # the output data
    for i in range(0, len(data_samples) - sequence_length + 1, sequence_length):
        sample = data_samples[i:i + sequence_length]
        target = data_targets.iloc[i + sequence_length - 1]

        samples.append(sample)
        targets.append(target)

    samples = np.array(samples)
    targets = np.array(targets)

    return samples, targets



def create_bilstm_model(hidden_size:int, learning_rate:float) -> tf.keras.models.Sequential:
    """
    Build a BiLSTM Keras model with the provided hidden_size (units) 
    and learning_rate (for the Adam optimizer).
    """
    print(f'model information -> hidden_size: {hidden_size}, learning_rate: {learning_rate}')
    
    l1 = LSTM(hidden_size)                      # forward LSTM
    l2 = LSTM(hidden_size, go_backwards=True)   # backward LSTM
    
    model = Sequential([
        Input(shape=(20, 4)),  # (timesteps=None, features=4)
        Bidirectional(l1, backward_layer=l2),
        Dense(1)  # single output for regression (loss='mse')
    ])

    # model = Sequential([
    #     Bidirectional(LSTM(14, return_sequences=False, input_shape=(20, 4))),
    #     Dense(1)  # Single output
    # ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model



def bilstm_optmize(trial,train_samples,train_targets,valid_samples,valid_targets):
    """
    An Optuna objective function that:
      - samples hyperparameters (hidden_size, learning_rate)
      - creates a BiLSTM model
      - trains the model
      - returns the best validation loss for that trial
    """
    
    # -- SUGGEST HYPERPARAMETERS --
    # hidden_size = trial.suggest_int('hidden_size', 8, 32, step=8)
    hidden_size = 14
    learning_rate = trial.suggest_float('learning_rate', 5e-3, 1e-2, log=True)
    
    # Build the model
    model = create_bilstm_model(hidden_size, learning_rate)
    
    # Train the model (shorter epochs for speed in HPO)
    early_stop = EarlyStopping(patience=10, restore_best_weights=True)
    history = model.fit(
        train_samples,
        train_targets,
        epochs=30,
        validation_data=(valid_samples, valid_targets),
        verbose=0,
        callbacks=[early_stop]
    )
    # We'll minimize validation loss
    val_loss = min(history.history['val_loss'])
    return val_loss


