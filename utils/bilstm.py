import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

__all__ = ['data_reshape','create_bilstm_model','bilstm_optmize','data_reshape_groupby','data_reshape_generator']

def data_reshape(data:pd.DataFrame, data_groupby:list, sequence_length:int, features:list, labels:list) -> tuple[np.array, np.array]:
    # Reshape the data to be three-dimensional: samples, timesteps, features
    samples = []    # the input data
    targets = []    # the output data

    useful_cols = list({*data_groupby, *features, *labels})
    data = data[useful_cols].copy()

    for idx, group in data.groupby(data_groupby):
        for i in range(0, len(group) - sequence_length + 1, sequence_length):
            group_samples = group[features]
            group_targets = group[labels]
            sample = group_samples[i:i + sequence_length]
            target = group_targets.iloc[i + sequence_length - 1]

            samples.append(sample)
            targets.append(target)        

    samples = np.array(samples)
    targets = np.array(targets)

    return samples, targets


def data_reshape_generator(
    data: pd.DataFrame,
    data_groupby: list,
    sequence_length: int,
    features: list,
    labels: list,
    batch_size: int = 1024
):
    useful_cols = list(set(data_groupby + features + labels))
    data = data[useful_cols].copy()
    
    batch_samples, batch_targets = [], []
    for _, group in data.groupby(data_groupby, sort=False):
        group_features = group[features].to_numpy()
        group_labels = group[labels].to_numpy()
        n = len(group)
        for start in range(0, n - sequence_length + 1, sequence_length):
            end = start + sequence_length
            batch_samples.append(group_features[start:end])
            batch_targets.append(group_labels[end - 1])
            
            if len(batch_samples) >= batch_size:
                yield np.array(batch_samples), np.array(batch_targets)
                batch_samples, batch_targets = [], []
    
    if batch_samples:
        yield np.array(batch_samples), np.array(batch_targets)



# def create_bilstm_model(hidden_size:int, learning_rate:float, input_shape:tuple) -> tf.keras.models.Sequential:
#     """
#     Build a BiLSTM Keras model with the provided hidden_size (units) 
#     and learning_rate (for the Adam optimizer).
#     """
#     print(f'model information -> hidden_size: {hidden_size}, learning_rate: {learning_rate}')
    
#     # l1 = LSTM(hidden_size)                      # forward LSTM
#     # l2 = LSTM(hidden_size, go_backwards=True)   # backward LSTM
    
#     # model = Sequential([
#     #     Input(shape=input_shape),  # (timesteps=None, features=4)
#     #     Bidirectional(l1, backward_layer=l2),
#     #     Dense(1)  # single output for regression (loss='mse')
#     # ])

#     model = Sequential([
#         Bidirectional(LSTM(hidden_size, return_sequences=False, input_shape=input_shape)),
#         Dense(1)  # Single output
#     ])
    
#     model.compile(
#         optimizer=Adam(learning_rate=learning_rate),
#         loss='mse',
#         metrics=['mae']
#     )
    
#     return model



# def bilstm_optmize(trial,train_samples,train_targets,valid_samples,valid_targets,input_shape):
#     """
#     An Optuna objective function that:
#       - samples hyperparameters (hidden_size, learning_rate)
#       - creates a BiLSTM model
#       - trains the model
#       - returns the best validation loss for that trial
#     """
    
#     # -- SUGGEST HYPERPARAMETERS --    
#     hidden_size = trial.suggest_categorical('hidden_size', [8,16,32,64,128,256])
#     # hidden_size = 14
#     learning_rate = trial.suggest_float('learning_rate', 5e-3, 1e-2, log=True)
#     epochs = trial.suggest_categorical("epochs", [50, 100])
#     batch_size = trial.suggest_categorical("batch_size", [32, 64])

#     # Build the model
#     model = create_bilstm_model(hidden_size, learning_rate,input_shape)
#     print(f'epochs: {epochs}, batch_size: {batch_size}')
    
#     # Train the model (shorter epochs for speed in HPO)
#     early_stop = EarlyStopping(patience=10, restore_best_weights=True)
#     history = model.fit(
#         train_samples,
#         train_targets,
#         # epochs=30,
#         epochs = epochs,
#         batch_size = batch_size,
#         validation_data=(valid_samples, valid_targets),
#         verbose=0,
#         callbacks=[early_stop]
#     )
#     # We'll minimize validation loss
#     val_loss = min(history.history['val_loss'])
#     return val_loss


def create_bilstm_model(hidden_size: int, learning_rate: float, window_size: int,
                        dropout_rate: float, recurrent_dropout: float,
                        dense_units: int, num_layers: int) -> tf.keras.models.Sequential:
    """
    Build a more complex BiLSTM model with multiple tunable parameters.
    """
    print(f'Model config -> layers: {num_layers}, hidden: {hidden_size}, '
          f'lr: {learning_rate:.2e}, dropout: {dropout_rate}, rec_dropout: {recurrent_dropout}')
 
    model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Input(shape=(window_size,)))

    for i in range(num_layers):
        return_sequences = i < num_layers - 1  # True for all but last layer
        if i == 0:
            # Input layer
            model.add(tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(hidden_size,
                                   return_sequences=return_sequences,
                                   dropout=dropout_rate,
                                   recurrent_dropout=recurrent_dropout),
                                   kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                   recurrent_initializer=tf.keras.initializers.Orthogonal(),
                input_shape=(window_size, 4)
            ))
        else:
            model.add(tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(hidden_size,
                                   return_sequences=return_sequences,
                                   dropout=dropout_rate,
                                   recurrent_dropout=recurrent_dropout),
                                   kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                   recurrent_initializer=tf.keras.initializers.Orthogonal()
            ))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    # Add final dense layers
    model.add(tf.keras.layers.Dense(dense_units, activation='relu'),kernel_initializer=tf.keras.initializers.HeNormal())
    model.add(tf.keras.layers.Dropout(dropout_rate/2))
    model.add(tf.keras.layers.Dense(1),kernel_initializer=tf.keras.initializers.GlorotUniform())
 
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mae']
    )
    return model
 
def bilstm_optmize(trial, train_samples, train_targets, valid_samples, valid_targets, window_size):
    """
    Enhanced Optuna objective with expanded search space:
    - Variable number of BiLSTM layers
    - Larger hidden sizes
    - Additional dense layers
    - Regularization parameters
    - Batch size tuning
    """
    # Hyperparameter suggestions
    params = {
        'num_layers': trial.suggest_categorical('num_layers', [1,2,3,4,5]),
        'hidden_size': trial.suggest_categorical('hidden_size', [8, 16, 32, 64, 128, 256]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.7, step=0.1),
        'recurrent_dropout': trial.suggest_float('recurrent_dropout', 0.1, 0.5, step=0.1),
        'dense_units': trial.suggest_categorical('dense_units', [32, 64, 128, 256]),        
    }

    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    epochs = trial.suggest_categorical("epochs", [50, 100])
 
    model = create_bilstm_model(window_size=window_size, **params)
    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=15,
        restore_best_weights=True,
        min_delta=1e-4
    )
    history = model.fit(
        train_samples,
        train_targets,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(valid_samples, valid_targets),
        verbose=0,
        callbacks=[early_stop]
    )
    return min(history.history['val_loss'])

