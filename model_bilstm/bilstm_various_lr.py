import os
import sys
import csv
from pathlib import Path
from types import SimpleNamespace
import yaml

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 0
tf.keras.utils.set_random_seed(SEED)


# ==================== FUNCTION ====================
def bilstm_optmize(trial,train_samples,train_targets,valid_samples,valid_targets,input_shape):
    """
    An Optuna objective function that:
      - samples hyperparameters (hidden_size, learning_rate)
      - creates a BiLSTM model
      - trains the model
      - returns the best validation loss for that trial
    """
    
    # -- SUGGEST HYPERPARAMETERS --
    hidden_size = 14
    learning_rate = trial.suggest_float('learning_rate', 5e-3, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])

    # Build the model
    tf.keras.utils.set_random_seed(42)
    model = create_bilstm_model(hidden_size, learning_rate,input_shape)
    
    # Train the model (shorter epochs for speed in HPO)
    early_stop = EarlyStopping(patience=10, restore_best_weights=True)
    history = model.fit(
        train_samples,
        train_targets,
        epochs=30,
        batch_size=batch_size,
        validation_data=(valid_samples, valid_targets),
        verbose=0,
        callbacks=[early_stop]
    )
    # We'll minimize validation loss
    val_loss = min(history.history['val_loss'])
    return val_loss


def data_reshape_generator(
    data: pd.DataFrame,
    data_groupby: list,
    sequence_length: int,
    features: list,
    labels: list,
    batch_size: int = 1024
):
    # convert the data to tensor format
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


def create_bilstm_model(hidden_size:int, learning_rate:float, input_shape:tuple) -> tf.keras.models.Sequential:
    """
    Build a BiLSTM Keras model with the provided hidden_size (units) 
    and learning_rate (for the Adam optimizer).
    """
    model = Sequential([
        Bidirectional(LSTM(hidden_size, return_sequences=False, input_shape=input_shape)),
        Dense(1)  # Single output
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def load_parameters(params_file_path:str) -> SimpleNamespace:
    with open(params_file_path, 'r') as file:
        para_dict = yaml.safe_load(file)
    for key, value in para_dict.items():
        print(f'{key}: {value}')

    # create the namespace for parameters, use params.features to access
    params = SimpleNamespace(**para_dict)
    return params
# ==================== FUNCTION ====================


def main(args):
    """Main function to process arguments"""
    print(f"Received {len(args)} argument(s): {args}")


    # load parameters ====================
    params = load_parameters('model_bilstm/parameters.yaml')
    params.sequence_length = int(args[1])
    print(f'params.sequence_length = {params.sequence_length}')


    # load and reshape the data ====================
    # data read path
    normalized_data = Path(f'model_bilstm/data_{params.dataset}')
    
    data_samples = {}
    data_targets = {}
    for prefix in ['train', 'valid', 'test']:
        # init a list to save the reshaped data for all csv files
        data_samples[prefix] = []
        data_targets[prefix] = []
        # load all data from the source folder
        for csv_file in normalized_data.glob(f'{prefix}_*.csv'):
            df = pd.read_csv(csv_file)        

            # reshape the data
            gen = data_reshape_generator(df, params.data_groupby, params.sequence_length, params.features, params.labels)

            while True:
                try:
                    df_sample, df_target = next(gen)
                    data_samples[prefix].append(df_sample)
                    data_targets[prefix].append(df_target)
                except StopIteration:
                    break  # iteration finish


    # generate the train, valid, test samples ====================
    train_samples = np.concatenate(data_samples['train'], axis=0)
    train_targets = np.concatenate(data_targets['train'], axis=0)

    valid_samples = np.concatenate(data_samples['valid'], axis=0)
    valid_targets = np.concatenate(data_targets['valid'], axis=0)

    test_samples = np.concatenate(data_samples['test'], axis=0)
    test_targets = np.concatenate(data_targets['test'], axis=0)


    # get the best parameters ====================
    input_shape = (params.sequence_length, len(params.features))
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda trial: bilstm_optmize(trial,train_samples,train_targets,valid_samples,valid_targets,input_shape),
        n_trials=params.n_trials,    # e.g., 10 trials; increase as needed
        n_jobs=1
    ) 

    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")

    # create BiLSTM model ====================
    hidden_size = 14    
    learning_rate = best_params['learning_rate']
    batch_size = best_params['batch_size']

    tf.keras.utils.set_random_seed(42)
    final_model = create_bilstm_model(    
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        input_shape=input_shape
    )


    # fit the model ====================    
    history = final_model.fit(
        train_samples,
        train_targets,
        epochs=100,
        batch_size=batch_size,
        validation_data=(valid_samples, valid_targets),
        verbose=0
    )


    # get the result and print ====================
    final_train_loss = history.history['loss'][-1]
    final_train_mae = history.history['mae'][-1]
    final_valid_loss = history.history['val_loss'][-1]
    final_valid_mae = history.history['val_mae'][-1]
    pred = final_model.predict(test_samples, verbose=0)
    test_loss = mean_squared_error(test_targets, pred)
    test_mae = mean_absolute_error(test_targets, pred)
    r2 = r2_score(test_targets, pred)

    print(f"Final Training Loss (MSE): {final_train_loss}, MAE: {final_train_mae}")
    print(f"Final Validation Loss (MSE): {final_valid_loss}, MAE: {final_valid_mae}")
    print(f"Final Test Loss (MSE): {test_loss}, MAE: {test_mae}")


    # save the final best model ====================
    # model save path
    model_save_folder = Path(f'model_bilstm/model_{params.dataset}')
    model_save_folder.mkdir(parents=True, exist_ok=True)
    # generate model name based on test MAE
    mae_decimal_part = f"{test_mae:.8f}".split('.')[1]    
    model_name = f'{params.dataset}_len{params.sequence_length}_{int(test_mae)}_{mae_decimal_part}.keras'
    final_model.save(model_save_folder/model_name)
    print(f'already saved into {model_save_folder/model_name}')


    # save the result ====================
    # check the csv file exists or not
    filename = 'result_new1.csv'
    if not os.path.exists(f'model_bilstm/{filename}'):
        columns = [
            'Date', 
            'dataset', 
            'model_name', 
            'sequence_length', 
            'hidden_size',
            'learning_rate',
            'batch_size',
            'train_loss',
            'train_mae',
            'val_loss',
            'val_mae',
            'test_loss',
            'test_mae',
            'r2',
        ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(f'model_bilstm/{filename}', index=False)
        print(f"'model_bilstm/{filename}' does not exist. A new empty file has been created.")
    else:
        print(f"'model_bilstm/{filename}' already exists.")

    # generate the noew record for this training
    new_record = [
        np.datetime64("now", "m"),      # date
        params.dataset,                 # dataset
        model_name,                     # model_name
        params.sequence_length,         # seq_len
        hidden_size,                    # hidden_size
        learning_rate,                  # learning_rate
        batch_size,
        final_train_loss,
        final_train_mae,
        final_valid_loss,
        final_valid_mae,
        test_loss,                      # test_loss
        test_mae,                       # test_mae
        r2
    ]

    # write the new record into csv file
    result_path = Path(f'model_bilstm/{filename}')
    with open(result_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(new_record)
    print(f'file updated, with mae={test_mae}, on {np.datetime64("now", "m")}')

    return 0


if __name__ == "__main__":
    # The first argument is the script name (bilstm.py), followed by the actual arguments
    input_args = sys.argv[1:]  
    
    # Argument validation
    if not input_args:
        print("Error: At least one argument is required", file=sys.stderr)
        sys.exit(1)
        
    # Call the main logic
    exit_code = main(input_args)
    sys.exit(exit_code)
