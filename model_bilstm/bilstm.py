import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf

import optuna

import yaml
from types import SimpleNamespace
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# optuna.logging.set_verbosity(optuna.logging.WARNING)

random_seed = 42
np.random.seed(random_seed)

def bilstm_optmize(trial,train_samples,train_targets,valid_samples,valid_targets,input_shape):
    """
    An Optuna objective function that:
      - samples hyperparameters (hidden_size, learning_rate)
      - creates a BiLSTM model
      - trains the model
      - returns the best validation loss for that trial
    """
    
    # -- SUGGEST HYPERPARAMETERS --
    hidden_size = trial.suggest_int('hidden_size', 8, 32, step=8)
    # hidden_size = 14
    learning_rate = trial.suggest_float('learning_rate', 5e-3, 1e-2, log=True)
    
    # Build the model
    model = create_bilstm_model(hidden_size, learning_rate,input_shape)
    
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


def create_bilstm_model(hidden_size:int, learning_rate:float, input_shape:tuple) -> tf.keras.models.Sequential:
    """
    Build a BiLSTM Keras model with the provided hidden_size (units) 
    and learning_rate (for the Adam optimizer).
    """
    # print(f'model information -> hidden_size: {hidden_size}, learning_rate: {learning_rate}')
    
    # l1 = LSTM(hidden_size)                      # forward LSTM
    # l2 = LSTM(hidden_size, go_backwards=True)   # backward LSTM
    
    # model = Sequential([
    #     Input(shape=input_shape),  # (timesteps=None, features=4)
    #     Bidirectional(l1, backward_layer=l2),
    #     Dense(1)  # single output for regression (loss='mse')
    # ])

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

def main(args):
    """Main function to process arguments"""
    print(f"Received {len(args)} argument(s): {args}")



    params = load_parameters('model_bilstm/parameters.yaml')
    params.sequence_length = int(args[1])
    print(f'params.sequence_length = {params.sequence_length}')
    if params.sequence_length==5:
        learning_rate = 0.006962677097559403
    if params.sequence_length==10:
        learning_rate = 0.006386744479933583
    if params.sequence_length==20:
        learning_rate = 0.006694034786224293
    if params.sequence_length==30:
        learning_rate = 0.00969264043125348
    if params.sequence_length==50:
        learning_rate = 0.005794911438597224
    hidden_size = 14
    input_shape = (params.sequence_length, len(params.features))

    # data read path
    normalized_data = Path(f'model_bilstm/data_{params.dataset}')
    # model save path
    model_save_folder = Path(f'model_bilstm/model_{params.dataset}')
    model_save_folder.mkdir(parents=True, exist_ok=True)

    # load the data --------------------------------------------------------
    data_samples = {}
    data_targets = {}
    for prefix in ['train', 'valid', 'test']:
        # init a list to save the reshaped data for all csv files
        data_samples[prefix] = []
        data_targets[prefix] = []
        # load all data from the source folder
        for csv_file in normalized_data.glob(f'{prefix}_*.csv'):
            df = pd.read_csv(csv_file)

            # for test data, take the first n rows from each group
            if prefix=='test':
                first_n_list = []
                for idx, group in df.groupby(params.data_groupby):
                    if params.sequence_length <= len(group):
                        group_head = group[:params.sequence_length]
                        first_n_list.append(group_head)
                df = pd.concat(first_n_list,ignore_index=True)            

            # reshape the data
            gen = data_reshape_generator(df, params.data_groupby, params.sequence_length, params.features, params.labels)

            while True:
                try:
                    df_sample, df_target = next(gen)
                    data_samples[prefix].append(df_sample)
                    data_targets[prefix].append(df_target)
                except StopIteration:
                    break  # iteration finish
    train_samples = np.concatenate(data_samples['train'], axis=0)
    train_targets = np.concatenate(data_targets['train'], axis=0)

    valid_samples = np.concatenate(data_samples['valid'], axis=0)
    valid_targets = np.concatenate(data_targets['valid'], axis=0)

    test_samples = np.concatenate(data_samples['test'], axis=0)
    test_targets = np.concatenate(data_targets['test'], axis=0)


    # # get the best parameters ---------------------------------------------------------------------------
    # study = optuna.create_study(direction='minimize')
    # study.optimize(
    #     lambda trial: bilstm_optmize(trial,train_samples,train_targets,valid_samples,valid_targets,input_shape),
    #     n_trials=params.n_trials,    # e.g., 10 trials; increase as needed
    #     n_jobs=-1
    # ) 

    # best_params = study.best_params
    # print(f"Best hyperparameters: {best_params}")

    # # create BiLSTM model with the best parameters
    # hidden_size = best_params['hidden_size']
    # final_model = create_bilstm_model(    
    #     hidden_size=hidden_size,
    #     learning_rate=best_params['learning_rate'],
    #     input_shape=input_shape
    # )

    final_model = create_bilstm_model(    
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        input_shape=input_shape
    )

    # fit the model ------------------------------------------------------------------------------------------
    final_model.fit(
        train_samples,
        train_targets,
        epochs=100,
        batch_size=32,
        validation_data=(valid_samples, valid_targets),
        verbose=0
    )

    # Evaluate on test set -----------------------------------------------------------------------------------
    test_loss, test_mae = final_model.evaluate(test_samples, test_targets, verbose=0)
    print(f"Final model - Test Loss: {test_loss}, Test MAE: {test_mae}")

    # Save the final best model -----------------------------------------------------------------
    mae_decimal_part = f"{test_mae:.4f}".split('.')[1]
    model_name = f'{params.dataset}_len{params.sequence_length}_{int(test_mae)}_{mae_decimal_part}.keras'
    final_model.save(model_save_folder/model_name)
    print(f'already saved into {model_save_folder/model_name}')

    import csv
    new_record = [
        np.datetime64("now", "m"),      # date
        params.dataset,                 # dataset
        model_name,                     # model_name
        params.sequence_length,         # seq_len
        hidden_size,                    # hidden_size
        learning_rate,   # learning_rate
        test_loss,                      # test_loss
        test_mae,                       # test_mae
    ]

    result_path = Path('model_bilstm/result.csv')
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
