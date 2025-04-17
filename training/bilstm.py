import os
import sys
import argparse
import numpy as np

# SEED and environment variables
SEED = 0
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
np.random.seed(SEED)

# Limit CPU cores if needed
from cpu import set_cores
set_cores(8)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, mean_squared_error

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Local imports (adjust paths if needed)
from preprocess import preprocess_and_window, load_parameters
early_stop = lambda: EarlyStopping(
    patience=10, restore_best_weights=True, verbose=0, monitor="val_mae"
)


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


def bilstm_optmize(
    trial,
    train_samples,
    train_targets,
    valid_samples,
    valid_targets,
    input_shape,
    hidden_size,
    epochs
):
    """
    An Optuna objective function that:
      - samples hyperparameters (learning_rate, batch_size)
      - creates a BiLSTM model
      - trains the model
      - returns the best validation loss (MAE).
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    # Build the model
    tf.keras.utils.set_random_seed(SEED)
    model = create_bilstm_model(hidden_size, learning_rate, input_shape)

    # Train the model (shorter epochs for speed in HPO)
    es = early_stop()
    history = model.fit(
        train_samples,
        train_targets,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(valid_samples, valid_targets),
        verbose=0,
        callbacks=[es],
    )

    # We'll minimize the best validation MAE recorded
    val_loss = es.best
    return val_loss


def train_for_one_window(params, window_size):
    """
    Handles:
      1) Loading data at the given window_size
      2) Running Optuna hyperparam search to find best LR/batch
      3) Training a final BiLSTM model with best hyperparams
      4) Evaluating on the test set
      5) Saving the model to `models/bilstm/{dataset}_bilstm_win{window_size}.h5`

    Returns: (test_mae, best_params)
    """
    print(f"\n--- Training for window_size={window_size} ---")

    # 1) Load data
    (
        train_samples,
        train_targets,
        valid_samples,
        valid_targets,
        test_samples,
        test_targets
    ) = preprocess_and_window(
        data_path=params.data_path,
        sequence_length=window_size,
        overlap=params.overlap,
        normalize=params.normalize,
        features=params.features,
        labels=params.labels,
        data_groupby=params.data_groupby  # e.g. ['cycle'] or ['filename','Date']
    )

    # Print some shape info for clarity
    print(f"  Training set size: {len(train_samples)}  Validation set size: {len(valid_samples)}  Test set size: {len(test_samples)}")

    # 2) Optuna hyperparam search
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    input_shape = (window_size, len(params.features))
    study.optimize(
        lambda trial: bilstm_optmize(
            trial,
            train_samples,
            train_targets,
            valid_samples,
            valid_targets,
            input_shape,
            hidden_size=params.hidden_size,
            epochs=params.epochs
        ),
        n_trials=params.n_trials,
        n_jobs=1,
        show_progress_bar=True
    )

    best_params = study.best_params
    print(f"  [window={window_size}] Best hyperparameters from Optuna: {best_params}")

    # 3) Retrain final model with best hyperparams
    tf.keras.utils.set_random_seed(SEED)
    best_model = create_bilstm_model(
        params.hidden_size,
        best_params["learning_rate"],
        input_shape
    )
    es = early_stop()
    best_model.fit(
        train_samples,
        train_targets,
        epochs=params.epochs,
        batch_size=best_params["batch_size"],
        validation_data=(valid_samples, valid_targets),
        verbose=0,
        callbacks=[es],
    )

    # 4) Evaluate on test
    test_predictions = best_model.predict(test_samples)
    test_mae = mean_absolute_error(test_targets, test_predictions)
    r2 = r2_score(test_targets, test_predictions)
    rmse = root_mean_squared_error(test_targets, test_predictions)
    mse = mean_squared_error(test_targets, test_predictions)
    print(f"  [window={window_size}] Final Test MAE: {test_mae:.6f} r2: {r2:.6f} RMSE: {rmse:.6f} MSE: {mse:.6f}")

    # 5) Save model
    dataset_name = params.dataset  # e.g. "st" or "randomized"
    os.makedirs("models/bilstm", exist_ok=True)
    model_path = f"models/bilstm/{dataset_name}_bilstm_win{window_size}.h5"
    best_model.save(model_path)

    print(f"  [window={window_size}] Model saved -> {model_path}")

    return test_mae, best_params, r2, rmse, mse


def main():
    """
    Main entry point:
    - Parse command line (for --dataset)
    - Load bilstm.yaml (with possible override)
    - Loop over multi_window_size (or single sequence_length)
    - Print summary at the end
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Which dataset to train on (e.g. 'st' or 'randomized'). Overrides 'dataset:' in YAML."
    )
    args = parser.parse_args()

    # 1) Load parameters from YAML
    #    This merges top-level keys with the dataset block if it exists.
    params = load_parameters("training/bilstm.yaml", dataset_override=args.dataset)

    # 2) Identify window sizes
    #    If multi_window_size is present, loop over it. Otherwise, use the single sequence_length.
    if hasattr(params, "multi_window_size"):
        window_sizes = params.multi_window_size
    else:
        window_sizes = [params.sequence_length]

    print("=========================================================")
    print("     BiLSTM Training Pipeline")
    print("=========================================================")
    print(f"Dataset selected: {params.dataset}")
    #print(f"Data path: {params.data_path}")
    print(f"Features: {params.features}")
    print(f"Labels: {params.labels}")
    print(f"Overlap: {params.overlap}, Normalize: {params.normalize}")
    print(f"Will train on window sizes: {window_sizes}")
    print("=========================================================\n")

    # 3) Train for each window and store results
    results = []
    for wlen in window_sizes:
        test_mae, best_params, r2, rmse, mse = train_for_one_window(params, wlen)
        results.append((wlen, test_mae, best_params, r2, rmse, mse))

    # 4) Print final summary
    print("\n================ FINAL SUMMARY ================")
    print(f"Dataset: {params.dataset}")
    for (wlen, mae, bpar, r2, rmse, mse) in results:
        print(f" - Window={wlen:2d} | Test MAE={mae:.6f} | r2={r2:.6f} | RMSE={rmse:.6f} | MSE={mse:.6f}| Best Params={bpar}")
    print("===============================================")


if __name__ == "__main__":
    main()
