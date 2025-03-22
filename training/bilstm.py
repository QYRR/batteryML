import os
import sys
import os
import numpy as np


SEED = 0
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
np.random.seed(SEED)

from cpu import set_cores

set_cores(8)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import mean_absolute_error
from preprocess import preprocess_and_window, load_parameters

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
early_stop = lambda: EarlyStopping(
    patience=10, restore_best_weights=True, verbose=0, monitor="val_mae"
)


def create_bilstm_model(
    hidden_size: int, learning_rate: float, input_shape: tuple
) -> tf.keras.models.Sequential:
    """
    Build a BiLSTM Keras model with the provided hidden_size (units)
    and learning_rate (for the Adam optimizer).
    """
    model = Sequential(
        [
            Input(shape=input_shape),
            Bidirectional(LSTM(hidden_size, return_sequences=False)),
            Dense(1),  # Single output
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate), loss="mae", metrics=["mae"]
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
    epochs,
):
    """
    An Optuna objective function that:
      - samples hyperparameters (hidden_size, learning_rate)
      - creates a BiLSTM model
      - trains the model
      - returns the best validation loss for that trial
    """

    # -- SUGGEST HYPERPARAMETERS --
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])

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
    # We'll minimize validation loss
    val_loss = es.best
    return val_loss


def search():
    # Read the params
    params = load_parameters("model_bilstm/parameters.yaml")
    (
        train_samples,
        train_targets,
        valid_samples,
        valid_targets,
        test_samples,
        test_targets,
    ) = preprocess_and_window(
        params.data_path,
        params.sequence_length,
        params.overlap,
        params.normalize,
        params.features,
        params.labels,
    )

    # get the best parameters ====================
    sampler = optuna.samplers.TPESampler(seed=SEED)
    input_shape = (params.sequence_length, len(params.features))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda trial: bilstm_optmize(
            trial,
            train_samples,
            train_targets,
            valid_samples,
            valid_targets,
            input_shape,
            hidden_size=params.hidden_size,
            epochs=params.epochs,
        ),
        n_trials=params.n_trials,
        n_jobs=1,
        show_progress_bar=True,
    )

    # Re-get the best model
    best_params = study.best_params
    print(f"Best parameters: {best_params}")

    # Build the best model
    tf.keras.utils.set_random_seed(SEED)
    best_model = create_bilstm_model(
        params.hidden_size, best_params["learning_rate"], input_shape
    )
    es = early_stop()
    history = best_model.fit(
        train_samples,
        train_targets,
        epochs=params.epochs,
        batch_size=best_params["batch_size"],
        validation_data=(valid_samples, valid_targets),
        verbose=0,
        callbacks=[es],
    )
    test_predictions = best_model.predict(test_samples, batch_size=len(test_samples))
    valid_predictions = best_model.predict(
        valid_samples, batch_size=best_params["batch_size"]
    )
    test_mae = mean_absolute_error(test_targets, test_predictions)
    valid_mae = mean_absolute_error(valid_targets, valid_predictions)
    print("ES : ", es.best)

    print(f"Test MAE: {test_mae}")
    print(f"Validation MAE: {valid_mae} - Original {history.history['val_mae'][-1]}")


if __name__ == "__main__":
    search()
