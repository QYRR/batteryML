from cpu import set_cores
import numpy as np

SEED = 0
np.random.seed(SEED)
set_cores(8)


import lightgbm as lgbm
import optuna

from preprocess import preprocess_and_window, load_parameters
from feature_extraction import extract_features

from sklearn.metrics import (
    mean_absolute_error,
)
import math
from typing import Optional

optuna.logging.set_verbosity(optuna.logging.WARNING)


# A default model
def lgbm_model(**params):
    lgbm_params = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "verbose": -1,
        "seed": SEED,
        "n_jobs": 8,
        "deterministic": True,
    }
    params.update(lgbm_params)
    return lgbm.LGBMRegressor(**params)


def lgbm_optimize(trial, x, y, vx, vy):
    """
    Args:
        x: input of tarining data
        y: output of training data
        vx: input of valid data
        vy: output of valid data

    Returns: mean abs error

    """
    lgbm_params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "num_leaves": trial.suggest_int("num_leaves", 10, 200),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 1.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
    }

    model = lgbm_model(**lgbm_params)
    model.fit(x, y, eval_set=[(vx, vy)], eval_metric="mae")

    valid_pred = model.predict(vx)
    mae = mean_absolute_error(vy, valid_pred)

    return mae


def load_params_and_data():
    params = load_parameters("training/lightgbm.yaml")
    train_samples = []
    train_targets = []
    valid_samples = []
    valid_targets = []
    test_samples = []
    test_targets = []

    for sequence_length in params.multi_split_size:
        (
            train_s,
            train_t,
            valid_s,
            valid_t,
            test_s,
            test_t,
        ) = preprocess_and_window(
            params.data_path,
            sequence_length,
            params.overlap,
            params.normalize,
            params.raw_features,
            params.labels,
        )

        train_s = extract_features(train_s, params.raw_features, params.feature_list)
        valid_s = extract_features(valid_s, params.raw_features, params.feature_list)
        test_s = extract_features(test_s, params.raw_features, params.feature_list)

        train_samples.append(train_s)
        train_targets.append(train_t)
        valid_samples.append(valid_s)
        valid_targets.append(valid_t)
        test_samples.append(test_s)
        test_targets.append(test_t)

    train_samples = np.concatenate(train_samples, axis=0)
    valid_samples = np.concatenate(valid_samples, axis=0)
    train_targets = np.concatenate(train_targets, axis=0)
    valid_targets = np.concatenate(valid_targets, axis=0)
    return (
        params,
        train_samples,
        train_targets,
        valid_samples,
        valid_targets,
        test_samples,
        test_targets,
    )


def estimate_memory_usage(
    model: lgbm.LGBMRegressor, limit_to_trees: Optional[list] = None
) -> float:
    """
    Return the estimated kB of memory usage for the model

    Parameters
    ----------
    model : lgbm.LGBMRegressor

    limit_to_trees : Optional[list], optional
        List of trees to consider, by default None (all trees)

    """
    booster = model.booster_
    model_dump = booster.dump_model()

    def count_nodes(tree):
        if "left_child" in tree and "right_child" in tree:
            return (
                1 + count_nodes(tree["left_child"]) + count_nodes(tree["right_child"])
            )
        else:
            return 1

    # Calculate the total number of nodes
    if limit_to_trees is not None:
        num_nodes = sum(
            count_nodes(tree["tree_structure"])
            for idx, tree in enumerate(model_dump["tree_info"])
            if idx in limit_to_trees
        )
    else:
        num_nodes = sum(
            count_nodes(tree["tree_structure"]) for tree in model_dump["tree_info"]
        )

    # Struct of nodes
    # 1. right child
    child_mem = 32 if math.log2(num_nodes) > 16 else 16
    feat_mem = 8
    threshold_mem = 32  # Assume float32

    # NOTE: I am excluding the input buffer size, it's an estimate
    total_mem = num_nodes * (child_mem + feat_mem + threshold_mem)

    return (total_mem / 8) / 1024


def search():
    (
        params,
        train_samples,
        train_targets,
        valid_samples,
        valid_targets,
        test_samples,
        test_targets,
    ) = load_params_and_data()

    # Print the sizes of the datasets
    print(f"Train samples: {train_samples.shape}")
    print(f"Valid samples: {valid_samples.shape}")
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study_nasa = optuna.create_study(direction="minimize", sampler=sampler)
    study_nasa.optimize(
        lambda trial: lgbm_optimize(
            trial, train_samples, train_targets, valid_samples, valid_targets
        ),
        n_trials=params.num_trials,
        n_jobs=1,
        show_progress_bar=True,
    )
    best_hyperparameters = study_nasa.best_params
    best_model = lgbm_model(**best_hyperparameters)
    best_model.fit(
        train_samples,
        train_targets,
        eval_set=[(valid_samples, valid_targets)],
        eval_names=["train", "valid"],
        eval_metric=["mae"],
    )
    print(f"Best hyperparameters: {best_hyperparameters}")
    print("Benchmarking...")
    for idx, split_length in enumerate(params.multi_split_size):
        test_pred = best_model.predict(test_samples[idx], verbose=0)
        test_mae = mean_absolute_error(test_targets[idx], test_pred)
        print(f"Test MAE - WLEN= {split_length}: {test_mae}")


if __name__ == "__main__":
    search()
