# training/ensemble.py

import os
import sys
import argparse
import numpy as np

SEED = 0
np.random.seed(SEED)

from cpu import set_cores
set_cores(4)

import lightgbm as lgbm
import optuna

from preprocess import preprocess_and_window, load_parameters
from feature_extraction import extract_features
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import pickle
from eden.frontend.lightgbm import parse_boosting_trees
from eden.model import Ensemble
import pandas as pd
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
        "n_jobs": 4,
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
        'learning_rate': trial.suggest_float('learning_rate', 0.00001, 0.7, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'num_leaves': trial.suggest_int('num_leaves', 2, 25),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 450),
        'min_child_samples': trial.suggest_int('min_child_samples', 2, 25),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
    }

    model = lgbm_model(**lgbm_params)
    model.fit(x, y, eval_set=[(vx, vy)], eval_metric="mae")

    valid_pred = model.predict(vx)
    mae = mean_absolute_error(vy, valid_pred)

    return mae


# def estimate_memory_usage(model: lgbm.LGBMRegressor):
#     """
#     Roughly estimate the memory usage (KB) of a trained LightGBM model.
#     """
#     booster = model.booster_
#     model_dump = booster.dump_model()

#     def count_nodes(tree):
#         if "left_child" in tree and "right_child" in tree:
#             return 1 + count_nodes(tree["left_child"]) + count_nodes(tree["right_child"])
#         else:
#             return 1

#     # Count total nodes in all trees
#     nodi = [count_nodes(tree["tree_structure"]) for tree in model_dump["tree_info"]]
#     num_nodes = sum(nodi)

#     # Very rough memory estimate
#     child_mem = 32       # bits
#     feat_mem = 8         # bits
#     threshold_mem = 32   # bits
#     total_mem_bits = num_nodes * (child_mem + feat_mem + threshold_mem)
#     total_mem_kb = (total_mem_bits / 8) / 1024
#     return total_mem_kb


def get_memory_usage(model: lgbm.LGBMRegressor):
    """
    Get the memory usage (KB) of a trained LightGBM model by eden.
    """
    emodel: Ensemble = parse_boosting_trees(model=model)
    memory_cost = emodel.get_memory_cost()
    total_mem_bytes = sum(memory_cost.values())
    total_mem_kb = total_mem_bytes / 1024 

    return total_mem_kb


def clean_after_feature_extraction(samples, targets, names=None):
    # Remove rows with NaN values
    temp_train = pd.DataFrame(samples, columns=names)
    temp_train["target"] = targets
    temp_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    temp_train.dropna(inplace=True)
    samples = temp_train.values[:, :-1]
    targets = temp_train.values[:, -1]
    #print(targets)
    return samples, targets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
        help="Choose which dataset block to use in lightgbm.yaml (e.g. 'st' or 'randomized').")
    args = parser.parse_args()

    # 1) Load all parameters from lightgbm.yaml (merges top-level + dataset block)
    params = load_parameters("training/lightgbm.yaml", dataset_override=args.dataset)
    # For convenience, store some variables
    dataset_name = params.dataset
    mw_sizes = params.multi_split_size if hasattr(params, "multi_split_size") else []
    print("==========================================================")
    print("   LightGBM Ensemble Training with Multiple Windows")
    print("==========================================================")
    print(f"Selected dataset: {dataset_name}")
    print(f"Raw features  : {params.raw_features}")
    print(f"Feature list  : {params.feature_list}")
    print(f"Labels        : {params.labels}")
    print(f"Data path     : {params.data_path}")
    print(f"Overlap       : {params.overlap}, Normalize={params.normalize}")
    print(f"Window sizes  : {mw_sizes}")
    print("==========================================================\n")

    # 2) For each window size in multi_split_size, we will create data + extract features
    all_train_samples, all_train_targets = [], []
    all_valid_samples, all_valid_targets = [], []
    all_test_samples,  all_test_targets  = [], []

    for wlen in mw_sizes:
        print(f"--- Processing window_size={wlen} ---")
        # Window the data
        train_s, train_t, valid_s, valid_t, test_s, test_t = preprocess_and_window(
            data_path=params.data_path,
            sequence_length=wlen,
            overlap=params.overlap,
            normalize=params.normalize,
            features=params.raw_features,
            labels=params.labels,
            data_groupby=params.data_groupby,  # e.g. ["cycle"] or ["filename","Date"]
        )

        # Convert raw windows into your final set of features
        # (assuming you store final features in 'feature_list')
        train_s = extract_features(train_s, params.raw_features, params.feature_list)
        valid_s = extract_features(valid_s, params.raw_features, params.feature_list)
        test_s  = extract_features(test_s, params.raw_features, params.feature_list)

        # Print shapes for clarity
        print(f"  Train split shape: {train_s.shape}, Valid split shape: {valid_s.shape}, Test split shape: {test_s.shape}")

        # Collect them
        all_train_samples.append(train_s)
        all_train_targets.append(train_t)
        all_valid_samples.append(valid_s)
        all_valid_targets.append(valid_t)
        all_test_samples.append(test_s)
        all_test_targets.append(test_t)

    # 3) Concatenate all window-based training sets into one big set
    train_samples = np.concatenate(all_train_samples, axis=0)
    train_targets = np.concatenate(all_train_targets, axis=0)
    valid_samples = np.concatenate(all_valid_samples, axis=0)
    valid_targets = np.concatenate(all_valid_targets, axis=0)

    train_samples, train_targets = clean_after_feature_extraction(train_samples, train_targets)
    valid_samples, valid_targets = clean_after_feature_extraction(valid_samples, valid_targets)
    print("\n--- Combined Training Data ---")
    print(f"train_samples shape: {train_samples.shape}")
    print(f"valid_samples shape: {valid_samples.shape}")

    # 4) Use Optuna to find best hyperparams on this combined data
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    print("\n=== Starting Optuna Hyperparam Search ===")
    study.optimize(
        lambda trial: lgbm_optimize(trial, train_samples, train_targets, valid_samples, valid_targets),
        n_trials=params.num_trials,
        n_jobs=1,
        show_progress_bar=True
    )
    best_params = study.best_params
    print("\n=== Optuna Search Complete ===")
    print("Best hyperparameters:", best_params)

    # 5) Retrain final model with best hyperparams
    print("\n--- Training Final LightGBM Model with Best Hyperparams ---")
    final_model = lgbm_model(**best_params)
    final_model.fit(
        train_samples,
        train_targets,
        eval_set=[(valid_samples, valid_targets)],
        eval_metric=["mae"],
    )

    # 6) Estimate memory usage
    mem_kb = get_memory_usage(final_model)
    print(f"Estimated final model memory usage: {mem_kb:.2f} KB")

    # 7) Evaluate on each test set
    print("\n--- Per-Window Test Performance ---")
    for idx, wlen in enumerate(mw_sizes):
        all_test_samples[idx], all_test_targets[idx] = clean_after_feature_extraction(
            all_test_samples[idx], all_test_targets[idx]
        )
        preds = final_model.predict(all_test_samples[idx])
        mae  = mean_absolute_error(all_test_targets[idx], preds)
        r2 = r2_score(all_test_targets[idx], preds)
        rmse = root_mean_squared_error(all_test_targets[idx], preds)
        print(f"  Window={wlen:2d} => Test MAE={mae:.6f}, r2={r2:.6f}, RMSE={rmse:.6f}")

    # 8) Save final model
    os.makedirs("models/lightgbm", exist_ok=True)
    model_path = f"models/lightgbm/lgbm_{dataset_name}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"\nFinal LightGBM model saved to: {model_path}")


if __name__ == "__main__":
    main()
