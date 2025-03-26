import pickle
import lightgbm as lgb
import numpy as np
from ensemble import load_params_and_data


def run_inference_per_tree(model_path, X):
    """
    Loads a LightGBM model and runs inference on each tree of the ensemble.

    Args:
        model_path (str): Path to the LightGBM model file (pickle format).
        input_data (np.ndarray): Input data for inference.

    Returns:
        list: A list of predictions from each tree.
    """
    # Load the LightGBM model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Ensure the model is a LightGBM Booster
    if not isinstance(model, lgb.Booster):
        raise ValueError("The loaded model is not a LightGBM Booster.")

    # Get the number of trees in the model
    num_trees = model.num_trees()

    # Run inference for each tree
    tree_predictions = []
    for tree_idx in range(num_trees):
        pred = model.predict(X, start_iteration=tree_idx, num_iteration=1)
        tree_predictions.append(pred)

    return tree_predictions


def prune_trees():
    (
        params,
        train_samples,
        train_targets,
        valid_samples,
        valid_targets,
        test_samples,
        test_targets,
    ) = load_params_and_data()

    # Predict by tree on the validation set
    tree_predictions = run_inference_per_tree(params.model_path, valid_samples)

    # Calculate the mae for an ensemble with less and less trees
    scores = np.zeros(len(tree_predictions))
    for i in range(len(tree_predictions)):
        ensemble_pred = np.mean(tree_predictions[: i + 1], axis=0)
        scores[i] = np.mean(np.abs(ensemble_pred - valid_targets))
    
    # Plot the scores vs memory


if __name__ == "__main__":
    prune_trees()
