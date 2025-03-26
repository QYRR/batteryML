import pickle
import lightgbm as lgb
import numpy as np
from ensemble import load_params_and_data, estimate_memory_usage
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def run_inference_per_tree(model, X):
    """
    Loads a LightGBM model and runs inference on each tree of the ensemble.

    Args:
        model_path (str): Path to the LightGBM model file (pickle format).
        input_data (np.ndarray): Input data for inference.

    Returns:
        np.array: A matrix (n_trees, predictions)
    """
    # Get the number of trees in the model
    num_trees = model.num_trees()

    # Run inference for each tree
    tree_predictions = []
    for tree_idx in range(num_trees):
        pred = model.predict(X, start_iteration=tree_idx, num_iteration=1)
        tree_predictions.append(pred)

    tree_predictions = np.array(tree_predictions)
    return tree_predictions


def prune_trees(model_path):
    (
        params,
        train_samples,
        train_targets,
        valid_samples,
        valid_targets,
        test_samples,
        test_targets,
    ) = load_params_and_data()

    # Load the LightGBM model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Ensure the model is a LightGBM Booster
    if not isinstance(model.booster_, lgb.Booster):
        raise ValueError("The loaded model is not a LightGBM Booster.")

    # Print some info about the model
    print("Pruning model with {} trees".format(model.booster_.num_trees()))

    # Predict by tree on the test set
    tree_predictions = run_inference_per_tree(
        model.booster_, test_samples[params.multi_split_size.index(20)]
    )
    nodes_per_tree = []

    # Get the learning rate
    lr = model.learning_rate

    # Calculate the mae for an ensemble with less and less trees
    # Using 20 as the sequence length
    scores = np.zeros(len(tree_predictions))
    memory = np.zeros(len(tree_predictions))
    for i in range(len(tree_predictions)):
        ensemble_pred = np.sum(tree_predictions[: i + 1], axis=0)
        memory[i] = estimate_memory_usage(
            model, limit_to_trees=[*range(i + 1)], nodes_per_tree=nodes_per_tree
        )
        scores[i] = mean_absolute_error(
            test_targets[params.multi_split_size.index(20)], ensemble_pred
        )
        print(f"MAE with {i + 1} trees: {scores[i]} - Memory: {memory[i]} kB")

    # Add the full ensemble
    ensemble_pred = model.predict(test_samples[params.multi_split_size.index(20)])
    ensemble_score = mean_absolute_error(
        test_targets[params.multi_split_size.index(20)], ensemble_pred
    )
    print(f"Full ensemble MAE: {ensemble_score}")

    # Plot the scores vs memory
    fig, ax1 = plt.subplots()

    # Primary x-axis (memory)
    ax1.plot(memory, scores, "bo-", label="MAE vs Memory", markersize=3)
    ax1.set_xlabel("Memory Usage")
    ax1.set_ylabel("Mean Absolute Error (MAE)")
    ax1.hlines(
        ensemble_score,
        0,
        memory[-1],
        colors="r",
        linestyles="--",
        label="Full Ensemble MAE",
    )
    ax1.legend(loc="upper right")

    plt.savefig("pruning.png")


if __name__ == "__main__":
    prune_trees("models/lightgbm_model.pkl")
