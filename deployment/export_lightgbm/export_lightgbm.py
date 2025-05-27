"""
under main folder, use 
"python deployment/export_lightgbm/export_lightgbm.py --dataset 'st'" or
"python deployment/export_lightgbm/export_lightgbm.py --dataset 'randomized'"
to run this script
"""


from eden.frontend.lightgbm import parse_boosting_trees
from eden.model import Ensemble
from eden.backend.deployment import deploy_model
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import pickle
import argparse



def generate_random_data(len_features: int, labels: list, num_samples: int = 20, seed: int = 42):
    # generate random data for export model
    np.random.seed(seed)

    X = pd.DataFrame(np.random.rand(num_samples, len_features), columns=[f"feature_{i}" for i in range(len_features)])
    y = pd.DataFrame(np.random.rand(num_samples, len(labels)), columns=labels)
    
    # Note the .values here, as we support only numpy arrays
    return X.values, y.values


def main():

    # load the main function parameter
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Which dataset to train on (e.g. 'st' or 'randomized'). Overrides 'dataset:' in YAML."
    )
    args = parser.parse_args()
    dataset = args.dataset

    # define the parameters for model export
    model_path = f"result/models/lightgbm/lgbm_{dataset}_model.pkl"
    data_path = f"data/{dataset}/test.csv"  
    export_path = f"deployment/export_lightgbm/{dataset}"
    if dataset == "st":
        len_features = 10
        labels = ['capacity']
    if dataset == "randomized":
        len_features = 10
        labels = ['SOH']
    
    X, y = generate_random_data(len_features, labels)
    model = pickle.load(open(model_path, "rb"))
    golden = model.predict(X)
    emodel: Ensemble = parse_boosting_trees(model=model)
    # if 'remove_list' in globals():    
    #     emodel.remove_trees(idx =remove_list)
    # Note the .values here, as we support only numpy arrays
    # eden_preds = emodel.predict(X)
    # print("Sklearn-Eden prediction error", mean_absolute_error(golden, eden_preds))
    # print("Sklearn MAE", mean_absolute_error(golden, y[:,0]))
    # print("Eden MAE", mean_absolute_error(eden_preds, y[:,0]))

    # Deployment step
    subsampled_X = X[:10]
    deploy_model(
        ensemble=emodel,
        target="default",
        output_path=export_path,
        input_data=subsampled_X,
        data_structure="arrays",
    )

    memory_cost = emodel.get_memory_cost()
    print(f'memory cost: {memory_cost}')
    total_memory_cost = sum(memory_cost.values())
    print(f'total memory: {total_memory_cost} B ({total_memory_cost/1024:.2f} KiB)')



if __name__ == "__main__":
    main()