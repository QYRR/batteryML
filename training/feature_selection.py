from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from preprocess import preprocess_and_window, load_parameters
from feature_extraction import extract_features
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
import lightgbm as lgbm
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor


def clean_after_feature_extraction(samples, targets, names=None):
    # Remove rows with NaN values
    temp_train = pd.DataFrame(samples, columns=names)
    temp_train["target"] = targets
    temp_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    temp_train.dropna(inplace=True)
    samples = temp_train.values[:, :-1]
    targets = temp_train.values[:, -1]
    print(targets)
    return samples, targets


def feature_select():
    params = load_parameters("training/lightgbm.yaml", dataset_override='st')
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
            params.data_groupby
        )

        train_s, fnames = extract_features(train_s, params.raw_features, ["all"], return_names=True)
        valid_s = extract_features(valid_s, params.raw_features, ["all"])
        test_s = extract_features(test_s, params.raw_features, ["all"])

        # scaler = MinMaxScaler()
        # train_s = scaler.fit_transform(train_s)
        # valid_s = scaler.transform(valid_s)
        # test_s = scaler.transform(test_s)

        train_samples.append(train_s)
        train_targets.append(train_t)
        valid_samples.append(valid_s)
        valid_targets.append(valid_t)
        test_samples.append(test_s)
        test_targets.append(test_t)

    train_samples = np.concatenate(train_samples, axis=0)
    valid_samples = np.concatenate(valid_samples, axis=0)
    train_targets = np.concatenate(train_targets, axis=0).ravel()
    valid_targets = np.concatenate(valid_targets, axis=0).ravel()
    
    train_samples, train_targets = clean_after_feature_extraction(train_samples, train_targets)
    valid_samples, valid_targets = clean_after_feature_extraction(valid_samples, valid_targets)
    # kb = SelectKBest(score_func=mutual_info_regression, k=10).fit(
    #     train_samples, train_targets
    # )
    # scores = {fnames[i]: abs(kb.scores_[i]) for i in range(len(fnames))}

    # # Print the score for each feature, sorted by decreasing score
    # for feature, score in sorted(scores.items(), key=lambda x: -x[1]):
    #     print(f"{feature}: {score}")
    '''lgbm_params = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "verbose": -1,
        "seed": 42,
        "n_jobs": 4,
        "deterministic": True,
        "force_col_wise": True,
    }
    pipe = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("rfe", RFE(estimator=lgbm.LGBMRegressor(**lgbm_params), n_features_to_select=8, step=1)),
        ]
    )

    pipe.fit(train_samples, train_targets.ravel())
<<<<<<< HEAD
    pipe.fit(train_samples, train_targets.ravel())
=======
>>>>>>> 131ede9 (Refactor training scripts and enhance feature extraction)
    selected_mask = pipe.named_steps["rfe"].support_
    selected_features = [name for name, selected in zip(fnames, selected_mask) if selected]'''
    # Convert train_samples and train_targets to DataFrame for compatibility
    x_all = pd.DataFrame(train_samples, columns=fnames)
    y_all = pd.Series(train_targets)

    # Train a baseline model on the training set
    baseline_model = lgbm.LGBMRegressor(force_col_wise=True, verbosity=-1)

    # Reduce the float to 32 for all the numerical float features
    float_cols = x_all.select_dtypes(include=['float64']).columns
    #x_all[float_cols] = x_all[float_cols].astype('float32')
    # Replace infinity values with NaN, then drop them
    
    x_all.dropna(inplace=True)
    baseline_model.fit(x_all, y_all)

    # Compute permutation importance on training set using a simple CV approach
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    importances = []

    for train_idx, val_idx in cv.split(x_all):
        X_tr, X_vr = x_all.iloc[train_idx], x_all.iloc[val_idx]
        y_tr, y_vr = y_all.iloc[train_idx], y_all.iloc[val_idx]
        
        model = lgbm.LGBMRegressor(force_col_wise=True, verbosity=-1)
        model.fit(X_tr, y_tr)
        result = permutation_importance(model, X_vr, y_vr, random_state=0)
        importances.append(result.importances_mean)
        
    # Average the importances across folds
    avg_importances = np.mean(importances, axis=0)

    # Map importance scores to feature names
    feature_importances = pd.Series(avg_importances, index=fnames)
    feature_importances = feature_importances.sort_values(ascending=False)

    # Select the top 8 features from permutation importance
    top_perm_features = feature_importances.head(10)
    print("Top features from permutation importance:\n", top_perm_features)

    # Create the estimator for RFE
    #estimator = RandomForestRegressor(n_jobs=4, random_state=0)
    # Replace RandomForestRegressor with ExtraTreesRegressor for a lighter model

    # Create the estimator for RFE
    estimator = ExtraTreesRegressor(n_jobs=4, random_state=0)
    # Set RFE to select the top 8 features
    rfe_selector = RFE(estimator, n_features_to_select=10, step=1)
    rfe_selector = rfe_selector.fit(x_all, y_all)

    # Get the feature mask and the list of features selected by RFE
    top_rfe_features = x_all.columns[rfe_selector.support_].tolist()
    print()
    print("Top features from RFE:", top_rfe_features)

    print()
    # Take the intersection (agreeable features that appear in both methods)
    agreeable_features = list(set(top_perm_features.index) & set(top_rfe_features))
    print("Agreeable features (intersection):", agreeable_features)

    # Fallback to union if intersection is too small
    if len(agreeable_features) < 2:
        agreeable_features = list(set(top_perm_features.index).union(top_rfe_features))
        print("Using union as fallback. Agreeable features:", agreeable_features)
    print('selected features:')
    for f in agreeable_features:
        print(f"- {f}")
    


if __name__ == "__main__":
    feature_select()
