from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from preprocess import preprocess_and_window, load_parameters
from feature_extraction import extract_features
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
import lightgbm as lgbm
from sklearn.pipeline import Pipeline


def feature_select():
    params = load_parameters("training/lightgbm.yaml", dataset_override='randomized')
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
    train_targets = np.concatenate(train_targets, axis=0)
    valid_targets = np.concatenate(valid_targets, axis=0)

    # kb = SelectKBest(score_func=mutual_info_regression, k=10).fit(
    #     train_samples, train_targets
    # )
    # scores = {fnames[i]: abs(kb.scores_[i]) for i in range(len(fnames))}

    # # Print the score for each feature, sorted by decreasing score
    # for feature, score in sorted(scores.items(), key=lambda x: -x[1]):
    #     print(f"{feature}: {score}")

    pipe = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("rfe", RFE(estimator=lgbm.LGBMRegressor(), n_features_to_select=5, step=1)),
        ]
    )

    pipe.fit(train_samples, train_targets)
    selected_mask = pipe.named_steps["rfe"].support_
    selected_features = [name for name, selected in zip(fnames, selected_mask) if selected]

    print('selected features:')
    for f in selected_features:
        print(f"- {f}")
    


if __name__ == "__main__":
    feature_select()
