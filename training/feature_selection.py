from sklearn.feature_selection import SelectKBest, f_regression
from preprocess import preprocess_and_window, load_parameters
from feature_extraction import extract_features
import numpy as np


def feature_select():
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

        train_s, fnames = extract_features(train_s, params.raw_features, ["all"], return_names=True)
        valid_s = extract_features(valid_s, params.raw_features, ["all"])
        test_s = extract_features(test_s, params.raw_features, ["all"])

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

    kb = SelectKBest(score_func=f_regression, k=10).fit(
        train_samples, train_targets
    )
    scores = {fnames[i]: abs(kb.scores_[i]) for i in range(len(fnames))}

    # Print the score for each feature, sorted by decreasing score
    for feature, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"{feature}: {score}")

    


if __name__ == "__main__":
    feature_select()
