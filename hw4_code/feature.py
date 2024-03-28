import numpy as np


def create_nl_feature(X):
    """
    Create additional features and add it to the dataset.

    Returns:
        X_new - (N, d + num_new_features) array with
                additional features added to X such that it
                can classify the points in the dataset.
    """
    # TODO

    N, d = X.shape
    degree = 13

    features = []

    for i in range(1, degree + 1):
        for j in range(i + 1):
            if i - j == 0 and j == 0:
                continue  # skips the bias term, include_bias=False
            features.append((X[:, 0] ** (i - j)) * (X[:, 1] ** j))

    X_new = np.column_stack(features)

    return X_new
