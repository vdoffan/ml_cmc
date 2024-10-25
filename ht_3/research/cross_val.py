import numpy as np
import typing
from collections import defaultdict


def kfold_split(
    num_objects: int, num_folds: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds
       (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects: number of objects in train set
    num_folds: number of folds for cross-validation split

    Returns:
    list of length num_folds, where i-th element of list
    contains tuple of 2 numpy arrays, he 1st numpy array
    contains all indexes without i-th fold while the 2nd
    one contains i-th fold
    """
    sec_arr_len = int(num_objects / num_folds)

    all_ids = np.array([i for i in range(num_objects)])

    cur = 0
    res = []
    for i in range(num_folds - 1):
        first_part = all_ids[all_ids < cur]
        first_part = np.concatenate(
            [first_part, all_ids[all_ids > cur + sec_arr_len - 1]]
        )

        tmp = all_ids[all_ids >= cur]
        second_part = tmp[tmp <= cur + sec_arr_len - 1]
        res.append((first_part, second_part))
        cur += sec_arr_len

    last_first = all_ids[all_ids < cur]
    last_second = all_ids[all_ids >= cur]
    res.append((last_first, last_second))

    return res


def knn_cv_score(
    X: np.ndarray,
    y: np.ndarray,
    parameters: dict[str, list],
    score_function: callable,
    folds: list[tuple[np.ndarray, np.ndarray]],
    knn_class: object,
) -> dict[str, float]:
    """Takes train data, counts cross-validation score over
    grid of parameters (all possible parameters combinations)

    Parameters:
    X: train set
    y: train labels
    parameters: dict with keys from
        {n_neighbors, metrics, weights, normalizers}, values of type list,
        parameters['normalizers'] contains tuples (normalizer, normalizer_name)
        see parameters example in your jupyter notebook

    score_function: function with input (y_true, y_predict)
        which outputs score metric
    folds: output of kfold_split
    knn_class: class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight),
    value - mean score over all folds
    """
    result = {}

    n_neighbors_list = parameters.get("n_neighbors", [])
    metrics_list = parameters.get("metrics", [])
    weights_list = parameters.get("weights", [])
    normalizers = parameters.get("normalizers", [])

    for normalizer, normalizer_name in normalizers:
        for n_neighbors in n_neighbors_list:
            for metric in metrics_list:
                for weights in weights_list:

                    scores = []

                    for train_id, test_id in folds:
                        if normalizer is not None:
                            normalizer.fit(X[train_id])
                            x_train = normalizer.transform(X[train_id])
                            x_test = normalizer.transform(X[test_id])
                        else:
                            x_train = X[train_id]
                            x_test = X[test_id]

                        y_train = y[train_id]
                        y_test = y[test_id]

                        model = knn_class(
                            n_neighbors=n_neighbors, metric=metric, weights=weights
                        )
                        model.fit(x_train, y_train)
                        y_pred = model.predict(x_test)

                        score = score_function(y_test, y_pred)

                        scores.append(score)

                    key = (normalizer_name, n_neighbors, metric, weights)
                    result[key] = np.mean(scores)

    return result
