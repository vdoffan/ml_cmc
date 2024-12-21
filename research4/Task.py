import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        self.categories = {}
        self.encoded_feature_names = []

        for col in X.columns:
            unique_vals = X[col].unique()
            self.categories[col] = sorted(unique_vals)
            self.encoded_feature_names.extend(
                [f"{col}_{str(val)}" for val in self.categories[col]]
            )

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        encoded_features = []

        for col in X.columns:
            col_encoded = np.zeros(
                (X.shape[0], len(self.categories[col])), dtype=self.dtype
            )
            category_to_index = {
                category: idx for idx, category in enumerate(self.categories[col])
            }
            for i, val in enumerate(X[col]):
                if val in category_to_index:
                    col_encoded[i, category_to_index[val]] = 1
                else:
                    pass
            encoded_features.append(col_encoded)

        res = np.hstack(encoded_features).astype(self.dtype)
        return res

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        self.encoders = []
        X_np = X.values
        Y_np = Y.values
        n_features = X_np.shape[1]
        n_samples = X_np.shape[0]

        for feature_idx in range(n_features):
            feature_dict = {}
            unique_vals = np.unique(X_np[:, feature_idx])

            for val in unique_vals:
                mask = X_np[:, feature_idx] == val
                count = np.sum(mask)
                success_total = np.sum(Y_np[mask])
                success_rate = success_total / count if count > 0 else 0
                proportion = count / n_samples
                feature_dict[val] = np.array(
                    [success_rate, proportion, 0], dtype=self.dtype
                )

            self.encoders.append(feature_dict)

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        X_np = X.values
        n_features = X_np.shape[1]
        n_samples = X_np.shape[0]
        transformed = []

        for feature_idx in range(n_features):
            encoded_feature = np.empty((n_samples, 3), dtype=self.dtype)
            encoder = self.encoders[feature_idx]

            for sample_idx in range(n_samples):
                val = X_np[sample_idx, feature_idx]
                if val in encoder:
                    success, proportion, _ = encoder[val]
                else:
                    success, proportion = 0.0, 0.0

                adjusted_success = (success + a) / (proportion + b)
                encoded_feature[sample_idx] = [success, proportion, adjusted_success]

            transformed.append(encoded_feature)

        return np.hstack(transformed)

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack(
            (idx[: i * n_], idx[(i + 1) * n_:])
        )
    yield idx[(n_splits - 1) * n_:], idx[: (n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        self.fold_encoders = []
        for val_indices, train_indices in group_k_fold(X.shape[0], self.n_folds, seed):
            encoder = SimpleCounterEncoder(dtype=self.dtype)
            encoder.fit(X.iloc[train_indices], Y.iloc[train_indices])
            self.fold_encoders.append((val_indices, encoder))

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        transformed_matrix = None
        for val_indices, encoder in self.fold_encoders:
            transformed_fold = encoder.transform(X.iloc[val_indices], a, b)
            fold_data = np.hstack((transformed_fold, val_indices.reshape(-1, 1)))
            if transformed_matrix is None:
                transformed_matrix = fold_data
            else:
                transformed_matrix = np.vstack((transformed_matrix, fold_data))
        sorted_indices = transformed_matrix[:, -1].astype(int).argsort()
        final_transformed = transformed_matrix[sorted_indices, :-1]
        return final_transformed.astype(self.dtype)

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    unique_categories, inverse_indices = np.unique(x, return_inverse=True)
    sums = np.bincount(inverse_indices, weights=y)
    counts = np.bincount(inverse_indices)
    weights_array = np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)
    return weights_array
