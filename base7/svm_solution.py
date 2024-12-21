from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_svm_and_predict(train_features, train_target, test_features):
    """
    train_features: np.array, (num_elements_train x num_features) - train data description, the same features and the same order as in train data
    train_target: np.array, (num_elements_train) - train data target
    test_features: np.array, (num_elements_test x num_features) -- some test data, features are in the same order as train features

    return: np.array, (num_elements_test) - test data predicted target, 1d array
    """

    pipeline = Pipeline([("scaler", StandardScaler()), ("svm", SVC())])

    params = {
        "svm__kernel": ["linear", "rbf", "poly"],
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto"],
        "svm__degree": [1, 2, 8],
        "svm__class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)

    grid_search = GridSearchCV(
        pipeline, param_grid=params, cv=cv, scoring="accuracy", verbose=3, n_jobs=-1
    )

    grid_search.fit(train_features, train_target)

    bm = grid_search.best_estimator_
    bm.fit(train_features, train_target)

    predictions = bm.predict(test_features)
    print(grid_search.best_params_)

    return predictions
