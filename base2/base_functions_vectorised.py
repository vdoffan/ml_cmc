import numpy as np


def get_part_of_array(X: np.ndarray) -> np.ndarray:

    return X[::4, 120:500:5]



def sum_non_neg_diag(X: np.ndarray) -> int:
    diag_elements = np.diag(X)
    non_neg_elements = diag_elements[diag_elements >= 0]
    
    if len(non_neg_elements) == 0:
        return -1
    else:
        return np.sum(non_neg_elements)


def replace_values(X: np.ndarray) -> np.ndarray:
    result = X.copy()
    M = np.mean(X, axis=0)

    mask = (X < 0.25 * M) | (X > 1.5 * M)
    result[mask] = -1

    return result

 