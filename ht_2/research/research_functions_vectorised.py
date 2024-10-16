import numpy as np


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    x = np.sort(x)
    y = np.sort(y)
    return np.array_equal(x, y)


def max_prod_mod_3(x: np.ndarray) -> int:
    if x.size < 2:
        return -1

    a = x[:-1]
    b = x[1:]

    products = a * b
    condition = (a % 3 == 0) | (b % 3 == 0)
    valid_products = products[condition]
    
    if valid_products.size == 0:
        return -1

    return int(np.max(valid_products))


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    image = np.array(image)
    weights = np.array(weights)

    result = np.tensordot(image, weights, axes=([2], [0]))

    return result



def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    x_array = np.array(x)
    y_array = np.array(y)

    vec_x = np.repeat(x_array[:, 0], x_array[:, 1])
    vec_y = np.repeat(y_array[:, 0], y_array[:, 1])

    if vec_x.size != vec_y.size:
        return -1

    scalar_product = np.dot(vec_x, vec_y)

    return int(scalar_product)



def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    norm_X = np.linalg.norm(X, axis=1)
    norm_Y = np.linalg.norm(Y, axis=1)

    norm_X_matrix = norm_X[:, np.newaxis]
    norm_Y_matrix = norm_Y[np.newaxis, :]

    dot_product = np.dot(X, Y.T)
    cosine_similarity = dot_product / (norm_X_matrix * norm_Y_matrix)

    valid = (norm_X_matrix != 0) & (norm_Y_matrix != 0)
    cosine_similarity = np.where(valid, cosine_similarity, 1)

    cosine_similarity = np.nan_to_num(cosine_similarity, nan=1.0)

    return cosine_similarity
