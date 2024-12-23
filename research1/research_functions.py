from collections import Counter
from typing import List


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    x.sort()
    y.sort()
    return True if x == y else False


def max_prod_mod_3(x: List[int]) -> int:
    max_pr = -1
    for a, b in zip(x, x[1:]):
        if a * b % 3 == 0:
            max_pr = max(max_pr, a*b)
    return max_pr


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    height = len(image)
    width = len(image[0])
    num_channels = len(image[0][0])

    result = [[0 for _ in range(width)] for _ in range(height)]

    for h in range(height):
        for w in range(width):
            pixel_sum = 0
            for c in range(num_channels):
                pixel_sum += image[h][w][c] * weights[c]
            result[h][w] = pixel_sum

    return result

def decode(rle):
        decoded = []
        for pair in rle:
            element, count = pair
            for _ in range(count):
                decoded.append(element)
        return decoded

def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    decoded_x = decode(x)
    decoded_y = decode(y)

    if len(decoded_x) != len(decoded_y):
        return -1

    scalar_product = 0
    for a, b in zip(decoded_x, decoded_y):
        scalar_product += a * b

    return scalar_product

def norm(v: List[float]) -> float:
    summ = 0
    for x in v:
        summ += x*x
    return summ ** 0.5

def dot(v1: List[float], v2: List[float]) -> float:
    summ = 0
    for a, b in zip(v1, v2):
        summ += a*b
    return summ

def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    norm_X = [norm(x) for x in X]
    norm_Y = [norm(y) for y in Y]

    res = []
    for i, x in enumerate(X):
        row = []
        for j, y in enumerate(Y):
            if norm_X[i] == 0 or norm_Y[j] == 0:
                sim = 1.0
            else:
                sim = dot(x, y) / (norm_X[i] * norm_Y[j])
            row.append(sim)
        res.append(row)
    return res