from typing import List
from copy import deepcopy


def get_part_of_array(X: List[List[float]]) -> List[List[float]]:
    result = []
    for i in range(0, len(X), 4):
        result.append(X[i][120:500:5])

    return result



def sum_non_neg_diag(X: List[List[int]]) -> int:
    n = min(len(X), len(X[0]))
    diag_sum = 0
    found_non_neg = False
    
    for i in range(n):
        if X[i][i] >= 0:
            diag_sum += X[i][i]
            found_non_neg = True

    return diag_sum if found_non_neg else -1


def replace_values(X: List[List[float]]) -> List[List[float]]:
    n = len(X)
    m = len(X[0])
    result = [row[:] for row in X]
    
    for j in range(m):
        column = [X[i][j] for i in range(n)]
        M = sum(column) / n
        
        for i in range(n):
            if X[i][j] < 0.25 * M or X[i][j] > 1.5 * M:
                result[i][j] = -1

    return result
