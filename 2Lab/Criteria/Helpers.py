from scipy import stats
import numpy as np
from math import *

def GetRanks(data):
    sortedIndices = np.argsort(data)
    ranks = np.empty_like(sortedIndices, dtype=float)

    ranks[sortedIndices] = np.arange(1, len(data) + 1)

    for i in range(1, len(data)):
        if data[sortedIndices[i]] == data[sortedIndices[i - 1]]:
            ranks[sortedIndices[i]] = ranks[sortedIndices[i - 1]]

    uniqueValues, inverseIndices = np.unique(data, return_inverse=True)
    counts = np.bincount(inverseIndices)
    ranks = np.array([np.mean(ranks[inverseIndices == i]) for i in range(len(uniqueValues))])[inverseIndices]

    return ranks


def CountInversions(array):
    count = 0
    for i in range(len(array)):
        count += np.sum(array[i] > array[i+1:])

    return count


def СalculateT(ranks_x, ranks_y):
    unique_x, counts_x = np.unique(ranks_x, return_counts=True)
    unique_y, counts_y = np.unique(ranks_y, return_counts=True)
    T1 = 0.5 * np.sum(counts_x ** 2 - counts_x)
    T2 = 0.5 * np.sum(counts_y ** 2 - counts_y)
    return T1, T2


def GetEmpiricalDistribution(x: list) -> list:
    n = len(x)
    F_x = []
    for i in range(n):
        func = 0
        for j in range(n):
            if x[j] < x[i]:
                func += 1
        func /= n
        F_x.append(func)
    return F_x