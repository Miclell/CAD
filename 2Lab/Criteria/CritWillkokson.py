from scipy import stats
import numpy as np
from math import *
from .Helpers import GetRanks

def CritWillkokson(x: list, y: list, alpha: float):
    n = len(x)
    if n != len(y):
        raise ValueError("Разная длина объектов")

    differences = [xi - yi for xi, yi in zip(x, y)]
    differences = [d for d in differences if d != 0]

    ranks = GetRanks(differences)
    rPlus = sum(rank for rank, diff in zip(ranks, differences) if diff > 0)
    rMinus = sum(rank for rank, diff in zip(ranks, differences) if diff < 0)

    tStat = min(rPlus, rMinus)
    zSelect = np.abs(tStat - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)

    return zSelect < stats.norm.ppf(1 - alpha / 2)