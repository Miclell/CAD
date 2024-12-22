from scipy import stats
import numpy as np
from math import *
from .Helpers import *

def CritSign(x: list, y: list, alpha):
    n = len(x)

    if n != len(y):
        raise ValueError("Разная длина объектов")

    delta = [xi - yi for xi, yi in zip(x, y)]
    delta = [d for d in delta if d != 0]
    m = len(delta)

    mu = sum(1 for d in delta if d > 0)

    if m < 50:
        wStat = 1 / (2**m) * sum(comb(m, i) for i in range(mu))
    else:
        wStat = stats.norm.cdf((2 * mu - m) / np.sqrt(m))

    if wStat > 1 - alpha:
        return "p > 0.5"
    elif wStat < alpha:
        return "p < 0.5"
    elif wStat < alpha / 2 or wStat > 1 - alpha / 2:
        return "p != 0.5"
    else:
        return "p = 0.5"