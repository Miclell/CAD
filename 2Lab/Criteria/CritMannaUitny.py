from scipy import stats
import numpy as np
from math import *
from .Helpers import *

def CritMannaUitny(x: list, y: list, alpha: float):
    n1, n2 = len(x), len(y)

    R1, R2 = sum(GetRanks(x)), sum(GetRanks(y))

    w1 = n1 * n2 + 0.5 * n1 * (n1 + 1) - R1
    w2 = n1 * n2 + 0.5 * n2 * (n2 + 1) - R2

    WSelect = min(w1, w2)
    ZSelect = (WSelect - (1/2) * n1 * n2) / np.sqrt((1 / 12) * n1 * n2 * (n1 + n2 + 1))

    return np.abs(ZSelect) < stats.norm.ppf(1 - alpha / 2)