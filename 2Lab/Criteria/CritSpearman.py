from scipy import stats
import numpy as np
from collections import Counter
from .Helpers import *

def CritSpearman(x: list, y: list, alpha: float):
    n = len(x)

    if n != len(y):
        raise ValueError("Разная длина объектов")

    rankX = [rank for rank, _ in sorted(enumerate(x, 1), key=lambda x: x[1])]
    rankY = [rank for rank, _ in sorted(enumerate(y, 1), key=lambda x: x[1])]

    tX = sum([n**3 - n for n in Counter(rankX).values()]) / 12
    tY = sum([n**3 - n for n in Counter(rankY).values()]) / 12

    summa = sum((rankX[i] - rankY[i])**2 for i in range(n))
    nDelta = n**3 - n
    rho = (nDelta / 6 - summa - tX - tY) / np.sqrt((nDelta / 6 - 2 * tX) * (nDelta / 6 - 2 * tY))

    uQuantile = stats.norm.ppf(1 - alpha) / np.sqrt(n - 1)
    uQuantileTwoSided = stats.norm.ppf(1 - alpha / 2) / np.sqrt(n - 1)

    if rho >= uQuantile:
        result = 'rho < 0'
    elif rho <= -uQuantile:
        result = 'rho > 0'
    elif np.abs(rho) >= uQuantileTwoSided:
        result = 'rho != 0'
    else:
        result = 'rho = 0'

    return result