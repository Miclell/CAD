from scipy import stats
import numpy as np
from .Helpers import *

def CritKendall(x: list, y: list, alpha: float):
    n = len(x)

    if n != len(y):
        raise ValueError("Разная длина объектов")
    
    xRanks = GetRanks(x)
    yRanks = GetRanks(y)
    m1 = len(np.unique(xRanks))
    m2 = len(np.unique(yRanks))
    k = CountInversions(yRanks)

    if m1 == m2 == n:
        tauK = 1 - (4 * k) / (n**2 - n)
    else:
        T1, T2 = СalculateT(xRanks, yRanks)
        nIt = n
        tauK = (1 - (4 * k + 2 * (T1 + T2)) / ((nIt)**2 - nIt)) * (
                np.sqrt(1 - 2 * T1 / (n**2 - nIt)) * np.sqrt(1 - 2 * T2 / (n**2 - nIt)))**(-1)
        
    zAlpha = stats.norm.ppf(1 - alpha / 2)
    tauAlpha = zAlpha * np.sqrt(2 * (2 * n + 5) / (9 * n * (n + 1)))

    return abs(tauK) <= tauAlpha