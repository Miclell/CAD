import numpy as np
from scipy import stats


def CritEqualityVar(x: np.ndarray, y: np.ndarray, alpha: float = 0.05):
    n, m = len(x), len(y)
    dx = np.var(x, ddof=1)
    dy = np.var(y, ddof=1)

    if dx > dy:
        fSelect = dx / dy
    else:
        fSelect = dy / dx

    fQuantile = stats.f.ppf(1 - alpha, n - 1, m - 1)
    return fSelect < fQuantile
