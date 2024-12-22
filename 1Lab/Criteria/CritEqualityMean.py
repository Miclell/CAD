import numpy as np
from scipy import stats


def _Calculate(x: np.ndarray, y: np.ndarray):
    m = len(x)
    n = len(y)
    meanx = np.mean(x)
    meany = np.mean(y)
    dx = np.var(x, ddof=1)
    dy = np.var(y, ddof=1)

    return n, m, meanx, meany, dx, dy

def CritEqualityMean(x: np.ndarray, y: np.ndarray, alpha: float = 0.05):
    n, m, meanx, meany, dx, dy =  _Calculate(x, y)

    normalQuantille = stats.norm().ppf(1 - alpha / 2)
    left = meanx - meany - normalQuantille * np.sqrt(dx / m + dy / n)
    right = meanx - meany + normalQuantille * np.sqrt(dx / m + dy / n)

    return left < 0 < right

def CritEqualityMeanNorm(x: np.ndarray, y: np.ndarray, alpha: float = 0.05):
    n, m, meanx, meany, dx, dy =  _Calculate(x, y)

    tSelection = (meanx - meany) / np.sqrt(m * dx + n * dy) * np.sqrt((m * n * (m + n - 2)) / (m + n))
    tCritical = stats.t(df=(m + n)).ppf(1 - alpha / 2)

    return np.abs(tSelection) < tCritical