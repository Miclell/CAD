from scipy import stats
import numpy as np

def CritQ(sels_table, alpha):
    n, k = sels_table.shape

    U = np.sum(sels_table, axis = 0)
    V = np.sum(sels_table, axis = 1)

    Q = (k - 1) * (k * (np.sum(U**2)) - np.sum(U) ** 2) / (k * np.sum(V) - np.sum(V**2))
    Q_kr = stats.chi2(df = k-1).ppf(1 - alpha)

    return np.abs(Q) < Q_kr