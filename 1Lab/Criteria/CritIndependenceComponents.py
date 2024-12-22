from scipy import stats

def CritIndependenceComponents(freqMatrix, n: int, m: int, alpha: float = 0.05):
    freqX = [sum(freqMatrix[i]) for i in range(n)]
    freqY = [sum(freqMatrix[i][j] for i in range(n)) for j in range(m)]

    chi_select = sum((freqMatrix[i][j]**2) / (freqX[i] * freqY[j]) for i in range(n) for j in range(m)) - 1

    N = freqMatrix.sum()
    chi_select *= N
    
    chi_quantile = stats.chi2(df=n + m).ppf(1 - alpha)

    return chi_select <= chi_quantile