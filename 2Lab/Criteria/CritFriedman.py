from scipy import stats
from .Helpers import GetRanks

def CritFriedman(dataset: list, alpha: float):
    k, n = len(dataset), len(dataset[0])

    if not all(len(sample) == n for sample in dataset):
        raise ValueError("Разная длина объектов")
    
    R = [GetRanks(dataset[i]) for i in range(k)]

    sumRanks = 0
    for j in range(n):
        column_sum = sum(R[i][j] for i in range(k))
        sumRanks += column_sum**2

    fSelect = (12 / (k * n * (n + 1)) * sumRanks) - (3 * k * (n + 1))

    return fSelect < stats.chi2(df=n - 1).ppf(1 - alpha)