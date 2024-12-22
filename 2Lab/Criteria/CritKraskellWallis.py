from scipy import stats
from .Helpers import GetRanks

def CritKraskellWallis(x1: list, x2: list, x3: list, x4: list, alpha: float):
    n1, n2, n3, n4 = len(x1), len(x2), len(x3), len(x4)
    n = n1 + n2 + n3 + n4

    R1, R2, R3, R4 = sum(GetRanks(x1)), sum(GetRanks(x2)), sum(GetRanks(x3)), sum(GetRanks(x4))

    HSelect = -3 * (n + 1) + 12 / (n * (n + 1)) * ((R1**2) / n1 + (R2**2) / n2 + (R3**2) / n3 + (R4**2) / n4)

    k = 4
    return HSelect <= stats.chi2(df=k - 1).ppf(1 - alpha)