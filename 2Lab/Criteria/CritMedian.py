from scipy import stats
import numpy as np

def CritMedian(data1, data2, alpha):
    data1 = np.array(data1)
    data2 = np.array(data2)

    combinedData = np.concatenate((data1, data2))

    totalMedian = np.median(combinedData)

    nPlus1 = np.count_nonzero(data1 > totalMedian)
    nMinus1 = np.count_nonzero(data1 <= totalMedian)
    nPlus2 = np.count_nonzero(data2 > totalMedian)
    nMinus2 = np.count_nonzero(data2 <= totalMedian)

    n1, n2 = len(data1), len(data2)

    mStatistic = (nPlus1**2 + nMinus1**2) / (n1 / 2) + (nPlus2**2 + nMinus2**2) / (n2 / 2) - (n1 + n2)
    mCritical = stats.chi2(df=1).ppf(q=1 - alpha ** 0.5)

    return mStatistic <= mCritical