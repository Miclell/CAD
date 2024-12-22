import numpy as np
import pandas as pd
import os


os.chdir(os.path.dirname(os.path.abspath(__file__)))
dataAi = pd.read_csv('../Data/swilk-coeff-ai.txt', sep='\t').astype('float') # таблица точек
dataCritical = pd.read_csv('../Data/swilk-critical.txt', sep='\t').set_index('n\p').astype('float') # критические

# Критерий Шапиро-Уилка (проверяется гипотеза о том, что генеральная совокупность иммеет нормальный характер)
def CritShapiroWilk(x: np.ndarray, alpha: float = 0.05):
    n = len(x)
    k = int(n / 2)
    mean = x.mean()
    x = sorted(x)
    a = dataAi[str(n)].to_numpy()

    wV = sum([a[i] * (x[n - i - 1] - x[i]) for i in range(k)])**2 / sum([(x[i] - mean)**2 for i in range(n)])

    wCritical = dataCritical[format(alpha, ".2f")][n]

    return wV >= wCritical