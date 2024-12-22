import numpy as np

def CritExcess(x: np.ndarray):
    n = len(x)
    mean = np.mean(x)

    central4 = sum((x[i] - mean)** 4 for i in range(n)) / n
    std = np.std(x)
    
    excess = central4 / std**4 - 3
    excessCritical = (24 * n * (n - 2) * (n - 3)) / ((n + 1) * (n + 3) * (n + 5))

    return excess < excessCritical