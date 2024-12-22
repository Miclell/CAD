import numpy as np

def CritSymmetry(x: np.ndarray):
    n = len(x)
    mean = np.mean(x)

    central3 = sum((x[i] - mean)**3 for i in range(n)) / n
    std = np.std(x)

    asymmetry = central3 / (std**3)
    asymmetryCritical = 6 * (n - 2) / ((n + 1) * (n + 3))

    return asymmetry < asymmetryCritical