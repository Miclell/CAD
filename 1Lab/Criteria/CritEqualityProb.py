import numpy as np
from scipy import stats
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning)

# Критерий равенства вероятностей (проверяется гипотеза о равенстве вероятностей в двух выборках по схеме Бернулли)
def CritEqualityProb(x: np.ndarray, y: np.ndarray, alpha: float = 0.05):
    m, n = len(x), len(y)
    p_x = np.mean(x) / m
    p_y = np.mean(y) / n
    
    normalQuantille = stats.norm().ppf(1 - alpha / 2)
    eps = normalQuantille * np.sqrt(p_x * (1 - p_x) / m + p_y * (1 - p_y) / n)

    left = p_x - p_y - eps
    right = p_x - p_y + eps
    return left < 0 < right