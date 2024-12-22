from scipy import stats as sps
import numpy as np
from math import *
from .Helpers import *

def CritKholmogorovSmirnov(x: list, y: list, alpha: float):
    n1, n2 = len(x), len(y)

    Fx = GetEmpiricalDistribution(x)
    Fy = GetEmpiricalDistribution(y)

    differentes = []    
    for i in range(min(n1, n2)):
        differentes.append(np.abs(Fx[i] - Fy[i]))
    
    D = 0
    if alpha == 0.05:
        D = 1.36 * np.sqrt(n1 * n2 / (n1 + n2))
    elif alpha == 0.1:
        D = 1.22 * np.sqrt(n1 * n2 / (n1 + n2))

    DSelect = max(differentes)
    return DSelect <= D