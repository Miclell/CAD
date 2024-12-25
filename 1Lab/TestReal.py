from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import os

from Criteria.CritEqualityProb import CritEqualityProb
from Criteria.CritShapiroWilk import CritShapiroWilk
from Criteria.CritEqualityVar import CritEqualityVar
from Criteria.CritEqualityMean import CritEqualityMean
from Criteria.CritExcess import CritExcess
from Criteria.CritSymmetry import CritSymmetry
from Criteria.CritIndependenceComponents import CritIndependenceComponents

def HistPlot(X):
    k = 1.72 * (len(X) ** (1 / 3))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Количество забитых мячей')
    ax.set_xlabel('Значения выборки')
    ax.hist(X, color='purple', bins=int(k), edgecolor='black', density=1)
    plt.tight_layout()
    kde = stats.gaussian_kde(X)
    x = np.linspace(min(X), max(X), 1000)
    #ax.plot(x, kde(x))
    plt.show()
    return 0

os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('Data/nbagoals.csv')

x = df['Забитые']
y = df['Незабитые']

HistPlot(x)

if CritShapiroWilk(x):
    print('Распределение генеральной совокупности нормальное')
else:
    print('Распределение генеральной совокупности не нормальное')


dfVR = pd.read_csv('Data/vaccine_russia.csv')
dfVU = pd.read_csv('Data/vaccine_usa.csv')
if CritEqualityProb(dfVR, dfVU):
    print('Вероятности успехов совпадают')
else:
    print('Вероятности успехов не совпадают')

if CritEqualityVar(x, y):
    print('Дисперсии распределений равны')
else:
    print('Дисперсии распределений не равны')

if CritEqualityMean(x, y):
    print('Мат ожидания распределений равны')
else:
    print('Мат ожидания распределений не равны')

if CritSymmetry(x):
    print('Коэффициент асимметрии равен 0')
else:
    print('Коэффициент асимметрии не равен 0')

if CritExcess(x):
    print('Коэффициент эксцесса равен 0')
else:
    print('Коэффициент эксцесса не равен 0')

dfUR = pd.read_csv('Data/Urov_11subg-nm.csv')
dfUR = dfUR.drop(dfUR.columns[0], axis=1)
freqMatrix = dfUR / dfUR.sum()

if CritIndependenceComponents(freqMatrix.to_numpy(), n=12, m=34):
    print('Компоненты случайной величины независимы')
else:
    print('Компоненты случайной величины зависимы')