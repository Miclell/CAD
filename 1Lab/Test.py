import random
import numpy as np
from scipy import stats

from Criteria.CritEqualityProb import CritEqualityProb
from Criteria.CritShapiroWilk import CritShapiroWilk
from Criteria.CritEqualityVar import CritEqualityVar
from Criteria.CritEqualityMean import CritEqualityMean, CritEqualityMeanNorm
from Criteria.CritExcess import CritExcess
from Criteria.CritSymmetry import CritSymmetry
from Criteria.CritIndependenceComponents import CritIndependenceComponents


print('-----------Критерий Шапиро-Уилка-----------')
n = 30
alpha = 0.05
a = random.uniform(-5, 5)
x = np.random.normal(loc=a, scale=alpha, size=n)
x1 = np.random.randint(5, 10, n)

if CritShapiroWilk(x):
    print('Распределение генеральной совокупности нормальное (+)')
else:
    print('Распределение генеральной совокупности не нормальное (-)')

if CritShapiroWilk(x1):
    print('Распределение генеральной совокупности нормальное (-)')
else:
    print('Распределение генеральной совокупности не нормальное (+)')
print('-------------------------------------------')

#

print('------Критерий равенства вероятностей------')
m = 1000
n = 800
x1 = sorted(stats.binom(m, p=0.4).rvs(size=m))
y1 = sorted(stats.binom(n, p=0.4).rvs(size=n))
x2 = sorted(stats.binom(m, p=0.8).rvs(size=m))
y2 = sorted(stats.binom(n, p=0.6).rvs(size=n))

if CritEqualityProb(x1, y1):
    print('Вероятности успехов совпадают (+)')
else:
    print('Вероятности успехов не совпадают (-)')

if CritEqualityProb(x2, y2):
    print('Вероятности успехов совпадают (-)')
else:
    print('Вероятности успехов не совпадают (+)')
print('-------------------------------------------')

#

print('--------Критерий равенства дисперсий-------')
n = 100
m = 100
x1 = sorted(stats.norm(loc=1, scale=0.51).rvs(size=n))
y1 = sorted(stats.norm(loc=0, scale=0.49).rvs(size=m))
x2 = sorted(stats.norm(loc=1, scale=0.01).rvs(size=n))
y2 = sorted(stats.norm(loc=0, scale=0.1).rvs(size=m))

if CritEqualityVar(x1, y1):
    print('Дисперсии распределений равны (+)')
else:
    print('Дисперсии распределений не равны (-)')

if CritEqualityVar(x2, y2):
    print('Дисперсии распределений равны (-)')
else:
    print('Дисперсии распределений не равны (+)')
print('-------------------------------------------')

#

print('---------Критерий равенства средних--------')
n = 100
m = 100
x1 = sorted(stats.norm(loc=1, scale=0.51).rvs(size=n))
y1 = sorted(stats.norm(loc=1, scale=0.49).rvs(size=m))
x2 = sorted(stats.norm(loc=1, scale=0.01).rvs(size=n))
y2 = sorted(stats.norm(loc=0, scale=0.1).rvs(size=m))

if CritEqualityMean(x1, y1):
    print('Мат ожидания распределений равны (+)')
else:
    print('Мат ожидания распределений не равны (-)')

if CritEqualityMean(x2, y2):
    print('Мат ожидания распределений равны (-)')
else:
    print('Мат ожидания распределений не равны (+)')

if CritEqualityMeanNorm(x1, y1):
    print('Мат ожидания распределений равны (+)')
else:
    print('Мат ожидания распределений не равны (-)')

if CritEqualityMeanNorm(x2, y2):
    print('Мат ожидания распределений равны (-)')
else:
    print('Мат ожидания распределений не равны (+)')
print('-------------------------------------------')

#

print('-----Критерий независимости компонент------')
n = 100
m = 100
randomValues = np.random.rand(n, m)
freqMatrix = randomValues / randomValues.sum()

if CritIndependenceComponents(freqMatrix, n=50, m=50):
    print('Компоненты случайной величины независимы')
else:
    print('Компоненты случайной величины зависимы')
print('-------------------------------------------')

#

print('-----------Эксцесс и асимметрия------------')
print('Для нормального:')
n = 100
x = sorted(stats.norm(loc=0.5, scale=0.5).rvs(size=n))

if CritSymmetry(x):
    print('Коэффициент асимметрии равен 0')
else:
    print('Коэффициент асимметрии не равен 0')

if CritExcess(x):
    print('Коэффициент эксцесса равен 0')
else:
    print('Коэффициент эксцесса не равен 0')

print('Для экспоненциального:')
x1 = sorted(stats.expon(loc=0.45, scale=0.7).rvs(size=n))
if CritSymmetry(x1):
    print('Коэффициент асимметрии равен 0')
else:
    print('Коэффициент асимметрии не равен 0')

if CritExcess(x1):
    print('Коэффициент эксцесса равен 0')
else:
    print('Коэффициент эксцесса не равен 0')
print('-------------------------------------------')
