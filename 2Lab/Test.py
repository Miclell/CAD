from scipy import stats
import numpy as np

from Criteria.CritFriedman import CritFriedman
from Criteria.CritKendall import CritKendall
from Criteria.CritKholmogorovSmirnov import CritKholmogorovSmirnov
from Criteria.CritKraskellWallis import CritKraskellWallis
from Criteria.CritMannaUitny import CritMannaUitny
from Criteria.CritMedian import CritMedian
from Criteria.CritSign import CritSign
from Criteria.CritSpearman import CritSpearman
from Criteria.CritWillkokson import CritWillkokson
from Criteria.CritQ import CritQ

print('-------------Критерий Фридмена-------------')
dataset = [
    np.random.normal(loc=0, scale=1, size=10),
    np.random.normal(loc=0.5, scale=1, size=10),
    np.random.normal(loc=1, scale=1, size=10)
]
alpha = 0.05
if CritFriedman(dataset, alpha):
    print('Гипотеза о равенстве средних рангов отклоняется')
else:
    print('Гипотеза о равенстве средних рангов не отклоняется')
print('-------------------------------------------')

print('-------------Критерий Кендалла-------------')
x = np.random.normal(0, 1, 20).tolist()
y = np.random.normal(0.5, 1, 20).tolist()
if CritKendall(x, y, alpha):
    print('Корреляция Кендалла отсутствует')
else:
    print('Корреляция Кендалла существует')
print('-------------------------------------------')

print('-------Критерий Колмогорова-Смирнова-------')
x = np.random.normal(0, 1, 30).tolist()
y = np.random.normal(0.5, 1, 30).tolist()
if CritKholmogorovSmirnov(x, y, alpha):
    print('Распределения совпадают')
else:
    print('Распределения различаются')
print('-------------------------------------------')

print('---------Критерий Краскелла-Уоллиса--------')
x1 = np.random.normal(0, 1, 15).tolist()
x2 = np.random.normal(0.5, 1, 15).tolist()
x3 = np.random.normal(1, 1, 15).tolist()
x4 = np.random.normal(1.5, 1, 15).tolist()
if CritKraskellWallis(x1, x2, x3, x4, alpha):
    print('Гипотеза о равенстве медиан отклоняется')
else:
    print('Гипотеза о равенстве медиан не отклоняется')
print('-------------------------------------------')

print('-----------Критерий Манна-Уитни-----------')
x = np.random.normal(0, 1, 20).tolist()
y = np.random.normal(0.5, 1, 20).tolist()
alpha = 0.05
if CritMannaUitny(x, y, alpha):
    print('Нет значимых различий между выборками')
else:
    print('Есть значимые различия между выборками')
print('-------------------------------------------')

print('-------------Критерий медианы-------------')
data1 = np.random.normal(0, 1, 25).tolist()
data2 = np.random.normal(0.5, 1, 25).tolist()
if CritMedian(data1, data2, alpha):
    print('Нет значимых различий в медианах выборок')
else:
    print('Есть значимые различия в медианах выборок')
print('-------------------------------------------')

print('---------------Критерий знаков---------------')
x = np.random.normal(0, 1, 30).tolist()
y = np.random.normal(0, 1, 30).tolist()
alpha = 0.05
print('Результат критерия знаков:', CritSign(x, y, alpha))
print('--------------------------------------------')

print('----------Критерий рангов Спирмена----------')
x = np.random.normal(0, 1, 20).tolist()
y = np.random.normal(0.5, 1, 20).tolist()
print('Результат критерия Спирмена:', CritSpearman(x, y, alpha))
print('--------------------------------------------')

print('-----------Критерий Уилкоксона--------------')
x = np.random.normal(0, 1, 25).tolist()
y = np.random.normal(0.5, 1, 25).tolist()
if CritWillkokson(x, y, alpha):
    print('Нет значимых различий между выборками')
else:
    print('Есть значимые различия между выборками')
print('--------------------------------------------')

print('-------------Критерий Q-Кохрена-----------')
sels_table = np.random.randint(1, 10, (5, 3))
if CritQ(sels_table, alpha):
    print('Гипотеза об однородности дисперсий не отклоняется')
else:
    print('Гипотеза об однородности дисперсий отклоняется')
print('-------------------------------------------')