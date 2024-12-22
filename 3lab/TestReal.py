from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from Criteria.CritRegression import CritRegression

print('-----------------Регрессия-----------------')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('Data/nba.csv')

par1 = 'Win%'
par2 = 'Зарплатный фонд'
d1 = df[df['Сезон'] == '2008-09'][par1].reset_index(drop=True)
d2 = df[df['Сезон'] == '2008-09'][par2].reset_index(drop=True)

print(pd.DataFrame({par1: d1, par2: d2}))
regr_coef = CritRegression(d1, d2, 0.05)
Y = lambda x: regr_coef[0] + regr_coef[1] * x
regr_points_x = np.linspace(min(d1), max(d1), 1000)
regr_points_y = Y(regr_points_x)

plt.plot(d1, d2, 'o')
plt.plot(regr_points_x, regr_points_y, color='red')
ax = plt.gca()
ax.set_xlabel(par1)
ax.set_ylabel(par2)
plt.show()
print('-------------------------------------------')