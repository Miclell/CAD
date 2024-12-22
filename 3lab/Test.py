from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

from Criteria.CritRegression import CritRegression

print('-----------------Регрессия-----------------')
sd1 = np.round(np.sort(stats.norm().rvs(size=100)), decimals = 3)
sd2 = np.sort(stats.norm().rvs(size=100))
regr_coef = CritRegression(sd1, sd2, 0.05)
SY = lambda x: regr_coef[0] + regr_coef[1] * x

sregr_points_x = np.linspace(min(sd1), max(sd1), 2)
sregr_points_y = SY(sregr_points_x)
plt.plot(sd1, sd2, 'o')
plt.plot(sregr_points_x, sregr_points_y)
plt.show()
print('-------------------------------------------')