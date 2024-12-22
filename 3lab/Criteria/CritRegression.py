from scipy import stats
import numpy as np
from math import sqrt

def CritRegression(sel_1, sel_2, alpha):
    n = len(sel_1)
    mx = np.mean(sel_1)
    my = np.mean(sel_2)
    cent_x = sel_1 - mx
    cent_y = sel_2 - my
    Qx = np.sum(cent_x ** 2)
    Qy = np.sum(cent_y ** 2)
    Qxy = np.sum(cent_x * cent_y)
    print(f"Qx = {Qx}")
    print(f"Qy = {Qy}")
    print(f"Qxy = {Qxy}")
    
    rho = Qxy / sqrt(Qx * Qy)
    print(f"rho = {rho}")
    
    Beta1 = Qxy / Qx
    Beta0 = my - Beta1*mx
    print(f"y = {Beta0} + {Beta1}x")
    Y = lambda x: Beta0 + Beta1 * x
    
    errors = sel_2 - Y(sel_1)
    Qe = np.sum(errors ** 2)
    print(f"Остаточная сумма квадратов: Qe = {Qe}")
    QR = Qy - Qe
    print(f"QR = {QR}")
    
    S2e = Qe / (n - 2)
    print(f"Остаточная дисперсия: S2e = {S2e}")
    #print(f"std ошибок наблюдений: {sqrt(S2e)}")  
    #С интервалом ДИСПЕРСИИ ошибок наблюдений что-то не так, там должно быть что-то другое
    left  = Qe / stats.chi2(df = n - 2).ppf(1 - alpha/2)
    right = Qe / stats.chi2(df = n - 2).ppf(alpha/2)
    print(f"({left}; {right})")
    
    R2 = 1 - Qe / Qy
    print(f"Коэффициент детерминации: R2 = {R2}")
    
    print("----- Значимость модели -----")
    F_v = Beta1 ** 2 * Qx / S2e
    F_crit = stats.f(1, n - 2).ppf(1 - alpha)
    print(f"F = {F_v}; crit = {F_crit}")
    if F_v < F_crit:
        print("Модель незначима, beta1 = 0 (H0)")
    else:
        print("Модель имеет значимость, beta1 =/= 0 (H1)")
        
    print("----- Критерий Дарбина-Уотсона -----")
    #неоткуда брать критическое значение
    d_v = np.sum([
        (errors[i] - errors[i - 1])**2 for i in range(1, n)
    ]) / Qe
    print(f"Значение критерия: {d_v}")
    
    print("----- Мера адекватности -----")
    
    grouped_by_x = dict()
    for x, y in zip(sel_1, sel_2):
        if x not in grouped_by_x:
            grouped_by_x[x] = [y]
        else:
            grouped_by_x[x].append(y)
    m = len(grouped_by_x.keys())
            
    Qn = 0
    for x in grouped_by_x.keys():
        ni = len(grouped_by_x[x])
        myi = sum(grouped_by_x[x]) / ni
        regr_yi = Y(x)
        Qn += ni * (myi - regr_yi)**2
        
    print(f"Мера адекватности: Qn = {Qn}")
    
    Qp = Qe - Qn
    print(f"Сумма квадратов чистой ошибки: Qp = {Qp}")
    
    F_v = (Qn * (n - m)) / (Qp * (m - 2))
    F_crit = stats.f(m - 2, n - m).ppf(1 - alpha)
    print(f"F = {F_v}, crit = {F_crit}")
    if F_v < F_crit:
        print("Модель адекватная (H0)")
    else:
        print("Модель неадекватна (H1)")
    
    return (Beta0, Beta1)