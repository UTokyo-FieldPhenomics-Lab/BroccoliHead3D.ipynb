import numpy as np
import statsmodels.api as sm
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

def lowess(x, y, frac=0.5):
    lowess = sm.nonparametric.lowess
    z = lowess(y, x, frac)
    x_points = []
    y_points = []
    for x_y in z:
        x_points.append(x_y[0])
        y_points.append(x_y[1])
    return x_points, y_points

def linear(x, y, printf=True):
    def func(x, a, b):
        y = a * x + b
        return y

    def fitans(xdata, ydata):
        popt, pcov = curve_fit(func, xdata, ydata)
        residuals = ydata - func(xdata, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        return popt, pcov, r_squared
  
    t, p, r2 = fitans(x, y)
  
    x_points = np.linspace(min(x), max(x), num=max(int(max(x)-min(x))+1, 100), endpoint=True)
    y_points = func(x_points, *t)
    y_predict = func(x, *t)
    rmse_v = rmse(y_predict, y)
    if printf:
        print(f'y={round(t[0], 3)}x+{round(t[1], 3)}  R^2={round(r2, 2)}')
        print(f'RMSE = {rmse_v}')
        print('-----------------')
    return x_points, y_points, y_predict, t, r2, rmse_v

def rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))