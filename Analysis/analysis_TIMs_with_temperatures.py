# 这个文档中函数作用是分析不同温度下的TIMs特征

import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit

def CV(data):
    return np.std(data) / np.mean(data)

def rCV_MAD(data):
    x = np.asarray(data, dtype=float)
    m = np.median(x)
    mad = np.median(np.abs(x - m))
    return (1.4826 * mad) / m

def avg(data):
    return np.mean(data)

# 求平均值和中位数
def mean_and_median(data):
    return np.mean(data), np.median(data)

# 求标准差，方差，幅值
def std_and_var_and_amplitude(data):
    return np.std(data), np.var(data), np.max(data) - np.min(data)

# 求一阶敏感度,做线性拟合
def first_order_sensitivity(data):
    x = np.arange(len(data))
    y = data
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope

# 求二阶敏感度,做非线性拟合
# 求二阶敏感度,做非线性拟合
def second_order_sensitivity(data):
    x = np.arange(len(data))
    y = data
    popt, pcov = curve_fit(lambda x, a, b, c: a * x**2 + b * x + c, x, y)
    a, b, c = popt  # 解包参数：a是二次项系数，b是一次项系数，c是常数项
    return a  # 返回二次项系数 a

# 分析集合函数
def analysis_TIMs_with_temperatures(data):
    results = {}
    results['CV'] = CV(data)
    results['rCV_MAD'] = rCV_MAD(data)
    results['avg'] = avg(data)
    results['mean'], results['median'] = mean_and_median(data)
    results['std'], results['var'], results['amplitude'] = std_and_var_and_amplitude(data)
    # results['first_order_sensitivity'] = first_order_sensitivity(data)
    # results['second_order_sensitivity'] = second_order_sensitivity(data)
    return results

