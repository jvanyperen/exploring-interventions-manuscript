import numpy as np
from scipy.optimize import curve_fit


def get_hosp_remove_coefficient(dis_pats, dead_pats, verbose=True):
    popt, R2 = fit_linear(dis_pats, dead_pats, c=True)

    if verbose:
        print(f"y intercept = {popt[0]}, rH = {popt[1]}")
        print(f"R2 = {R2}")

    return popt[1]


def fit_linear(xdata, ydata, c=True):
    func = linear
    if not c:
        func = linear_no_intercept

    try:
        popt, _ = curve_fit(func, xdata, ydata)
        R2 = calc_R2(func, xdata, ydata, popt)

        if np.isnan(R2):
            R2 = -np.inf

        return popt, R2
    except RuntimeError:
        return [0, 0, 0, 0], -np.inf


def linear(x, a, b):
    return a + b * x


def linear_no_intercept(x, b):
    return b * x


def calc_R2(func, xdata, ydata, coeff):
    res = np.array([y - func(x, *coeff) for x, y in zip(xdata, ydata)])
    ss_res = sum(res**2)
    ss_tot = sum((ydata - np.mean(ydata)) ** 2)

    return 1.0 - ss_res / ss_tot
