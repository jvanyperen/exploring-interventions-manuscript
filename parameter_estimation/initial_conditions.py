import numpy as np
from scipy.optimize import curve_fit


def get_initial_conditions(data_dict, population_size, p, gI):
    S0 = get_S0_no_data(population_size)
    E0 = get_E0_no_data(population_size)
    I0 = get_compartment_from_data(data_dict["adm"], gI)
    U0 = get_U0_no_data(I0, 1.0 - p)
    H0 = get_compartment_from_data(data_dict["occ"], None)

    return [S0, E0, U0, I0, H0]


def get_S0_no_data(N, fac=0.975):
    return fac * N


def get_E0_no_data(N, fac=0.005):
    return fac * N


def get_U0_no_data(I0, prob_hosp):
    return I0 * (1.0 - prob_hosp) / prob_hosp


def get_compartment_from_data(comp_data, rate=None):
    try:
        integral = True
        t_data = np.array([t[1] for t in comp_data["days"][:21]])
    except IndexError:
        integral = False
        t_data = comp_data["days"][:21]

    Gauss_coefs, R2G = fit_Gaussian(t_data, comp_data["data"][:21])
    cubic_coefs, R2p = fit_cubic(
        t_data, comp_data["data"][:21], integral=integral
    )

    if R2G >= R2p:
        # Gaussian fits better than cubic
        if integral:
            return Gaussian_derivative(0.0, *Gauss_coefs) / rate
        return Gaussian(0.0, *Gauss_coefs)

    # cubic fits better than Gaussian
    if integral:
        return cubic_derivative(0.0, *cubic_coefs) / rate
    return cubic(0.0, *cubic_coefs)


def fit_Gaussian(xdata, ydata):
    weighted_mean = sum(xdata * ydata) / sum(ydata)
    weighted_sd = np.sqrt(
        sum(ydata * (xdata - weighted_mean) ** 2) / sum(ydata)
    )

    try:
        popt, _ = curve_fit(
            Gaussian,
            xdata,
            ydata,
            p0=[max(ydata), weighted_mean, weighted_sd],
        )
        R2 = calc_R2(Gaussian, xdata, ydata, popt)

        if np.isnan(R2):
            R2 = -np.inf

        return popt, R2
    except RuntimeError:
        return [0, 0, 0], -np.inf


def Gaussian(x, a, mu, sigma):
    return a * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))


def Gaussian_derivative(x, a, mu, sigma):
    return (mu - x) * Gaussian(x, a, mu, sigma) / (sigma**2)


def fit_cubic(xdata, ydata, integral=True):

    try:
        popt, _ = curve_fit(
            cubic,
            xdata,
            ydata,
            bounds=cubic_bounds(integral),
        )
        R2 = calc_R2(cubic, xdata, ydata, popt)

        if np.isnan(R2):
            R2 = -np.inf

        return popt, R2
    except RuntimeError:
        return [0, 0, 0, 0], -np.inf


def cubic_bounds(integral):
    if integral:
        # need to keep linear coefficient positive so that
        # derivative is positive at origin
        return (
            [-np.inf, -np.inf, 0.0, -np.inf],
            [np.inf, np.inf, np.inf, np.inf],
        )

    # need to keep y intercept positive so that
    # function is positive at origin
    return (
        [-np.inf, -np.inf, -np.inf, 0.0],
        [np.inf, np.inf, np.inf, np.inf],
    )


def cubic(x, a, b, c, d):
    return a + b * x + c * x**2 + d * x**3


def cubic_derivative(x, a, b, c, d):
    return b + 2.0 * c * x + 3.0 * d * x**2


def calc_R2(func, xdata, ydata, coeff):
    res = np.array([y - func(x, *coeff) for x, y in zip(xdata, ydata)])
    ss_res = sum(res**2)
    ss_tot = sum((ydata - np.mean(ydata)) ** 2)

    return 1.0 - ss_res / ss_tot
