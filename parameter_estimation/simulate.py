import numpy as np
from scipy.integrate import ode

import parameter_estimation.ode_systems as os


def standard_ode_solver(
    initial_conditions, params, population_size, end_time, dt=0.01
):
    s = ode(os.standard_model)
    s.set_f_params(params, population_size)
    s.set_initial_value(initial_conditions, 0)
    s.set_integrator("lsoda")

    T = np.arange(0, end_time + dt, dt)

    X = np.zeros([len(T), len(initial_conditions)])
    X[0, :] = initial_conditions

    # solve loop
    for idx, _ in enumerate(T[1:]):
        X[idx + 1, :] = s.integrate(s.t + dt)

        if not s.successful:
            print("An error in LSODA")
            return T[: idx + 2], X[: idx + 2, :], True

    return T, X, False


def observational_solver(
    initial_conditions, params, population_size, end_time, dt=0.01
):
    obs_initial_conditions = observational_initial_conditions(
        params, initial_conditions, population_size
    )
    s = ode(os.observational_model)
    s.set_f_params(params, population_size)
    s.set_initial_value(obs_initial_conditions, 0)
    s.set_integrator("lsoda")

    T = np.arange(0, end_time + dt, dt)

    X = np.zeros([len(T), len(obs_initial_conditions)])
    X[0, :] = obs_initial_conditions

    # solve loop
    for idx, _ in enumerate(T[1:]):
        X[idx + 1, :] = s.integrate(s.t + dt)

        if not s.successful:
            print("An error in LSODA")
            return T[: idx + 2], X[: idx + 2, :], True

    sir_X = observational_to_standard_mapping(X, params, population_size)

    return T, sir_X, False


def observational_initial_conditions(
    params, initial_conditions, population_size
):
    observational = np.zeros(len(initial_conditions))

    S, E, U, I, H = initial_conditions

    b, gE, p, _, gI, *_ = params

    observational[0] = I
    observational[3] = U
    observational[4] = H

    Ip = (1.0 - p) * gE * E - gI * I
    observational[1] = Ip

    inf_prop = (U + I) / population_size
    Ep = b * inf_prop * S - gE * E
    observational[2] = (1.0 - p) * gE * Ep - gI * Ip

    return observational


def observational_to_standard_mapping(observational, params, population_size):
    standard = np.zeros_like(observational)

    I = observational[:, 0]
    Ip = observational[:, 1]
    Ipp = observational[:, 2]
    U = observational[:, 3]
    H = observational[:, 4]

    b, gE, p, _, gI, *_ = params

    standard[:, 3] = I
    standard[:, 2] = U
    standard[:, 4] = H

    E = (Ip + gI * I) / ((1.0 - p) * gE)
    standard[:, 1] = E

    inf_prop = (U + I) / population_size
    Ep = (Ipp + gI * Ip) / ((1.0 - p) * gE)
    standard[:, 0] = (Ep + gE * E) / (b * inf_prop)

    return standard
