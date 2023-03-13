import numpy as np

import parameter_estimation.simulate as sim
import parameter_estimation.metrics_loglikelihoods as ml

MAX_ITER = 100
RES_TOL = 1.0e-8
TRISECTION_INFO = True


def trisection_wrapper(
    ic_idx,
    initial_conditions,
    params,
    fac,
    T,
    population_size,
    data_dict,
    weights_dict,
):
    p_bounds = [
        initial_conditions[ic_idx] / fac,
        initial_conditions[ic_idx] * fac,
    ]

    res = trisection(
        loglikelihood_wrapper,
        p_bounds,
        params,
        initial_conditions,
        T,
        ic_idx,
        population_size,
        data_dict,
        weights_dict,
        max_iter=MAX_ITER,
        res_tol=RES_TOL,
    )

    if TRISECTION_INFO:
        print("----------------------")
        print("result: " + str(res["x"]))
        print("nit: " + str(res["iterations"]))
        print("sucess: " + str(res["success"]))
        print("cauchy error: " + str(res["cauchy_err"]))
        print("----------------------")

    initial_conditions[ic_idx] = res["x"]

    return initial_conditions


def trisection(
    func,
    lims,
    params,
    initial_conditions,
    T,
    ic_idx,
    population_size,
    data_dict,
    weights_dict,
    max_iter=100,
    res_tol=1.0e-8,
):
    k = 0
    b0, b1 = lims
    while k < max_iter:
        x0 = b0 + (b1 - b0) / 3.0
        x1 = b0 + 2.0 * (b1 - b0) / 3.0

        f0 = func(
            x0,
            params,
            initial_conditions,
            T,
            ic_idx,
            population_size,
            data_dict,
            weights_dict,
        )
        f1 = func(
            x1,
            params,
            initial_conditions,
            T,
            ic_idx,
            population_size,
            data_dict,
            weights_dict,
        )

        if f0 < f1:
            b1 = x1
        else:
            b0 = x0

        if np.abs(b1 - b0) < res_tol:
            conv = True
            break

        k += 1
    else:
        conv = False

    return {
        "x": x1 * (f0 > f1) + x0 * (f0 <= f1),
        "success": conv,
        "iterations": k,
        "cauchy_err": np.abs(b1 - b0),
    }


def loglikelihood_wrapper(
    P,
    params,
    initial_conditions,
    T,
    ic_idx,
    population_size,
    data_dict,
    weights_dict,
):
    new_initial_conditions = initial_conditions.copy()
    new_initial_conditions[ic_idx] = P

    _, X, _ = sim.standard_ode_solver(
        new_initial_conditions, params, population_size, T
    )
    metrics_dict = ml.get_metrics(X, params, T, population_size)
    ll_value, _ = ml.get_loglikelihood(
        metrics_dict, data_dict, weights_dict, population_size
    )

    return ll_value
