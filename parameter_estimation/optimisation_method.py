import numpy as np
from scipy.optimize import minimize

import parameter_estimation.simulate as sim
import parameter_estimation.metrics_loglikelihoods as ml


MINIMIZE_OPTIONS = {"iprint": 1, "maxiter": 5}  # -1
MINIMIZE_INFO = True
ABS_ERR = 1.0
REL_ERR = 0.05


def optimisation_method(
    idxs,
    bounds,
    initial_conditions,
    params,
    T,
    population_size,
    data_dict,
    weights_dict,
    ll_value,
):
    flags = [False for _ in range(len(idxs))]
    out = False
    count = 0

    while out is False:
        for idx, combined_idx in enumerate(idxs):
            if not (flags[idx] is True):
                flags[idx], params, initial_conditions = run_optimisation(
                    combined_idx,
                    bounds[idx],
                    params,
                    initial_conditions,
                    T,
                    population_size,
                    data_dict,
                    weights_dict,
                    ll_value,
                )

                _, X, _ = sim.standard_ode_solver(
                    initial_conditions, params, population_size, T
                )

                metrics_dict = ml.get_metrics(X, params, T, population_size)
                ll_value, _ = ml.get_loglikelihood(
                    metrics_dict, data_dict, weights_dict, population_size
                )

            count += 1

        out = True
        for flag in flags:
            if flag is False:
                out = False

    return params, initial_conditions


def run_optimisation(
    combined_idx,
    bounds,
    params,
    initial_conditions,
    T,
    population_size,
    data_dict,
    weights_dict,
    ll_value,
):

    res = optimisation_wrapper(
        combined_idx,
        bounds,
        params,
        initial_conditions,
        T,
        population_size,
        data_dict,
        weights_dict,
    )

    new_params = params.copy()
    new_initial_conditions = initial_conditions.copy()
    try:
        for idx, p_idx in enumerate(combined_idx[0]):
            new_params[p_idx] = res[idx]

        try:
            for idx, ic_idx in enumerate(combined_idx[1]):
                new_initial_conditions[ic_idx] = res[idx + len(combined_idx[0])]
        except TypeError:
            pass
    except TypeError:
        for idx, ic_idx in enumerate(combined_idx[1]):
            new_initial_conditions[ic_idx] = res[idx]

    accept = error_check(
        new_params,
        new_initial_conditions,
        T,
        population_size,
        data_dict,
        weights_dict,
        ll_value,
    )
    if accept == 0:
        return True, params, initial_conditions

    if accept == 1:
        # good fit but not converged yet
        return False, new_params, new_initial_conditions
    else:
        # fit has converged
        return True, new_params, new_initial_conditions


def optimisation_wrapper(
    combined_idx,
    bounds,
    params,
    initial_conditions,
    T,
    population_size,
    data_dict,
    weights_dict,
):
    try:
        PP0 = [params[p] for p in combined_idx[0]]
        try:
            PP0.extend([initial_conditions[p] for p in combined_idx[1]])
        except TypeError:
            pass
    except TypeError:
        PP0 = [initial_conditions[p] for p in combined_idx[1]]

    res = minimize(
        observational_wrapper,
        PP0,
        args=(
            combined_idx,
            params,
            initial_conditions,
            T,
            population_size,
            data_dict,
            weights_dict,
        ),
        bounds=bounds,
        options=MINIMIZE_OPTIONS,
    )

    if MINIMIZE_INFO:
        print("----------------------")
        print("result: " + str(res.x))
        print("message: " + str(res.message))
        print("nfev: " + str(res.nfev))
        print("nit: " + str(res.nit))
        print("status: " + str(res.status))
        print("sucess: " + str(res.success))
        print("----------------------")

    return res.x


def observational_wrapper(
    P,
    combined_idx,
    params,
    initial_conditions,
    T,
    population_size,
    data_dict,
    weights_dict,
):
    new_params = params.copy()
    new_initial_conditions = initial_conditions.copy()

    try:
        for idx, p_idx in enumerate(combined_idx[0]):
            new_params[p_idx] = P[idx]

        try:
            for idx, ic_idx in enumerate(combined_idx[1]):
                new_initial_conditions[ic_idx] = P[idx + len(combined_idx[0])]
        except TypeError:
            pass
    except TypeError:
        for idx, ic_idx in enumerate(combined_idx[1]):
            new_initial_conditions[ic_idx] = P[idx]

    _, X, err = sim.observational_solver(
        new_initial_conditions, new_params, population_size, T
    )

    if err is True:
        return np.inf

    metrics_dict = ml.get_metrics(X, new_params, T, population_size)
    ll_value, _ = ml.get_loglikelihood(
        metrics_dict, data_dict, weights_dict, population_size
    )

    return ll_value


def error_check(
    params,
    initial_conditions,
    T,
    population_size,
    data_dict,
    weights_dict,
    ll_value,
):
    abs_residual, rel_residual = get_residual_errors(
        params,
        initial_conditions,
        T,
        population_size,
        data_dict,
        weights_dict,
        ll_value,
    )

    if abs_residual < 0:
        print("Negative log-likelihood error. This fit was not accepted")
        return 0
    elif rel_residual < 1.0e-5:
        print("Bad line search. This fit was not accepted")
        return 0

    if (abs_residual < ABS_ERR) or (rel_residual < REL_ERR):
        return 2

    return 1


def get_residual_errors(
    params,
    initial_conditions,
    T,
    population_size,
    data_dict,
    weights_dict,
    ll_value,
):
    _, X, _ = sim.standard_ode_solver(
        initial_conditions, params, population_size, T
    )

    metrics_dict = ml.get_metrics(X, params, T, population_size)
    new_ll_value, _ = ml.get_loglikelihood(
        metrics_dict, data_dict, weights_dict, population_size
    )

    return get_residual(new_ll_value, ll_value)


def get_residual(new_value, old_value):
    abs_err = old_value - new_value
    rel_err = abs_err / np.abs(old_value)

    return abs_err, rel_err
