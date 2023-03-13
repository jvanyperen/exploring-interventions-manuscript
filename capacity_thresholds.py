import json
import numpy as np
from scipy.integrate import ode
from scipy.optimize import root_scalar

import Rt_formula as Rf
import ode_systems as odes
import abm_branch_approach as abm

TIME_STEP = 0.01
MONTE_CARLO_SIMS = 5000
ABM_TIME_STEP = 0.25


def explore_objective(region_name, R0, Hufac, init_prop=0.999, end_time=1000):
    model_file = f"outputs/{region_name}.json"

    with open(model_file, "r") as json_file:
        model_dict = json.load(json_file)

    params = np.array(model_dict["params"])
    initial_conditions = np.zeros(len(model_dict["initial_conditions"]))
    population_size = model_dict["population_size"]

    initial_conditions[1] = max(
        np.floor((1.0 - init_prop) * population_size), 1.0
    )
    initial_conditions[0] = population_size - initial_conditions[1]

    initial_R0 = Rf.calculate_R0(params)
    CR0 = R0 / initial_R0

    Hu = Hufac * population_size

    _, solution, _ = intervention_ode_solver(
        initial_conditions, params, population_size, Hu, CR0, end_time
    )

    H_dict = {
        "max_H": np.amax(solution[:, 4]),
        "pN": 100.0 / population_size,
    }

    return H_dict


def minimise_objective(
    region_name, R0, Hufac=0.0012, init_prop=0.999, end_time=1000
):
    model_file = f"outputs/{region_name}.json"

    with open(model_file, "r") as json_file:
        model_dict = json.load(json_file)

    params = np.array(model_dict["params"])
    initial_conditions = np.zeros(len(model_dict["initial_conditions"]))
    population_size = model_dict["population_size"]

    initial_conditions[1] = max(
        np.floor((1.0 - init_prop) * population_size), 1.0
    )
    initial_conditions[0] = population_size - initial_conditions[1]

    initial_R0 = Rf.calculate_R0(params)

    Hmax = Hufac * population_size

    CR0 = R0 / initial_R0

    res = root_scalar(
        objective_func,
        args=(
            initial_conditions,
            params,
            CR0,
            population_size,
            end_time,
            Hmax,
        ),
        x0=Hmax,
        x1=0.0,
    )

    Hu = res.root

    Hudict = {
        "Hu": Hu,
        "HupHmax": 100.0 * Hu / Hmax,
        "pN": 100.0 / population_size,
    }

    return Hudict


def objective_func(
    Hu, initial_conditions, params, CR0, population_size, end_time, Hmax
):
    _, solution, _ = intervention_ode_solver(
        initial_conditions, params, population_size, Hu, CR0, end_time
    )

    max_H = np.amax(solution[:, 4])

    return max_H - Hmax


def intervention_ode_solver(
    initial_conditions, params, population_size, Hu, CR0, end_time
):
    new_params = np.copy(params)
    new_params[0] *= CR0

    s = ode(odes.standard_model)
    s.set_f_params(new_params, population_size)
    s.set_initial_value(initial_conditions, 0)
    s.set_integrator("lsoda", max_step=TIME_STEP)
    # max_step to dt for force solver to make change to params immediately

    T = np.arange(0, end_time + TIME_STEP, TIME_STEP)

    X = np.zeros([len(T), len(initial_conditions)])
    X[0, :] = initial_conditions

    ell = False
    # no intervention

    # solve loop
    for idx, _ in enumerate(T[1:]):
        X[idx + 1, :] = s.integrate(s.t + TIME_STEP)

        if not s.successful:
            print("An error in LSODA")
            return T[: idx + 2], X[: idx + 2, :], True

        if s.t < 1.0:
            continue

        if sum(X[idx + 1, 1:4]) < 1.0:
            # stop if E + U + I < 1.0, i.e. no infection
            return T[: idx + 2], X[: idx + 2, :], False

        if (not ell) and (X[idx + 1, 4] >= Hu):
            # not in intervention and demand is over capacity
            ell = True
            s.set_f_params(params, population_size)
            continue

    return T, X, False


def capacity_thresholds_abm(
    region_name,
    R0,
    Hup=1.0,
    Hmaxfac=0.0012,
    init_prop=0.999,
    end_time=1000,
    C=9.0,
    lb=0.01,
    ub=0.99,
    mc_trials=None,
):
    model_file = f"outputs/{region_name}.json"

    with open(model_file, "r") as json_file:
        model_dict = json.load(json_file)

    params = np.array(model_dict["params"])
    initial_conditions = np.zeros(len(model_dict["initial_conditions"]))
    population_size = model_dict["population_size"]

    abm_params = np.zeros(len(params) + 1)
    abm_params[0] = C
    abm_params[1] = params[0] / (C * population_size)
    abm_params[2:] = params[1:]

    for idx in range(len(abm_params)):
        if not (idx in [0, 3, 7, 8]):
            abm_params[idx] *= ABM_TIME_STEP

    initial_conditions[1] = max(
        np.floor((1.0 - init_prop) * population_size), 1.0
    )
    initial_conditions[0] = population_size - initial_conditions[1]

    initial_R0 = Rf.calculate_R0(params)
    CR0 = R0 / initial_R0

    thresholds_file = f"outputs/capacity_thresholds_{region_name}.json"

    with open(thresholds_file, "r") as json_file:
        region_dict = json.load(json_file)

    thresholds_R0s = np.array(region_dict["R0"])
    thresholds_HupHmaxs = np.array(region_dict["HupHmax"])
    Hup_idx = np.where(np.abs(thresholds_R0s - R0) < 1.0e-3)[0]

    Hmax = Hmaxfac * population_size
    Hup_max = thresholds_HupHmaxs[Hup_idx] / 100.0

    _, _, mc_breach_dict = intervention_abm_solver(
        initial_conditions,
        abm_params,
        Hup * Hup_max * Hmax,
        Hmax,
        CR0,
        end_time,
        mc_trials=mc_trials,
    )

    mc_sims = MONTE_CARLO_SIMS if mc_trials is None else mc_trials

    nb = mc_breach_dict["num"] / mc_sims
    breaching_dict = {
        "num": 100.0 * nb,
        "diff_from_mean": np.mean(mc_breach_dict["diff"]),
        "diff_from_lb": np.quantile(mc_breach_dict["diff"], lb),
        "diff_from_ub": np.quantile(mc_breach_dict["diff"], ub),
        "max_over_mean": np.mean(mc_breach_dict["max"]) if nb > 0 else 0,
        "max_over_lb": np.quantile(mc_breach_dict["max"], lb) if nb > 0 else 0,
        "max_over_ub": np.quantile(mc_breach_dict["max"], ub) if nb > 0 else 0,
        "Hu": Hup_max,
    }

    return breaching_dict


def intervention_abm_solver(
    initial_conditions, params, Hu, Hmax, CR0, end_time, mc_trials=None
):
    new_params = np.copy(params)
    new_params[1] *= CR0

    X_mc = {}
    T = np.arange(0, end_time + ABM_TIME_STEP, ABM_TIME_STEP)
    num_breaches = 0
    max_breach = []
    max_diff_breach_Hmax = []

    mc_sims = MONTE_CARLO_SIMS if mc_trials is None else mc_trials

    for idx in range(mc_sims):
        X = np.zeros([len(T), len(initial_conditions)])
        X[0, :] = initial_conditions

        ell = False
        breach = False
        # no intervention

        # solve loop
        for t_idx, t in enumerate(T[1:]):
            if ell:
                X[t_idx + 1, :] = abm.standard_model(X[t_idx, :], params)
            else:
                X[t_idx + 1, :] = abm.standard_model(X[t_idx, :], new_params)

            if t < 1.0:
                continue

            if sum(X[t_idx + 1, 1:4]) < 1.0:
                # stop if E + U + I < 1.0, i.e. no infection
                X[t_idx + 2 :, :] = X[t_idx + 1, :]
                break

            if (not ell) and (X[t_idx + 1, 4] >= Hu):
                # not in intervention and demand is over capacity
                ell = True

            if (X[t_idx + 1, 4] >= Hmax) and (not breach):
                num_breaches += 1
                breach = True

        X_mc[idx] = X
        max_H = np.max(X[:, 4])
        diff = 100.0 * (max_H - Hmax) / Hmax
        max_diff_breach_Hmax.append(diff)
        if breach:
            max_breach.append(diff)

    X_mcc = {}
    for idx in range(len(initial_conditions)):
        mat = np.zeros([len(T), mc_sims])
        for idx2 in range(mc_sims):
            mat[:, idx2] = X_mc[idx2][:, idx]
        X_mcc[idx] = mat

    breach_dict = {
        "num": num_breaches,
        "max": max_breach,
        "diff": max_diff_breach_Hmax,
    }

    return T, X_mcc, breach_dict
