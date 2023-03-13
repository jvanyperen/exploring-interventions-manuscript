import json
import numpy as np
from scipy.integrate import ode

import Rt_formula as Rf
import ode_systems as odes
import abm_branch_approach as abm

TIME_STEP = 0.01
MONTE_CARLO_SIMS = 1000
ABM_TIME_STEP = 0.5


def hospital_intervention_R0_approach(
    region_name,
    R0,
    Hlfac=0.5,
    Hufac=0.0012,
    init_prop=0.999,
    end_time=2000,
    cut_off=600,
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
    CR0 = R0 / initial_R0

    Hu = Hufac * population_size
    Hl = Hlfac * Hu

    t, solution, in_intervention, _ = intervention_ode_solver(
        initial_conditions, params, population_size, Hl, Hu, CR0, end_time
    )

    new_params = np.copy(params)
    new_params[0] *= CR0

    cut_idx = min(int(cut_off / TIME_STEP), len(t))

    Rt_dict = {
        "t": t[:cut_idx],
        "Rt": Rf.calculate_Rt(
            new_params, solution[:cut_idx, 0], population_size
        ),
        "Rt_in": Rf.calculate_Rt(
            new_params, in_intervention[:cut_idx, 0], population_size
        ),
    }

    H_dict = {
        "t": t[:cut_idx],
        "H": solution[:cut_idx, 4],
        "H_in": in_intervention[:cut_idx, 4],
        "pN": 100.0 / population_size,
        "Hufac": Hufac,
    }

    DU = solution[-1, 6]
    DH = solution[-1, 8]

    met_dict = {
        "dead": DU + DH,
        "pN": 100.0 / population_size,
    }

    return H_dict, Rt_dict, met_dict


def intervention_ode_solver(
    initial_conditions, params, population_size, Hl, Hu, CR0, end_time
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
    in_X = np.zeros_like(X)
    X[0, :] = initial_conditions
    in_X[0, :] = np.nan

    ell = False
    # no intervention

    # solve loop
    for idx, _ in enumerate(T[1:]):
        X[idx + 1, :] = s.integrate(s.t + TIME_STEP)
        in_X[idx + 1, :] = np.nan

        if not s.successful:
            print("An error in LSODA")
            return T[: idx + 2], X[: idx + 2, :], in_X[: idx + 2, :], True

        if s.t < 1.0:
            continue

        if sum(X[idx + 1, 1:4]) < 1.0:
            # stop if E + U + I < 1.0, i.e. no infection
            return T[: idx + 2], X[: idx + 2, :], in_X[: idx + 2, :], False

        if (not ell) and (X[idx + 1, 4] >= Hu):
            # not in intervention and demand is over capacity
            ell = True
            in_X[idx + 1, :] = X[idx + 1, :]
            s.set_f_params(params, population_size)
            continue

        if ell:
            # in intervention
            in_X[idx + 1, :] = X[idx + 1, :]
            if X[idx + 1, 4] < Hl:
                # demand has gone below threshold
                ell = False
                s.set_f_params(new_params, population_size)

    return T, X, in_X, False


def hospital_intervention_R0_abm_approach(
    region_name,
    R0,
    Hlfac=0.5,
    Hufac=0.0012,
    init_prop=0.999,
    end_time=2000,
    C=9.0,
    lb=0.01,
    ub=0.99,
    cut_off=1000,
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

    Hu = Hufac * population_size
    Hl = Hlfac * Hu

    t, mc_solution, mc_in_intervention = intervention_abm_solver(
        initial_conditions, abm_params, Hl, Hu, CR0, end_time
    )

    new_params = np.copy(params)
    new_params[0] *= CR0

    cut_idx = min(int(cut_off / ABM_TIME_STEP), len(t))

    Rt_dict = {
        "t": t[:cut_idx],
        "Rt_mean": Rf.calculate_Rt(
            new_params,
            np.mean(mc_solution[0], axis=1)[:cut_idx],
            population_size,
        ),
        "Rt_lb": Rf.calculate_Rt(
            new_params,
            np.quantile(mc_solution[0], lb, axis=1)[:cut_idx],
            population_size,
        ),
        "Rt_ub": Rf.calculate_Rt(
            new_params,
            np.quantile(mc_solution[0], ub, axis=1)[:cut_idx],
            population_size,
        ),
        "Rt_in_mean": Rf.calculate_Rt(
            new_params,
            np.mean(mc_in_intervention[0], axis=1)[:cut_idx],
            population_size,
        ),
        "Rt_in_lb": Rf.calculate_Rt(
            new_params,
            np.quantile(mc_in_intervention[0], lb, axis=1)[:cut_idx],
            population_size,
        ),
        "Rt_in_ub": Rf.calculate_Rt(
            new_params,
            np.quantile(mc_in_intervention[0], ub, axis=1)[:cut_idx],
            population_size,
        ),
    }

    H_dict = {
        "t": t[:cut_idx],
        "H_mean": np.mean(mc_solution[4], axis=1)[:cut_idx],
        "H_lb": np.quantile(mc_solution[4], lb, axis=1)[:cut_idx],
        "H_ub": np.quantile(mc_solution[4], ub, axis=1)[:cut_idx],
        "H_in_mean": np.mean(mc_in_intervention[4], axis=1)[:cut_idx],
        "H_in_lb": np.quantile(mc_in_intervention[4], lb, axis=1)[:cut_idx],
        "H_in_ub": np.quantile(mc_in_intervention[4], ub, axis=1)[:cut_idx],
        "pN": 100.0 / population_size,
        "Hufac": Hufac,
    }

    DU = mc_solution[6][-1, :]
    DH = mc_solution[8][-1, :]
    DUH = DU + DH

    DUH_mean = np.mean(DUH)
    DUH_lb = np.quantile(DUH, lb)
    DUH_ub = np.quantile(DUH, ub)

    met_dict = {
        "dead_mean": DUH_mean,
        "dead_lb": DUH_lb,
        "dead_ub": DUH_ub,
        "pN": 100.0 / population_size,
    }

    return H_dict, Rt_dict, met_dict


def intervention_abm_solver(initial_conditions, params, Hl, Hu, CR0, end_time):
    new_params = np.copy(params)
    new_params[1] *= CR0

    X_mc = {}
    in_X_mc = {}
    T = np.arange(0, end_time + ABM_TIME_STEP, ABM_TIME_STEP)

    for idx in range(MONTE_CARLO_SIMS):
        X = np.zeros([len(T), len(initial_conditions)])
        in_X = np.zeros_like(X)
        X[0, :] = initial_conditions
        in_X[0, :] = np.nan

        ell = False
        # no intervention

        # solve loop
        for t_idx, t in enumerate(T[1:]):
            if ell:
                X[t_idx + 1, :] = abm.standard_model(X[t_idx, :], params)
                in_X[t_idx + 1, :] = X[t_idx + 1, :]
            else:
                X[t_idx + 1, :] = abm.standard_model(X[t_idx, :], new_params)
                in_X[t_idx + 1, :] = np.nan

            if t < 1.0:
                continue

            if sum(X[t_idx + 1, 1:4]) < 1.0:
                # stop if E + U + I < 1.0, i.e. no infection
                X[t_idx + 2 :, :] = X[t_idx + 1, :]
                in_X[t_idx + 2 :, :] = in_X[t_idx + 1, :]
                break

            if (not ell) and (X[t_idx + 1, 4] >= Hu):
                # not in intervention and demand is over capacity
                ell = True
                in_X[t_idx + 1, :] = X[t_idx + 1, :]
                continue

            if ell:
                # in intervention
                if X[t_idx + 1, 4] < Hl:
                    # demand has gone below threshold
                    ell = False

        X_mc[idx] = X
        in_X_mc[idx] = in_X

    X_mcc = {}
    in_X_mcc = {}
    for idx in range(len(initial_conditions)):
        mat = np.zeros([len(T), MONTE_CARLO_SIMS])
        mat2 = np.zeros([len(T), MONTE_CARLO_SIMS])
        for idx2 in range(MONTE_CARLO_SIMS):
            mat[:, idx2] = X_mc[idx2][:, idx]
            mat2[:, idx2] = in_X_mc[idx2][:, idx]
        X_mcc[idx] = mat
        in_X_mcc[idx] = mat2

    return T, X_mcc, in_X_mcc
