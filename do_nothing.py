import json
import numpy as np
from scipy.integrate import ode
from tqdm import tqdm

import Rt_formula as Rf
import ode_systems as odes
import abm_branch_approach as abm

TIME_STEP = 0.01
MONTE_CARLO_SIMS = 1000
ABM_TIME_STEP = 0.25


def do_nothing_approach(
    region_name, R0, init_prop=0.999, end_time=1000, cut_off=400
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
    params[0] *= R0 / initial_R0

    t, solution, _ = do_nothing_ode_solver(
        initial_conditions, params, population_size, end_time
    )

    Rt_dict = {
        "t": t,
        "Rt": Rf.calculate_Rt(params, solution[:, 0], population_size),
    }

    cut_idx = min(int(cut_off / TIME_STEP), len(t))

    H_dict = {
        "t": t[:cut_idx],
        "H": solution[:cut_idx, 4],
        "pN": 100.0 / population_size,
    }

    H_argmax = np.argmax(solution[:, 4])
    t_mho = t[H_argmax]
    max_hosp_occ = solution[H_argmax, 4]

    DU = solution[-1, 6]
    DH = solution[-1, 8]

    met_dict = {
        "max_hosp": max_hosp_occ,
        "peak_beds": t_mho,
        "dead": DU + DH,
        "pN": 100.0 / population_size,
    }

    return H_dict, Rt_dict, met_dict


def do_nothing_ode_solver(
    initial_conditions, params, population_size, end_time
):
    s = ode(odes.standard_model)
    s.set_f_params(params, population_size)
    s.set_initial_value(initial_conditions, 0)
    s.set_integrator("lsoda")

    T = np.arange(0, end_time + TIME_STEP, TIME_STEP)

    X = np.zeros([len(T), len(initial_conditions)])
    X[0, :] = initial_conditions

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

    return T, X, False


def do_nothing_abm_approach(
    region_name,
    R0,
    info=False,
    init_prop=0.999,
    C=9.0,
    scale=1,
    end_time=1000,
    lb=0.01,
    ub=0.99,
    cut_off=400,
):
    model_file = f"outputs/{region_name}.json"

    with open(model_file, "r") as json_file:
        model_dict = json.load(json_file)

    params = np.array(model_dict["params"])
    initial_conditions = np.zeros(len(model_dict["initial_conditions"]))
    population_size = model_dict["population_size"]

    initial_R0 = Rf.calculate_R0(params)

    abm_params = np.zeros(len(params) + 1)
    abm_params[0] = C
    abm_params[1] = params[0] / (C * population_size)
    abm_params[2:] = params[1:]

    if scale == 0:
        # 1.3 is the first row of Table 3
        # keeps a and C the same in the first row (think: bench mark)
        abm_params[0] *= R0 / 1.3
        abm_params[1] *= 1.3 / initial_R0
    else:
        abm_params[1] *= R0 / initial_R0

    params[0] *= R0 / initial_R0

    for idx in range(len(abm_params)):
        if not (idx in [0, 3, 7, 8]):
            abm_params[idx] *= ABM_TIME_STEP

    initial_conditions[1] = max(
        np.floor((1.0 - init_prop) * population_size), 1.0
    )
    initial_conditions[0] = population_size - initial_conditions[1]

    t, mc_solution = do_nothing_abm_solver(
        initial_conditions, abm_params, end_time
    )

    cut_idx = min(int(cut_off / ABM_TIME_STEP), len(t))

    Rt_dict = {
        "t": t[:cut_idx],
        "Rt_mean": Rf.calculate_Rt(
            params,
            np.mean(mc_solution[0], axis=1)[:cut_idx],
            population_size,
        ),
        "Rt_lb": Rf.calculate_Rt(
            params,
            np.quantile(mc_solution[0], lb, axis=1)[:cut_idx],
            population_size,
        ),
        "Rt_ub": Rf.calculate_Rt(
            params,
            np.quantile(mc_solution[0], ub, axis=1)[:cut_idx],
            population_size,
        ),
    }

    H_dict = {
        "t": t[:cut_idx],
        "H_mean": np.mean(mc_solution[4], axis=1)[:cut_idx],
        "H_lb": np.quantile(mc_solution[4], lb, axis=1)[:cut_idx],
        "H_ub": np.quantile(mc_solution[4], ub, axis=1)[:cut_idx],
        "pN": 100.0 / population_size,
    }

    t_mho = np.zeros(MONTE_CARLO_SIMS)
    max_hosp_occ = np.zeros(MONTE_CARLO_SIMS)

    for idx in range(MONTE_CARLO_SIMS):
        H_argmax = np.argmax(mc_solution[4][:, idx])
        t_mho[idx] = t[H_argmax]
        max_hosp_occ[idx] = mc_solution[4][H_argmax, idx]

    DU = mc_solution[6][-1, :]
    DH = mc_solution[8][-1, :]
    DUH = DU + DH

    max_hosp_occ_mean = np.mean(max_hosp_occ)
    max_hosp_occ_lb = np.quantile(max_hosp_occ, lb)
    max_hosp_occ_ub = np.quantile(max_hosp_occ, ub)

    t_mho_mean = np.mean(t_mho)
    t_mho_lb = np.quantile(t_mho, lb)
    t_mho_ub = np.quantile(t_mho, ub)

    DUH_mean = np.mean(DUH)
    DUH_lb = np.quantile(DUH, lb)
    DUH_ub = np.quantile(DUH, ub)

    met_dict = {
        "max_hosp_mean": max_hosp_occ_mean,
        "max_hosp_lb": max_hosp_occ_lb,
        "max_hosp_ub": max_hosp_occ_ub,
        "peak_beds_mean": t_mho_mean,
        "peak_beds_lb": t_mho_lb,
        "peak_beds_ub": t_mho_ub,
        "dead_mean": DUH_mean,
        "dead_lb": DUH_lb,
        "dead_ub": DUH_ub,
        "pN": 100.0 / population_size,
    }

    if info:
        max_IQR = np.amax(max_hosp_occ_ub - max_hosp_occ_lb)
        print(f"max IQR of H = {max_IQR:.4f}")
        print("-------------------------")

    return H_dict, Rt_dict, met_dict


def do_nothing_abm_solver(initial_conditions, params, end_time):
    X_mc = {}
    T = np.arange(0, end_time + ABM_TIME_STEP, ABM_TIME_STEP)
    for idx in tqdm(range(MONTE_CARLO_SIMS), desc="MC"):
        X = np.zeros([len(T), len(initial_conditions)])
        X[0, :] = initial_conditions

        for t_idx, t in enumerate(T[1:]):
            X[t_idx + 1, :] = abm.standard_model(X[t_idx, :], params)

            if t < 1.0:
                continue

            if sum(X[t_idx + 1, 1:4]) < 1.0:
                # E + U + I < 1.0, i.e. no infections
                X[t_idx + 2 :, :] = X[t_idx + 1, :]
                break

        X_mc[idx] = X

    X_mcc = {}
    for idx in range(len(initial_conditions)):
        mat = np.zeros([len(T), MONTE_CARLO_SIMS])
        for idx2 in range(MONTE_CARLO_SIMS):
            mat[:, idx2] = X_mc[idx2][:, idx]
        X_mcc[idx] = mat

    return T, X_mcc
