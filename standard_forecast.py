import json
import numpy as np
from scipy.integrate import ode
from tqdm import tqdm

import ode_systems as odes
import abm_branch_approach as branch
import abm_LoS_approach as los

TIME_STEP = 0.01
B_MONTE_CARLO_SIMS = 1000
ABM_MONTE_CARLO_SIMS = 20


def standard_forecast(region_name, end_time=100):
    model_file = f"outputs/{region_name}.json"

    with open(model_file, "r") as json_file:
        model_dict = json.load(json_file)

    params = np.array(model_dict["params"])
    initial_conditions = np.array(model_dict["initial_conditions"])
    population_size = model_dict["population_size"]

    t, solution, _ = standard_ode_solver(
        initial_conditions, params, population_size, end_time
    )

    return {
        "t": t,
        "H": solution[:, 4],
        "pN": 100.0 / population_size,
    }


def forecast_wrong_parameters(region_name, other_region_name, end_time=100):
    model_file = f"outputs/{region_name}.json"

    with open(model_file, "r") as json_file:
        model_dict = json.load(json_file)
        initial_conditions = np.array(model_dict["initial_conditions"])

    population_size = model_dict["population_size"]

    other_model_file = f"outputs/{other_region_name}.json"
    with open(other_model_file, "r") as json_file:
        model_dict = json.load(json_file)
        params = np.array(model_dict["params"])

    t, solution, _ = standard_ode_solver(
        initial_conditions, params, population_size, end_time
    )

    return {
        "t": t,
        "H": solution[:, 4],
        "pN": 100.0 / population_size,
    }


def standard_ode_solver(initial_conditions, params, population_size, end_time):
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


def standard_forecast_branch(
    region_name, end_time=100, C=9.0, dt=1.0, lb=0.01, ub=0.99, info=False
):
    model_file = f"outputs/{region_name}.json"

    with open(model_file, "r") as json_file:
        model_dict = json.load(json_file)

    params = np.array(model_dict["params"])
    initial_conditions = np.array(model_dict["initial_conditions"]).astype(int)
    population_size = model_dict["population_size"]

    b_params = np.zeros(len(params) + 1)
    b_params[0] = C
    b_params[1] = params[0] / (C * population_size)
    b_params[2:] = params[1:]

    for idx in range(len(b_params)):
        if not (idx in [0, 3, 7, 8]):
            b_params[idx] *= dt

    t, mc_solution = standard_branch_solver(
        initial_conditions, b_params, end_time, dt
    )

    if info:
        H_lb = np.quantile(mc_solution[4], lb, axis=1)
        H_ub = np.quantile(mc_solution[4], ub, axis=1)
        H_mean = np.mean(mc_solution[4], axis=1)

        print(f"dt = {dt}")
        print(f"Max of mean = {np.max(H_mean)}")
        print(f"Max diff in range = {np.max(H_ub-H_lb)}")

    return {
        "t": t,
        "H_mean": np.mean(mc_solution[4], axis=1),
        "H_lb": np.quantile(mc_solution[4], lb, axis=1),
        "H_ub": np.quantile(mc_solution[4], ub, axis=1),
        "pN": 100.0 / population_size,
    }


def standard_branch_solver(initial_conditions, params, end_time, dt):
    X_mc = {}
    T = np.arange(0, end_time + dt, dt)
    for idx in tqdm(range(B_MONTE_CARLO_SIMS), desc="MC"):
        X = np.zeros([len(T), len(initial_conditions)])
        X[0, :] = initial_conditions

        for t_idx, t in enumerate(T[1:]):
            X[t_idx + 1, :] = branch.standard_model(X[t_idx, :], params)

            if t < 1.0:
                continue

            if sum(X[t_idx + 1, 1:4]) < 1.0:
                # E + U + I < 1.0, i.e. no infections
                X[t_idx + 2 :, :] = X[t_idx + 1, :]
                break

        X_mc[idx] = X

    X_mcc = {}
    for idx in range(len(initial_conditions)):
        mat = np.zeros([len(T), B_MONTE_CARLO_SIMS])
        for idx2 in range(B_MONTE_CARLO_SIMS):
            mat[:, idx2] = X_mc[idx2][:, idx]
        X_mcc[idx] = mat

    return T, X_mcc


def standard_forecast_LoS(
    region_name, end_time=100, C=9.0, dt=1.0, lb=0.01, ub=0.99, correction=True
):
    model_file = f"outputs/{region_name}.json"

    with open(model_file, "r") as json_file:
        model_dict = json.load(json_file)

    params = np.array(model_dict["params"])
    initial_conditions = np.array(model_dict["initial_conditions"]).astype(int)
    population_size = model_dict["population_size"]

    abm_params = np.zeros(len(params) + 1)
    abm_params[0] = C
    abm_params[1] = params[0] / C
    abm_params[2:] = params[1:]
    abm_params[6] = abm_params[6] * (1.0 + abm_params[7])

    if correction:
        for p_idx in range(len(abm_params)):
            if p_idx in [2, 4, 5, 6]:
                abm_params[p_idx] += los.correct_time(abm_params[p_idx], dt)

    t, mc_solution, _ = standard_LoS_solver(
        initial_conditions, abm_params, end_time, dt
    )

    return {
        "t": t,
        "H_mean": np.mean(mc_solution[4], axis=1),
        "H_lb": np.quantile(mc_solution[4], lb, axis=1),
        "H_ub": np.quantile(mc_solution[4], ub, axis=1),
        "pN": 100.0 / population_size,
    }


def standard_LoS_solver(initial_conditions, params, end_time, dt):
    X_mc = {}
    t = np.arange(0, end_time + dt, dt)
    param_dist_dict = {"E": {}, "U": {}, "I": {}, "H": {}}

    for idx in tqdm(range(ABM_MONTE_CARLO_SIMS), desc="ABM Monte Carlo"):
        X = np.zeros([len(t), len(initial_conditions)])
        X[0, :] = initial_conditions
        abm_df, param_dist_dict = los.init_dataframe(
            initial_conditions, params, dt, param_dist_dict
        )
        num_infectious = sum(initial_conditions[2:4])

        for t_idx, _ in enumerate(t[1:]):
            abm_df, param_dist_dict, new_infections = los.infect_agents(
                abm_df, num_infectious, params, dt, param_dist_dict
            )
            abm_df, param_dist_dict, comp_diffs = los.update_agents(
                abm_df, params, dt, param_dist_dict
            )
            comp_diffs[0] -= new_infections
            comp_diffs[1] += new_infections

            X[t_idx + 1, :] = X[t_idx, :] + comp_diffs

            if sum(X[t_idx + 1, 1:4]) < 1.0:
                X[t_idx + 2 :, :] = X[t_idx + 1, :]
                break

            num_infectious = sum(X[t_idx + 1, 2:4])
            abm_df["time"] = abm_df["time"] - dt

        X_mc[idx] = X

    T_idxs = len(t)
    X_mcc = {}
    for idx in range(len(initial_conditions)):
        mat = np.zeros([T_idxs, ABM_MONTE_CARLO_SIMS])
        for idx2 in range(ABM_MONTE_CARLO_SIMS):
            mat[:, idx2] = X_mc[idx2][:, idx]
        X_mcc[idx] = mat

    return t, X_mcc, param_dist_dict
