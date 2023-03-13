import numpy as np
from tqdm import tqdm
import json

import abm_LoS_approach as abm


def parameter_testing(
    region_name, T, dt, mc_sims, correction=True, lb=0.01, ub=0.99
):
    model_file = region_name + ".json"

    with open(model_file, "r") as json_file:
        model_dict = json.load(json_file)

    params = np.array(model_dict["params"])
    initial_conditions = np.array(model_dict["initial_conditions"], dtype=int)

    abm_params = np.zeros(len(params) + 1)
    abm_params[0] = 9.0
    abm_params[1] = params[0] / abm_params[0]
    abm_params[2:] = params[1:]
    abm_params[6] = abm_params[6] * (1.0 + abm_params[7])

    param_dist_dict = {}

    new_abm_params = np.copy(abm_params)
    if correction:
        for p_idx in range(len(abm_params)):
            if p_idx in [2, 4, 5, 6]:
                new_abm_params[p_idx] += abm.correct_time(
                    new_abm_params[p_idx], dt
                )

    _, solution_abm, param_dist_dict = abm_solver(
        initial_conditions, new_abm_params, T, dt, mc_sims
    )
    param_dist_dict["params"] = list(abm_params)

    prob_dist_dict = get_prob_dist(
        solution_abm, new_abm_params, dt, lb, ub, mc_sims
    )
    prob_dist_dict["params"] = list(abm_params)

    return param_dist_dict, prob_dist_dict


def get_prob_dist(solution, params, unit, lb, ub, mc_sims):
    # approximate mU
    mUs = (solution[6][-1, :] - solution[6][0, :]) / (
        solution[5][-1, :]
        + solution[6][-1, :]
        - solution[5][0, :]
        - solution[6][0, :]
    )

    # approximate mH
    mHs = (solution[8][-1, :] - solution[8][0, :]) / (
        solution[7][-1, :]
        + solution[8][-1, :]
        - solution[7][0, :]
        - solution[8][0, :]
    )

    # approximate p
    Uint = np.array([trap(solution[2][:, idx], unit) for idx in range(mc_sims)])
    Iint = np.array([trap(solution[3][:, idx], unit) for idx in range(mc_sims)])
    ps = (solution[2][-1, :] - solution[2][0, :] + params[4] * Uint) / (
        solution[2][-1, :]
        + solution[3][-1, :]
        - solution[2][0, :]
        - solution[3][0, :]
        + params[4] * Uint
        + params[5] * Iint
    )

    return {
        "p_mean": np.mean(ps),
        "p_lb": np.quantile(ps, lb),
        "p_ub": np.quantile(ps, ub),
        "mU_mean": np.mean(mUs),
        "mU_lb": np.quantile(mUs, lb),
        "mU_ub": np.quantile(mUs, ub),
        "mH_mean": np.mean(mHs),
        "mH_lb": np.quantile(mHs, lb),
        "mH_ub": np.quantile(mHs, ub),
    }


def trap(f, dx):
    return 0.5 * dx * (f[0] + f[-1] + 2.0 * np.sum(f[1:-1]))


def abm_solver(initial_conditions, params, end_time, dt, mc_sims):
    X_mc = {}
    t = np.arange(0, end_time + dt, dt)
    param_dist_dict = {"E": {}, "U": {}, "I": {}, "H": {}}

    for idx in tqdm(range(mc_sims), desc="ABM Monte Carlo"):
        X = np.zeros([len(t), len(initial_conditions)])
        X[0, :] = initial_conditions
        abm_df, param_dist_dict = abm.init_dataframe(
            initial_conditions, params, dt, param_dist_dict
        )
        num_infectious = sum(initial_conditions[2:4])

        for t_idx, _ in enumerate(t[1:]):
            abm_df, param_dist_dict, new_infections = abm.infect_agents(
                abm_df, num_infectious, params, dt, param_dist_dict
            )
            abm_df, param_dist_dict, comp_diffs = abm.update_agents(
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
        mat = np.zeros([T_idxs, mc_sims])
        for idx2 in range(mc_sims):
            mat[:, idx2] = X_mc[idx2][:, idx]
        X_mcc[idx] = mat

    return t, X_mcc, param_dist_dict
