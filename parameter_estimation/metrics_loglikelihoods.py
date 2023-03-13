import numpy as np
import copy
import Rt_formula as rt

RT_BOUNDS = [0.5, 2.0]
N0_BOUNDS = [0.95, 1.0]


def trapz(fx, dx):
    return dx * (sum(fx) - 0.5 * fx[0] - 0.5 * fx[-1])


def solution_to_data(X, t0, dt, intervals):
    i_intervals = [[get_i_time(t, t0, dt) for t in I] for I in intervals]

    return [trapz(X[a:b], dt) for (a, b) in i_intervals]


def get_i_time(t, t0, dt):
    return int((t - t0) / dt)


def get_metrics(solution, params, T, population_size, t0=0, dt=0.01):
    metrics_dict = {}
    days = np.arange(t0 + 1, T + 1, step=1)
    day_intervals = np.array([[t - 1, t] for t in days])

    beds = solution[[get_i_time(t, t0, dt) for t in days], 4]
    adm = params[4] * np.array(
        solution_to_data(solution[:, 3], t0, dt, day_intervals)
    )
    dis = params[5] * np.array(
        solution_to_data(solution[:, 4], t0, dt, day_intervals)
    )
    dnh = (
        params[3]
        * params[7]
        * np.array(solution_to_data(solution[:, 2], t0, dt, day_intervals))
    )
    dhp = dis * params[6]

    metrics_dict = {
        "adm": adm,
        "occ": beds,
        "dis": dis,
        "dnh": dnh,
        "dhp": dhp,
    }

    metrics_dict["N0"] = np.sum(solution[0, :])
    metrics_dict["Rt"] = rt.calculate_Rt(
        params, solution[:, 0], population_size
    )

    return copy.deepcopy(metrics_dict)


def get_loglikelihood(metrics_dict, data_dict, weights_dict, population_size):
    ll_dict = {}
    run_total = 0

    for metric in data_dict:
        ll_dict[metric] = (
            0.5
            * weights_dict[metric]
            * sum(
                [
                    (s - d) ** 2
                    for s, d in zip(
                        metrics_dict[metric],
                        data_dict[metric]["data"],
                    )
                ]
            )
            / len(metrics_dict[metric])
        )
        run_total += ll_dict[metric]

    ll_dict["Rt"] = (
        0.5
        * weights_dict["Rt"]
        * (RT_BOUNDS[1] - metrics_dict["Rt"][0])
        * (metrics_dict["Rt"][0] - RT_BOUNDS[0])
    )

    run_total -= ll_dict["Rt"]

    ll_dict["N0"] = (
        0.5
        * weights_dict["N0"]
        * (N0_BOUNDS[1] * population_size - metrics_dict["N0"])
        * (metrics_dict["N0"] - N0_BOUNDS[0] * population_size)
    )

    run_total += ll_dict["N0"]

    return run_total, ll_dict.copy()
