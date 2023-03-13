import numpy as np

import parameter_estimation.simulate as sim
import parameter_estimation.metrics_loglikelihoods as ml


def weights_distribution(initial=False):
    # keys are levels (level 1 -> 0.1 * size in level 0)
    # entries are which data to put in each level
    if initial:
        weights_user_data = {
            0: ["adm"],
            1: ["dis", "dhp"],
            2: ["dnh", "Rt", "N0"],
            3: ["occ"],
        }
    else:
        weights_user_data = {
            0: ["occ"],
            1: ["adm"],
            2: ["dis", "dnh", "Rt", "N0"],
            3: ["dhp"],
        }

    return weights_user_data


def get_weights(
    params, initial_conditions, population_size, T, data_dict, initial=False
):
    weights_dist = weights_distribution(initial=initial)
    weights_dict = initialise_weights(weights_dist)

    _, X, _ = sim.standard_ode_solver(
        initial_conditions, params, population_size, T
    )
    metrics_dict = ml.get_metrics(X, params, T, population_size)
    _, ll_dict = ml.get_loglikelihood(
        metrics_dict, data_dict, weights_dict, population_size
    )

    return get_adaptive_weights(ll_dict, weights_dict, weights_dist)


def initialise_weights(weights_dist):
    weights_dict = {}
    for _, metrics in weights_dist.items():
        for metric in metrics:
            weights_dict[metric] = 1.0

    return weights_dict.copy()


def get_adaptive_weights(ll_dict, weights_dict, weights_dist):
    weights_dict = weights_dict.copy()
    standard_order = np.amax(
        [get_magnitude(ll_dict[metric]) for metric in weights_dist[0]]
    )

    for level, metrics in weights_dist.items():
        for metric in metrics:
            order = get_magnitude(ll_dict[metric])
            weights_dict[metric] *= 10.0 ** (standard_order - level - order)

    return weights_dict.copy()


def get_magnitude(x):
    return np.round(np.log10(np.abs(x)))
