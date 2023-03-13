import parameter_estimation.hosp_remove_regression as hrr
import parameter_estimation.initial_conditions as ic
import parameter_estimation.initial_parameters as ip
import parameter_estimation.likelihood_weights as lw
import parameter_estimation.trisection_method as tm
import parameter_estimation.simulate as sim
import parameter_estimation.metrics_loglikelihoods as ml
import parameter_estimation.optimisation_method as om


def parameter_estimation(data_dict, population_size):
    rH = hrr.get_hosp_remove_coefficient(
        data_dict["dis"]["data"], data_dict["dhp"]["data"]
    )

    initial_params = (ip.INITIAL_PARAMETERS).copy()
    initial_params[6] = rH

    initial_conditions = ic.get_initial_conditions(
        data_dict, population_size, initial_params[2], initial_params[4]
    )

    T = data_dict["occ"]["days"][-1]

    weights_dict = lw.get_weights(
        initial_params,
        initial_conditions,
        population_size,
        T,
        data_dict,
        initial=True,
    )

    _, X, _ = sim.standard_ode_solver(
        initial_conditions, initial_params, population_size, T
    )

    metrics_dict = ml.get_metrics(X, initial_params, T, population_size)
    ll_value, _ = ml.get_loglikelihood(
        metrics_dict, data_dict, weights_dict, population_size
    )

    for _ in range(2):
        for idx, fac in zip([3, 1], [ip.I_FACTOR, ip.E_FACTOR]):
            weights_dict = lw.get_weights(
                initial_params,
                initial_conditions,
                population_size,
                T,
                data_dict,
                initial=True,
            )
            initial_conditions = tm.trisection_wrapper(
                idx,
                initial_conditions,
                initial_params,
                fac,
                T,
                population_size,
                data_dict,
                weights_dict,
            )

    _, X, _ = sim.standard_ode_solver(
        initial_conditions, initial_params, population_size, T
    )

    metrics_dict = ml.get_metrics(X, initial_params, T, population_size)
    ll_value, _ = ml.get_loglikelihood(
        metrics_dict, data_dict, weights_dict, population_size
    )

    weights_dict = lw.get_weights(
        initial_params,
        initial_conditions,
        population_size,
        T,
        data_dict,
        initial=True,
    )

    idxs = [[[0], None], [[4, 5], None], [None, [1, 2, 3]]]
    bounds = [
        [[initial_params[0] / ip.B_FACTOR, initial_params[0] * ip.B_FACTOR]],
        [ip.GI_BOUNDS, ip.GH_BOUNDS],
        [
            [
                initial_conditions[1] / ip.E_FACTOR,
                initial_conditions[1] * ip.E_FACTOR,
            ],
            [
                initial_conditions[2] / ip.U_FACTOR,
                initial_conditions[2] * ip.U_FACTOR,
            ],
            [
                initial_conditions[3] / ip.I_FACTOR,
                initial_conditions[3] * ip.I_FACTOR,
            ],
        ],
    ]

    params, initial_conditions = om.optimisation_method(
        idxs,
        bounds,
        initial_conditions,
        initial_params,
        T,
        population_size,
        data_dict,
        weights_dict,
        ll_value,
    )

    weights_dict = lw.get_weights(
        params,
        initial_conditions,
        population_size,
        T,
        data_dict,
    )
    _, X, _ = sim.standard_ode_solver(
        initial_conditions, params, population_size, T
    )

    metrics_dict = ml.get_metrics(X, params, T, population_size)
    ll_value, _ = ml.get_loglikelihood(
        metrics_dict, data_dict, weights_dict, population_size
    )

    idxs = [[[0], None], [[4, 5], None]]
    bounds = [
        [[params[0] / ip.B_FACTOR, params[0] * ip.B_FACTOR]],
        [ip.GI_BOUNDS, ip.GH_BOUNDS],
    ]

    params, initial_conditions = om.optimisation_method(
        idxs,
        bounds,
        initial_conditions,
        params,
        T,
        population_size,
        data_dict,
        weights_dict,
        ll_value,
    )

    weights_dict = lw.get_weights(
        params,
        initial_conditions,
        population_size,
        T,
        data_dict,
    )
    _, X, _ = sim.standard_ode_solver(
        initial_conditions, params, population_size, T
    )

    metrics_dict = ml.get_metrics(X, params, T, population_size)
    ll_value, _ = ml.get_loglikelihood(
        metrics_dict, data_dict, weights_dict, population_size
    )

    idxs = [[[7], None]]
    bounds = [[ip.MU_BOUNDS]]

    params, initial_conditions = om.optimisation_method(
        idxs,
        bounds,
        initial_conditions,
        params,
        T,
        population_size,
        data_dict,
        weights_dict,
        ll_value,
    )

    initial_conditions.extend([0.0, 0.0, 0.0, 0.0])

    return params, initial_conditions
