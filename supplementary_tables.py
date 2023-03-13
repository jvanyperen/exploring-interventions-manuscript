import numpy as np
import os
import json
from scipy.stats import norm

import do_nothing as dn
import supplementary_material as sm


def tables_1_2():
    regions = ["england", "northWest"]
    for region in regions:
        jname = f"outputs/table_4_{region}.json"

        if os.path.exists(jname):
            with open(jname, "r") as json_file:
                res_dict = json.load(json_file)
        else:
            R0s = np.arange(1.3, 2.1, 0.1)
            res_dict = {}

            for R0 in R0s:
                _, _, met_dict0 = dn.do_nothing_approach(region, R0)
                pN0 = met_dict0["pN"]
                _, _, met_dict1 = dn.do_nothing_abm_approach(region, R0)
                pN1 = met_dict1["pN"]

                res_dict[R0] = {
                    "SEIR-D": {
                        "beds": met_dict0["max_hosp"],
                        "beds_pN": met_dict0["max_hosp"] * pN0,
                        "peak": met_dict0["peak_beds"],
                    },
                    "Branching Model": {
                        "beds_mean": met_dict1["max_hosp_mean"],
                        "beds_lb": met_dict1["max_hosp_lb"],
                        "beds_ub": met_dict1["max_hosp_ub"],
                        "beds_mean_pN": met_dict1["max_hosp_mean"] * pN1,
                        "beds_lb_pN": met_dict1["max_hosp_lb"] * pN1,
                        "beds_ub_pN": met_dict1["max_hosp_ub"] * pN1,
                        "peak_mean": met_dict1["peak_beds_mean"],
                        "peak_lb": met_dict1["peak_beds_lb"],
                        "peak_ub": met_dict1["peak_beds_ub"],
                    },
                }

            with open(jname, "w") as json_file:
                json.dump(res_dict, json_file)

        print("-----------------")
        print(f"----{region}----")
        print("-----------------")
        for R0 in res_dict:
            print(f"R0 = {R0:.2f}")
            print("-----------------")
            for t in res_dict[R0]:
                print(t)
                try:
                    print(
                        f"max beds = {res_dict[R0][t]['beds']:,3f}, {res_dict[R0][t]['beds_pN']:.3f} (\%N)"
                    )
                    print(f"peak beds = {res_dict[R0][t]['peak']}")
                except KeyError:
                    print(
                        f"mean max beds = {res_dict[R0][t]['beds_mean']:.3f}, IRP = {res_dict[R0][t]['beds_ub'] - res_dict[R0][t]['beds_lb']:.3f}"
                    )
                    print(
                        f"mean max beds {res_dict[R0][t]['beds_mean_pN']:.3f} (\%N), range = ({res_dict[R0][t]['beds_lb_pN']:.3f}, {res_dict[R0][t]['beds_ub_pN']:.3f})"
                    )
                    print(
                        f"mean peak beds = {res_dict[R0][t]['peak_mean']}, range = ({res_dict[R0][t]['peak_lb']:.3f}, {res_dict[R0][t]['peak_ub']:.3f})"
                    )
                print("-----------------")


def table_3():
    region = "southEast"
    jname = f"outputs/rate_params_testing_nc_{region}.json"

    param_letters = ["E", "U"]
    param_idx = [2, 4]
    alp = 0.05

    if os.path.exists(jname):
        with open(jname, "r") as json_file:
            param_dist_dict = json.load(json_file)
    else:
        dts = [1.0, 0.5, 0.25]
        mc = 5
        param_dist_dict = {dt: {} for dt in dts}
        for dt in dts:
            param_dist_dict_dt, _ = sm.parameter_testing(region, 100, dt, mc)
            param_dist_dict[dt] = param_dist_dict_dt

        with open(jname, "w") as json_file:
            json.dump(param_dist_dict, json_file)

    for dt in param_dist_dict:
        print("----------------")
        print(f"dt = {dt} (day)")
        print("----------------")
        for _, (letter, p_idx) in enumerate(zip(param_letters, param_idx)):
            times = np.array(
                list(param_dist_dict[dt][letter].keys()), dtype=float
            )
            sorted_times = np.sort(times)
            times = np.arange(
                start=float(dt),
                stop=sorted_times[-1] + float(dt),
                step=float(dt),
            )

            try:
                counts = np.array(
                    [param_dist_dict[dt][letter].get(t, 0) for t in times],
                    dtype=float,
                )
                tot_counts = sum(counts)
                if np.sum(counts) == 0:
                    raise ZeroDivisionError
                counts = counts / (float(dt) * np.sum(counts))
            except ZeroDivisionError:
                counts = np.array(
                    [param_dist_dict[dt][letter].get(f"{t}", 0) for t in times],
                    dtype=float,
                )
                tot_counts = sum(counts)
                counts = counts / (float(dt) * np.sum(counts))

            p_mean = np.average(times, weights=counts)
            zscore = norm.ppf(1.0 - alp / 2.0)
            lb = p_mean * (1.0 - zscore / np.sqrt(tot_counts))
            ub = p_mean * (1.0 + zscore / np.sqrt(tot_counts))

            print(
                f"True value of gamma_{letter}^-1 = {1./param_dist_dict[dt]['params'][p_idx]:.3f}"
            )
            print(
                f"estimated mean = {p_mean:.3f}, approx CI = ({lb:.3f}, {ub:.3f})"
            )


def table_4():
    region = "southEast"
    jname = f"outputs/rate_params_testing_{region}.json"

    param_letters = ["E", "U"]
    param_idx = [2, 4]
    alp = 0.05

    if os.path.exists(jname):
        with open(jname, "r") as json_file:
            param_dist_dict = json.load(json_file)
    else:
        dts = [1.0, 0.5, 0.25]
        mc = 5
        param_dist_dict = {dt: {} for dt in dts}
        for dt in dts:
            param_dist_dict_dt, _ = sm.parameter_testing(region, 100, dt, mc)
            param_dist_dict[dt] = param_dist_dict_dt

        with open(jname, "w") as json_file:
            json.dump(param_dist_dict, json_file)

    for dt in param_dist_dict:
        print("----------------")
        print(f"dt = {dt} (day)")
        print("----------------")
        for _, (letter, p_idx) in enumerate(zip(param_letters, param_idx)):
            times = np.array(
                list(param_dist_dict[dt][letter].keys()), dtype=float
            )
            sorted_times = np.sort(times)
            times = np.arange(
                start=float(dt),
                stop=sorted_times[-1] + float(dt),
                step=float(dt),
            )

            try:
                counts = np.array(
                    [param_dist_dict[dt][letter].get(t, 0) for t in times],
                    dtype=float,
                )
                tot_counts = sum(counts)
                if np.sum(counts) == 0:
                    raise ZeroDivisionError
                counts = counts / (float(dt) * np.sum(counts))
            except ZeroDivisionError:
                counts = np.array(
                    [param_dist_dict[dt][letter].get(f"{t}", 0) for t in times],
                    dtype=float,
                )
                tot_counts = sum(counts)
                counts = counts / (float(dt) * np.sum(counts))

            p_mean = np.average(times, weights=counts)
            zscore = norm.ppf(1.0 - alp / 2.0)
            lb = p_mean * (1.0 - zscore / np.sqrt(tot_counts))
            ub = p_mean * (1.0 + zscore / np.sqrt(tot_counts))

            print(
                f"True value of gamma_{letter}^-1 = {1./param_dist_dict[dt]['params'][p_idx]:.3f}"
            )
            print(
                f"estimated mean = {p_mean:.3f}, approx CI = ({lb:.3f}, {ub:.3f})"
            )


def table_5():
    region = "southEast"
    jname = f"outputs/prob_params_testing_{region}.json"
    dt = 0.25
    if os.path.exists(jname):
        with open(jname, "r") as json_file:
            res_dict = json.load(json_file)
    else:
        mc_sims = [1, 5, 10, 20]
        res_dict = {mc: {} for mc in mc_sims}
        for mc in mc_sims:
            _, prob_dist_dict_mc = sm.parameter_testing(
                "southEast", 100, dt, mc
            )
            res_dict[mc] = prob_dist_dict_mc

        with open(jname, "w") as json_file:
            json.dump(res_dict, json_file)

    for mc in res_dict:
        print("----------------")
        print(f"Number of MC sims = {mc}")
        print("----------------")
        print(
            f"true p = {res_dict[mc]['params'][3]:.4f}, mean p = {res_dict[mc]['p_mean']:.4f}, PR = ({res_dict[mc]['p_lb']:.4f}, {res_dict[mc]['p_ub']:.4f})"
        )
        print(
            f"true mU = {res_dict[mc]['params'][8]:.4f}, mean m_U = {res_dict[mc]['mU_mean']:.4f}, PR = ({res_dict[mc]['mU_lb']:.4f}, {res_dict[mc]['mU_ub']:.4f})"
        )
        mH = res_dict[mc]["params"][7] / (1.0 + res_dict[mc]["params"][7])
        print(
            f"true mH = {mH:.4f}, mean m_H = {res_dict[mc]['mH_mean']:.4f}, PR = ({res_dict[mc]['mH_lb']:.4f}, {res_dict[mc]['mH_ub']:.4f})"
        )


if __name__ == "__main__":
    # tables_1_2()
    # table_3()
    # table_4()
    # table_5()
