import numpy as np
import os
import json

import run_parameter_estimation as rp
import do_nothing as dn


def table_1():
    regions = ["england", "northWest", "southEast"]
    for region in regions:
        print("-----------------")
        print(f"----{region}----")
        print("-----------------")
        rp.parameter_information(
            region, parameters_info=True, initial_conditions_info=False
        )


def table_2():
    regions = ["england", "northWest", "southEast"]
    for region in regions:
        print("-----------------")
        print(f"----{region}----")
        print("-----------------")
        rp.parameter_information(
            region, parameters_info=False, initial_conditions_info=True
        )


def table_3():
    region = "southEast"
    jname = f"outputs/table_3_{region}.json"

    if os.path.exists(jname):
        with open(jname, "r") as json_file:
            res_dict = json.load(json_file)
    else:
        R0s = np.arange(1.3, 2.1, 0.1)
        res_dict = {}

        for R0 in R0s:
            _, _, met_dict0 = dn.do_nothing_abm_approach(region, R0, scale=0)
            pN0 = met_dict0["pN"]
            _, _, met_dict1 = dn.do_nothing_abm_approach(region, R0, scale=1)
            pN1 = met_dict1["pN"]

            res_dict[R0] = {
                "a": {
                    "mean": met_dict0["dead_mean"],
                    "lb": met_dict0["dead_lb"],
                    "ub": met_dict0["dead_ub"],
                    "mean_pN": met_dict0["dead_mean"] * pN0,
                    "lb_pN": met_dict0["dead_lb"] * pN0,
                    "ub_pN": met_dict0["dead_ub"] * pN0,
                },
                "C": {
                    "mean": met_dict1["dead_mean"],
                    "lb": met_dict1["dead_lb"],
                    "ub": met_dict1["dead_ub"],
                    "mean_pN": met_dict1["dead_mean"] * pN1,
                    "lb_pN": met_dict1["dead_lb"] * pN1,
                    "ub_pN": met_dict1["dead_ub"] * pN1,
                },
            }

        with open(jname, "r") as json_file:
            json.dump(res_dict, json_file)

    print("-----------------")
    print(f"----{region}----")
    print("-----------------")
    for R0 in res_dict:
        print(f"R0 = {R0:.2f}")
        print("-----------------")
        for t in res_dict[R0]:
            print(f"fix {t}")

            print(
                f"mean dead = {res_dict[R0][t]['mean']}, IPR = {res_dict[R0][t]['ub'] - res_dict[R0][t]['lb']:.3f}"
            )
            print(
                f"mean dead = {res_dict[R0][t]['mean_pN']:.3f} (%N), range = ({res_dict[R0][t]['lb_pN']:.3f} , {res_dict[R0][t]['ub_pN']:.3f})"
            )
            print("-----------------")


def table_4():
    region = "southEast"
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
                    "beds": met_dict0["max_hosp_mean"],
                    "beds_lb": met_dict0["max_hosp_lb"],
                    "beds_ub": met_dict0["max_hosp_ub"],
                    "beds_pN": met_dict0["max_hosp_mean"] * pN1,
                    "beds_lb_pN": met_dict0["max_hosp_lb"] * pN1,
                    "beds_ub_pN": met_dict0["max_hosp_ub"] * pN1,
                    "peak": met_dict0["peak_beds_mean"],
                    "peak_lb": met_dict0["peak_beds_lb"],
                    "peak_ub": met_dict0["peak_beds_ub"],
                },
            }

        with open(jname, "r") as json_file:
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
                    f"max beds = {res_dict[R0][t]['beds']}, {res_dict[R0][t]['beds_pN']} (\%N)"
                )
                print(f"peak beds = {res_dict[R0][t]['peak']}")
            except KeyError:
                print(
                    f"mean max beds = {res_dict[R0][t]['beds_mean']}, IRP = {res_dict[R0][t]['beds_ub'] - res_dict[R0][t]['beds_lb']:.3f}"
                )
                print(
                    f"mean max beds {res_dict[R0][t]['beds_mean_pN']} (\%N), range = ({res_dict[R0][t]['beds_lb_pN']:.3f}, {res_dict[R0][t]['beds_ub_pN']:.3f})"
                )
                print(
                    f"mean peak beds = {res_dict[R0][t]['peak_mean']}, range = ({res_dict[R0][t]['peak_lb']:.3f}, {res_dict[R0][t]['peak_ub']:.3f})"
                )
            print("-----------------")


if __name__ == "__main__":
    # table_1()
    # table_2()
    # table_3()
    # table_4()
