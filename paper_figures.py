import warnings
import numpy as np
import json
import os.path
from tqdm import tqdm

import standard_forecast as sf
import run_parameter_estimation as rp
import parameter_estimation.get_data as gd
import do_nothing as dn
import interventions as ins
import capacity_thresholds as ct
import plotting_scripts.forecast_plots as fp
import plotting_scripts.do_nothing_plots as dnp
import plotting_scripts.interventions_plots as inp
import plotting_scripts.capacity_plots as cp


def figure_3():
    eng_dict = sf.standard_forecast("england")
    NW_dict = sf.standard_forecast("northWest")
    SE_dict = sf.standard_forecast("southEast")
    region_dicts = {
        "England": eng_dict,
        "North West": NW_dict,
        "South East": SE_dict,
    }

    dates = [rp.START_DATE, rp.END_DATE]
    fname = "parameter_estimation/data_management/england.csv"
    eng_data_dict, *_ = gd.form_data_dictionary(fname, dates[0], dates[1])
    fname = "parameter_estimation/data_management/northWest.csv"
    NW_data_dict, *_ = gd.form_data_dictionary(fname, dates[0], dates[1])
    fname = "parameter_estimation/data_management/southEast.csv"
    SE_data_dict, *_ = gd.form_data_dictionary(fname, dates[0], dates[1])
    data_dicts = {
        "England": eng_data_dict,
        "North West": NW_data_dict,
        "South East": SE_data_dict,
    }

    fname = "figures/beds_occupied.png"

    fp.beds_from_fits_data_plot(region_dicts, data_dicts, dates, fname=fname)


def figure_4():
    eng_dict = sf.standard_forecast("england")
    NW_dict = sf.standard_forecast("northWest")
    SE_dict = sf.standard_forecast("southEast")
    region_dicts = {"England": eng_dict, "NW": NW_dict, "SE": SE_dict}

    dates = [rp.START_DATE, rp.END_DATE]

    fname = "figures/prop_beds.png"

    fp.prop_beds_from_fit_plot(region_dicts, dates, fname=fname)


def figure_5():
    SE_dict = sf.standard_forecast("southEast")
    SE_dict["HpN"] = SE_dict["H"] * SE_dict["pN"]

    SE_eng_dict = sf.forecast_wrong_parameters("southEast", "england")
    SE_eng_dict["HpN"] = SE_eng_dict["H"] * SE_eng_dict["pN"]

    SE_NW_dict = sf.forecast_wrong_parameters("southEast", "northWest")
    SE_NW_dict["HpN"] = SE_NW_dict["H"] * SE_NW_dict["pN"]

    SE_eng_diff = np.amax(
        np.abs(SE_dict["H"][: len(SE_eng_dict["H"])] - SE_eng_dict["H"])
    )
    SE_NW_diff = np.amax(np.abs(SE_dict["H"] - SE_NW_dict["H"]))

    print(f"Max diff in beds occupied using England params = {SE_eng_diff:.1f}")
    print(f"Max diff in beds occupied using NW params = {SE_NW_diff:.1f}")

    region_dict = {
        "SE": SE_dict,
        "SE + Eng": SE_eng_dict,
        "SE + NW": SE_NW_dict,
    }

    fname = "figures/beds_wrong_params.png"
    fp.beds_wrong_params(region_dict, fname=fname)


def figure_6():
    region = "southEast"
    jname = f"outputs/beds_occupied_branch_{region}.json"

    if os.path.exists(jname):
        with open(jname, "r") as json_file:
            res_dict = json.load(json_file)
            EBM_dict = res_dict["EBM"]
            ABM_dict = res_dict["ABM"]
    else:
        region_dict = sf.standard_forecast(region)
        EBM_dict = {
            key: list(region_dict[key]) for key in region_dict if key != "pN"
        }

        dts = [1.0, 0.5, 0.25]
        ABM_dict = {dt: {} for dt in dts}
        for dt in dts:
            branch_dict = sf.standard_forecast_branch(region, dt=dt)
            ABM_dict[dt] = {
                key: list(branch_dict[key])
                for key in branch_dict
                if key != "pN"
            }

        save_dict = {"EBM": EBM_dict, "ABM": ABM_dict}

        with open(jname, "w") as json_file:
            json.dump(save_dict, json_file)

    fname = f"figures/beds_occupied_branch_{region}.png"
    fp.compare_abms_plot(EBM_dict, ABM_dict, fname=fname)


def figure_7():
    region = "southEast"
    jname = f"outputs/beds_occupied_LoS_{region}.json"

    if os.path.exists(jname):
        with open(jname, "r") as json_file:
            res_dict = json.load(json_file)
            EBM_dict = res_dict["EBM"]
            ABM_dict = res_dict["ABM"]
    else:
        region_dict = sf.standard_forecast(region)
        EBM_dict = {
            key: list(region_dict[key]) for key in region_dict if key != "pN"
        }

        dts = [1.0, 0.5, 0.25]
        ABM_dict = {dt: {} for dt in dts}
        for dt in dts:
            branch_dict = sf.standard_forecast_LoS(region, dt=dt)
            ABM_dict[dt] = {
                key: list(branch_dict[key])
                for key in branch_dict
                if key != "pN"
            }

        save_dict = {"EBM": EBM_dict, "ABM": ABM_dict}

        with open(jname, "w") as json_file:
            json.dump(save_dict, json_file)

    fname = f"figures/beds_occupied_LoS_{region}.png"
    fp.compare_abms_plot(EBM_dict, ABM_dict, fname=fname)


def figure_8():
    R0s = np.arange(1.3, 2.1, 0.1)

    Rt_dict = {R0: {} for R0 in R0s}
    for R0 in R0s:
        _, rt_dict, _ = dn.do_nothing_approach("england", R0)
        Rt_dict[R0] = rt_dict

    fname = "figures/donoRt.png"
    dnp.do_nothing_Rt_plot(Rt_dict, fname=fname)


def figure_9():
    R0s = np.arange(1.3, 2.1, 0.2)
    regions = ["england", "northWest", "southEast"]
    r_names = ["England", "NW", "SE"]
    region_dict = {R0: {} for R0 in R0s}
    for R0 in R0s:
        for region, rN in zip(regions, r_names):
            H_dict, _, _ = dn.do_nothing_approach(region, R0)
            region_dict[R0][rN] = H_dict

    fname = "figures/donoH.png"
    dnp.do_nothing_hospital_plot(region_dict, fname=fname)


def figure_10():
    jname = "outputs/donoD.json"
    if os.path.exists(jname):
        with open(jname, "r") as json_file:
            sim_dicts = json.load(json_file)
            region_dict = sim_dicts["EB"]
            region_abm_dict = sim_dicts["BM"]
    else:
        R0s = np.arange(1.3, 2.01, 0.01)
        regions = ["england", "northWest", "southEast"]
        r_names = ["England", "NW", "SE"]
        D_arr = np.zeros([len(R0s), len(regions)])

        for idx, R0 in enumerate(R0s):
            for ridx, region in enumerate(regions):
                _, _, met_dict = dn.do_nothing_approach(region, R0)
                D_arr[idx, ridx] = met_dict["dead"] * met_dict["pN"]

        region_dict = {
            rN: {"R0": list(R0s), "D": list(D_arr[:, idx])}
            for idx, rN in enumerate(r_names)
        }

        R0s = np.arange(1.3, 2.1, 0.1)
        D_mean_arr = np.zeros([len(R0s), len(regions)])
        D_lb_arr = np.zeros([len(R0s), len(regions)])
        D_ub_arr = np.zeros([len(R0s), len(regions)])

        for idx, R0 in enumerate(R0s):
            for ridx, region in enumerate(regions):
                _, _, met_dict = dn.do_nothing_abm_approach(region, R0)
                D_mean_arr[idx, ridx] = met_dict["dead_mean"] * met_dict["pN"]
                D_lb_arr[idx, ridx] = met_dict["dead_lb"] * met_dict["pN"]
                D_ub_arr[idx, ridx] = met_dict["dead_ub"] * met_dict["pN"]

        region_abm_dict = {
            rN: {
                "R0": list(R0s),
                "D_mean": list(D_mean_arr[:, idx]),
                "D_lb": list(D_lb_arr[:, idx]),
                "D_ub": list(D_ub_arr[:, idx]),
            }
            for idx, rN in enumerate(r_names)
        }

        json_dict = {"EB": region_dict, "BM": region_abm_dict}
        with open(jname, "w") as json_obj:
            json.dump(json_dict, json_obj)

    fname = "figures/donoD.png"
    dnp.do_nothing_deaths_plot(region_dict, region_abm_dict, fname=fname)


def figure_11():
    R0s = np.arange(1.3, 2.1, 0.2)
    regions = ["england", "northWest", "southEast"]
    r_names = ["England", "NW", "SE"]
    region_dict = {R0: {} for R0 in R0s}
    for R0 in R0s:
        for region, rN in zip(regions, r_names):
            H_dict, _, _ = ins.hospital_intervention_R0_approach(
                region, R0, Hlfac=0.25
            )
            region_dict[R0][rN] = H_dict

    fname = "figures/changing_r0_all_regions.png"
    inp.intervention_hospital_plot_all(region_dict, fname=fname)


def figure_12():
    region = "southEast"
    jname = f"outputs/changing_r0_D_{region}.json"
    if os.path.exists(jname):
        with open(jname, "r") as json_file:
            sim_dicts = json.load(json_file)
            Hl_dict = sim_dicts["EB"]
            Hl_abm_dict = sim_dicts["BM"]
    else:
        R0s = np.arange(1.3, 2.0005, 0.0005)

        Hls = [0.25, 0.5]
        D_arr = np.zeros([len(R0s), len(Hls)])

        for idx, R0 in enumerate(tqdm(R0s, "SEIR-D")):
            for hidx, Hl in enumerate(Hls):
                _, _, met_dict = ins.hospital_intervention_R0_approach(
                    region, R0, Hlfac=Hl
                )
                D_arr[idx, hidx] = met_dict["dead"] * met_dict["pN"]

        Hl_dict = {
            Hl: {
                "R0": list(R0s),
                "D": list(D_arr[:, idx]),
            }
            for idx, Hl in enumerate(Hls)
        }

        R0s = np.arange(1.3, 2.1, 0.1)
        dono_D_arr = np.zeros(len(R0s))

        for idx, R0 in enumerate(tqdm(R0s, "dono")):
            _, _, met_dict = dn.do_nothing_approach(region, R0)
            dono_D_arr[idx] = met_dict["dead"] * met_dict["pN"]

        Hl_dict["dono"] = {"R0": list(R0s), "D": list(dono_D_arr)}

        R0s = np.arange(1.3, 2.01, 0.01)
        D_mean_arr = np.zeros([len(R0s), len(Hls)])
        D_lb_arr = np.zeros([len(R0s), len(Hls)])
        D_ub_arr = np.zeros([len(R0s), len(Hls)])

        for idx, R0 in enumerate(tqdm(R0s, desc="MC")):
            for hidx, Hl in enumerate(Hls):
                _, _, met_dict = ins.hospital_intervention_R0_abm_approach(
                    region, R0, Hlfac=Hl
                )
                D_mean_arr[idx, hidx] = met_dict["dead_mean"] * met_dict["pN"]
                D_lb_arr[idx, hidx] = met_dict["dead_lb"] * met_dict["pN"]
                D_ub_arr[idx, hidx] = met_dict["dead_ub"] * met_dict["pN"]

        Hl_abm_dict = {
            Hl: {
                "R0": list(R0s),
                "D_mean": list(D_mean_arr[:, idx]),
                "D_lb": list(D_lb_arr[:, idx]),
                "D_ub": list(D_ub_arr[:, idx]),
            }
            for idx, Hl in enumerate(Hls)
        }

        json_dict = {"EB": Hl_dict, "BM": Hl_abm_dict}
        with open(jname, "w") as json_obj:
            json.dump(json_dict, json_obj)

    fname = f"figures/changing_r0_D_{region}.png"
    inp.intervention_death_plot(Hl_dict, Hl_abm_dict, fname=fname)


def figure_13():
    R0s = [1.485, 1.486, 1.488, 1.51]
    region = "southEast"
    R0s_dict = {R0: {} for R0 in R0s}
    Hlfac = 0.5
    Hufac = 0.0012
    for _, R0 in enumerate(R0s):
        H_dict, _, _ = ins.hospital_intervention_R0_approach(
            region, R0, Hlfac=Hlfac, Hufac=Hufac
        )

        R0s_dict[R0] = H_dict

    fname = f"figures/changing_r0_{region}.png"
    inp.intervention_hospital_plot(R0s_dict, fname=fname)


def figure_14():
    R0s = [1.485, 1.486, 1.488, 1.51]
    region = "southEast"
    R0s_dict = {R0: {} for R0 in R0s}
    Hlfac = 0.5
    Hufac = 0.0012
    for _, R0 in enumerate(R0s):
        _, Rt_dict, _ = ins.hospital_intervention_R0_approach(
            region, R0, Hlfac=Hlfac, Hufac=Hufac
        )

        R0s_dict[R0] = Rt_dict

    fname = f"figures/changing_r0_rt_{region}.png"
    inp.intervention_hospital_plot_rt(R0s_dict, fname=fname)


def figure_15():
    region = "southEast"
    jname = f"outputs/changing_Hl_D_{region}.json"
    if os.path.exists(jname):
        with open(jname, "r") as json_file:
            sim_dicts = json.load(json_file)
            Hl_dict = sim_dicts["EB"]
            Hl_abm_dict = sim_dicts["BM"]
    else:
        R0s = [1.5]

        Hls = np.arange(0.025, 0.756, 0.001)
        D_arr = np.zeros([len(Hls), len(R0s)])

        for idx, Hl in enumerate(tqdm(Hls, "SEIR-D")):
            for ridx, R0 in enumerate(R0s):
                _, _, met_dict = ins.hospital_intervention_R0_approach(
                    region, R0, Hlfac=Hl
                )
                D_arr[idx, ridx] = met_dict["dead"] * met_dict["pN"]

        Hl_dict = {
            R0: {
                "Hl": list(Hls),
                "D": list(D_arr[:, idx]),
            }
            for idx, R0 in enumerate(R0s)
        }

        dono_D_arr = np.zeros(len(R0s))

        for idx, R0 in enumerate(R0s):
            _, _, met_dict = dn.do_nothing_approach(region, R0)
            dono_D_arr[idx] = met_dict["dead"] * met_dict["pN"]

        Hl_dict["dono"] = {"D": list(dono_D_arr)}

        Hls = np.arange(0.025, 0.76, 0.01)
        D_mean_arr = np.zeros([len(Hls), len(R0s)])
        D_lb_arr = np.zeros([len(Hls), len(R0s)])
        D_ub_arr = np.zeros([len(Hls), len(R0s)])

        for idx, Hl in enumerate(tqdm(Hls, "MC")):
            for ridx, R0 in enumerate(R0s):
                _, _, met_dict = ins.hospital_intervention_R0_abm_approach(
                    region, R0, Hlfac=Hl
                )
                D_mean_arr[idx, ridx] = met_dict["dead_mean"] * met_dict["pN"]
                D_lb_arr[idx, ridx] = met_dict["dead_lb"] * met_dict["pN"]
                D_ub_arr[idx, ridx] = met_dict["dead_ub"] * met_dict["pN"]

        Hl_abm_dict = {
            R0: {
                "Hl": list(Hls),
                "D_mean": list(D_mean_arr[:, idx]),
                "D_lb": list(D_lb_arr[:, idx]),
                "D_ub": list(D_ub_arr[:, idx]),
            }
            for idx, R0 in enumerate(R0s)
        }

        json_dict = {"EB": Hl_dict, "BM": Hl_abm_dict}
        with open(jname, "w") as json_obj:
            json.dump(json_dict, json_obj)

    fname = f"figures/changing_Hl_D_{region}.png"
    inp.intervention_death_Hl_plot(Hl_dict, Hl_abm_dict, fname=fname)


def figure_16():
    Hls = [0.4, 0.45]
    region = "southEast"
    Hls_dict = {Hl: {} for Hl in Hls}
    R0 = 1.5
    Hufac = 0.0012
    for _, Hl in enumerate(Hls):
        H_dict, _, _ = ins.hospital_intervention_R0_approach(
            region, R0, Hlfac=Hl, Hufac=Hufac
        )

        Hls_dict[Hl] = H_dict

    fname = f"figures/changing_Hl_{region}.png"
    inp.intervention_hospital_Hl_plot(Hls_dict, fname=fname)


def figure_17():
    R0s = [1.3, 1.5]
    Hmaxfac = 0.0012
    Hufacs = np.linspace(0.00075, 0.0015)
    max_H = np.zeros([len(R0s), len(Hufacs)])
    max_HpN = np.zeros([len(R0s), len(Hufacs)])
    region = "southEast"
    for idx, R0 in enumerate(R0s):
        for hidx, Hufac in enumerate(tqdm(Hufacs, "Hu")):
            H_dict = ct.explore_objective(region, R0, Hufac)
            max_H[idx, hidx] = H_dict["max_H"]
            max_HpN[idx, hidx] = H_dict["max_H"] * H_dict["pN"]

    R0s_dict = {
        R0: {
            "Hu": Hufacs * 100.0 / H_dict["pN"],
            "HupN": Hufacs * 100.0,
            "max_H": max_H[idx, :],
            "max_HpN": max_HpN[idx, :],
        }
        for idx, R0 in enumerate(R0s)
    }

    R0s_dict["Hmaxfac"] = Hmaxfac * 100.0

    fname = "figures/capacity_func.png"
    cp.explore_objective_plot(R0s_dict, fname=fname)


def figure_18():
    jname = "outputs/capacity_thresholds.json"
    if os.path.exists(jname):
        with open(jname, "r") as json_file:
            region_dict = json.load(json_file)
    else:
        regions = ["england", "northWest", "southEast"]
        r_names = ["England", "NW", "SE"]
        R0s = np.linspace(1.3, 2.0, num=36)
        HupN = np.zeros([len(regions), len(R0s)])
        HupHmax = np.zeros([len(regions), len(R0s)])
        region_dict = {rn: {} for rn in r_names}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # warnings from root solver clogs terminal outputs
            for idx, (region, rn) in enumerate(zip(regions, r_names)):
                for ridx, R0 in enumerate(tqdm(R0s, region)):
                    res_dict = ct.minimise_objective(region, R0)
                    HupN[idx, ridx] = res_dict["Hu"] * res_dict["pN"]
                    HupHmax[idx, ridx] = res_dict["HupHmax"]

                region_dict[rn] = {
                    "R0": list(R0s),
                    "HupN": list(HupN[idx, :]),
                    "HupHmax": list(HupHmax[idx, :]),
                }

        with open(jname, "w") as json_file:
            json.dump(region_dict, json_file)

    fname = "figures/opt_capacity.png"
    cp.mimin_objective_plot(region_dict, fname=fname)


def figure_19():
    region = "southEast"
    jname = f"outputs/num_breaches_Hmax_{region}.json"
    if os.path.exists(jname):
        with open(jname, "r") as json_file:
            region_dict = json.load(json_file)
    else:
        R0s = [1.3, 1.5, 1.7, 1.9]
        Hufacs = np.arange(0.94, 1.04 + 0.005, 0.005)
        trials = 50
        lb = 0.01
        ub = 0.99
        mc_trials = 100

        number_breaches_mean = np.zeros([len(R0s), len(Hufacs)])
        number_breaches_lb = np.zeros([len(R0s), len(Hufacs)])
        number_breaches_ub = np.zeros([len(R0s), len(Hufacs)])

        HupHmax = np.zeros([len(R0s), len(Hufacs)])
        Hu = np.zeros(len(R0s))

        for idx, R0 in enumerate(R0s):
            for Hidx, Hufac in enumerate(tqdm(Hufacs, f"R0 = {R0:.1f}, Hus")):
                num_breaches = np.zeros(trials)
                for t in range(trials):
                    breaching_dict = ct.capacity_thresholds_abm(
                        region, R0, Hup=Hufac, mc_trials=mc_trials
                    )
                    num_breaches[t] = breaching_dict["num"]

                number_breaches_mean[idx, Hidx] = np.mean(num_breaches)
                number_breaches_lb[idx, Hidx] = np.quantile(num_breaches, lb)
                number_breaches_ub[idx, Hidx] = np.quantile(num_breaches, ub)
                HupHmax[idx, Hidx] = Hufac * breaching_dict["Hu"]
                Hu[idx] = breaching_dict["Hu"]

        region_dict = {
            R0: {
                "num_mean": list(number_breaches_mean[idx, :]),
                "num_lb": list(number_breaches_lb[idx, :]),
                "num_ub": list(number_breaches_ub[idx, :]),
                "HupHmax": list(HupHmax[idx, :] * 100.0),
                "Hu": Hu[idx] * 100.0,
            }
            for idx, R0 in enumerate(R0s)
        }

        with open(jname, "w") as json_file:
            json.dump(region_dict, json_file)

    fname = f"figures/num_breaches_Hmax_{region}.png"
    cp.number_breaches_Hmax_plot(region_dict, fname=fname)


def figure_20():
    region = "southEast"
    jname = f"outputs/num_breaches_Hu_{region}.json"
    if os.path.exists(jname):
        with open(jname, "r") as json_file:
            region_dict = json.load(json_file)
    else:
        R0s = [1.3, 1.5, 1.7, 1.9]
        Hufacs = np.arange(0.94, 1.04 + 0.005, 0.005)

        number_breaches = np.zeros([len(R0s), len(Hufacs)])

        for idx, R0 in enumerate(R0s):
            for Hidx, Hufac in enumerate(tqdm(Hufacs, f"R0 = {R0:.1f}, Hus")):
                breaching_dict = ct.capacity_thresholds_abm(
                    region, R0, Hup=Hufac
                )
                number_breaches[idx, Hidx] = breaching_dict["num"]

        region_dict = {
            R0: {
                "num": list(number_breaches[idx, :]),
                "Hufacs": list(Hufacs * 100.0),
            }
            for idx, R0 in enumerate(R0s)
        }

        with open(jname, "w") as json_file:
            json.dump(region_dict, json_file)

    fname = f"figures/num_breaches_Hu_{region}.png"
    cp.number_breaches_Hu_plot(region_dict, fname=fname)


def figure_21():
    region = "southEast"
    jname = f"outputs/diff_breach_{region}.json"
    if os.path.exists(jname):
        with open(jname, "r") as json_file:
            region_dict = json.load(json_file)
    else:
        R0s = [1.3, 1.5, 1.7, 1.9]
        Hufacs = np.arange(0.94, 1.04 + 0.005, 0.005)

        diff_breach_mean = np.zeros([len(R0s), len(Hufacs)])
        diff_breach_lb = np.zeros([len(R0s), len(Hufacs)])
        diff_breach_ub = np.zeros([len(R0s), len(Hufacs)])

        for idx, R0 in enumerate(R0s):
            for Hidx, Hufac in enumerate(tqdm(Hufacs, f"R0 = {R0:.1f}, Hus")):
                breaching_dict = ct.capacity_thresholds_abm(
                    region, R0, Hup=Hufac
                )
                diff_breach_mean[idx, Hidx] = breaching_dict["diff_from_mean"]
                diff_breach_lb[idx, Hidx] = breaching_dict["diff_from_lb"]
                diff_breach_ub[idx, Hidx] = breaching_dict["diff_from_ub"]

        region_dict = {
            R0: {
                "mean": list(diff_breach_mean[idx, :]),
                "lb": list(diff_breach_lb[idx, :]),
                "ub": list(diff_breach_ub[idx, :]),
                "Hufacs": list(Hufacs * 100.0),
            }
            for idx, R0 in enumerate(R0s)
        }

        with open(jname, "w") as json_file:
            json.dump(region_dict, json_file)

    fname = f"figures/diff_breach_{region}.png"
    title = r"$\mathcal{M}_{diff}$ (\%)"
    cp.metrics_plot(region_dict, title, fname=fname)


def figure_22():
    region = "southEast"
    jname = f"outputs/max_breach_{region}.json"
    if os.path.exists(jname):
        with open(jname, "r") as json_file:
            region_dict = json.load(json_file)
    else:
        R0s = [1.3, 1.5, 1.7, 1.9]
        Hufacs = np.arange(0.94, 1.04 + 0.005, 0.005)

        max_breach_mean = np.zeros([len(R0s), len(Hufacs)])
        max_breach_lb = np.zeros([len(R0s), len(Hufacs)])
        max_breach_ub = np.zeros([len(R0s), len(Hufacs)])

        for idx, R0 in enumerate(R0s):
            for Hidx, Hufac in enumerate(tqdm(Hufacs, f"R0 = {R0:.1f}, Hus")):
                breaching_dict = ct.capacity_thresholds_abm(
                    region, R0, Hup=Hufac
                )
                max_breach_mean[idx, Hidx] = breaching_dict["max_over_mean"]
                max_breach_lb[idx, Hidx] = breaching_dict["max_over_lb"]
                max_breach_ub[idx, Hidx] = breaching_dict["max_over_ub"]

        region_dict = {
            R0: {
                "mean": list(max_breach_mean[idx, :]),
                "lb": list(max_breach_lb[idx, :]),
                "ub": list(max_breach_ub[idx, :]),
                "Hufacs": list(Hufacs * 100.0),
            }
            for idx, R0 in enumerate(R0s)
        }

        with open(jname, "w") as json_file:
            json.dump(region_dict, json_file)

    fname = f"figures/max_breach_{region}.png"
    title = r"$\mathcal{M}_{max}$ (\%)"
    cp.metrics_plot(region_dict, title, fname=fname)


if __name__ == "__main__":
    # figure_3()
    # figure_4()
    # figure_5()
    # figure_6()
    # figure_7()
    # figure_8()
    # figure_9()
    # figure_10()
    # figure_11()
    # figure_12()
    # figure_13()
    # figure_14()
    # figure_15()
    # figure_16()
    # figure_17()
    # figure_18()
    # figure_19()
    # figure_20()
    # figure_21()
    # figure_22()
    
