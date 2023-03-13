import numpy as np
import json
import os.path
from tqdm import tqdm

import standard_forecast as sf
import do_nothing as dn
import interventions as ins
import supplementary_material as sm
import capacity_thresholds as ct
import plotting_scripts.capacity_plots as cp
import plotting_scripts.forecast_plots as fp
import plotting_scripts.interventions_plots as inp
import plotting_scripts.supplementary_plots as sp


def figures_1_2():
    regions = ["england", "northWest"]
    for region in regions:
        jname = f"outputs/beds_occupied_branch_{region}.json"

        if os.path.exists(jname):
            with open(jname, "r") as json_file:
                res_dict = json.load(json_file)
                EBM_dict = res_dict["EBM"]
                ABM_dict = res_dict["ABM"]
        else:
            region_dict = sf.standard_forecast(region)
            EBM_dict = {
                key: list(region_dict[key])
                for key in region_dict
                if key != "pN"
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


def figures_3_4():
    regions = ["england", "northWest"]
    for region in regions:
        jname = f"outputs/beds_occupied_LoS_{region}.json"

        if os.path.exists(jname):
            with open(jname, "r") as json_file:
                res_dict = json.load(json_file)
                EBM_dict = res_dict["EBM"]
                ABM_dict = res_dict["ABM"]
        else:
            region_dict = sf.standard_forecast(region)
            EBM_dict = {
                key: list(region_dict[key])
                for key in region_dict
                if key != "pN"
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


def figures_5_6():
    regions = ["england", "northWest"]
    for region in regions:
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
                    D_mean_arr[idx, hidx] = (
                        met_dict["dead_mean"] * met_dict["pN"]
                    )
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


def figure_7():
    R0s = [1.508, 1.51, 1.512, 1.53]
    region = "england"
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


def figure_8():
    R0s = [1.553, 1.554, 1.557, 1.58]
    region = "northWest"
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


def figure_9():
    R0s = [1.508, 1.51, 1.512, 1.53]
    region = "england"
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


def figure_10():
    R0s = [1.553, 1.554, 1.557, 1.58]
    region = "northWest"
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


def figures_11_12():
    regions = ["england", "northWest"]
    for region in regions:
        jname = f"outputs/changing_Hl_D_{region}_small.json"
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
                    D_mean_arr[idx, ridx] = (
                        met_dict["dead_mean"] * met_dict["pN"]
                    )
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


def figure_13():
    Hls = [0.54, 0.6]
    region = "england"
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


def figure_14():
    Hls = [0.68, 0.72]
    region = "northWest"
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


def figures_15_16():
    regions = ["england", "northWest"]
    for region in regions:
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
                for Hidx, Hufac in enumerate(
                    tqdm(Hufacs, f"R0 = {R0:.1f}, Hus")
                ):
                    num_breaches = np.zeros(trials)
                    for t in range(trials):
                        breaching_dict = ct.capacity_thresholds_abm(
                            region, R0, Hup=Hufac, mc_trials=mc_trials
                        )
                        num_breaches[t] = breaching_dict["num"]

                    number_breaches_mean[idx, Hidx] = np.mean(num_breaches)
                    number_breaches_lb[idx, Hidx] = np.quantile(
                        num_breaches, lb
                    )
                    number_breaches_ub[idx, Hidx] = np.quantile(
                        num_breaches, ub
                    )
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
        cp.number_breaches_Hmax_plot(region_dict, fname=fname, ps=False)


def figures_17_18():
    regions = ["england", "northWest"]
    for region in regions:
        jname = f"outputs/num_breaches_Hu_{region}.json"
        if os.path.exists(jname):
            with open(jname, "r") as json_file:
                region_dict = json.load(json_file)
        else:
            R0s = [1.3, 1.5, 1.7, 1.9]
            Hufacs = np.arange(0.94, 1.04 + 0.005, 0.005)

            number_breaches = np.zeros([len(R0s), len(Hufacs)])

            for idx, R0 in enumerate(R0s):
                for Hidx, Hufac in enumerate(
                    tqdm(Hufacs, f"R0 = {R0:.1f}, Hus")
                ):
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
        cp.number_breaches_Hu_plot(region_dict, fname=fname, ps=False)


def figures_19_20():
    regions = ["england", "northWest"]
    for region in regions:
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
                for Hidx, Hufac in enumerate(
                    tqdm(Hufacs, f"R0 = {R0:.1f}, Hus")
                ):
                    breaching_dict = ct.capacity_thresholds_abm(
                        region, R0, Hup=Hufac
                    )
                    diff_breach_mean[idx, Hidx] = breaching_dict[
                        "diff_from_mean"
                    ]
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
        cp.metrics_plot(region_dict, title, fname=fname, ps=False)


def figures_21_22():
    regions = ["england", "northWest"]
    for region in regions:
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
                for Hidx, Hufac in enumerate(
                    tqdm(Hufacs, f"R0 = {R0:.1f}, Hus")
                ):
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
        cp.metrics_plot(region_dict, title, fname=fname, ps=False)


def figure_23():
    region = "southEast"
    jname = f"outputs/beds_occupied_LoS_nc_{region}.json"

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
            branch_dict = sf.standard_forecast_LoS(
                region, dt=dt, correction=False
            )
            ABM_dict[dt] = {
                key: list(branch_dict[key])
                for key in branch_dict
                if key != "pN"
            }

        save_dict = {"EBM": EBM_dict, "ABM": ABM_dict}

        with open(jname, "w") as json_file:
            json.dump(save_dict, json_file)

    fname = f"figures/beds_occupied_LoS_nc_{region}.png"
    fp.compare_abms_plot(EBM_dict, ABM_dict, fname=fname, ps=False)


def figure_24():
    # note, this is the same plot as Fig~7 in the manuscript
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
    fp.compare_abms_plot(EBM_dict, ABM_dict, fname=fname, ps=False)


def figure_25():
    region = "southEast"
    jname = f"outputs/rate_params_testing_nc_{region}.json"
    if os.path.exists(jname):
        with open(jname, "r") as json_file:
            param_dist_dict = json.load(json_file)
    else:
        dts = [1.0, 0.5, 0.25]
        mc = 5
        param_dist_dict = {dt: {} for dt in dts}
        for dt in dts:
            param_dist_dict_dt, _ = sm.parameter_testing(
                region, 100, dt, mc, correction=False
            )
            param_dist_dict[dt] = param_dist_dict_dt

        with open(jname, "w") as json_file:
            json.dump(param_dist_dict, json_file)

    fname = f"figures/hist_nc_{region}.png"
    sp.parameter_histogram(
        param_dist_dict, correction=False, fname=fname, ps=True
    )


def figure_26():
    region = "southEast"
    jname = f"outputs/rate_params_testing_{region}.json"
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

    fname = f"figures/hist_{region}.png"
    sp.parameter_histogram(param_dist_dict, fname=fname, ps=True)


if __name__ == "__main__":
    # figures_1_2()
    # figures_3_4()
    # figures_5_6() 
    # figure_7() 
    # figure_8()
    # figure_9()
    # figure_10()
    # figures_11_12()
    # figure_13()
    # figure_14()
    # figures_15_16()
    # figures_17_18()
    # figures_19_20()
    # figures_21_22()
    # figure_23()
    # figure_24() 
    # figure_25() 
    # figure_26()  
