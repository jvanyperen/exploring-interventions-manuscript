import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.usetex"] = True

LEGEND_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 20
TITLE_FONTSIZE = 20
CHART_SIZE = [10, 6]
LONG_CHART_SIZE = [10, 10]


def do_nothing_Rt_plot(Rt_dict, fname=None, ps=True):
    fig, ax = plt.subplots(1, 1, figsize=CHART_SIZE)
    ax.set_xlabel("Time (days)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"$\mathcal{R}_t$", fontsize=AXIS_LABEL_FONTSIZE)

    for R0 in Rt_dict:
        t_arr = Rt_dict[R0]["t"]
        Rt_arr = Rt_dict[R0]["Rt"]

        ax.plot(t_arr, Rt_arr, label=f"{R0:.1f}")

    ax.legend(
        loc="best",
        title=r"$\mathcal{R}_0$",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=TITLE_FONTSIZE,
    )
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)

    if not (fname is None):
        fig.savefig(fname)

    if ps:
        plt.show()


def do_nothing_hospital_plot(region_dict, fname=None, ps=True):
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=CHART_SIZE)
    R0s = [k for k in region_dict]

    R0 = R0s[0]
    hospital_do_nothing_plot(ax[0, 0], region_dict[R0], R0, xlabel=False)

    plt.gca().set_prop_cycle(None)
    R0 = R0s[1]
    hospital_do_nothing_plot(ax[0, 1], region_dict[R0], R0, xlabel=False)

    plt.gca().set_prop_cycle(None)
    R0 = R0s[2]
    hospital_do_nothing_plot(ax[1, 0], region_dict[R0], R0)

    plt.gca().set_prop_cycle(None)
    R0 = R0s[3]
    hospital_do_nothing_plot(ax[1, 1], region_dict[R0], R0)

    fig.suptitle(r"Beds Occupied $(\%N)$", fontsize=TITLE_FONTSIZE)

    if not (fname is None):
        fig.savefig(fname)

    if ps:
        plt.show()


def hospital_do_nothing_plot(ax, Hdict, R0, xlabel=True):
    for region in Hdict:
        pN = Hdict[region]["pN"]
        ax.plot(Hdict[region]["t"], Hdict[region]["H"] * pN, label=region)

    if xlabel:
        ax.set_xlabel("Time (days)", fontsize=AXIS_LABEL_FONTSIZE)

    ax.legend(
        loc="best",
        title="Region",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=TITLE_FONTSIZE,
    )
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_title(rf"$\mathcal{{R}}_0$ = {R0:.1f}", fontsize=TITLE_FONTSIZE)


def do_nothing_deaths_plot(region_dict, region_abm_dict, fname=None, ps=True):
    fig, ax = plt.subplots(1, 1, figsize=CHART_SIZE)

    for region in region_dict:
        (line,) = ax.plot(
            region_dict[region]["R0"],
            region_dict[region]["D"],
            label=region,
        )
        yerr = [
            np.array(region_abm_dict[region]["D_mean"])
            - np.array(region_abm_dict[region]["D_lb"]),
            np.array(region_abm_dict[region]["D_ub"])
            - np.array(region_abm_dict[region]["D_mean"]),
        ]
        ax.errorbar(
            region_abm_dict[region]["R0"],
            region_abm_dict[region]["D_mean"],
            yerr,
            label=f"PR: {region}",
            fmt=".",
            c=line.get_color(),
            capsize=5,
            # c="k",
        )

    ax.set_xlabel(r"$\mathcal{R}_0$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"Dead individuals $(\%N)$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.legend(
        loc="best",
        title="Region",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=TITLE_FONTSIZE,
    )
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)

    if not (fname is None):
        fig.savefig(fname)

    if ps:
        plt.show()
