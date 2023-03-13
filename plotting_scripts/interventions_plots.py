import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.usetex"] = True

LEGEND_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 20
TITLE_FONTSIZE = 20
CHART_SIZE = [10, 6]
LONG_CHART_SIZE = [10, 10]


def intervention_hospital_plot_all(region_dict, fname=None, ps=True):
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=CHART_SIZE)
    R0s = [k for k in region_dict]

    R0 = R0s[0]
    hospital_intervention_all_plot(ax[0, 0], region_dict[R0], R0, xlabel=False)

    plt.gca().set_prop_cycle(None)
    R0 = R0s[1]
    hospital_intervention_all_plot(ax[0, 1], region_dict[R0], R0, xlabel=False)

    plt.gca().set_prop_cycle(None)
    R0 = R0s[2]
    hospital_intervention_all_plot(ax[1, 0], region_dict[R0], R0)

    plt.gca().set_prop_cycle(None)
    R0 = R0s[3]
    hospital_intervention_all_plot(ax[1, 1], region_dict[R0], R0)

    fig.suptitle(r"Beds Occupied $(\%N)$", fontsize=TITLE_FONTSIZE)

    if not (fname is None):
        fig.savefig(fname)

    if ps:
        plt.show()


def hospital_intervention_all_plot(ax, Hdict, R0, xlabel=True):
    tend = 0
    for region in Hdict:
        pN = Hdict[region]["pN"]
        ax.plot(Hdict[region]["t"], Hdict[region]["H"] * pN, label=region)
        ax.plot(
            Hdict[region]["t"],
            Hdict[region]["H_in"] * pN,
            label="__nolabel__",
            c="lightgray",
        )
        tend = max(Hdict[region]["t"][-1], tend)

    ax.hlines(
        100.0 * Hdict[region]["Hufac"],
        Hdict[region]["t"][0],
        tend,
        colors="k",
        ls="--",
        label="__nolabel__",
    )

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


def intervention_death_plot(Hl_dict, Hl_abm_dict, fname=None, ps=True):
    fig, ax = plt.subplots(1, 1, figsize=CHART_SIZE)

    Hls = [Hl for Hl in Hl_dict if not Hl == "dono"]

    for Hl in Hls:
        (line,) = ax.plot(
            Hl_dict[Hl]["R0"],
            Hl_dict[Hl]["D"],
            label=rf"{Hl}$H_u$",
        )
        yerr = [
            np.array(Hl_abm_dict[Hl]["D_mean"])
            - np.array(Hl_abm_dict[Hl]["D_lb"]),
            np.array(Hl_abm_dict[Hl]["D_ub"])
            - np.array(Hl_abm_dict[Hl]["D_mean"]),
        ]
        ax.errorbar(
            Hl_abm_dict[Hl]["R0"],
            Hl_abm_dict[Hl]["D_mean"],
            yerr,
            label=rf"PR: {Hl}$H_u$",
            fmt=".",
            c=line.get_color(),
            capsize=5,
            # c="k",
        )

    ax.plot(
        Hl_dict["dono"]["R0"],
        Hl_dict["dono"]["D"],
        label="do nothing",
        ls="--",
        lw=2,
        c="k",
    )

    ax.set_xlabel(r"$\mathcal{R}_0$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"Dead individuals $(\%N)$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.legend(
        loc="best",
        title=r"$H_l$",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=TITLE_FONTSIZE,
    )
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)

    if not (fname is None):
        fig.savefig(fname)

    if ps:
        plt.show()


def intervention_hospital_plot(region_dict, fname=None, ps=True):
    fig, ax = plt.subplots(1, 1, figsize=CHART_SIZE)

    for R0 in region_dict:
        pN = region_dict[R0]["pN"]
        ax.plot(
            region_dict[R0]["t"], region_dict[R0]["H"] * pN, label=f"{R0:.3f}"
        )
        ax.plot(
            region_dict[R0]["t"],
            region_dict[R0]["H_in"] * pN,
            label="__nolabel__",
            c="lightgray",
        )

    ax.set_xlabel("Time (days)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"Beds occupied $(\% N)$", fontsize=AXIS_LABEL_FONTSIZE)

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


def intervention_hospital_plot_rt(region_dict, fname=None, ps=True):
    fig, ax = plt.subplots(1, 1, figsize=CHART_SIZE)

    tend = 0
    for R0 in region_dict:
        ax.plot(region_dict[R0]["t"], region_dict[R0]["Rt"], label=f"{R0:.3f}")
        ax.plot(
            region_dict[R0]["t"],
            region_dict[R0]["Rt_in"],
            label="__nolabel__",
            c="lightgray",
        )
        tend = max(region_dict[R0]["t"][-1], tend)

    ax.hlines(
        1.0,
        region_dict[R0]["t"][0],
        tend,
        colors="k",
        ls="--",
        label="__nolabel__",
    )

    ax.set_xlabel("Time (days)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"$\mathcal{R}_t$", fontsize=AXIS_LABEL_FONTSIZE)

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


def intervention_death_Hl_plot(Hl_dict, Hl_abm_dict, fname=None, ps=True):
    fig, ax = plt.subplots(1, 1, figsize=CHART_SIZE)

    R0s = [R0 for R0 in Hl_dict if not R0 == "dono"]
    for R0 in R0s:
        ax.plot(Hl_dict[R0]["Hl"], Hl_dict[R0]["D"], label=R0, lw=2)
        yerr = [
            np.array(Hl_abm_dict[R0]["D_mean"])
            - np.array(Hl_abm_dict[R0]["D_lb"]),
            np.array(Hl_abm_dict[R0]["D_ub"])
            - np.array(Hl_abm_dict[R0]["D_mean"]),
        ]
        ax.errorbar(
            Hl_abm_dict[R0]["Hl"],
            Hl_abm_dict[R0]["D_mean"],
            yerr,
            label=f"PR: {R0}",
            fmt=".",
            c="gray",
            capsize=5,
            # c="k",
        )

        ax.hlines(
            Hl_dict["dono"]["D"],
            Hl_dict[R0]["Hl"][0],
            Hl_dict[R0]["Hl"][-1],
            label=f"do nothing: {R0}",
            ls="--",
            lw=2,
            colors="k",
        )

    ax.set_xlabel(r"$H_l H_u^{-1}$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"Dead individuals $(\%N)$", fontsize=AXIS_LABEL_FONTSIZE)
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


def intervention_hospital_Hl_plot(region_dict, fname=None, ps=True):
    fig, ax = plt.subplots(1, 1, figsize=CHART_SIZE)

    for Hl in region_dict:
        pN = region_dict[Hl]["pN"]
        ax.plot(
            region_dict[Hl]["t"],
            region_dict[Hl]["H"] * pN,
            label=rf"{Hl:.2f} $H_u$",
        )
        ax.plot(
            region_dict[Hl]["t"],
            region_dict[Hl]["H_in"] * pN,
            label="__nolabel__",
            c="lightgray",
        )

    ax.set_xlabel("Time (days)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"Beds occupied $(\% N)$", fontsize=AXIS_LABEL_FONTSIZE)

    ax.legend(
        loc="best",
        title=r"$H_l$",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=TITLE_FONTSIZE,
    )
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)

    if not (fname is None):
        fig.savefig(fname)

    if ps:
        plt.show()
