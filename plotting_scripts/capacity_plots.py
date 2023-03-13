import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.usetex"] = True

LEGEND_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 20
TITLE_FONTSIZE = 20
CHART_SIZE = [10, 6]
LONG_CHART_SIZE = [10, 10]
WIDE_CHART_SIZE = [12, 6]


def explore_objective_plot(R0s_dict, fname=None, ps=True):
    fig, ax = plt.subplots(1, 1, figsize=CHART_SIZE)
    axy = ax.twinx()
    axx = ax.twiny()

    R0s = [R0 for R0 in R0s_dict if not (R0 == "Hmaxfac")]

    HuL = np.inf
    HuR = 0
    for R0 in R0s:
        ax.plot(R0s_dict[R0]["Hu"], R0s_dict[R0]["max_H"], label=R0)
        axy.plot(R0s_dict[R0]["Hu"], R0s_dict[R0]["max_HpN"], ls=None)
        axx.plot(R0s_dict[R0]["HupN"], R0s_dict[R0]["max_H"], ls=None)
        HuL = min(R0s_dict[R0]["Hu"][0], HuL)
        HuR = max(R0s_dict[R0]["Hu"][-1], HuR)

    axy.hlines(
        R0s_dict["Hmaxfac"],
        HuL,
        HuR,
        colors="k",
        ls="--",
        label="__nolabel__",
    )

    ax.set_xlabel(r"$H_u$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"max $H$", fontsize=AXIS_LABEL_FONTSIZE)
    axy.set_ylabel(r"max $H$ $(\% N)$", fontsize=AXIS_LABEL_FONTSIZE)
    axx.set_xlabel(r"$H_u$ $(\% N)$", fontsize=AXIS_LABEL_FONTSIZE)

    ax.legend(
        loc="best",
        title=r"$\mathcal{R}_0$",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=TITLE_FONTSIZE,
    )
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    axy.tick_params(axis="y", which="major", labelsize=TICK_LABEL_FONTSIZE)
    axx.tick_params(axis="x", which="major", labelsize=TICK_LABEL_FONTSIZE)

    if not (fname is None):
        fig.savefig(fname)

    if ps:
        plt.show()


def mimin_objective_plot(res_dict, fname=None, ps=True):
    fig, ax = plt.subplots(1, 1, figsize=CHART_SIZE)
    axy = ax.twinx()

    for region in res_dict:
        ax.plot(
            res_dict[region]["R0"], res_dict[region]["HupHmax"], label=region
        )
        axy.plot(res_dict[region]["R0"], res_dict[region]["HupN"], ls="None")

    ax.set_xlabel(r"$\mathcal{R}_0$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"$H_u$ $(\%H_{max})$", fontsize=AXIS_LABEL_FONTSIZE)
    axy.set_ylabel(r"$H_u$ $(\%N)$", fontsize=AXIS_LABEL_FONTSIZE)

    ax.legend(
        loc="best",
        title="Region",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=TITLE_FONTSIZE,
    )

    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    axy.tick_params(axis="y", which="major", labelsize=TICK_LABEL_FONTSIZE)

    if not (fname is None):
        fig.savefig(fname)

    if ps:
        plt.show()


def number_breaches_Hmax_plot(res_dict, fname=None, ps=True):
    fig, ax = plt.subplots(1, 1, figsize=CHART_SIZE)

    ymin = 100
    ymax = 0
    for idx, R0 in enumerate(res_dict):
        if idx == len(res_dict) - 1:
            label = "PR"
        else:
            label = "__nolabel__"
        ax.plot(res_dict[R0]["HupHmax"], res_dict[R0]["num_mean"], label=R0)
        ax.fill_between(
            res_dict[R0]["HupHmax"],
            res_dict[R0]["num_lb"],
            res_dict[R0]["num_ub"],
            alpha=0.5,
            label=label,
            color="grey",
            ec="black",
        )

        ymin = min(res_dict[R0]["num_lb"][0], ymin)
        ymax = max(res_dict[R0]["num_ub"][-1], ymax)

    for R0 in res_dict:
        ax.vlines(
            res_dict[R0]["Hu"],
            ymin,
            ymax,
            colors="k",
            ls="--",
            lw=2,
            label="__nolabel__",
        )

    ax.set_xlabel(r"$\%H_{max}$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Proportion of breaches (\%)", fontsize=AXIS_LABEL_FONTSIZE)

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


def number_breaches_Hu_plot(res_dict, fname=None, ps=True):
    fig, ax = plt.subplots(1, 1, figsize=CHART_SIZE)

    ymin = 100
    ymax = 0
    for R0 in res_dict:
        ax.plot(res_dict[R0]["Hufacs"], res_dict[R0]["num"], label=R0)

        ymin = min(res_dict[R0]["num"][0], ymin)
        ymax = max(res_dict[R0]["num"][-1], ymax)

    ax.vlines(
        100.0,
        ymin,
        ymax,
        colors="k",
        ls="--",
        lw=2,
        label="__nolabel__",
    )

    ax.set_xlabel(r"$\%H_u$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Proportion of breaches (\%)", fontsize=AXIS_LABEL_FONTSIZE)

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


def metrics_plot(res_dict, ptitle, fname=None, ps=True):
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=CHART_SIZE)

    R0s = [R0 for R0 in res_dict]
    xlabel = r"$\%H_u$"

    R0 = R0s[0]
    title = rf"$\mathcal{{R}}_0$ = {float(R0):.1f}"
    threshold_subplot(ax[0, 0], res_dict[R0], title)

    R0 = R0s[1]
    title = rf"$\mathcal{{R}}_0$ = {float(R0):.1f}"
    threshold_subplot(ax[0, 1], res_dict[R0], title)

    R0 = R0s[2]
    title = rf"$\mathcal{{R}}_0$ = {float(R0):.1f}"
    threshold_subplot(ax[1, 0], res_dict[R0], title, xlabel)

    R0 = R0s[3]
    title = rf"$\mathcal{{R}}_0$ = {float(R0):.1f}"
    threshold_subplot(ax[1, 1], res_dict[R0], title, xlabel)

    fig.suptitle(ptitle, fontsize=TITLE_FONTSIZE)

    if not (fname is None):
        fig.savefig(fname)

    if ps:
        plt.show()


def threshold_subplot(ax, to_plot_dict, title, xlabel=None):
    ax.plot(to_plot_dict["Hufacs"], to_plot_dict["mean"], label="mean")

    ax.fill_between(
        to_plot_dict["Hufacs"],
        to_plot_dict["lb"],
        to_plot_dict["ub"],
        alpha=0.5,
        label="PR",
        color="grey",
        ec="black",
    )

    ax.vlines(
        100.0,
        to_plot_dict["lb"][0],
        to_plot_dict["ub"][-1],
        colors="k",
        ls="--",
        lw=2,
        label="__nolabel__",
    )

    ax.legend(loc="best", fontsize=LEGEND_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)

    if not (xlabel is None):
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
