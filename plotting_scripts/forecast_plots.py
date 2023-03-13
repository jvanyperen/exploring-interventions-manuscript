import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

plt.rcParams["text.usetex"] = True

LEGEND_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 20
TITLE_FONTSIZE = 20
CHART_SIZE = [10, 6]
LONG_CHART_SIZE = [10, 10]


def prop_beds_from_fit_plot(region_dicts, dates, fname=None, ps=True):
    fig, ax = plt.subplots(1, 1, figsize=CHART_SIZE)

    dt = 0

    for key in region_dicts:
        ax.plot(region_dicts[key]["t"], region_dicts[key]["HpN"], label=key)
        dt = max(region_dicts[key]["t"][1], dt)
        daily_H = region_dicts[key]["HpN"][:: int(1 / dt)]

    d = dates[0]
    full_dates = [d]
    while d < dates[1]:
        d = d + datetime.timedelta(days=1)
        full_dates.append(d)

    ax2 = ax.twiny()
    ax2.plot(full_dates, daily_H[: len(full_dates)], ls="None")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))

    ax.set_xlabel("Time (days)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"Beds Occupied (\%$N$)", fontsize=AXIS_LABEL_FONTSIZE)
    ax2.set_xlabel("Date (d/m/20)", fontsize=AXIS_LABEL_FONTSIZE)

    ax.legend(
        loc="best",
        title="Region",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=TITLE_FONTSIZE,
    )
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    ax2.tick_params(axis="x", which="major", labelsize=TICK_LABEL_FONTSIZE)

    if not (fname is None):
        fig.savefig(fname)

    if ps:
        plt.show()


def beds_from_fits_data_plot(
    region_dicts, data_dicts, dates, fname=None, ps=True
):
    fig, ax = plt.subplots(
        len(region_dicts), 1, sharex=True, figsize=LONG_CHART_SIZE
    )

    d = dates[0]
    full_dates = [d]
    while d < dates[1]:
        d = d + datetime.timedelta(days=1)
        full_dates.append(d)

    ax2 = ax[0].twiny()
    for idx, (rk, dk) in enumerate(zip(region_dicts, data_dicts)):
        ax[idx].plot(region_dicts[rk]["t"], region_dicts[rk]["H"], label=rk)
        ax[idx].scatter(
            data_dicts[dk]["occ"]["days"],
            data_dicts[dk]["occ"]["data"],
            label="Data",
            c="red",
            marker="x",
        )

        ax[idx].legend(
            loc="best",
            fontsize=LEGEND_FONTSIZE,
        )
        ax[idx].tick_params(
            axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE
        )

        ax[idx].set_ylabel("Beds Occupied", fontsize=AXIS_LABEL_FONTSIZE)

        if idx == 0:
            ax2.plot(full_dates[1:], data_dicts[dk]["occ"]["data"], ls="None")
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))

    ax[-1].set_xlabel("Time (days)", fontsize=AXIS_LABEL_FONTSIZE)
    ax2.set_xlabel("Date (d/m/20)", fontsize=AXIS_LABEL_FONTSIZE)
    ax2.tick_params(axis="x", which="major", labelsize=TICK_LABEL_FONTSIZE)

    if not (fname is None):
        fig.savefig(fname)

    if ps:
        plt.show()


def compare_abms_plot(region_dict, abm_dict, fname=None, ps=True):
    fig, ax = plt.subplots(
        len(abm_dict), 1, sharex=True, figsize=LONG_CHART_SIZE
    )

    for idx, dt in enumerate(abm_dict):
        compare_abms(ax[idx], region_dict, abm_dict[dt], dt)

    ax[-1].set_xlabel("Time (days)", fontsize=AXIS_LABEL_FONTSIZE)

    if not (fname is None):
        fig.savefig(fname)

    if ps:
        plt.show()


def compare_abms(ax, region_dict, dt_sims, dt):
    ax.plot(
        region_dict["t"],
        region_dict["H"],
        label="SEIR-D",
        ls="dashed",
        c="k",
        lw=2,
    )
    ax.plot(
        dt_sims["t"],
        dt_sims["H_mean"],
        label="mean",
    )
    ax.fill_between(
        dt_sims["t"],
        dt_sims["H_lb"],
        dt_sims["H_ub"],
        alpha=0.5,
        label="PR",
        ec="black",
        color="grey",
    )

    ax.legend(
        loc="best",
        fontsize=LEGEND_FONTSIZE,
    )
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_ylabel("Beds Occupied", fontsize=AXIS_LABEL_FONTSIZE)

    ax.set_title(rf"$\Delta t$ = {dt}", fontsize=TITLE_FONTSIZE)


def beds_wrong_params(region_dict, fname=None, ps=True):
    fig, ax = plt.subplots(1, 1, figsize=CHART_SIZE)
    axy = ax.twinx()

    for region in region_dict:
        ax.plot(
            region_dict[region]["t"], region_dict[region]["H"], label=region
        )
        axy.plot(region_dict[region]["t"], region_dict[region]["HpN"], ls=None)

    ax.legend(
        loc="best",
        fontsize=LEGEND_FONTSIZE,
    )
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    axy.tick_params(axis="y", which="major", labelsize=TICK_LABEL_FONTSIZE)

    ax.set_ylabel("Beds Occupied", fontsize=AXIS_LABEL_FONTSIZE)
    axy.set_ylabel(r"Beds Occupied $(\%N)$", fontsize=AXIS_LABEL_FONTSIZE)

    if not (fname is None):
        fig.savefig(fname)

    if ps:
        plt.show()
