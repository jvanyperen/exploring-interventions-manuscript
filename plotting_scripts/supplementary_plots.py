import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.usetex"] = True

LEGEND_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 20
TITLE_FONTSIZE = 20
CHART_SIZE = [13, 10]


def parameter_histogram(param_dist_dict, fname=None, ps=True, correction=True):
    fig, ax = plt.subplots(2, 3, sharex=True, sharey=False, figsize=CHART_SIZE)

    param_letters = ["E", "U"]
    param_idx = [2, 4]

    for idx, dt in enumerate(param_dist_dict):
        for l_idx, (letter, p_idx) in enumerate(zip(param_letters, param_idx)):
            ylabel = False
            xlabel = False
            if idx == 0:
                ylabel = True
            if l_idx == 2:
                xlabel = True

            prepare_histogram(
                ax[l_idx, idx],
                param_dist_dict[dt][letter],
                param_dist_dict[dt]["params"][p_idx],
                letter,
                float(dt),
                xlabel=xlabel,
                ylabel=ylabel,
            )

    if correction:
        fig.suptitle("Using correction term", fontsize=TITLE_FONTSIZE)
    else:
        fig.suptitle("Not using correction term", fontsize=TITLE_FONTSIZE)

    if not (fname is None):
        fig.savefig(fname)

    if ps:
        plt.show()


def prepare_histogram(ax, g_dist_dict, g, letter, dt, xlabel=True, ylabel=True):
    times = np.array(list(g_dist_dict.keys()), dtype=float)
    sorted_times = np.sort(times)
    times = np.arange(
        start=dt,
        stop=sorted_times[-1] + dt,
        step=dt,
    )
    try:
        counts = np.array([g_dist_dict.get(t, 0) for t in times])
        counts = counts / (dt * np.sum(counts))
    except ZeroDivisionError:
        counts = np.array([g_dist_dict.get(f"{t}", 0) for t in times])
        counts = counts / (dt * np.sum(counts))

    ft = 15
    ax.set_title(
        rf"$(\gamma_{letter})^{{-1}}$, $\Delta t$: {dt} (day)",
        fontsize=TITLE_FONTSIZE,
    )
    if xlabel:
        ax.set_xlabel("Times (days)", fontsize=AXIS_LABEL_FONTSIZE)

    if ylabel:
        ax.set_ylabel("Probability Density", fontsize=AXIS_LABEL_FONTSIZE)

    ax.bar(
        times[: int(ft / dt)],
        counts[: int(ft / dt)],
        width=dt,
    )
    ax.plot(
        times[: int(ft / dt)],
        g * np.exp(-g * times[: int(ft / dt)]),
        c="red",
    )

    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
