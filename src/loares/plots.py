from pymoo.visualization.scatter import Scatter
from pymoo.visualization.heatmap import Heatmap
from pymoo.visualization.pcp import PCP

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from matplotlib.backends.backend_pdf import PdfPages


def save_scatter_plots(F, spec, path):
    if spec["problem"].n_obj <= 3:
        plot = Scatter(legend=True)
        plot.add(F, label=spec["algorithm_name"], **spec["plot_kwargs"])
        plot.save(path)
    else:
        plot = PCP()
        plot.add(F)
        plot.save(path)


def save_heatmap(F, x_labels, y_labels, path):
    plot = Heatmap(
        bounds=[0, 1],
        title=("Optimization", {"pad": 15}),
        cmap="Oranges_r",
        solution_labels=y_labels,
        labels=x_labels,
    )
    plot.add(F)
    plot.save(path)


colors = sns.color_palette("tab10", 10)


def multi_line_plot(
    data,
    pdf,
    legend_fontsize=8,
    label_fontsize=None,
    tick_fontsize=None,
):
    fig = plt.figure()
    for iy, (xdata, ydata) in enumerate(zip(data["xdata"], data["ydata"])):
        plt.plot(xdata, ydata, linestyle="-", marker="", color=colors[iy])
    if "vline" in data:
        for iv, pt in enumerate(data["vline"]):
            y_point, x_point = pt
            plt.axvline(x=x_point, linestyle="--", color=colors[iv])
    if "point" in data:
        for ip, pt in enumerate(data["point"]):
            y_point, x_point = pt
            plt.plot(x_point, y_point, linestyle="", marker="x", color=colors[ip])
    plt.legend(labels=data["legend"], loc="best", fontsize=legend_fontsize)
    plt.grid(which="both", linestyle="--", alpha=0.7)
    plt.xlabel(f"{data['xlabel']}", fontsize=label_fontsize)
    plt.ylabel(f"{data['ylabel']}", fontsize=label_fontsize)
    if tick_fontsize is not None:
        plt.tick_params(labelsize=tick_fontsize)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()
