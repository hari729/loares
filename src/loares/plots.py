from pymoo.visualization.scatter import Scatter
from pymoo.visualization.heatmap import Heatmap
from pymoo.visualization.pcp import PCP

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
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


def save_heatmap(F, x_labels, y_labels, path, annotate=False, fmt=".3f"):
    if annotate:
        _save_annotated_heatmap(F, x_labels, y_labels, path, fmt)
        return
    plot = Heatmap(
        bounds=[0, 1],
        title=("Optimization", {"pad": 15}),
        cmap="Oranges_r",
        solution_labels=y_labels,
        labels=x_labels,
    )
    plot.add(F)
    plot.save(path)


def _save_annotated_heatmap(F, x_labels, y_labels, path, fmt):
    """Annotated seaborn heatmap, normalized to [0, 1] like the pymoo
    Heatmap used in the non-annotated path (suitable for both A12 effect
    sizes and p-values)."""
    F = np.asarray(F, dtype=float)
    n_x, n_y = len(x_labels), len(y_labels)
    fig, ax = plt.subplots(
        figsize=(max(3.0, 0.6 * n_x), max(3.0, 0.6 * n_y))
    )
    sns.heatmap(
        F,
        xticklabels=list(x_labels),
        yticklabels=list(y_labels),
        annot=True,
        fmt=fmt,
        cmap="Oranges_r",
        vmin=0.0,
        vmax=1.0,
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


colors = sns.color_palette("tab10", 10)


def multi_line_plot(
    data,
    pdf=None,
    path=None,
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
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    else:
        if path is None:
            filename = f"{data['ylabel']}-vs-{data['xlabel']}.pdf".replace(" ", "-")
            plt.savefig(filename, bbox_inches="tight")
    plt.close()
