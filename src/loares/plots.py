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


class AnnotatedHeatmap(Heatmap):
    """pymoo Heatmap with the cell value drawn inside each cell, plus
    optional direction glyphs (▲/▼ around the midpoint) and significance
    markers (*)."""

    def __init__(self, fmt=".3f", significance=None, glyph=False, **kwargs):
        super().__init__(**kwargs)
        self.fmt = fmt
        self.significance = significance
        self.glyph = glyph

    def _do(self):
        super()._do()
        F = np.asarray(self.to_plot[0][0], dtype=float)
        if self.bounds is None:
            lo, hi = F.min(axis=0), F.max(axis=0)
        else:
            bounds = np.asarray(self.bounds, dtype=float)
            lo, hi = bounds[0], bounds[1]
        if np.ndim(lo) == 0:
            lo = np.full(F.shape[1], lo)
        if np.ndim(hi) == 0:
            hi = np.full(F.shape[1], hi)
        for i in range(F.shape[0]):
            for j in range(F.shape[1]):
                v = (F[i, j] - lo[j]) / (hi[j] - lo[j])
                if self.reverse:
                    v = 1.0 - v
                significant = bool(
                    self.significance is not None and self.significance[i, j]
                )
                text = f"{F[i, j]:{self.fmt}}"
                if significant:
                    text += "*"
                rgba = self.cmap(v)
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if luminance < 0.5 else "black"
                value_text = self.ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold" if significant else "normal",
                )
                if self.glyph:
                    mid = (lo[j] + hi[j]) / 2
                    marker = None
                    if F[i, j] > mid:
                        marker = "^"
                    elif F[i, j] < mid:
                        marker = "v"
                    if marker is not None:
                        # Drawn as a geometric marker (not a text glyph) so it
                        # sits exactly on the value's vertical center, placed
                        # just right of the measured text extent.
                        self.fig.canvas.draw()
                        renderer = self.fig.canvas.get_renderer()
                        bbox = value_text.get_window_extent(renderer=renderer)
                        inv = self.ax.transData.inverted()
                        right = inv.transform((bbox.x1 + 3, bbox.y0))[0]
                        self.ax.scatter(
                            right,
                            i,
                            marker=marker,
                            s=28,
                            color=text_color,
                            linewidths=0,
                        )


def save_heatmap(
    F,
    x_labels,
    y_labels,
    path,
    annotate=False,
    fmt=".3f",
    significance=None,
    glyph=False,
    cmap="Oranges_r",
    reverse=True,
    title="Optimization",
):
    plot_cls = AnnotatedHeatmap if annotate else Heatmap
    kwargs = {}
    if annotate:
        kwargs = {
            "fmt": fmt,
            "significance": significance,
            "glyph": glyph,
        }
    plot = plot_cls(
        bounds=[0, 1],
        title=title,
        cmap=cmap,
        reverse=reverse,
        solution_labels=y_labels,
        labels=x_labels,
        **kwargs,
    )
    plot.add(F)
    plot.save(path)


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
