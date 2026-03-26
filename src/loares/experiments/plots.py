import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

colors = sns.color_palette("tab10", 10)


def multi_line_plot(
    data,
    filepath,
    filename=None,
    legend_fontsize=8,
    label_fontsize=None,
    tick_fontsize=None,
):
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
    if filename is None:
        filename = f"{data['ylabel']}-vs-{data['xlabel']}".replace(" ", "-")
    plt.savefig(f"{filepath}/{filename}.pdf", bbox_inches="tight")
    plt.close()


def plot_2d(data, filepath, cid=0):
    color = sns.color_palette("tab10")[cid]
    legend = [data["name"]]
    plt.figure()
    plt.plot(
        data["f1"],
        data["f2"],
        linestyle="",
        marker="o",
        markeredgecolor="white",
        markeredgewidth=0.5,
        markerfacecolor=color,
    )
    plt.legend(labels=legend, loc="best", fontsize=8)
    plt.grid(which="both", linestyle="--", alpha=0.7)
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.tight_layout()
    plt.savefig(f"{filepath}/{data['name']}-pareto-front.pdf", bbox_inches="tight")
    plt.close()


def plot_3d(pareto_dict, filepath, cid=0, mid=0):
    sns.set_theme(style="whitegrid")

    markers = ["o", "X", "s", "P", "v", "^", "D", "p", "*"]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        pareto_dict["f1"],
        pareto_dict["f2"],
        pareto_dict["f3"],
        s=40,
        alpha=0.9,
        color=sns.color_palette("tab10")[cid],
        marker=markers[mid],
        edgecolors="white",
        linewidths=0.4,
        label=pareto_dict.get("name", None),
    )

    # Axis labels
    ax.set_xlabel("f1", labelpad=10)
    ax.set_ylabel("f2", labelpad=12)
    ax.text2D(0.03, 0.8, "f3", transform=ax.transAxes, fontsize=12)

    # Grid styling (matches your previous function)
    grid_style = {"linestyle": "--", "color": (0.8, 0.8, 0.8, 0.5)}
    ax.xaxis._axinfo["grid"].update(**grid_style)
    ax.yaxis._axinfo["grid"].update(**grid_style)
    ax.zaxis._axinfo["grid"].update(**grid_style)

    # Pane & spine styling
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.line.set_color("black")
    ax.yaxis.line.set_color("black")
    ax.zaxis.line.set_color("black")

    ax.locator_params(axis="x", nbins=8)
    ax.locator_params(axis="y", nbins=8)
    ax.locator_params(axis="z", nbins=8)

    ax.view_init(elev=20, azim=45)

    # Legend (only if name provided)
    if pareto_dict.get("name") is not None:
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.95, 0.95),
            frameon=True,
            framealpha=0.5,
            edgecolor="black",
            fontsize=10,
            bbox_transform=ax.transAxes,
        )

    plt.tight_layout()
    out_path = f"{filepath}/{pareto_dict['name']}-pareto-front.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def parallel_coordinates_plot(data, filepath, alpha=0.6):
    keys = sorted(
        [k for k in data.keys() if re.fullmatch(r"f\d+", k)], key=lambda x: int(x[1:])
    )
    values = np.array([data[f] for f in keys]).T  # shape: (n_points, n_dims)

    # Normalize each dimension to [0,1]
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    norm = (values - mins) / (maxs - mins + 1e-12)

    fig, ax = plt.subplots(figsize=(10, 4))

    for row in norm:
        ax.plot(range(len(keys)), row, alpha=alpha)

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=15)
    ax.set_ylabel("Normalized value")

    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    out_path = f"{filepath}/{data['name']}-pareto-front.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def parallel_coordinates_plot_v2(
    data, filepath, alpha=0.6, axis_mins=None, axis_maxs=None
):
    keys = sorted(
        [k for k in data.keys() if re.fullmatch(r"f\d+", k)], key=lambda x: int(x[1:])
    )
    values = np.array([data[f] for f in keys]).T  # shape: (n_points, n_dims)

    if values.size == 0:
        return

    n_dims = len(keys)
    data_mins = np.nanmin(values, axis=0)
    data_maxs = np.nanmax(values, axis=0)

    def _resolve_bounds(custom, keys_, fallback):
        if custom is None:
            return fallback.astype(float)
        if isinstance(custom, dict):
            return np.array([float(custom[k]) for k in keys_], dtype=float)
        arr = np.asarray(custom, dtype=float)
        if arr.shape[0] != len(keys_):
            raise ValueError("Custom axis bounds must match number of objective axes")
        return arr

    mins = _resolve_bounds(axis_mins, keys, data_mins)
    maxs = _resolve_bounds(axis_maxs, keys, data_maxs)
    spans = maxs - mins
    if np.any(spans < 0.0):
        raise ValueError("Each custom axis min must be <= corresponding axis max")
    safe_spans = np.where(spans == 0.0, 1.0, spans)

    # Independent-axis scaling for each objective, then mapped to a shared render range.
    host_min, host_max = mins[0], maxs[0]
    host_span = host_max - host_min
    if host_span == 0.0:
        host_span = 1.0

    z = (values - mins) / safe_spans
    y = z * host_span + host_min

    const_idx = np.where(spans == 0.0)[0]
    if const_idx.size:
        y[:, const_idx] = host_min + 0.5 * host_span

    def _fmt(v):
        if not np.isfinite(v):
            return str(v)
        av = abs(v)
        if av != 0 and (av < 1e-3 or av >= 1e4):
            return f"{v:.2e}"
        return f"{v:.6g}"

    fig_width = max(10, 0.95 * n_dims)
    fig, ax = plt.subplots(figsize=(fig_width, 4.6))

    x = np.arange(n_dims)
    for row in y:
        ax.plot(x, row, alpha=alpha, linewidth=1.0)

    for xi in x:
        ax.axvline(x=float(xi), linestyle="-", color="0.55", linewidth=0.8, zorder=0)

    y_pad = 0.16 * host_span
    y_lo = host_min - y_pad
    y_hi = host_max + y_pad
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(-0.25, n_dims - 1 + 0.25)

    label_fontsize = 11
    for i, xi in enumerate(x):
        xif = float(xi)
        ax.text(
            xif,
            host_max + 0.05 * host_span,
            _fmt(maxs[i]),
            ha="center",
            va="bottom",
            fontsize=label_fontsize,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.85},
            clip_on=False,
        )
        ax.text(
            xif,
            host_min - 0.05 * host_span,
            _fmt(mins[i]),
            ha="center",
            va="top",
            fontsize=label_fontsize,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.85},
            clip_on=False,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=15)
    ax.set_ylabel("")
    ax.set_yticks(np.linspace(host_min, host_max, 6))
    ax.set_yticklabels([])
    ax.grid(which="both", linestyle="--", alpha=0.6)

    # Keep classic boxed frame like the original plot style.
    ax.spines["left"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    plt.tight_layout()
    out_path = f"{filepath}/{data['name']}-pareto-front.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
