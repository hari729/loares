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
    # keys = list(data.keys())
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
