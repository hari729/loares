import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_convergence_comparison(convergence_data_list, legend, comparison_path, problem_name):
    headers = list(convergence_data_list[0].columns)
    for key in headers:
        if key != "evals":
            plt.figure()
            for data in convergence_data_list:
                plt.plot(data['evals'], data[key], linestyle='-',marker='')

            plt.legend(labels=legend, loc='best', fontsize=8)
            plt.grid(which='both',linestyle='--',alpha=0.7)
            plt.xlabel("Function Evaluations")
            plt.ylabel(key)
            plt.tight_layout()
            plt.savefig(f"{comparison_path}/{problem_name}-{key}-comparison.png", dpi=600, bbox_inches='tight')
            plt.close()

def plot_pareto_2d_comparison(pareto_df, comparison_path, problem_name):
    plt.figure()
    sc = sns.relplot(data = pareto_df, x = "f1", y = "f2", hue = "algorithm", style = "algorithm", s = 50,
                     col = "algorithm", col_wrap = 2,alpha = 1, linewidth = 0.3, facet_kws = {'legend_out':False})
    # sc.get_legend().set_title(None)
    plt.tight_layout()
    for ax in sc.axes.flat:
        ax.grid(which='both',linestyle='--',alpha=0.7)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        # ax.set_ylim(top=1, bottom=0)
    sc.set_titles("")
    sc.legend.set_title("Algorithm")
    plt.savefig(f"{comparison_path}/{problem_name}-pareto-fronts-comparison.png", dpi=600, bbox_inches='tight')
    plt.close()

def plot_pareto_3d_individual(pareto_df, comparison_path, problem_name):
    sns.set_theme(style="whitegrid")

    algorithms = pareto_df["algorithm"].unique()
    n_algos = len(algorithms)

    # Consistent colors and markers across figures
    colors = sns.color_palette("tab10", n_algos)
    markers = ["o", "X", "s", "P", "v", "^", "D", "p", "*"]

    color_map = {algo: colors[i % len(colors)] for i, algo in enumerate(algorithms)}
    marker_map = {algo: markers[i % len(markers)] for i, algo in enumerate(algorithms)}

    for algo in algorithms:
        df_algo = pareto_df[pareto_df["algorithm"] == algo]

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Scatter plot for current algorithm
        ax.scatter(
            df_algo["f1"], df_algo["f2"], df_algo["f3"],
            label=algo,
            s=40,
            alpha=0.9,
            color=color_map[algo],
            marker=marker_map[algo],
            edgecolors="white",
            linewidths=0.4
        )

        # Labels and styling
        ax.set_xlabel("f1", labelpad=10)
        ax.set_ylabel("f2", labelpad=12)
        ax.text2D(0.03, 0.8, "f3", transform=ax.transAxes, rotation=0, fontsize=12)

        grid_style = {"linestyle": "--", "color": (0.8, 0.8, 0.8, 0.5)}
        ax.xaxis._axinfo["grid"].update(**grid_style)
        ax.yaxis._axinfo["grid"].update(**grid_style)
        ax.zaxis._axinfo["grid"].update(**grid_style)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.line.set_color("black")
        ax.yaxis.line.set_color("black")
        ax.zaxis.line.set_color("black")
        ax.locator_params(axis='x', nbins=8)
        ax.locator_params(axis='y', nbins=8)
        ax.locator_params(axis='z', nbins=8)

        ax.view_init(elev=20, azim=45)

        # Legend
        legend = ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.95, 0.95),
            frameon=True,
            framealpha=0.5,
            edgecolor='black',
            fontsize=10,
            bbox_transform=ax.transAxes
        )
        # legend.get_frame().set_alpha(0.9)
        # legend.get_frame().set_edgecolor("black")

        plt.tight_layout()
        psize = pareto_df[pareto_df["algorithm"] == algo]["psize"].unique()
        out_path = f"{comparison_path}/{problem_name}-{algo}-pareto-front-3D.png"
        plt.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved: {out_path}")

def plot_pareto_3d_many_obj(pareto_df, comparison_path, problem_name,
                            n_obj, combinations_list=None):

    sns.set_theme(style="whitegrid")

    algorithms = pareto_df["algorithm"].unique()
    n_algos = len(algorithms)

    # Consistent colors and markers across figures
    colors = sns.color_palette("tab10", n_algos)
    markers = ["o", "X", "s", "P", "v", "^", "D", "p", "*"]

    color_map = {algo: colors[i % len(colors)] for i, algo in enumerate(algorithms)}
    marker_map = {algo: markers[i % len(markers)] for i, algo in enumerate(algorithms)}

    if combinations_list is None:
        obj_list = [f"f{i}" for i in range(1, n_obj+1)]
        combinations_list = [obj_list[j:j+3] for j in range(0,n_obj,2)]
        if len(combinations_list[-1]) < 3:
            for k in range(1,4 - len(combinations_list[-1]),1):
                combinations_list[-1].append(f"f{k}" )

    for algo in algorithms:
        df_algo = pareto_df[pareto_df["algorithm"] == algo]

        for combination in combinations_list:
            data = [df_algo[f] for f in combination]

            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                *data,
                label=algo,
                s=40,
                alpha=0.9,
                color=color_map[algo],
                marker=marker_map[algo],
                edgecolors="white",
                linewidths=0.4
            )

            # Labels and styling
            ax.set_xlabel(combination[0], labelpad=10)
            ax.set_ylabel(combination[1], labelpad=12)
            ax.text2D(0.03, 0.8, combination[2], transform=ax.transAxes, rotation=0, fontsize=12)

            grid_style = {"linestyle": "--", "color": (0.8, 0.8, 0.8, 0.5)}
            ax.xaxis._axinfo["grid"].update(**grid_style)
            ax.yaxis._axinfo["grid"].update(**grid_style)
            ax.zaxis._axinfo["grid"].update(**grid_style)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.line.set_color("black")
            ax.yaxis.line.set_color("black")
            ax.zaxis.line.set_color("black")
            ax.locator_params(axis='x', nbins=8)
            ax.locator_params(axis='y', nbins=8)
            ax.locator_params(axis='z', nbins=8)

            ax.view_init(elev=20, azim=45)

            # Legend
            legend = ax.legend(
                loc="upper right",
                bbox_to_anchor=(0.95, 0.95),
                frameon=True,
                framealpha=0.5,
                edgecolor='black',
                fontsize=10,
                bbox_transform=ax.transAxes
            )
            # legend.get_frame().set_alpha(0.9)
            # legend.get_frame().set_edgecolor("black")

            plt.tight_layout()
            psize = pareto_df[pareto_df["algorithm"] == algo]["psize"].unique()
            out_path = f"{comparison_path}/{algo}"
            os.makedirs(out_path, exist_ok=True)
            plt.savefig(f"{out_path}/{problem_name}-{algo}-{combination}-pareto-front-3D.png", 
                        dpi=600, bbox_inches="tight")
            plt.close()

            print(f"✅ Saved: {out_path}")

            
def plot_pareto_2d_comparison_external(pareto_df, external_df, problem_name, comparison_path):
    sns.set_theme(
        context="talk",
        style="whitegrid",
        font_scale=1.1,
        rc={"axes.edgecolor": "black", "axes.linewidth": 1.2}
    )

    algorithms = pareto_df["algorithm"].unique()
    n = len(algorithms)
    ncols = 2
    nrows = (n + 1) // ncols

    palette = sns.color_palette("tab10", n)       # distinct colors
    markers = ["o", "X", "s", "P", "v", "P", "^"] # distinct markers (recycle if needed)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10* ncols, 10 * nrows),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for i, algo in enumerate(algorithms):
        ax = axes[i]
        subset = pareto_df[pareto_df["algorithm"] == algo]

        sns.scatterplot(
            data=subset,
            x="f1", y="f2",
            color=palette[i % len(palette)],
            marker=markers[i % len(markers)],
            s=170, edgecolor="white", alpha=0.8,
            ax=ax,
            label=algo, linewidth=0.5
        )

        sns.scatterplot(
            data=external_df,
            x="f1", y="f2",
            color="indigo", marker="X", s=170, ax=ax, label=external_df["algorithm"].iloc[0]
        )

        # ax.set_ylim(top = 1, bottom=0)
        ax.grid(ls="--", alpha=1)
        ax.legend(loc="best")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.2)

    # Remove unused axes if odd count
    for ax in axes[len(algorithms):]:
        ax.remove()

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{comparison_path}/{problem_name}-pareto-fronts-comparison.png",
                dpi=600, bbox_inches="tight")
    plt.close()
