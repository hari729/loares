from pathlib import Path
import os
import __main__
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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
            plt.savefig(f"{comparison_path}/{problem_name}_{key}_comparison.png", dpi=600, bbox_inches='tight')
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
    plt.savefig(f"{comparison_path}/{problem_name}_pareto_fronts_comparison.png", dpi=600, bbox_inches='tight')
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

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.line.set_color("black")
        ax.yaxis.line.set_color("black")
        ax.zaxis.line.set_color("black")

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
        out_path = f"{comparison_path}/{problem_name}_{algo}_{psize}_pareto_front_3D.png"
        plt.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved: {out_path}")

def basic_compare(problem_name, n_obj = None, selection_metric = "HV", minmax = 1, dir = None):
    root_dir = Path(dir) if dir else Path(__main__.__file__).parent.resolve()
    master_list_path = Path(f"{root_dir}/results/{problem_name}")
    master_lists = [a for a in master_list_path.iterdir() if not a.is_dir()]
    for master_list in tqdm(master_lists):
        comparison_path = Path(f"{master_list_path}/comparison/{master_list.stem}")
        os.makedirs(comparison_path, exist_ok=True)
        master_df = pd.read_csv(master_list)
        df = master_df.loc[master_df.groupby("algorithm")[f"{selection_metric}"].idxmax()]
        df.to_csv(f"{comparison_path}/best_metrics.csv")
        convergence_data_list = []
        pareto_front_list = []
        for index, row in df.iterrows():
            convergence_data_list.append(pd.read_csv(f"{row["save_path"]}/convergence.csv"))
            temp = pd.read_csv(f"{row["save_path"]}/solutions.csv")
            temp["algorithm"] = row["algorithm"]
            temp["psize"] = row["psize"]
            pareto_front_list.append(temp)

        pareto_df = pd.concat(pareto_front_list, ignore_index=True)

        plot_convergence_comparison(convergence_data_list, df["algorithm"], comparison_path, problem_name)
        
        if n_obj == 2:
            plot_pareto_2d_comparison(pareto_df, comparison_path, problem_name)
        if n_obj == 3:
            plot_pareto_3d_individual(pareto_df, comparison_path, problem_name)


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
    plt.savefig(f"{comparison_path}/{problem_name}_pareto_fronts_comparison.png",
                dpi=600, bbox_inches="tight")
    plt.close()

def compare_solutions(problem, objectives_dict,
                      selection_metric = "HV", minmax = 1, dir = None):
    problem_name = problem.get_info()["name"]
    root_dir = Path(dir) if dir else Path(__main__.__file__).parent.resolve()
    master_list_path = Path(f"{root_dir}/results/{problem_name}")
    master_lists = [a for a in master_list_path.iterdir() if not a.is_dir()]
    external_df = pd.DataFrame(objectives_dict)
    obj_cols = [f"f{o+1}" for o in range(problem.n_obj)]
    for master_list in tqdm(master_lists):
        comparison_path = Path(f"{master_list_path}/compare_solutions/{master_list.stem}")
        os.makedirs(comparison_path, exist_ok=True)
        master_df = pd.read_csv(master_list)
        df = master_df.loc[master_df.groupby("algorithm")[f"{selection_metric}"].idxmax()]
        pareto_front_list = []
        for index, row in df.iterrows():
            temp = pd.read_csv(f"{row["save_path"]}/solutions.csv")
            temp["algorithm"] = row["algorithm"]
            pareto_front_list.append(temp)
            matched_rows = []
            for i, e_row in external_df.iterrows():
                better_mask = np.all((temp[obj_cols].values - e_row[obj_cols].values) * problem.minmax <= 0, axis=1)
                matches = temp[better_mask]
                # print(matches)

                if len(matches) > 0:
                    flipped_temp = temp[obj_cols].values * problem.minmax
                    flipped_ext = e_row[obj_cols].values * problem.minmax
                    diff = np.asarray(flipped_temp[better_mask] - flipped_ext, dtype=float)
                    dist = np.linalg.norm(diff, axis=1)
                    matches = matches.assign(distance=dist)
                    matches = matches.sort_values(by="distance", ascending=True)
                    matches = matches.head(5)
                    for j, col in enumerate(obj_cols):
                        matches[f"e{col}"] = e_row[col]
                    matched_rows.append(matches)


            if matched_rows:
                matched_df = pd.concat(matched_rows, ignore_index=True)
            else:
                matched_df = pd.DataFrame(columns=temp.columns.tolist() + ["distance", "external_id"])

            output_path = comparison_path / f"{row['algorithm']}_matches.csv"
            matched_df.to_csv(output_path, index=False)
            print(f"✅ Saved matches for {row['algorithm']} → {output_path}")

        pareto_df = pd.concat(pareto_front_list, ignore_index=True)
        plot_pareto_fronts(pareto_df, external_df, problem_name, comparison_path)
