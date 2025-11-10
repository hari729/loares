from pathlib import Path
import os
import __main__
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from opti.core.process import ResultProcessor
from opti.core.problem import Problem
from opti.moo.population import MoPopulation
from opti.moo.sorting import ranking_crowding

class MOProcessor(ResultProcessor):
    def __init__(self,
                 results_list,
                 root_dir = None):
        super().__init__(results_list,
                         root_dir)

    def plot_convergence(self, convergence_data, legend, file_path):
        colors = {
            "GD" : "red",
            "IGD" : "blue",
            "SPC" : "green",
            "SPR" : "orange",
            "HV" : "indigo"
        }
        for key in convergence_data:
            if key == "evals":
                continue

            plt.figure()
            plt.plot(convergence_data["evals"], convergence_data[key], linestyle='-',marker='',
                    color=colors[key],
                    markerfacecolor='cyan',markersize='5',
                    markeredgecolor='black',markeredgewidth=0.1)
            plt.legend(labels=legend, loc='right', fontsize=8)
            plt.grid(which='both',linestyle='--',alpha=0.7)
            plt.xlabel("Function Evaluations")
            plt.ylabel(key)
            plt.tight_layout()
            plt.savefig(f"{file_path}/{key}.png", dpi=600, bbox_inches='tight')
            plt.close()

    def plot_pareto_front(self, objective_values, tf, legend, file_path):
        n_obj = objective_values.shape[1]
        if n_obj == 2:
            plt.figure()
            plt.plot(objective_values[:,0], objective_values[:,1], linestyle='',marker='s',
                    markerfacecolor='cyan',markersize='5'
                    ,markeredgecolor='black',markeredgewidth=0.1)
            if tf is not None:
                plt.plot(tf[:,0],tf[:,1],linestyle='',marker='.',color='black'
                        ,markersize='5',alpha=1)
                legend.append("True Front")
            plt.legend(labels=legend, loc='upper right', fontsize=8)
            plt.grid(which='both',linestyle='--',alpha=0.7)
            plt.xlabel("f1")
            plt.ylabel("f2")
            plt.tight_layout()
            plt.savefig(f"{file_path}/pareto_front.png", dpi=600, bbox_inches='tight')
            plt.close()
        
        if n_obj == 3:
            plt.figure()
            ax = plt.axes(projection='3d')
            ax.view_init(elev=30, azim=30)
            ax.set_xlabel("f1")
            ax.set_ylabel("f2")
            ax.set_zlabel("f3")
            
            plt.plot(objective_values[:,0], objective_values[:,1],objective_values[:,2], linestyle='',marker='s',
                        markerfacecolor='cyan',markersize='5',markeredgecolor='black',markeredgewidth=0.1)
            if tf is not None:
                plt.plot(tf[:,0],tf[:,1],tf[:,2],linestyle='',marker='.',color='black',markersize='5')
                legend.append("True Front")
            plt.legend(labels=legend, loc='upper right', fontsize=8)
            ax.grid(which='both',linestyle='--',alpha=0.3)
            plt.savefig(f"{file_path}/pareto_front.png", dpi=600, bbox_inches='tight')
            plt.close()

    def generate_plots(self, result, save_path):
        _,pareto_front,_,_ = result.population.get_pareto()
        legend = [result.algorithm.get_info()["name"]]
        self.plot_convergence(result.get_convergence_data(), legend, save_path)
        self.plot_pareto_front(pareto_front,
                                result.problem.get_true_front(),
                                legend,
                                save_path)

def compare_base(problem_name, selection_metric = "HV", minmax = 1, dir = None):
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
            pareto_front_list.append(temp)

        pareto_df = pd.concat(pareto_front_list, ignore_index=True)

        headers = list(convergence_data_list[0].columns)
        legend = df["algorithm"]
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
        sc.set_titles("")
        sc.legend.set_title("Algorithm")
        plt.savefig(f"{comparison_path}/{problem_name}_pareto_fronts_comparison.png", dpi=600, bbox_inches='tight')
        plt.close()


def plot_pareto_fronts(pareto_df, external_df, problem_name, comparison_path):
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
            x="x", y="y",
            color="indigo", marker="X", s=170, ax=ax, label=external_df["algorithm"].iloc[0]
        )

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

def compare_solutions(problem_name, objectives_dict,
                      selection_metric = "HV", minmax = 1, dir = None):
    root_dir = Path(dir) if dir else Path(__main__.__file__).parent.resolve()
    master_list_path = Path(f"{root_dir}/results/{problem_name}")
    master_lists = [a for a in master_list_path.iterdir() if not a.is_dir()]
    external_df = pd.DataFrame(objectives_dict)
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

        # pareto_front_list.append(external_df)
        pareto_df = pd.concat(pareto_front_list, ignore_index=True)
        plot_pareto_fronts(pareto_df, external_df, problem_name, comparison_path)
        # plt.figure()
        # sc = sns.relplot(data = pareto_df, x = "f1", y = "f2", hue = "algorithm", style = "algorithm", s = 50,
        #                  col = "algorithm", col_wrap = 2,alpha = 1, linewidth = 0.3, facet_kws = {'legend_out':False})
        # # sc.get_legend().set_title(None)]
        # sc.map_dataframe(
        #     lambda data, color, **kwargs: sns.scatterplot(
        #         data=external_df, x="x", y="y", color="red", marker="X", s=60, **kwargs
        #     )
        # )
        # plt.tight_layout()
        # for ax in sc.axes.flat:
        #     ax.grid(which='both',linestyle='--',alpha=0.7)
        #     ax.spines['top'].set_visible(True)
        #     ax.spines['right'].set_visible(True)
        #     ax.spines['bottom'].set_visible(True)
        #     ax.spines['left'].set_visible(True)
        # sc.set_titles("")
        # sc.figure.axes[0].scatter([], [], color="red", marker="X", s=60, label="Reference (paper)")
        # sc.add_legend()
        # sc.legend.set_title("Algorithm")
        # plt.savefig(f"{comparison_path}/{problem_name}_pareto_fronts_comparison.png", dpi=600, bbox_inches='tight')
        # plt.close()
