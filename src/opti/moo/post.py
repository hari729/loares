from pathlib import Path
import os
import __main__
import pandas as pd
import numpy as np
from tqdm import tqdm
from opti.moo.plots import plot_pareto_2d_comparison, plot_pareto_2d_comparison_external, plot_pareto_3d_individual
from opti.moo.plots import plot_convergence_comparison, plot_pareto_3d_many_obj
from pymoo.util.normalization import normalize

# def find_compromise_solution(pareto_df, comparison_path, problem):
#
#     algorithms = pareto_df["algorithm"].unique()
#     obj_cols = [f"f{o+1}" for o in range(problem.n_obj)]
#     best_solutions = []
#     for algo in algorithms:
#         df = pareto_df[pareto_df["algorithm"] == algo]
#         flipped = df[obj_cols].values * problem.minmax 
#         optimal_point = flipped.min(axis=0)
#         direct_ed = np.linalg.norm(flipped - optimal_point, axis=1)
#         df = df.assign(direct_ed= direct_ed)
#         df.sort_values(by="direct_ed", ascending=True)
#         best_solutions.append( df.head(1))
#         normalized = normalize(flipped, flipped.min(axis=0), flipped.max(axis=0))
#         normalized_ed = np.linalg.norm(normalized, axis=1)
#         df["normalized_ed"] = normalized_ed
#         df.sort_values(by="normalized_ed", ascending=True)
#         best_solutions.append(df.head(1))
#
#     best_df = pd.concat(best_solutions, ignore_index=True)
#     output_path = comparison_path / f"best-solutions.csv"
#     best_df.to_csv(output_path, index=False)
#     tqdm.write(f"✅ Saved best-solutions to → {output_path}")

def get_master_dfs(problem, selection_metric = "HV", minmax = 1, dir = None):
    problem_name = problem.get_info()["name"]
    root_dir = Path(dir) if dir else Path(__main__.__file__).parent.resolve()
    master_list_path = Path(f"{root_dir}/results/{problem_name}")
    all_lists = [a for a in master_list_path.iterdir() if not a.is_dir()]
    master_data_frames = []
    for master_list in all_lists:
        temp = pd.read_csv(master_list)
        master_data_frames.append({"class":master_list.stem, "dataF":temp})
    return master_data_frames

def basic_compare(problem, selection_metric = "HV", minmax = 1, dir = None):
    problem_info = problem.get_info()
    problem_name = problem_info["name"]
    n_obj = problem_info["n_obj"]
    root_dir = Path(dir) if dir else Path(__main__.__file__).parent.resolve()
    master_list_path = Path(f"{root_dir}/results/{problem_name}")
    master_lists = [a for a in master_list_path.iterdir() if not a.is_dir()]
    for master_list in tqdm(master_lists, desc="Running Comparison"):
        comparison_path = Path(f"{master_list_path}/comparison/{master_list.stem}")
        os.makedirs(comparison_path, exist_ok=True)
        master_df = pd.read_csv(master_list)
        df = master_df.loc[master_df.groupby("algorithm")[f"{selection_metric}"].idxmax()]
        df.to_csv(f"{comparison_path}/best-metrics.csv")
        convergence_data_list = []
        pareto_front_list = []
        for index, row in df.iterrows():
            convergence_data_list.append(pd.read_csv(f"{row["save_path"]}/convergence.csv"))
            temp = pd.read_csv(f"{row["save_path"]}/solutions.csv")
            temp["algorithm"] = row["algorithm"]
            temp["psize"] = row["psize"]
            pareto_front_list.append(temp)

        pareto_df = pd.concat(pareto_front_list, ignore_index=True)
        
        # find_compromise_solution(pareto_df, comparison_path, problem)

        plot_convergence_comparison(convergence_data_list, df["algorithm"], comparison_path, problem_name)
        
        if n_obj == 2:
            plot_pareto_2d_comparison(pareto_df, comparison_path, problem_name)
        if n_obj == 3:
            plot_pareto_3d_individual(pareto_df, comparison_path, problem_name)
        if n_obj > 3:
            plot_pareto_3d_many_obj(pareto_df, comparison_path, problem_name, n_obj)


def compare_solutions(problem, objectives_dict,
                      selection_metric = "HV", minmax = 1, dir = None):
    problem_name = problem.get_info()["name"]
    root_dir = Path(dir) if dir else Path(__main__.__file__).parent.resolve()
    master_list_path = Path(f"{root_dir}/results/{problem_name}")
    master_lists = [a for a in master_list_path.iterdir() if not a.is_dir()]
    external_df = pd.DataFrame(objectives_dict)
    obj_cols = [f"f{o+1}" for o in range(problem.n_obj)]
    for master_list in tqdm(master_lists, desc="Running external comparison"):
        comparison_path = Path(f"{master_list_path}/compare-solutions/{master_list.stem}")
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

            output_path = comparison_path / f"{row['algorithm']}-matches.csv"
            matched_df.to_csv(output_path, index=False)
            tqdm.write(f"✅ Saved matches for {row['algorithm']} → {output_path}")

        pareto_df = pd.concat(pareto_front_list, ignore_index=True)

        if problem.n_obj == 2:
            plot_pareto_2d_comparison_external(pareto_df, external_df, problem_name, comparison_path)

        else:
            print("Plotter not defined yet")

