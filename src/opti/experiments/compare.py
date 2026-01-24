from pathlib import Path
import pickle
import gzip
import sys
import os
from multiprocessing import Pool
import __main__
from typing import final
from opti.algorithms.moo.sorting import ranking_crowding
import pandas as pd
import numpy as np
import re
from opti.core.population import Population
from opti.analysis.plots import multi_line_plot, plot_2d, plot_3d, parallel_coordinates_plot
from opti.analysis.moo.metrics import raw_performance_metrics
from opti.algorithms.moo.base import MOPopulationHandler
from opti.core.results import ResultProcessor
from pymoo.algorithms.moo.mopso_cd import MOPSO_CD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def get_suffix_priority(result_dict):
    name = result_dict['Info']['Algorithm']['name']
    # Priority order: None -> archive -> opposition -> samp
    if name.endswith('SAMP'):
        return 3
    if name.endswith('OPPOSITION'):
        return 2
    if name.endswith('ARCHIVE'):
        return 1
    return 0  # No suffix (Base version)

resultProcessor = ResultProcessor()

def extract_results_paths(test_dir, selction_metric):
    algo_paths = [a for a in test_dir.iterdir() if a.is_dir()]
    result_paths = []
    for path in algo_paths:
        master_df = pd.read_csv(path/"master.csv")
        best_idx = master_df[f"{selction_metric}(mean)"].idxmax()
        psize = master_df['Psize'][best_idx]
        evals = master_df['Max-evals'][best_idx]
        result_paths.append(path/f"{psize}-{evals}")
    return result_paths

def extract_population_paths_custom(test_dir, psize, evals):
    algo_paths = [a for a in test_dir.iterdir() if a.is_dir()]
    result_paths = []
    for path in algo_paths:
        if re.search(r'\bBMWR\b', str(path)) or re.search(r'\bBMR\b', str(path)) or re.search(r'\bBWR\b', str(path)):
            result_paths.append(path/f"{psize}-{evals}")
        else:
            result_paths.append(path/f"{100}-{evals}")
    return result_paths

def extract_population_paths(test_dir, psize, evals):
    algo_paths = [a for a in test_dir.iterdir() if a.is_dir()]
    result_paths = []
    for path in algo_paths:
        result_paths.append(path/f"{psize}-{evals}")
    return result_paths

# Define which metrics to minimize vs maximize (for mean values)
# Only HV(mean) is higher is better, all others are lower is better
MINIMIZE_MEAN_METRICS = ["GD(mean)", "IGD(mean)", "SPC(mean)", "SPR(mean)"]
MAXIMIZE_MEAN_METRICS = ["HV(mean)"]

# All std metrics - lower is better (more consistent)
STD_METRICS = ["GD(std)", "IGD(std)", "SPC(std)", "SPR(std)", "HV(std)"]
STD_METRICS = []

def get_best_algorithm(df: pd.DataFrame, metric: str, minimize: bool = True) -> str:
    """
    Find the algorithm with the best value for a given metric.

    Args:
        df: DataFrame with algorithm results
        metric: Column name of the metric
        minimize: If True, lower is better; if False, higher is better

    Returns:
        Name of the best performing algorithm
    """
    if metric not in df.columns:
        return "N/A"

    if minimize:
        idx = df[metric].idxmin()
    else:
        idx = df[metric].idxmax()

    # return df.loc[idx, "Algorithm"]
    return df.loc[idx, "Algorithm"], df.loc[idx, metric]

def process_comparison_folder(comparison_folder: Path) -> pd.DataFrame:
    """
    Process a comparison folder and find best algorithms for each population size.

    Args:
        comparison_folder: Path to the comparison folder

    Returns:
        DataFrame with population sizes and best algorithms per metric
    """
    results = []

    # Find all population subfolders
    population_folders = sorted(
        [f for f in comparison_folder.iterdir() if f.is_dir() and f.name.isdigit()],
        key=lambda x: int(x.name),
    )

    if not population_folders:
        print(f"  No population subfolders found in {comparison_folder}")
        return pd.DataFrame()

    for pop_folder in population_folders:
        net_results_path = pop_folder / "net-results.csv"

        if not net_results_path.exists():
            print(f"  Warning: net-results.csv not found in {pop_folder}")
            continue

        df = pd.read_csv(net_results_path)

        # Get population size and max evals from the data
        psize = int(pop_folder.name)

        # Try to get Max-evals from the data (use first row's value)
        max_evals = df["Max-evals"].iloc[0] if "Max-evals" in df.columns else "N/A"

        row = {
            "Population": psize,
            "Max-evals": max_evals,
        }

        # Find best algorithm for metrics to minimize (mean)
        for metric in MINIMIZE_MEAN_METRICS:
            best_algo,best_value = get_best_algorithm(df, metric, minimize=True)
            row[metric] = best_algo
            row[f"{metric}_value"] = best_value
        # Find best algorithm for metrics to maximize (mean)
        for metric in MAXIMIZE_MEAN_METRICS:
            best_algo ,best_value= get_best_algorithm(df, metric, minimize=False)
            row[metric] = best_algo
            row[f"{metric}_value"] = best_value
        # Find best algorithm for std metrics (always minimize - lower variance is better)
        for metric in STD_METRICS:
            best_algo,best_value = get_best_algorithm(df, metric, minimize=True)
            row[metric] = best_algo
            row[f"{metric}_value"] = best_value
        results.append(row)

    # Create DataFrame with results
    if results:
        result_df = pd.DataFrame(results)
        # Order columns to match net-results.csv structure
        cols = [
                    "Population",
                    "Max-evals",
                    "GD(mean)",
                    "GD(mean)_value",
                    "GD(std)",
                    "GD(std)_value",
                    "IGD(mean)",
                    "IGD(mean)_value",
                    "IGD(std)",
                    "IGD(std)_value",
                    "SPC(mean)",
                    "SPC(mean)_value",
                    "SPC(std)",
                    "SPC(std)_value",
                    "SPR(mean)",
                    "SPR(mean)_value",
                    "SPR(std)",
                    "SPR(std)_value",
                    "HV(mean)",
                    "HV(mean)_value",
                    "HV(std)",
                    "HV(std)_value",
                ]
        result_df = result_df[[c for c in cols if c in result_df.columns]]
        return result_df

    return pd.DataFrame()

def compare_metrics(problem_name, compare_dir_path):
    """Main function to process a specific test's comparison folder."""


    # Build the comparison folder path
    comparison_folder = Path(compare_dir_path)

    if not comparison_folder.exists():
        print(f"Error: Comparison folder not found: {comparison_folder}")
        sys.exit(1)

    if not comparison_folder.is_dir():
        print(f"Error: Path is not a directory: {comparison_folder}")
        sys.exit(1)

    print(f"Processing: {comparison_folder}")

    result_df = process_comparison_folder(comparison_folder)

    if result_df.empty:
        print(f"  No valid data found in {comparison_folder}")
        sys.exit(1)

    # Save the result CSV just outside the comparison folder
    # Name it the same as the comparison folder name
    output_name = comparison_folder.name + ".csv"
    output_path = comparison_folder / output_name

    result_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"\nResults:")
    print(result_df.to_string(index=False))

class compare_experiments_all():
    def __init__(self, problem, test_name, psizes, CTF=False, all=False):
        self.problem = problem
        self.problem_info = problem.get_info()
        self.test_name = test_name
        self.psizes = psizes
        self.CTF = CTF
        self.all = all
        self.test_dir = (Path.home()/"OptiResults"
                                    /self.problem_info["name"]
                                    /self.test_name)
        suffix = ''
        if CTF:
            suffix += "-CTF"
            if all:
                suffix += "-FLL"
            else:
                suffix += "-Psize"
        self.main_comparison_dir = (Path.home()/"OptiResults"
                                /self.problem_info["name"]
                                /f"{self.test_name}-comparison{suffix}")

        print(f"Comparing {self.test_name} test for {self.problem_info['name']} " 
                f"with CTF={self.CTF} and CTF size ={'Full' if self.all else 'Psize'}"
              )
    def run(self, psize):
        print(f"at Psize = {psize}")
        comparison_dir = Path(self.main_comparison_dir/f"{psize}")
        os.makedirs(comparison_dir, exist_ok=True)

        result_paths = extract_population_paths(self.test_dir, psize, self.problem_info['max_evals'])
        results = {"BMR": [], "BWR": [], "BMWR":[]}
        others = []
        pareto_list = []
        for path in result_paths:
            with gzip.open(path / "results.pkl.gz", 'rb') as f:
                results_obj_list = pickle.load(f)
            temp = {
                    # 'pf': pd.read_csv(path/"pareto-front.csv"),
                    'Info': pd.read_json(path/"Info.json"),
                    'res_obj': results_obj_list
                    }
            final_metrics = pd.read_csv(path/"raw-results.csv")
            best_seed = final_metrics['seed'][final_metrics['HV'].idxmax()]
            for robj in results_obj_list:
                if robj.seed == best_seed:
                    pareto_list.append(robj.population)
            name = temp['Info']['Algorithm']['name']
            if re.search(r'\bBMWR\b', name):
                results["BMWR"].append(temp)
            elif re.search(r'\bBMR\b', name):
                results["BMR"].append(temp)
            elif re.search(r'\bBWR\b', name):
                results["BWR"].append(temp)
            else:
                others.append(temp)
        for rt in results:
            results[rt].sort(key=get_suffix_priority)
        others.sort(key=lambda x: x['Info']['Algorithm']['name'])
        results['others'] = others

        TF = None
        if self.CTF:
            CTf_path = Path(comparison_dir / "composite_true_front.npy")
            if CTf_path.exists():
                TF = np.load(CTf_path)
            else:
                populationHandler = MOPopulationHandler()
                combined_pop = populationHandler.merge(pareto_list)
                composite_population_raw = Population(*ranking_crowding(
                                                        self.problem, combined_pop, psize,
                                                        ndf=True, seed = 1, all=self.all))
                composite_population = populationHandler.get_refined(composite_population_raw)

                TF = composite_population.objectives
                np.save(comparison_dir / "composite_true_front.npy", TF)

        for algo_class in results:
            for algo in results[algo_class]:
                output = algo['res_obj']
                metrics_list = []
                for res in output:
                    temp_dict = resultProcessor.get_metrics_history(res, raw_performance_metrics,
                                                                    TF=TF)
                    temp_dict["seed"] = [res.seed]
                    metrics_list.append(temp_dict)

                metrics = metrics_list[0].keys()
                mean = {"name": f"{algo['Info']['Algorithm']['name']} (Mean)"}
                std = {"name": f"{algo['Info']['Algorithm']['name']} (Std)"}
                net = {'Psize':algo['Info']['Problem']['psize'],
                    'Max-evals':algo['Info']['Problem']['max_evals']}

                recording_interval = int(algo['Info']['Problem']['max_evals'] * 0.05)
                eval_grid = np.arange(recording_interval, 
                                    algo['Info']['Problem']['max_evals'] + 1, 
                                    recording_interval)
                # Interpolate each run's metrics to the common grid
                mean['evals'] = eval_grid
                std['evals'] = eval_grid
                convergence = {"name": f"{algo['Info']['Algorithm']['name']} (convergence pts)"}
                ind_metrics = []
                for m in metrics:
                    if m not in ["seed", "evals"]:
                        interpolated_values = []
                        for r in metrics_list:
                            # Linear interpolation to common evaluation grid
                            interp_vals = np.interp(eval_grid, r['evals'], r[m])
                            interpolated_values.append(interp_vals)
                        
                        values = np.array(interpolated_values, dtype=float)
                        mean[m] = np.mean(values, axis=0)
                        std[m] = np.std(values, axis=0)
                        
                        # Rest of convergence logic remains the same...
                        delta = np.diff(mean[m]) / mean[m][:-1]
                        convergence_pt = np.where(np.abs(delta) < 1e-3)[0]
                        convergence[m] = [np.nan, np.nan]
                        if len(convergence_pt) > 1:
                            cidx = np.where(np.diff(convergence_pt) == 1)[0]
                            if len(cidx) > 0:
                                idx = cidx[0]
                                convergence[m] = [mean[m][convergence_pt[idx]], 
                                                mean['evals'][convergence_pt[idx]]]


                        net[f"{m}(mean)"] = [mean[m][-1]]
                        net[f"{m}(std)"] = [std[m][-1]]

                algo['mean-history'] = pd.DataFrame(mean)
                algo['convergence-pts'] = pd.DataFrame(convergence)
                algo['net-result'] = pd.DataFrame(net)

        metrics = [k for k in results['BMR'][0]['convergence-pts'] if k != 'name']

        net_res = {}
        for rc in results:
            if rc not in ['others']:
                for m in metrics:
                    plot_data = {'ydata' : [],
                                'xdata': [],
                                'xlabel' : "Function Evaluations",
                                'point' : [],
                                'legend':[]}
                    for r in results[rc]+results['others']:
                        plot_data['ylabel']=f"{m}"
                        plot_data['ydata' ].append(r['mean-history'][m])
                        plot_data['xdata'].append(r['mean-history']['evals']) 
                        plot_data['point' ].append(r['convergence-pts'][m])
                        plot_data['legend'].append(r['Info']['Algorithm']['name'])
                    multi_line_plot(plot_data, comparison_dir, f"{m}-{rc}")
            
            for rn in results[rc]:
                net_res[rn['Info']['Algorithm']['name']] = rn['net-result']

        net_res = pd.concat(net_res, names=["Algorithm"]).reset_index(level=0)
        net_res.to_csv(f"{comparison_dir}/net-results.csv", index=False, float_format="%.5f")

    def multi_thread(self, threads=5):
        with Pool(processes=threads) as pool:
            pool.map(self.run, self.psizes)
        return self.main_comparison_dir



