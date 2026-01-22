from pathlib import Path
import pickle
import gzip
import os
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

def extract_population_paths(test_dir, psize, evals):
    algo_paths = [a for a in test_dir.iterdir() if a.is_dir()]
    result_paths = []
    for path in algo_paths:
        result_paths.append(path/f"{psize}-{evals}")
    return result_paths

def compare_experiments(problem, test_name, selction_metric='HV', CTF=False):
    problem_info = problem.get_info()
    print(f"Comparing {test_name} test for {problem_info['name']}  with CTF={CTF}")
    test_dir = (Path.home()/"OptiResults"
                            /problem_info["name"]
                            /test_name)
    comparison_dir = (Path.home()/"OptiResults"
                            /problem_info["name"]
                            /f"{test_name}-comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    result_paths = extract_results_paths(test_dir, selction_metric)
    results = {"BMR": [], "BWR": [], "BMWR":[]}
    others = []
    pareto_list = []
    for path in result_paths:
        with gzip.open(path / "results.pkl.gz", 'rb') as f:
            results_obj_list = pickle.load(f)
        temp = {
                'pf': pd.read_csv(path/"pareto-front.csv"),
                'Info': pd.read_json(path/"Info.json"),
                'res_obj': results_obj_list
                }
        final_metrics = pd.read_csv(path/"final-metrics-per-run.csv")
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

    print(len(pareto_list))
    TF = None
    if CTF:
        populationHandler = MOPopulationHandler()
        combined_pop = populationHandler.merge(pareto_list)
        composite_population_raw = Population(*ranking_crowding(
                                                problem, combined_pop, 100,
                                                ndf=True, seed = 1))
        composite_population = populationHandler.get_refined(composite_population_raw)
        TF = composite_population.objectives

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

    # print(algo)
    metrics = [k for k in results['BMR'][0]['convergence-pts'] if k != 'name']
    print(metrics)

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


def compare_experiments_all(problem, test_name, psizes, CTF=False):
    problem_info = problem.get_info()
    test_dir = (Path.home()/"OptiResults"
        /problem_info["name"]
        /test_name)
    main_comparison_dir = (Path.home()/"OptiResults"
                            /problem_info["name"]
                            /f"{test_name}-comparison")
    for psize in psizes:
        print(f"Comparing {test_name} test for {problem_info['name']} at Psize = {psize} with CTF={CTF}")
        comparison_dir = Path(main_comparison_dir/f"{psize}")
        os.makedirs(comparison_dir, exist_ok=True)

        result_paths = extract_population_paths(test_dir, psize, problem_info['max_evals'])
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
        if CTF:
            populationHandler = MOPopulationHandler()
            combined_pop = populationHandler.merge(pareto_list)
            composite_population_raw = Population(*ranking_crowding(
                                                    problem, combined_pop, 100,
                                                    ndf=True, seed = 1))
            composite_population = populationHandler.get_refined(composite_population_raw)
            TF = composite_population.objectives

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

        # print(algo)
        metrics = [k for k in results['BMR'][0]['convergence-pts'] if k != 'name']
        print(metrics)

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
