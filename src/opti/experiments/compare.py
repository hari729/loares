from operator import mul
from pathlib import Path
import os
import __main__
from matplotlib._api import recursive_subclasses
import pandas as pd
import numpy as np
import re
from opti.analysis.plots import multi_line_plot, plot_2d, plot_3d, parallel_coordinates_plot

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

def compare_experiments(problem, test_name, selction_metric='HV'):
    problem_info = problem.get_info()
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
    for path in result_paths:
        temp = {'pf': pd.read_csv(path/"pareto-front.csv"),
                'Info': pd.read_json(path/"Info.json"),
                'mean-history': pd.read_csv(path/"mean-history.csv"),
                'convergence-pts': pd.read_csv(path/"convergence-points.csv")}
        name = temp['Info']['Algorithm']['name']
        if re.search(r'\bBMWR\b', name):
            results["BMWR"].append(temp)
        elif re.search(r'\bBMR\b', name):
            results["BMR"].append(temp)
        elif re.search(r'\bBWR\b', name):
            results["BWR"].append(temp)
        else:
            others.append(temp)

    metrics = results['BMR'][0]['convergence-pts'].columns.tolist()
    metrics.remove('name')
    print(metrics)

    # Sort each category (BMR, BWR, BMWR) internally
    for rt in results:
        results[rt].sort(key=get_suffix_priority)   
    
    for rc in results:
        for m in metrics:
            plot_data = {'ydata' : [],
                        'xdata': [],
                        'xlabel' : "Mean-Function-Evaluations",
                        'point' : [],
                        'legend':[]}
            for r in results[rc]:
                plot_data['ylabel']=f"{m}"
                plot_data['ydata' ].append(r['mean-history'][m])
                plot_data['xdata'].append(r['mean-history']['evals']) 
                plot_data['point' ].append(r['convergence-pts'][m])
                plot_data['legend'].append(r['Info']['Algorithm']['name'])
            multi_line_plot(plot_data, comparison_dir, f"{m}-{rc}")




