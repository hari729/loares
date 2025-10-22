import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import argparse

def extract(test_name,problem,algo,psize):
    results_path = Path(__file__).resolve().parent.parent/'results'
    data_path =f"{results_path}/{test_name}/{problem}/{algo.upper()}/{psize}" 
    with open(f"{data_path}/settings.json") as f:
        temp_dict = json.load(f)

    temp_dict.update(np.load(f"{data_path}/convergence_data.npz"))

    temp_dict.update(np.load(f"{data_path}/pareto_data.npz"))

    return(temp_dict)

def compare(list_of_result_paths,comparison_result_path):

    save_path = Path(__file__).resolve().parent.parent/'results'/'comparison'/comparison_result_path
    os.makedirs(save_path, exist_ok=True)
    data_dict = []
    for data_path in list_of_result_paths:
        
        with open(f"{data_path}/settings.json") as f:
            temp_dict = json.load(f)

        temp_dict.update(np.load(f"{data_path}/convergence_data.npz"))

        temp_dict.update(np.load(f"{data_path}/pareto_data.npz"))

        data_dict.append(temp_dict)

    metrics = data_dict[0]['metrics']["headers"]

    for key in metrics:
        if key == 'evals':
            continue
        
        plt.figure()
        legend = []
        for data in data_dict:
            plt.plot(data['evals'], data[key], linestyle='-',marker='')
            legend.append(f"MO-{data['algo_name'].upper()}" +
                (f" - {data['selection_pool']}" if data['selection_pool'] != 'population' else "")+ 
                (f"{" - ".join(f"{p}" for p in ['']+data['modifiers'] if p != 'local_search')}"))

        plt.legend(labels=legend, loc='right', fontsize=8)
        plt.grid(which='both',linestyle='--',alpha=0.7)
        plt.xlabel("Function Evaluations")
        plt.ylabel(key)
        plt.tight_layout()
        plt.savefig(f"{save_path}/{key}_comparison.png", dpi=600, bbox_inches='tight')
        plt.close()

def run_comparison(list_of_tests):

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
    test_path = Path(f"/home/hari/projects/opti/results/{list_of_tests[0]}")
    problems = [p.name for p in test_path.iterdir() if p.is_dir()]
    algos = ['BMR','BWR','BMWR']

    for prob in problems:
        for algo in algos:
            pops_path = Path(f"/home/hari/projects/opti/results/{list_of_tests[0]}/{prob}/{algo}")
            psizes = [ps.name for ps in pops_path.iterdir() if ps.is_dir()]
            for psize in psizes:
                list_of_result_paths = [f"/home/hari/projects/opti/results/{test}/{prob}/{algo}/{psize}"
                                        for test in list_of_tests]
                compare(list_of_result_paths,f"{timestamp}/{prob}/{algo}/{psize}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate comparison")

    parser.add_argument("--test_names", "-t", nargs="+" , type=str, required=True, help="List of tests")

    run_comparison(parser.parse_args().test_names)
