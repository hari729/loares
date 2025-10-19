import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os 

def extract(test_name,problem,algo,psize):
    results_path = Path(__file__).resolve().parent.parent/'results'
    data_path =f"{results_path}/{test_name}/{problem}/{algo.upper()}/{psize}" 
    with open(f"{data_path}/settings.json") as f:
        temp_dict = json.load(f)

    temp_dict.update(np.load(f"{data_path}/convergence_data.npz"))

    temp_dict.update(np.load(f"{data_path}/pareto_data.npz"))

    return(temp_dict)

def compare(list_of_result_paths):
    save_path = Path(__file__).resolve().parent.parent/'results'/'comparison'
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
            legend.append(f"{data['algo_name'].upper()}-{data['selection_pool']}-{data['modifiers']}")

        plt.legend(labels=legend, loc='right', fontsize=8)
        plt.grid(which='both',linestyle='--',alpha=0.7)
        plt.xlabel("Function Evaluations")
        plt.ylabel(key)
        plt.tight_layout()
        plt.savefig(f"{save_path}/{key}_comparison.png", dpi=600, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":

    # print(extract("npz_test_20251017_113101","mou.zdt2","bwr","100").keys())

    compare([
        '/home/hari/projects/opti/results/regular+local_search_20251019_143059/mou.zdt1/BMR/100/',
        '/home/hari/projects/opti/results/archive+local_search_20251019_143204/mou.zdt1/BMR/100/'
    ])
