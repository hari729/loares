import json
from pathlib import Path
import numpy as np

def extract(test_name,problem,algo,psize):
    results_path = Path(__file__).resolve().parent.parent/'results'
    data_path =f"{results_path}/{test_name}/{problem}/{algo.upper()}/{psize}" 
    with open(f"{data_path}/settings.json") as f:
        temp_dict = json.load(f)

    temp_dict.update(np.load(f"{data_path}/convergence_data.npz"))

    temp_dict.update(np.load(f"{data_path}/pareto_data.npz"))

    return(temp_dict)

if __name__ == "__main__":

    print(extract("npz_test_20251017_113101","mou.zdt2","bwr","100"))
