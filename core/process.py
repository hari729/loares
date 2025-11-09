import numpy as np
import datetime
import sys
import os
import __main__
from pathlib import Path
import csv
import json
import pandas as pd

class ResultProcessor():
    def __init__(self,
                 results_list,
                 root_dir = None):
        self.results = results_list
        self.root_dir = Path(root_dir) if root_dir else Path(__main__.__file__).parent.resolve()

    def new_set(self, new_list):
        self.results = new_list

    def set_path(self, result):
        problem_info = result.problem.get_info()
        algo_info = result.algorithm.get_info()
        psize = problem_info['psize']
        max_evals = problem_info['max_evals']
        seed = algo_info["seed"]
        save_path = f"{self.root_dir}/results/{problem_info["name"]}/{algo_info["name"]}/{psize}_{max_evals}/{seed}"
        os.makedirs(save_path, exist_ok=True)
        master_list_path = Path(f"{self.root_dir}/results/{problem_info["name"]}/{algo_info["BaseFunction"]}.csv")
        return problem_info, algo_info, save_path, master_list_path

    def generate_plots(self,*args):
        pass

    def generate_results(self):
        master_list = []
        for result in self.results:
            # print(result.algorithm.get_info())
            problem_info, algorithm_info, save_path, master_list_path = self.set_path(result)

            master_list.append({"problem" : problem_info["name"],
                                "algorithm" : algorithm_info['name'],
                                "psize" : problem_info["psize"],
                                "max_evals" : problem_info["max_evals"],
                                "seed" : algorithm_info['seed'],
                                **result.final_metrics,
                                "save_path": save_path})

            with open(f"{save_path}/problem_settings.json", "w") as write_file:
                json.dump(problem_info, write_file)

            with open(f"{save_path}/algorithm_settings.json", "w") as write_file:
                json.dump(algorithm_info, write_file)

            df = pd.DataFrame(result.population.get_pareto_dict())
            df.to_csv(f"{save_path}/solutions.csv", index=False)

            df = pd.DataFrame(result.get_convergence_data())
            df.to_csv(f"{save_path}/convergence.csv", index=False)

            self.generate_plots(result, save_path)


        mf = pd.DataFrame(master_list)
        if master_list_path.exists():
            existing = pd.read_csv(master_list_path)
            combined = pd.concat([existing, mf], ignore_index=True)
            combined.drop_duplicates(subset=["algorithm", "psize", "max_evals", "seed"], keep="last", inplace=True)
            combined.to_csv(master_list_path, index=False)
        else:
            mf.to_csv(master_list_path, mode='w', header=True, index = False)
