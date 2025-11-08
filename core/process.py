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
        self.problem_info = self.results[0].problem.get_info()
        self.root_dir = Path(root_dir) if root_dir else Path(__main__.__file__).parent.resolve()
        self.save_path = None
        self.master_list_path = None

    def new_set(self, new_list):
        self.results = new_list
        self.problem_info = self.results[0].problem.get_info()
        self.algorithm_info = self.results[0].algorithm.get_info()

    def set_path(self):
        problem_name = self.problem_info['name']
        algo_name = self.results[0].algorithm.get_info()["name"]
        psize = self.problem_info['psize']
        max_evals = self.problem_info['max_evals']
        self.save_path = f"{self.root_dir}/results/{problem_name}/{algo_name}/{psize}_{max_evals}"
        os.makedirs(self.save_path, exist_ok=True)
        self.master_list_path = Path(f"{self.root_dir}/results/{problem_name}/master_list.csv")

    def generate_plots(self,*args):
        pass

    def generate_results(self):
        self.set_path()
        master_list = []
        for result in self.results:
            algorithm_info = result.algorithm.get_info()
            save_path = f"{self.save_path}/{algorithm_info["seed"]}"
            os.makedirs(save_path, exist_ok=True)

            master_list.append({"algorithm" : algorithm_info['name'],
                                "seed" : algorithm_info['seed'],
                                "psize" : self.problem_info["psize"],
                                "max_evals" : self.problem_info["max_evals"],
                                **result.final_metrics})

            with open(f"{save_path}/problem_settings.json", "w") as write_file:
                json.dump(self.problem_info, write_file)

            with open(f"{save_path}/algorithm_settings.json", "w") as write_file:
                json.dump(algorithm_info, write_file)

            df = pd.DataFrame(result.population.get_pareto_dict())
            df.to_csv(f"{save_path}/solutions.csv", index=False)

            df = pd.DataFrame(result.get_convergence_data())
            df.to_csv(f"{save_path}/convergence.csv", index=False)

            self.generate_plots(result, save_path)


        mf = pd.DataFrame(master_list)
        if self.master_list_path.exists():
            mf.to_csv(self.master_list_path, mode='a', header=False, index = False)
        else:
            mf.to_csv(self.master_list_path, mode='w', header=True, index = False)
