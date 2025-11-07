import numpy as np
import datetime
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import csv
import json
import pandas as pd

class ResultProcessor():
    def __init__(self,
                 results_list,
                 root_dir = None):
        self.results = results_list
        self.problem_info = self.results[0].problem.get_info()
        self.algorithm_info = self.results[0].algorithm.get_info()
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.save_path = None

    def new_set(self, new_list):
        self.results = new_list
        self.problem_info = self.results[0].problem.get_info()
        self.algorithm_info = self.results[0].algorithm.get_info()

    def set_path(self):
        problem_name = self.problem_info['name']
        algo_name = self.algorithm_info['name']
        psize = self.problem_info['psize']
        max_evals = self.problem_info['max_evals']
        self.save_path = f"{self.root_dir}/results/{problem_name}/{algo_name}/{psize}_{max_evals}"
        os.makedirs(self.save_path, exist_ok=True)

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

    def generate_results(self):
        self.set_path()
        for i, result in enumerate(self.results):
            save_path = f"{self.save_path}/{i+1}"
            os.makedirs(save_path, exist_ok=True)

            with open(f"{save_path}/problem_settings.json", "w") as write_file:
                json.dump(self.problem_info, write_file)

            with open(f"{save_path}/algorithm_settings.json", "w") as write_file:
                json.dump(self.algorithm_info, write_file)

            df = pd.DataFrame(result.get_pareto_dict())
            df.to_csv(f"{save_path}/solutions.csv", index=False)

            df = pd.DataFrame(result.get_convergence_data())
            df.to_csv(f"{save_path}/convergence.csv", index=False)
            
            _,pareto_front,_,_ = result.final_population.get_pareto()
            legend = [self.algorithm_info["name"]]
            self.plot_convergence(result.get_convergence_data(), legend, save_path)
            self.plot_pareto_front(pareto_front,
                                   result.problem.get_true_front(),
                                   legend,
                                   save_path)
