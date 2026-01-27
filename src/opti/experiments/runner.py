from pathlib import Path
import pandas as pd
from matplotlib.pyplot import ylabel
import numpy as np
import pickle, gzip
from math import comb
from multiprocessing import Pool
from opti.algorithms.moo.base import MOPopulationHandler
from opti.analysis.moo import metrics
from opti.core.problem import ProblemHandler
from opti.analysis.moo.metrics import raw_performance_metrics
from opti.core.results import ResultProcessor
from opti.analysis.utils import dict_to_csv, dict_to_json, modify_master_list
from opti.analysis.plots import multi_line_plot, plot_2d, plot_3d, parallel_coordinates_plot
from opti.core.adapters import opti_to_pymoo_prob, pymoo_to_opti_res
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

class ExperimentRunner:
    def __init__(self, problem, algorithm, test_name, TF=None):
        self.problem = problem
        self.problemHandler = ProblemHandler(self.problem)
        self.algorithm_class = algorithm
        self.algorithm = algorithm(self.problemHandler)
        self.problem_info = problem.get_info()
        self.algorithm_info = self.algorithm.get_info()
        self.update_info = self.algorithm.updateRule.get_info()
        self.test_name = test_name
        output_dir = (Path.home()/"OptiResults"
                                /self.problem_info["name"]
                                /test_name
                                /self.algorithm_info["name"]
                                /f"{self.problem_info['psize']}-{self.problem_info['max_evals']}")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processor = ResultProcessor()
        self.TF = TF

    def run(self, seed):
        np.random.seed(seed)
        self.problemHandler = ProblemHandler(self.problem)
        self.algorithm = self.algorithm_class(self.problemHandler)
        result = self.algorithm.run(seed)
        return result

    def multi_thread(self, seeds, threads=5, get=False, min=False):
        print(f"\nOptimizing {self.problem_info['name']} using {self.algorithm_info['name']}")
        print(f"| Population Size: {self.problem_info['psize']} | " +
                f"Max Evals: {self.problem_info['max_evals']} | Runs: {len(seeds)} |")
        with Pool(processes=threads) as pool:
            output = pool.map(self.run, seeds)
        if get:
            return output
        elif min:
            print("Processing (Minimal)")
            return self._minimal_post_process(output)
        else:
            print("\nProcessing\n")
            self._post_process(output)
            print(f"\nResults saved to {self.output_dir}")

    def _post_process(self, output):

        info_dict = {"Problem": self.problem_info,
                       "Algorithm": self.algorithm_info,
                       "UpdateRule": self.update_info}
        dict_to_json(info_dict, self.output_dir, "Info")
        
        with gzip.open(self.output_dir / "results.pkl.gz", 'wb') as f:
            pickle.dump(output, f)
        
        metrics_list = []
        for res in output:
            temp_dict = self.processor.get_metrics_history(res, raw_performance_metrics,
                                                           self.TF)
            temp_dict["seed"] = [res.seed]
            metrics_list.append(temp_dict)

        metrics = metrics_list[0].keys()
        mean = {"name": f"{self.algorithm_info['name']} (Mean)"}
        std = {"name": f"{self.algorithm_info['name']} (Std)"}
        net = {}
        recording_interval = int(self.problem_info['max_evals'] * 0.05)
        eval_grid = np.arange(recording_interval, 
                            self.problem_info['max_evals'] + 1, 
                            recording_interval)
        # Interpolate each run's metrics to the common grid
        mean['evals'] = eval_grid
        std['evals'] = eval_grid
        convergence = {"name": f"{self.algorithm_info['name']} (convergence pts)"}
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
                
                ind_metrics.append({
                    'ydata': [mean[m], std[m]], 
                    'ylabel': f"{m}",
                    'xdata': [mean['evals'], std['evals']], 
                    'xlabel': "Mean-Function-Evaluations",
                    'point': [convergence[m]],
                    'legend': [mean['name'], std['name']]
                })
                net[f"{m}(mean)"] = [mean[m][-1]]
                net[f"{m}(std)"] = [std[m][-1]]
                print(f"{m}(mean) :  {mean[m][-1]}")
                print(f"{m}(std) :  {std[m][-1]}")

        dict_to_csv(mean, self.output_dir, "mean-history")
        dict_to_csv(std, self.output_dir, "std-history")
        dict_to_csv(net, self.output_dir, "net-result")
        dict_to_csv(convergence, self.output_dir, "convergence-points")

        final_metrics = {k:[] for k in metrics_list[0].keys()}
        for d in metrics_list:
            for i,j in d.items():
                final_metrics[i].append(j[-1])
        dict_to_csv(final_metrics, self.output_dir, "final-metrics-per-run")

        master_dict = {"Problem": self.problem_info["name"],
                       "Algorithm": self.algorithm_info['name'],
                       "Max-evals": self.problem_info['max_evals'],
                       "Psize": self.problem_info["psize"],
                       "Runs": len(final_metrics['seed']),
                       **net,
                       }
        modify_master_list(master_dict, Path(self.output_dir.parent/"master.csv"))

        for ind in ind_metrics:
            multi_line_plot(ind, self.output_dir)


        highest_hv_result = output[np.argmax(final_metrics["HV"])]
        plot_data = highest_hv_result.final_dict
        dict_to_csv(plot_data, self.output_dir, "pareto-front")
        plot_data["name"] = highest_hv_result.algorithm_info["name"]
        plot_data["seed"] = highest_hv_result.seed
        n_obj = self.problem_info["n_obj"]
        if n_obj == 1:
            pass
        elif n_obj == 2:
            plot_2d(plot_data, self.output_dir)
        elif n_obj == 3:
            plot_3d(plot_data, self.output_dir)
        else:
            parallel_coordinates_plot(plot_data, self.output_dir)

    def _minimal_post_process(self, output):
 
        with gzip.open(self.output_dir / "results.pkl.gz", 'wb') as f:
            pickle.dump(output, f)

        metrics_list = []
        for res in output:
            temp_dict = raw_performance_metrics(res.population.objectives,self.TF)
            temp_dict["seed"] = res.seed
            temp_dict["res"] = res
            metrics_list.append(temp_dict)

        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.drop(columns=['res']).to_csv(self.output_dir /"raw-results.csv", index=False, float_format="%.5f")
        metrics_df['Algorithm'] = self.algorithm_info['name']

        info_dict = {"Problem": self.problem_info,
                       "Algorithm": self.algorithm_info,
                       "UpdateRule": self.update_info,
                     "seeds": str(metrics_df['seed'].tolist())}
        dict_to_json(info_dict, self.output_dir, "Info")

        # highest_hv_result = output[np.argmax(raw_metrics["HV"])]
        highest_hv_result = metrics_df['res'][metrics_df['HV'].idxmax()]
        plot_data = highest_hv_result.final_dict
        dict_to_csv(plot_data, self.output_dir, "pareto-front")
        plot_data["name"] = highest_hv_result.algorithm_info["name"]
        plot_data["seed"] = highest_hv_result.seed
        n_obj = self.problem_info["n_obj"]
        
        # np.save(self.output_dir / "pareto_front.npy", highest_hv_result.population.objectives)

        if n_obj == 1:
            pass
        elif n_obj == 2:
            plot_2d(plot_data, self.output_dir)
        elif n_obj == 3:
            plot_3d(plot_data, self.output_dir)
        else:
            parallel_coordinates_plot(plot_data, self.output_dir)
        print(f"Raw results: HV = {metrics_df['HV'].mean():4f}")
        print(f"Results saved to {self.output_dir}")
        return {'name':self.algorithm_info['name'],
                'psize':self.problem_info['psize'],
                'data': metrics_df }

def get_das_dennis_partitions(n_obj, target_psize):
    # Find n_partitions that gives closest to target_psize
    for p in range(1, 1000):
        n_points = comb(p + n_obj - 1, n_obj - 1)
        if n_points >= target_psize:
            return p
    return p

class PymooExptRunner(ExperimentRunner):
    def __init__(self, problem, algorithm, test_name, TF=None):
        self.problem = problem
        self.pymoo_problem = opti_to_pymoo_prob(self.problem)
        self.algorithm = algorithm
        self.problem_info = problem.get_info()
        self.algorithm_info = {'name':(self.algorithm.__name__).replace("_", "-")}
        self.update_info = {'name':f"pymoo defaults for {self.algorithm.__name__}"}
        self.test_name = test_name
        output_dir = (Path.home()/"OptiResults"
                                /self.problem_info["name"]
                                /test_name
                                /self.algorithm_info["name"]
                                /f"{self.problem_info['psize']}-{self.problem_info['max_evals']}")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processor = ResultProcessor()
        self.TF = TF
        if self.problem_info['n_obj']>1:
            self.populationHandler = MOPopulationHandler()
        else:
            print("Implement SO PopulationHandler First")
            raise

    def run(self, seed):
        if self.algorithm_info['name'] in ['MOEAD', 'NSGA3']:
            n_partitions = get_das_dennis_partitions(self.problem.n_obj,
                                                     self.problem.psize)
            ref_dirs = get_reference_directions('das-dennis', 
                                                self.problem.n_obj, 
                                                n_partitions=n_partitions)
            algorithm = self.algorithm(ref_dirs=ref_dirs, pop_size=len(ref_dirs))
        else:
            algorithm = self.algorithm(pop_size=self.problem.psize)

        res = minimize(self.pymoo_problem,
                    algorithm,
                    ('n_eval', self.problem.max_evals),
                    seed=int(seed),
                    save_history=True,
                    )
        return pymoo_to_opti_res(self.problem_info,
                                 self.algorithm_info,
                                 seed,
                                 res,
                                 self.populationHandler)
