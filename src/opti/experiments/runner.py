from pathlib import Path
from matplotlib.pyplot import ylabel
import numpy as np
from multiprocessing import Pool
from opti.core.problem import ProblemHandler
from opti.analysis.moo.metrics import performance_metrics, raw_performance_metrics
from opti.core.population import PopulationHDF5Reader, PopulationRecorderHDF5
from opti.core.results import ResultProcessor
from opti.analysis.utils import dict_to_csv, dict_to_json, modify_master_list
from opti.analysis.plots import multi_line_plot, plot_2d, plot_3d, parallel_coordinates_plot

class ExperimentRunner:
    def __init__(self, problem, algorithm, test_name, TF=None):
        self.problem = problem
        self.problemHandler = ProblemHandler(self.problem)
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
        Path(self.output_dir/"H5").mkdir(parents=True, exist_ok=True)
        self.processor = ResultProcessor()
        self.TF = TF

    def run(self, seed):
        np.random.seed(seed)
        result = self.algorithm.run(seed)
        return result

    def multi_thread(self, seeds, threads=5):
        print(f"\nOptimizing {self.problem_info['name']} using {self.algorithm_info['name']}")
        print(f"| Population Size: {self.problem_info['psize']} | " +
                f"Max Evals: {self.problem_info['max_evals']} | Runs: {len(seeds)} |")
        with Pool(processes=threads) as pool:
            output = pool.map(self.run, seeds)
        print("\nProcessing\n")
        self._post_process(output)
        print(f"\nResults saved to {self.output_dir}")

    def _post_process(self, output):

        info_dict = {"Problem": self.problem_info,
                       "Algorithm": self.algorithm_info,
                       "UpdateRule": self.update_info}
        dict_to_json(info_dict, self.output_dir, "Info")

        metrics_list = []
        for res in output:
            recorder = PopulationRecorderHDF5(f"{self.output_dir}/H5/{res.seed:03d}.h5")
            for i, eval in enumerate(res.history['evals']):
                recorder.record(res.history['pop'][i], eval)
            recorder.close()
            temp_dict = self.processor.get_metrics_history(res, raw_performance_metrics,
                                                           self.TF)
            temp_dict["seed"] = [res.seed]
            metrics_list.append(temp_dict)

        metrics = metrics_list[0].keys()
        mean = {"name": f"{self.algorithm_info['name']} (Mean)"}
        std = {"name": f"{self.algorithm_info['name']} (Std)"}
        net = {}
        evals = np.array([r['evals'] for r in metrics_list], dtype=float)
        mean['evals'] = np.mean(evals, axis=0)
        std['evals'] = mean['evals']
        convergence = {"name": f"{self.algorithm_info['name']} (convergence pts)"}
        ind_metrics = []
        for m in metrics:
            if m not in ["seed", "evals"]:
                values = np.array([r[m] for r in metrics_list], dtype=float)
                mean[m] = np.mean(values, axis=0)
                delta = np.diff(mean[m])/mean[m][:-1]
                convergence_pt = np.where(np.abs(delta) < 1e-3)[0]
                if len(convergence_pt)>0:
                    convergence[m] = [mean[m][convergence_pt+1][0],mean['evals'][convergence_pt[0]+1]]
                else:
                    convergence[m] = [np.nan, np.nan]
                std[m]  = np.std(values, axis=0)
                ind_metrics.append({'ydata' : [mean[m],std[m]], 'ylabel': f"{m}",
                                    'xdata': [mean['evals'],std['evals']], 'xlabel' : "Mean-Function-Evaluations",
                                    'point' : [convergence[m]],
                                    'legend':[mean['name'],std['name']]})
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




class pymooExptRunner(ExperimentRunner):
    def __init__(self, problem, algorithm, test_name):
        super().__init__(problem, algorithm, test_name)
