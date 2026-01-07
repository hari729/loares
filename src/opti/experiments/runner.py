from datetime import datetime
from pathlib import Path
from typing import final
from matplotlib.pyplot import plot
import numpy as np
from multiprocessing import Pool, Manager
from opti.core.problem import ProblemHandler
from opti.analysis.moo.metrics import performance_metrics, raw_performance_metrics
from opti.core.population import PopulationHDF5Reader
from opti.core.results import ResultProcessor
from opti.analysis.utils import dict_to_csv
from opti.analysis.plots import multi_line_plot, plot_2d, plot_3d, parallel_coordinates_plot

class ExperimentRunner:
    def __init__(self, problem, algorithm):
        self.problem = problem
        self.problemHandler = ProblemHandler(self.problem)
        self.algorithm = algorithm(self.problemHandler)
        self.problem_info = problem.get_info()
        self.algorithm_info = self.algorithm.get_info()
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_dir = (Path.home()/"OptiResults"
                                /self.problem_info["name"]
                                /self.algorithm_info["name"]
                                /timestamp)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processor = ResultProcessor()

    def run(self, seed):
        np.random.seed(seed)
        result = self.algorithm.run(seed)
        return result

    def multi_thread(self, seeds, threads=5):
        print(f"\nOptimizing {self.problem_info['name']} using {self.algorithm_info['name']}")
        with Pool(processes=threads) as pool:
            output = pool.map(self.run, seeds)
        print("Processing\n")
        self._post_process(output)
        print(f"\nResults saved to {self.output_dir}")

    def _post_process(self, output):
        metrics_list = []
        for res in output:
            temp_dict = self.processor.get_metrics_history(res, raw_performance_metrics)
            temp_dict["seed"] = [res.seed]
            metrics_list.append(temp_dict)
            dict_to_csv(res.final_dict, self.output_dir, res.seed)

        metrics = metrics_list[0].keys()
        mean = {"name": "Mean"}
        std = {"name": "Std"}
        net = {}
        for m in metrics:
            if m != "seed":
                values = np.array([r[m] for r in metrics_list], dtype=float)
                mean[m] = np.mean(values, axis=0)
                std[m]  = np.std(values, axis=0)
                net[f"{m}(mean)"] = [mean[m][-1]]
                net[f"{m}(std)"] = [std[m][-1]]
                if m!='evals':
                    print(f"{m}(mean) :  {mean[m][-1]}")
                    print(f"{m}(std) :  {std[m][-1]}")

        dict_to_csv(mean, self.output_dir, "mean-history")
        dict_to_csv(std, self.output_dir, "std-history")
        dict_to_csv(net, self.output_dir, "net-result")

        final_metrics = {k:[] for k in metrics_list[0].keys()}
        for d in metrics_list:
            for i,j in d.items():
                final_metrics[i].append(j[-1])

        dict_to_csv(final_metrics, self.output_dir, "final-metrics")

        multi_line_plot([mean, std], self.output_dir)

        highest_hv_result = output[np.argmax(final_metrics["HV"])]
        plot_data = highest_hv_result.final_dict
        plot_data["name"] = highest_hv_result.algorithm_info["name"]
        plot_data["seed"] = highest_hv_result.seed
        dict_to_csv(plot_data, self.output_dir, "plot-data")
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
    def __init__(self, problem, algorithm):
        super().__init__(problem, algorithm)
