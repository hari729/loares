from datetime import datetime
from pathlib import Path
from typing import final
from matplotlib.pyplot import plot
import numpy as np
from multiprocessing import Pool, Manager
from opti.core.problem import ProblemHandler
from opti.analysis.moo.metrics import performance_metrics
from opti.core.population import PopulationHDF5Reader
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
        self.reader = PopulationHDF5Reader(problem, performance_metrics)

    def run(self, seed):
        np.random.seed(seed)

        filename = f"{seed}-{self.problem_info["psize"]}-{self.problem_info["max_evals"]}.h5" 
        filepath = self.output_dir / filename
        refined_dict = self.algorithm.run(filepath)

        return {"file": str(filepath), "seed": seed, "final_pop":refined_dict}

    def multi_thread(self, seeds, threads=5):
        print(f"Optimizing {self.problem_info["name"]} using {self.algorithm_info["name"]}")
        with Pool(processes=threads) as pool:
            output = pool.map(self.run, seeds)
        print("Processing\n")
        self._post_process(output)
        print(f"\nResults saved to {self.output_dir}")

    def _post_process(self, output):
        dict_list = []
        for o in output:
            temp_dict = self.reader.get_metrics_history(o["file"], 'function_evals')
            temp_dict["seed"] = [o["seed"]]
            dict_list.append(temp_dict)
            dict_to_csv(o["final_pop"], self.output_dir, o["seed"])

        metrics = dict_list[0].keys()
        mean = {"name": "Mean"}
        std = {"name": "Std"}
        net = {}
        for m in metrics:
            if m != "seed":
                values = np.array([r[m] for r in dict_list], dtype=float)
                mean[m] = np.mean(values, axis=0)
                std[m]  = np.std(values, axis=0)
                net[f"{m}(mean)"] = [mean[m][-1]]
                net[f"{m}(std)"] = [std[m][-1]]
                if m!='evals':
                    print(f"{m}(mean) :  {mean[m][-1]}")
                    print(f"{m}(std) :  {std[m][-1]}")

        dict_to_csv(mean, self.output_dir, "mean_history")
        dict_to_csv(std, self.output_dir, "std_history")
        dict_to_csv(net, self.output_dir, "net_result")

        final_metrics = {k:[] for k in dict_list[0].keys()}
        for d in dict_list:
            for i,j in d.items():
                final_metrics[i].append(j[-1])

        dict_to_csv(final_metrics, self.output_dir, "final_metrics")

        multi_line_plot([mean, std], self.output_dir)

        highest_hv_result = output[np.argmax(final_metrics["HV"])]
        plot_data = highest_hv_result["final_pop"]
        plot_data["name"] = self.algorithm_info["name"]
        plot_data["seed"] = highest_hv_result["seed"]
        dict_to_csv(plot_data, self.output_dir, "plot_data")
        n_obj = self.problem_info["n_obj"]
        if n_obj == 1:
            pass
        elif n_obj == 2:
            plot_2d(plot_data, self.output_dir)
        elif n_obj == 3:
            plot_3d(plot_data, self.output_dir)
        else:
            parallel_coordinates_plot(plot_data, self.output_dir)




