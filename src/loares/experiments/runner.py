from pathlib import Path
import inspect
import numpy as np
from math import comb
from multiprocessing import Pool
from loares.algorithms.moo.base import MOPopulationHandler
from loares.algorithms.soo.base import SOPopulationHandler
from loares.core.problem import ProblemHandler
from loares.metrics.moo import raw_performance_metrics
from loares.metrics.soo import bw_fitness
from loares.experiments.utils import dict_to_json
from loares.core.adapters import loares_to_pymoo_prob, pymoo_to_loares_h5
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
        caller_frame = inspect.stack()[1]
        caller_dir = Path(caller_frame.filename).resolve().parent
        output_dir = (
            caller_dir
            / test_name
            / "raw_data"
            / self.algorithm_info["name"]
            / f"{self.problem_info['psize']}-{self.problem_info['max_evals']}"
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.TF = TF
        if self.problem_info["n_obj"] > 1:
            self.populationHandler = MOPopulationHandler()
            self.metrics_calculator = raw_performance_metrics
            self.control_metric = "HV"
        else:
            self.populationHandler = SOPopulationHandler()
            self.metrics_calculator = bw_fitness
            self.control_metric = "best"

    def run(self, seed):
        np.random.seed(seed)
        self.problemHandler = ProblemHandler(self.problem)
        self.algorithm = self.algorithm_class(self.problemHandler)
        hdf5_path = self.output_dir / f"seed_{int(seed):03d}.h5"
        self.algorithm.run(seed, hdf5_path)

    def multi_thread(self, seeds, threads=5, get=False):
        print(
            f"\nOptimizing {self.problem_info['name']} using {self.algorithm_info['name']}"
        )
        print(
            f"| Population Size: {self.problem_info['psize']} | "
            + f"Max Evals: {self.problem_info['max_evals']} | Runs: {len(seeds)} |"
        )
        with Pool(processes=threads) as pool:
            pool.map(self.run, seeds)
        self._minimal_post_process(seeds)
        print(f"Results saved to {self.output_dir}")

    def _minimal_post_process(self, seeds):
        info_dict = {
            "Problem": self.problem_info,
            "Algorithm": self.algorithm_info,
            "UpdateRule": self.update_info,
            "seeds": str(seeds.tolist()),
        }
        dict_to_json(info_dict, self.output_dir, "Info")


def get_das_dennis_partitions(n_obj, target_psize):
    for p in range(1, 1000):
        n_points = comb(p + n_obj - 1, n_obj - 1)
        if n_points >= target_psize:
            return p
    return p


class PymooExptRunner(ExperimentRunner):
    def __init__(self, problem, algorithm, test_name, TF=None):
        self.problem = problem
        self.pymoo_problem = loares_to_pymoo_prob(self.problem)
        self.algorithm = algorithm
        self.problem_info = problem.get_info()
        self.algorithm_info = {"name": (self.algorithm.__name__).replace("_", "-")}
        self.update_info = {"name": f"pymoo defaults for {self.algorithm.__name__}"}
        self.test_name = test_name
        caller_frame = inspect.stack()[1]
        caller_dir = Path(caller_frame.filename).resolve().parent
        output_dir = (
            caller_dir
            / test_name
            / "raw_data"
            / self.algorithm_info["name"]
            / f"{self.problem_info['psize']}-{self.problem_info['max_evals']}"
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.TF = TF
        if self.problem_info["n_obj"] > 1:
            self.populationHandler = MOPopulationHandler()
            self.metrics_calculator = raw_performance_metrics
            self.control_metric = "HV"
        else:
            self.populationHandler = SOPopulationHandler()
            self.metrics_calculator = bw_fitness
            self.control_metric = "best"

    def run(self, seed):
        if self.algorithm_info["name"] in ["MOEAD", "NSGA3"]:
            n_partitions = get_das_dennis_partitions(
                self.problem.n_obj, self.problem.psize
            )
            ref_dirs = get_reference_directions(
                "das-dennis", self.problem.n_obj, n_partitions=n_partitions
            )
            algorithm = self.algorithm(ref_dirs=ref_dirs, pop_size=len(ref_dirs))
        else:
            algorithm = self.algorithm(pop_size=self.problem.psize)

        res = minimize(
            self.pymoo_problem,
            algorithm,
            ("n_eval", self.problem.max_evals),
            seed=int(seed),
            save_history=True,
        )
        hdf5_path = self.output_dir / f"seed_{int(seed):03d}.h5"
        pymoo_to_loares_h5(
            self.problem_info,
            self.algorithm_info,
            seed,
            res,
            self.populationHandler,
            hdf5_path,
        )
