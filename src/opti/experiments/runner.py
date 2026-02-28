from pathlib import Path
import numpy as np
from math import comb
from multiprocessing import Pool
from opti.algorithms.moo.base import MOPopulationHandler
from opti.algorithms.soo.base import SOPopulationHandler
from opti.core.problem import ProblemHandler
from opti.analysis.moo.metrics import raw_performance_metrics
from opti.analysis.soo.metrics import bw_fitness
from opti.core.results import ResultProcessor
from opti.analysis.utils import dict_to_json
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
        if self.problem_info['n_obj']>1:
            self.populationHandler = MOPopulationHandler()
            self.metrics_calculator = raw_performance_metrics
            self.control_metric = 'HV'
        else:
            self.populationHandler = SOPopulationHandler()
            self.metrics_calculator = bw_fitness
            self.control_metric = 'best'

    def run(self, seed):
        np.random.seed(seed)
        self.problemHandler = ProblemHandler(self.problem)
        self.algorithm = self.algorithm_class(self.problemHandler)
        result = self.algorithm.run(seed)
        return result

    def multi_thread(self, seeds, threads=5, get=False):
        print(f"\nOptimizing {self.problem_info['name']} using {self.algorithm_info['name']}")
        print(f"| Population Size: {self.problem_info['psize']} | " +
                f"Max Evals: {self.problem_info['max_evals']} | Runs: {len(seeds)} |")
        with Pool(processes=threads) as pool:
            output = pool.map(self.run, seeds)
        print("Processing (Minimal)")
        self._minimal_post_process(output, seeds)
        if get:
            return output

    def _minimal_post_process(self, output, seeds):
 
        info_dict = {"Problem": self.problem_info,
                       "Algorithm": self.algorithm_info,
                       "UpdateRule": self.update_info,
                    "seeds": str(seeds.tolist())}
        dict_to_json(info_dict, self.output_dir, "Info")
        self.processor.to_hdf5(output, self.output_dir / "history.h5")
        print(f"Results saved to {self.output_dir}")



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
            self.metrics_calculator = raw_performance_metrics
            self.control_metric = 'HV'
        else:
            self.populationHandler = SOPopulationHandler()
            self.metrics_calculator = bw_fitness
            self.control_metric = 'best'

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
