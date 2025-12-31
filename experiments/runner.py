from datetime import datetime
from pathlib import Path
import numpy as np

from opti.core.problem import ProblemHandler

class ExperimentRunner:
    def __init__(self, problem, algorithm):
        self.problem = problem
        self.problemHandler = ProblemHandler(self.problem)
        self.algorithm = algorithm(self.problemHandler)
        self.problem_info = problem.get_info()
        self.algorithm_info = self.algorithm.get_info()
        output_dir = (Path.home()/"OptiResults"
                                /self.problem_info["name"]
                                /self.algorithm_info["name"]
                                /f"{self.problem_info["psize"]}-{self.problem_info["max_evals"]}")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, seed):
        np.random.seed(seed)

        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        filename = f"{seed}_{timestamp}.h5"
        filepath = self.output_dir / filename
        # ---- run ----
        self.algorithm.run(filepath)

        # ---- minimal post-processing ----
        return self._post_process(filepath)

    def _post_process(self, filepath):
        # intentionally minimal
        return {
            "file": str(filepath),
        }
