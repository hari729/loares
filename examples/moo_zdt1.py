"""
Multi-objective optimization example on ZDT1.

Compares BXR variants (MO-BMR, MO-BWR, MO-BMWR) against NSGA-II,
runs post-processing, and generates statistical analysis.
"""

import os
import numpy as np
from pathlib import Path
import inspect

from pymoo.problems.multi import ZDT1
from pymoo.algorithms.moo.nsga2 import NSGA2

from loares.algorithms.bxr.moo import MO_BMR, MO_BWR, MO_BMWR
from loares.experiments.pymoo_runner import ExperimentRunner, AlgoFactory

from loares.experiments.pymoo_process import PostProcess

if __name__ == "__main__":
    runs = 5
    threads = min(8, os.cpu_count() or 8)
    seeds = np.arange(1, runs + 1, 1)
    ps = 100
    max_evals = 25000
    test_name = f"zdt1-example-r{runs}-p{ps}"

    problem = ZDT1()

    algorithms = [
        AlgoFactory(MO_BMR, pop_size=ps),
        AlgoFactory(MO_BWR, pop_size=ps),
        AlgoFactory(MO_BMWR, pop_size=ps),
        AlgoFactory(NSGA2, pop_size=ps),
    ]

    for factory in algorithms:
        runner = ExperimentRunner(problem, factory, max_evals, test_name)
        runner.multi_run(seeds, threads=threads)

    print(f"\nCompleted ZDT1 MOO experiment. Output: {test_name}/raw_data/")

    caller_dir = Path(inspect.stack()[0].filename).resolve().parent
    output_dir = (
        caller_dir
        / test_name
        / "raw_data")

    pp = PostProcess(
        test_dir=output_dir,
        algo_grps={"BXR": ["MO-BMR", "MO-BWR"], "common": ["NSGA2"]},
        true_front=ZDT1().pareto_front(500),
        gen_rf=False,
        plot_convergence=True,
        plot_pareto=True,
    )
    pp.run(threads=4)
