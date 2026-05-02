"""
Single-objective optimization example on Sphere.

Compares BXR variants (SO-BMR, SO-BWR, SO-BMWR).
"""

import os
import numpy as np

from pymoo.problems.single import Sphere

from loares.algorithms.bxr.soo import SO_BMR, SO_BWR, SO_BMWR
from loares.experiments.pymoo_runner import ExperimentRunner, AlgoFactory


if __name__ == "__main__":
    runs = 5
    threads = min(8, os.cpu_count() or 8)
    seeds = np.arange(1, runs + 1, 1)
    ps = 50
    max_evals = 5000
    test_name = f"sphere-example-r{runs}-p{ps}"

    problem = Sphere(n_var=10)

    algorithms = [
        AlgoFactory(SO_BMR, pop_size=ps),
        AlgoFactory(SO_BWR, pop_size=ps),
        AlgoFactory(SO_BMWR, pop_size=ps),
    ]

    for factory in algorithms:
        runner = ExperimentRunner(problem, factory, max_evals, test_name)
        runner.multi_run(seeds, threads=threads)

    print(f"\nCompleted Sphere SOO experiment. Output: {test_name}/raw_data/")
