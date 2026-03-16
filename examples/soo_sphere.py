import os
import sys
import numpy as np

from loares.core.problem import Problem
from loares.algorithms.soo import SO_BMR, SO_BWR, SO_BMWR
from loares.experiments.runner import ExperimentRunner
from loares.experiments.process import post_process


def sphere(X):
    F = np.sum(X**2, axis=1, keepdims=True)
    G = np.zeros((X.shape[0], 1))
    return F, G


class Sphere(Problem):
    def __init__(self, n_vars=10, psize=50, max_evals=5000):
        super().__init__(
            function=sphere,
            n_vars=n_vars,
            n_obj=1,
            n_constr=0,
            psize=psize,
            max_evals=max_evals,
            bounds=np.column_stack([np.full(n_vars, -5.12), np.full(n_vars, 5.12)]),
            minmax=["min"],
        )


if __name__ == "__main__":
    runs = 5
    threads = min(8, os.cpu_count() or 8)
    seeds = np.arange(1, runs + 1, 1)
    ps = 50
    psizes = [ps]
    test_name = f"sphere-example-r{runs}-p{ps}"

    custom_algos = [SO_BMR, SO_BWR, SO_BMWR]

    algo_grps = {
        "BMR": ["SO-BMR"],
        "BWR": ["SO-BWR"],
        "BMWR": ["SO-BMWR"],
        "common": [],
    }

    for algo in custom_algos:
        for psize in psizes:
            runner = ExperimentRunner(Sphere(psize=psize), algo, test_name)
            runner.multi_thread(seeds, threads=threads)

    processor = post_process(
        Sphere(),
        test_name,
        psizes,
        algo_grps=algo_grps,
    )
    compare_dir = processor.multi_thread(threads=threads)

    print(f"Completed Sphere SOO pipeline. Outputs: {compare_dir}")
