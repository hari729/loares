import os
import numpy as np

from loares.core.problem import Problem
from loares.algorithms.moo import MO_BMR, MO_BWR, MO_BMWR
from pymoo.algorithms.moo.nsga2 import NSGA2
from loares.experiments.runner import ExperimentRunner, PymooExptRunner
from loares.experiments.process import post_process
from loares.experiments.analysis.compare import compare_metrics
from loares.experiments.analysis.stats import run as run_statistics


def zdt1(X):
    n = X.shape[1]
    f1 = X[:, 0]
    g = 1 + 9 * np.sum(X[:, 1:], axis=1) / (n - 1)
    f2 = g * (1 - np.sqrt(f1 / g))
    F = np.column_stack([f1, f2])
    G = np.zeros((X.shape[0], 1))
    return F, G


class ZDT1(Problem):
    def __init__(self, n_vars=30, psize=100, max_evals=25000):
        super().__init__(
            function=zdt1,
            n_vars=n_vars,
            n_obj=2,
            n_constr=0,
            psize=psize,
            max_evals=max_evals,
            bounds=np.column_stack([np.zeros(n_vars), np.ones(n_vars)]),
            minmax=["min", "min"],
        )

    def get_true_front(self, pts=500):
        f1 = np.linspace(0, 1, pts)
        f2 = 1 - np.sqrt(f1)
        return np.column_stack([f1, f2])


if __name__ == "__main__":
    runs = 5
    threads = min(8, os.cpu_count() or 8)
    seeds = np.arange(1, runs + 1, 1)
    ps = 100
    psizes = [ps]
    test_name = f"zdt1-example-r{runs}-p{ps}"

    custom_algos = [MO_BMR, MO_BWR, MO_BMWR]
    pymoo_algos = [NSGA2]

    algo_grps = {
        "BMR": ["MO-BMR"],
        "BWR": ["MO-BWR"],
        "BMWR": ["MO-BMWR"],
        "common": ["NSGA2"],
    }

    for algo in custom_algos:
        for psize in psizes:
            runner = ExperimentRunner(ZDT1(psize=psize), algo, test_name)
            runner.multi_thread(seeds, threads=threads)

    for algo in pymoo_algos:
        for psize in psizes:
            runner = PymooExptRunner(ZDT1(psize=psize), algo, test_name)
            runner.multi_thread(seeds, threads=threads)

    processor = post_process(
        ZDT1(),
        test_name,
        psizes,
        algo_grps=algo_grps,
        gen_rf=True,
        rf_size=5000,
    )
    compare_dir = processor.multi_thread(threads=threads)

    compare_metrics(ZDT1().get_info()["name"], compare_dir)

    for psize in psizes:
        run_statistics(compare_dir / str(psize), alpha=0.05)

    print(f"Completed ZDT1 MOO pipeline. Outputs: {compare_dir}")
