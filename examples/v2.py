"""
Multi-objective optimization example on ZDT1.

Compares BXR variants (MO-BMR, MO-BWR, MO-BMWR) against NSGA-II,
runs post-processing, and generates statistical analysis.
"""

import os
import numpy as np
from pathlib import Path
import inspect

from loares import indicator
from pymoo.problems.multi import ZDT1
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import HV
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from pymoo.indicators.spacing import SpacingIndicator

from loares.algorithms.bxr.moo import MO_BMR, MO_BWR, MO_BMWR
from loares.run import parallel_run
from loares.indicator import indicator_multi_run, mean_history_multi_run
from loares.statistics import statistical_test_1

if __name__ == "__main__":
    runs = 5
    seeds = np.arange(1, runs + 1, 1)
    ps = 100
    max_evals = 25000
    problem = ZDT1()

    common = {
        "algorithm_kwargs": {"pop_size": ps},
        "problem_name": "ZDT1",
        "problem": problem,
    }

    algos = [
        {"algorithm_name": "MO-BMR", "algorithm": MO_BMR(pop_size=ps)},
        {"algorithm_name": "MO-BWR", "algorithm": MO_BWR(pop_size=ps)},
        {"algorithm_name": "NSGA-II", "algorithm": NSGA2(pop_size=ps)},
    ]
    algo_specs = [
        {
            "solver_kwargs": {
                "seed": s,
                "termination": ("n_eval", max_evals),
                "save_history": True,
            },
            **common,
            **algo,
        }
        for s in seeds
        for algo in algos
    ]

    output_dir = Path(inspect.stack()[0].filename).resolve().parent
    parallel_run(
        algo_specs,
        output_dir,
        n_jobs=5,
    )

    print("\nCompleted ZDT1 MOO experiment")

    true_front = ZDT1().pareto_front(500)

    indicator_specs = [
        {"indicator_name": "GD", "indicator": GD(true_front)},
        {"indicator_name": "IGD", "indicator": IGD(true_front)},
        {"indicator_name": "Spacing", "indicator": SpacingIndicator()},
        {"indicator_name": "HV", "indicator": HV(ref_point=[1.1, 1.1])},
    ]

    indicator_multi_run(
        indicator_specs,
        output_dir,
        n_jobs=5,
    )
    #
    # stat_specs = [
    #     {
    #         "filter": {
    #             "indicator_name": "HV",
    #             "pop_size": ps,
    #             "termination_metric": "n_eval",
    #             "termination_value": max_evals,
    #             "source": "optimum",
    #         },
    #         "pivot": {
    #             "index": "seed",
    #             "columns": "algorithm_name",
    #             "values": "indicator_value",
    #         },
    #     }
    # ]
    #
    # statistical_test_1(
    #     stat_specs,
    #     Path(inspect.stack()[0].filename).resolve().parent / "metrics.csv",
    #     Path(inspect.stack()[0].filename).resolve().parent,
    # )
    #
    # from loares.indicator import indicator_history_multi_run, plot_convergence
    #
    # indicator_history_multi_run(
    #     indicator_specs, algo_specs, output_dir, output_dir, n_jobs=5
    # )
    # mean_history_multi_run(output_dir / "history.parquet", output_dir)
