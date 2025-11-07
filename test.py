import numpy as np

from moo.algorithms.bw_samp import MO_BMR_SAMP, MO_BWR_SAMP, MO_BMWR_SAMP
from moo.problems.robotics import mau
from moo.population_modifiers import local_search

args = { "problem": mau,
         "pmods": [local_search]
        }

algos = [MO_BMR_SAMP(**args), MO_BWR_SAMP(**args), MO_BMWR_SAMP(**args)]

for algo in algos:
    while algo.tracker.remaining_evals() > 0:
        algo.advance()

    result = algo.get_result()
    print(result.get_convergence_data())
    print(f"{np.min(result.final_population.objectives[:,0])}, {np.min(result.final_population.objectives[:,1])}")
    print(result.final_metrics)
    print(result.algorithm)
    print(result.problem.get_info())
    print(result.algorithm.get_info())
