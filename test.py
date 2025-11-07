import numpy as np

from moo.algorithms.bw_samp import MO_BMR_SAMP, MO_BWR_SAMP, MO_BMWR_SAMP
from moo.problems.robotics import mau
from moo.population_modifiers import local_search

args = { "problem": mau,
         "pmods": [local_search]
        }

algo_classes = [MO_BMR_SAMP, MO_BWR_SAMP, MO_BMWR_SAMP]

def optimizer(algo_class, seed_list):
    results = []
    for i in seed_list:
        args["seed"] = i
        algo = algo_class(**args)
        while algo.tracker.remaining_evals() > 0:
            algo.advance()
        results.append(algo.get_result())
    return results

for algo_class in algo_classes:
    results = optimizer(algo_class, [1])


    result = results[0]
    print(result.get_convergence_data())
    print(f"{np.min(result.final_population.objectives[:,0])}, {np.min(result.final_population.objectives[:,1])}")
    print(result.final_metrics)
    print(result.algorithm)
    print(result.problem.get_info())
    print(result.algorithm.get_info())
