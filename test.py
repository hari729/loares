import numpy as np

from moo.algorithms.bwr import MO_BMR, MO_BWR, MO_BMWR
from moo.problems.robotics import mau
from moo.population_modifiers import local_search

args = { "problem": mau,
         "pmods": [local_search]
        }

algos = [MO_BMR(**args), MO_BWR(**args), MO_BMWR(**args)]

for algo in algos:
    while algo.tracker.remaining_evals() > 0:
        algo.advance()

    result = algo.get_result()
    print()
    print(f"{np.min(result.final_population.objectives[:,0])}, {np.min(result.final_population.objectives[:,1])}")
    print(result.final_metrics)
