import numpy as np

from pymoo.core.population import Population as PymooPopulation
from pymoo.core.problem import Problem


def bw_sorting(problem, population, limit, seed, ndf=False, all=False):
    if limit is None:
        limit = problem.psize
    # print(population.constraints)
    violation_count = np.atleast_2d((population.constraints > 0).sum(axis=1)).T
    # print(population.solutions)
    # print(population.objectives)
    # print(violation_count)
    sorted_idx = np.lexsort((population.objectives[:,0], violation_count[:,0]))[:limit]
    # print(sorted_idx)
    sols = population.solutions[sorted_idx]
    objs = population.objectives[sorted_idx]
    constr = population.constraints[sorted_idx]
    metadata = violation_count[sorted_idx]
    # print(population.objectives)
    # print(objs[0])
    return sols, objs, constr, metadata
