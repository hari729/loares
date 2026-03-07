import numpy as np


def bw_sorting(problem, population, limit, seed, ndf=False, all=False):
    if limit is None:
        limit = problem.psize
    violation_count = np.atleast_2d((population.constraints > 0).sum(axis=1)).T
    sorted_idx = np.lexsort((population.objectives[:, 0], violation_count[:, 0]))[
        :limit
    ]
    sols = population.solutions[sorted_idx]
    objs = population.objectives[sorted_idx]
    constr = population.constraints[sorted_idx]
    metadata = violation_count[sorted_idx]
    return sols, objs, constr, metadata
