import numpy as np
from scipy.spatial.distance import cdist
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
from pymoo.core.population import Population as PymooPopulation
from pymoo.core.problem import Problem
from pymoo.util.normalization import normalize

def ranking_crowding(problem, population, limit, seed, ndf=False, all=False):

    class DummyProblem(Problem):
        def __init__(self, n_var, n_obj, n_constr):
            super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr)

        def _evaluate(self, x, out, *args, **kwargs):
            pass

    pop = PymooPopulation.new(
        "X",
        population.solutions,
        "F",
        population.objectives,
        "G",
        population.constraints,
    )

    dummy_problem = DummyProblem(
        n_var=problem.n_vars, n_obj=problem.n_obj, n_constr=problem.n_constr
    )

    survival = RankAndCrowdingSurvival()
    if all:
        survivors = survival.do(dummy_problem, pop, seed=seed)
    else:
        survivors = survival.do(dummy_problem, pop, n_survive=limit, seed=seed)

    target_pop = survivors
    if ndf:
        target_pop = survivors[survivors.get("rank") == 0]

    p_array = target_pop.get("X")
    o_array = target_pop.get("F")
    c_array = target_pop.get("G")
    metadata = np.column_stack([target_pop.get("rank"), target_pop.get("crowding")])

    if np.all([x is None for x in np.ravel(metadata)]):
        metadata = target_pop.get("CV")
        metadata = metadata - np.min(metadata)
        metadata = metadata.reshape(-1, 1)

    return p_array, o_array, c_array, metadata


nds = NonDominatedSorting()

def nds_cd(population, limit=None, ndf=False):
    limit = len(population.objectives) if limit is None else limit
    N = len(population.solutions)
    ranks = np.full(N, np.inf)
    cd = np.zeros(N)
    if ndf:
        idx = nds.do(population.objectives, only_non_dominated_front=True)
        cd[idx] = calc_crowding_distance(population.objectives[idx])
        ranks[idx] = 0
    else:
        fronts = nds.do(population.objectives, n_stop_if_ranked=limit)
        for i in range(len(fronts)):
            cd[fronts[i]] = calc_crowding_distance(population.objectives[fronts[i]])
            ranks[fronts[i]] = i
    sorted_idx = np.lexsort((-cd, ranks))
    selected = sorted_idx[:limit]
    pm = np.column_stack([ranks[selected], cd[selected]])
    ps = population.solutions[selected]
    po = population.objectives[selected]
    pc = population.constraints[selected]

    return ps, po, pc, pm


def farthest_point_sampling(points, n_samples):
    n_obj = points.shape[1]
    selected = []
    npoints = normalize(points, np.min(points,axis=0), np.max(points, axis=0))
    for j in range(n_obj):
        selected.append(np.argmin(npoints[:, j]))
        selected.append(np.argmax(npoints[:, j]))
    selected = list(dict.fromkeys(selected))  # deduplicate, preserve order

    min_dist = cdist(npoints, npoints[selected]).min(axis=1)

    for _ in range(n_samples - len(selected)):
        idx = np.argmax(min_dist)
        selected.append(idx)
        new_dist = cdist(npoints, npoints[idx : idx + 1]).flatten()
        min_dist = np.minimum(min_dist, new_dist)
    return selected


def nds_fps(prob, population, limit, seed, ndf=False, all=False):
    selected_idx = nds.do(population.objectives, only_non_dominated_front=True)
    if limit < len(selected_idx):
        selected = farthest_point_sampling(population.objectives[selected_idx], limit)
    else:
        selected = np.arange(len(selected_idx))
    ps = population.solutions[selected_idx][selected]
    po = population.objectives[selected_idx][selected]
    pc = population.constraints[selected_idx][selected]
    pm = np.zeros((len(selected), 1))

    return ps, po, pc, pm
