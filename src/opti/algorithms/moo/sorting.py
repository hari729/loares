import numpy as np

from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.population import Population as PymooPopulation
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.problem import Problem

from opti.moo.population import MoPopulation

def ranking_crowding(problem, population, limit, seed, ndf=False):

    class DummyProblem(Problem):
        def __init__(self, n_var, n_obj, n_constr):
            super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr)
        
        def _evaluate(self, x, out, *args, **kwargs):
            pass

    pop = PymooPopulation.new("X", population.solutions, "F", population.objectives, "G", population.constraints)
 
    dummy_problem = DummyProblem(n_var=problem.n_vars, 
                                 n_obj=problem.n_obj, 
                                 n_constr=problem.n_constr)

    survival = RankAndCrowdingSurvival()
    survivors = survival.do(dummy_problem, pop, n_survive=limit, seed=seed)

    target_pop = survivors
    if ndf:
        target_pop = survivors[survivors.get("rank") == 0]

    p_array = target_pop.get("X")
    o_array = target_pop.get("F")
    c_array = target_pop.get("G")
    metadata = np.column_stack([target_pop.get("rank"), target_pop.get("crowding") ])

    if np.all([x is None for x in np.ravel(metadata)]):
        metadata = target_pop.get("CV") 
        metadata = metadata - np.min(metadata)
        metadata = metadata.reshape(-1, 1)

    # return MoPopulation(p_array, o_array, c_array, metadata)
    return p_array, o_array, c_array, metadata
