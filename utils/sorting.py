import numpy as np
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.problem import Problem

def ranking_reference(population_data,
                      objective_values,
                      limit,
                      ndf=False,
                      constraint_values=None,
                      ref_dirs=None):
    """
    NSGA-III style survival using pymoo's ReferenceDirectionSurvival.

    Parameters
    ----------
    population_data : np.ndarray
        Decision variable matrix (N x D).
    objective_values : np.ndarray
        Objective values (N x M).
    limit : int
        Number of survivors to select.
    ndf : bool
        If True, return only the non-dominated front.
    constraint_values : np.ndarray, optional
        Constraint violations (N x C).
    ref_dirs : np.ndarray, optional
        Reference directions (K x M). If None, will be generated.

    Returns
    -------
    Depending on `constraint_values` and `ndf`:
        X, F, [G], metadata
    metadata columns:
        [rank, niche_index, niche_distance]
    """
    M = objective_values.shape[1]

    # Generate reference directions if not provided
    if ref_dirs is None:
        ref_dirs = get_reference_directions("das-dennis", M, n_partitions=13)

    # Build population
    if constraint_values is None:
        pop = Population.new("X", population_data, "F", objective_values)

        from pymoo.core.problem import Problem
        class DummyProblem(Problem):
            def __init__(self, n_obj):
                super().__init__(n_var=1, n_obj=n_obj, n_constr=0)

        dummy_problem = DummyProblem(n_obj=M)

        survival = ReferenceDirectionSurvival(ref_dirs)
        survivors = survival.do(dummy_problem, pop, n_survive=limit)

        if ndf:
            nd_front = survivors[survivors.get("rank") == 0]
            metadata = np.column_stack([
                nd_front.get("rank"),
                nd_front.get("niche"),
                nd_front.get("dist_to_niche")
            ])
            return nd_front.get("X"), nd_front.get("F"), None, metadata
        else:
            metadata = np.column_stack([
                survivors.get("rank"),
                survivors.get("niche"),
                survivors.get("dist_to_niche")
            ])
            return survivors.get("X"), survivors.get("F"), None, metadata

    else:
        pop = Population.new("X", population_data, "F", objective_values, "G", constraint_values)

        from pymoo.core.problem import Problem
        class DummyProblem(Problem):
            def __init__(self, n_obj, n_constr):
                super().__init__(n_var=population_data.shape[1], n_obj=n_obj, n_constr=n_constr)

        dummy_problem = DummyProblem(n_obj=M, n_constr=constraint_values.shape[1])

        survival = ReferenceDirectionSurvival(ref_dirs)
        survivors = survival.do(dummy_problem, pop, n_survive=limit)

        if ndf:
            nd_front = survivors[survivors.get("rank") == 0]
            metadata = np.column_stack([
                nd_front.get("rank"),
                nd_front.get("niche"),
                nd_front.get("dist_to_niche")
            ])
            return nd_front.get("X"), nd_front.get("F"), nd_front.get("G"), metadata
        else:
            metadata = np.column_stack([
                survivors.get("rank"),
                survivors.get("niche"),
                survivors.get("dist_to_niche")
            ])
            # mark infeasible solutions (rank=None)
            infeasible = np.equal(survivors.get("rank"), None)
            metadata[infeasible] = [np.inf, -1, np.inf]
            return survivors.get("X"), survivors.get("F"), survivors.get("G"), metadata


def ranking_crowding_general(population_data, objective_values, constraint_values, limit, ndf=False):

    class DummyProblem(Problem):
        def __init__(self, n_var, n_obj, n_constr):
            super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr)
        
        def _evaluate(self, x, out, *args, **kwargs):
            pass

    pop = Population.new("X", population_data, "F", objective_values, "G", constraint_values)
    n_constr = constraint_values.shape[1]
    
    dummy_problem = DummyProblem(n_var=population_data.shape[1], 
                                 n_obj=objective_values.shape[1], 
                                 n_constr=n_constr)

    survival = RankAndCrowdingSurvival()
    survivors = survival.do(dummy_problem, pop, n_survive=limit)

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

    return p_array, o_array, c_array, metadata


