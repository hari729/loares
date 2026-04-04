"""
Tournament selection functions compatible with ModularAlgorithm.

pymoo's binary_tournament (from nsga2.py) depends on algorithm.tournament_type,
which is NSGA-II specific. These functions work with any algorithm class that
uses RankAndCrowding survival (which sets rank and crowding on individuals).
"""

import numpy as np
from pymoo.operators.selection.tournament import compare
from pymoo.util.dominator import Dominator


def rank_and_crowding_tournament(pop, P, random_state=None, **kwargs):
    """
    Binary tournament using non-dominated rank and crowding distance.

    Compares two individuals:
    1. If either is infeasible, prefer the one with lower constraint violation.
    2. If both feasible, prefer lower rank (better Pareto front).
    3. If same rank, prefer higher crowding distance (more isolated).

    This is equivalent to NSGA-II's 'comp_by_rank_and_crowding' mode but
    doesn't require algorithm.tournament_type.

    Use with: TournamentSelection(func_comp=rank_and_crowding_tournament)
    """
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):
        a, b = P[i, 0], P[i, 1]
        a_cv, b_cv = pop[a].CV[0], pop[b].CV[0]

        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(a, a_cv, b, b_cv,
                           method='smaller_is_better',
                           return_random_if_equal=True,
                           random_state=random_state)
        else:
            rank_a, cd_a = pop[a].get("rank", "crowding")
            rank_b, cd_b = pop[b].get("rank", "crowding")

            S[i] = compare(a, rank_a, b, rank_b, method='smaller_is_better')

            if np.isnan(S[i]):
                S[i] = compare(a, cd_a, b, cd_b,
                               method='larger_is_better',
                               return_random_if_equal=True,
                               random_state=random_state)

    return S[:, None].astype(int, copy=False)


def dominance_and_crowding_tournament(pop, P, random_state=None, **kwargs):
    """
    Binary tournament using Pareto dominance and crowding distance.

    Like rank_and_crowding_tournament but uses direct dominance comparison
    instead of pre-computed rank. Slightly more expensive per comparison
    but doesn't require rank to be pre-set.

    Use with: TournamentSelection(func_comp=dominance_and_crowding_tournament)
    """
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):
        a, b = P[i, 0], P[i, 1]
        a_cv, b_cv = pop[a].CV[0], pop[b].CV[0]

        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(a, a_cv, b, b_cv,
                           method='smaller_is_better',
                           return_random_if_equal=True,
                           random_state=random_state)
        else:
            rel = Dominator.get_relation(pop[a].F, pop[b].F)
            if rel == 1:
                S[i] = a
            elif rel == -1:
                S[i] = b

            if np.isnan(S[i]):
                cd_a = pop[a].get("crowding")
                cd_b = pop[b].get("crowding")
                S[i] = compare(a, cd_a, b, cd_b,
                               method='larger_is_better',
                               return_random_if_equal=True,
                               random_state=random_state)

    return S[:, None].astype(int, copy=False)
