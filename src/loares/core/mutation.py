"""
Mutation operators for the BWR/BMR/BMWR algorithm family.

These subclass pymoo's Mutation and follow its do/_do contract.
The outer Mutation.do() handles per-individual probability gating.
The inner _do() implements the variable-level mutation logic.

RandomReinit corresponds to the r4 <= 0.5 branch in the paper:
    X'[x,k,i] = Hx - (Hx - Lx) * r3

This is the "fallback to uniform random within bounds" that gives the
algorithms their exploration capability.

Note: Neither mutation clips to bounds. That is Repair's job.
"""

import numpy as np
from pymoo.core.mutation import Mutation


class RandomReinit(Mutation):
    """
    Random reinitialization mutation.

    For each mutated individual, each variable is independently replaced
    with a uniform random value within bounds with probability pb.

    This is the r4 <= 0.5 branch from the BWR/BMR/BMWR paper:
        X' = Hx - (Hx - Lx) * r3
    which simplifies to:
        X' = Lx + (Hx - Lx) * r    (uniform random in [Lx, Hx])

    Parameters
    ----------
    prob : float
        Probability that an individual is mutated at all.
        This is the outer probability (pymoo's Mutation handles this).
        Corresponds to r4 <= 0.5 check, so default is 0.5.
    """

    def __init__(self, prob=0.5, **kwargs):
        super().__init__(prob=prob, **kwargs)

    def _do(self, problem, X, random_state=None, **kwargs):
        xl, xu = problem.bounds()
        n, n_var = X.shape
        r = random_state.random((n, n_var))          # (N, 1) not (N, n_var)
        return xu - r * (xu - xl)                 # matches loares formula

class QOPPReinit(Mutation):
    """
    Quasi-Opposition-based Population reinitialization mutation.

    For each mutated individual, variables are replaced with quasi-opposition
    points: random values between the midpoint and the opposition point.

    Parameters
    ----------
    prob : float
        Probability that an individual is mutated at all.
    """

    def __init__(self, prob=0.5, **kwargs):
        super().__init__(prob=prob, **kwargs)

    def _do(self, problem, X, random_state=None, **kwargs):
        xl, xu = problem.bounds()
        n, n_var = X.shape

        mid = (xl + xu) / 2.0
        opp = (xl + xu) - X
        low = np.minimum(mid, opp)
        high = np.maximum(mid, opp)

        r = random_state.random((n, n_var))
        qopp = low + r * (high - low)

        return qopp
