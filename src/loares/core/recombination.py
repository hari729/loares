"""
Population-level recombination operators for composable evolutionary algorithms.

These are the update equations from the BWR/BMR/BMWR family. Unlike pymoo's
Crossover (which operates on parent pairs), these operate on the entire population
simultaneously using named parent roles (best, worst, mean, random).

The recombination does NOT clip to bounds — that is the Repair's job.
"""

import numpy as np
from abc import abstractmethod


class Recombination:
    """
    Base class for population-level recombination.

    A Recombination takes the current population and a pool of parent arrays
    (identified by role — best, worst, mean, random, etc.) and produces new
    X values for every individual simultaneously.

    Subclasses implement _do() with the specific equation.
    """

    def __init__(self):
        pass

    def do(self, problem, pop, pool, random_state=None, **kwargs):
        """
        Execute the recombination.

        Parameters
        ----------
        problem : pymoo Problem
        pop : pymoo Population — the current population
        pool : dict — parent arrays keyed by role name (e.g. "best", "worst", "mean")
        random_state : numpy Generator

        Returns
        -------
        X : np.ndarray of shape (n, n_var) — new decision variable values
        """
        return self._do(problem, pop, pool, random_state=random_state, **kwargs)

    @abstractmethod
    def _do(self, problem, pop, pool, random_state=None, **kwargs):
        """Override this with the recombination equation."""
        pass

    @property
    def requires(self):
        """
        Return a set of pool keys this recombination needs.
        Used for validation — if the selection doesn't provide a required key,
        fail early with a clear message instead of a KeyError deep in the math.
        """
        return set()


class BWR(Recombination):
    """
    Best-Worst-Random recombination.

    Equation (from Rao & Davim 2025):
        X' = X + r1 * (best - F * random) - r2 * (worst - random)

    where F is randomly 1 or 2 per individual, and r1, r2 are uniform [0, 1].
    """

    @property
    def requires(self):
        return {"best", "worst", "random"}

    def _do(self, problem, pop, pool, random_state=None, **kwargs):
        X = pop.get("X")
        n, n_var = X.shape

        best = pool["best"]
        worst = pool["worst"]
        rand = pool["random"]

        r1 = random_state.random((n, 1))
        r2 = random_state.random((n, 1))
        F = random_state.choice([1, 2], size=(n, 1))

        return X + r1 * (best - F * rand) - r2 * (worst - rand)


class BMR(Recombination):
    """
    Best-Mean-Random recombination.

    Equation (from Rao & Davim 2025):
        X' = X + r1 * (best - F * mean) + r2 * (best - random)
    """

    @property
    def requires(self):
        return {"best", "mean", "random"}

    def _do(self, problem, pop, pool, random_state=None, **kwargs):
        X = pop.get("X")
        n, n_var = X.shape

        best = pool["best"]
        mean = pool["mean"]
        rand = pool["random"]

        r1 = random_state.random((n, 1))
        r2 = random_state.random((n, 1))
        F = random_state.choice([1, 2], size=(n, 1))

        return X + r1 * (best - F * mean) + r2 * (best - rand)


class BMWR(Recombination):
    """
    Best-Mean-Worst-Random recombination.

    Equation (from Rao & Davim 2025):
        X' = X + r1 * (best - F * mean) - r2 * (worst - random)
    """

    @property
    def requires(self):
        return {"best", "worst", "mean", "random"}

    def _do(self, problem, pop, pool, random_state=None, **kwargs):
        X = pop.get("X")
        n, n_var = X.shape

        best = pool["best"]
        worst = pool["worst"]
        mean = pool["mean"]
        rand = pool["random"]

        r1 = random_state.random((n, 1))
        r2 = random_state.random((n, 1))
        F = random_state.choice([1, 2], size=(n, 1))

        return X + r1 * (best - F * mean) - r2 * (worst - rand)
