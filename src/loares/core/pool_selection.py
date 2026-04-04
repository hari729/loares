"""
Pool selection operators for population-level recombination.

A PoolSelection examines the current population and constructs a dict of
parent arrays keyed by role name (best, worst, mean, random). The
Recombination then uses these arrays in its equation.

This is NOT the same as pymoo's Selection (which returns index pairs for
Crossover). PoolSelection returns actual solution arrays organized by
population-level roles.

The selection strategy is what changes between algorithm contexts:
    - SO: best/worst by penalized objective value
    - MO: best/worst by non-dominated rank
    - Archive: best from an external archive, worst from current population
    - Custom: any criteria the user defines
"""

import numpy as np
from abc import abstractmethod


class PoolSelection:
    """
    Base class for pool-based parent selection.

    Subclasses implement _do() to construct the pool dict.
    """

    def __init__(self):
        pass

    def do(self, pop, algorithm=None, random_state=None, **kwargs):
        """
        Construct the parent pool from the current population.

        Parameters
        ----------
        pop : pymoo Population — current population with metadata (rank, etc.)
        algorithm : the running algorithm instance (access to archive, generation, etc.)
        random_state : numpy Generator

        Returns
        -------
        pool : dict — arrays keyed by role name, each shape (pop_size, n_var)
        """
        return self._do(pop, algorithm=algorithm, random_state=random_state, **kwargs)

    @abstractmethod
    def _do(self, pop, algorithm=None, random_state=None, **kwargs):
        pass

    @property
    def provides(self):
        """Return a set of pool keys this selection provides."""
        return set()


class BestWorstSelection(PoolSelection):
    """
    Splits population by non-dominated rank into best (rank 0) and worst pools,
    then randomly samples from each to build pop_size-length arrays.

    Also provides the population mean.

    For single-objective use: survival should set rank (FitnessSurvival does this).
    For multi-objective use: survival should set rank (RankAndCrowding does this).

    Works with any survival that sets a "rank" attribute on individuals.
    """

    @property
    def provides(self):
        return {"best", "worst", "mean", "random"}

    def _do(self, pop, algorithm=None, random_state=None, **kwargs):
        X = pop.get("X")
        n = len(X)
        rank = pop.get("rank")

        # Split by rank: best = rank 0, worst = everything else
        # If all solutions have rank 0 (or rank not set), split by halves
        if rank is not None and np.any(rank != 0):
            best_X = X[rank == 0]
            worst_X = X[rank != 0]
        else:
            half = n // 2
            best_X = X[:half]
            worst_X = X[half:]

        # Sample pop_size parents from each group (with replacement)
        best_idx = random_state.integers(0, len(best_X), size=n)
        worst_idx = random_state.integers(0, len(worst_X), size=n)
        rand_idx = random_state.integers(0, n, size=n)

        return {
            "best": best_X[best_idx],
            "worst": worst_X[worst_idx],
            "mean": np.mean(X, axis=0),  # broadcast-compatible (n_var,)
            "random": X[rand_idx],
        }


class ArchiveBestWorstSelection(PoolSelection):
    """
    Like BestWorstSelection, but draws the "best" pool from an external
    archive population instead of the current population.

    The archive is accessed via algorithm.archive or a stored attribute.
    Falls back to BestWorstSelection behavior if no archive is available.
    """

    @property
    def provides(self):
        return {"best", "worst", "mean", "random"}

    def _do(self, pop, algorithm=None, random_state=None, **kwargs):
        X = pop.get("X")
        n = len(X)
        rank = pop.get("rank")

        # Best from archive if available
        archive_pop = getattr(algorithm, "archive_pop", None)
        if archive_pop is not None and len(archive_pop) > 0:
            best_X = archive_pop.get("X")
        elif rank is not None and np.any(rank == 0):
            best_X = X[rank == 0]
        else:
            best_X = X[: n // 2]

        # Worst from current population
        if rank is not None and np.any(rank != 0):
            worst_X = X[rank != 0]
        else:
            worst_X = X[n // 2 :]

        best_idx = random_state.integers(0, len(best_X), size=n)
        worst_idx = random_state.integers(0, len(worst_X), size=n)
        rand_idx = random_state.integers(0, n, size=n)

        return {
            "best": best_X[best_idx],
            "worst": worst_X[worst_idx],
            "mean": np.mean(X, axis=0),
            "random": X[rand_idx],
        }
