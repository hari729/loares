"""
Modification operators (Mods) for composable evolutionary algorithms.

A Mod generates additional candidate solutions from the current population
state. These are merged with the core offspring and all compete through
survival. Mods are additive — they contribute extra individuals, they
don't transform existing offspring.

This is the concept that pymoo completely lacks and loares proved out.
Each mod independently generates candidates:
    - LocalSearchMod: perturbs Pareto front members for fine-grained exploitation
    - OppositionMod: generates opposition-based points for diversity

Mods do NOT clip to bounds. The algorithm applies Repair after collecting
all mod outputs.
"""

import numpy as np
from abc import abstractmethod
from pymoo.core.population import Population


class Mod:
    """
    Base class for additive infill modification operators.

    Subclasses implement _do() to generate extra candidate solutions.
    """

    def __init__(self):
        pass

    def do(self, problem, pop, algorithm=None, random_state=None, **kwargs):
        """
        Generate additional candidate solutions.

        Parameters
        ----------
        problem : pymoo Problem
        pop : pymoo Population — current population (evaluated, with rank etc.)
        algorithm : the running algorithm instance
        random_state : numpy Generator

        Returns
        -------
        off : pymoo Population — new individuals with X set (unevaluated)
        """
        return self._do(problem, pop, algorithm=algorithm,
                        random_state=random_state, **kwargs)

    @abstractmethod
    def _do(self, problem, pop, algorithm=None, random_state=None, **kwargs):
        pass


class LocalSearchMod(Mod):
    """
    Local search around Pareto-front members.

    Selects sqrt(|pareto_front|) solutions from the non-dominated front,
    applies small random perturbations, and returns them as new candidates.

    Parameters
    ----------
    factor : float
        Scale of the perturbation relative to variable range.
    """

    def __init__(self, factor=0.05):
        super().__init__()
        self.factor = factor

    def _do(self, problem, pop, algorithm=None, random_state=None, **kwargs):
        rank = pop.get("rank")

        # Select Pareto front (rank 0). If rank not available, use first half.
        if rank is not None and np.any(rank == 0):
            pareto_X = pop[rank == 0].get("X")
        else:
            pareto_X = pop[: len(pop) // 2].get("X")

        if len(pareto_X) == 0:
            return Population.new("X", np.empty((0, problem.n_var)))

        n_search = max(1, int(np.sqrt(len(pareto_X))))
        n_search = min(n_search, len(pareto_X))

        idx = random_state.choice(len(pareto_X), n_search, replace=False)
        base = pareto_X[idx]

        n_var = base.shape[1]
        perturb = (random_state.random((n_search, n_var)) - 0.5) * self.factor
        scale = random_state.random((n_search, 1))

        searched = base + scale * perturb
        return Population.new("X", searched)


class OppositionMod(Mod):
    """
    Opposition-based learning.

    For every individual in the population, generates the opposition point:
        X_opp = (xl + xu) - X

    This produces pop_size extra candidates, significantly increasing
    selection pressure through survival.
    """

    def _do(self, problem, pop, algorithm=None, random_state=None, **kwargs):
        X = pop.get("X")
        xl, xu = problem.bounds()
        opp = (xl + xu) - X
        return Population.new("X", opp)


class EdgeBoostMod(Mod):
    """
    Edge boosting for multi-objective problems.

    Generates solutions near the extreme ends of the Pareto front
    to improve boundary coverage. For each objective, takes the
    best-performing Pareto member and perturbs it.

    Parameters
    ----------
    factor : float
        Perturbation scale.
    """

    def __init__(self, factor=0.02):
        super().__init__()
        self.factor = factor

    def _do(self, problem, pop, algorithm=None, random_state=None, **kwargs):
        rank = pop.get("rank")
        F = pop.get("F")

        if rank is not None and np.any(rank == 0):
            pareto_mask = rank == 0
            pareto_X = pop[pareto_mask].get("X")
            pareto_F = F[pareto_mask]
        else:
            pareto_X = pop.get("X")
            pareto_F = F

        if len(pareto_X) == 0:
            return Population.new("X", np.empty((0, problem.n_var)))

        edges = []
        n_obj = pareto_F.shape[1]
        n_var = pareto_X.shape[1]

        for j in range(n_obj):
            best_idx = np.argmin(pareto_F[:, j])
            base = pareto_X[best_idx]
            perturb = (random_state.random(n_var) - 0.5) * self.factor
            edges.append(base + perturb)

        return Population.new("X", np.array(edges))
