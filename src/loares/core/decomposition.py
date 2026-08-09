"""
Decomposition-based algorithm framework (MOEA/D-style).

DecompositionAlgorithm decomposes a multi-objective problem into scalar
subproblems using weight vectors. Each subproblem maintains one solution,
and offspring compete against neighborhood solutions via scalarization.

Composable: any Recombination + PoolSelection + Mutation can be plugged in.
"""

import numpy as np
from scipy.spatial.distance import cdist

from pymoo.core.algorithm import Algorithm
from pymoo.core.infill import InfillCriterion
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.core.repair import NoRepair
from pymoo.core.duplicate import NoDuplicateElimination
from pymoo.util.optimum import filter_optimum
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from pymoo.util.normalization import normalize

class NeighborhoodPoolSelection:
    """
    Pool selection for decomposition: picks best/worst per subproblem
    from its neighborhood, ranked by scalarized fitness.
    """

    provides = {"best", "worst", "mean", "random"}

    def do(self, pop, algorithm=None, random_state=None, **kwargs):
        return self._do(pop, algorithm=algorithm, random_state=random_state)

    def _do(self, pop, algorithm=None, random_state=None, **kwargs):
        X = pop.get("X")
        F = pop.get("F")
        n = len(pop)

        best_list = []
        worst_list = []

        for i in range(n):
            if random_state.random() < algorithm.prob_neighbor:
                pool_idx = algorithm.neighbors[i]
            else:
                pool_idx = np.arange(n)

            scores = np.array([algorithm._scalar(F[j], i) for j in pool_idx])
            best_list.append(X[pool_idx[np.argmin(scores)]])
            worst_list.append(X[pool_idx[np.argmax(scores)]])

        return {
            "best": np.array(best_list),
            "worst": np.array(worst_list),
            "mean": np.mean(X, axis=0),
            "random": X[random_state.integers(0, n, size=n)],
        }


class DecompositionAlgorithm(Algorithm):
    """
    MOEA/D-style decomposition algorithm with composable operators.

    Parameters
    ----------
    name : str
        Algorithm name.
    ref_dirs : ndarray (N, M)
        Weight vectors defining subproblems. pop_size = len(ref_dirs).
    infill : InfillCriterion
        Offspring generation (RecombinationVariant with NeighborhoodPoolSelection).
    n_neighbors : int
        Neighborhood size T.
    scalarization : str
        "tchebycheff", "pbi", or "ws".
    prob_neighbor : float
        Probability of mating within neighborhood vs whole population.
    pbi_theta : float
        Penalty parameter for PBI scalarization.
    sampling : pymoo Sampling
        Initial population generator.
    repair : pymoo Repair or None
        Bounds repair.
    """

    def __init__(self, name, ref_dirs, infill, n_neighbors=20,
                 scalarization="tchebycheff", prob_neighbor=0.9,
                 pbi_theta=5.0, sampling=None, repair=None, **kwargs):

        super().__init__(**kwargs)

        self.name = name
        self.ref_dirs = np.array(ref_dirs)
        self.pop_size = len(ref_dirs)
        self.n_neighbors = min(n_neighbors, self.pop_size)
        self.scalarization = scalarization
        self.prob_neighbor = prob_neighbor
        self.pbi_theta = pbi_theta
        self.infill_criterion = infill
        self.repair = repair if repair is not None else NoRepair()

        dists = cdist(self.ref_dirs, self.ref_dirs)
        self.neighbors = np.argsort(dists, axis=1)[:, :self.n_neighbors]

        from pymoo.operators.sampling.rnd import FloatRandomSampling
        if sampling is None:
            sampling = FloatRandomSampling()

        self.initialization = Initialization(
            sampling,
            repair=self.repair,
            eliminate_duplicates=NoDuplicateElimination(),
        )

        self.z_ = None

    def _initialize_infill(self):
        return self.initialization.do(
            self.problem, self.pop_size,
            algorithm=self, random_state=self.random_state
        )

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.z_ = self.pop.get("F").min(axis=0).copy()
        self.z_nadir = self.pop.get("F").max(axis=0).copy()
        self.nadir_norm = np.ones(self.pop.get("F").shape[1])

    def _infill(self):
        off = self.infill_criterion.do(
            self.problem, self.pop, self.pop_size,
            algorithm=self, random_state=self.random_state
        )
        if off is None or len(off) == 0:
            self.termination.force_termination = True
            return None
        off = self.repair.do(self.problem, off, random_state=self.random_state)
        return off

    def _advance(self, infills=None, **kwargs):
        F_off = infills.get("F")

        self.z_ = np.minimum(self.z_, F_off.min(axis=0))
        self.z_nadir = np.maximum(self.z_nadir, F_off.max(axis=0))
        # self.z_nadir = F_off.max(axis=0)

        F_pop = self.pop.get("F").copy()

        for i in range(len(infills)):
            f_off = F_off[i]
            for j in self.neighbors[i]:
                if self._scalar(f_off, j) < self._scalar(F_pop[j], j):
                    self.pop[j] = infills[i]
                    F_pop[j] = f_off

    def _scalar(self, f, idx):
        lam = self.ref_dirs[idx]
        if self.scalarization == "tchebycheff":
            return np.max(lam * np.abs(f - self.z_))
        elif self.scalarization == "ws":
            return np.dot(lam, f)
        elif self.scalarization == "pbi":
            diff = f - self.z_
            lam_norm = lam / np.linalg.norm(lam)
            d1 = np.dot(diff, lam_norm)
            d2 = np.linalg.norm(diff - d1 * lam_norm)
            return d1 + self.pbi_theta * d2
        elif self.scalarization == "i-pbi":
            # f_norm = normalize(f, self.z_, self.z_nadir)
            # diff = self.nadir_norm - f_norm
            diff = self.z_nadir - f
            lam_norm = lam / np.linalg.norm(lam)
            d1 = np.dot(diff, lam_norm)
            d2 = np.linalg.norm(diff - d1 * lam_norm)
            return -d1 + self.pbi_theta * d2

    def _set_optimum(self):
        F = self.pop.get("F")
        ndf_idx = find_non_dominated(F)
        self.opt = self.pop[ndf_idx]
