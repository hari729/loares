"""
Custom survival operators extending pymoo's Survival.

NDSFarthestPointSurvival provides an alternative to crowding distance
for maintaining diversity on the Pareto front, using farthest point
sampling instead.
"""

import numpy as np
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize
from scipy.spatial.distance import cdist


def farthest_point_sampling(points, n_samples):
    """
    Select n_samples points from the given set using farthest point sampling.

    Greedily selects the point farthest from all previously selected points,
    starting from the extreme points in each objective dimension.

    Parameters
    ----------
    points : np.ndarray of shape (n, n_obj)
    n_samples : int

    Returns
    -------
    selected : list of int — indices into points
    """
    n_obj = points.shape[1]
    npoints = normalize(points, np.min(points, axis=0), np.max(points, axis=0))

    # Start with extreme points in each objective
    selected = []
    for j in range(n_obj):
        selected.append(int(np.argmin(npoints[:, j])))
        selected.append(int(np.argmax(npoints[:, j])))
    selected = list(dict.fromkeys(selected))  # deduplicate, preserve order

    if len(selected) >= n_samples:
        return selected[:n_samples]

    min_dist = cdist(npoints, npoints[selected]).min(axis=1)

    while len(selected) < n_samples:
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        new_dist = cdist(npoints, npoints[idx:idx + 1]).flatten()
        min_dist = np.minimum(min_dist, new_dist)

    return selected


class NDSFarthestPointSurvival(Survival):
    """
    Non-dominated sorting with farthest point sampling for diversity.

    On the splitting front (the last front that fits within n_survive),
    uses farthest point sampling instead of crowding distance to select
    which individuals survive.

    Parameters
    ----------
    nds : NonDominatedSorting instance or None
    """

    def __init__(self, nds=None):
        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()

    def _do(self, problem, pop, *args, n_survive=None, random_state=None, **kwargs):
        F = pop.get("F").astype(float, copy=False)
        survivors = []

        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):
            # Set rank on all front members
            for i in front:
                pop[i].set("rank", k)

            if len(survivors) + len(front) <= n_survive:
                survivors.extend(front)
            else:
                # Splitting front: use FPS to select from this front
                n_remaining = n_survive - len(survivors)
                if n_remaining > 0:
                    selected = farthest_point_sampling(F[front], n_remaining)
                    survivors.extend(front[selected])
                break

        return pop[survivors]
