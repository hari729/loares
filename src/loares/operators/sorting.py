import numpy as np
from scipy.spatial.distance import cdist

from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize


def farthest_point_sampling(points, n_samples):
    n_obj = points.shape[1]
    selected = []
    npoints = normalize(points, np.min(points, axis=0), np.max(points, axis=0))
    for j in range(n_obj):
        selected.append(np.argmin(npoints[:, j]))
        selected.append(np.argmax(npoints[:, j]))
    selected = list(dict.fromkeys(selected))

    min_dist = cdist(npoints, npoints[selected]).min(axis=1)

    for _ in range(n_samples - len(selected)):
        idx = np.argmax(min_dist)
        selected.append(idx)
        new_dist = cdist(npoints, npoints[idx : idx + 1]).flatten()
        min_dist = np.minimum(min_dist, new_dist)
    return selected


class NDSFarthestPointSurvival(Survival):
    """
    Non-dominated sorting + farthest point sampling survival.

    Adds fronts (rank 0, 1, 2, ...) until n_survive is reached.
    On the splitting front, uses FPS to select the remaining slots
    for maximum spread. Same pattern as RankAndCrowding but with
    FPS replacing crowding distance for the splitting decision.
    """

    def __init__(self):
        super().__init__(filter_infeasible=True)
        self._nds = NonDominatedSorting()

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        F = pop.get("F").astype(float, copy=False)

        if n_survive is None:
            n_survive = len(pop)

        fronts = self._nds.do(F, n_stop_if_ranked=n_survive)

        survivors = []
        for k, front in enumerate(fronts):
            remaining = n_survive - len(survivors)

            if len(front) <= remaining:
                for i in front:
                    pop[i].set("rank", k)
                survivors.extend(front)
            else:
                front_F = F[front]
                fps_idx = farthest_point_sampling(front_F, remaining)
                selected = front[fps_idx]
                for i in selected:
                    pop[i].set("rank", k)
                survivors.extend(selected)
                break

        return pop[survivors]
