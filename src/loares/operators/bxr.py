from pymoo.core.operator import Operator
import numpy as np


class BMR(Operator):
    def __init__(self, name=None, vtype=None, repair=None) -> None:
        super().__init__(name, vtype, repair)

    def _do(self, problem, X, pool, random_state, **kwargs):
        n, n_var = X.shape

        best = pool["best"]
        rand = pool["random"]

        r1 = random_state.random((n, 1))
        r2 = random_state.random((n, 1))
        F = random_state.choice([1, 2], size=(n, 1))

        return X + r1 * (best - F * np.mean(X, axis=0)) + r2 * (best - rand)


class BWR(Operator):
    def __init__(self, name=None, vtype=None, repair=None) -> None:
        super().__init__(name, vtype, repair)

    def _do(self, problem, X, pool, random_state, **kwargs):
        n, n_var = X.shape

        best = pool["best"]
        worst = pool["worst"]
        rand = pool["random"]

        r1 = random_state.random((n, 1))
        r2 = random_state.random((n, 1))
        F = random_state.choice([1, 2], size=(n, 1))

        return X + r1 * (best - F * rand) - r2 * (worst - rand)


class BMWR(Operator):
    def __init__(self, name=None, vtype=None, repair=None) -> None:
        super().__init__(name, vtype, repair)

    def _do(self, problem, X, pool, random_state, **kwargs):
        n, n_var = X.shape

        best = pool["best"]
        worst = pool["worst"]
        rand = pool["random"]

        r1 = random_state.random((n, 1))
        r2 = random_state.random((n, 1))
        F = random_state.choice([1, 2], size=(n, 1))

        return X + r1 * (best - F * np.mean(X, axis=0)) - r2 * (worst - rand)
