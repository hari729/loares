"""
Sub-population policies for adaptive population splitting.

A SubPopPolicy decides:
    - WHEN to change the number of sub-populations (adapt)
    - HOW to partition self.pop into sub-populations (split)

When plugged into ModularAlgorithm via sub_pop_policy, each sub-population
evolves independently per generation, then merges back into self.pop.
"""

import numpy as np
from abc import ABC, abstractmethod
from pymoo.indicators.hv import HV
from pymoo.util.normalization import normalize


class SubPopPolicy(ABC):
    """
    Base class for sub-population splitting policies.

    Handles interval checking logic. Subclasses implement _adapt() for
    metric-based decisions and split() for partitioning.

    Parameters
    ----------
    initial_n : int
        Starting number of sub-populations.
    check_every : tuple
        When to check for adaptation. Formats:
            ("n_eval", 100)  — every 100 evaluations
            ("n_gen", 5)     — every 5 generations
            ("frac", 0.05)   — every 5% of max_evals, falls back to
                               generation-based if max_evals not detectable
    """

    def __init__(self, initial_n=2, check_every=("frac", 0.05),
                 min_subpop_size=None):
        self.initial_n = initial_n
        self.n = initial_n
        self.check_every = check_every
        self._min_subpop_size = min_subpop_size
        self._max_n = None
        self._check_mode = None
        self._check_interval = None
        self._last_check = 0

    def setup(self, algorithm):
        pop_size = algorithm.pop_size

        if self._min_subpop_size is None:
            self._min_subpop_size = max(4, pop_size // 10)
        self._max_n = max(1, pop_size // self._min_subpop_size)

        if self.initial_n > self._max_n:
            self.initial_n = self._max_n
            self.n = self._max_n

        mode, value = self.check_every

        if mode == "n_eval":
            self._check_mode = "n_eval"
            self._check_interval = max(int(value), 1)
        elif mode == "n_gen":
            self._check_mode = "n_gen"
            self._check_interval = max(int(value), 1)
        elif mode == "frac":
            max_evals = self._try_get_max_evals(algorithm)
            if max_evals is not None:
                self._check_mode = "n_eval"
                self._check_interval = max(int(max_evals * value), 1)
            else:
                self._check_mode = "n_gen"
                self._check_interval = max(1, int(1.0 / value))
        else:
            raise ValueError(f"Unknown check_every mode: {mode}")

        self._last_check = self._current_counter(algorithm)
        self._setup(algorithm)

    def _try_get_max_evals(self, algorithm):
        t = algorithm.termination
        if hasattr(t, "n_max_evals"):
            return t.n_max_evals
        if hasattr(t, "max_evals") and hasattr(t.max_evals, "n_max_evals"):
            return t.max_evals.n_max_evals
        return None

    def _current_counter(self, algorithm):
        if self._check_mode == "n_eval":
            return algorithm.evaluator.n_eval
        return algorithm.n_gen

    def _should_check(self, algorithm):
        current = self._current_counter(algorithm)
        if (current // self._check_interval) > (self._last_check // self._check_interval):
            self._last_check = current
            return True
        return False

    def adapt(self, algorithm) -> int:
        if not self._should_check(algorithm):
            return self.n
        self.n = self._adapt(algorithm)
        return self.n

    @abstractmethod
    def _setup(self, algorithm):
        """Called once after interval resolution. Initialize metric state."""

    @abstractmethod
    def _adapt(self, algorithm) -> int:
        """Evaluate metrics and return desired sub-population count."""

    @abstractmethod
    def split(self, pop, n_subpops, random_state) -> list:
        """Partition population into n sub-populations."""


class HVAdaptiveSplit(SubPopPolicy):
    """
    Adapts sub-population count based on hypervolume change.

    HV improves  -> increase sub-pops (more exploration)
    HV stagnates -> decrease sub-pops (consolidate)

    Parameters
    ----------
    initial_n : int
        Starting number of sub-populations.
    check_every : tuple
        Interval specification (see SubPopPolicy).
    max_n_frac : float
        Maximum sub-pops as fraction of pop_size.
    ref_point : np.ndarray or None
        HV reference point. If None, uses ones(n_obj) + 1e-5.
    """

    def __init__(self, initial_n=2, check_every=("frac", 0.05),
                 min_subpop_size=None, ref_point=None):
        super().__init__(initial_n=initial_n, check_every=check_every,
                         min_subpop_size=min_subpop_size)
        self._ref_point = ref_point
        self._prev_hv = None

    def _setup(self, algorithm):
        n_obj = algorithm.problem.n_obj

        if self._ref_point is None:
            self._ref_point = np.ones(n_obj) + 1e-5

        self._prev_hv = self._calc_hv(algorithm)

    def _calc_hv(self, algorithm):
        F = algorithm.pop.get("F")
        if F is None or len(F) == 0:
            return 0.0

        fmax = F.max(axis=0)
        fmin = F.min(axis=0)
        denom = fmax - fmin
        denom[denom < 1e-12] = 1.0
        F_norm = (F - fmin) / denom

        hv = HV(ref_point=self._ref_point)
        return hv(F_norm)

    def _adapt(self, algorithm) -> int:
        new_hv = self._calc_hv(algorithm)

        if new_hv > self._prev_hv:
            new_n = min(self.n + 1, self._max_n)  # _max_n from base class
        elif self.n > 1:
            new_n = self.n - 1
        else:
            new_n = self.n

        self._prev_hv = new_hv
        return new_n

    def split(self, pop, n_subpops, random_state) -> list:
        n = len(pop)
        if n_subpops > n:
            n_subpops = n
        idx = np.arange(n)
        random_state.shuffle(idx)
        parts = np.array_split(idx, n_subpops)
        return [pop[p] for p in parts]
