"""
Pre-configured single-objective algorithms.
"""

from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.operators.repair.to_bound import ToBoundOutOfBoundsRepair
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.util.display.single import SingleObjectiveOutput

from loares.algorithms.composable import ModularAlgorithm, RecombinationVariant
from loares.operators.recombination import BMR, BWR, BMWR
from loares.operators.pool_selection import BestWorstSelection
from loares.operators.mutation import RandomReinit


def _build_so(recombination_cls, pop_size, mods, mutation, sampling, survival,
              pool_selection, repair, output, termination, **kwargs):
    """Internal builder for SO algorithm variants."""

    if pool_selection is None:
        pool_selection = BestWorstSelection()
    if mutation is None:
        mutation = RandomReinit(prob=0.5, pb=1.0)
    if sampling is None:
        sampling = FloatRandomSampling()
    if survival is None:
        survival = FitnessSurvival()
    if repair is None:
        repair = ToBoundOutOfBoundsRepair()
    if mods is None:
        mods = []
    if output is None:
        output = SingleObjectiveOutput()
    if termination is None:
        termination = DefaultSingleObjectiveTermination()

    algo = ModularAlgorithm(
        pop_size=pop_size,
        sampling=sampling,
        infill=RecombinationVariant(
            pool_selection=pool_selection,
            recombination=recombination_cls(),
            mutation=mutation,
            repair=repair,
        ),
        survival=survival,
        mods=mods,
        repair=repair,
        advance_after_initial_infill=True,
        output=output,
        **kwargs,
    )
    algo.termination = termination
    return algo


def SO_BMR(pop_size=100, mods=None, mutation=None, sampling=None,
           survival=None, pool_selection=None, repair=None,
           output=None, termination=None, **kwargs):
    """Single-objective BMR algorithm."""
    return _build_so(BMR, pop_size, mods, mutation, sampling, survival,
                     pool_selection, repair, output, termination, **kwargs)


def SO_BWR(pop_size=100, mods=None, mutation=None, sampling=None,
           survival=None, pool_selection=None, repair=None,
           output=None, termination=None, **kwargs):
    """Single-objective BWR algorithm."""
    return _build_so(BWR, pop_size, mods, mutation, sampling, survival,
                     pool_selection, repair, output, termination, **kwargs)


def SO_BMWR(pop_size=100, mods=None, mutation=None, sampling=None,
            survival=None, pool_selection=None, repair=None,
            output=None, termination=None, **kwargs):
    """Single-objective BMWR algorithm."""
    return _build_so(BMWR, pop_size, mods, mutation, sampling, survival,
                     pool_selection, repair, output, termination, **kwargs)
