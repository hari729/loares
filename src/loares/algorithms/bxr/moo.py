"""
Pre-configured multi-objective algorithms.

These are convenience constructors that assemble ModularAlgorithm with
the right components for each algorithm variant. Users who want the
standard configurations use these. Users who want to experiment with
novel combinations use ModularAlgorithm directly.
"""

from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.operators.repair.to_bound import ToBoundOutOfBoundsRepair
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.archive import MultiObjectiveArchive, SurvivalTruncation

from loares.core.composable import ModularAlgorithm, RecombinationVariant
from loares.core.recombination import BMR, BWR, BMWR
from loares.core.pool_selection import BestWorstSelection, ArchiveBestWorstSelection
from loares.core.mutation import RandomReinit
from loares.core.mods import LocalSearchMod, OppositionMod, EdgeBoostMod


def _build_mo(recombination_cls, pop_size, mods, mutation, sampling, survival,
              pool_selection, repair, output, termination, **kwargs):
    """Internal builder for MO algorithm variants."""

    if pool_selection is None:
        pool_selection = BestWorstSelection()
    if mutation is None:
        mutation = RandomReinit(prob=0.5, pb=1.0)
    if sampling is None:
        sampling = FloatRandomSampling()
    if survival is None:
        survival = RankAndCrowding()
    if repair is None:
        repair = ToBoundOutOfBoundsRepair()
    if mods is None:
        mods = [LocalSearchMod()]
    if output is None:
        output = MultiObjectiveOutput()
    if termination is None:
        termination = DefaultMultiObjectiveTermination()

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

class MORankingCrowding(ModularAlgorithm):
    """Base for BXR family: BW selection + RankAndCrowding survival."""
    
    def __init__(self, infill, pop_size, mods, **kwargs):
        super().__init__(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            infill=infill,
            survival=RankAndCrowding(),
            mods=mods,
            repair=ToBoundOutOfBoundsRepair(),
            advance_after_initial_infill=True,
            output=MultiObjectiveOutput(),
            **kwargs,
        )
        self.termination = DefaultMultiObjectiveTermination()


BMR_DInfill = RecombinationVariant(
    pool_selection=BestWorstSelection(),
    recombination=BMR(),
    mutation=RandomReinit(prob=0.5, pb=1.0),
    repair=ToBoundOutOfBoundsRepair(),
)
BWR_DInfill = RecombinationVariant(
    pool_selection=BestWorstSelection(),
    recombination=BWR(),
    mutation=RandomReinit(prob=0.5, pb=1.0),
    repair=ToBoundOutOfBoundsRepair(),
)
BMWR_DInfill = RecombinationVariant(
    pool_selection=BestWorstSelection(),
    recombination=BMWR(),
    mutation=RandomReinit(prob=0.5, pb=1.0),
    repair=ToBoundOutOfBoundsRepair(),
)

class MO_BMR_py(MORankingCrowding):
    def __init__(self, pop_size=100, **kwargs):
        super().__init__(BMR_DInfill, pop_size, [LocalSearchMod(),EdgeBoostMod()], **kwargs)

class MO_BWR(MORankingCrowding):
    def __init__(self, pop_size=100, **kwargs):
        super().__init__(BWR_DInfill, pop_size, [LocalSearchMod(),EdgeBoostMod()], **kwargs)

class MO_BMWR(MORankingCrowding):
    def __init__(self, pop_size=100, **kwargs):
        super().__init__(BMWR_DInfill, pop_size, [LocalSearchMod(),EdgeBoostMod()], **kwargs)

class MO_BMR_Opposition(MORankingCrowding):
    def __init__(self, pop_size=100, **kwargs):
        super().__init__(BMR_DInfill, pop_size, [LocalSearchMod(),EdgeBoostMod(), OppositionMod()], **kwargs)

class MO_BWR_Opposition(MORankingCrowding):
    def __init__(self, pop_size=100, **kwargs):
        super().__init__(BWR_DInfill, pop_size, [LocalSearchMod(),EdgeBoostMod(), OppositionMod()], **kwargs)

class MO_BMWR_Opposition(MORankingCrowding):
    def __init__(self, pop_size=100, **kwargs):
        super().__init__(BMWR_DInfill, pop_size, [LocalSearchMod(),EdgeBoostMod(), OppositionMod()], **kwargs)



# Archive with crowding-based truncation — matching loares behavior
Crowding_Archive = MultiObjectiveArchive(
    max_size=200,
    truncation=SurvivalTruncation(RankAndCrowding()),
)

BMR_ArchiveInfill = RecombinationVariant(
    pool_selection=ArchiveBestWorstSelection(),  # reads from algorithm.archive
    recombination=BMR(),
    mutation=RandomReinit(prob=0.5, pb=1.0),
    repair=ToBoundOutOfBoundsRepair(),
)
BWR_ArchiveInfill = RecombinationVariant(
    pool_selection=ArchiveBestWorstSelection(),  # reads from algorithm.archive
    recombination=BWR(),
    mutation=RandomReinit(prob=0.5, pb=1.0),
    repair=ToBoundOutOfBoundsRepair(),
)
BMWR_ArchiveInfill = RecombinationVariant(
    pool_selection=ArchiveBestWorstSelection(),  # reads from algorithm.archive
    recombination=BMWR(),
    mutation=RandomReinit(prob=0.5, pb=1.0),
    repair=ToBoundOutOfBoundsRepair(),
)

class MO_BMR_Archive_py(MORankingCrowding):
    def __init__(self, pop_size=100, **kwargs):
        archive = MultiObjectiveArchive(
            max_size=pop_size * 2,
            truncation=SurvivalTruncation(RankAndCrowding()),
        )
        super().__init__(BMR_ArchiveInfill, pop_size, [LocalSearchMod()],
                         archive=archive, **kwargs)

class MO_BWR_Archive(MORankingCrowding):
    def __init__(self, pop_size=100, **kwargs):
        archive = MultiObjectiveArchive(
            max_size=pop_size * 2,
            truncation=SurvivalTruncation(RankAndCrowding()),
        )
        super().__init__(BWR_ArchiveInfill, pop_size, [LocalSearchMod()],
                         archive=archive, **kwargs)

class MO_BMWR_Archive(MORankingCrowding):
    def __init__(self, pop_size=100, **kwargs):
        archive = MultiObjectiveArchive(
            max_size=pop_size * 2,
            truncation=SurvivalTruncation(RankAndCrowding()),
        )
        super().__init__(BMWR_ArchiveInfill, pop_size, [LocalSearchMod()],
                         archive=archive, **kwargs)
