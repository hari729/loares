from loares.base.bxr import BxR
from loares.operators.bxr import BMR, BWR, BMWR
from loares.operators.selection.bxr_cd import CD_BW_selection
from loares.operators.mutation import RandomReinit

from pymoo.operators.survival.rank_and_crowding import RankAndCrowding


class MO_BMR_CD(BxR):
    def __init__(
        self,
        pop_size,
        mutation_prob=0.5,
        **kwargs,
    ):
        super().__init__(
            pop_size=pop_size,
            core_operator=BMR(),
            selection=CD_BW_selection,
            survival=RankAndCrowding(),
            mutation=RandomReinit(prob=mutation_prob),
            **kwargs,
        )


class MO_BWR_CD(BxR):
    def __init__(
        self,
        pop_size,
        mutation_prob=0.5,
        **kwargs,
    ):
        super().__init__(
            pop_size=pop_size,
            core_operator=BWR(),
            selection=CD_BW_selection,
            survival=RankAndCrowding(),
            mutation=RandomReinit(prob=mutation_prob),
            **kwargs,
        )


class MO_BMWR_CD(BxR):
    def __init__(
        self,
        pop_size,
        mutation_prob=0.5,
        **kwargs,
    ):
        super().__init__(
            pop_size=pop_size,
            core_operator=BMWR(),
            selection=CD_BW_selection,
            survival=RankAndCrowding(),
            mutation=RandomReinit(prob=mutation_prob),
            **kwargs,
        )
