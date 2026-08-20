from numpy.random import rand
from pandas.core.common import random_state
from pymoo.core import algorithm
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.repair.to_bound import ToBoundOutOfBoundsRepair


class BxR(Algorithm):
    def __init__(
        self,
        *,
        pop_size,
        core_operator=None,
        selection=None,
        survival=None,
        mutation=None,
        sampling=FloatRandomSampling(),
        archive=None,
        repair=ToBoundOutOfBoundsRepair(),
        **kwargs,
    ):
        super().__init__(
            archive=archive,
            **kwargs,
        )

        self.pop_size = pop_size
        self.core_operator = core_operator
        self.selection = selection
        self.mutation = mutation
        self.survival = survival
        self.repair = repair
        self.sampling = sampling

    def _initialize_infill(self):
        pop = self.sampling.do(
            self.problem, self.pop_size, random_state=self.random_state
        )
        pop = self.repair.do(self.problem, pop)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = self.survival.do(
            self.problem,
            infills,
            n_survive=len(infills),
            random_state=self.random_state,
            algorithm=self,
            **kwargs,
        )

    def _infill(self):
        pool = self.selection(self.pop, random_state=self.random_state, algorithm=self)
        X_new = self.core_operator.do(
            self.problem, self.pop.get("X"), pool, random_state=self.random_state
        )
        infills = Population.new("X", X_new)
        infills = self.mutation.do(
            self.problem, infills, random_state=self.random_state
        )
        infills = self.repair.do(self.problem, infills)
        return infills

    def _advance(self, infills=None, **kwargs):
        pop = self.pop

        if infills is not None:
            pop = Population.merge(self.pop, infills)

        self.pop = self.survival.do(
            self.problem,
            pop,
            n_survive=self.pop_size,
            random_state=self.random_state,
            algorithm=self,
            **kwargs,
        )
