import numpy as np

from loares.algorithms.moo.base import MORankingCrowdingAlgo
from loares.base.bmr import bmr
from loares.base.bwr import bwr
from loares.base.bmwr import bmwr
from loares.base.mutation import random_reinit
from loares.core.update import UpdateRule
from loares.algorithms.moo.selection import archive_bw_selection as bw_selection_a

# def bw_selection_a(population, archive):
#     best = archive.solutions[0,:]
#     worst = population.solutions[-1,:]
#     return {"best":best, "worst":worst}


class UpdateRuleA(UpdateRule):
    def __init__(self, selection, base_function, mutation):
        super().__init__(selection, base_function, mutation)

    def next_gen_a(self, problem, population, archive):
        new_gen = self.base_function(
            problem, population, self.selection(population, archive)
        )
        new_gen = self.mutation(problem, new_gen)
        return new_gen


BMR_a = UpdateRuleA(bw_selection_a, bmr, random_reinit)
BWR_a = UpdateRuleA(bw_selection_a, bwr, random_reinit)
BMWR_a = UpdateRuleA(bw_selection_a, bmwr, random_reinit)


class MORankingCrowdingArchive(MORankingCrowdingAlgo):
    def __init__(self, ProblemHandler, UpdateRuleA):
        super().__init__(ProblemHandler, UpdateRuleA)

    def initialize(self, seed, hdf5_path):
        super().initialize(seed, hdf5_path)
        self.basepopulation = self.populationHandler.get_refined(self.population)

    def step(self):
        temp_X = self.updateRule.next_gen_a(
            self.problemHandler.problem, self.basepopulation, self.population
        )
        for mod in self.mods:
            temp_X = np.vstack(
                [
                    temp_X,
                    mod(
                        self.problemHandler.problem,
                        self.basepopulation,
                        self.populationHandler,
                    ),
                ]
            )

        temp_population = self.problemHandler.evaluate(temp_X)
        self.basepopulation = self.populationHandler.update(
            [self.basepopulation, temp_population], self.problemHandler
        )

        self.population = self.populationHandler.update(
            [self.population, self.populationHandler.get_refined(self.basepopulation)],
            self.problemHandler,
            self.problemHandler.problem.psize * 2,
        )


class MO_BMR_Archive(MORankingCrowdingArchive):
    def __init__(self, problemHandler):
        super().__init__(problemHandler, BMR_a)


class MO_BWR_Archive(MORankingCrowdingArchive):
    def __init__(self, problemHandler):
        super().__init__(problemHandler, BWR_a)


class MO_BMWR_Archive(MORankingCrowdingArchive):
    def __init__(self, problemHandler):
        super().__init__(problemHandler, BMWR_a)
