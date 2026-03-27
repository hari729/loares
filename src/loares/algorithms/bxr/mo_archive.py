import numpy as np
from loares.core.flow import FlowHandler
from loares.core.algorithm import Algorithm
from loares.operators.sorting import ranking_crowding
from loares.operators.bxr import *
from loares.operators.selection import archive_bw_selection
from loares.operators.mutation import random_reinit
from loares.operators.mods import local_search

class MOArchiveFlow(FlowHandler):

    def initialize(self, seed, hdf5_path):
        super().initialize(seed, hdf5_path)
        self.basepopulation = self.populationHandler.get_refined(self.population)

    def step(self):
        new_gen = self.updateRule.base_function(
            self.problemHandler.problem, self.basepopulation,
            self.updateRule.selection(self.basepopulation, self.population)
        )
        temp_X = self.updateRule.mutation(self.problemHandler.problem, new_gen)
        # temp_X = self.updateRule.next_gen_a(
        #     self.problemHandler.problem, self.basepopulation, self.population
        # )
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

MO_BMR_A = Algorithm("MO-BMR-Archive", bmr, archive_bw_selection, random_reinit, ranking_crowding, [local_search],
                     flowhandler=MOArchiveFlow)
MO_BWR_A = Algorithm("MO-BWR-Archive", bwr, archive_bw_selection, random_reinit, ranking_crowding, [local_search],
                     flowhandler=MOArchiveFlow)
MO_BMWR_A = Algorithm("MO-BMWR-Archive", bmwr, archive_bw_selection, random_reinit, ranking_crowding, [local_search],
                      flowhandler=MOArchiveFlow)

