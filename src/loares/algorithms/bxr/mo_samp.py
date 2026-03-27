import numpy as np
from loares.core.algorithm import Algorithm
from loares.core.flow import FlowHandler
from loares.operators.sorting import ranking_crowding
from loares.operators.bxr import *
from loares.operators.selection import random_bw_selection
from loares.operators.mutation import random_reinit
from loares.operators.mods import local_search
from loares.metrics.moo import performance_metrics

class MOSAMP(FlowHandler):
    def __init__(self, ProblemHandler, UpdateRule, PopulationHandler, Mods):
        self.n = 2
        super().__init__(ProblemHandler, UpdateRule, PopulationHandler, Mods)

    def initialize(self, seed, hdf5_path):
        super().initialize(seed, hdf5_path)
        self.sub_populations = self.populationHandler.random_split(self.population, self.n)
        self.indicator = performance_metrics(
            self.problemHandler.problem,
            self.populationHandler.get_refined(self.population),
        )["HV"]
        for k in range(self.n):
            self.sub_populations[k] = self.populationHandler.get_sorted(
                self.sub_populations[k], self.problemHandler
            )

    def step(self):
        for i in range(self.n):
            temp_X = self.updateRule.next_gen(
                self.problemHandler.problem, self.sub_populations[i]
            )
            for mod in self.mods:
                temp_X = np.vstack(
                    [
                        temp_X,
                        mod(
                            self.problemHandler.problem,
                            self.sub_populations[i],
                            self.populationHandler,
                        ),
                    ]
                )
            temp_population = self.problemHandler.evaluate(temp_X)
            self.sub_populations[i] = self.populationHandler.update(
                [self.sub_populations[i], temp_population], self.problemHandler,
                limit=self.sub_populations[i].solutions.shape[0]
            )
        if self.n > 1:
            self.population = self.populationHandler.merge(self.sub_populations)
        else:
            self.population = self.sub_populations[0]

        self.population = self.populationHandler.get_sorted(
            self.population, self.problemHandler
        )
        new_indicator = performance_metrics(
            self.problemHandler.problem,
            self.populationHandler.get_refined(self.population),
        )["HV"]

        if self.problemHandler.interval_status():
            if new_indicator > self.indicator:
                new_n = min(
                    self.n + 1, max(2, int(0.1 * self.problemHandler.problem.psize))
                )
            elif self.n > 1:
                new_n = self.n - 1
            else:
                new_n = self.n

            self.indicator = new_indicator

            if new_n != self.n:
                self.n = new_n
                self.sub_populations = self.populationHandler.random_split(self.population, self.n)
                for j in range(self.n):
                    self.sub_populations[j] = self.populationHandler.get_sorted(
                        self.sub_populations[j], self.problemHandler
                    )

MO_BMR_S = Algorithm("MO-BMR-SAMP", bmr, random_bw_selection, random_reinit, ranking_crowding, [local_search],
                     flowhandler=MOSAMP)
MO_BWR_S = Algorithm("MO-BWR-SAMP", bwr, random_bw_selection, random_reinit, ranking_crowding, [local_search],
                     flowhandler=MOSAMP)
MO_BMWR_S = Algorithm("MO-BMWR-SAMP", bmwr, random_bw_selection, random_reinit, ranking_crowding, [local_search],
                     flowhandler=MOSAMP)


