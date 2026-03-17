import numpy as np
from loares.algorithms.moo.base import BMR, BWR, BMWR, MORankingCrowdingAlgo
from loares.metrics.moo import performance_metrics


class MORankingCrowdingSAMP(MORankingCrowdingAlgo):
    def __init__(self, ProblemHandler, UpdateRule):
        self.n = 2
        super().__init__(ProblemHandler, UpdateRule)

    def initialize(self, seed, hdf5_path):
        super().initialize(seed, hdf5_path)
        self.sub_populations = self.populationHandler.split(self.population, self.n)
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
            self.sub_populations = self.populationHandler.split(self.population, self.n)
            for j in range(self.n):
                self.sub_populations[j] = self.populationHandler.get_sorted(
                    self.sub_populations[j], self.problemHandler
                )


class MO_BMR_SAMP(MORankingCrowdingSAMP):
    def __init__(self, ProblemHandler):
        super().__init__(ProblemHandler, BMR)


class MO_BWR_SAMP(MORankingCrowdingSAMP):
    def __init__(self, ProblemHandler):
        super().__init__(ProblemHandler, BWR)


class MO_BMWR_SAMP(MORankingCrowdingSAMP):
    def __init__(self, ProblemHandler):
        super().__init__(ProblemHandler, BMWR)
