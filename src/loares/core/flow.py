import numpy as np
from loares.core.results import ResultProcessor


class FlowHandler:
    def __init__(self, ProblemHandler, UpdateRule, PopulationHandler, Mods):
        self.problemHandler = ProblemHandler
        self.updateRule = UpdateRule
        self.populationHandler = PopulationHandler
        self.mods = Mods

    def step(self):
        temp_X = self.updateRule.next_gen(self.problemHandler.problem, self.population)
        for mod in self.mods:
            temp_X = np.vstack(
                [
                    temp_X,
                    mod(
                        self.problemHandler.problem,
                        self.population,
                        self.populationHandler,
                    ),
                ]
            )
        temp_population = self.problemHandler.evaluate(temp_X)
        self.population = self.populationHandler.update(
            [self.population, temp_population], self.problemHandler
        )

    def record(self):
        if self.problemHandler.interval_status():
            ResultProcessor.write_snapshot(
                self._h5,
                self.populationHandler.get_refined(self.population),
                self.problemHandler.evals,
            )
            self.problemHandler.update_evals()

    def stop_record(self, final_dict):
        ResultProcessor.write_final(self._h5, final_dict)
        ResultProcessor.close(self._h5)
        self._h5 = None

    def initialize(self, seed, hdf5_path):
        np.random.seed(seed)
        self._h5 = ResultProcessor.open(
            hdf5_path, self.problemHandler.get_info(), self.get_info(), seed
        )
        self.population = self.populationHandler.initialize(self.problemHandler, seed)
        self.record()

    def run(self, seed, hdf5_path):
        self.initialize(seed, hdf5_path)
        while self.problemHandler.remaining_evals() > 0:
            self.step()
            self.record()
        self.stop_record(self.populationHandler.get_refined_dict(self.population))

    def get_info(self):
        dictionary = {
            "name": str(self.__class__.__name__).replace("_", "-"),
            "mods": [mod.__name__ for mod in self.mods],
        }
        return dictionary
