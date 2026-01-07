from typing import final
import numpy as np
from opti.core.population import PopulationRecorderHDF5
from opti.core.results import Result

class FlowHandler():
    def __init__(self, ProblemHandler, UpdateRule, PopulationHandler, Mods):
        self.problemHandler = ProblemHandler
        self.updateRule = UpdateRule
        self.populationHandler = PopulationHandler
        self.mods = Mods

    def step(self):
        temp_X = self.updateRule.next_gen(self.problemHandler.problem,
                                            self.population)
        for mod in self.mods:
            temp_X = np.vstack([temp_X,mod(self.problemHandler.problem,
                                           self.population, self.populationHandler)])
        temp_population = self.problemHandler.evaluate(temp_X)
        self.population = self.populationHandler.update([self.population, temp_population],
                                                            self.problemHandler)

    def record(self):
        if self.problemHandler.interval_status():
            # self.populationRecorder.record(self.populationHandler.get_refined(self.population),
            #                                 self.problemHandler.evals)
            self.result.record(self.populationHandler.get_refined(self.population),
                                self.problemHandler.evals)
            self.problemHandler.update_evals()

    def stop_record(self, final_dict):
        # self.populationRecorder.close()
        self.result.stop(final_dict)

    def initialize(self,seed):
        # self.populationRecorder = PopulationRecorderHDF5(filedir)
        self.result = Result(self.problemHandler.get_info(), self.get_info(), seed)
        self.population = self.populationHandler.initialize(self.problemHandler)
        self.record()

    def run(self, seed):
        self.initialize(seed)
        while self.problemHandler.remaining_evals() > 0:
            self.step()
            self.record()
        self.stop_record(self.populationHandler.get_refined_dict(self.population))
        return self.result

    def get_info(self):
        dictionary = {
            "name": str(self.__class__.__name__).replace("_", "-"),
        }
        return dictionary
