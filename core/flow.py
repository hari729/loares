import numpy as np
from opti.core.population import PopulationRecorderHDF5
from opti.analysis.utils import dict_to_csv

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
            self.populationRecorder.record(self.populationHandler.get_refined(self.population),
                                            self.problemHandler.evals)
            self.problemHandler.update_evals()

    def stop_record(self):
        self.populationRecorder.close()

    def initialize(self, filedir):
        self.populationRecorder = PopulationRecorderHDF5(filedir)
        self.population = self.populationHandler.initialize(self.problemHandler)
        self.record()

    def run(self, filedir):
        self.initialize(filedir)
        while self.problemHandler.remaining_evals() > 0:
            self.step()
            self.record()
        self.stop_record()
        return self.populationHandler.get_refined_dict(self.population)

    def get_info(self):
        dict = {
            "name": str(self.__class__.__name__).replace("_", "-"),
        }
        return dict
