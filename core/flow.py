import numpy as np
from opti.core.population import PopulationRecorderHDF5

class FlowHandler():
    def __init__(self, ProblemHandler, UpdateRule, PopulationHandler, Mods):
        self.ProblemHandler = ProblemHandler
        self.UpdateRule = UpdateRule
        self.PopulationHandler = PopulationHandler
        self.Mods = Mods

    def step(self):
        temp_X = self.UpdateRule.next_gen(self.ProblemHandler.problem,
                                self.PopulationHandler.population) 
        for mod in self.Mods:
            temp_X = np.vstack([temp_X,mod(self.ProblemHandler.problem
                                           , self.PopulationHandler)])
        temp_X, temp_F, temp_G = self.ProblemHandler.evaluate(temp_X)
        self.PopulationHandler.update(temp_X, temp_F, temp_G, self.ProblemHandler)

    def record(self):
        if self.ProblemHandler.interval_status():
            self.PopulationRecorder.record(self.PopulationHandler.get(),
                                            self.ProblemHandler.evals)

    def stop_record(self):
        self.PopulationRecorder.close()

    def run(self, filedir):
        self.PopulationRecorder = PopulationRecorderHDF5(filedir)
        self.PopulationHandler.initialize(self.ProblemHandler)
        self.record()
        while self.ProblemHandler.remaining_evals() > 0:
            self.step()
            self.record()
        self.stop_record()

    def get_info(self):
        dict = {
            "name": str(self.__class__.__name__).replace("_", "-"),
        }
        return dict
