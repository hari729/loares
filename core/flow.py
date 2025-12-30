

class FlowHandler():
    def __init__(self, ProblemHandler, UpdateRule, PopulationHandler, PopulationRecorder):
        self.ProblemHandler = ProblemHandler
        self.UpdateRule = UpdateRule
        self.PopulationHandler = PopulationHandler
        self.PopulationRecorder = PopulationRecorder

    def run(self):
        while self.ProblemHandler.remaining_evals > 0:
            temp_X = self.UpdateRule(self.ProblemHandler.problem,
                                    self.PopulationHandler.population)
            temp_F, temp_G = self.ProblemHandler.evaluate(temp_X)
            self.PopulationHandler.update(temp_X, temp_G, temp_G)
            self.PopulationRecorder.recodrd(PopulationHandler.get(),
                                            ProblemHandler.evals)

