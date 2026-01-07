import numpy as np

from opti.core.population import Population

def no_modifier(X):
    return X 

class Problem():

    def __init__(self,
                 function,
                 name = None,
                 n_vars = 1,
                 n_obj = 1,
                 n_constr = 0,
                 psize = 10,
                 max_evals = 100,
                 bounds = None,
                 minmax = None,
                 variable_modifier = None):
        self.name = name
        self.function = function
        self.n_vars = n_vars
        self.n_obj = n_obj
        self.n_constr = n_constr
        self.psize = psize
        self.max_evals = max_evals
        self.bounds = bounds
        self.minmax = minmax

        if variable_modifier is None:
            self.variable_modifier = no_modifier
        else:
            self.variable_modifier = variable_modifier

    def evaluate(self, solutions):
        return self.function(solutions)

    def get_true_front(self):
        return None

    def get_info(self):
        if self.name is not None:
            name = self.name
        else:
            name = str(self.__class__.__name__).replace("_", "-")
        dict = {
            "name": name,
            "n_obj" : self.n_obj,
            "n_vars" : self.n_vars,
            "bounds" : self.bounds.tolist(),
            "psize" : self.psize,
            "max_evals" : self.max_evals,
            "minmax" : self.minmax.tolist(),
            "variable_modifier" : str(self.variable_modifier.__name__)
        }
        return dict

    def objective_correction(self, population):
        population.objectives *= self.minmax
        return population


class ProblemHandler():
    def __init__(self, problem):
        self.problem = problem
        self.max_evals = problem.max_evals
        self.evals = 0
        self.prev_evals = 0
        self.recording_interval = int(self.max_evals * 0.05)

    def remaining_evals(self):
        return self.max_evals - self.evals

    def get_evals(self):
        return self.evals

    def evaluate(self, solutions):
        if self.remaining_evals() < solutions.shape[0]:
            solutions = solutions[:self.remaining_evals(),:]
        self.evals += solutions.shape[0]
        solutions = self.problem.variable_modifier(solutions)
        objectives, constraints =  self.problem.evaluate(solutions)
        return Population(solutions, objectives, constraints)

    def interval_status(self):
        if ((self.evals//self.recording_interval) > (self.prev_evals//self.recording_interval)) | (self.prev_evals == 0):
            return 1
        else:
            return 0

    def update_evals(self):
        self.prev_evals = self.get_evals()

