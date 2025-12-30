import numpy as np

from opti.core.population import Population
from opti.moo.population import MoPopulation

def no_modifier(X):
    return X 

class Problem():

    def __init__(self,
                 function = None,
                 n_vars = 1,
                 n_obj = 1,
                 n_constr = 0,
                 psize = 10,
                 max_evals = 100,
                 bounds = None,
                 minmax = None,
                 variable_modifier = None):

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

    def get_minmax_applied_pop(self, solutions):
        population = self.create_population(solutions)
        population.objectives *= self.minmax
        return population

    def get_true_front(self):
        return None

    def get_info(self):
        dict = {
            "name": str(self.__class__.__name__).replace("_", "-"),
            "n_obj" : self.n_obj,
            "n_vars" : self.n_vars,
            "bounds" : self.bounds.tolist(),
            "psize" : self.psize,
            "max_evals" : self.max_evals,
            "minmax" : self.minmax.tolist(),
            "variable_modifier" : str(self.variable_modifier.__name__)
        }
        return dict

    def create_population(self, solutions):
        X = solutions
        F, G  = self.evaluate(solutions)
        if self.n_obj > 2:
            population = MoPopulation(X, F, G)
        else:
            population = Population(X, F, G)
        return population

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

    def evaluate(self, solutions):
        if self.remaining_evals() < solutions.shape[0]:
            solutions = solutions[:self.remaining_evals(),:]
        self.evals += solutions.shape[0]
        objectives, constraints =  self.problem.evaluate(solutions)
        return solutions, objectives, constraints

    def interval_status(self):
        if (((self.evals//self.recording_interval) > (self.prev_evals//self.recording_interval)) 
            | (self.prev_evals == 0)):
            return 1
        else:
            return 0

