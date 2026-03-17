import numpy as np

from loares.core.population import Population


def no_modifier(X):
    return X


def dummy_function(X):
    return X


class Problem:
    def __init__(
        self,
        function=dummy_function,
        name=None,
        n_vars=1,
        n_obj=1,
        n_constr=0,
        psize=10,
        max_evals=100,
        bounds=None,
        minmax=None,
        variable_modifier=None,
    ):
        self.name = name
        self.function = function
        self.n_vars = n_vars
        self.n_obj = n_obj
        self.n_constr = n_constr
        self.psize = psize
        self.max_evals = max_evals
        self.bounds = bounds
        tmm = np.ones((1,self.n_obj))
        if minmax is not None:
            self.minmax = np.array([[-1 if m == "max" else 1 for m in minmax]])
        else:
            self.minmax = tmm

        if variable_modifier is None:
            self.variable_modifier = no_modifier
        else:
            self.variable_modifier = variable_modifier

    def evaluate(self, solutions):
        return self.function(solutions)

    def get_true_front(self, pts=500):
        return None

    def get_info(self):
        if self.name is not None:
            name = self.name
        else:
            name = str(self.__class__.__name__).replace("_", "-")
        dictionary = {
            "name": name,
            "n_obj": self.n_obj,
            "n_vars": self.n_vars,
            "n_constr": self.n_constr,
            "bounds": str(self.bounds.tolist()),
            "psize": self.psize,
            "max_evals": self.max_evals,
            "minmax": str(self.minmax.tolist()),
            "variable_modifier": str(self.variable_modifier.__name__),
        }
        return dictionary

class ProblemHandler:
    def __init__(self, problem):
        self.problem = problem
        self.max_evals = problem.max_evals
        self.evals = 0
        self.prev_evals = 0
        if self.problem.n_obj > 1:
            self.recording_interval = max(int(self.max_evals * 0.05), 1)
        else:
            self.recording_interval = max(int(self.max_evals * 0.005), 1)

    def remaining_evals(self):
        return self.max_evals - self.evals

    def get_evals(self):
        return self.evals

    def evaluate(self, solutions):
        if self.remaining_evals() < solutions.shape[0]:
            solutions = solutions[: self.remaining_evals(), :]
        self.evals += solutions.shape[0]
        solutions = self.problem.variable_modifier(solutions)
        objectives, constraints = self.problem.evaluate(solutions)
        return Population(solutions, objectives*self.problem.minmax, constraints)

    def interval_status(self):
        if (
            (self.evals // self.recording_interval)
            > (self.prev_evals // self.recording_interval)
        ) | (self.prev_evals == 0):
            return 1
        else:
            return 0

    def update_evals(self):
        self.prev_evals = self.get_evals()

    def get_info(self):
        return self.problem.get_info()
