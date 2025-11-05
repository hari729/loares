import numpy as np

class Problem():

    def __init__(self,
                 function = None,
                 n_vars = 1,
                 n_obj = 1,
                 psize = 10,
                 max_evals = 100,
                 bounds = None,
                 minmax = np.full(n_obj, 1),
                 variable_modifier = None):

        self.function = function
        self.n_vars = n_vars
        self.n_obj = n_obj
        self.psize = psize
        self.max_evals = max_evals
        self.bounds = bounds
        self.minmax = minmax

        if variable_modifier is None:
            def no_modifier(X):
                return X 
            self.variable_modifier = no_modifier
        else:
            self.variable_modifier = variable_modifier

    def evaluate(self, solutions):
        return self.function(solutions)

    def get_true_front(self):
        return None

    def get_settings(self):
        dict = {
            "name" : self.function.__name__(),
            "n_obj" : self.n_obj,
            "n_vars" : self.n_vars,
            "bounds" : self.bounds,
            "psize" : self.psize,
            "max_evals" : self.max_evals,
            "minmax" : self.minmax,
            "variable_modifier" : self.variable_modifier.__name__()
        }
        return dict

