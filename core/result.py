import numpy as np 

class Result():
    def __init__(self,
                 problem = None,
                 algorithm_settings = None,
                 metrics_function = None):
        self.problem = problem
        self.algorithm = algorithm_settings
        self.convergence_data = []
        self.final_metrics = None
        self.final_population = None
        self.metrics_function = metrics_function

    def add_convergence_data(self, population, evals):
        self.convergence_data.append(np.append(self.metrics_function(self.problem, population), evals))

    def set_final_population(self, population):
        self.final_population = population
        self.final_metrics = self.metrics_function(self.problem, population)
