import numpy as np 

class Result():
    def __init__(self,
                 problem = None,
                 algorithm = None,
                 metrics_function = None):
        self.problem = problem
        self.algorithm = algorithm
        self.convergence_data = {}
        self.final_metrics = None
        self.final_population = None
        self.metrics_function = metrics_function

    def add_convergence_data(self, population, evals):
        metrics = self.metrics_function(self.problem, population)
        for key, value in metrics.items():
            self.convergence_data.setdefault(key, []).append(value)
        self.convergence_data.setdefault("evals", []).append(evals)

    def get_convergence_data(self):
        return self.convergence_data

    def set_final_population(self, population):
        self.final_population = population
        self.final_metrics = self.metrics_function(self.problem, population)
