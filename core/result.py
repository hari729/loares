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
        self.population = None
        self.metrics_function = metrics_function

    def add_convergence_data(self, population, evals):
        metrics = self.metrics_function(self.problem, population)
        for key, value in metrics.items():
            self.convergence_data.setdefault(key, []).append(value)
        self.convergence_data.setdefault("evals", []).append(evals)

    def get_convergence_data(self):
        return self.convergence_data

    def set_final_population(self, population):
        self.population = population
        self.final_metrics = self.metrics_function(self.problem, population)

    def show_results(self):
        print(f"Problem settings: {self.problem.get_info()}")
        print(f"Algorithm settings: {self.algorithm.get_info()}")
        print(f"Final metrics: {self.final_metrics}")
