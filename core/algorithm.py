import numpy as np
from core.population import Population
from core.result import Result

class Algorithm():

    def __init__(self, 
                 basefunction=None, 
                 problem=None,
                 selection_function=None,
                 sorting_function=None,
                 pmods=[]):

        self.basefunction = basefunction
        self.problem = problem
        self.evals = 0

        original_evaluate = problem.evaluate
        def counted_evaluate(X):
            self.evals += X.shape[0]
            return original_evaluate(X)
 
        self.problem.evaluate = counted_evaluate

        self.population = Population(problem)
        self.selection = selection_function
        self.sorting_function = sorting_function
        self.pmode = pmods
        self.result = Result(self.problem.get_settings(), self.get_settings())

    def remaining_evals(self):
        return self.problem.max_evals - self.evals

    def advance(self):
        new_gen = self.basefunction(self.population, self.selection)
        for mod in self.pmods:
            new_gen = np.vstack([next_gen,mod(population,self.problem.bounds)])

        if next_gen.shape[0] > self.remaining_evals():
            next_gen = next_gen[:self.remaining_evals(),:]

        new_gen = self.problem.variable_modifer(next_gen)

        new_obj, new_constr = self.population.evaluate(next_gen)

        self.population.update(self.sorting_function(self.population,
                                                     new_gen,
                                                     new_obj,
                                                     new_constr,
                                                     self.population.get_size()))

    def get_population(self):
        return self.population

    def get_settings(self):
        dict = {
            "BaseFunction" : self.basefunction.__name__(),
            "Sorting" : self.sorting_function.__name__(),
            "Selection" : self.selection_function.__name__(),
            "Mods" : self.pmods
        }
        return dict
