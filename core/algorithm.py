import numpy as np
from core.population import Population
from core.result import Result
from core.tracker import Tracker
from core.initializer import random_initialize

class Algorithm():

    def __init__(self, 
                 basefunction=None,
                 mutation = None,
                 problem=None,
                 selection_function=None,
                 sorting_function=None,
                 pmods=[],
                 # tracker = Tracker(),
                 initializer = random_initialize,
                 metrics_function = None,
                 seed = 1 ):

        self.seed = seed
        np.random.seed(self.seed)
        self.basefunction = basefunction
        if mutation is None:
            def null_mutator(problem, new_gen):
                return new_gen
            self.mutation = null_mutator
        else:
            self.mutation = mutation
        self.problem = problem
        self.evals = 0
        self.tracker = Tracker(self.problem)
        solutions = initializer(self.problem)
        solutions, objectives, constraints = self.tracker.evaluate(self.problem, solutions)
        self.population = Population(solutions, objectives, constraints)
        self.selection = selection_function
        self.sorting_function = sorting_function
        self.population = self.sorting_function(self.problem, self.population, self.population.get_size())
        self.pmods = pmods
        self.result = Result(self.problem, self.get_settings(), metrics_function)

    def advance(self):
        if self.tracker.remaining_evals() > 0:

            new_gen = self.basefunction(self.problem, self.population, self.selection)
            new_gen = self.mutation(self.problem, new_gen)

            for mod in self.pmods:
                new_gen = np.vstack([new_gen,mod(self.problem, self.population)])

            new_gen = self.problem.variable_modifier(new_gen)

            new_gen, new_obj, new_constr = self.tracker.evaluate(self.problem, new_gen)

            self.population = self.sorting_function(self.problem,
                                                    self.population.merge(
                                                                new_gen, new_obj, new_constr ),
                                                    self.population.get_size())

            self.tracker.record(self.problem, self.population, self.result)

    def get_population(self):
        return self.population

    def get_result(self):
        self.result.set_final_population(self.population)
        return self.result

    def get_settings(self):
        dict = {
            "BaseFunction" : self.basefunction.__name__,
            "Sorting" : self.sorting_function.__name__,
            "Selection" : self.selection.__name__,
            "Mods" : self.pmods
        }
        return dict
