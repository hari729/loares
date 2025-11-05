import numpy as np
from core.population import Population
from core.result import Result
from core.tracker import Tracker
from core.initializer import random_initialize
from core.sorting import ranking_crowding

class Algorithm():

    def __init__(self, 
                 basefunction=None,
                 mutation = None,
                 problem=None,
                 selection_function=None,
                 sorting_function=ranking_crowding,
                 pmods=[],
                 tracker = Tracker(),
                 intitializer = random_initialize):

        self.basefunction = basefunction
        self.mutation = mutation
        self.problem = problem
        self.evals = 0
        self.tracker = tracker
        solutions = initializer(self.problem)
        objectives, constraints = self.tracker.evaluate(self.problem, solutions)
        self.population = Population(solutions, objectives, constraints)
        self.selection = selection_function
        self.sorting_function = sorting_function
        self.pmods = pmods
        self.result = Result(self.problem.get_settings(), self.get_settings())

    def advance(self):
        if self.tracker.remaining_evals() > 0:

            new_gen = self.basefunction(self.problem, self.population, self.selection)
            new_gen = self.mutation.do(new_gen)

            for mod in self.pmods:
                new_gen = np.vstack([new_gen,mod(self.problem, self.population)])

            new_gen = self.problem.variable_modifier(new_gen)

            new_gen, new_obj, new_constr = self.tracker.evaluate(self.problem, new_gen)

            self.population = self.sorting_function(self.problem,
                                                    self.population.merge(
                                                                new_gen, new_obj, new_constr ),
                                                    self.population.get_size()))

            self.tracker.record(self.problem, self.population, self.result)

    def get_population(self):
        return self.population

    def get_settings(self):
        dict = {
            "BaseFunction" : self.basefunction.__name__,
            "Sorting" : self.sorting_function.__name__,
            "Selection" : self.selection_function.__name__,
            "Mods" : self.pmods
        }
        return dict
