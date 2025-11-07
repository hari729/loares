import numpy as np
from opti.core.population import Population
from opti.core.result import Result
from opti.core.tracker import Tracker
from opti.core.initializer import random_initialize


def null_mutator(problem, new_gen):
    return new_gen

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
        self.result = Result(self.problem, self, metrics_function)
        self.tracker.record(self.problem, self.population, self.result)

    def advance(self):
        if self.tracker.remaining_evals() > 0:

            new_gen = self.basefunction(self.problem, self.population, self.selection)
            new_gen = self.mutation(self.problem, new_gen)

            for mod in self.pmods:
                new_gen = np.vstack([new_gen,mod(self.problem, self.population)])

            new_gen = self.problem.variable_modifier(new_gen)

            new_pop = Population(*self.tracker.evaluate(self.problem, new_gen))

            self.population = self.sorting_function(self.problem,
                                                    self.population + new_pop,
                                                    self.population.get_size())

            self.tracker.record(self.problem, self.population, self.result)
        else:
            print("Stopped")

    def get_population(self):
        return self.population

    def get_result(self):
        self.result.set_final_population(self.population)
        return self.result

    def get_info(self):
        dict = {
            "name" : str(self.__class__.__name__),
            "BaseFunction" : str(self.basefunction.__name__),
            "Sorting" : str(self.sorting_function.__name__),
            "Selection" : str(self.selection.__name__),
            "Mods" : [str(f.__name__) for f in self.pmods],
            "seed" : self.seed
        }
        return dict

class SAMP(Algorithm):
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
                 seed = 1 ,
                 n_sub_pops = 2):

        super().__init__(basefunction = basefunction,
                        mutation = mutation,
                        problem = problem,
                        selection_function = selection_function,
                        sorting_function = sorting_function,
                        pmods = pmods,
                        initializer = initializer,
                        metrics_function = metrics_function,
                        seed = seed)
        self.n_sub_pops = n_sub_pops
        self.sub_pops = self.population.split(self.n_sub_pops)

        for n in range(self.n_sub_pops):
            pop = self.sub_pops[n]
            self.sub_pops[n] = self.sorting_function(self.problem,
                                                     pop,
                                                     pop.get_size())

    def advance(self):
        if self.tracker.remaining_evals() > 0:
            for nc in range(self.n_sub_pops):
                pop = self.sub_pops[nc]

                new_gen = self.basefunction(self.problem, pop, self.selection)
                new_gen = self.mutation(self.problem, new_gen)

                for mod in self.pmods:
                    new_gen = np.vstack([new_gen,mod(self.problem, pop)])

                new_gen = self.problem.variable_modifier(new_gen)

                new_pop = Population(*self.tracker.evaluate(self.problem, new_gen))

                self.sub_pops[nc] = self.sorting_function(self.problem,
                                                        pop + new_pop,
                                                        pop.get_size())

            if self.n_sub_pops > 1:
                self.population = sum(self.sub_pops[1:], self.sub_pops[0])
            else:
                self.population = self.sub_pops[0]

            self.population = self.sorting_function(self.problem,
                                                    self.population,
                                                    self.population.get_size())

            better = self.tracker.record(self.problem, self.population, self.result)

            if isinstance(better, np.bool) and better and (self.n_sub_pops > 1):
                new_n = self.n_sub_pops - 1
            elif isinstance(better, np.bool) and not better:
                new_n = self.n_sub_pops + 1
            else:
                new_n = self.n_sub_pops

            if new_n != self.n_sub_pops:
                self.n_sub_pops = new_n
                self.sub_pops = self.population.split(self.n_sub_pops)

                for ns in range(self.n_sub_pops):
                    self.sub_pops[ns] = self.sorting_function(self.problem,
                                                            self.sub_pops[ns],
                                                            self.sub_pops[ns].get_size())

        else:
            print("Stopped")
