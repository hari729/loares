import numpy as np
from opti.core.algorithm import Algorithm
from opti.moo.sorting import ranking_crowding
from opti.moo.metrics import performance_metrics

class Ranking_Crowding_Algo(Algorithm):

    def __init__(self,
                 basefunction = None,
                 mutation = None,
                 problem = None,
                 selection_function = None,
                 pmods = [],
                 initializer = None,
                 seed = 1):

        super().__init__(
                 basefunction = basefunction,
                 mutation = mutation,
                 problem = problem,
                 selection_function = selection_function,
                 sorting_function = ranking_crowding,
                 pmods = pmods,
                 initializer = initializer,
                 metrics_function = performance_metrics,
                 seed = seed)


class MOArchive(Ranking_Crowding_Algo):
    def __init__(self, 
                 basefunction = None,
                 mutation = None,
                 problem = None,
                 selection_function = None,
                 pmods = [],
                 initializer = None,
                 seed = 1):

        super().__init__(
                 basefunction = basefunction,
                 mutation = mutation,
                 problem = problem,
                 selection_function = selection_function,
                 pmods = pmods,
                 initializer = initializer,
                 seed = seed)

        self.archive = self.population.get_pareto_population()


    def advance(self):
        if self.tracker.remaining_evals() > 0:
            pool = self.selection(self.population, self.archive)
            new_gen = self.basefunction(self.problem, self.population, pool)
            new_gen = self.mutation(self.problem, new_gen)

            for mod in self.pmods:
                new_gen = np.vstack([new_gen,mod(self.problem, self.population)])

            new_gen = self.problem.variable_modifier(new_gen)

            new_pop = self.tracker.create_population(self.problem, new_gen)

            self.population = self.sorting_function(self.problem,
                                                    self.population + new_pop,
                                                    self.population.get_size())

            self.archive = self.sorting_function(self.problem,
                                                 self.archive + self.population.get_pareto_population(),
                                                 self.population.get_size() * 2,
                                                 ndf = True)

            self.tracker.record(self.problem, self.archive, self.result)
        else:
            print("Stopped")

 
    def get_result(self):
        self.result.set_final_population(self.archive)
        return self.result

    def get_archive(self):
        return self.archive
