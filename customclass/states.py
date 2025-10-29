import numpy as np

class PopulationState():
    def __init__(self,population_array, function, max_evals, sorting_function, selection_pool):

        self.population = population_array
        self.objective_values, self.constraint_values = function(population_array)
        self.metadata = None
        self.psize = population_array.shape[0]
        self.init_updated = False
        self.evals = population_array.shape[0]
        self.max_evals = max_evals
        self.function = function
        self.sorting_function = sorting_function
        self.selection_pool = selection_pool

        self.best_idx = None
        self.worst_idx = None
        self.best = None
        self.worst = None

        self.new_solutions = []

        self.function_evals = None
        self.normalized_obj = None

        self.metrics = None

        self.temp_population = None
        self.temp_objectives = None
        self.temp_constraints = None
        self.temp_metadata = None

        self.pareto_pop = None
        self.pareto_objectives = None
        self.pareto_constraints = None
        self.pareto_metadata = None

        self.convergence_data = []

    def add_convergance_data(self, metrics):
        self.convergence_data.append(np.append(metrics, self.evals))

    def get_convergence_data(self):
        return np.vstack(self.convergence_data)

    def add_solutions(self, new_p):
        if new_p is not None:
            self.new_solutions.append(new_p)

    def evaluate(self):
        pending_solutions = np.vstack(self.new_solutions)
        if self.evals + pending_solutions.shape[0] > self.max_evals:
            remaining_evals = self.max_evals - self.evals
            pending_solutions = pending_solutions[:remaining_evals,:]
        
        self.temp_population = pending_solutions
        self.temp_objectives, self.temp_constraints = self.function(pending_solutions)
        self.evals += pending_solutions.shape[0]
        self.new_solutions.clear()

    def regular_update_generation(self):

        (self.population,
        self.objective_values, self.constraint_values,
        self.metadata) = self.sorting_function(np.row_stack([self.population,self.temp_population]),
                                        np.row_stack([self.objective_values,self.temp_objectives]),
                                        np.row_stack([self.constraint_values,self.temp_constraints]),
                                        self.psize,ndf=False)
        
        ndfront = (self.metadata[:,0] == 0)

        if self.selection_pool ==  "archive":
            (self.pareto_pop,
            self.pareto_objectives, self.pareto_constraints,
            self.pareto_metadata)  = self.sorting_function(np.row_stack([self.pareto_pop,self.population[ndfront]]),
                                                        np.row_stack([self.pareto_objectives,self.objective_values[ndfront]]),
                                                        np.row_stack([self.pareto_constraints, self.constraint_values[ndfront]]),
                                                        self.psize*2,ndf=True)
        else:
            (self.pareto_pop,
            self.pareto_objectives,
            self.pareto_constraints,
            self.pareto_metadata) = (self.population[ndfront],
                                        self.objective_values[ndfront],
                                        self.constraint_values[ndfront],
                                        self.metadata[ndfront])

        # if self.pareto_objectives.size == 0:
        #     self.metadata = np.zeros(self.metadata.shape)
        #     (self.pareto_pop,
        #     self.pareto_objectives,
        #     self.pareto_constraints,
        #     self.pareto_metadata) = (self.population,
        #                                 self.objective_values,
        #                                 self.constraint_values,
        #                                 self.metadata)
    
    def first_update_generation(self):

        (self.population,
        self.objective_values,
        self.constraint_values,
        self.metadata) = self.sorting_function(self.population,
                                            self.objective_values,
                                            self.constraint_values,
                                            self.psize,ndf=False)

        ndfront = (self.metadata[:,0] == 0)

        (self.pareto_pop,
        self.pareto_objectives,
        self.pareto_constraints,
        self.pareto_metadata) = (self.population[ndfront],
                                    self.objective_values[ndfront],
                                    self.constraint_values[ndfront],
                                    self.metadata[ndfront])

        # if self.pareto_objectives.size == 0:
        #     self.metadata = np.zeros(self.metadata.shape)
        #     (self.pareto_pop,
        #     self.pareto_objectives,
        #     self.pareto_constraints,
        #     self.pareto_metadata) = (self.population,
        #                                 self.objective_values,
        #                                 self.constraint_values,
        #                                 self.metadata)

    def update_generation(self):
        if ~self.init_updated:
            self.first_update_generation()
            self.update_generation = self.regular_update_generation
            self.init_updated = True
        else:
            self.regular_update_generation(sorting_function)


class population():
    def __init__(self,population_array):

        self.population = population_array
        self.objective_values = None
        self.constraint_values = None
        self.metadata = None
        self.best_idx = None
        self.worst_idx = None
        self.best = None
        self.worst = None
        self.function_evals = None
        self.normalized_obj = None

        self.metrics = None

        self.convergence_data = None
