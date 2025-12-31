import numpy as np 
import warnings
from opti.core.initializer import random_initialize
import h5py

class Population():

    def __init__(self, X, F, G, M = None):
        self.solutions = X 
        self.objectives = F
        self.constraints = G
        self.metadata = M

    def update(self, X, F, G, M):
        self.solutions = X
        self.objectives = F
        self.constraints = G
        self.metadata = M

    def get_size(self):
        return self.solutions.shape[0]

    def merge(self, new_gen, new_obj, new_constr):
        temp_population = Population(np.row_stack([self.solutions, new_gen]),
                                     np.row_stack([self.objectives, new_obj]),
                                     np.row_stack([self.constraints, new_constr]))
        return temp_population

    def __add__(self, other):
        if isinstance(other, Population):
            X = np.row_stack([self.solutions, other.solutions])
            F = np.row_stack([self.objectives, other.objectives])
            G = np.row_stack([self.constraints, other.constraints])
            return Population(X, F, G)
        else:
            raise TypeError("Can only add another instance of Population class")

    def split(self, n_sub_pops):
        if n_sub_pops > self.get_size():
            warnings.warn("No. of sub opulations exceed population size, value is automatically reduced.", Warning)
            n_sub_pops = self.get_size()
        idx = np.arange(self.get_size())
        np.random.shuffle(idx)
        parts = np.array_split(idx, n_sub_pops)
        return [Population(self.solutions[i], self.objectives[i], self.constraints[i]) for i in parts]

    def get_dict(self):
        combined = np.hstack([self.solutions, self.objectives, self.constraints])
        col_labels = (
            [f"x{i+1}" for i in range(self.solutions.shape[1])] +
            [f"f{j+1}" for j in range(self.objectives.shape[1])] +
            [f"g{k+1}" for k in range(self.constraints.shape[1])]
        )
        return {name: combined[:, idx] for idx, name in enumerate(col_labels)}


class PopulationHandler():
    def __init__(self, sorting_function, initializer=None):
        if initializer is None:
            initializer = random_initialize
        self.initializer = initializer
        self.sort = sorting_function

    def initialize(self, ProblemHandler):
        X = self.initializer(ProblemHandler.problem)
        X, F, G = ProblemHandler.evaluate(X)
        self.population = Population(X, F, G)
        self.raw_update(*self.sort(ProblemHandler.problem,
                                    self.population, self.population.solutions.shape[0]))

    def raw_update(self, X, F, G, M = None):
        self.population.solutions = X
        self.population.objectives = F
        self.population.constraints = G
        self.population.metadata = M

    def raw_replace(self, population):
        self.population = population

    def get_size(self):
        return self.population.solutions.shape[0]

    def merge(self, X, F, G):
        temp_population = Population(np.row_stack([self.population.solutions, X]),
                                      np.row_stack([self.population.objectives, F]),
                                      np.row_stack([self.population.constraints, G]))
        return temp_population

    def split(self, n_sub_pops):
        if n_sub_pops > self.get_size():
            warnings.warn("No. of sub opulations exceed population size, value is automatically reduced.", Warning)
            n_sub_pops = self.get_size()
        idx = np.arange(self.get_size())
        np.random.shuffle(idx)
        parts = np.array_split(idx, n_sub_pops)
        return [Population(self.population.solutions[i], self.population.objectives[i], self.population.constraints[i]) for i in parts]

    def get_dict(self):
        combined = np.hstack([self.population.solutions, self.population.objectives, self.population.constraints])
        col_labels = (
            [f"x{i+1}" for i in range(self.population.solutions.shape[1])] +
            [f"f{j+1}" for j in range(self.population.objectives.shape[1])] +
            [f"g{k+1}" for k in range(self.population.constraints.shape[1])]
        )
        return {name: combined[:, idx] for idx, name in enumerate(col_labels)}

    def self_sort(self, ProblemHandler):
        self.raw_update(*self.sort(ProblemHandler.problem, self.population, self.get_size()))

    def update(self, X, F, G, ProblemHandler):
        nX, nF, nG, nM = self.sort(ProblemHandler.problem,self.merge(X, F, G),
                                    self.get_size())
        self.raw_update(nX, nG, nF, nM)

    def get(self):
        return self.population

class PopulationRecorderHDF5():
    def __init__(self, filename):
        self.file = h5py.File(filename, "w")
        self.iter_group = self.file.create_group("function_evals")

    def record(self, population, evals):
        grp = self.iter_group.create_group(f"{evals:06d}")
        grp.create_dataset("X", data=population.solutions)
        grp.create_dataset("F", data=population.objectives)
        grp.create_dataset("G", data=population.constraints)
        grp.create_dataset("M", data=population.metadata)
    def close(self):
        self.file.close()


