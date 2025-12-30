import numpy as np 
import warnings
from opti.core.initializer import random_initiatlize
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

class PopulationX():
    def __init__(self, X = None, F = None, G = None, M = None):
        self.X = X
        self.F = F
        self.G = G
        self.M = M

class PopulationHandler():
    def __init__(self, ProblemHandler, initializer, sorting_function):
        if initializer is None:
            initializer = random_initiatlize
        self.population = PopulationX(
                ProblemHandler.evaluate(initializer(ProblemHandler.problem)))
        self.sort = sorting_function

    def raw_udpate(self, X, F, G, M = None):
        self.population.X = X
        self.population.F = F
        self.population.G = G
        self.population.M = M

    def get_size(self):
        return self.population.X.shape[0]

    def merge(self, X, F, G):
        temp_population = PopulationX(np.row_stack([self.population.X, X]),
                                      np.row_stack([self.population.F, F]),
                                      np.row_stack([self.population.G, G]))
        return temp_population

    def split(self, n_sub_pops):
        if n_sub_pops > self.get_size():
            warnings.warn("No. of sub opulations exceed population size, value is automatically reduced.", Warning)
            n_sub_pops = self.get_size()
        idx = np.arange(self.get_size())
        np.random.shuffle(idx)
        parts = np.array_split(idx, n_sub_pops)
        return [PopulationX(self.population.X[i], self.population.F[i], self.population.G[i]) for i in parts]

    def get_dict(self):
        combined = np.hstack([self.population.X, self.population.F, self.population.G])
        col_labels = (
            [f"x{i+1}" for i in range(self.population.X.shape[1])] +
            [f"f{j+1}" for j in range(self.population.F.shape[1])] +
            [f"g{k+1}" for k in range(self.population.G.shape[1])]
        )
        return {name: combined[:, idx] for idx, name in enumerate(col_labels)}

    def update(self, X, F, G):
        self.raw_update(self.sort(self.merge(X, F, G)))

    def get(self):
        return self.population

class PopulationRecorderHDF5():
    def __init__(self, filename):
        self.file = h5py.File(filename, "w")
        self.iter_group = self.file.create_group("function_evals")
        self.recording_interval = int(self.max_evals * 0.05)

    def record(self, population, evals):
        if (((self.evals//self.recording_interval) > (self.prev_evals//self.recording_interval)) 
            | (self.prev_evals == 0)):
            grp = self.iter_group.create_group(f"{evals:06d}")
            grp.create_dataset("X", data=population.X)
            grp.create_dataset("F", data=population.F)
            grp.create_dataset("G", data=population.G)

    def close(self):
        self.file.close()
