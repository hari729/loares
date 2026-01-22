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

    def get_size(self):
        return self.solutions.shape[0]

    def __add__(self, other):
        if isinstance(other, Population):
            X = np.vstack([self.solutions, other.solutions])
            F = np.vstack([self.objectives, other.objectives])
            G = np.vstack([self.constraints, other.constraints])
            return Population(X, F, G)
        else:
            raise TypeError("Can only add another instance of Population class")

class PopulationHandler():
    def __init__(self, sorting_function, initializer=None):
        if initializer is None:
            initializer = random_initialize
        self.initializer = initializer
        self.sort = sorting_function

    def initialize(self, ProblemHandler, seed):
        self.seed = seed
        X = self.initializer(ProblemHandler.problem)
        population = ProblemHandler.evaluate(X)
        return Population(*self.sort(ProblemHandler.problem,
                            population, population.solutions.shape[0], self.seed))

    def raw_update(self,population, X, F, G, M):
        population.solutions = X
        population.objectives = F
        population.constraints = G
        population.metadata = M

    def get_size(self, population):
        return population.solutions.shape[0]

    def merge(self, population_list):
        temp_population = sum(population_list[1:], population_list[0])
        return temp_population

    def split(self,population, n_sub_pops):
        if n_sub_pops > self.get_size(population):
            warnings.warn("No. of sub populations exceed population size, value is automatically reduced.", Warning)
            n_sub_pops = self.get_size(population)
        idx = np.arange(self.get_size(population))
        np.random.shuffle(idx)
        parts = np.array_split(idx, n_sub_pops)
        return [Population(population.solutions[i], population.objectives[i], population.constraints[i]) for i in parts]

    def get_dict(self, population):
        combined = np.hstack([population.solutions, population.objectives, population.constraints])
        col_labels = (
            [f"x{i+1}" for i in range(population.solutions.shape[1])] +
            [f"f{j+1}" for j in range(population.objectives.shape[1])] +
            [f"g{k+1}" for k in range(population.constraints.shape[1])]
        )
        return {name: combined[:, idx] for idx, name in enumerate(col_labels)}

    def get_sorted(self,population, ProblemHandler, limit=None):
        if limit is None:
            limit = ProblemHandler.problem.psize
        return Population(*self.sort(ProblemHandler.problem, population, limit, self.seed))

    def update(self, population_list, ProblemHandler, limit=None):
        temp_population = self.merge(population_list)
        return self.get_sorted(temp_population, ProblemHandler, limit)

    def get_refined(self, population):
        return population

    def get_refined_dict(self, population):
        return population

class PopulationRecorderHDF5():
    def __init__(self, filename):
        self.file = h5py.File(filename, "w")
        self.iter_group = self.file.create_group("function_evals")

    def record(self, population, evals):
        grp = self.iter_group.create_group(f"{evals:06d}")
        grp.create_dataset("X", data=population.solutions)
        grp.create_dataset("F", data=population.objectives)
        grp.create_dataset("G", data=population.constraints)
        # grp.create_dataset("M", data=population.metadata)
    def close(self):
        self.file.close()

class PopulationHDF5Reader:
    def __init__(self, problem, perofrmance_metrics):
        self.perofrmance_metrics = perofrmance_metrics
        self.problem = problem

    def list_keys(self, filepath):
        file = h5py.File(filepath, "r")
        keys = sorted(file.keys())
        file.close()
        return keys

    def get_metrics_history(self, filepath, group):
        file = h5py.File(filepath, "r")
        convergence_data = {}
        for it in file[group]:
            X = file[group][it]['X'][:]
            F = file[group][it]['F'][:]
            G = file[group][it]['G'][:]
            M = file[group][it]['M'][:]

            population = Population(X, F, G, M)

            metrics = self.perofrmance_metrics(self.problem, population)

            for key, value in metrics.items():
                convergence_data.setdefault(key, []).append(value)
            convergence_data.setdefault("evals", []).append(it)

        file.close()
        return convergence_data

    def get_final_population(self, filepath, group):
        file = h5py.File(filepath, "r")
        it = f"{self.problem.max_evals:06d}"
        X = file[group][it]['X'][:]
        F = file[group][it]['F'][:]
        G = file[group][it]['G'][:]
        M = file[group][it]['M'][:]

        population = Population(X, F, G, M)

        return population
