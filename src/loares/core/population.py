import numpy as np
import warnings
from loares.core.initializer import random_initialize

class Population:
    def __init__(self, X, F, G, M=None):
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


class PopulationHandler:
    def __init__(self, sorting_function, initializer=None):
        if initializer is None:
            initializer = random_initialize
        self.initializer = initializer
        self.sort = sorting_function

    def initialize(self, ProblemHandler, seed):
        self.seed = seed
        X = self.initializer(ProblemHandler.problem)
        population = ProblemHandler.evaluate(X)
        return Population(
            *self.sort(
                ProblemHandler.problem,
                population,
                population.solutions.shape[0],
                self.seed,
            )
        )

    def get_empty_pop(self, nX, nO, nG):
        return Population(np.empty((0,nX)), np.empty((0,nO)), np.empty((0,nG)))

    def raw_update(self, population, X, F, G, M):
        population.solutions = X
        population.objectives = F
        population.constraints = G
        population.metadata = M

    def get_size(self, population):
        return population.solutions.shape[0]

    def merge(self, population_list):
        temp_population = sum(population_list[1:], population_list[0])
        return temp_population

    def random_split(self, population, n_sub_pops):
        if n_sub_pops > self.get_size(population):
            warnings.warn(
                "No. of sub populations exceed population size, value is automatically reduced.",
                Warning,
            )
            n_sub_pops = self.get_size(population)
        idx = np.arange(self.get_size(population))
        np.random.shuffle(idx)
        parts = np.array_split(idx, n_sub_pops)
        return [
            Population(
                population.solutions[i],
                population.objectives[i],
                population.constraints[i],
            )
            for i in parts
        ]

    def NN_split(self, population, n_sub_pops):

        from loares.operators.selection import get_nn_dist
        if n_sub_pops > self.get_size(population):
            warnings.warn(
                "No. of sub populations exceed population size, value is automatically reduced.",
                Warning,
            )
            n_sub_pops = self.get_size(population)
        n = self.get_size(population)
        idx, _ = get_nn_dist(population.objectives, k=min(5, n - 1))
        visited = np.zeros(n, dtype=bool)
        order = []
        current = np.random.randint(n)
        for _ in range(n):
            order.append(current)
            visited[current] = True
            found = False
            for neighbor in idx[current]:
                if not visited[neighbor]:
                    current = neighbor
                    found = True
                    break
            if not found:
                remaining = np.where(~visited)[0]
                if len(remaining) > 0:
                    current = remaining[0]
        parts = np.array_split(order, n_sub_pops)
        return [
            Population(
                population.solutions[i],
                population.objectives[i],
                population.constraints[i],
            )
            for i in parts
        ]

    def get_dict(self, population):
        combined = np.hstack(
            [population.solutions, population.objectives, population.constraints]
        )
        col_labels = (
            [f"x{i + 1}" for i in range(population.solutions.shape[1])]
            + [f"f{j + 1}" for j in range(population.objectives.shape[1])]
            + [f"g{k + 1}" for k in range(population.constraints.shape[1])]
        )
        return {name: combined[:, idx] for idx, name in enumerate(col_labels)}

    def get_sorted(self, population, ProblemHandler, limit=None):
        if limit is None:
            limit = ProblemHandler.problem.psize
        return Population(
            *self.sort(ProblemHandler.problem, population, limit, self.seed)
        )

    def update(self, population_list, ProblemHandler, limit=None):
        temp_population = self.merge(population_list)
        return self.get_sorted(temp_population, ProblemHandler, limit)

    def get_refined(self, population):
        ps, po, pc, pm = self.get_raw_pareto(population)
        return Population(ps, po, pc, pm.astype(float))

    def get_refined_dict(self, population):
        ps, po, pc, _ = self.get_raw_pareto(population)
        combined = np.hstack([ps, po, pc])
        col_labels = (
            [f"x{i + 1}" for i in range(ps.shape[1])]
            + [f"f{j + 1}" for j in range(po.shape[1])]
            + [f"g{k + 1}" for k in range(pc.shape[1])]
        )
        return {name: combined[:, idx] for idx, name in enumerate(col_labels)}

    def get_raw_pareto(self, population):
        mask = population.metadata[:, 0] == 0
        ps = population.solutions[mask]
        po = population.objectives[mask]
        pc = population.constraints[mask]
        pm = population.metadata[mask]

        _, unique_idx = np.unique(po, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)

        return ps[unique_idx], po[unique_idx], pc[unique_idx], pm[unique_idx]
