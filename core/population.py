import numpy as np 
import warnings

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

