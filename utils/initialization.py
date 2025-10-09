import numpy as np
from pymoo.operators.sampling.lhs import LHS
from pymoo.core.problem import Problem

def random_initialize(pop_size,n_vars,bounds):
    pop = np.zeros([pop_size,n_vars])
    for i in range(n_vars):
        pop[:,i] = np.random.uniform(bounds[i,0],
                                    bounds[i,1], 
                                    pop_size)
    return pop

def lhs_initialize(pop_size, n_vars, bounds):
    sampler = LHS()
    # X = sampler.do(None, pop_size, n_vars)
    X = LHS().do(Problem(n_var=n_vars, xl=0, xu=1), pop_size).get("X")
    # Scale from [0,1] to bounds
    scaled = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * X
    return scaled
