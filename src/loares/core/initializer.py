import numpy as np
from pymoo.operators.sampling.lhs import LHS
from pymoo.core.problem import Problem as PymooProblem


def random_initialize(problem):
    pop_size = problem.psize
    n_vars = problem.n_vars
    bounds = problem.bounds
    pop = np.zeros([pop_size, n_vars])
    for i in range(n_vars):
        pop[:, i] = np.random.uniform(bounds[i, 0], bounds[i, 1], pop_size)
    return pop


def lhs_initialize(problem):
    pop_size = problem.psize
    n_vars = problem.n_vars
    bounds = problem.bounds
    X = LHS().do(PymooProblem(n_var=n_vars, xl=0, xu=1), pop_size).get("X")
    # Scale from [0,1] to bounds
    scaled = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * X
    return scaled
