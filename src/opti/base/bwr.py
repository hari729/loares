import numpy as np


def bwr(problem, population, pool):
    pop_size = population.get_size()
    bounds = problem.bounds

    r_i = np.random.randint(0, pop_size, size=pop_size)
    r1 = np.random.rand(pop_size, 1)
    r2 = np.random.rand(pop_size, 1)
    T = np.random.choice([1, 2], size=(pop_size, 1))

    best_pop = pool["best"]
    worst_pop = pool["worst"]
    random_pop = population.solutions[r_i]

    new = (
        population.solutions
        + r1 * (best_pop - T * random_pop)
        - r2 * (worst_pop - random_pop)
    )

    new = np.clip(new, bounds[:, 0], bounds[:, 1])

    return new
