import numpy as np

def bmr(bounds, population, pool):
    pop_size = population.get_size()

    r_i = np.random.randint(0, pop_size, size=pop_size)
    r1 = np.random.rand(pop_size, 1)
    r2 = np.random.rand(pop_size, 1)
    T = np.random.choice([1, 2], size=(pop_size, 1))

    best_pop = pool["best"]
    mean = np.mean(population, axis=0)
    random_pop = population[r_i]

    new = (
        population + r1 * (best_pop - T * mean) + r2 * (best_pop - random_pop)
    )

    r3 = np.random.rand(pop_size, 1)
    r4 = np.random.rand(pop_size, 1)
    mask = r4 > 0.5

    random_init = bounds[None, :, 1] - r3  * (bounds[None, :, 1] - bounds[None,:, 0])
    new = np.where(mask, new, random_init)    
    new = np.clip(new,bounds[:,0],bounds[:,1])
    
    return new


def bw_selection(population):
    best = population[0, :]
    worst = population[-1, :]
    return {"best": best, "worst": worst}


def bw_sorting(problem, population, limit, seed, ndf=False, all=False):
    if limit is None:
        limit = problem.psize
    violation_count = np.atleast_2d((population.constraints > 0).sum(axis=1)).T
    sorted_idx = np.lexsort((population.objectives[:, 0], violation_count[:, 0]))[
        :limit
    ]
    sols = population.solutions[sorted_idx]
    objs = population.objectives[sorted_idx]
    constr = population.constraints[sorted_idx]
    metadata = violation_count[sorted_idx]
    return sols, objs, constr, metadata
