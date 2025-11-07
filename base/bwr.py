import numpy as np

def bwr(problem, population, selector):
    pop_size = population.get_size()
    variables = problem.n_vars
    bounds = problem.bounds
    # Generate random numbers
    r_i = np.random.randint(0, pop_size, size=pop_size)

    r1 = np.random.rand(pop_size, 1)
    r2 = np.random.rand(pop_size, 1)
    r3 = np.random.rand(pop_size, 1)
    r4 = np.random.rand(pop_size, 1)

    T = np.random.choice([1, 2], size=(pop_size, 1))
    
    mask = r4 > 0.5
    # First compute all the term arrays
    best_pop ,worst_pop = selector(population)
    
    random_pop = population.solutions[r_i]
    
    # Compute BWR formula for all positions
    new = (population.solutions + 
                 r1  * (best_pop - T * random_pop) -
                 r2  * (worst_pop - random_pop))


    # Compute random initialization for all positions
    random_init = bounds[None, :, 1] - r3  * (bounds[None, :, 1] - bounds[None,:, 0])
    
    # Use numpy.where to choose between BWR and random based on mask
    new = np.where(mask, new, random_init)    
    
    new = np.clip(new,bounds[:,0],bounds[:,1])

    return new
