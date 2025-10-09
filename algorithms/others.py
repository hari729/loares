import numpy as np

def booster(pop,bounds,no_sols,b_range):

    r_i = np.random.randint(0, pop.shape[0], size=no_sols)

    boost_values = np.random.choice(b_range, 
                                    size=(no_sols, pop.shape[1]))

    new = pop

    new[-no_sols:,:] = new[-no_sols:,:] + boost_values

    new = np.clip(new,bounds[:,0],bounds[:,1])

    return new

def edge_local_boost(current_set,new_pop, bounds, n_sols, selection_pool="archive", factor_eb=0.1, factor_ls=0.05):
    """
    Applies edge boosting and local search on a subset of the current archive population.

    Parameters:
    - current_set: object with archive_pop (ndarray of solutions)
    - bounds: ndarray (n_vars, 2), lower and upper bounds
    - n_sols: total number of boosted solutions desired
    - factor_eb: perturbation scale for edge boosting
    - factor_ls: perturbation scale for local search

    Returns:
    - ndarray of boosted solutions (n_sols, n_vars)
    """
    if n_sols == 0 :
        return new_pop
    
    b = bounds
    variables = b.shape[0]
    boosted_solutions = np.empty((0, variables))

    eb_n = n_sols // 3
    ls_n = n_sols - eb_n

    if selection_pool == "archive":
        population_b = current_set.archive_pop
    else:
        population_b = current_set.population

    # Edge Boosting
    if len(population_b) > 0:
        eb_base = population_b[
            np.random.choice(
                len(population_b),
                size=min(eb_n, len(population_b)),
                replace=len(population_b) < eb_n
            )
        ]
        eb_factors = (np.random.rand(len(eb_base), variables) - 0.5) * factor_eb
        eb_rand = np.random.rand(len(eb_base), 1)
        eb_perturbed = eb_base + eb_rand * eb_factors
        ebpop = np.clip(eb_perturbed, b[:, 0], b[:, 1])
        boosted_solutions = np.vstack([boosted_solutions, ebpop])

    # Local Search
    if len(population_b) >= ls_n:
        ls_base = population_b[:ls_n]
        ls_factors = (np.random.rand(ls_n, variables) - 0.5) * factor_ls
        ls_rand = np.random.rand(ls_n, 1)
        ls_perturbed = ls_base + ls_rand * ls_factors
        lspop = np.clip(ls_perturbed, b[:, 0], b[:, 1])
        boosted_solutions = np.vstack([boosted_solutions, lspop])

    new_pop = np.row_stack([new_pop,boosted_solutions])

    return new_pop

# def eqn1(current_set, b,):
#     pop_size = current_set.population.shape[0]
#     variables = current_set.population.shape[1]
    
#     # Generate random numbers
#     r_i = np.random.randint(0, pop_size, size=pop_size)
#     r = np.random.rand(pop_size, variables, 3)
#     r4 = np.random.rand(pop_size,1)
#     T = np.random.choice([1, 2], size=(pop_size, variables))
    
#     # Create mask
#     mask = r4 > 0.5

#     best_pop = current_set.best
#     random_pop = current_set.population[r_i]
#     worst_pop = current_set.worst
    
#     # Compute BWR formula for all positions
#     bwr_result = (current_set.population + 
#                  r[:, :, 0] * (best_pop - current_set.population)*T -
#                  r[:, :, 1] * (worst_pop - random_pop))
    
#     # Compute random initialization for all positions
#     random_init = b[None, :, 1] - r[:, :, 2] * (b[None, :, 1] - b[None, :, 0])
    
#     # Use numpy.where to choose between BWR and random based on mask
#     temp_population = np.where(mask, bwr_result, random_init)

#     # Apply bounds
#     bwr_result = np.clip(bwr_result, b[None,:, 0], b[None,:, 1])

#     return temp_population