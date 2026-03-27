import numpy as np 

def local_search(problem, population, PopulationHandler, factor=0.05):
    """
    Generates new solutions by perturbing the top solutions from the base population.

    Parameters:
    - base_population: ndarray of solutions to select from (e.g., s_pop),
                       assumed to be sorted with the best solutions first.
    - bounds: ndarray (n_vars, 2), lower and upper bounds for clipping.
    - n_search: The number of new solutions to generate.
    - factor: The scale of the random perturbation.

    Returns:
    - ndarray of new, locally searched solutions.
    """
    base_population,_,_,_ = PopulationHandler.get_raw_pareto(population)
    bounds = problem.bounds

    n_search = int(np.sqrt(base_population.shape[0]))

    # Determine how many solutions to actually process 
    num_to_process = min(n_search, base_population.shape[0])
    selected = np.random.choice(base_population.shape[0], num_to_process, replace=False)
    base_solutions = base_population[selected]
    
    variables = bounds.shape[0]

    # Generate random perturbation factors
    perturb_factors = (np.random.rand(num_to_process, variables) - 0.5) * factor
    rand_scale = np.random.rand(num_to_process, 1)

    # Apply perturbation and clip the results to the bounds
    perturbed_solutions = base_solutions + rand_scale * perturb_factors
    searched_solutions = np.clip(perturbed_solutions, bounds[:, 0], bounds[:, 1])

    return searched_solutions

def opposition(problem, population, PopulationHandler):
    bounds = problem.bounds
    current_p = population.solutions
    opp_p = np.sum(bounds, axis=1) - current_p
    opp_p = np.clip(opp_p, bounds[:,0], bounds[:,1])
    return opp_p
