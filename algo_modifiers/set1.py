import numpy as np

def edge_boost(base_population, bounds, n_boost=5, factor=0.1):
    """
    Generates new solutions by perturbing a random subset of the base population.

    Parameters:
    - base_population: ndarray of solutions to select from (e.g., s_pop).
    - bounds: ndarray (n_vars, 2), lower and upper bounds for clipping.
    - n_boost: The number of boosted solutions to generate.
    - factor: The scale of the random perturbation.

    Returns:
    - ndarray of new, boosted solutions.
    """

    # Randomly select solutions from the base population to perturb
    base_solutions = base_population[
        np.random.choice(
            base_population.shape[0],
            size=min(n_boost, base_population.shape[0]),
            replace=base_population.shape[0] < n_boost
        )
    ]

    variables = bounds.shape[0]
    
    # Generate random perturbation factors
    perturb_factors = (np.random.rand(base_solutions.shape[0], variables) - 0.5) * factor
    rand_scale = np.random.rand(base_solutions.shape[0], 1)
    
    # Apply perturbation and clip the results to the bounds
    perturbed_solutions = base_solutions + rand_scale * perturb_factors
    boosted_solutions = np.clip(perturbed_solutions, bounds[:, 0], bounds[:, 1])

    return boosted_solutions

def local_search(base_population, bounds, n_search=5, factor=0.05):
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

    # Determine how many solutions to actually process
    num_to_process = min(n_search, base_population.shape[0])

    # Select the 'best' solutions from the top of the base population
    base_solutions = base_population[:num_to_process]
    
    variables = bounds.shape[0]

    # Generate random perturbation factors
    perturb_factors = (np.random.rand(num_to_process, variables) - 0.5) * factor
    rand_scale = np.random.rand(num_to_process, 1)

    # Apply perturbation and clip the results to the bounds
    perturbed_solutions = base_solutions + rand_scale * perturb_factors
    searched_solutions = np.clip(perturbed_solutions, bounds[:, 0], bounds[:, 1])

    return searched_solutions

def null(new_p, bounds):
    return new_p

def opposition(new_p, bounds):
    opp_p = np.sum(bounds, axis=1) - new_p
    return opp_p


# def q_opposition(new_p, bounds)

get = {
    "null":null,
    "opposition":opposition,
}

def variations(new_p,bounds,list_of_variants):
    for v_name in list_of_variants:
        variant = get[v_name]
        variant_p = variant(new_p, bounds)
    
    return variant_p


if __name__ == "__main__":

    new_p = np.array([[5,4,3,2]])

    bounds = np.array([[1,6],[2,10],[1,5],[1,10]])

    print(opposition(new_p,bounds))