import numpy as np


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
