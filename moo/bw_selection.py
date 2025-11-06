import numpy as np

def random_selection(population):
    pool_size = population.solutions.shape[0]
    if np.any(population.metadata[:,0]!=0):
        h=0
        population_b = population.solutions[population.metadata[:,0]==0]
        population_w = population.solutions[population.metadata[:,0]!=0]

        M_b = len(population_b)
        M_w = len(population_w)

        selected_b = np.random.randint(0,M_b,pool_size)
        selected_w = np.random.randint(0,M_w,pool_size)

    else: 
        h=1
        half = pool_size//2
        population_b = population.solutions[:half,:]
        population_w = population.solutions[half:,:]

        M_b = len(population_b)
        M_w = len(population_w)

        selected_b = np.random.randint(0,M_b,pool_size)
        selected_w = np.random.randint(0,M_w,pool_size)

    return population_b[selected_b], population_w[selected_w]
