import numpy as np

def random_bw_selection(population):
    pool_size = population.solutions.shape[0]
    if np.any(population.metadata[:,0]!=0):
        population_b = population.solutions[population.metadata[:,0]==0]
        population_w = population.solutions[population.metadata[:,0]!=0]

        M_b = len(population_b)
        M_w = len(population_w)

        selected_b = np.random.randint(0,M_b,pool_size)
        selected_w = np.random.randint(0,M_w,pool_size)

    else: 
        half = pool_size//2
        population_b = population.solutions[:half,:]
        population_w = population.solutions[half:,:]

        M_b = len(population_b)
        M_w = len(population_w)

        selected_b = np.random.randint(0,M_b,pool_size)
        selected_w = np.random.randint(0,M_w,pool_size)

    return {"best":population_b[selected_b], "worst":population_w[selected_w]}

def bw_selection(population):
    best = population.solutions[0,:]
    worst = population.solutions[-1,0]
    return {"best":best, "worst":worst}

def bw_selection_a(population, archive):
    best = archive.solutions[0,:]
    worst = population.solutions[-1,0]
    return {"best":best, "worst":worst}


def archive_bw_selection(population, archive):
    pool_size = population.get_size()
    if np.any(population.metadata[:,0]!=0):
        population_b = archive.solutions
        population_w = population.solutions[population.metadata[:,0]!=0]

        M_b = len(population_b)
        M_w = len(population_w)

        selected_b = np.random.randint(0,M_b,pool_size)
        selected_w = np.random.randint(0,M_w,pool_size)

    else: 
        half = pool_size//2
        population_b = archive.solutions
        population_w = population.solutions[half:,:]

        M_b = len(population_b)
        M_w = len(population_w)

        selected_b = np.random.randint(0,M_b,pool_size)
        selected_w = np.random.randint(0,M_w,pool_size)

    return {"best":population_b[selected_b], "worst":population_w[selected_w]}
