import numpy as np
from metrics.performance import hv

def simple_selection(current_set,minimize=True):
    if minimize:
        best = current_set.population[np.argmin(current_set.objective_values)]
        worst = current_set.population[np.argmax(current_set.objective_values)]
    else:
        best = current_set.population[np.argmax(current_set.objective_values)]
        worst = current_set.population[np.argmin(current_set.objective_values)]
    return best,worst

def tournament_selection(current_set,selection_pool,pool_size):
    if selection_pool == "archive":
        metadata = current_set.archive_metadata
        population = current_set.archive_pop
    else:
        metadata = current_set.metadata
        population = current_set.population

    M = len(metadata)

    id1 = np.random.randint(0,M,pool_size)
    id2 = np.random.randint(0,M,pool_size)

    better_rank = metadata[id1,0] < metadata[id2,0]
    equal_rank = metadata[id1,0] == metadata[id2,0]
    better_cd = metadata[id1,1] > metadata[id2,1]

    selected_b = np.where(better_rank | equal_rank & better_cd,
                        id2,id1)

    selected_w = np.where(better_rank | equal_rank & better_cd,
                        id1,id2)

    return population[selected_b],population[selected_w]

def random_selection(current_set,selection_pool,pool_size):

    if selection_pool == "population":
        if np.any(current_set.metadata[:,0]!=0):
            h=0
            population_b = current_set.population[current_set.metadata[:,0]==0]
            population_w = current_set.population[current_set.metadata[:,0]!=0]

            M_b = len(population_b)
            M_w = len(population_w)

            selected_b = np.random.randint(0,M_b,pool_size)
            selected_w = np.random.randint(0,M_w,pool_size)

        else: 
            h=1
            half = pool_size//2
            population_b = current_set.population[:half,:]
            population_w = current_set.population[half:,:]

            M_b = len(population_b)
            M_w = len(population_w)

            selected_b = np.random.randint(0,M_b,pool_size)
            selected_w = np.random.randint(0,M_w,pool_size)

    elif selection_pool == "archive":

        population_b = current_set.pareto_pop

        if np.any(current_set.metadata[:,0]!=0):
            h=0
            population_w = current_set.population[current_set.metadata[:,0]!=0]

            M_b = len(population_b)
            M_w = len(population_w)

            selected_b = np.random.randint(0,M_b,pool_size)
            selected_w = np.random.randint(0,M_w,pool_size)

        else: 
            h=1
            half = pool_size//2
            population_w = current_set.population[half:,:]

            M_b = len(population_b)
            M_w = len(population_w)

            selected_b = np.random.randint(0,M_b,pool_size)
            selected_w = np.random.randint(0,M_w,pool_size)

    # print(f"Gen {gen_i}: Best Pool Size: {M_b}, Worst Pool Size: {M_w}, Hit:{h}")

    return population_b[selected_b], population_w[selected_w]


def hv_random_selection(current_set,selection_pool,pool_size):

    if selection_pool == "population":
        population_b = current_set.population[current_set.metadata[:,0]==0]
    elif selection_pool == "archive":
        population_b = current_set.archive_pop

    if np.any(current_set.metadata[:,0]!=0):
        
        population_w = current_set.population[current_set.metadata[:,0]!=0]

        M_b = len(population_b)
        M_w = len(population_w)

        selected_b = np.random.randint(0,M_b,pool_size)
        selected_w = np.random.randint(0,M_w,pool_size)

    else: 
        half = pool_size//2
        hvs = hv(current_set.objective_values,[1.1,1.1,1.1])
        order = np.argsort(hvs)

        population_b = current_set.population[order[half:],:]
        population_w = current_set.population[order[:half],:]

        M_b = len(population_b)
        M_w = len(population_w)

        selected_b = np.random.randint(0,M_b,pool_size)
        selected_w = np.random.randint(0,M_w,pool_size)

    return population_b[selected_b], population_w[selected_w]