import numpy as np
from scipy.spatial import KDTree
from scipy.stats import norm
from pymoo.util.normalization import normalize

def bw_selection(population):
    best = population.solutions[0, :]
    worst = population.solutions[-1, :]
    return {"best": best, "worst": worst}

def random_bw_selection(population):
    pool_size = population.solutions.shape[0]
    if np.any(population.metadata[:, 0] != 0):
        population_b = population.solutions[population.metadata[:, 0] == 0]
        population_w = population.solutions[population.metadata[:, 0] != 0]
    else:
        half = pool_size // 2
        population_b = population.solutions[:half, :]
        population_w = population.solutions[half:, :]

    M_b = len(population_b)
    M_w = len(population_w)
    selected_b = np.random.randint(0, M_b, pool_size)
    selected_w = np.random.randint(0, M_w, pool_size)
    return {"best": population_b[selected_b], "worst": population_w[selected_w]}

def archive_bw_selection(population, archive):
    pool_size = population.get_size()
    if np.any(population.metadata[:, 0] != 0):
        population_b = archive.solutions
        population_w = population.solutions[population.metadata[:, 0] != 0]
    else:
        half = pool_size // 2
        population_b = archive.solutions
        population_w = population.solutions[half:, :]

    M_b = len(population_b)
    M_w = len(population_w)
    selected_b = np.random.randint(0, M_b, pool_size)
    selected_w = np.random.randint(0, M_w, pool_size)
    return {"best": population_b[selected_b], "worst": population_w[selected_w]}

def get_nn_dist(X, k=5):

    normalized_values = normalize(X,np.min(X, axis=0), np.max(X, axis=0))
    tree = KDTree(normalized_values)

    # k+1 because first neighbor is itself
    dist, idx = tree.query(normalized_values, k=k+1)

    # remove self (first column)
    return idx[:, 1:], dist[:, 1:]

