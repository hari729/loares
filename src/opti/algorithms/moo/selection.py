import numpy as np

SELECTION_RATIO = 0.1


def _top_bottom_indices(M_b, M_w, pool_size):
    top_k = max(1, int(SELECTION_RATIO * M_b))
    bottom_k = max(1, int(SELECTION_RATIO * M_w))
    selected_b = np.random.randint(0, top_k, pool_size)
    selected_w = np.random.randint(M_w - bottom_k, M_w, pool_size)
    return selected_b, selected_w


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
    selected_b, selected_w = _top_bottom_indices(M_b, M_w, pool_size)
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
    selected_b, selected_w = _top_bottom_indices(M_b, M_w, pool_size)
    return {"best": population_b[selected_b], "worst": population_w[selected_w]}
