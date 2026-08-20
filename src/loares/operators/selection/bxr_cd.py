import numpy as np


def CD_BW_selection(pop, random_state, algorithm=None, **kwargs):
    X = pop.get("X")
    n = len(X)
    rank = pop.get("rank")

    # Split by rank: best = rank 0, worst = everything else
    # If all solutions have rank 0 (or rank not set), split by halves
    if rank is not None and np.any(rank != 0):
        best_X = X[rank == 0]
        worst_X = X[rank != 0]
    else:
        half = n // 2
        best_X = X[:half]
        worst_X = X[half:]

    # Sample pop_size parents from each group (with replacement)
    best_idx = random_state.integers(0, len(best_X), size=n)
    worst_idx = random_state.integers(0, len(worst_X), size=n)
    rand_idx = random_state.integers(0, n, size=n)

    return {
        "best": best_X[best_idx],
        "worst": worst_X[worst_idx],
        "random": X[rand_idx],
    }


def CD_BW_archive_selection(pop, random_state, algorithm=None, **kwargs):
    X = pop.get("X")
    n = len(X)
    rank = pop.get("rank")

    # Best from archive if available
    archive_pop = getattr(algorithm, "archive", None)
    if archive_pop is not None and len(archive_pop) > 0:
        best_X = archive_pop.get("X")
    elif rank is not None and np.any(rank == 0):
        best_X = X[rank == 0]
    else:
        best_X = X[: n // 2]

    # Worst from current population
    if rank is not None and np.any(rank != 0):
        worst_X = X[rank != 0]
    else:
        worst_X = X[n // 2 :]

    best_idx = random_state.integers(0, len(best_X), size=n)
    worst_idx = random_state.integers(0, len(worst_X), size=n)
    rand_idx = random_state.integers(0, n, size=n)

    return {
        "best": best_X[best_idx],
        "worst": worst_X[worst_idx],
        "random": X[rand_idx],
    }
