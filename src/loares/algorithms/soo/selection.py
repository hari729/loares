import numpy as np


def bw_selection(population):
    best = population.solutions[0, :]
    worst = population.solutions[-1, :]
    return {"best": best, "worst": worst}
