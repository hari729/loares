from opti.algorithms.soo.selection import bw_selection

def bw_fitness(objective_values, truefront):
    metrics = {'best':float(objective_values[0]), 'worst':float(objective_values[-1])}
    return metrics
