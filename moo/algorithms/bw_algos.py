import numpy as np

from opti.core.algorithm import Algorithm
from opti.moo.sorting import ranking_crowding
from opti.moo.metrics import performance_metrics
from opti.base.bwr import bwr
from opti.base.bmr import bmr
from opti.base.bmwr import bmwr
from opti.moo.bw_selection import random_selection
from opti.moo.population_modifiers import local_search

class MO_BMR(Algorithm):
    def __init__(self,
                 basefunction=bmr,
                 mutation = None,
                 problem=None,
                 selection_function=random_selection,
                 sorting_function=ranking_crowding,
                 pmods=[local_search],
                 # tracker = Tracker(),
                 # intitializer = random_initialize,
                 seed = 1,
                 metrics_function = performance_metrics):

        super().__init__(
                 basefunction=basefunction,
                 mutation = mutation,
                 problem=problem,
                 selection_function=selection_function,
                 sorting_function=sorting_function,
                 pmods=pmods,
                 # tracker = Tracker(),
                 # intitializer = intitializer,
                 metrics_function = metrics_function,
                 seed = seed)


class MO_BWR(Algorithm):
    def __init__(self,
                 basefunction=bwr,
                 mutation = None,
                 problem=None,
                 selection_function=random_selection,
                 sorting_function=ranking_crowding,
                 pmods=[local_search],
                 # tracker = Tracker(),
                 # intitializer = random_initialize,
                 metrics_function = performance_metrics,
                 seed = 1):

        super().__init__(
                 basefunction=basefunction,
                 mutation = mutation,
                 problem=problem,
                 selection_function=selection_function,
                 sorting_function=sorting_function,
                 pmods=pmods,
                 # tracker = Tracker(),
                 # intitializer = intitializer,
                 metrics_function = metrics_function,
                 seed = seed)


class MO_BMWR(Algorithm):
    def __init__(self,
                 basefunction=bmwr,
                 mutation = None,
                 problem=None,
                 selection_function=random_selection,
                 sorting_function=ranking_crowding,
                 pmods=[local_search],
                 # tracker = Tracker(),
                 # intitializer = random_initialize,
                 metrics_function = performance_metrics,
                 seed = 1):

        super().__init__(
                 basefunction=basefunction,
                 mutation = mutation,
                 problem=problem,
                 selection_function=selection_function,
                 sorting_function=sorting_function,
                 pmods=pmods,
                 # tracker = Tracker(),
                 # intitializer = intitializer,
                 metrics_function = metrics_function,
                 seed = seed)
