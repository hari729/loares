import numpy as np

from core.algorithm import Algorithm
from moo.sorting import ranking_crowding
from moo.metrics import performance_metrics
from base.bwr import bwr
from base.bmr import bmr
from base.bmwr import bmwr
from moo.bw_selection import random_selection

class MO_BMR(Algorithm):
    def __init__(self,
                 basefunction=bmr,
                 mutation = None,
                 problem=None,
                 selection_function=random_selection,
                 sorting_function=ranking_crowding,
                 pmods=[],
                 # tracker = Tracker(),
                 # intitializer = random_initialize,
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
                 metrics_function = metrics_function)


class MO_BWR(Algorithm):
    def __init__(self,
                 basefunction=bwr,
                 mutation = None,
                 problem=None,
                 selection_function=random_selection,
                 sorting_function=ranking_crowding,
                 pmods=[],
                 # tracker = Tracker(),
                 # intitializer = random_initialize,
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
                 metrics_function = metrics_function)


class MO_BMWR(Algorithm):
    def __init__(self,
                 basefunction=bmwr,
                 mutation = None,
                 problem=None,
                 selection_function=random_selection,
                 sorting_function=ranking_crowding,
                 pmods=[],
                 # tracker = Tracker(),
                 # intitializer = random_initialize,
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
                 metrics_function = metrics_function)
