
import numpy as np

from opti.core.algorithm import SAMP
from opti.core.initializer import random_initialize
from opti.moo.sorting import ranking_crowding
from opti.moo.metrics import performance_metrics
from opti.base.bwr import bwr
from opti.base.bmr import bmr
from opti.base.bmwr import bmwr
from opti.moo.bw_selection import random_bw_selection
from opti.moo.population_modifiers import local_search

class MO_BMR_SAMP(SAMP):
    def __init__(self,
                 mutation = None,
                 problem=None,
                 selection_function=random_bw_selection,
                 sorting_function=ranking_crowding,
                 pmods=[local_search],
                 initializer = random_initialize,
                 metrics_function = performance_metrics,
                 seed = 1):

        super().__init__(
                 basefunction=bmr,
                 mutation = mutation,
                 problem=problem,
                 selection_function=selection_function,
                 sorting_function=sorting_function,
                 pmods=pmods,
                 initializer = initializer,
                 metrics_function = metrics_function,
                 seed = seed)


class MO_BWR_SAMP(SAMP):
    def __init__(self,
                 mutation = None,
                 problem=None,
                 selection_function=random_bw_selection,
                 sorting_function=ranking_crowding,
                 pmods=[local_search],
                 initializer = random_initialize,
                 metrics_function = performance_metrics,
                 seed = 1):

        super().__init__(
                 basefunction=bwr,
                 mutation = mutation,
                 problem=problem,
                 selection_function=selection_function,
                 sorting_function=sorting_function,
                 pmods=pmods,
                 initializer = initializer,
                 metrics_function = metrics_function,
                 seed = seed)


class MO_BMWR_SAMP(SAMP):
    def __init__(self,
                 mutation = None,
                 problem=None,
                 selection_function=random_bw_selection,
                 sorting_function=ranking_crowding,
                 pmods=[local_search],
                 initializer = random_initialize,
                 metrics_function = performance_metrics,
                 seed = 1):

        super().__init__(
                 basefunction=bmwr,
                 mutation = mutation,
                 problem=problem,
                 selection_function=selection_function,
                 sorting_function=sorting_function,
                 pmods=pmods,
                 initializer = initializer,
                 metrics_function = metrics_function,
                 seed = seed)

class_list = [MO_BMR_SAMP, MO_BWR_SAMP, MO_BMWR_SAMP]
