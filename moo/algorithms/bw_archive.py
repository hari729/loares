
from opti.core.initializer import random_initialize
from opti.base.bwr import bwr
from opti.base.bmr import bmr
from opti.base.bmwr import bmwr
from opti.moo.algorithms.general import MOArchive
from opti.moo.bw_selection import bw_selection_a
from opti.moo.population_modifiers import local_search


class MO_BMR_ARCHIVE(MOArchive):
    def __init__(self,
                 mutation = None,
                 problem = None,
                 selection_function = bw_selection_a,
                 pmods=[local_search],
                 initializer = random_initialize,
                 seed = 1):

        super().__init__(
                 basefunction = bmr,
                 mutation = mutation,
                 problem = problem,
                 selection_function = selection_function,
                 pmods = pmods,
                 initializer = initializer,
                 seed = seed)


class MO_BWR_ARCHIVE(MOArchive):
    def __init__(self,
                 mutation = None,
                 problem = None,
                 selection_function = bw_selection_a,
                 pmods=[local_search],
                 initializer = random_initialize,
                 seed = 1):

        super().__init__(
                 basefunction = bwr,
                 mutation = mutation,
                 problem = problem,
                 selection_function = selection_function,
                 pmods = pmods,
                 initializer = initializer,
                 seed = seed)


class MO_BMWR_ARCHIVE(MOArchive):
    def __init__(self,
                 mutation = None,
                 problem = None,
                 selection_function = bw_selection_a,
                 pmods=[local_search],
                 initializer = random_initialize,
                 seed = 1):

        super().__init__(
                 basefunction = bmwr,
                 mutation = mutation,
                 problem = problem,
                 selection_function = selection_function,
                 pmods = pmods,
                 initializer = initializer,
                 seed = seed)

class_list = [MO_BMR_ARCHIVE, MO_BWR_ARCHIVE, MO_BMWR_ARCHIVE]
