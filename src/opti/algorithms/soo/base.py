from opti.core.population import PopulationHandler
from opti.core.flow import FlowHandler

from opti.algorithms.soo.sorting import bw_sorting
from opti.algorithms.soo.selection import bw_selection

from opti.base.bmr import bmr
from opti.base.bwr import bwr
from opti.base.bmwr import bmwr
from opti.base.mutation import random_reinit
from opti.core.update import UpdateRule

BMR = UpdateRule(bw_selection, bmr, random_reinit)
BWR = UpdateRule(bw_selection, bwr, random_reinit)
BMWR = UpdateRule(bw_selection, bmwr, random_reinit)


class SOPopulationHandler(PopulationHandler):
    def __init__(self):
        super().__init__(bw_sorting)


class SOAlgo(FlowHandler):
    def __init__(self, ProblemHandler, UpdateRule, Mods=None):
        if Mods is None:
            Mods = []
        super().__init__(ProblemHandler, UpdateRule, SOPopulationHandler(), Mods)


class SO_BMR(SOAlgo):
    def __init__(self, ProblemHandler):
        super().__init__(ProblemHandler, BMR)


class SO_BWR(SOAlgo):
    def __init__(self, ProblemHandler):
        super().__init__(ProblemHandler, BWR)


class SO_BMWR(SOAlgo):
    def __init__(self, ProblemHandler):
        super().__init__(ProblemHandler, BMWR)
