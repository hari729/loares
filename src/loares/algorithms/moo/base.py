from loares.core.population import PopulationHandler
from loares.core.flow import FlowHandler

from loares.algorithms.moo.mods import local_search
from loares.algorithms.moo.sorting import ranking_crowding
from loares.algorithms.moo.selection import random_bw_selection as bw_selection

from loares.base.bmr import bmr
from loares.base.bwr import bwr
from loares.base.bmwr import bmwr
from loares.base.mutation import random_reinit
from loares.core.update import UpdateRule

BMR = UpdateRule(bw_selection, bmr, random_reinit)
BWR = UpdateRule(bw_selection, bwr, random_reinit)
BMWR = UpdateRule(bw_selection, bmwr, random_reinit)


class MOPopulationHandler(PopulationHandler):
    def __init__(self):
        super().__init__(ranking_crowding)


class MORankingCrowdingAlgo(FlowHandler):
    def __init__(self, ProblemHandler, UpdateRule, Mods=None):
        if Mods is None:
            Mods = [local_search]
        super().__init__(ProblemHandler, UpdateRule, MOPopulationHandler(), Mods)


class MO_BMR(MORankingCrowdingAlgo):
    def __init__(self, ProblemHandler):
        super().__init__(ProblemHandler, BMR)


class MO_BWR(MORankingCrowdingAlgo):
    def __init__(self, ProblemHandler):
        super().__init__(ProblemHandler, BWR)


class MO_BMWR(MORankingCrowdingAlgo):
    def __init__(self, ProblemHandler):
        super().__init__(ProblemHandler, BMWR)
