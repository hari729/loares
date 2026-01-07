from opti.base.bw_rules import BMR, BWR, BMWR
from opti.algorithms.moo.base import MORankingCrowdingAlgo
from opti.algorithms.moo.mods import local_search, opposition

Mods = [local_search, opposition]

class MO_BMR_OPPOSITION(MORankingCrowdingAlgo):
    def __init__(self, ProblemHandler):
        super().__init__(ProblemHandler, BMR, Mods)

class MO_BWR_OPPOSITION(MORankingCrowdingAlgo):
    def __init__(self, ProblemHandler):
        super().__init__(ProblemHandler, BWR, Mods)

class MO_BMWR_OPPOSITION(MORankingCrowdingAlgo):
    def __init__(self, ProblemHandler):
        super().__init__(ProblemHandler, BMWR, Mods)
