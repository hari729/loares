from loares.algorithms.moo.base import BMR, BWR, BMWR, MORankingCrowdingAlgo
from loares.algorithms.moo.mods import local_search, opposition

Mods = [local_search, opposition]


class MO_BMR_Opposition(MORankingCrowdingAlgo):
    def __init__(self, ProblemHandler):
        super().__init__(ProblemHandler, BMR, Mods)


class MO_BWR_Opposition(MORankingCrowdingAlgo):
    def __init__(self, ProblemHandler):
        super().__init__(ProblemHandler, BWR, Mods)


class MO_BMWR_Opposition(MORankingCrowdingAlgo):
    def __init__(self, ProblemHandler):
        super().__init__(ProblemHandler, BMWR, Mods)
