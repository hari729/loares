import numpy as np
from opti.core.population import Population, PopulationHandler
from opti.core.flow import FlowHandler

from opti.algorithms.moo.mods import local_search
from opti.algorithms.moo.sorting import ranking_crowding
from opti.algorithms.moo.selection import random_bw_selection as bw_selection

from opti.base.bmr import bmr
from opti.base.bwr import bwr
from opti.base.bmwr import bmwr
from opti.base.mutation import random_reinit
from opti.core.update import UpdateRule

BMR = UpdateRule(bw_selection, bmr, random_reinit)
BWR = UpdateRule(bw_selection, bwr, random_reinit)
BMWR = UpdateRule(bw_selection, bmwr, random_reinit)

class MOPopulationHandler(PopulationHandler):
    def __init__(self):
        super().__init__(ranking_crowding)

    def get_raw_pareto(self, population):
        mask = (population.metadata[:,0] == 0)
        ps = population.solutions[mask]
        po = population.objectives[mask]
        pc = population.constraints[mask]
        pm = population.metadata[mask]

        _, unique_idx = np.unique(po, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)

        return ps[unique_idx], po[unique_idx], pc[unique_idx], pm[unique_idx]

    def get_refined(self, population):
        ps,po,pc,pm = self.get_raw_pareto(population)
        return Population(ps, po, pc, pm.astype(float))

    def get_refined_dict(self, population):
        ps,po,pc,_ = self.get_raw_pareto(population)
        combined = np.hstack([ps, po, pc])
        col_labels = (
            [f"x{i+1}" for i in range(ps.shape[1])] +
            [f"f{j+1}" for j in range(po.shape[1])] +
            [f"g{k+1}" for k in range(pc.shape[1])]
        )
        return {name: combined[:, idx] for idx, name in enumerate(col_labels)}

class MORankingCrowdingAlgo(FlowHandler):
    def __init__(self, ProblemHandler, UpdateRule, Mods=[local_search]):
        super().__init__(ProblemHandler, UpdateRule, MOPopulationHandler(), 
                         Mods)

class MO_BMR(MORankingCrowdingAlgo):
    def __init__(self, ProblemHandler):
        super().__init__(ProblemHandler, BMR)

class MO_BWR(MORankingCrowdingAlgo):
    def __init__(self, ProblemHandler):
        super().__init__(ProblemHandler, BWR)

class MO_BMWR(MORankingCrowdingAlgo):
    def __init__(self, ProblemHandler):
        super().__init__(ProblemHandler, BMWR)
