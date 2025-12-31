import numpy as np
from opti.core.population import Population, PopulationHandler
from opti.core.flow import FlowHandler

from opti.base.bw_rules import BMR, BWR, BMWR
from opti.algorithms.moo.mods import local_search
from opti.algorithms.moo.sorting import ranking_crowding


class MOPopulationHandler(PopulationHandler):
    def __init__(self):
        super().__init__(ranking_crowding)

    def get_pareto(self):
        mask = (self.population.metadata[:,0] == 0)
        ps = self.population.solutions[mask]
        po = self.population.objectives[mask]
        pc = self.population.constraints[mask]
        pm = self.population.metadata[mask]

        _, unique_idx = np.unique(po, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)

        return ps[unique_idx], po[unique_idx], pc[unique_idx], pm[unique_idx]

    def get_pareto_population(self):
        ps,po,pc,pm = self.get_pareto()
        return Population(ps, po, pc, pm)

    def get_pareto_dict(self):
        ps,po,pc,_ = self.get_pareto()
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

    def record(self):
            if self.ProblemHandler.interval_status():
                self.PopulationRecorder.record(self.PopulationHandler.get_pareto_population(),
                                                self.ProblemHandler.evals)

class MO_BMR(MORankingCrowdingAlgo):
    def __init__(self, ProblemHandler):
        super().__init__(ProblemHandler, BMR)

class MO_BWR(MORankingCrowdingAlgo):
    def __init__(self, ProblemHandler):
        super().__init__(ProblemHandler, BWR)

class MO_BMWR(MORankingCrowdingAlgo):
    def __init__(self, ProblemHandler):
        super().__init__(ProblemHandler, BMWR)
