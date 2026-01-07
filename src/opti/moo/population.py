import numpy as np
from opti.core.population import Population, PopulationHandler

class MoPopulation(Population):
    def __init__(self, X, F, G, M = None):
        super().__init__(X, F, G, M)


    def get_pareto(self):
        mask = (self.metadata[:,0] == 0)
        ps = self.solutions[mask]
        po = self.objectives[mask]
        pc = self.constraints[mask]
        pm = self.metadata[mask]

        _, unique_idx = np.unique(po, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)

        return ps[unique_idx], po[unique_idx], pc[unique_idx], pm[unique_idx]

    def get_pareto_population(self):
        ps,po,pc,_ = self.get_pareto()
        return Population(ps, po, pc)

    def get_pareto_dict(self):
        ps,po,pc,_ = self.get_pareto()
        combined = np.hstack([ps, po, pc])
        col_labels = (
            [f"x{i+1}" for i in range(ps.shape[1])] +
            [f"f{j+1}" for j in range(po.shape[1])] +
            [f"g{k+1}" for k in range(pc.shape[1])]
        )
        return {name: combined[:, idx] for idx, name in enumerate(col_labels)}

