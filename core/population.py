
class Population():

    def __init__(self, X, F, G, M = None):
        self.solutions = X 
        self.objectives = F
        self.constraints = G
        self.metadata = M

    def update(self, X, F, G, M):
        self.solutions = X
        self.objectives = F
        self.constraints = G
        self.metadata = M

    def get_pareto(self):
        mask = [self.metadata[:,0] == 0]
        ps = self.solutions[mask]
        po = self.objectives[mask]
        pc = self.constraints[mask]
        pm = self.metadata[mask]
        return ps, po, pc, pm

    def merge(self, new_gen, new_obj, new_constr):
        temp_population = Population(np.row_stack([self.solutions, new_gen]),
                                     np.row_stack([self.objectives, new_obj]),
                                     np.row_stack([self.constraints, new_constr]))
        return temp_population
