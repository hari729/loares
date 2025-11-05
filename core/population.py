
class Population():

    def __init__(self, problem, initializer):
        self.solutions = initializer(problem)
        self.objectives, self.constraints = problem.evaluate(self.solutions)
        self.metadata = None
        self.true_front = problem.get_true_front()

    def update(self, X, F, G, M):
        self.solutions = X
        self.objectives = F
        self.constraints = G
        self.metadata = M

    def get_pareto(self):
        mask = [self.metadata[:,0] == 0]
        ps = self.solutions[mask]
        po = self.objectives[mask]
        pg = self.constraints[mask]
        pm = self.metadata[mask]
