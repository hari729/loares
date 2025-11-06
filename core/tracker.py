
class Tracker():
    def __init__(self, problem):
        self.max_evals = problem.max_evals
        self.evals = 0
        self.prev_evals = 0
        self.tracking_interval = int(self.max_evals * 0.05)

    def remaining_evals(self):
        return self.max_evals - self.evals

    def evaluate(self, problem, solutions):
        if self.remaining_evals() < solutions.shape[0]:
            solutions = solutions[:self.remaining_evals,:]
        self.prev_evals = self.evals
        self.evals += solutions.shape[0]
        objectives, constraints =  problem.evaluate(solutions)
        return solutions, objectives, constraints

    def record(self, problem, population, result):
        if (self.evals//self.tracking_interval) > (self.prev_evals//self.tracking_interval):
           result.add_convergence_data(population, self.evals)
