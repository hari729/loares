from core.metrics import performance_metrics as metrics

class Result():
    def __init__(self,
                 problem_settings = None,
                 algorithm_settings = None):
        self.problem = problem_settings
        self.algorithm = algorithm_settings
        self.convergence_data = []
        self.final_metrics = None
        self.final_population = None

    def add_convergence_data(self, problem, population, evals):
        self.convergence_data.append(np.append(metrics(problem, population), evals))

    def set_final_population(self, problem, population):
        self.final_population = population
        self.final_metrics = metrics(problem, population)
