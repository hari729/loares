
class Result():
    def __init__(self,
                 problem_settings = None,
                 algorithm_settings = None):
        self.problem = problem_settings
        self.algorithm = algorithm_settings
        self.convergence_data = []
        self.final_metrics = None
        self.final_population = None

    def add_convergance_data(self, population, evals):
        self.convergence_data.append(np.append(metrics(population), evals))

    def set_final_population(self, population):
        self.final_population = population
        self.final_metrics = metrics(population)
