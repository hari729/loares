
class Result():
    def __init__(self, problem_info, algo_info, seed):
        self.problem_info = problem_info
        self.algorithm_info = algo_info
        self.seed = seed
        self.history = {'pop':[],'evals':[]}

    def record(self, population, evals):
        self.history['pop'].append(population)
        self.history['evals'].append(evals)

    def stop(self, final_dict):
        self.population = self.history['pop'][-1]
        self.final_dict = final_dict

class ResultProcessor():
    def __init__(self):
        pass

    def get_metrics_history(self, result, performance_metrics):
        metrics_history = {}
        for i,evals in enumerate(result.history['evals']):
            metrics = performance_metrics(result.history['pop'][i].objectives,
                                          result.problem_info["TF"])

            for key, value in metrics.items():
                metrics_history.setdefault(key, []).append(value)
            metrics_history.setdefault("evals", []).append(evals)

        return metrics_history

    def get_final_pop(self, result):
        return result.history['pop'][-1]
