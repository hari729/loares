import numpy as np
from opti.core.problem import Problem as optiProblem
from opti.core.results import Result as optiResult
from opti.core.population import Population
from pymoo.core.problem import Problem as pymooProblem
from pymoo.core.result import Result as pymooResult
from opti.algorithms.moo.base import MOPopulationHandler

class pymoo_to_opti_prob(optiProblem):
    def __init__(self,
                pymoo_prob,
                psize = 100,
                max_evals = 10000):

        super().__init__(name = pymoo_prob.__class__.__name__,
                            function = pymoo_prob.evaluate,
                            n_vars = pymoo_prob.n_var,
                            n_obj = pymoo_prob.n_obj,
                            n_constr = pymoo_prob.n_constr,
                            psize = psize,
                            max_evals = max_evals,
                            bounds = np.vstack(pymoo_prob.bounds()).T,
                            minmax = np.ones([1, pymoo_prob.n_obj]))

        if self.n_constr == 0:
            self.evaluate = self.evaluate_no_contr

        self.pareto_front = pymoo_prob.pareto_front(100)

    def get_true_front(self):
        return self.pareto_front


    def evaluate_no_contr(self, solutions):
        F = self.function(solutions)
        return F, np.full((solutions.shape[0], 1), -1)

class opti_to_pymoo_prob(pymooProblem):

    def __init__(self, opti_prob):
        self.custom_eval = opti_prob.evaluate
        super().__init__(n_var=opti_prob.n_vars,
                         n_obj=opti_prob.n_obj, 
                         n_constr=opti_prob.n_constr, 
                         xl=opti_prob.bounds[:,0],
                         xu=opti_prob.bounds[:,1],)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"], out["G"] = self.custom_eval(x)

def pymoo_to_opti_res(problem_info, algorithm_info, seed, pymooResult, populationHandler):
    result = optiResult(problem_info, algorithm_info, seed)
    for algo in pymooResult.history:
        feasible = np.all(algo.opt.get("G") < 0, axis=1)
        pop = Population(algo.opt.get("X")[feasible],
                         algo.opt.get("F")[feasible],
                         algo.opt.get("G")[feasible])
        result.record(pop, algo.evaluator.n_eval)
    result.stop(populationHandler.get_dict(pop)) 
    return result
