import numpy as np
from opti.core.problem import Problem as optiProblem
from pymoo.core.problem import Problem as pymooProblem

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

        self.pareto_front = pymoo_prob.pareto_front

    def get_true_front(self, pts=500):
        return self.pareto_front(pts)


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
