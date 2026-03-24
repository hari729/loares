import numpy as np
from loares.core.problem import Problem as loaresProblem
from loares.core.population import Population
from loares.core.results import ResultProcessor
from pymoo.core.problem import Problem as pymooProblem


class pymoo_to_loares_prob(loaresProblem):
    def __init__(self, pymoo_prob, psize=100, max_evals=10000):

        super().__init__(
            name=str(pymoo_prob.__class__.__name__),
            function=pymoo_prob.evaluate,
            n_vars=pymoo_prob.n_var,
            n_obj=pymoo_prob.n_obj,
            n_constr=pymoo_prob.n_constr,
            psize=psize,
            max_evals=max_evals,
            bounds=np.vstack(pymoo_prob.bounds()).T,
            minmax=["min"] * pymoo_prob.n_obj,
        )

        if self.n_constr == 0:
            self.evaluate = self.evaluate_no_constr

        self.pareto_front = pymoo_prob.pareto_front(100)

    def get_true_front(self):
        return self.pareto_front

    def evaluate_no_constr(self, solutions):
        F = self.function(solutions)
        return F, np.full((solutions.shape[0], 1), -1)


class loares_to_pymoo_prob(pymooProblem):
    def __init__(self, loares_prob):
        self.custom_eval = loares_prob.evaluate
        self.minmax = loares_prob.minmax
        self.variable_modifier = loares_prob.variable_modifier
        super().__init__(
            n_var=loares_prob.n_vars,
            n_obj=loares_prob.n_obj,
            n_constr=loares_prob.n_constr,
            xl=loares_prob.bounds[:, 0],
            xu=loares_prob.bounds[:, 1],
        )

    def _evaluate(self, x, out, *args, **kwargs):
        x = self.variable_modifier(x)
        out["F"], out["G"] = self.custom_eval(x)
        out["F"] = out["F"]*self.minmax

def pymoo_to_loares_h5(
    problem_info, algorithm_info, seed, pymooResult, populationHandler, hdf5_path
):
    h5 = ResultProcessor.open(hdf5_path, problem_info, algorithm_info, seed)
    pop = None
    for algo in pymooResult.history:
        feasible = np.all(algo.opt.get("G") <= 0, axis=1)
        pop = Population(
            algo.opt.get("X")[feasible],
            algo.opt.get("F")[feasible],
            algo.opt.get("G")[feasible],
        )
        ResultProcessor.write_snapshot(h5, pop, algo.evaluator.n_eval)
    if pop is not None:
        ResultProcessor.write_final(h5, populationHandler.get_dict(pop))
    ResultProcessor.close(h5)
