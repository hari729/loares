import numpy as np
from opti.core.problem import Problem


def drag_lift(population):
    x1,x2,x3,x4 = population.T
    lift = (-5.57529 + 0.12894*x1 + 0.21604*x2
            + 0.93850*x3 - 0.07689*x4 - 0.00011*x1**2
            - 0.00353*x2**2 - 0.07071*x3**2 + 0.00243*x4**2
            - 0.00018*x1*x2 + 0.01703*x1*x3 + 0.00295*x1*x4
            + 0.00356*x2*x3 + 0.0003*x2*x4 - 0.01148*x3*x4)
    drag = (-0.00745 + 8.10185*x1 + 0.00045*x2
            + 0.00124*x3 - 4.27546e-5 *x4 + 1.48611e-6 *x1**2
            - 5.7778e-6 *x2**2 + 0.0003*x3**2 + 1.25e-7 *x4**2
            + 1.52778e-6 *x1*x2 - 1.09722e-5 *x1*x3 + 2.43751*x1*x4
            - 6.94444e-5 *x2*x3 + 1.55556e-6 *x2*x4 - 5.51667e-6 *x3*x4)

    return np.column_stack([drag,-lift]), np.full((population.shape[0], 1), -1)

def dl_var_modifier(solutions):
    x1,x2,x3,x4 = solutions.T
    x3 = np.rint(x3)
    x2 = np.round(x2/10) * 10
    x4 = np.round(x4/10) * 10
    return np.column_stack([x1,x2,x3,x4])

class QuadCopter_drag_lift(Problem):
    def __init__(self,
                 psize = 300,
                 max_evals = 60000):

        super().__init__(function = drag_lift,
                            n_vars = 4,
                            n_obj = 2,
                            n_constr = 2,
                            psize = psize,
                            max_evals = max_evals,
                            bounds = np.array([[10,50],[20,40],[1,3],[0,40]]),
                            minmax = np.array([1,-1]),
                         variable_modifier = dl_var_modifier)

