import numpy as np 
from opti.core.problem import Problem 

def hir(solutions):
    L1, L2, L3, L4, L5, L7 = solutions.T

    f_alpha = (6.61 
            - 24.1e-3 * L1
            - 18.1e-3 * L2
            - 51.7e-3 * L3
            - 9.5e-3 * L5
            + 27.1e-3 * L7
            + 2.1e-4 * L1 * L3)
    
    f_beta = (43.85
            - 94.3e-3 * L1
            - 91.2e-3 * L2
            + 92.3e-3 * L3
            + 110.5e-3 * L4
            - 114.1e-3 * L5
            - 73.9e-3 * L7
            - 2.1e-4 * L4 * L5)
    
    f_gamma = (0.74
            + 10.5e-3 * L1
            + 11.4e-3 * L7
            + 1.9e-5 * L1 * L7)
    
    # return np.column_stack([f_alpha, f_beta, f_gamma]), np.full((solutions.shape[0],1), -1)
    return np.column_stack([f_alpha, f_beta, f_gamma]), -np.column_stack([f_alpha, f_beta, f_gamma])


class HIR_deflection(Problem):

    def __init__(self,
                 psize = 100,
                 max_evals = 20000):

        super().__init__(function = hir,
                            n_vars = 6,
                            n_obj = 3,
                            n_constr = 0,
                            psize = psize,
                            max_evals = max_evals,
                            bounds = np.array([[130,210],
                                               [30,90],[30,90],
                                               [40,80],[40,80],
                                               [200,250]]),
                            minmax = np.array([1,1,1]))
