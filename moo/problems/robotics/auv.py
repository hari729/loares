import numpy as np 
from opti.core.problem import Problem 


BIG = 1e6  # penalty constant

def safe_log(x):
    out = np.empty_like(x)
    mask = x > 0
    out[mask] = np.log(x[mask])
    out[~mask] = BIG  # invalid -> huge -> arctan(BIG) ~ π/2
    return out

def safe_arccos(x):
    out = np.empty_like(x)
    mask = (x >= -1) & (x <= 1)
    out[mask] = np.arccos(x[mask])
    out[~mask] = np.pi  # invalid -> max penalty
    return out

def safe_arcsin(x):
    out = np.empty_like(x)
    mask = (x >= -1) & (x <= 1)
    out[mask] = np.arcsin(x[mask])
    out[~mask] = 0.0  # invalid -> small g -> large 1/g
    return out

def auv_gep(population):
    x1, x2, x3, x4 = population.T
    
    # Resistance surrogate
    f = (10**(np.arctan(safe_arccos(np.maximum((1.6126-x4)*1.8852, np.tan(x1)))))
         + np.cos(x1*2.0651)*x3
         + 1/x2
         + np.maximum(x4, -3.0430)
         + np.arctan(safe_log(5.4558 - np.tan(np.tan(8.4771 + x4)))**2))
    
    # Volume surrogate
    v = (np.cbrt(np.exp(np.cbrt(np.cos(x3)*(1/-9.5433 - x2)))**2)
         + np.cbrt(np.minimum(np.arctan(1/(8.6543-x4-((x3+x1)/2))),
                              np.arctan(1/(x3+6.6160))))
         + np.cbrt(safe_arcsin(np.maximum(0.1969, x1)*0.1174**2
                               - np.minimum((x1+x2)/2, x2))))
    
    # Clamp g to avoid division by zero
    v = np.clip(v, 1e-8, None)
    
    return np.column_stack([f, 1/v]), np.full((population.shape[0], 1), -1)

class AUV_gep(Problem):
    def __init__(self,
                 psize = 200,
                 max_evals = 80000):

        super().__init__(function = auv_gep,
                            n_vars = 4,
                            n_obj = 2,
                            n_constr = 0,
                            psize = psize,
                            max_evals = max_evals,
                            bounds = np.array([[0.148,0.223],[0.185,0.285],[1.5,4],[1.5,3]]),
                            minmax = np.array([1,1]))
