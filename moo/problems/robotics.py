import numpy as np

from opti.core.problem import Problem


def mau(population): 
    b,h,l,s = population.T
    f = np.zeros([population.shape[0],2])
    g = np.zeros([population.shape[0],1])

    f[:,0] = (6.4852 + 0.036*l + 0.023*s*s + 0.0025*b*h
                - 0.0007*b*l + 0.0007*h*l - 0.014*h*s
                - 0.0015*l*s)
    f[:,1] = 7.86e-3 * (b - 7) * (h - 3) * (l - 7)

    F = (-63.446 - 1.4887*(h-3)**2 + 1.1434*(b-7)*(h-3)
            + 0.0911*(b-7)*(l-7) + 0.3495*(h-3)*(l-7))
    g[:,0] = 630 - F

    return f,g

class MAU(Problem):
    def __init__(self,
                 psize = 100,
                 max_evals = 30000):

        super().__init__(function = mau,
                            n_vars = 4,
                            n_obj = 2,
                            n_constr = 1,
                            psize = psize,
                            max_evals = max_evals,
                            bounds = np.array([[24.5,45.5],[9,17],[60,110],[0,15]]),
                            minmax = np.array([1,1]))


def drag_lift(population):
    x1,x2,x3,x4 = population.T
    x3 = np.rint(x3)
    # x2 = np.round(x2/10) * 10
    # x4 = np.round(x4/10) * 10
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

    return np.column_stack([-1/drag,1/lift]), np.column_stack([-lift,-drag])

class DRAG_LIFT(Problem):
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
                            minmax = np.array([-1,1]))


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

class AUV_g(Problem):
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
