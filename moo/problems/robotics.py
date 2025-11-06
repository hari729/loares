import numpy as np

from core.problem import Problem


def mau_funciton(population): 
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

mau = Problem(function = mau_funciton,
              n_vars = 4,
              n_obj = 2,
              n_constr = 1,
              psize = 100,
              max_evals = 30000,
              bounds = np.array([[24.5,45.5],[9,17],[60,110],[0,15]]),
              minmax = np.array([1,1]))
