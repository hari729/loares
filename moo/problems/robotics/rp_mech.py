
import numpy as np
from opti.core.problem import Problem

def rpmech(population):

    b, lambda_B, lambda_L = population.T
    lambda_e = lambda_B - 0.15
    lambda_l1 = 0.6
    lambda_l2 = 1.05

    N = population.shape[0]
    f = np.zeros([N, 6])
    g = -np.ones([N, 1])

    etaW_mean = (
        -3432.52053485
        - 6.21861267*b
        + 13908.56659919*lambda_B
        + 1739.85437824*lambda_L
        + 0.09077242*b**2
        - 23574.95864703*lambda_B**2
        - 2176.21861868*lambda_L**2
        + 0.00702898*b*lambda_B
        - 0.00239827*b*lambda_L
        - 2.57075884*lambda_B*lambda_L
        - 0.00058842*b**3
        + 17733.02708304*lambda_B**3
        + 1209.69560550*lambda_L**3
        + 1.429e-6*b**4
        - 4993.83500365*lambda_B**4
        - 251.74415634*lambda_L**4
    )

    etaW_std = (
        -0.63418796
        + 0.0000213*b
        + 1.02576783*lambda_B
        - 0.10525385*lambda_L
    )

    V1T2R = (
        49582134.03146700
        - 141997.43299023*b
        - 209863908.10123200*lambda_B
        + 1134167.69712162*lambda_L
        + 2197.88225988*b**2
        + 355404454.09198600*lambda_B**2
        - 2092057.70252440*lambda_L**2
        - 996.75469469*b*lambda_B
        - 460.91139663*b*lambda_L
        + 119185.17454121*lambda_B*lambda_L
        - 14.94064098*b**3
        - 266637521.17325500*lambda_B**3
        + 1481074.86183301*lambda_L**3
        + 0.038056668*b**4
        + 74751229.37562560*lambda_B**4
        - 374138.36239284*lambda_L**4
    )

    S = (
        68898.97051581
        - 445.58791653*b
        + 1358.44806545*lambda_B
        - 100196.89445736*lambda_L
        + 1.44761511*b**2
        - 1545.32830287*lambda_B**2
        + 39740.10716568*lambda_L**2
        + 0.94212994*b*lambda_B
        + 322.89733917*b*lambda_L
        - 83.71115823*lambda_B*lambda_L
        - 0.00191225*b**3
        + 590.84556373*lambda_B**3
        - 2554.59470115*lambda_L**3
    )

    Fmax = (
        -2896.84517833
        + 2.26246212*b
        - 287.80447259*lambda_B
        + 9365.17110769*lambda_L
        - 0.00088772*b**2
        + 288.68258499*lambda_B**2
        - 11107.70189219*lambda_L**2
        - 0.31908913*b*lambda_B
        + 0.05912618*b*lambda_L
        + 62.02176312*lambda_B*lambda_L
        + 3.855e-6*b**3
        - 179.95070708*lambda_B**3
        + 5881.11713850*lambda_L**3
        - 4.355e-9*b**4
        + 44.99995029*lambda_B**4
        - 1182.03132555*lambda_L**4
    )

    v_hat = (
        0.03207663
        + 9.921e-7*b
        - 0.13237850*lambda_B
        - 1.489e-8*b**2
        + 0.22002280*lambda_B**2
        - 3.857e-9*b*lambda_B
        + 9.9219e-11*b**3
        - 0.16016749*lambda_B**3
        - 2.471e-13*b**4
        + 0.04540062*lambda_B**4
    )

    f[:, 0] = -etaW_mean
    f[:, 1] =  etaW_std
    f[:, 2] = -V1T2R
    f[:, 3] = -S
    f[:, 4] = -Fmax
    f[:, 5] =  v_hat

    return f, g


class RPMechanismProblem(Problem):
    def __init__(self, psize=100, max_evals=10000):
        bounds = np.array([
            [80, 120],
            [0.75, 1.05],
            [1.10, 1.30],
        ])

        minmax = np.array([-1, 1, -1, -1, -1, 1])

        super().__init__(function=rpmech,
                         n_vars=3,
                         n_obj=6,
                         n_constr=1,
                         psize=psize,
                         max_evals=max_evals,
                         bounds=bounds,
                         minmax=minmax)
