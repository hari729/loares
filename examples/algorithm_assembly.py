import numpy as np

from loares.core.problem import Problem, ProblemHandler
from loares.core.update import UpdateRule
from loares.core.population import PopulationHandler
from loares.core.flow import FlowHandler
from loares.core.initializer import lhs_initialize

from loares.base.bmr import bmr
from loares.base.bwr import bwr
from loares.base.bmwr import bmwr
from loares.base.mutation import random_reinit

from loares.algorithms.soo.selection import bw_selection
from loares.algorithms.soo.sorting import bw_sorting

from loares.algorithms.moo.selection import random_bw_selection
from loares.algorithms.moo.sorting import ranking_crowding
from loares.algorithms.moo.mods import local_search


def sphere(X):
    F = np.sum(X**2, axis=1, keepdims=True)
    G = np.zeros((X.shape[0], 1))
    return F, G


def zdt1(X):
    n = X.shape[1]
    f1 = X[:, 0]
    g = 1 + 9 * np.sum(X[:, 1:], axis=1) / (n - 1)
    f2 = g * (1 - np.sqrt(f1 / g))
    F = np.column_stack([f1, f2])
    G = np.zeros((X.shape[0], 1))
    return F, G


# ---------- SOO assembly ----------
# 1. Define the problem
soo_problem = Problem(
    function=sphere,
    n_vars=10,
    n_obj=1,
    n_constr=0,
    psize=50,
    max_evals=5000,
    bounds=np.column_stack([np.full(10, -5.12), np.full(10, 5.12)]),
    minmax=["min"],
)

# 2. Wrap in a ProblemHandler (tracks evals, recording intervals)
soo_ph = ProblemHandler(soo_problem)

# 3. Build an UpdateRule from selection + base function + mutation
soo_update_rule = UpdateRule(bw_selection, bmr, random_reinit)

# 4. Build a PopulationHandler with a sorting function
soo_pop_handler = PopulationHandler(bw_sorting)

# 5. Assemble FlowHandler (no mods for SOO)
soo_algo = FlowHandler(soo_ph, soo_update_rule, soo_pop_handler, Mods=[])

print("SOO algorithm assembled:")
print(f"  Algorithm info: {soo_algo.get_info()}")
print(f"  UpdateRule info: {soo_update_rule.get_info()}")
print(f"  Problem info: {soo_problem.get_info()}")

# Run it
soo_algo.run(seed=42, hdf5_path="/tmp/soo_assembly_test.h5")
print(f"SOO run complete — {soo_ph.evals} evaluations used\n")


# ---------- MOO assembly ----------
# 1. Define the problem
moo_problem = Problem(
    function=zdt1,
    n_vars=30,
    n_obj=2,
    n_constr=0,
    psize=100,
    max_evals=10000,
    bounds=np.column_stack([np.zeros(30), np.ones(30)]),
    minmax=["min", "min"],
)

# 2. Wrap in a ProblemHandler
moo_ph = ProblemHandler(moo_problem)

# 3. Build an UpdateRule (MOO uses random_bw_selection from top/bottom 10%)
moo_update_rule = UpdateRule(random_bw_selection, bwr, random_reinit)

# 4. Build a PopulationHandler with ranking+crowding, using LHS initialization
moo_pop_handler = PopulationHandler(ranking_crowding, initializer=lhs_initialize)

# 5. Assemble FlowHandler with local_search mod
moo_algo = FlowHandler(moo_ph, moo_update_rule, moo_pop_handler, Mods=[local_search])

print("MOO algorithm assembled:")
print(f"  Algorithm info: {moo_algo.get_info()}")
print(f"  UpdateRule info: {moo_update_rule.get_info()}")
print(f"  Problem info: {moo_problem.get_info()}")

# Run it
moo_algo.run(seed=42, hdf5_path="/tmp/moo_assembly_test.h5")
print(f"MOO run complete — {moo_ph.evals} evaluations used")
