import numpy as np
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions


def get_true_fronts(function_name,n_vars):
    function_name = function_name.lower()
    if any(f"dtlz{i}" in function_name for i in [1, 2, 3, 4]) or "wfg" in function_name:
        ref_dirs = get_reference_directions("das-dennis", 3, n_points=105)
        true_front = get_problem(function_name,n_var=n_vars).pareto_front(ref_dirs)
    elif any(f"dtlz{i}" in function_name for i in [5, 6, 7]):
        true_front = get_problem(function_name,n_var=n_vars).pareto_front()
    elif "zdt" in function_name:
        true_front = get_problem(function_name).pareto_front(100)
    else:
        raise ValueError(f"Unsupported problem: {function_name}")
    return true_front

def benchmark_bounds(function_name,variables=None):
    """Return appropriate bounds for ZDT and DTLZ problems"""
    function_name = function_name.upper()
    if function_name in ["ZDT1", "ZDT2", "ZDT3" ]:
        # For ZDT1, ZDT2, ZDT3, ZDT6: All variables are in [0,1]
        return np.array([[0, 1]] * 30)  # Return bounds for up to 30 variables

    elif function_name == "ZDT6":
        return np.array([[0, 1]] * 10)
        
    elif function_name == "ZDT4":
        bounds = np.zeros((10, 2))
        bounds[0] = [0, 1]  # First variable bounds
        bounds[1:] = [-5, 5]  # Remaining variable bounds
        return bounds
    
    elif "DTLZ" in str(function_name):
        # For all DTLZ problems: All variables are in [0,1]
        return np.array([[0, 1]] * variables)  # Return bounds for up to 30 variables
    elif "WFG" in function_name:
        return np.array([[0,2]]*variables)
    else:
        raise ValueError(f"Unsupported problem: {function_name}")


def zdt1(population):
    F = get_problem("zdt1", n_var=30).evaluate(population)
    return F, np.full((population.shape[0], 1), -1)

def zdt2(population):
    F = get_problem("zdt2", n_var=30).evaluate(population)
    return F, np.full((population.shape[0], 1), -1)

def zdt3(population):
    F = get_problem("zdt3", n_var=30).evaluate(population)
    return F, np.full((population.shape[0], 1), -1)

def zdt4(population):
    F = get_problem("zdt4", n_var=10).evaluate(population)
    return F, np.full((population.shape[0], 1), -1)

def zdt6(population):
    F = get_problem("zdt6", n_var=10).evaluate(population)
    return F, np.full((population.shape[0], 1), -1)

def dtlz1(population):
    F = get_problem("dtlz1", n_var=7, n_obj=3).evaluate(population)
    return F, np.full((population.shape[0], 1), -1)

def dtlz2(population):
    F = get_problem("dtlz2", n_var=12, n_obj=3).evaluate(population)
    return F, np.full((population.shape[0], 1), -1)

def dtlz3(population):
    F = get_problem("dtlz3", n_var=12, n_obj=3).evaluate(population)
    return F, np.full((population.shape[0], 1), -1)

def dtlz4(population):
    F = get_problem("dtlz4", n_var=12, n_obj=3).evaluate(population)
    return F, np.full((population.shape[0], 1), -1)

def dtlz5(population):
    F = get_problem("dtlz5", n_var=12, n_obj=3).evaluate(population)
    return F, np.full((population.shape[0], 1), -1)

def dtlz6(population):
    F = get_problem("dtlz6", n_var=12, n_obj=3).evaluate(population)
    return F, np.full((population.shape[0], 1), -1)

def dtlz7(population):
    F = get_problem("dtlz7", n_var=22, n_obj=3).evaluate(population)
    return F, np.full((population.shape[0], 1), -1)

def wfg1(population):
    F = get_problem("wfg1", n_var=12, n_obj=3).evaluate(population)
    return F, np.full((population.shape[0], 1), -1)

def wfg2(population):
    F = get_problem("wfg2", n_var=12, n_obj=3).evaluate(population)
    return F, np.full((population.shape[0], 1), -1)

def wfg3(population):
    F = get_problem("wfg3", n_var=12, n_obj=3).evaluate(population)
    return F, np.full((population.shape[0], 1), -1)

def wfg4(population):
    F = get_problem("wfg4", n_var=12, n_obj=3).evaluate(population)
    return F, np.full((population.shape[0], 1), -1)

def wfg5(population):
    F = get_problem("wfg5", n_var=12, n_obj=3).evaluate(population)
    return F, np.full((population.shape[0], 1), -1)

def wfg6(population):
    F = get_problem("wfg6", n_var=12, n_obj=3).evaluate(population)
    return F, np.full((population.shape[0], 1), -1)

def wfg7(population):
    F = get_problem("wfg7", n_var=12, n_obj=3).evaluate(population)
    return F, np.full((population.shape[0], 1), -1)

def wfg8(population):
    F = get_problem("wfg8", n_var=12, n_obj=3).evaluate(population)
    return F, np.full((population.shape[0], 1), -1)

def wfg9(population):
    F = get_problem("wfg9", n_var=12, n_obj=3).evaluate(population)
    return F, np.full((population.shape[0], 1), -1)


# --- Final Dictionary with Wrapped Functions ---
get = {
    "zdt1": [zdt1, 30, benchmark_bounds("zdt1", 30), 2, 1, None],
    "zdt2": [zdt2, 30, benchmark_bounds("zdt2", 30), 2, 1, None],
    "zdt3": [zdt3, 30, benchmark_bounds("zdt3", 30), 2, 1, None],
    "zdt4": [zdt4, 10, benchmark_bounds("zdt4", 10), 2, 1, None],
    "zdt6": [zdt6, 10, benchmark_bounds("zdt6", 10), 2, 1, None],
    "dtlz1": [dtlz1, 7, benchmark_bounds("dtlz1", 7), 3, 1, None],
    "dtlz2": [dtlz2, 12, benchmark_bounds("dtlz2", 12), 3, 1, None],
    "dtlz3": [dtlz3, 12, benchmark_bounds("dtlz3", 12), 3, 1, None],
    "dtlz4": [dtlz4, 12, benchmark_bounds("dtlz4", 12), 3, 1, None],
    "dtlz5": [dtlz5, 12, benchmark_bounds("dtlz5", 12), 3, 1, None],
    "dtlz6": [dtlz6, 12, benchmark_bounds("dtlz6", 12), 3, 1, None],
    "dtlz7": [dtlz7, 22, benchmark_bounds("dtlz7", 22), 3, 1, None],
    "wfg1": [wfg1, 12, benchmark_bounds("wfg1", 12), 3, 1, None],
    "wfg2": [wfg2, 12, benchmark_bounds("wfg2", 12), 3, 1, None],
    "wfg3": [wfg3, 12, benchmark_bounds("wfg3", 12), 3, 1, None],
    "wfg4": [wfg4, 12, benchmark_bounds("wfg4", 12), 3, 1, None],
    "wfg5": [wfg5, 12, benchmark_bounds("wfg5", 12), 3, 1, None],
    "wfg6": [wfg6, 12, benchmark_bounds("wfg6", 12), 3, 1, None],
    "wfg7": [wfg7, 12, benchmark_bounds("wfg7", 12), 3, 1, None],
    "wfg8": [wfg8, 12, benchmark_bounds("wfg8", 12), 3, 1, None],
    "wfg9": [wfg9, 12, benchmark_bounds("wfg9", 12), 3, 1, None],
}