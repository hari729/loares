import importlib

def get(problem_name: str):
    """
    Load a problem definition from a module-level 'get' dictionary.
    Example: 'robotics.robot_arm' → problems/robotics.py → get['robot_arm']
    """
    try:
        module_name, key = problem_name.split('.', 1)
    except ValueError:
        raise ValueError(f"Invalid problem name '{problem_name}'. Expected 'module.entry' format.")
    
    module_path = f"{__name__}.{module_name}"  # e.g. 'problems.robotics'
    module = importlib.import_module(module_path)
    
    try:
        return module.get[key]
    except (AttributeError, KeyError):
        raise KeyError(f"Problem '{key}' not found in '{module_path}.get'")


def get_true_front(problem_name: str, n_vars: int):
    """
    Call the correct module's get_true_front(function_name, n_vars)
    Example: problems.get_true_front("robotics.robot_arm", 10)
    """
    try:
        module_name, func_name = problem_name.split('.', 1)
    except ValueError:
        raise ValueError(f"Invalid problem name '{problem_name}'. Expected 'module.entry' format.")

    module_path = f"{__name__}.{module_name}"
    module = importlib.import_module(module_path)

    if not hasattr(module, "get_true_fronts"):
        raise AttributeError(f"Module '{module_path}' has no 'get_true_front' function")

    return module.get_true_fronts(func_name, n_vars)
