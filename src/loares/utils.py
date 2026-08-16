import numpy as np
import pathlib
from pathlib import Path

import gzip
import pickle


def unzip_result(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def get_spec_path(spec):
    return Path(
        spec["problem_name"]
        + f"/{spec['solver_kwargs']['termination'][0]}-{spec['solver_kwargs']['termination'][1]}/"
        + spec["algorithm_name"]
        + f"/{spec['algorithm'].pop_size}"
        + f"/seed_{int(spec['solver_kwargs']['seed']):03d}"
    )


def get_problem_info(spec):
    p = spec["problem"]
    bounds = "None"
    if p.xl is not None and p.xu is not None:
        bounds = str(np.column_stack([p.xl, p.xu]).tolist())
    return {
        "name": spec["problem_name"],
        "n_obj": int(p.n_obj),
        "n_vars": int(p.n_var),
        "n_constr": int(getattr(p, "n_ieq_constr", 0) + getattr(p, "n_eq_constr", 0)),
        "bounds": bounds,
        "minmax": getattr(p, "minmax", np.ones(p.n_obj)).tolist(),
    }


def get_spec_info(spec):
    return {
        "algorithm_name": spec["algorithm_name"],
        "problem_name": spec["problem_name"],
        "pop_size": spec["algorithm_kwargs"]["pop_size"],
        "seed": spec["solver_kwargs"]["seed"],
        "termination_metric": spec["solver_kwargs"]["termination"][0],
        "termination_value": spec["solver_kwargs"]["termination"][1],
    }


def json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pathlib.PosixPath,)):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")
