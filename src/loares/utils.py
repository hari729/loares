import contextlib
import numpy as np
import pathlib
from pathlib import Path
import pandas as pd
import gzip
import pickle
import joblib
from tqdm import tqdm


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def unzip_result(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def get_spec_path(spec):
    return Path(
        spec["problem_name"]
        + f"/{spec['solver_kwargs']['termination'][0]}-{spec['solver_kwargs']['termination'][1]}/"
        + spec["algorithm_name"]
        + f"/{spec['algorithm_kwargs']['pop_size']}"
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
        "save_history": spec["solver_kwargs"]["save_history"],
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


def update_manifest(dir_path, new_rows, spec_key_cols, manifest_name=""):
    dir_path.mkdir(parents=True, exist_ok=True)
    existing_path = dir_path / f"{manifest_name}_manifest.csv"
    df = pd.read_csv(existing_path) if existing_path.exists() else pd.DataFrame()
    new_df = pd.DataFrame(new_rows)
    if not df.empty:
        mask = (
            df[spec_key_cols]
            .apply(tuple, axis=1)
            .isin(set(new_df[spec_key_cols].apply(tuple, axis=1)))
        )
        df = df[~mask]
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(existing_path, index=False)
