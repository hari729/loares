from multiprocessing import Pool
from pathlib import Path
from pymoo.optimize import minimize
import pickle
import gzip
from loares.utils import get_spec_path


def single_run(spec):
    res = minimize(
        spec["problem"],
        spec["algorithm"],
        **spec["solver_kwargs"],
    )

    res_path = get_spec_path(spec)
    res_path.mkdir(parents=True, exist_ok=True)
    with gzip.open(res_path / "result.pkl.gz", "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)


def pending_specs(spec_list, overwrite=False):
    if overwrite:
        return spec_list
    return [
        spec
        for spec in spec_list
        if not (get_spec_path(spec) / "result.pkl.gz").exists()
    ]


def parallel_run(spec_list, n_threads, overwrite=False):
    specs_to_run = pending_specs(spec_list, overwrite)
    with Pool(processes=n_threads) as pool:
        pool.map(single_run, specs_to_run)
