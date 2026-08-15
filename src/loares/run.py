from multiprocessing import Pool
from pathlib import Path
from pymoo.optimize import minimize

from loares.utils import write_results, get_spec_path


def single_run(spec):
    res = minimize(
        spec["problem"],
        spec["algorithm"],
        **spec["solver_kwargs"],
        save_history=True,
    )
    write_results(res, spec)


def parallel_run(spec_list, n_threads, overwrite=False):
    if not overwrite:
        pending_specs = [
            spec
            for spec in spec_list
            if not get_spec_path(spec).with_suffix(".h5").exists()
        ]
    else:
        pending_specs = spec_list
    with Pool(processes=n_threads) as pool:
        pool.map(single_run, pending_specs)
