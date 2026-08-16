from pathlib import Path
import pickle
import gzip
from joblib import Parallel, delayed
from pymoo.optimize import minimize
from loares.utils import get_spec_path


def single_run(spec, output_dir):
    res = minimize(
        spec["problem"],
        spec["algorithm"],
        **spec["solver_kwargs"],
    )
    res_path = Path(output_dir) / get_spec_path(spec)
    res_path.mkdir(parents=True, exist_ok=True)
    with gzip.open(res_path / "result.pkl.gz", "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)


def pending_specs(spec_list, output_dir, overwrite=False):
    if overwrite:
        return spec_list
    return [
        spec
        for spec in spec_list
        if not Path(Path(output_dir) / get_spec_path(spec) / "result.pkl.gz").exists()
    ]


def parallel_run(spec_list, output_dir, n_jobs, overwrite=False):
    specs_to_run = pending_specs(spec_list, output_dir, overwrite)
    Parallel(n_jobs=n_jobs)(
        delayed(single_run)(spec, output_dir) for spec in specs_to_run
    )
