from pathlib import Path
import pickle
import gzip
from joblib import Parallel, delayed
from pymoo.optimize import minimize
from loares.utils import get_spec_path, get_spec_info, update_manifest


def pending_specs(spec_list, output_dir, overwrite=False):
    if overwrite:
        return spec_list
    return [
        spec
        for spec in spec_list
        if not Path(Path(output_dir) / get_spec_path(spec) / "result.pkl.gz").exists()
    ]


def single_run(spec, output_dir):
    res_path = Path(output_dir) / get_spec_path(spec)
    res_path.mkdir(parents=True, exist_ok=True)
    spec_info = get_spec_info(spec)
    spec_info["result_path"] = res_path / "result.pkl.gz"
    spec_info["error"] = ""
    try:
        res = minimize(
            spec["problem"],
            spec["algorithm"](**spec["algorithm_kwargs"]),
            **spec["solver_kwargs"],
        )
    except Exception as e:
        spec_info["error"] = f"{type(e).__name__}: {e}"
        return spec_info
    with gzip.open(res_path / "result.pkl.gz", "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    return spec_info


def parallel_run(spec_list, output_dir, n_jobs, overwrite=False):
    specs_to_run = pending_specs(spec_list, output_dir, overwrite)
    status = Parallel(n_jobs=n_jobs)(
        delayed(single_run)(spec, output_dir) for spec in specs_to_run
    )
    if status:
        update_manifest(output_dir, status, ["result_path"], "run")
