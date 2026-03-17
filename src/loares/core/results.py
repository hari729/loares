import h5py
import json
import numpy as np


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


class ResultProcessor:
    @staticmethod
    def open(path, problem_info, algo_info, seed):
        f = h5py.File(path, "w")
        meta = f.create_group("metadata")
        meta.attrs["problem_info_json"] = json.dumps(problem_info)
        meta.attrs["algorithm_info_json"] = json.dumps(algo_info)
        meta.attrs["seed"] = int(seed)
        f.create_group("function_evals")
        return f

    @staticmethod
    def write_snapshot(handle, population, evals):
        grp = handle["function_evals"].create_group(f"{evals:06d}")
        grp.create_dataset("X", data=population.solutions)
        grp.create_dataset("F", data=population.objectives)
        grp.create_dataset("G", data=population.constraints)

    @staticmethod
    def write_final(handle, final_dict):
        handle.attrs["final_dict_json"] = json.dumps(final_dict, default=_json_default)

    @staticmethod
    def close(handle):
        handle.close()

    @staticmethod
    def stream_metrics(hdf5_path, metrics_fn, TF=None):
        with h5py.File(hdf5_path, "r") as f:
            fe = f["function_evals"]
            for ek in sorted(fe.keys(), key=lambda k: int(k)):
                grp = fe[ek]
                F = grp["F"][:]
                evals = int(ek)
                metrics = metrics_fn(F, TF)
                yield evals, metrics

    @staticmethod
    def read_final_population(hdf5_path):
        from loares.core.population import Population

        with h5py.File(hdf5_path, "r") as f:
            fe = f["function_evals"]
            last_key = sorted(fe.keys(), key=lambda k: int(k))[-1]
            grp = fe[last_key]
            return Population(grp["X"][:], grp["F"][:], grp["G"][:])

    @staticmethod
    def read_final_dict(hdf5_path):
        with h5py.File(hdf5_path, "r") as f:
            return json.loads(f.attrs["final_dict_json"])

    @staticmethod
    def read_seed(hdf5_path):
        with h5py.File(hdf5_path, "r") as f:
            return int(f["metadata"].attrs["seed"])

    @staticmethod
    def read_metadata(hdf5_path):
        with h5py.File(hdf5_path, "r") as f:
            meta = f["metadata"]
            problem_info = json.loads(meta.attrs["problem_info_json"])
            algo_info = json.loads(meta.attrs["algorithm_info_json"])
            seed = int(meta.attrs["seed"])
            return problem_info, algo_info, seed
