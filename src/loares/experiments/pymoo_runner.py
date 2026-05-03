"""
Factory-based experiment runner for pymoo algorithms.

Takes a pymoo Problem directly and an algorithm factory callable.
No loares-internal conversions — works entirely within pymoo's ecosystem.
HDF5 output is compatible with post_process.
"""

import inspect
import json
from math import comb
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from pymoo.optimize import minimize


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


class AlgoFactory:
    """
    Picklable algorithm factory — works with multiprocessing.

    For stock pymoo classes (no .name), sets name from class name or explicit arg.
    For ModularAlgorithm subclasses (already have .name), passes through.

    Usage:
        AlgoFactory(NSGA2, pop_size=100)
        AlgoFactory(MO_BMR, pop_size=50)
        AlgoFactory(MOEAD, name="MOEAD", ref_dirs=ref_dirs, pop_size=len(ref_dirs))
    """

    def __init__(self, cls, name=None, **kwargs):
        self.cls = cls
        self._name = name
        self.kwargs = kwargs

    def __call__(self):
        algo = self.cls(**self.kwargs)
        if self._name is not None:
            algo.name = self._name
        elif not hasattr(algo, "name") or callable(getattr(algo, "name")):
            algo.name = self.cls.__name__.replace("_", "-")
        return algo


def get_das_dennis_partitions(n_obj, target_psize):
    for p in range(1, 1000):
        n_points = comb(p + n_obj - 1, n_obj - 1)
        if n_points >= target_psize:
            return p
    return p


class ExperimentRunner:
    """
    Runs a pymoo algorithm across multiple seeds and writes HDF5 results.

    Parameters
    ----------
    problem : pymoo Problem
        The optimization problem.
    algo_factory : callable
        Returns a fully configured pymoo Algorithm instance.
        Must set .name on the algorithm (or use pymoo_factory helper).
    max_evals : int
        Maximum function evaluations per run.
    test_name : str
        Name for the output directory structure.
    TF : np.ndarray or None
        True Pareto front for metric computation.
    """

    def __init__(self, problem, algo_factory, max_evals, test_name, TF=None):
        self.problem = problem
        self.algo_factory = algo_factory
        self.max_evals = max_evals
        self.test_name = test_name
        self.TF = TF

        probe = algo_factory()
        self.algo_name = getattr(probe, "name", probe.__class__.__name__.replace("_", "-"))
        self.pop_size = probe.pop_size

        self.problem_info = self._build_problem_info()
        self.algorithm_info = {"name": self.algo_name}

        caller_dir = Path(inspect.stack()[1].filename).resolve().parent
        self.output_dir = (
            caller_dir
            / test_name
            / "raw_data"
            / self.algo_name
            / f"{self.pop_size}-{self.max_evals}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_problem_info(self):
        p = self.problem
        bounds = "None"
        if p.xl is not None and p.xu is not None:
            bounds = str(np.column_stack([p.xl, p.xu]).tolist())
        return {
            "name": p.__class__.__name__.replace("_", "-"),
            "n_obj": int(p.n_obj),
            "n_vars": int(p.n_var),
            "n_constr": int(getattr(p, "n_ieq_constr", 0) + getattr(p, "n_eq_constr", 0)),
            "bounds": bounds,
            "psize": self.pop_size,
            "max_evals": self.max_evals,
            "minmax": getattr(p, "minmax", np.ones(p.n_obj)).tolist(),
        }

    def run(self, seed):
        algorithm = self.algo_factory()
        res = minimize(
            self.problem,
            algorithm,
            ("n_eval", self.max_evals),
            seed=int(seed),
            save_history=True,
        )
        hdf5_path = self.output_dir / f"seed_{int(seed):03d}.h5"
        self._write_h5(seed, res, hdf5_path)

    def _write_h5(self, seed, res, hdf5_path):
        with h5py.File(hdf5_path, "w") as h5:
            meta = h5.create_group("metadata")
            meta.attrs["problem_info_json"] = json.dumps(self.problem_info)
            meta.attrs["algorithm_info_json"] = json.dumps(self.algorithm_info)
            meta.attrs["seed"] = int(seed)
            fe = h5.create_group("function_evals")

            X, F, G = None, None, None
            for algo in res.history:
                source = algo.archive if algo.archive is not None and len(algo.archive) > 0 else algo.opt

                G_raw = source.get("G")
                if G_raw is not None and G_raw.shape[1] > 0:
                    feasible = np.all(G_raw <= 0, axis=1)
                else:
                    feasible = np.ones(len(source), dtype=bool)

                X = source.get("X")[feasible]
                F = source.get("F")[feasible]
                G = G_raw[feasible] if G_raw is not None else np.full((feasible.sum(), 1), -1)

                grp = fe.create_group(f"{algo.evaluator.n_eval:06d}")
                grp.create_dataset("X", data=X)
                grp.create_dataset("F", data=F)
                grp.create_dataset("G", data=G)

            if X is not None:
                h5.attrs["final_dict_json"] = json.dumps(
                    self._make_final_dict(X, F, G), default=_json_default
                )

    @staticmethod
    def _make_final_dict(X, F, G):
        labels = (
            [f"x{i + 1}" for i in range(X.shape[1])]
            + [f"f{j + 1}" for j in range(F.shape[1])]
            + [f"g{k + 1}" for k in range(G.shape[1])]
        )
        combined = np.hstack([X, F, G])
        return {name: combined[:, idx] for idx, name in enumerate(labels)}

    def multi_run(self, seeds, threads=5):
        print(
            f"\nOptimizing {self.problem_info['name']} using {self.algo_name}"
            f" | Pop: {self.pop_size} | Max Evals: {self.max_evals} | Runs: {len(seeds)}"
        )
        with Pool(processes=threads) as pool:
            pool.map(self.run, seeds)
        self._save_info(seeds)
        print(f"Results saved to {self.output_dir}")

    def _save_info(self, seeds):
        info = {
            "Problem": self.problem_info,
            "Algorithm": self.algorithm_info,
            "seeds": str(list(seeds)),
        }
        info_path = self.output_dir / "Info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2, default=_json_default)
