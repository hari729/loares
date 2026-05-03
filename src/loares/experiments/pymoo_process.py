"""
Post-processing for pymoo experiment results.

Reads HDF5 seed files produced by ExperimentRunner, computes metrics,
generates reference fronts, convergence histories, and Pareto plots.

No dependency on old loares pipeline — reads problem_info directly
from HDF5/Info.json metadata. Metric indicator objects are created once
and reused across all snapshots.
"""

import json
import os
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from pymoo.indicators.hv import HV
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.util.normalization import normalize

from pymoo.core.population import Population as PymooPopulation
from pymoo.core.problem import Problem as PymooProblem

from loares.operators.sorting import NDSFarthestPointSurvival
from loares.experiments.utils import dict_to_csv
from loares.experiments.plots import (
    multi_line_plot,
    plot_2d,
    plot_3d,
    parallel_coordinates_plot,
)


# ── HDF5 readers (inline, no ResultProcessor dependency) ──


def _read_metadata(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        meta = f["metadata"]
        return (
            json.loads(meta.attrs["problem_info_json"]),
            json.loads(meta.attrs["algorithm_info_json"]),
            int(meta.attrs["seed"]),
        )


def _read_seed(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        return int(f["metadata"].attrs["seed"])


def _read_final_dict(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        return json.loads(f.attrs["final_dict_json"])


def _read_final_arrays(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        fe = f["function_evals"]
        last_key = sorted(fe.keys(), key=lambda k: int(k))[-1]
        grp = fe[last_key]
        return grp["X"][:], grp["F"][:], grp["G"][:]


def _stream_snapshots(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        fe = f["function_evals"]
        for ek in sorted(fe.keys(), key=lambda k: int(k)):
            yield int(ek), fe[ek]["F"][:]


# ── Metrics ──


class MOOMetrics:
    """
    Multi-objective metrics calculator. Creates indicator objects once,
    reuses across all calls.

    Parameters
    ----------
    true_front : np.ndarray or None
        Reference Pareto front for GD/IGD. If None, only HV and SPC computed.
    n_obj : int
        Number of objectives.
    """

    def __init__(self, true_front, n_obj):
        self.ref_point = np.ones(n_obj) + 0.1
        self._hv = HV(ref_point=self.ref_point)
        self._spacing = SpacingIndicator()
        self.true_front = true_front

        if true_front is not None:
            self._fmax = true_front.max(axis=0)
            self._fmin = true_front.min(axis=0)
            self._tf_norm = normalize(true_front, self._fmin, self._fmax)
            self._gd = GD(self._tf_norm)
            self._igd = IGD(self._tf_norm)
        else:
            self._fmax = None
            self._fmin = None
            self._tf_norm = None
            self._gd = None
            self._igd = None

    def __call__(self, F):
        if F.shape[0] == 0:
            return {"HV": np.nan, "SPC": np.nan, "GD": np.nan, "IGD": np.nan}

        metrics = {}

        if self._tf_norm is not None:
            F_norm = normalize(F, self._fmin, self._fmax)
            metrics["GD"] = float(self._gd(F_norm))
            metrics["IGD"] = float(self._igd(F_norm))
            metrics["SPC"] = float(self._spacing(F_norm)) if F.shape[0] > 1 else np.nan
            metrics["HV"] = float(self._hv(F_norm))
        else:
            fmax = F.max(axis=0)
            fmin = F.min(axis=0)
            denom = fmax - fmin
            denom[denom < 1e-12] = 1.0
            F_norm = (F - fmin) / denom
            metrics["SPC"] = float(self._spacing(F_norm)) if F.shape[0] > 1 else np.nan
            metrics["HV"] = float(self._hv(F_norm))

        return metrics


class SOOMetrics:
    """Single-objective metrics. Stateless — no setup needed."""

    def __call__(self, F):
        return {"best": float(F.min()), "worst": float(F.max())}


# ── Reference front generation ──


def _generate_reference_front(seed_files, rf_size, n_vars, n_obj, n_constr, survival):
    """Merge final populations from all seeds, apply survival operator, return non-dominated."""

    class _DummyProblem(PymooProblem):
        def __init__(self, n_var, n_obj, n_ieq_constr):
            super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr)
        def _evaluate(self, x, out, *args, **kwargs):
            pass

    all_X, all_F, all_G = [], [], []
    for sf in tqdm(seed_files, desc="Loading populations"):
        X, F, G = _read_final_arrays(sf)
        if G.shape[1] == 0:
            G = np.full((X.shape[0], 1), -1.0)
        all_X.append(X)
        all_F.append(F)
        all_G.append(G)

    pop = PymooPopulation.new(
        "X", np.vstack(all_X),
        "F", np.vstack(all_F),
        "G", np.vstack(all_G),
    )

    dummy = _DummyProblem(n_var=n_vars, n_obj=n_obj, n_ieq_constr=max(n_constr, 1))
    survivors = survival.do(dummy, pop, n_survive=rf_size)

    ndf = survivors[survivors.get("rank") == 0]
    return ndf.get("F")


# ── Score sort for Pareto display ──


def _score_sort(data, n_obj):
    f_keys = [f"f{i+1}" for i in range(n_obj)]
    objectives = np.column_stack([np.array(data[k]) for k in f_keys])
    mins = objectives.min(axis=0)
    maxs = objectives.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    score = ((maxs - objectives) / ranges).sum(axis=1)
    order = np.argsort(-score)
    for key in data:
        data[key] = np.array(data[key])[order]
    return data


# ── Main class ──


class PostProcess:
    """
    Post-process pymoo experiment results.

    Auto-discovers problem metadata from HDF5 files. Creates metric
    indicator objects once and reuses them.

    Parameters
    ----------
    test_dir : str or Path
        Path to the raw_data directory containing algorithm subdirectories.
    algo_grps : dict
        Algorithm grouping for convergence plots.
        e.g. {"BMR": ["MO-BMR"], "BWR": ["MO-BWR"], "common": ["NSGA2"]}
    true_front : np.ndarray or None
        Known true Pareto front. If None and gen_rf=True, one is generated.
    gen_rf : bool
        Generate reference front from all seed populations.
    rf_size : int
        Max points in generated reference front.
    rf_survival : pymoo Survival or None
        Survival operator for reference front generation. Defaults to
        NDSFarthestPointSurvival. Any pymoo Survival works (e.g. RankAndCrowding).
    plot_convergence : bool
        Generate convergence plots per metric per algo group.
    plot_pareto : bool
        Generate Pareto front plots for best seed per algorithm.
    pcid : int
        Color index for Pareto plots.
    """

    def __init__(
        self,
        test_dir,
        algo_grps,
        true_front=None,
        gen_rf=False,
        rf_size=1000,
        rf_survival=None,
        plot_convergence=True,
        plot_pareto=True,
        pcid=1,
    ):
        self.test_dir = Path(test_dir)
        self.algo_grps = algo_grps
        self.true_f = true_front
        self.gen_rf = gen_rf
        self.rf_size = rf_size
        self.rf_survival = rf_survival or NDSFarthestPointSurvival()
        self.plot_convergence = plot_convergence
        self.plot_pareto = plot_pareto
        self.pcid = pcid

        self.problem_info = self._discover_problem_info()
        self.n_obj = self.problem_info["n_obj"]
        self.minmax = np.array(self.problem_info.get("minmax", np.ones(self.n_obj)))

        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.result_dir = self.test_dir.parent / f"analysis-{self.timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)

        if self.n_obj > 1:
            self.recording_interval = 0.05
            self.control_metric = "HV"
        else:
            self.recording_interval = 0.005
            self.control_metric = "best"

        self._per_algo_accumulator = []

        print(
            f"\nPost-processing {self.problem_info['name']}"
            f" | n_obj={self.n_obj} | gen_rf={self.gen_rf}"
        )

    def _discover_problem_info(self):
        for algo_dir in self.test_dir.iterdir():
            if not algo_dir.is_dir():
                continue
            for config_dir in algo_dir.iterdir():
                info_path = config_dir / "Info.json"
                if info_path.exists():
                    info = json.loads(info_path.read_text())
                    return info["Problem"]
                seed_files = sorted(config_dir.glob("seed_*.h5"))
                if seed_files:
                    pinfo, _, _ = _read_metadata(seed_files[0])
                    return pinfo
        raise FileNotFoundError(f"No seed files or Info.json found in {self.test_dir}")

    def _discover_configs(self):
        """Find all {algo}/{psize}-{max_evals} directories."""
        configs = {}
        for algo_dir in sorted(self.test_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
            for config_dir in sorted(algo_dir.iterdir()):
                if not config_dir.is_dir():
                    continue
                seed_files = sorted(config_dir.glob("seed_*.h5"))
                if not seed_files:
                    continue
                info_path = config_dir / "Info.json"
                if info_path.exists():
                    info = json.loads(info_path.read_text())
                    name = info["Algorithm"]["name"]
                else:
                    _, ainfo, _ = _read_metadata(seed_files[0])
                    name = ainfo["name"]
                    info = {"Problem": self.problem_info, "Algorithm": {"name": name}}
                configs.setdefault(config_dir.parent.name, []).append({
                    "name": name,
                    "path": config_dir,
                    "seed_files": seed_files,
                    "info": info,
                })
        return configs

    def _build_metrics(self):
        if self.n_obj > 1:
            return MOOMetrics(self.true_f, self.n_obj)
        return SOOMetrics()

    def _compute_seed_metrics(self, args):
        hdf5_path, metrics = args
        seed = _read_seed(hdf5_path)
        history = {}
        final = None
        for evals, F in _stream_snapshots(hdf5_path):
            m = metrics(F)
            for key, value in m.items():
                history.setdefault(key, []).append(value)
            history.setdefault("evals", []).append(evals)
            final = m
        if final is not None:
            final["seed"] = seed
        history["seed"] = [seed]
        return history, final

    def _generate_rf(self):
        rf_path = self.result_dir.parent / "ref_front.npy"
        if rf_path.exists():
            print(f"Using existing reference front: {rf_path}")
            self.true_f = np.load(rf_path)
            return

        print("Generating reference front...")
        all_seeds = []
        for algo_dir in self.test_dir.iterdir():
            if not algo_dir.is_dir():
                continue
            for config_dir in algo_dir.iterdir():
                if config_dir.is_dir():
                    all_seeds.extend(sorted(config_dir.glob("seed_*.h5")))

        self.true_f = _generate_reference_front(
            all_seeds, self.rf_size,
            self.problem_info["n_vars"],
            self.problem_info["n_obj"],
            self.problem_info.get("n_constr", 0),
            self.rf_survival,
        )
        np.save(rf_path, self.true_f)
        print(f"Reference front saved: {rf_path} ({len(self.true_f)} points)")

    def run(self, threads=5):
        if self.gen_rf and self.n_obj > 1 and self.true_f is None:
            self._generate_rf()

        metrics = self._build_metrics()
        configs = self._discover_configs()

        by_psize = {}
        for algo_name, config_list in configs.items():
            for config in config_list:
                self._process_config(config, metrics, threads)
                psize = config["info"]["Problem"]["psize"]
                by_psize.setdefault(psize, []).append(config)

        for psize, psize_configs in by_psize.items():
            pop_dir = self.result_dir / str(psize)
            self._plot_convergence_charts(psize_configs, pop_dir)

        self._write_per_algo_csvs()
        return self.result_dir

    def _process_config(self, config, metrics, threads):
        name = config["name"]
        seed_files = config["seed_files"]
        info = config["info"]

        psize = info["Problem"]["psize"]
        max_evals = info["Problem"]["max_evals"]
        pop_dir = self.result_dir / str(psize)
        os.makedirs(pop_dir / "parquets", exist_ok=True)

        print(f"  Processing {name} ({len(seed_files)} seeds)")

        args_list = [(sf, metrics) for sf in seed_files]
        with Pool(processes=threads) as pool:
            rows = pool.map(self._compute_seed_metrics, args_list)

        metrics_list = [h for h, _ in rows]
        final_per_run = [f for _, f in rows]

        # Final metrics CSV
        final_df = pd.DataFrame(final_per_run)
        final_df.to_csv(
            pop_dir / f"{name}-final-metrics.csv",
            index=False, float_format="%.5f",
        )

        # Pareto front plots
        if self.plot_pareto and self.n_obj > 1 and "HV" in final_df.columns:
            self._plot_pareto(name, seed_files, final_df, pop_dir, info)

        # Convergence aggregation
        recording_interval = max(1, int(max_evals * self.recording_interval))
        eval_grid = np.arange(recording_interval, max_evals + 1, recording_interval)

        mean_row = {"name": f"{name} (Mean)", "evals": eval_grid}
        std_row = {"name": f"{name} (Std)", "evals": eval_grid}
        net = {"Psize": psize, "Max-evals": max_evals}
        convergence = {"name": f"{name} (convergence pts)"}

        metric_keys = [k for k in metrics_list[0].keys() if k not in ("seed", "evals")]
        for m in metric_keys:
            interpolated = []
            for r in metrics_list:
                interpolated.append(np.interp(eval_grid, r["evals"], r[m]))
            values = np.array(interpolated, dtype=float)
            mean_row[m] = np.mean(values, axis=0)
            std_row[m] = np.std(values, axis=0)
            convergence[m] = [np.nan, np.nan]
            net[f"{m}(mean)"] = [mean_row[m][-1]]
            net[f"{m}(std)"] = [std_row[m][-1]]

        history_df = pd.DataFrame(mean_row)
        history_df.to_parquet(
            pop_dir / "parquets" / f"{name}-mean-history.parquet",
            engine="pyarrow",
        )

        net_df = pd.DataFrame(net)
        self._per_algo_accumulator.append(
            pd.DataFrame({"Algorithm": [name], **net})
        )

        # Net results CSV (append per psize)
        net_csv = pop_dir / "net-results.csv"
        if net_csv.exists():
            existing = pd.read_csv(net_csv)
            combined = pd.concat([existing, pd.DataFrame({"Algorithm": [name], **net})], ignore_index=True)
            combined.to_csv(net_csv, index=False, float_format="%.5f")
        else:
            pd.DataFrame({"Algorithm": [name], **net}).to_csv(
                net_csv, index=False, float_format="%.5f"
            )

        # Store for convergence plotting
        config["_history"] = history_df
        config["_convergence"] = pd.DataFrame(convergence)

    def _plot_pareto(self, name, seed_files, final_df, pop_dir, info):
        best_idx = np.argmax(final_df["HV"])
        best_seed_file = seed_files[best_idx]
        plot_data = _read_final_dict(best_seed_file)
        plot_data = _score_sort(plot_data, self.n_obj)

        minmax_flat = self.minmax.flatten()
        for j in range(self.n_obj):
            key = f"f{j + 1}"
            if key in plot_data:
                plot_data[key] = np.array(plot_data[key]) * minmax_flat[j]

        _, algo_info, best_seed = _read_metadata(best_seed_file)
        pareto_dir = pop_dir / "pareto_fronts" / name
        os.makedirs(pareto_dir, exist_ok=True)
        dict_to_csv(plot_data, pareto_dir, f"{algo_info['name']}-pareto-front")

        plot_data["name"] = info["Algorithm"]["name"]
        plot_data["seed"] = best_seed

        if self.n_obj == 2:
            plot_2d(plot_data, pareto_dir, cid=self.pcid)
        elif self.n_obj == 3:
            plot_3d(plot_data, pareto_dir, cid=self.pcid)
        elif self.n_obj > 3 and self.true_f is not None:
            ref_actual = self.true_f * minmax_flat
            scale = 1000
            axis_mins = np.floor(np.min(ref_actual, axis=0) * scale) / scale
            axis_maxs = np.ceil(np.max(ref_actual, axis=0) * scale) / scale
            parallel_coordinates_plot(
                plot_data, pareto_dir,
                axis_mins=axis_mins, axis_maxs=axis_maxs,
            )

    def _plot_convergence_charts(self, configs, pop_dir):
        if not self.plot_convergence:
            return

        all_histories = {}
        for config in configs:
            if "_history" in config:
                all_histories[config["name"]] = {
                    "history": config["_history"],
                    "convergence": config["_convergence"],
                }

        if not all_histories:
            return

        sample = next(iter(all_histories.values()))["history"]
        metric_keys = [c for c in sample.columns if c not in ("name", "evals")]

        for grp_name, grp_algos in self.algo_grps.items():
            if grp_name == "common":
                continue
            combined = grp_algos + self.algo_grps.get("common", [])
            available = [a for a in combined if a in all_histories]
            if not available:
                continue
            for m in metric_keys:
                plot_data = {
                    "ydata": [all_histories[a]["history"][m] for a in available],
                    "xdata": [all_histories[a]["history"]["evals"] for a in available],
                    "xlabel": "Function Evaluations",
                    "ylabel": m,
                    "point": [all_histories[a]["convergence"][m] for a in available],
                    "legend": available,
                }
                multi_line_plot(plot_data, pop_dir, f"{m}-{grp_name}")

    def _write_per_algo_csvs(self):
        if not self._per_algo_accumulator:
            return
        combined = pd.concat(self._per_algo_accumulator, ignore_index=True)
        per_algo_dir = self.result_dir / "per-algo"
        os.makedirs(per_algo_dir, exist_ok=True)
        for algo_name, group in combined.groupby("Algorithm"):
            group.to_csv(
                per_algo_dir / f"{algo_name}-net-results.csv",
                index=False, float_format="%.5f",
            )
        print(f"Per-algorithm CSVs: {per_algo_dir}")
