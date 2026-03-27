import inspect
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import pytest

from loares.core.population import Population
from loares.core.results import ResultProcessor


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def _fake_stack(caller_dir):
    """Return a mock inspect.stack that makes [1].filename point to caller_dir."""
    real_stack = inspect.stack()
    fake_frame = type(real_stack[0])(
        real_stack[0].frame,
        str(caller_dir / "fake_caller.py"),
        real_stack[0].lineno,
        real_stack[0].function,
        real_stack[0].code_context,
        real_stack[0].index,
    )
    # [0] is the callee (runner/process __init__), [1] is the "caller"
    return [real_stack[0], fake_frame] + real_stack[2:]


def _make_population(n=20, n_vars=3, n_obj=2):
    X = np.random.rand(n, n_vars)
    F = np.random.rand(n, n_obj)
    G = np.full((n, 1), -1.0)
    return Population(X, F, G)


# ── Test 1: ResultProcessor roundtrip ────────────────────────────────────────


class TestResultProcessorRoundtrip:
    def test_open_creates_metadata_and_function_evals(self, tmp_dir):
        path = tmp_dir / "test.h5"
        pinfo = {"name": "TestProblem", "n_obj": 2}
        ainfo = {"name": "TestAlgo"}
        h5 = ResultProcessor.open(path, pinfo, ainfo, seed=42)
        assert "metadata" in h5
        assert "function_evals" in h5
        assert h5["metadata"].attrs["seed"] == 42
        assert json.loads(h5["metadata"].attrs["problem_info_json"]) == pinfo
        assert json.loads(h5["metadata"].attrs["algorithm_info_json"]) == ainfo
        ResultProcessor.close(h5)

    def test_write_snapshot_stores_arrays(self, tmp_dir):
        path = tmp_dir / "test.h5"
        h5 = ResultProcessor.open(path, {}, {}, seed=1)
        pop = _make_population(n=15, n_vars=4, n_obj=3)
        ResultProcessor.write_snapshot(h5, pop, evals=500)
        grp = h5["function_evals"]["000500"]
        assert grp["X"].shape == (15, 4)
        assert grp["F"].shape == (15, 3)
        assert grp["G"].shape == (15, 1)
        ResultProcessor.close(h5)

    def test_write_final_stores_json_attr(self, tmp_dir):
        path = tmp_dir / "test.h5"
        h5 = ResultProcessor.open(path, {}, {}, seed=1)
        final = {"x1": [1.0, 2.0], "f1": [0.5, 0.6]}
        ResultProcessor.write_final(h5, final)
        recovered = json.loads(h5.attrs["final_dict_json"])
        assert recovered["x1"] == [1.0, 2.0]
        ResultProcessor.close(h5)

    def test_write_final_handles_numpy_types(self, tmp_dir):
        path = tmp_dir / "test.h5"
        h5 = ResultProcessor.open(path, {}, {}, seed=1)
        final = {
            "x1": np.array([1.0, 2.0]),
            "f1": np.float64(0.5),
            "count": np.int64(10),
            "flag": np.bool_(True),
        }
        ResultProcessor.write_final(h5, final)
        recovered = json.loads(h5.attrs["final_dict_json"])
        assert recovered["x1"] == [1.0, 2.0]
        assert recovered["f1"] == 0.5
        assert recovered["count"] == 10
        assert recovered["flag"] is True
        ResultProcessor.close(h5)

    def test_full_roundtrip(self, tmp_dir):
        path = tmp_dir / "seed_001.h5"
        pinfo = {"name": "ZDT1", "n_obj": 2, "n_vars": 30}
        ainfo = {"name": "MO-BMR"}
        np.random.seed(99)

        h5 = ResultProcessor.open(path, pinfo, ainfo, seed=7)
        pops = []
        eval_points = [100, 200, 300]
        for ev in eval_points:
            pop = _make_population(n=10, n_vars=30, n_obj=2)
            pops.append(pop)
            ResultProcessor.write_snapshot(h5, pop, ev)
        final_dict = {
            "x1": pops[-1].solutions[:, 0].tolist(),
            "f1": pops[-1].objectives[:, 0].tolist(),
        }
        ResultProcessor.write_final(h5, final_dict)
        ResultProcessor.close(h5)

        rp_info, ra_info, rseed = ResultProcessor.read_metadata(path)
        assert rp_info == pinfo
        assert ra_info == ainfo
        assert rseed == 7

        assert ResultProcessor.read_seed(path) == 7

        rd = ResultProcessor.read_final_dict(path)
        assert len(rd["x1"]) == 10

        final_pop = ResultProcessor.read_final_population(path)
        np.testing.assert_array_equal(final_pop.solutions, pops[-1].solutions)
        np.testing.assert_array_equal(final_pop.objectives, pops[-1].objectives)
        np.testing.assert_array_equal(final_pop.constraints, pops[-1].constraints)

    def test_stream_metrics_yields_all_snapshots(self, tmp_dir):
        path = tmp_dir / "seed_001.h5"
        h5 = ResultProcessor.open(path, {}, {}, seed=1)
        for ev in [100, 200, 300]:
            ResultProcessor.write_snapshot(h5, _make_population(n=10, n_obj=2), ev)
        ResultProcessor.write_final(h5, {})
        ResultProcessor.close(h5)

        def dummy_metrics(F, TF):
            return {"mean_f1": float(F[:, 0].mean())}

        results = list(ResultProcessor.stream_metrics(path, dummy_metrics, TF=None))
        assert len(results) == 3
        assert [ev for ev, _ in results] == [100, 200, 300]
        for _, m in results:
            assert "mean_f1" in m

    def test_stream_metrics_sorted_order(self, tmp_dir):
        path = tmp_dir / "seed_001.h5"
        h5 = ResultProcessor.open(path, {}, {}, seed=1)
        for ev in [300, 100, 200]:
            ResultProcessor.write_snapshot(h5, _make_population(n=5, n_obj=2), ev)
        ResultProcessor.write_final(h5, {})
        ResultProcessor.close(h5)

        def dummy_metrics(F, TF):
            return {"v": 1.0}

        evals_seen = [
            ev for ev, _ in ResultProcessor.stream_metrics(path, dummy_metrics)
        ]
        assert evals_seen == [100, 200, 300]


# ── Test 2: FlowHandler.run() produces valid per-seed HDF5 ──────────────────


class TestFlowHandlerRun:
    def test_run_produces_valid_hdf5(self, tmp_dir):
        from loares.core.adapters import pymoo_to_loares_prob
        from pymoo.problems.multi import ZDT1
        from loares.algorithms.bxr.mo_basic import MO_BMR

        bench = ZDT1()
        prob = pymoo_to_loares_prob(bench, psize=20, max_evals=200)
        algo = MO_BMR(prob)

        hdf5_path = tmp_dir / "seed_001.h5"
        algo.run(seed=1, hdf5_path=hdf5_path)

        assert hdf5_path.exists()

        with h5py.File(hdf5_path, "r") as f:
            assert "metadata" in f
            assert "function_evals" in f
            assert f["metadata"].attrs["seed"] == 1

            fe_keys = sorted(f["function_evals"].keys(), key=lambda k: int(k))
            assert len(fe_keys) >= 2

            last_grp = f["function_evals"][fe_keys[-1]]
            assert last_grp["X"].shape[1] == 30
            assert last_grp["F"].shape[1] == 2
            assert last_grp["G"].shape[1] == 1

            final_dict = json.loads(f.attrs["final_dict_json"])
            assert "x1" in final_dict
            assert "f1" in final_dict

    def test_run_soo_produces_valid_hdf5(self, tmp_dir):
        from loares.core.problem import Problem
        from loares.algorithms.bxr.so_basic import SO_BMR

        def sphere(X):
            F = np.sum(X**2, axis=1, keepdims=True)
            G = np.full((X.shape[0], 1), -1.0)
            return F, G

        prob = Problem(
            function=sphere,
            name="Sphere",
            n_vars=5,
            n_obj=1,
            n_constr=0,
            psize=10,
            max_evals=200,
            bounds=np.column_stack([np.full(5, -5), np.full(5, 5)]),
            minmax=["min"],
        )
        algo = SO_BMR(prob)

        hdf5_path = tmp_dir / "seed_001.h5"
        algo.run(seed=1, hdf5_path=hdf5_path)

        assert hdf5_path.exists()
        final_pop = ResultProcessor.read_final_population(hdf5_path)
        assert final_pop.solutions.shape[1] == 5
        assert final_pop.objectives.shape[1] == 1

    def test_stream_metrics_from_run_output(self, tmp_dir):
        from loares.core.adapters import pymoo_to_loares_prob
        from pymoo.problems.multi import ZDT1
        from loares.algorithms.bxr.mo_basic import MO_BMR
        from loares.metrics.moo import raw_performance_metrics

        bench = ZDT1()
        tf = bench.pareto_front(100)
        prob = pymoo_to_loares_prob(bench, psize=20, max_evals=200)
        algo = MO_BMR(prob)

        hdf5_path = tmp_dir / "seed_001.h5"
        algo.run(seed=1, hdf5_path=hdf5_path)

        results = list(
            ResultProcessor.stream_metrics(hdf5_path, raw_performance_metrics, TF=tf)
        )
        assert len(results) >= 2
        for evals, metrics in results:
            assert isinstance(evals, int)
            assert "HV" in metrics
            assert "GD" in metrics
            assert "IGD" in metrics


# ── Test 3: ExperimentRunner produces per-seed HDF5 + Info.json ──────────────


class TestExperimentRunner:
    def test_multi_thread_produces_seed_files_and_info(self, tmp_dir):
        from loares.core.adapters import pymoo_to_loares_prob
        from pymoo.problems.multi import ZDT1
        from loares.algorithms.bxr.mo_basic import MO_BMR
        from loares.experiments.runner import ExperimentRunner

        bench = ZDT1()
        tf = bench.pareto_front(100)
        prob = pymoo_to_loares_prob(bench, psize=20, max_evals=200)

        with patch(
            "loares.experiments.runner.inspect.stack",
            return_value=_fake_stack(tmp_dir),
        ):
            runner = ExperimentRunner(prob, MO_BMR, "test-run", TF=tf)
        seeds = np.array([1, 2, 3])
        runner.multi_thread(seeds, threads=2)

        assert runner.output_dir.exists()
        assert (runner.output_dir / "Info.json").exists()

        seed_files = sorted(runner.output_dir.glob("seed_*.h5"))
        assert len(seed_files) == 3

        for sf in seed_files:
            seed_val = ResultProcessor.read_seed(sf)
            assert seed_val in [1, 2, 3]

        info = json.loads((runner.output_dir / "Info.json").read_text())
        assert "Problem" in info
        assert "Algorithm" in info


# ── Test 4: post_process.run() smoke test ────────────────────────────────────


class TestPostProcess:
    @pytest.fixture
    def setup_raw_data(self, tmp_dir):
        from loares.core.adapters import pymoo_to_loares_prob
        from pymoo.problems.multi import ZDT1
        from loares.algorithms.bxr.mo_basic import MO_BMR, MO_BWR
        from loares.experiments.runner import ExperimentRunner

        bench = ZDT1()
        tf = bench.pareto_front(100)
        prob = pymoo_to_loares_prob(bench, psize=20, max_evals=200)
        seeds = np.array([1, 2, 3])

        for algo_cls in [MO_BMR, MO_BWR]:
            with patch(
                "loares.experiments.runner.inspect.stack",
                return_value=_fake_stack(tmp_dir),
            ):
                runner = ExperimentRunner(prob, algo_cls, "smoke-test", TF=tf)
            runner.multi_thread(seeds, threads=2)

        yield tmp_dir, prob, tf

    def test_run_produces_expected_outputs(self, setup_raw_data):
        tmp_dir, prob, tf = setup_raw_data
        from loares.experiments.process import post_process

        algo_grps = {
            "base": ["MO-BMR", "MO-BWR"],
            "common": [],
        }

        with patch(
            "loares.experiments.process.inspect.stack",
            return_value=_fake_stack(tmp_dir),
        ):
            pp = post_process(prob, "smoke-test", [20], algo_grps, true_f=tf)
        pp._per_algo_accumulator = []
        pp.run(20)

        result_dir = pp.result_dir
        pop_dir = result_dir / "20"
        assert pop_dir.exists()

        net_results = pop_dir / "net-results.csv"
        assert net_results.exists()
        df = pd.read_csv(net_results)
        assert len(df) == 2
        assert "Algorithm" in df.columns
        assert "HV(mean)" in df.columns

        final_csvs = list(pop_dir.glob("*-final-metrics.csv"))
        assert len(final_csvs) == 2
        for csv_path in final_csvs:
            fdf = pd.read_csv(csv_path)
            assert len(fdf) == 3
            assert "seed" in fdf.columns
            assert "HV" in fdf.columns

        parquets = list((pop_dir / "parquets").glob("*.parquet"))
        assert len(parquets) == 2

    def test_multi_thread_writes_per_algo_csvs(self, setup_raw_data):
        tmp_dir, prob, tf = setup_raw_data
        from loares.experiments.process import post_process

        algo_grps = {
            "base": ["MO-BMR", "MO-BWR"],
            "common": [],
        }

        with patch(
            "loares.experiments.process.inspect.stack",
            return_value=_fake_stack(tmp_dir),
        ):
            pp = post_process(prob, "smoke-test", [20], algo_grps, true_f=tf)
        result_dir = pp.multi_thread(threads=2)

        per_algo_dir = Path(result_dir) / "per-algo"
        assert per_algo_dir.exists()
        per_algo_csvs = list(per_algo_dir.glob("*-net-results.csv"))
        assert len(per_algo_csvs) == 2


# ── Test 5: compare_metrics and stats smoke test ─────────────────────────────


class TestAnalysis:
    @pytest.fixture
    def setup_analysis_data(self, tmp_dir):
        pop_dir = tmp_dir / "200"
        pop_dir.mkdir(parents=True)

        np.random.seed(42)
        for algo in ["MO-BMR", "MO-BWR", "MO-BMWR"]:
            final_df = pd.DataFrame(
                {
                    "seed": [1, 2, 3, 4, 5],
                    "GD": np.random.rand(5) * 0.1,
                    "IGD": np.random.rand(5) * 0.2,
                    "SPC": np.random.rand(5) * 0.05,
                    "SPR": np.random.rand(5) * 0.3,
                    "HV": 0.5 + np.random.rand(5) * 0.5,
                }
            )
            final_df.to_csv(pop_dir / f"{algo}-final-metrics.csv", index=False)

        net_df = pd.DataFrame(
            {
                "Algorithm": ["MO-BMR", "MO-BWR", "MO-BMWR"],
                "Psize": [200, 200, 200],
                "Max-evals": [10000, 10000, 10000],
                "GD(mean)": [0.05, 0.04, 0.06],
                "GD(std)": [0.01, 0.02, 0.01],
                "IGD(mean)": [0.1, 0.08, 0.12],
                "IGD(std)": [0.02, 0.03, 0.01],
                "SPC(mean)": [0.02, 0.03, 0.01],
                "SPC(std)": [0.005, 0.004, 0.006],
                "SPR(mean)": [0.15, 0.12, 0.18],
                "SPR(std)": [0.03, 0.02, 0.04],
                "HV(mean)": [0.8, 0.85, 0.75],
                "HV(std)": [0.05, 0.04, 0.06],
            }
        )
        net_df.to_csv(pop_dir / "net-results.csv", index=False, float_format="%.5f")

        yield tmp_dir

    def test_compare_metrics_produces_summary_csv(self, setup_analysis_data):
        from loares.experiments.analysis.compare import compare_metrics

        compare_metrics("TestProblem", setup_analysis_data)

        folder_name = setup_analysis_data.name
        summary = setup_analysis_data / f"{folder_name}.csv"
        assert summary.exists()
        df = pd.read_csv(summary)
        assert "HV(mean)" in df.columns
        assert len(df) == 1

    def test_stats_load_problem_data(self, setup_analysis_data):
        from loares.experiments.analysis.stats import load_problem_data, build_pivot

        pop_dir = setup_analysis_data / "200"
        df = load_problem_data(pop_dir)
        assert "Algorithm" in df.columns
        assert len(df) == 15

        pivot = build_pivot(df, "HV")
        assert pivot.shape[1] == 3
        assert pivot.shape[0] == 5

    def test_stats_run_produces_friedman_and_posthoc(self, setup_analysis_data):
        from loares.experiments.analysis.stats import run as run_statistics

        pop_dir = setup_analysis_data / "200"
        stats_dir = run_statistics(pop_dir, alpha=0.05)

        assert stats_dir.exists()
        assert (stats_dir / "friedman-results.csv").exists()

        friedman = pd.read_csv(stats_dir / "friedman-results.csv")
        assert "Metric" in friedman.columns
        assert "Significant" in friedman.columns
        assert len(friedman) >= 1

        # Average ranks should exist for every metric tested
        for metric in ["GD", "IGD", "SPC", "SPR", "HV"]:
            assert (stats_dir / f"{metric}-average-ranks.csv").exists()


# ── Test 6: pymoo_to_loares_h5 adapter ────────────────────────────────────────


class TestPymooAdapter:
    def test_pymoo_to_loares_h5_writes_valid_file(self, tmp_dir):
        from loares.core.adapters import pymoo_to_loares_prob, pymoo_to_loares_h5
        from loares.core.population import PopulationHandler
        from loares.operators.sorting import ranking_crowding
        from pymoo.problems.multi import ZDT1
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize

        bench = ZDT1()

        algorithm = NSGA2(pop_size=20)
        res = minimize(bench, algorithm, ("n_eval", 200), seed=1, save_history=True)

        prob = pymoo_to_loares_prob(bench, psize=20, max_evals=200)

        hdf5_path = tmp_dir / "seed_001.h5"
        pymoo_to_loares_h5(
            prob.get_info(),
            {"name": "NSGA2"},
            1,
            res,
            PopulationHandler(ranking_crowding),
            hdf5_path,
        )

        assert hdf5_path.exists()

        with h5py.File(hdf5_path, "r") as f:
            assert "metadata" in f
            assert "function_evals" in f
            assert f["metadata"].attrs["seed"] == 1

            fe_keys = list(f["function_evals"].keys())
            assert len(fe_keys) >= 1

            final_dict = json.loads(f.attrs["final_dict_json"])
            assert any(k.startswith("x") for k in final_dict)
            assert any(k.startswith("f") for k in final_dict)

        final_pop = ResultProcessor.read_final_population(hdf5_path)
        assert final_pop.solutions.shape[1] == 30
        assert final_pop.objectives.shape[1] == 2


# ── Test 7: Reference front generation (NDS + FPS) ──────────────────────────


class TestReferenceFront:
    def _make_mixed_population(self, n=50, n_vars=5, n_obj=2):
        X = np.random.rand(n, n_vars)
        f1 = np.random.rand(n)
        f2 = 1 - f1 + np.random.rand(n) * 0.3
        F = np.column_stack([f1, f2])
        G = np.full((n, 1), -1.0)
        return Population(X, F, G)

    def _make_dummy_problem(self, n_vars=5, n_obj=2, n_constr=1):
        from loares.core.problem import Problem

        return Problem(n_vars=n_vars, n_obj=n_obj, n_constr=n_constr)

    def test_nds_fps_returns_only_non_dominated(self):
        from loares.operators.sorting import nds_fps
        from pymoo.util.nds.non_dominated_sorting import find_non_dominated

        np.random.seed(42)
        pop = self._make_mixed_population(n=100, n_obj=2)
        prob = self._make_dummy_problem(n_vars=5, n_obj=2)

        ps, po, pc, pm = nds_fps(prob, pop, limit=50, seed=1)

        expected_ndf_idx = find_non_dominated(pop.objectives)
        ndf_objectives = pop.objectives[expected_ndf_idx]

        for i in range(po.shape[0]):
            match = np.any(np.all(np.isclose(po[i], ndf_objectives), axis=1))
            assert match, f"Returned point {po[i]} is not on the non-dominated front"

    def test_nds_fps_respects_limit(self):
        from loares.operators.sorting import nds_fps

        np.random.seed(42)
        pop = self._make_mixed_population(n=200, n_obj=2)
        prob = self._make_dummy_problem(n_vars=5, n_obj=2)
        limit = 15

        ps, po, pc, pm = nds_fps(prob, pop, limit=limit, seed=1)

        assert ps.shape[0] <= limit
        assert po.shape[0] <= limit
        assert pc.shape[0] <= limit
        assert pm.shape[0] <= limit

    def test_nds_fps_returns_all_when_front_smaller_than_limit(self):
        from loares.operators.sorting import nds_fps
        from pymoo.util.nds.non_dominated_sorting import find_non_dominated

        np.random.seed(42)
        pop = self._make_mixed_population(n=50, n_obj=2)
        prob = self._make_dummy_problem(n_vars=5, n_obj=2)

        ndf_size = len(find_non_dominated(pop.objectives))
        limit = ndf_size + 100

        ps, po, pc, pm = nds_fps(prob, pop, limit=limit, seed=1)

        assert ps.shape[0] == ndf_size

    def test_nds_fps_metadata_shape(self):
        from loares.operators.sorting import nds_fps

        np.random.seed(42)
        pop = self._make_mixed_population(n=100, n_obj=2)
        prob = self._make_dummy_problem(n_vars=5, n_obj=2)

        ps, po, pc, pm = nds_fps(prob, pop, limit=30, seed=1)

        assert pm.ndim == 2
        assert pm.shape[0] == ps.shape[0]
        assert pm.shape[1] == 1
        assert np.all(pm == 0)

    def test_fps_preserves_extreme_points(self):
        from loares.operators.sorting import farthest_point_sampling

        np.random.seed(42)
        points = np.random.rand(200, 2)

        selected = farthest_point_sampling(points, n_samples=20)
        selected_points = points[selected]

        for j in range(points.shape[1]):
            assert np.isclose(selected_points[:, j].min(), points[:, j].min())
            assert np.isclose(selected_points[:, j].max(), points[:, j].max())

    def test_fps_spread_better_than_random(self):
        from loares.operators.sorting import farthest_point_sampling
        from scipy.spatial.distance import cdist

        np.random.seed(42)
        points = np.random.rand(500, 2)
        n_samples = 30

        fps_idx = farthest_point_sampling(points, n_samples)
        fps_points = points[fps_idx]
        fps_dists = cdist(fps_points, fps_points)
        np.fill_diagonal(fps_dists, np.inf)
        fps_min_spacing = fps_dists.min(axis=1).mean()

        random_spacings = []
        for trial in range(20):
            rng = np.random.RandomState(trial)
            rand_idx = rng.choice(len(points), n_samples, replace=False)
            rand_points = points[rand_idx]
            rand_dists = cdist(rand_points, rand_points)
            np.fill_diagonal(rand_dists, np.inf)
            random_spacings.append(rand_dists.min(axis=1).mean())

        assert fps_min_spacing > np.mean(random_spacings)

    def test_nds_fps_output_shapes_consistent(self):
        from loares.operators.sorting import nds_fps

        np.random.seed(42)
        n_vars = 5
        n_obj = 3
        pop = self._make_mixed_population(n=100, n_vars=n_vars, n_obj=n_obj)
        pop.objectives = np.column_stack([pop.objectives, np.random.rand(100)])
        prob = self._make_dummy_problem(n_vars=n_vars, n_obj=3, n_constr=1)

        ps, po, pc, pm = nds_fps(prob, pop, limit=20, seed=1)

        assert ps.shape[1] == n_vars
        assert po.shape[1] == 3
        assert pc.shape[1] == 1
        assert ps.shape[0] == po.shape[0] == pc.shape[0] == pm.shape[0]


# ── Test 8: FPS selection quality (deeper validation) ────────────────────────


class TestFPSQuality:
    def test_fps_no_duplicate_indices(self):
        from loares.operators.sorting import farthest_point_sampling

        np.random.seed(7)
        for ndim, n_pts, n_samples in [(2, 300, 50), (3, 500, 80)]:
            points = np.random.rand(n_pts, ndim)
            selected = farthest_point_sampling(points, n_samples=n_samples)
            assert len(selected) == len(set(selected))

    def test_fps_deterministic(self):
        from loares.operators.sorting import farthest_point_sampling

        np.random.seed(0)
        points = np.random.rand(200, 2)
        s1 = farthest_point_sampling(points, n_samples=30)
        s2 = farthest_point_sampling(points, n_samples=30)
        assert s1 == s2

    def test_fps_preserves_extreme_points(self):
        from loares.operators.sorting import farthest_point_sampling

        np.random.seed(42)
        for ndim in [2, 3]:
            points = np.random.rand(400, ndim)
            selected = farthest_point_sampling(points, n_samples=30)
            selected_pts = points[selected]
            for j in range(ndim):
                assert np.isclose(selected_pts[:, j].min(), points[:, j].min())
                assert np.isclose(selected_pts[:, j].max(), points[:, j].max())

    def test_fps_spacing_uniformity(self):
        from loares.operators.sorting import farthest_point_sampling
        from scipy.spatial.distance import cdist

        np.random.seed(42)
        points = np.random.rand(1000, 2)
        selected = farthest_point_sampling(points, n_samples=50)
        sel_pts = points[selected]

        dists = cdist(sel_pts, sel_pts)
        np.fill_diagonal(dists, np.inf)
        nn_dists = dists.min(axis=1)

        cv = nn_dists.std() / nn_dists.mean()
        assert cv < 0.6, (
            f"Nearest-neighbor distance CV = {cv:.3f}, expected < 0.6 for uniform spread"
        )

    def test_nds_fps_no_duplicate_points(self):
        from loares.operators.sorting import nds_fps
        from loares.core.problem import Problem

        np.random.seed(42)
        n = 200
        X = np.random.rand(n, 5)
        f1 = np.random.rand(n)
        f2 = 1 - f1 + np.random.rand(n) * 0.3
        F = np.column_stack([f1, f2])
        G = np.full((n, 1), -1.0)
        pop = Population(X, F, G)
        prob = Problem(n_vars=5, n_obj=2, n_constr=1)

        ps, po, pc, pm = nds_fps(prob, pop, limit=30, seed=1)

        unique_rows = np.unique(po, axis=0)
        assert unique_rows.shape[0] == po.shape[0], (
            "nds_fps returned duplicate objective rows"
        )

    def test_nds_fps_solutions_match_objectives(self):
        from loares.operators.sorting import nds_fps
        from loares.core.problem import Problem

        np.random.seed(42)
        n = 150
        X = np.random.rand(n, 5)
        f1 = np.random.rand(n)
        f2 = 1 - f1 + np.random.rand(n) * 0.3
        F = np.column_stack([f1, f2])
        G = np.random.rand(n, 1) - 0.5
        pop = Population(X, F, G)
        prob = Problem(n_vars=5, n_obj=2, n_constr=1)

        ps, po, pc, pm = nds_fps(prob, pop, limit=25, seed=1)

        for i in range(ps.shape[0]):
            matches = np.where(
                np.all(np.isclose(X, ps[i]), axis=1)
                & np.all(np.isclose(F, po[i]), axis=1)
                & np.all(np.isclose(G, pc[i]), axis=1)
            )[0]
            assert len(matches) >= 1, f"Row {i}: solution/objective/constraint mismatch"


# ── Test 9: Algorithm composition ─────────────────────────────────────────────


class TestAlgorithmComposition:
    def _make_prob(self):
        from loares.core.adapters import pymoo_to_loares_prob
        from pymoo.problems.multi import ZDT1

        return pymoo_to_loares_prob(ZDT1(), psize=20, max_evals=200)

    def test_get_info_returns_expected_keys(self):
        from loares.algorithms.bxr.mo_basic import MO_BMR

        info = MO_BMR.get_info()
        assert info["name"] == "MO-BMR"
        assert "BaseFunction" in info
        assert "Mutation" in info
        assert "Selection" in info
        assert "Sorting" in info
        assert "mods" in info

    def test_call_returns_flowhandler(self):
        from loares.algorithms.bxr.mo_basic import MO_BMR
        from loares.core.flow import FlowHandler

        prob = self._make_prob()
        flow = MO_BMR(prob)
        assert isinstance(flow, FlowHandler)

    def test_call_wires_update_rule_correctly(self):
        from loares.algorithms.bxr.mo_basic import MO_BMR
        from loares.operators.bxr import bmr
        from loares.operators.selection import random_bw_selection
        from loares.operators.mutation import random_reinit

        prob = self._make_prob()
        flow = MO_BMR(prob)
        assert flow.updateRule.base_function is bmr
        assert flow.updateRule.selection is random_bw_selection
        assert flow.updateRule.mutation is random_reinit

    def test_call_wires_mods(self):
        from loares.algorithms.bxr.mo_basic import MO_BMR
        from loares.operators.mods import local_search

        prob = self._make_prob()
        flow = MO_BMR(prob)
        assert local_search in flow.mods

    def test_so_algorithm_has_no_mods(self):
        from loares.algorithms.bxr.so_basic import SO_BMR
        from loares.core.problem import Problem

        prob = Problem(
            function=lambda X: (np.sum(X**2, axis=1, keepdims=True), np.full((X.shape[0], 1), -1.0)),
            n_vars=5, n_obj=1, psize=10, max_evals=100,
            bounds=np.column_stack([np.full(5, -5), np.full(5, 5)]),
            minmax=["min"],
        )
        flow = SO_BMR(prob)
        assert flow.mods == []


# ── Test 10: Algorithm variants smoke tests ───────────────────────────────────


class TestAlgorithmVariants:
    @pytest.fixture
    def zdt1_prob(self):
        from loares.core.adapters import pymoo_to_loares_prob
        from pymoo.problems.multi import ZDT1

        return pymoo_to_loares_prob(ZDT1(), psize=20, max_evals=200)

    def _run_and_validate(self, algorithm, prob, tmp_dir, seed=1):
        flow = algorithm(prob)
        hdf5_path = tmp_dir / f"seed_{seed:03d}.h5"
        flow.run(seed=seed, hdf5_path=hdf5_path)
        assert hdf5_path.exists()
        final_pop = ResultProcessor.read_final_population(hdf5_path)
        assert final_pop.solutions.shape[0] > 0
        assert final_pop.solutions.shape[1] == 30
        assert final_pop.objectives.shape[1] == 2

    def test_archive_variant_runs(self, zdt1_prob, tmp_dir):
        from loares.algorithms.bxr.mo_archive import MO_BMR_A

        self._run_and_validate(MO_BMR_A, zdt1_prob, tmp_dir)

    def test_samp_variant_runs(self, zdt1_prob, tmp_dir):
        from loares.algorithms.bxr.mo_samp import MO_BMR_S

        self._run_and_validate(MO_BMR_S, zdt1_prob, tmp_dir)

    def test_opposition_variant_runs(self, zdt1_prob, tmp_dir):
        from loares.algorithms.bxr.mo_opp import MO_BMR_O

        self._run_and_validate(MO_BMR_O, zdt1_prob, tmp_dir)


# ── Test 11: Operators unit tests ─────────────────────────────────────────────


class TestOperators:
    @pytest.fixture
    def setup(self):
        from loares.core.problem import Problem
        from loares.operators.sorting import ranking_crowding

        np.random.seed(42)
        n, n_vars, n_obj = 20, 5, 2
        prob = Problem(
            n_vars=n_vars, n_obj=n_obj, n_constr=1, psize=n,
            bounds=np.column_stack([np.zeros(n_vars), np.ones(n_vars)]),
        )
        X = np.random.rand(n, n_vars)
        F = np.random.rand(n, n_obj)
        G = np.full((n, 1), -1.0)
        raw_pop = Population(X, F, G)
        pop = Population(*ranking_crowding(prob, raw_pop, n, seed=1))

        return prob, pop

    def test_bmr_within_bounds(self, setup):
        from loares.operators.bxr import bmr
        from loares.operators.selection import random_bw_selection

        prob, pop = setup
        pool = random_bw_selection(pop)
        result = bmr(prob, pop, pool)
        assert result.shape == pop.solutions.shape
        assert np.all(result >= prob.bounds[:, 0])
        assert np.all(result <= prob.bounds[:, 1])

    def test_bwr_within_bounds(self, setup):
        from loares.operators.bxr import bwr
        from loares.operators.selection import random_bw_selection

        prob, pop = setup
        pool = bwr(prob, pop, random_bw_selection(pop))
        assert np.all(pool >= prob.bounds[:, 0])
        assert np.all(pool <= prob.bounds[:, 1])

    def test_bmwr_within_bounds(self, setup):
        from loares.operators.bxr import bmwr
        from loares.operators.selection import random_bw_selection

        prob, pop = setup
        result = bmwr(prob, pop, random_bw_selection(pop))
        assert np.all(result >= prob.bounds[:, 0])
        assert np.all(result <= prob.bounds[:, 1])

    def test_random_reinit_within_bounds(self, setup):
        from loares.operators.mutation import random_reinit

        prob, pop = setup
        result = random_reinit(prob, pop.solutions)
        assert result.shape == pop.solutions.shape
        assert np.all(result >= prob.bounds[:, 0])
        assert np.all(result <= prob.bounds[:, 1])

    def test_bw_selection_structure(self, setup):
        from loares.operators.selection import bw_selection
        from loares.operators.sorting import bw_sorting

        prob, pop = setup
        sorted_pop = Population(*bw_sorting(prob, pop, prob.psize, seed=1))
        pool = bw_selection(sorted_pop)
        assert "best" in pool
        assert "worst" in pool
        assert pool["best"].shape == (prob.n_vars,)
        assert pool["worst"].shape == (prob.n_vars,)

    def test_random_bw_selection_structure(self, setup):
        from loares.operators.selection import random_bw_selection

        prob, pop = setup
        pool = random_bw_selection(pop)
        assert "best" in pool
        assert "worst" in pool
        assert pool["best"].shape == (pop.get_size(), prob.n_vars)
        assert pool["worst"].shape == (pop.get_size(), prob.n_vars)

    def test_local_search_within_bounds(self, setup):
        from loares.operators.mods import local_search
        from loares.core.population import PopulationHandler
        from loares.operators.sorting import ranking_crowding

        prob, pop = setup
        ph = PopulationHandler(ranking_crowding)
        ph.seed = 1
        result = local_search(prob, pop, ph)
        assert np.all(result >= prob.bounds[:, 0])
        assert np.all(result <= prob.bounds[:, 1])

    def test_opposition_within_bounds(self, setup):
        from loares.operators.mods import opposition
        from loares.core.population import PopulationHandler
        from loares.operators.sorting import ranking_crowding

        prob, pop = setup
        ph = PopulationHandler(ranking_crowding)
        result = opposition(prob, pop, ph)
        assert result.shape == pop.solutions.shape
        assert np.all(result >= prob.bounds[:, 0])
        assert np.all(result <= prob.bounds[:, 1])
