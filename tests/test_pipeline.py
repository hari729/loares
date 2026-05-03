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

from pymoo.problems.multi import ZDT1
from pymoo.problems.single import Sphere
from pymoo.algorithms.moo.nsga2 import NSGA2

from loares.experiments.pymoo_runner import AlgoFactory, ExperimentRunner
from loares.experiments.pymoo_process import (
    PostProcess,
    MOOMetrics,
    SOOMetrics,
    _read_metadata,
    _read_final_dict,
    _read_final_arrays,
    _stream_snapshots,
)
from loares.algorithms.bxr.moo import (
    MO_BMR_py,
    MO_BWR,
    MO_BMWR,
    MO_BMR_Archive_py,
    MO_BMR_Opposition,
    MO_BMR_S_py,
)
from loares.algorithms.bxr.soo import SO_BMR, SO_BWR, SO_BMWR
from loares.core.population import Population


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def _fake_stack(caller_dir):
    real_stack = inspect.stack()
    fake_frame = type(real_stack[0])(
        real_stack[0].frame,
        str(caller_dir / "fake_caller.py"),
        real_stack[0].lineno,
        real_stack[0].function,
        real_stack[0].code_context,
        real_stack[0].index,
    )
    return [real_stack[0], fake_frame] + real_stack[2:]


# ── AlgoFactory ──────────────────────────────────────────────────────────────


class TestAlgoFactory:
    def test_stock_pymoo_gets_name(self):
        factory = AlgoFactory(NSGA2, pop_size=20)
        algo = factory()
        assert algo.name == "NSGA2"
        assert algo.pop_size == 20

    def test_explicit_name_override(self):
        factory = AlgoFactory(NSGA2, name="Custom-NSGA2", pop_size=20)
        algo = factory()
        assert algo.name == "Custom-NSGA2"

    def test_modular_algorithm_keeps_name(self):
        factory = AlgoFactory(MO_BMR_py, pop_size=30)
        algo = factory()
        assert algo.name == "MO-BMR"
        assert algo.pop_size == 30

    def test_factory_is_picklable(self):
        import pickle

        factory = AlgoFactory(NSGA2, pop_size=20)
        restored = pickle.loads(pickle.dumps(factory))
        algo = restored()
        assert algo.name == "NSGA2"
        assert algo.pop_size == 20

    def test_each_call_returns_fresh_instance(self):
        factory = AlgoFactory(NSGA2, pop_size=20)
        a1 = factory()
        a2 = factory()
        assert a1 is not a2


# ── ExperimentRunner (MOO) ───────────────────────────────────────────────────


class TestExperimentRunnerMOO:
    def test_single_seed_produces_hdf5(self, tmp_dir):
        problem = ZDT1()
        factory = AlgoFactory(MO_BMR_py, pop_size=20)

        with patch(
            "loares.experiments.pymoo_runner.inspect.stack",
            return_value=_fake_stack(tmp_dir),
        ):
            runner = ExperimentRunner(problem, factory, max_evals=200, test_name="t1")

        runner.run(seed=1)

        seed_files = list(runner.output_dir.glob("seed_*.h5"))
        assert len(seed_files) == 1

        with h5py.File(seed_files[0], "r") as f:
            assert "metadata" in f
            assert "function_evals" in f
            assert f["metadata"].attrs["seed"] == 1

            pinfo = json.loads(f["metadata"].attrs["problem_info_json"])
            assert pinfo["n_obj"] == 2
            assert pinfo["n_vars"] == 30

            fe_keys = sorted(f["function_evals"].keys(), key=lambda k: int(k))
            assert len(fe_keys) >= 2

            last = f["function_evals"][fe_keys[-1]]
            assert last["X"].shape[1] == 30
            assert last["F"].shape[1] == 2

            final_dict = json.loads(f.attrs["final_dict_json"])
            assert "x1" in final_dict
            assert "f1" in final_dict

    def test_multi_run_produces_seeds_and_info(self, tmp_dir):
        problem = ZDT1()
        factory = AlgoFactory(MO_BMR_py, pop_size=20)

        with patch(
            "loares.experiments.pymoo_runner.inspect.stack",
            return_value=_fake_stack(tmp_dir),
        ):
            runner = ExperimentRunner(problem, factory, max_evals=200, test_name="t2")

        runner.multi_run(seeds=[1, 2, 3], threads=2)

        assert (runner.output_dir / "Info.json").exists()
        seed_files = sorted(runner.output_dir.glob("seed_*.h5"))
        assert len(seed_files) == 3

        info = json.loads((runner.output_dir / "Info.json").read_text())
        assert "Problem" in info
        assert "Algorithm" in info
        assert info["Algorithm"]["name"] == "MO-BMR"

    def test_stock_pymoo_algorithm(self, tmp_dir):
        problem = ZDT1()
        factory = AlgoFactory(NSGA2, pop_size=20)

        with patch(
            "loares.experiments.pymoo_runner.inspect.stack",
            return_value=_fake_stack(tmp_dir),
        ):
            runner = ExperimentRunner(problem, factory, max_evals=200, test_name="t3")

        runner.run(seed=1)
        seed_files = list(runner.output_dir.glob("seed_*.h5"))
        assert len(seed_files) == 1

        _, ainfo, _ = _read_metadata(seed_files[0])
        assert ainfo["name"] == "NSGA2"

    def test_problem_info_includes_minmax(self, tmp_dir):
        problem = ZDT1()
        problem.minmax = np.array([1, -1])
        factory = AlgoFactory(MO_BMR_py, pop_size=20)

        with patch(
            "loares.experiments.pymoo_runner.inspect.stack",
            return_value=_fake_stack(tmp_dir),
        ):
            runner = ExperimentRunner(problem, factory, max_evals=200, test_name="t4")

        assert runner.problem_info["minmax"] == [1, -1]


# ── ExperimentRunner (SOO) ───────────────────────────────────────────────────


class TestExperimentRunnerSOO:
    def test_soo_single_seed(self, tmp_dir):
        problem = Sphere(n_var=5)
        factory = AlgoFactory(SO_BMR, pop_size=20)

        with patch(
            "loares.experiments.pymoo_runner.inspect.stack",
            return_value=_fake_stack(tmp_dir),
        ):
            runner = ExperimentRunner(problem, factory, max_evals=200, test_name="soo1")

        runner.run(seed=1)

        seed_files = list(runner.output_dir.glob("seed_*.h5"))
        assert len(seed_files) == 1

        X, F, G = _read_final_arrays(seed_files[0])
        assert X.shape[1] == 5
        assert F.shape[1] == 1


# ── HDF5 readers ─────────────────────────────────────────────────────────────


class TestHDF5Readers:
    @pytest.fixture
    def seed_file(self, tmp_dir):
        problem = ZDT1()
        factory = AlgoFactory(MO_BMR_py, pop_size=20)

        with patch(
            "loares.experiments.pymoo_runner.inspect.stack",
            return_value=_fake_stack(tmp_dir),
        ):
            runner = ExperimentRunner(problem, factory, max_evals=200, test_name="rd")

        runner.run(seed=42)
        return list(runner.output_dir.glob("seed_*.h5"))[0]

    def test_read_metadata(self, seed_file):
        pinfo, ainfo, seed = _read_metadata(seed_file)
        assert pinfo["n_obj"] == 2
        assert ainfo["name"] == "MO-BMR"
        assert seed == 42

    def test_read_final_dict(self, seed_file):
        fd = _read_final_dict(seed_file)
        assert any(k.startswith("x") for k in fd)
        assert any(k.startswith("f") for k in fd)

    def test_read_final_arrays(self, seed_file):
        X, F, G = _read_final_arrays(seed_file)
        assert X.shape[1] == 30
        assert F.shape[1] == 2

    def test_stream_snapshots_sorted(self, seed_file):
        evals_list = [ev for ev, _ in _stream_snapshots(seed_file)]
        assert evals_list == sorted(evals_list)
        assert len(evals_list) >= 2


# ── MOOMetrics / SOOMetrics ──────────────────────────────────────────────────


class TestMetrics:
    def test_moo_metrics_with_true_front(self):
        tf = ZDT1().pareto_front(500)
        m = MOOMetrics(tf, n_obj=2)
        F = np.column_stack([np.linspace(0, 1, 20), np.linspace(1, 0, 20)])
        result = m(F)
        assert "GD" in result
        assert "IGD" in result
        assert "HV" in result
        assert "SPC" in result
        assert all(np.isfinite(v) for v in result.values())

    def test_moo_metrics_without_true_front(self):
        m = MOOMetrics(None, n_obj=2)
        F = np.column_stack([np.linspace(0, 1, 20), np.linspace(1, 0, 20)])
        result = m(F)
        assert "HV" in result
        assert "SPC" in result
        assert "GD" not in result

    def test_moo_metrics_empty_F(self):
        tf = ZDT1().pareto_front(100)
        m = MOOMetrics(tf, n_obj=2)
        result = m(np.empty((0, 2)))
        assert np.isnan(result["HV"])
        assert np.isnan(result["GD"])

    def test_moo_metrics_single_point(self):
        tf = ZDT1().pareto_front(100)
        m = MOOMetrics(tf, n_obj=2)
        result = m(np.array([[0.5, 0.5]]))
        assert np.isfinite(result["HV"])
        assert np.isnan(result["SPC"])

    def test_soo_metrics(self):
        m = SOOMetrics()
        F = np.array([[3.0], [1.0], [5.0]])
        result = m(F)
        assert result["best"] == 1.0
        assert result["worst"] == 5.0

    def test_moo_reuses_indicators(self):
        tf = ZDT1().pareto_front(100)
        m = MOOMetrics(tf, n_obj=2)
        gd_obj = m._gd
        igd_obj = m._igd
        hv_obj = m._hv
        F = np.random.rand(10, 2)
        m(F)
        m(F)
        assert m._gd is gd_obj
        assert m._igd is igd_obj
        assert m._hv is hv_obj


# ── PostProcess ──────────────────────────────────────────────────────────────


class TestPostProcess:
    @pytest.fixture
    def setup_raw_data(self, tmp_dir):
        problem = ZDT1()
        tf = problem.pareto_front(500)
        seeds = [1, 2, 3]

        for algo_cls in [MO_BMR_py, MO_BWR]:
            factory = AlgoFactory(algo_cls, pop_size=20)
            with patch(
                "loares.experiments.pymoo_runner.inspect.stack",
                return_value=_fake_stack(tmp_dir),
            ):
                runner = ExperimentRunner(
                    problem, factory, max_evals=200, test_name="pp-test"
                )
            runner.multi_run(seeds, threads=2)

        raw_dir = tmp_dir / "pp-test" / "raw_data"
        yield raw_dir, tf

    def test_discover_problem_info(self, setup_raw_data):
        raw_dir, tf = setup_raw_data
        pp = PostProcess(
            raw_dir,
            algo_grps={"base": ["MO-BMR", "MO-BWR"], "common": []},
            true_front=tf,
            plot_convergence=False,
            plot_pareto=False,
        )
        assert pp.problem_info["n_obj"] == 2
        assert pp.problem_info["n_vars"] == 30

    def test_run_produces_expected_outputs(self, setup_raw_data):
        raw_dir, tf = setup_raw_data
        pp = PostProcess(
            raw_dir,
            algo_grps={"base": ["MO-BMR", "MO-BWR"], "common": []},
            true_front=tf,
            plot_convergence=False,
            plot_pareto=False,
        )
        result_dir = pp.run(threads=2)

        pop_dir = result_dir / "20"
        assert pop_dir.exists()

        net_csv = pop_dir / "net-results.csv"
        assert net_csv.exists()
        df = pd.read_csv(net_csv)
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
            assert "GD" in fdf.columns

        parquets = list((pop_dir / "parquets").glob("*.parquet"))
        assert len(parquets) == 2

    def test_per_algo_csvs(self, setup_raw_data):
        raw_dir, tf = setup_raw_data
        pp = PostProcess(
            raw_dir,
            algo_grps={"base": ["MO-BMR", "MO-BWR"], "common": []},
            true_front=tf,
            plot_convergence=False,
            plot_pareto=False,
        )
        result_dir = pp.run(threads=2)

        per_algo_dir = result_dir / "per-algo"
        assert per_algo_dir.exists()
        per_algo_csvs = list(per_algo_dir.glob("*-net-results.csv"))
        assert len(per_algo_csvs) == 2


# ── Algorithm variants smoke tests ──────────────────────────────────────────


class TestAlgorithmVariants:
    def _run_variant(self, algo_cls, problem, tmp_dir, max_evals=200, pop_size=20):
        factory = AlgoFactory(algo_cls, pop_size=pop_size)
        with patch(
            "loares.experiments.pymoo_runner.inspect.stack",
            return_value=_fake_stack(tmp_dir),
        ):
            runner = ExperimentRunner(
                problem, factory, max_evals=max_evals, test_name="variant"
            )
        runner.run(seed=1)
        seed_files = list(runner.output_dir.glob("seed_*.h5"))
        assert len(seed_files) == 1
        X, F, G = _read_final_arrays(seed_files[0])
        assert X.shape[0] > 0
        return X, F

    def test_mo_bmr(self, tmp_dir):
        X, F = self._run_variant(MO_BMR_py, ZDT1(), tmp_dir)
        assert X.shape[1] == 30
        assert F.shape[1] == 2

    def test_mo_bwr(self, tmp_dir):
        X, F = self._run_variant(MO_BWR, ZDT1(), tmp_dir)
        assert F.shape[1] == 2

    def test_mo_bmwr(self, tmp_dir):
        X, F = self._run_variant(MO_BMWR, ZDT1(), tmp_dir)
        assert F.shape[1] == 2

    def test_mo_bmr_archive(self, tmp_dir):
        X, F = self._run_variant(MO_BMR_Archive_py, ZDT1(), tmp_dir)
        assert F.shape[1] == 2

    def test_mo_bmr_opposition(self, tmp_dir):
        X, F = self._run_variant(MO_BMR_Opposition, ZDT1(), tmp_dir)
        assert F.shape[1] == 2

    def test_mo_bmr_samp(self, tmp_dir):
        X, F = self._run_variant(MO_BMR_S_py, ZDT1(), tmp_dir)
        assert F.shape[1] == 2

    def test_so_bmr(self, tmp_dir):
        X, F = self._run_variant(SO_BMR, Sphere(n_var=5), tmp_dir)
        assert X.shape[1] == 5
        assert F.shape[1] == 1

    def test_so_bwr(self, tmp_dir):
        X, F = self._run_variant(SO_BWR, Sphere(n_var=5), tmp_dir)
        assert F.shape[1] == 1

    def test_so_bmwr(self, tmp_dir):
        X, F = self._run_variant(SO_BMWR, Sphere(n_var=5), tmp_dir)
        assert F.shape[1] == 1


# ── Reference front generation (NDS + FPS) ──────────────────────────────────


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

        ps, po, pc, pm = nds_fps(prob, pop, limit=15, seed=1)
        assert ps.shape[0] <= 15
        assert po.shape[0] <= 15

    def test_nds_fps_returns_all_when_front_smaller_than_limit(self):
        from loares.operators.sorting import nds_fps
        from pymoo.util.nds.non_dominated_sorting import find_non_dominated

        np.random.seed(42)
        pop = self._make_mixed_population(n=50, n_obj=2)
        prob = self._make_dummy_problem(n_vars=5, n_obj=2)

        ndf_size = len(find_non_dominated(pop.objectives))
        ps, po, pc, pm = nds_fps(prob, pop, limit=ndf_size + 100, seed=1)
        assert ps.shape[0] == ndf_size

    def test_nds_fps_output_shapes_consistent(self):
        from loares.operators.sorting import nds_fps

        np.random.seed(42)
        n_vars, n_obj = 5, 3
        X = np.random.rand(100, n_vars)
        F = np.random.rand(100, n_obj)
        G = np.full((100, 1), -1.0)
        pop = Population(X, F, G)
        prob = self._make_dummy_problem(n_vars=n_vars, n_obj=3, n_constr=1)

        ps, po, pc, pm = nds_fps(prob, pop, limit=20, seed=1)
        assert ps.shape[1] == n_vars
        assert po.shape[1] == 3
        assert pc.shape[1] == 1
        assert ps.shape[0] == po.shape[0] == pc.shape[0] == pm.shape[0]


# ── FPS selection quality ────────────────────────────────────────────────────


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
        assert cv < 0.6


# ── Analysis (compare_metrics + stats) ───────────────────────────────────────


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

        for metric in ["GD", "IGD", "SPC", "HV"]:
            assert (stats_dir / f"{metric}-average-ranks.csv").exists()
