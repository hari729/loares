import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymoo.problems.multi import ZDT1
from pymoo.problems.single import Sphere
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.gd import GD
from pymoo.indicators.hv import HV
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.core.population import Population as PymooPopulation
from pymoo.core.problem import Problem as PymooProblem

from loares.run import parallel_run, pending_specs
from loares.utils import get_spec_path, get_spec_info, unzip_result
from loares.indicator import (
    calculate_indicator,
    indicator_multi_run,
    calculate_indicator_history,
    indicator_history_multi_run,
    mean_history_multi_run,
    calculate_mean_history,
    build_convergence_lines,
    build_convergence_lines_for_algos,
)
from loares.statistics import (
    build_pivot,
    vargha_delaney_a12,
    compute_a12_matrix,
    friedman_connover_holm,
    statistical_test_1,
)
from loares.algorithms.bxr.moo import (
    MO_BMR,
    MO_BWR,
    MO_BMWR,
    MO_BMR_Archive,
    MO_BMR_Opposition,
    MO_BMR_S,
)
from loares.algorithms.bxr.soo import SO_BMR, SO_BWR, SO_BMWR


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def make_spec(
    output_dir,
    name,
    algorithm,
    seed=1,
    max_evals=60,
    pop_size=10,
    problem=None,
    problem_name="ZDT1",
):
    return {
        "algorithm_name": name,
        "algorithm": algorithm,
        "algorithm_kwargs": {"pop_size": pop_size},
        "problem_name": problem_name,
        "problem": problem,
        "output_dir": output_dir,
        "plot_kwargs": {},
        "solver_kwargs": {
            "seed": seed,
            "termination": ("n_eval", max_evals),
            "save_history": False,
        },
    }


# ── Spec path / info ─────────────────────────────────────────────────────────


class TestSpecPaths:
    def test_get_spec_path_layout(self, tmp_dir):
        spec = make_spec(tmp_dir, "NSGA-II", NSGA2(pop_size=10), seed=1, max_evals=60)
        path = get_spec_path(spec)
        assert path == Path("ZDT1/n_eval-60/NSGA-II/10/seed_001")

    def test_get_spec_info_fields(self, tmp_dir):
        spec = make_spec(tmp_dir, "NSGA-II", NSGA2(pop_size=10), seed=3, max_evals=60)
        info = get_spec_info(spec)
        assert info["algorithm_name"] == "NSGA-II"
        assert info["problem_name"] == "ZDT1"
        assert info["pop_size"] == 10
        assert info["seed"] == 3
        assert info["termination_metric"] == "n_eval"
        assert info["termination_value"] == 60


# ── parallel_run + write_results ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def run_env(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    spec = make_spec(
        out,
        "NSGA-II",
        NSGA2(pop_size=10),
        seed=1,
        max_evals=60,
        pop_size=10,
        problem=ZDT1(),
    )
    spec["solver_kwargs"]["save_history"] = True
    parallel_run([spec], out, n_jobs=1)
    return out, spec


class TestRun:
    def test_parallel_run_produces_result(self, run_env):
        out, spec = run_env
        result_path = out / get_spec_path(spec) / "result.pkl.gz"
        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_result_contains_history(self, run_env):
        out, spec = run_env
        result = unzip_result(out / get_spec_path(spec) / "result.pkl.gz")
        assert len(result.history) >= 2
        last = result.history[-1]
        assert last.opt.get("F").shape[1] == 2
        assert last.opt.get("X").shape[1] == 30

    def test_final_state_readable(self, run_env):
        out, spec = run_env
        result = unzip_result(out / get_spec_path(spec) / "result.pkl.gz")
        assert result.F.shape[1] == 2

    def test_overwrite_false_skips_existing(self, run_env):
        out, spec = run_env
        assert pending_specs([spec], out, overwrite=False) == []

    def test_overwrite_true_reruns(self, run_env):
        out, spec = run_env
        assert pending_specs([spec], out, overwrite=True) == [spec]

    def test_pending_specs_filters_only_missing(self, tmp_dir):
        existing = make_spec(
            tmp_dir, "NSGA-II", NSGA2(pop_size=10), problem=ZDT1()
        )
        missing = make_spec(
            tmp_dir, "MO-BMR", MO_BMR(pop_size=10), problem=ZDT1()
        )
        result_path = tmp_dir / get_spec_path(existing) / "result.pkl.gz"
        result_path.parent.mkdir(parents=True)
        result_path.touch()
        assert pending_specs([existing, missing], tmp_dir, overwrite=False) == [missing]


# ── Indicators ───────────────────────────────────────────────────────────────


def indicator_specs():
    tf = ZDT1().pareto_front(200)
    return [
        {"indicator_name": "GD", "indicator": GD(tf)},
        {"indicator_name": "HV", "indicator": HV(ref_point=[1.1, 1.1])},
    ]


class TestIndicators:
    def test_calculate_indicator_uses_indicator_value(self, run_env):
        out, spec = run_env
        run_spec = pd.read_csv(out / "run_manifest.csv").iloc[0].to_dict()
        rows = calculate_indicator((indicator_specs()[0], run_spec))
        assert len(rows) == 1
        for row in rows:
            assert "indicator_value" in row
            assert "value" not in row

    def test_metrics_csv_has_indicator_value_not_value(self, run_env, tmp_dir):
        out, spec = run_env
        indicator_multi_run(indicator_specs(), out, n_jobs=1)
        df = pd.read_csv(out / "metrics.csv")
        assert "indicator_value" in df.columns
        assert "value" not in df.columns
        assert "indicator_name" in df.columns
        assert set(df["indicator_name"]) == {"GD", "HV"}

    def test_metrics_dedup_skips_existing_rows(self, run_env, tmp_dir):
        out, spec = run_env
        indicator_multi_run(indicator_specs(), out, n_jobs=1)
        indicator_multi_run(indicator_specs(), out, n_jobs=1)
        df = pd.read_csv(out / "metrics.csv")
        assert len(df) == 2

    def test_calculate_indicator_history_uses_indicator_value(self, run_env):
        out, spec = run_env
        rows = calculate_indicator_history((indicator_specs(), spec), out)
        assert rows
        for row in rows:
            assert "indicator_value" in row
            assert "value" not in row
            assert "evals" in row

    def test_history_parquet_has_indicator_value(self, run_env, tmp_dir):
        out, spec = run_env
        indicator_history_multi_run(indicator_specs(), [spec], out, tmp_dir, n_jobs=1)
        df = pd.read_parquet(tmp_dir / "history.parquet")
        assert "indicator_value" in df.columns
        assert "value" not in df.columns
        assert len(df) > 0

    def test_mean_history_list_columns_and_plot_lookup(self, run_env, tmp_dir):
        out, spec = run_env
        indicator_history_multi_run(indicator_specs(), [spec], out, tmp_dir, n_jobs=1)
        mean_history_multi_run(tmp_dir / "history.parquet", tmp_dir)

        mean_df = pd.read_parquet(tmp_dir / "mean_history.parquet")
        assert "indicator_value" in mean_df.columns
        assert "value" not in mean_df.columns
        assert "evals" in mean_df.columns
        hv_rows = mean_df[mean_df["indicator_name"] == "HV"]
        assert len(hv_rows) == 1
        curve = hv_rows.iloc[0]["indicator_value"]
        assert isinstance(curve, (list, np.ndarray))
        assert len(curve) >= 2

        filt = {"indicator_name": "HV", "source": "opt"}
        data = build_convergence_lines(mean_df, {"filter": filt})
        assert len(data["xdata"]) == 1
        assert data["legend"] == ["NSGA-II"]
        assert data["ydata"][0].shape == data["xdata"][0].shape

        for_algos = build_convergence_lines_for_algos(
            mean_df, filt, ["NSGA-II"], "Function Evaluations", "Hypervolume"
        )
        assert len(for_algos["xdata"]) == 1
        assert np.asarray(for_algos["ydata"][0]).shape == np.asarray(
            for_algos["xdata"][0]
        ).shape

    def test_calculate_mean_history_collapses_across_seeds(self, tmp_dir):
        key_cols = ["algorithm_name", "indicator_name", "source"]
        rows = []
        for seed in [1, 2]:
            for ev in [0, 10, 20]:
                rows.append(
                    {
                        "algorithm_name": "A",
                        "indicator_name": "HV",
                        "source": "opt",
                        "seed": seed,
                        "evals": ev,
                        "indicator_value": 0.5 + 0.01 * ev + seed,
                    }
                )
        history = pd.DataFrame(rows)
        mean = calculate_mean_history(history, key_cols)
        assert len(mean) == 1
        assert len(mean.iloc[0]["indicator_value"]) == 3


# ── Statistics ───────────────────────────────────────────────────────────────


def synthetic_metrics(tmp_dir, n_seeds=5, rng_seed=0):
    rng = np.random.RandomState(rng_seed)
    base_hv = {"MO-BMR": 0.9, "MO-BWR": 0.5, "MO-BMWR": 0.1}
    rows = []
    for algo, base in base_hv.items():
        for seed in range(1, n_seeds + 1):
            rows.append(
                {
                    "algorithm_name": algo,
                    "problem_name": "ZDT1",
                    "pop_size": 100,
                    "seed": seed,
                    "termination_metric": "n_eval",
                    "termination_value": 25000,
                    "output_dir": str(tmp_dir),
                    "source": "opt",
                    "indicator_name": "HV",
                    "indicator_value": base + rng.uniform(-0.01, 0.01),
                }
            )
    return pd.DataFrame(rows)


def hv_stat_spec():
    return {
        "filter": {
            "indicator_name": "HV",
            "pop_size": 100,
            "termination_metric": "n_eval",
            "termination_value": 25000,
            "source": "opt",
        },
        "pivot": {
            "index": "seed",
            "columns": "algorithm_name",
            "values": "indicator_value",
        },
    }


class TestStatistics:
    def test_save_heatmap_annotated(self, tmp_dir):
        from loares.plots import save_heatmap

        matrix = np.array([[0.5, 0.8], [0.2, 0.5]])
        out = tmp_dir / "hm.pdf"
        save_heatmap(matrix, ["A", "B"], ["A", "B"], out, annotate=True)
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_save_heatmap_annotated_p_values(self, tmp_dir):
        from loares.plots import save_heatmap

        matrix = np.array([[1.0, 0.0012], [0.0012, 1.0]])
        out = tmp_dir / "pv.pdf"
        save_heatmap(matrix, ["A", "B"], ["A", "B"], out, annotate=True, fmt=".4f")
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_annotated_heatmap_glyph_significance(self):
        from loares.plots import AnnotatedHeatmap

        matrix = np.array([[0.5, 0.82], [0.18, 0.5]])
        sig = np.array([[False, True], [False, False]])
        hm = AnnotatedHeatmap(
            bounds=[0, 1],
            cmap="RdBu_r",
            reverse=False,
            solution_labels=["A", "B"],
            labels=["A", "B"],
            fmt=".3f",
            significance=sig,
            glyph=True,
        )
        hm.add(matrix)
        hm.do()

        artists = {t.get_position(): t for t in hm.ax.texts}
        assert artists[(1, 0)].get_text() == "0.820*"  # >0.5: row wins, significant
        assert artists[(0, 1)].get_text() == "0.180"  # <0.5: column wins
        assert artists[(0, 0)].get_text() == "0.500"  # tie: no marker, no asterisk
        renderer = hm.fig.canvas.get_renderer()
        inv = hm.ax.transData.inverted()
        rows = {}
        for (x, y), artist in artists.items():
            bbox = artist.get_window_extent(renderer=renderer)
            right = inv.transform((bbox.x1, bbox.y0))[0]
            rows.setdefault(y, []).append((x, right))
        markers = [tuple(o) for c in hm.ax.collections for o in c.get_offsets()]
        assert len(markers) == 2  # only the two non-tie cells
        for mx, my in markers:
            center, right = min(rows[my], key=lambda c: abs(c[0] - mx))
            assert right < mx < right + 0.5  # marker just right of its text

    def test_annotated_heatmap_no_markers_by_default(self):
        from loares.plots import AnnotatedHeatmap

        matrix = np.array([[0.5, 0.82], [0.18, 0.5]])
        hm = AnnotatedHeatmap(
            bounds=[0, 1],
            solution_labels=["A", "B"],
            labels=["A", "B"],
        )
        hm.add(matrix)
        hm.do()
        texts = {t.get_position(): t.get_text() for t in hm.ax.texts}
        assert texts[(1, 0)] == "0.820"

    def test_vargha_delaney_a12(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        assert np.isclose(vargha_delaney_a12(x, x), 0.5)
        assert np.isclose(
            vargha_delaney_a12(x, y) + vargha_delaney_a12(y, x), 1.0
        )

    def test_compute_a12_matrix(self, tmp_dir):
        pivot = synthetic_metrics(tmp_dir).pivot(
            index="seed", columns="algorithm_name", values="indicator_value"
        )
        matrix = compute_a12_matrix(pivot, ascending=False)
        assert sorted(matrix.index) == ["MO-BMR", "MO-BMWR", "MO-BWR"]
        assert np.all(np.diag(matrix.to_numpy()) == 0.5)
        assert matrix.loc["MO-BMR", "MO-BWR"] > 0.5

    def test_build_pivot(self, tmp_dir):
        df = synthetic_metrics(tmp_dir)
        pivot = build_pivot(df, hv_stat_spec())
        assert pivot.shape == (5, 3)
        assert sorted(pivot.columns) == ["MO-BMR", "MO-BMWR", "MO-BWR"]

    def test_friedman_significant_returns_posthoc(self, tmp_dir):
        pivot = synthetic_metrics(tmp_dir).pivot(
            index="seed", columns="algorithm_name", values="indicator_value"
        )
        result, posthoc = friedman_connover_holm(pivot)
        assert result["P-value"] < 0.05
        assert posthoc is not None
        assert posthoc.shape == (3, 3)

    def test_friedman_identical_data_no_posthoc(self):
        data = pd.DataFrame(
            {
                "A": [0.5, 0.5, 0.5, 0.5, 0.5],
                "B": [0.5, 0.5, 0.5, 0.5, 0.5],
                "C": [0.5, 0.5, 0.5, 0.5, 0.5],
            }
        )
        result, posthoc = friedman_connover_holm(data)
        assert posthoc is None

    def test_friedman_too_few_blocks(self):
        data = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]})
        result, posthoc = friedman_connover_holm(data)
        assert np.isnan(result["P-value"])
        assert posthoc is None

    def test_statistical_test_1_end_to_end(self, tmp_dir):
        metrics = synthetic_metrics(tmp_dir)
        metrics_path = tmp_dir / "metrics.csv"
        metrics.to_csv(metrics_path, index=False)

        statistical_test_1([hv_stat_spec()], metrics_path, tmp_dir)

        stats_dir = tmp_dir / "statistical-test-1"
        assert (stats_dir / "friedman-results.csv").exists()
        assert (stats_dir / "HV-a12.csv").exists()
        assert (stats_dir / "HV-average-ranks.csv").exists()
        assert (stats_dir / "HV-a12.pdf").exists()
        assert (stats_dir / "HV-conover-holm.csv").exists()
        assert (stats_dir / "HV-conover-holm.pdf").exists()

        friedman = pd.read_csv(stats_dir / "friedman-results.csv")
        assert len(friedman) == 1
        assert friedman.iloc[0]["indicator_name"] == "HV"


# ── Algorithm variants smoke tests ───────────────────────────────────────────


MO_VARIANTS = [
    ("MO-BMR", MO_BMR),
    ("MO-BWR", MO_BWR),
    ("MO-BMWR", MO_BMWR),
    ("MO-BMR-Archive", MO_BMR_Archive),
    ("MO-BMR-Opposition", MO_BMR_Opposition),
    ("MO-BMR-SAMP", MO_BMR_S),
]

SO_VARIANTS = [
    ("SO-BMR", SO_BMR),
    ("SO-BWR", SO_BWR),
    ("SO-BMWR", SO_BMWR),
]


class TestAlgorithmVariants:
    @pytest.mark.parametrize("name,algo_cls", MO_VARIANTS)
    def test_mo_variant_runs(self, tmp_dir, name, algo_cls):
        spec = make_spec(
            tmp_dir,
            name,
            algo_cls(pop_size=10),
            problem=ZDT1(),
            problem_name="ZDT1",
        )
        parallel_run([spec], tmp_dir, n_jobs=1)
        result = unzip_result(tmp_dir / get_spec_path(spec) / "result.pkl.gz")
        assert result.F.shape[1] == 2

    @pytest.mark.parametrize("name,algo_cls", SO_VARIANTS)
    def test_so_variant_runs(self, tmp_dir, name, algo_cls):
        spec = make_spec(
            tmp_dir,
            name,
            algo_cls(pop_size=10),
            problem=Sphere(n_var=5),
            problem_name="Sphere",
        )
        parallel_run([spec], tmp_dir, n_jobs=1)
        result = unzip_result(tmp_dir / get_spec_path(spec) / "result.pkl.gz")
        assert np.asarray(result.F).size == 1


# ── Reference front generation (NDS + FPS) ──────────────────────────────────


class TestNDSFarthestPointSurvival:
    def _make_pop(self, n=50, n_vars=5, n_obj=2):
        X = np.random.rand(n, n_vars)
        f1 = np.random.rand(n)
        f2 = 1 - f1 + np.random.rand(n) * 0.3
        F = np.column_stack([f1, f2])
        G = np.full((n, 1), -1.0)
        return PymooPopulation.new("X", X, "F", F, "G", G)

    def _make_problem(self, n_vars=5, n_obj=2, n_constr=1):
        class _Dummy(PymooProblem):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def _evaluate(self, x, out, *args, **kwargs):
                pass

        return _Dummy(n_var=n_vars, n_obj=n_obj, n_ieq_constr=n_constr)

    def test_returns_only_non_dominated_when_n_survive_large(self):
        from loares.operators.sorting import NDSFarthestPointSurvival
        from pymoo.util.nds.non_dominated_sorting import find_non_dominated

        np.random.seed(42)
        pop = self._make_pop(n=100, n_obj=2)
        prob = self._make_problem(n_vars=5, n_obj=2)
        F = pop.get("F")

        ndf_size = len(find_non_dominated(F))
        survival = NDSFarthestPointSurvival()
        survivors = survival.do(prob, pop, n_survive=ndf_size + 100)

        ndf = survivors[survivors.get("rank") == 0]
        ndf_F = ndf.get("F")
        expected_F = F[find_non_dominated(F)]

        for i in range(ndf_F.shape[0]):
            match = np.any(np.all(np.isclose(ndf_F[i], expected_F), axis=1))
            assert match, f"Point {ndf_F[i]} is not on the non-dominated front"

    def test_respects_n_survive(self):
        from loares.operators.sorting import NDSFarthestPointSurvival

        np.random.seed(42)
        pop = self._make_pop(n=200, n_obj=2)
        prob = self._make_problem(n_vars=5, n_obj=2)

        survival = NDSFarthestPointSurvival()
        survivors = survival.do(prob, pop, n_survive=15)
        assert len(survivors) == 15

    def test_fills_from_multiple_fronts(self):
        from loares.operators.sorting import NDSFarthestPointSurvival
        from pymoo.util.nds.non_dominated_sorting import find_non_dominated

        np.random.seed(42)
        pop = self._make_pop(n=100, n_obj=2)
        prob = self._make_problem(n_vars=5, n_obj=2)
        F = pop.get("F")

        ndf_size = len(find_non_dominated(F))
        n_survive = ndf_size + 10

        survival = NDSFarthestPointSurvival()
        survivors = survival.do(prob, pop, n_survive=n_survive)
        assert len(survivors) == n_survive

        ranks = survivors.get("rank")
        assert 0 in ranks
        assert np.max(ranks) >= 1


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
