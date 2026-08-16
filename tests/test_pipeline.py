import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.gd import GD
from pymoo.problems.multi import ZDT1
from pymoo.problems.single import Sphere

from loares.algorithms.bxr.moo import MO_BMR, MO_BMWR, MO_BWR
from loares.algorithms.bxr.soo import SO_BMR, SO_BMWR, SO_BWR
from loares.indicator import indicator_multi_run
from loares.run import parallel_run
from loares.statistics import statistical_test_1
from loares.utils import get_spec_path


def make_spec(output_dir, *, algorithm_name="NSGA-II", algorithm=None, seed=1):
    problem = ZDT1()
    return {
        "algorithm_name": algorithm_name,
        "algorithm": algorithm or NSGA2(pop_size=20),
        "algorithm_kwargs": {"pop_size": 20},
        "problem_name": "ZDT1",
        "problem": problem,
        "output_dir": output_dir,
        "plot_kwargs": {},
        "solver_kwargs": {"seed": seed, "termination": ("n_eval", 40)},
    }


def test_parallel_run_writes_result_files_and_skips_completed_specs(tmp_path):
    spec = make_spec(tmp_path)
    result_path = get_spec_path(spec).with_suffix(".h5")

    parallel_run([spec], n_threads=1)

    assert result_path.exists()
    assert result_path.with_name(f"{result_path.stem}_opt.csv").exists()
    assert result_path.with_name(f"{result_path.stem}_opt.pdf").exists()

    with h5py.File(result_path, "r") as result:
        spec_info = json.loads(result["metadata"].attrs["spec_info"])
        problem_info = json.loads(result["metadata"].attrs["problem_info"])
        snapshots = result["function_evals"]

        assert spec_info["algorithm_name"] == "NSGA-II"
        assert spec_info["seed"] == 1
        assert problem_info["n_obj"] == 2
        assert len(snapshots) > 0

        final_snapshot = snapshots[sorted(snapshots.keys(), key=int)[-1]]
        assert final_snapshot["optimum"]["X"].shape[1] == problem_info["n_vars"]
        assert final_snapshot["optimum"]["F"].shape[1] == problem_info["n_obj"]

    modified_at = result_path.stat().st_mtime_ns
    parallel_run([spec], n_threads=1)
    assert result_path.stat().st_mtime_ns == modified_at


def test_indicator_multi_run_compiles_metrics_for_completed_runs(tmp_path):
    spec = make_spec(tmp_path)
    parallel_run([spec], n_threads=1)

    indicator_multi_run(
        [{"indicator_name": "GD", "indicator": GD(ZDT1().pareto_front(100))}],
        [spec],
        tmp_path,
        n_threads=1,
    )

    metrics = pd.read_csv(tmp_path / "metrics.csv")
    assert len(metrics) == 1
    assert metrics.loc[0, "algorithm_name"] == "NSGA-II"
    assert metrics.loc[0, "indicator_name"] == "GD"
    assert metrics.loc[0, "source"] == "optimum"
    assert np.isfinite(metrics.loc[0, "value"])


def test_statistical_test_writes_summary_and_effect_size_outputs(tmp_path):
    metrics_path = tmp_path / "metrics.csv"
    pd.DataFrame(
        [
            {
                "algorithm_name": algorithm,
                "problem_name": "ZDT1",
                "pop_size": 20,
                "seed": seed,
                "termination_metric": "n_eval",
                "termination_value": 40,
                "source": "optimum",
                "indicator_name": "GD",
                "value": value,
            }
            for algorithm, values in {
                "MO-BMR": [0.1, 0.2, 0.3],
                "MO-BWR": [0.2, 0.3, 0.1],
                "NSGA-II": [0.3, 0.1, 0.2],
            }.items()
            for seed, value in enumerate(values, start=1)
        ]
    ).to_csv(metrics_path, index=False)

    statistical_test_1(
        [
            {
                "filter": {
                    "indicator_name": "GD",
                    "pop_size": 20,
                    "termination_metric": "n_eval",
                    "termination_value": 40,
                    "source": "optimum",
                },
                "pivot": {
                    "index": "seed",
                    "columns": "algorithm_name",
                    "values": "value",
                },
            }
        ],
        metrics_path,
        tmp_path,
    )

    results_dir = tmp_path / "statistical-test-1"
    assert (results_dir / "friedman-results.csv").exists()
    assert (results_dir / "GD-a12.csv").exists()
    assert (results_dir / "GD-a12.pdf").exists()
    assert (results_dir / "GD-average-ranks.csv").exists()

    results = pd.read_csv(results_dir / "friedman-results.csv")
    assert results.loc[0, "indicator_name"] == "GD"
    assert results.loc[0, "Algorithms"] == 3


def test_bxr_algorithm_constructors_expose_expected_names():
    assert MO_BMR(pop_size=20).name == "MO-BMR"
    assert MO_BWR(pop_size=20).name == "MO-BWR"
    assert MO_BMWR(pop_size=20).name == "MO-BMWR"
    assert SO_BMR(pop_size=20).name == "SO-BMR"
    assert SO_BWR(pop_size=20).name == "SO-BWR"
    assert SO_BMWR(pop_size=20).name == "SO-BMWR"


def test_single_objective_algorithm_is_compatible_with_current_spec_format(tmp_path):
    spec = make_spec(
        tmp_path,
        algorithm_name="SO-BMR",
        algorithm=SO_BMR(pop_size=20),
    )
    spec["problem_name"] = "Sphere"
    spec["problem"] = Sphere(n_var=5)

    parallel_run([spec], n_threads=1)

    result_path = get_spec_path(spec).with_suffix(".h5")
    with h5py.File(result_path, "r") as result:
        final_snapshot = result["function_evals"][
            sorted(result["function_evals"].keys(), key=int)[-1]
        ]
        assert final_snapshot["optimum"]["F"].shape[1] == 1
