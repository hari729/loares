# loares

Library for optimisation algorithm research built on [pymoo](https://pymoo.org/). loares provides BxR (best-x-worst recombination) algorithms plus a file-based experiment pipeline for running repeated trials, calculating quality indicators, and comparing algorithms statistically.

[![PyPI](https://img.shields.io/pypi/v/loares)](https://pypi.org/project/loares/)

## Installation

```bash
pip install loares
```

loares requires Python 3.11 or later.

## Quick Start

The following multi-objective ZDT1 experiment follows [`examples/v2.py`](examples/v2.py). It runs BxR variants and NSGA-II, then computes quality indicators and performs statistical analysis.

```python
import inspect
from pathlib import Path

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.gd import GD
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.problems.multi import ZDT1

from loares.algorithms.moo.mobxr_cd import MO_BMR_CD, MO_BWR_CD, MO_BMWR_CD
from loares.indicator import indicator_multi_run
from loares.run import parallel_run
from loares.statistics import statistical_test_1

runs = 5
pop_size = 100
max_evals = 25_000
problem = ZDT1()
output_dir = Path(inspect.stack()[0].filename).resolve().parent / "output"

common = {
    "algorithm_kwargs": {"pop_size": pop_size},
    "problem_name": "ZDT1",
    "problem": problem,
}

algorithms = [
    {"algorithm_name": "MO-BMR", "algorithm": MO_BMR_CD},
    {"algorithm_name": "MO-BWR", "algorithm": MO_BWR_CD},
    {"algorithm_name": "MO-BMWR", "algorithm": MO_BMWR_CD},
    {"algorithm_name": "NSGA-II", "algorithm": NSGA2},
]

algorithm_specs = [
    {
        "solver_kwargs": {
            "seed": seed,
            "termination": ("n_eval", max_evals),
            "save_history": True,
        },
        **common,
        **algorithm,
    }
    for seed in range(1, runs + 1)
    for algorithm in algorithms
]

parallel_run(algorithm_specs, output_dir, n_jobs=5)

true_front = problem.pareto_front(500)
indicator_specs = [
    {"indicator_name": "GD", "indicator": GD(true_front)},
    {"indicator_name": "IGD", "indicator": IGD(true_front)},
    {"indicator_name": "Spacing", "indicator": SpacingIndicator()},
    {"indicator_name": "HV", "indicator": HV(ref_point=[1.1, 1.1])},
]

indicator_multi_run(indicator_specs, output_dir, n_jobs=5)
```

`parallel_run` skips specs whose result already exists. Pass `overwrite=True` to run every spec again.

## Statistical Analysis

Use the generated `metrics_manifest.csv` to build a seed-by-algorithm comparison for each indicator. The example below analyses hypervolume; use `GD`, `IGD`, or `Spacing` to analyse a different metric.

```python
stat_specs = [
    {
        "filter": {
            "indicator_name": "HV",
            "pop_size": pop_size,
            "termination_metric": "n_eval",
            "termination_value": max_evals,
            "source": "opt",
        },
        "pivot": {"index": "seed", "columns": "algorithm_name", "values": "indicator_value"},
    }
]

statistical_test_1(stat_specs, output_dir / "metrics_manifest.csv", output_dir)
```

Results are written under `statistical-test-1/`, including Friedman results, average ranks, and Vargha-Delaney A12 matrices.

### Reading the heatmaps

Both the A12 and Conover-Holm heatmaps are annotated with the value in each cell and mark
statistically significant comparisons (p < alpha) with a `*` next to the value.

The A12 heatmap additionally encodes the *direction* of each comparison. A12 > 0.5 always
means the row algorithm is better (the direction is normalised per indicator, e.g. lower
is better for `GD`/`IGD`/`Spacing`, higher is better for `HV`), and this is shown in two
ways: the cell is coloured red (blue when the column algorithm is better, white for a tie
at 0.5) and a marker is drawn to the right of the value — `^` when the row wins, `v` when
the column wins.

## Algorithms

Preconfigured multi-objective BxR variants are available from `loares.algorithms.moo.mobxr_cd`:

- `MO_BMR_CD` -- Best-Mean-Recombination with Rank-and-Crowding survival
- `MO_BWR_CD` -- Best-Worst-Recombination with Rank-and-Crowding survival
- `MO_BMWR_CD` -- Best-Mean-Worst-Recombination with Rank-and-Crowding survival

Any compatible pymoo algorithm, such as `NSGA2`, can be used in an algorithm spec.

## Output

Each run creates a gzipped pickle result and a run manifest at:

```
{output_dir}/{problem_name}/{termination_metric}-{termination_value}/
    {algorithm_name}/{pop_size}/seed_{seed:03d}/result.pkl.gz
```

The result file stores the full pymoo `minimize` result object including `F`, `X`, and optionally `history`. `indicator_multi_run` writes or updates `{output_dir}/metrics_manifest.csv`.

## License

See [LICENSE](LICENSE).
