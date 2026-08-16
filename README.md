# loares

Library for optimisation algorithm research built on [pymoo](https://pymoo.org/). loares provides BxR (best-x-worst recombination) algorithms plus a file-based experiment pipeline for running repeated trials, calculating indicators, and comparing algorithms statistically.

[![PyPI](https://img.shields.io/pypi/v/loares)](https://pypi.org/project/loares/)

## Installation

```bash
pip install loares
```

loares requires Python 3.11 or later.

## Quick Start

The following multi-objective ZDT1 experiment follows [`examples/v2.py`](examples/v2.py). It runs BxR variants and NSGA-II for five seeds, then writes their final-population metrics to `metrics.csv`.

```python
import inspect
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.gd import GD
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.problems.multi import ZDT1

from loares.algorithms.bxr.moo import MO_BMR, MO_BWR
from loares.indicator import indicator_multi_run
from loares.run import parallel_run

runs = 5
pop_size = 100
max_evals = 25_000
problem = ZDT1()
output_dir = Path(inspect.stack()[0].filename).resolve().parent

common = {
    "algorithm_kwargs": {"pop_size": pop_size},
    "problem_name": "ZDT1",
    "problem": problem,
    "output_dir": output_dir,
    "plot_kwargs": {},
}

algorithms = [
    {"algorithm_name": "MO-BMR", "algorithm": MO_BMR(pop_size=pop_size)},
    {"algorithm_name": "MO-BWR", "algorithm": MO_BWR(pop_size=pop_size)},
    {"algorithm_name": "NSGA-II", "algorithm": NSGA2(pop_size=pop_size)},
]

algorithm_specs = [
    {
        "solver_kwargs": {"seed": seed, "termination": ("n_eval", max_evals)},
        **common,
        **algorithm,
    }
    for seed in range(1, runs + 1)
    for algorithm in algorithms
]

parallel_run(algorithm_specs, n_threads=5)

true_front = problem.pareto_front(500)
indicator_specs = [
    {"indicator_name": "GD", "indicator": GD(true_front)},
    {"indicator_name": "IGD", "indicator": IGD(true_front)},
    {"indicator_name": "Spacing", "indicator": SpacingIndicator()},
    {"indicator_name": "HV", "indicator": HV(ref_point=[1.1, 1.1])},
]

indicator_multi_run(indicator_specs, algorithm_specs, output_dir)
```

`parallel_run` skips specs whose HDF5 result already exists. Pass `overwrite=True` to run every spec again.

## Statistical Analysis

Use the generated `metrics.csv` to build a seed-by-algorithm comparison for each indicator. The example below analyses hypervolume; use `GD`, `IGD`, or `Spacing` to analyse a different metric.

```python
from loares.statistics import statistical_test_1

stat_specs = [
    {
        "filter": {
            "indicator_name": "HV",
            "pop_size": pop_size,
            "termination_metric": "n_eval",
            "termination_value": max_evals,
            "source": "optimum",
        },
        "pivot": {"index": "seed", "columns": "algorithm_name", "values": "indicator_value"},
    }
]

statistical_test_1(stat_specs, output_dir / "metrics.csv", output_dir)
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

Preconfigured multi-objective BxR variants are available from `loares.algorithms.bxr.moo`:

- `MO_BMR`, `MO_BWR`, and `MO_BMWR`
- `MO_BMR_Opposition`, `MO_BWR_Opposition`, and `MO_BMWR_Opposition`
- `MO_BMR_Archive`, `MO_BWR_Archive`, and `MO_BMWR_Archive`
- `MO_BMR_S`, `MO_BWR_S`, and `MO_BMWR_S`

Single-objective variants `SO_BMR`, `SO_BWR`, and `SO_BMWR` are available from `loares.algorithms.bxr.soo`. Any compatible pymoo algorithm, such as `NSGA2`, can be used in an algorithm spec.

## Output

Each run creates an HDF5 result, a CSV containing the final optimum, and a PDF scatter plot at:

```
{output_dir}/{problem_name}/{termination_metric}-{termination_value}/
    {algorithm_name}/{pop_size}/seed_{seed:03d}.h5
```

The HDF5 file stores problem and spec metadata and the `X`, `F`, and `G` arrays for each saved evaluation. `indicator_multi_run` writes or updates `{output_dir}/metrics.csv`.

## License

See [LICENSE](LICENSE).
