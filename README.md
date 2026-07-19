loares

# Library for Optimisation Algorithm Research

[![PyPI](https://img.shields.io/pypi/v/loares)](https://pypi.org/project/loares/)

loares provides modular, composable optimisation algorithms built on top of [pymoo](https://pymoo.org/). It includes the BxR (Best-x-worst Recombination) family of algorithms, an experiment runner with multiprocessing support, and automated post-processing with performance metrics and statistical analysis.

## Installation

```bash
pip install loares
```

## Usage

### Running Experiments

```python
import numpy as np
from pymoo.problems.multi import ZDT1
from pymoo.algorithms.moo.nsga2 import NSGA2

from loares.algorithms.bxr.moo import MO_BMR, MO_BWR
from loares.experiments.pymoo_runner import ExperimentRunner, AlgoFactory

problem = ZDT1()
seeds = np.arange(1, 6)

algorithms = [
    AlgoFactory(MO_BMR, pop_size=100),
    AlgoFactory(MO_BWR, pop_size=100),
    AlgoFactory(NSGA2, pop_size=100),  # stock pymoo algorithms work too
]

for factory in algorithms:
    runner = ExperimentRunner(problem, factory, max_evals=25000, test_name="zdt1-test")
    runner.multi_run(seeds, threads=4)
```

### Post-Processing

```python
from loares.experiments.pymoo_process import PostProcess

pp = PostProcess(
    test_dir="zdt1-test/raw_data",
    algo_grps={"BXR": ["MO-BMR", "MO-BWR"], "common": ["NSGA2"]},
    true_front=ZDT1().pareto_front(500),
    gen_rf=False,
    plot_convergence=True,
    plot_pareto=True,
)
pp.run(threads=4)
```

Post-processing generates:
- Per-seed final metrics (GD, IGD, HV, Spacing)
- Mean convergence histories as Parquet files
- Convergence plots per algorithm group
- Pareto front plots (2D, 3D, or parallel coordinates)
- Net summary CSV across all algorithms

### Statistical Analysis

```python
from pathlib import Path
from loares.experiments.analysis.stats import run as run_stats

stats_dir = run_stats(Path("zdt1-test/analysis-2026-01-01-12-00-00/100"), alpha=0.05)
# Produces Friedman test results and post-hoc comparisons
```

## Architecture

loares algorithms are built from composable pymoo operators:

- **Recombination**: BxR operators (BMR, BWR, BMWR)
- **Pool Selection**: Best-worst selection from sorted population
- **Mutation**: Random reinitialisation
- **Mods**: Local search, opposition-based learning, edge boosting
- **Survival**: pymoo's RankAndCrowding (MOO) or FitnessSurvival (SOO)
- **Sub-populations**: Adaptive splitting via HV-based policy

Custom algorithm assembly:

```python
from loares.core.composable import ModularAlgorithm, RecombinationVariant
from loares.core.recombination import BMR
from loares.core.pool_selection import BestWorstSelection
from loares.core.mutation import RandomReinit
from loares.core.mods import LocalSearchMod

algo = ModularAlgorithm(
    name="Custom-BMR",
    pop_size=100,
    infill=RecombinationVariant(
        pool_selection=BestWorstSelection(),
        recombination=BMR(),
        mutation=RandomReinit(prob=0.5),
    ),
    mods=[LocalSearchMod()],
)
```

## Output Format

Experiment results are stored as HDF5 files with one file per seed:

```
test_name/raw_data/{algorithm}/{psize}-{max_evals}/
    seed_001.h5
    seed_002.h5
    ...
    Info.json
```

Each HDF5 file contains:
- `metadata/` -- problem info, algorithm info, seed
- `function_evals/{n_eval}/` -- X (decision variables), F (objectives), G (constraints) at each snapshot
- `final_dict_json` -- final population as JSON attribute

## Requirements

Python >= 3.11

## License

See LICENSE file for details.
