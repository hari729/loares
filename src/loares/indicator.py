from loares.utils import (
    update_manifest,
    tqdm_joblib,
)
from loares.plots import multi_line_plot

from joblib import Parallel, delayed
import pandas as pd
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from tqdm import tqdm

from loares.utils import unzip_result


def calculate_indicator(config):
    indicator_spec, run_spec = config
    result = unzip_result(run_spec["result_path"])
    spec_dict = {key: value for key, value in run_spec.items() if key != "result_path"}
    calculated = []
    calculated.append(
        {
            **spec_dict,
            "source": "opt",
            "indicator_name": indicator_spec["indicator_name"],
            "indicator_value": indicator_spec["indicator"](result.F),
        }
    )
    if result.archive is not None and len(result.archive) > 0:
        calculated.append(
            {
                **spec_dict,
                "source": "archive",
                "indicator_name": indicator_spec["indicator_name"],
                "indicator_value": indicator_spec["indicator"](result.archive.get("F")),
            }
        )
    return calculated


spec_key_cols = [
    "algorithm_name",
    "problem_name",
    "pop_size",
    "seed",
    "termination_metric",
    "termination_value",
    "indicator_name",
    "source",
]


def pending_indicators(indicator_specs, run_manifest, metrics_manifest):
    indicator_key_cols = [key for key in spec_key_cols if key != "source"]
    completed = set()
    if not metrics_manifest.empty:
        completed = set(metrics_manifest[indicator_key_cols].apply(tuple, axis=1))
    return [
        (indicator_spec, run_spec)
        for run_spec in run_manifest.to_dict("records")
        for indicator_spec in indicator_specs
        if tuple(
            run_spec[key] if key != "indicator_name" else indicator_spec[key]
            for key in indicator_key_cols
        )
        not in completed
    ]


def indicator_multi_run(indicator_specs, output_dir, n_jobs=4):
    output_dir = Path(output_dir)
    run_manifest = pd.read_csv(output_dir / "run_manifest.csv")
    run_manifest = run_manifest[run_manifest["error"].isna()]
    metrics_path = output_dir / "metrics_manifest.csv"
    metrics_manifest = (
        pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    )
    args = pending_indicators(indicator_specs, run_manifest, metrics_manifest)
    pbar = tqdm(
        desc="Indicators",
        total=len(args),
        ascii=" -",
        bar_format="{desc}: {bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    with tqdm_joblib(pbar):
        output = Parallel(n_jobs=n_jobs)(
            delayed(calculate_indicator)(arg) for arg in args
        )
    flat = [item for sublist in output for item in sublist]
    if flat:
        update_manifest(output_dir, flat, spec_key_cols, "metrics")


def calculate_indicator_history(config):
    """Compute indicator values at every recorded snapshot for one run."""
    indicator_spec, run_spec = config
    result = unzip_result(run_spec["result_path"])
    spec_dict = {
        key: value
        for key, value in run_spec.items()
        if key not in ("result_path", "save_history")
    }
    calculated = []
    for state in result.history:
        try:
            value = indicator_spec["indicator"](state.opt.get("F"))
        except (ValueError, IndexError):
            value = np.nan
        calculated.append(
            {
                **spec_dict,
                "source": "opt",
                "indicator_name": indicator_spec["indicator_name"],
                "evals": state.evaluator.n_eval,
                "indicator_value": value,
            }
        )
        if state.archive is not None and len(state.archive) > 0:
            try:
                value = indicator_spec["indicator"](state.archive.get("F"))
            except (ValueError, IndexError):
                value = np.nan
            calculated.append(
                {
                    **spec_dict,
                    "source": "archive",
                    "indicator_name": indicator_spec["indicator_name"],
                    "evals": state.evaluator.n_eval,
                    "indicator_value": value,
                }
            )
    return calculated


history_key_cols = spec_key_cols + ["evals"]


def compile_history(dir_path, new_rows, history_key_cols=history_key_cols):
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    existing_path = dir_path / "history.parquet"
    df = pd.read_parquet(existing_path) if existing_path.exists() else pd.DataFrame()
    new_df = pd.DataFrame(new_rows)
    if not df.empty:
        mask = (
            df[history_key_cols]
            .apply(tuple, axis=1)
            .isin(set(new_df[history_key_cols].apply(tuple, axis=1)))
        )
        df = df[~mask]
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_parquet(existing_path, index=False)


def pending_history(indicator_specs, run_manifest, existing_df):
    history_key_cols_no_source = [
        c for c in history_key_cols if c not in ("source", "evals")
    ]
    completed = set()
    if not existing_df.empty:
        completed = set(
            existing_df[history_key_cols_no_source]
            .drop_duplicates()
            .apply(tuple, axis=1)
        )
    return [
        (indicator_spec, run_spec)
        for run_spec in run_manifest.to_dict("records")
        for indicator_spec in indicator_specs
        if tuple(
            run_spec[key] if key != "indicator_name" else indicator_spec[key]
            for key in history_key_cols_no_source
        )
        not in completed
    ]


def indicator_history_multi_run(
    indicator_specs,
    output_dir,
    n_jobs=4,
):
    output_dir = Path(output_dir)
    run_manifest = pd.read_csv(output_dir / "run_manifest.csv")
    run_manifest = run_manifest[
        (run_manifest["error"].isna()) & run_manifest["save_history"]
    ]
    history_path = output_dir / "history.parquet"
    existing_df = (
        pd.read_parquet(history_path) if history_path.exists() else pd.DataFrame()
    )
    args = pending_history(indicator_specs, run_manifest, existing_df)
    pbar = tqdm(
        desc="Indicator history",
        total=len(args),
        ascii=" -",
        bar_format="{desc}: {bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    with tqdm_joblib(pbar):
        output = Parallel(n_jobs=n_jobs)(
            delayed(calculate_indicator_history)(arg) for arg in args
        )
    flat = [item for sublist in output for item in sublist]
    if flat:
        compile_history(output_dir, flat)


def _mean_line(group):
    """Interpolate each seed's curve in `group` onto a shared eval grid,
    then average -> (eval_grid, mean_values)."""
    eval_grid = np.sort(group["evals"].unique())
    seed_curves = [
        np.interp(eval_grid, seed_group["evals"], seed_group["indicator_value"])
        for _, seed_group in group.sort_values("evals").groupby("seed")
    ]
    return eval_grid, np.mean(seed_curves, axis=0)


mean_history_key_cols = [c for c in spec_key_cols if c != "seed"]


def calculate_mean_history(history_df, key_cols=mean_history_key_cols):
    """Collapse history.parquet's per-eval-per-seed rows into one
    interpolated mean curve per group (default group: everything except
    seed -- i.e. one curve per algorithm/problem/pop_size/termination/
    indicator/source combination). One row per group; 'evals' and
    'indicator_value' are stored as list columns rather than exploded, since
    a group's curve isn't the same length as any other group's."""
    rows = []
    for key_vals, group in history_df.groupby(key_cols):
        eval_grid, mean_curve = _mean_line(group)
        row = dict(zip(key_cols, key_vals))
        row["evals"] = eval_grid.tolist()
        row["indicator_value"] = mean_curve.tolist()
        rows.append(row)
    return pd.DataFrame(rows)


def compile_mean_history(
    dir_path,
    new_rows_df,
    key_cols=mean_history_key_cols,
    filename="mean_history.parquet",
):
    """Same dedup-on-key-cols shape as update_manifest/compile_history, but
    a matching key here means 'replace with the freshly recomputed curve'
    rather than 'skip' -- a mean curve can't be updated incrementally when
    a new seed lands, it has to be rederived from history.parquet in full,
    so every call overwrites whatever groups it touches."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    existing_path = dir_path / filename
    df = pd.read_parquet(existing_path) if existing_path.exists() else pd.DataFrame()
    if not df.empty:
        mask = (
            df[key_cols]
            .apply(tuple, axis=1)
            .isin(set(new_rows_df[key_cols].apply(tuple, axis=1)))
        )
        df = df[~mask]
    df = pd.concat([df, new_rows_df], ignore_index=True)
    df.to_parquet(existing_path, index=False)


def mean_history_multi_run(
    history_path,
    output_dir,
    key_cols=mean_history_key_cols,
    filename="mean_history.parquet",
):
    """Compile stage: reads raw history.parquet, collapses it to one mean
    curve per group, and persists that separately from the raw table.
    Run this once after indicator_history_multi_run (and again whenever
    you've added seeds/algorithms) -- plot_convergence then only ever reads
    the compiled table, no per-plot-call interpolation or seed-averaging.
    """
    df = pd.read_parquet(history_path)
    mean_df = calculate_mean_history(df, key_cols)
    compile_mean_history(Path(output_dir), mean_df, key_cols, filename)


def build_convergence_lines(df, plot_spec):
    """Filter mean_history.parquet by plot_spec['filter'], group by
    plot_spec.get('group_by', 'algorithm_name') -> one line per group.
    Pure lookup -- the mean/interpolation work already happened in
    mean_history_multi_run, this just reads the stored curves.
    """
    filt = plot_spec["filter"]
    mask = (df[list(filt.keys())] == pd.Series(filt)).all(axis=1)
    filtered = df[mask]
    group_col = plot_spec.get("group_by", "algorithm_name")

    xdata = [np.asarray(e) for e in filtered["evals"]]
    ydata = [np.asarray(v) for v in filtered["indicator_value"]]
    legend = filtered[group_col].tolist()

    return {
        "xdata": xdata,
        "ydata": ydata,
        "legend": legend,
        "xlabel": plot_spec.get("xlabel", "Function Evaluations"),
        "ylabel": plot_spec.get(
            "ylabel", filt.get("indicator_name", "indicator_value")
        ),
    }


def build_convergence_lines_for_algos(df, filt, algo_names, xlabel, ylabel):
    """Like build_convergence_lines, but the lines are picked explicitly by
    algorithm_name (in the given order) instead of by grouping every distinct
    value in the filtered set. `filt` should not itself constrain
    algorithm_name -- that's what algo_names is for."""
    data = build_convergence_lines(df, {"filter": filt, "group_by": "algorithm_name"})
    order = {name: i for i, name in enumerate(algo_names)}
    indexed = [
        (order.get(name, float("inf")), i, name)
        for i, name in enumerate(data["legend"])
    ]
    indexed.sort()
    keep = [(i, name) for rank, i, name in indexed if rank < float("inf")]
    return {
        "xdata": [data["xdata"][i] for i, _ in keep],
        "ydata": [data["ydata"][i] for i, _ in keep],
        "legend": [name for _, name in keep],
        "xlabel": xlabel,
        "ylabel": ylabel,
    }


def plot_convergence(
    plot_specs, mean_history_path, output_dir, filename="convergence.pdf"
):
    """Spec-driven convergence plotting. One multi-page PDF at
    output_dir/filename. Reads only the compiled mean_history.parquet --
    no recomputation, so re-plotting a different algo_grps combination or
    indicator selection is just a filter+lookup.

    Each entry in plot_specs is either:
      - a plain spec (filter + optional group_by) -> one page, one line per
        distinct group_by value.
      - a grouped spec with 'algo_grps': {group_name: [algo_names, ...],
        "common": [algo_names, ...]} -> one page per non-"common" group,
        showing that group's algorithms plus the "common" ones on the same
        axes. Mirrors the old PostProcess(algo_grps=...) behaviour, e.g.
        {"BMR": ["MO-BMR"], "BWR": ["MO-BWR"], "common": ["NSGA2"]} produces
        a BMR-vs-NSGA2 page and a separate BWR-vs-NSGA2 page. 'filter' in
        this case should not constrain algorithm_name -- the group lists do.
    """
    df = pd.read_parquet(mean_history_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_dir / filename) as pdf:
        for plot_spec in plot_specs:
            algo_grps = plot_spec.get("algo_grps")
            if algo_grps is None:
                data = build_convergence_lines(df, plot_spec)
                multi_line_plot(data, pdf)
                continue

            filt = plot_spec["filter"]
            xlabel = plot_spec.get("xlabel", "Function Evaluations")
            ylabel = plot_spec.get(
                "ylabel", filt.get("indicator_name", "indicator_value")
            )
            common = algo_grps.get("common", [])
            for grp_name, grp_algos in algo_grps.items():
                if grp_name == "common":
                    continue
                algo_names = grp_algos + common
                data = build_convergence_lines_for_algos(
                    df, filt, algo_names, xlabel, ylabel
                )
                multi_line_plot(data, pdf)
