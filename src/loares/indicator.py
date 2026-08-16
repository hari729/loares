from loares.utils import (
    get_spec_path,
    get_spec_info,
    read_final_state,
    stream_snapshots,
)
from loares.plots import multi_line_plot

from multiprocessing import Pool
import pandas as pd
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def calculate_indicator(config):
    indicator_specs, algorithm_spec, source = config
    final_state = read_final_state(get_spec_path(algorithm_spec).with_suffix(".h5"))
    spec_dict = get_spec_info(algorithm_spec)
    calculated = []
    for i_spec in indicator_specs:
        calculated.append(
            {
                **spec_dict,
                "source": source,
                "indicator_name": i_spec["indicator_name"],
                "value": i_spec["indicator"](final_state[source]["F"][:]),
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


def compile_metrics(dir_path, new_rows, spec_key_cols=spec_key_cols):
    dir_path.mkdir(parents=True, exist_ok=True)
    existing_path = dir_path / "metrics.csv"
    df = pd.read_csv(existing_path) if existing_path.exists() else pd.DataFrame()
    new_df = pd.DataFrame(new_rows)
    if not df.empty:
        mask = (
            df[spec_key_cols]
            .apply(tuple, axis=1)
            .isin(set(new_df[spec_key_cols].apply(tuple, axis=1)))
        )
        df = df[~mask]
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(existing_path, index=False)


def indicator_multi_run(
    indicator_specs,
    algorithm_specs,
    output_dir,
    n_threads=4,
    source="optimum",
):
    args = [(indicator_specs, a, source) for a in algorithm_specs]
    with Pool(processes=n_threads) as pool:
        output = pool.map(calculate_indicator, args)
    flat = [item for sublist in output for item in sublist]
    compile_metrics(output_dir, flat)


def calculate_indicator_history(config):
    """Compute indicator values at every recorded snapshot (not just the
    final state), for one algorithm_spec. Mirrors calculate_indicator but
    streams the full evaluation history."""
    indicator_specs, algorithm_spec, source = config
    spec_path = get_spec_path(algorithm_spec).with_suffix(".h5")
    spec_dict = get_spec_info(algorithm_spec)
    calculated = []
    for evals, grp in stream_snapshots(spec_path):
        if source not in grp:
            continue
        F = grp[source]["F"][:]
        for i_spec in indicator_specs:
            calculated.append(
                {
                    **spec_dict,
                    "source": source,
                    "indicator_name": i_spec["indicator_name"],
                    "evals": evals,
                    "value": i_spec["indicator"](F),
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


def indicator_history_multi_run(
    indicator_specs,
    algorithm_specs,
    output_dir,
    n_threads=4,
    source="optimum",
):
    """Same spec-driven shape as indicator_multi_run, but computes and
    stores the indicator value at every eval snapshot rather than only
    the final one. Written to history.parquet (columnar/typed and far
    smaller than a CSV at this row count -- one row per eval snapshot
    per indicator per seed)."""
    args = [(indicator_specs, a, source) for a in algorithm_specs]
    with Pool(processes=n_threads) as pool:
        output = pool.map(calculate_indicator_history, args)
    flat = [item for sublist in output for item in sublist]
    compile_history(Path(output_dir), flat)


def _mean_line(group):
    """Interpolate each seed's curve in `group` onto a shared eval grid,
    then average -> (eval_grid, mean_values)."""
    eval_grid = np.sort(group["evals"].unique())
    seed_curves = [
        np.interp(eval_grid, seed_group["evals"], seed_group["value"])
        for _, seed_group in group.sort_values("evals").groupby("seed")
    ]
    return eval_grid, np.mean(seed_curves, axis=0)


mean_history_key_cols = [c for c in spec_key_cols if c != "seed"]


def calculate_mean_history(history_df, key_cols=mean_history_key_cols):
    """Collapse history.parquet's per-eval-per-seed rows into one
    interpolated mean curve per group (default group: everything except
    seed -- i.e. one curve per algorithm/problem/pop_size/termination/
    indicator/source combination). One row per group; 'evals' and 'value'
    are stored as list columns rather than exploded, since a group's curve
    isn't the same length as any other group's."""
    rows = []
    for key_vals, group in history_df.groupby(key_cols):
        eval_grid, mean_curve = _mean_line(group)
        row = dict(zip(key_cols, key_vals))
        row["evals"] = eval_grid.tolist()
        row["value"] = mean_curve.tolist()
        rows.append(row)
    return pd.DataFrame(rows)


def compile_mean_history(
    dir_path,
    new_rows_df,
    key_cols=mean_history_key_cols,
    filename="mean_history.parquet",
):
    """Same dedup-on-key-cols shape as compile_metrics/compile_history, but
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
    ydata = [np.asarray(v) for v in filtered["value"]]
    legend = filtered[group_col].tolist()

    return {
        "xdata": xdata,
        "ydata": ydata,
        "legend": legend,
        "xlabel": plot_spec.get("xlabel", "Function Evaluations"),
        "ylabel": plot_spec.get("ylabel", filt.get("indicator_name", "value")),
    }


def build_convergence_lines_for_algos(df, filt, algo_names, xlabel, ylabel):
    """Like build_convergence_lines, but the lines are picked explicitly by
    algorithm_name (in the given order) instead of by grouping every distinct
    value in the filtered set. `filt` should not itself constrain
    algorithm_name -- that's what algo_names is for."""
    mask = (df[list(filt.keys())] == pd.Series(filt)).all(axis=1)
    filtered = df[mask]

    xdata, ydata, legend = [], [], []
    for name in algo_names:
        row = filtered[filtered["algorithm_name"] == name]
        if row.empty:
            continue
        r = row.iloc[0]
        xdata.append(np.asarray(r["evals"]))
        ydata.append(np.asarray(r["value"]))
        legend.append(name)

    return {
        "xdata": xdata,
        "ydata": ydata,
        "legend": legend,
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
            ylabel = plot_spec.get("ylabel", filt.get("indicator_name", "value"))
            common = algo_grps.get("common", [])
            for grp_name, grp_algos in algo_grps.items():
                if grp_name == "common":
                    continue
                algo_names = grp_algos + common
                data = build_convergence_lines_for_algos(
                    df, filt, algo_names, xlabel, ylabel
                )
                multi_line_plot(data, pdf)
