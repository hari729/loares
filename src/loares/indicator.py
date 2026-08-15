from loares.utils import (
    get_spec_path,
    get_spec_info,
    read_final_state,
    stream_snapshots,
)

from multiprocessing import Pool
import pandas as pd
import os


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
