from loares import indicator
import pandas as pd
from pathlib import Path
import scikit_posthocs as sp
from scipy import stats

from loares.plots import save_heatmap


def build_pivot(df, stat_specs):
    mask = (
        df[list(stat_specs["filter"].keys())] == pd.Series(stat_specs["filter"])
    ).all(axis=1)
    filtered = df[mask]
    pivot = filtered.pivot(**stat_specs["pivot"])
    # Friedman requires complete blocks (same seeds across all algorithms).
    pivot = pivot.dropna(axis=0, how="any")
    return pivot


def vargha_delaney_a12(x: pd.Series, y: pd.Series) -> float:
    m, n = len(x), len(y)
    u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return float(u_stat / (m * n))


def compute_a12_matrix(pivot_df: pd.DataFrame, ascending: bool) -> pd.DataFrame:
    algorithms = pivot_df.columns.tolist()
    a12_matrix = pd.DataFrame(index=algorithms, columns=algorithms, dtype=float)

    for algo1 in algorithms:
        for algo2 in algorithms:
            if algo1 == algo2:
                a12_matrix.loc[algo1, algo2] = 0.5
                continue

            x = pivot_df[algo1]
            y = pivot_df[algo2]

            if ascending:
                a12 = vargha_delaney_a12(y, x)
            else:
                a12 = vargha_delaney_a12(x, y)

            a12_matrix.loc[algo1, algo2] = a12

    return a12_matrix


def friedman_connover_holm(pivot, alpha=0.05):
    n_blocks, n_algorithms = pivot.shape

    result = {"Blocks": n_blocks, "Algorithms": n_algorithms}
    posthoc = None
    if n_blocks < 2 or n_algorithms < 2:
        result["Statistic"] = float("nan")
        result["P-value"] = float("nan")
    else:
        statistic, p_value = stats.friedmanchisquare(
            *[pivot[col].to_numpy() for col in pivot.columns]
        )

        result["Statistic"] = statistic
        result["P-value"] = p_value

        if p_value < alpha:
            # Recommended post-hoc for Friedman blocked design.
            posthoc = sp.posthoc_conover_friedman(pivot, p_adjust="holm")

    return result, posthoc


def statistical_test_1(stat_specs, input_csv, output_dir, alpha=0.05):
    df = pd.read_csv(input_csv)
    statistics_dir = Path(output_dir) / "statistical-test-1"
    statistics_dir.mkdir(parents=True, exist_ok=True)
    friedman_rows = []
    for spec in stat_specs:
        pivot = build_pivot(df, spec)
        indicator_name = spec["filter"]["indicator_name"]
        ascending = indicator_name != "HV"

        significance, posthoc = friedman_connover_holm(pivot, alpha=alpha)
        sig_matrix = None
        if posthoc is not None:
            sig_matrix = posthoc.to_numpy(dtype=float) < alpha

        # Descriptive effect size matrix. a12 > 0.5 always means the row
        # algorithm is better (compute_a12_matrix flips per indicator), so
        # the diverging colormap and glyphs both encode comparison direction.
        a12_matrix = compute_a12_matrix(pivot, ascending=ascending)
        a12_matrix.to_csv(
            statistics_dir / f"{indicator_name}-a12.csv",
            float_format="%.3f",
        )

        algorithms = a12_matrix.columns.tolist()
        save_heatmap(
            a12_matrix.to_numpy(dtype=float),
            algorithms,
            algorithms,
            statistics_dir / f"{indicator_name}-a12.pdf",
            annotate=True,
            cmap="RdBu_r",
            reverse=False,
            significance=sig_matrix,
            glyph=True,
            title=f"{indicator_name} A12",
        )
        # Save average ranks for transparency regardless of significance.
        average_ranks = (
            pivot.rank(axis=1, ascending=ascending, method="average")
            .mean(axis=0)
            .sort_values()
        )
        average_ranks.to_csv(
            statistics_dir / f"{indicator_name}-average-ranks.csv",
            header=["average_rank"],
            float_format="%.6f",
        )

        significance["indicator_name"] = indicator_name
        friedman_rows.append(significance)

        if posthoc is not None:
            posthoc.to_csv(
                statistics_dir / f"{indicator_name}-conover-holm.csv",
                float_format="%.6f",
            )

            save_heatmap(
                posthoc.to_numpy(dtype=float),
                algorithms,
                algorithms,
                statistics_dir / f"{indicator_name}-conover-holm.pdf",
                annotate=True,
                fmt=".4f",
                cmap="Oranges",
                significance=sig_matrix,
                title=f"{indicator_name} Conover-Holm",
            )
    friedman_df = pd.DataFrame(friedman_rows)
    friedman_df.to_csv(
        statistics_dir / "friedman-results.csv",
        index=False,
        float_format="%.8f",
    )
