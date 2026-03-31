from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
from scipy import stats


DEFAULT_METRICS = ["GD", "IGD", "SPC", "SPR", "HV"]


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


def load_problem_data(final_metrics_dir: Path) -> pd.DataFrame:
    csv_files = sorted(final_metrics_dir.glob("*-final-metrics.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No '*-final-metrics.csv' files in {final_metrics_dir}"
        )

    all_data: list[pd.DataFrame] = []
    for csv_file in csv_files:
        algo_name = csv_file.stem.replace("-final-metrics", "")
        df = pd.read_csv(csv_file)
        df["Algorithm"] = algo_name
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


def build_pivot(problem_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = problem_df.pivot(index="seed", columns="Algorithm", values=metric)
    # Friedman requires complete blocks (same seeds across all algorithms).
    pivot = pivot.dropna(axis=0, how="any")
    return pivot


def run(
    final_metrics_dir: str | Path,
    alpha: float = 0.05,
    metrics: list[str] | None = None,
) -> Path:
    """Run Friedman + Conover-Holm + A12 pipeline on final-metrics CSVs.

    Parameters ----------
    final_metrics_dir : str or Path
        Directory containing ``*-final-metrics.csv`` files.
    alpha : float
        Significance threshold for the Friedman test (default 0.05).
    metrics : list[str] or None
        Metrics to analyse. ``None`` uses ``DEFAULT_METRICS``.

    Returns
    -------
    Path
        The ``statistical-results`` directory where outputs were saved.
    """
    final_metrics_dir = Path(final_metrics_dir).resolve()

    if not final_metrics_dir.exists() or not final_metrics_dir.is_dir():
        raise FileNotFoundError(f"Invalid final-metrics directory: {final_metrics_dir}")

    if metrics is None:
        metrics = DEFAULT_METRICS

    statistics_dir = final_metrics_dir / "statistical-results"
    statistics_dir.mkdir(parents=True, exist_ok=True)

    problem_df = load_problem_data(final_metrics_dir)

    friedman_rows: list[dict[str, float | str | int | bool]] = []

    with plt.rc_context({"figure.figsize": (12, 4),"axes.grid": False}):
        for metric in metrics:
            if metric not in problem_df.columns:
                continue

            pivot = build_pivot(problem_df, metric)
            n_blocks, n_algorithms = pivot.shape

            if n_blocks < 2 or n_algorithms < 2:
                friedman_rows.append(
                    {
                        "Metric": metric,
                        "Statistic": float("nan"),
                        "P-value": float("nan"),
                        "Significant": False,
                        "Blocks": n_blocks,
                        "Algorithms": n_algorithms,
                    }
                )
                continue

            statistic, p_value = stats.friedmanchisquare(
                *[pivot[col].to_numpy() for col in pivot.columns]
            )

            is_significant = bool(p_value < alpha)
            friedman_rows.append(
                {
                    "Metric": metric,
                    "Statistic": float(statistic),
                    "P-value": float(p_value),
                    "Significant": is_significant,
                    "Blocks": n_blocks,
                    "Algorithms": n_algorithms,
                }
            )

            ascending = metric != "HV"

            # Save average ranks for transparency regardless of significance.
            avg_ranks = (
                pivot.rank(axis=1, ascending=ascending, method="average")
                .mean(axis=0)
                .sort_values()
            )
            avg_ranks.to_csv(
                statistics_dir / f"{metric}-average-ranks.csv",
                header=["average_rank"],
                float_format="%.6f",
            )

            if not is_significant:
                continue

            # Recommended post-hoc for Friedman blocked design.
            posthoc = sp.posthoc_conover_friedman(pivot, p_adjust="holm")
            posthoc.to_csv(
                statistics_dir / f"{metric}-conover-holm.csv",
                float_format="%.6f",
            )

            # Descriptive effect size matrix.
            a12_matrix = compute_a12_matrix(pivot, ascending=ascending)
            a12_matrix.to_csv(
                statistics_dir / f"{metric}-a12.csv",
                float_format="%.3f",
            )

            # Average-rank diagram with non-significant cliques from Holm matrix.
            fig, ax = plt.subplots(figsize=(max(10, int(0.8 * n_algorithms)), 4))
            sp.critical_difference_diagram(avg_ranks, posthoc, ax=ax)
            ax.grid(False)
            ax.set_title(f"Average Ranks with Holm Cliques ({metric})")
            fig.tight_layout()
            fig.savefig(
                statistics_dir / f"{metric}-rank-cliques.pdf",
                bbox_inches="tight",
            )
            plt.close(fig)

    friedman_df = pd.DataFrame(friedman_rows)
    friedman_df.to_csv(
        statistics_dir / "friedman-results.csv",
        index=False,
        float_format="%.8f",
    )

    print(f"Saved statistical outputs to: {statistics_dir}")
    return statistics_dir


# ── CLI entry point ──────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Friedman + Conover-Holm pipeline from Loares final-metrics files."
        )
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=None,
        help=(
            "Path to one comparison population directory (contains final-metrics/). "
            "Example: /home/hari/OptiResults/IK-4/.../200"
        ),
    )
    parser.add_argument(
        "--final-metrics-dir",
        type=Path,
        default=None,
        help="Path to final-metrics directory directly.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for Friedman test.",
    )
    return parser.parse_args()


def _resolve_final_metrics_dir(args: argparse.Namespace) -> Path:
    if args.final_metrics_dir is not None:
        return args.final_metrics_dir
    if args.comparison_dir is not None:
        return args.comparison_dir / "final-metrics"
    raise ValueError("Provide either --comparison-dir or --final-metrics-dir.")


def main() -> None:
    args = _parse_args()
    final_metrics_dir = _resolve_final_metrics_dir(args)
    run(final_metrics_dir, alpha=args.alpha)


if __name__ == "__main__":
    main()
