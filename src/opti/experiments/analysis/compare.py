from pathlib import Path
import sys
import pandas as pd

# Define which metrics to minimize vs maximize (for mean values)
# Only HV(mean) is higher is better, all others are lower is better
MINIMIZE_MEAN_METRICS = ["GD(mean)", "IGD(mean)", "SPC(mean)", "SPR(mean)"]
MAXIMIZE_MEAN_METRICS = ["HV(mean)"]

# All std metrics - lower is better (more consistent)
STD_METRICS = ["GD(std)", "IGD(std)", "SPC(std)", "SPR(std)", "HV(std)"]
STD_METRICS = []


def get_best_algorithm(df: pd.DataFrame, metric: str, minimize: bool = True) -> str:
    """
    Find the algorithm with the best value for a given metric.

    Args:
        df: DataFrame with algorithm results
        metric: Column name of the metric
        minimize: If True, lower is better; if False, higher is better

    Returns:
        Name of the best performing algorithm
    """
    if metric not in df.columns:
        return "N/A"

    if minimize:
        idx = df[metric].idxmin()
    else:
        idx = df[metric].idxmax()

    # return df.loc[idx, "Algorithm"]
    return df.loc[idx, "Algorithm"], df.loc[idx, metric]


def process_comparison_folder(comparison_folder: Path) -> pd.DataFrame:
    """
    Process a comparison folder and find best algorithms for each population size.

    Args:
        comparison_folder: Path to the comparison folder

    Returns:
        DataFrame with population sizes and best algorithms per metric
    """
    results = []

    # Find all population subfolders
    population_folders = sorted(
        [f for f in comparison_folder.iterdir() if f.is_dir() and f.name.isdigit()],
        key=lambda x: int(x.name),
    )

    if not population_folders:
        print(f"  No population subfolders found in {comparison_folder}")
        return pd.DataFrame()

    for pop_folder in population_folders:
        net_results_path = pop_folder / "net-results.csv"

        if not net_results_path.exists():
            print(f"  Warning: net-results.csv not found in {pop_folder}")
            continue

        df = pd.read_csv(net_results_path)

        # Get population size and max evals from the data
        psize = int(pop_folder.name)

        # Try to get Max-evals from the data (use first row's value)
        max_evals = df["Max-evals"].iloc[0] if "Max-evals" in df.columns else "N/A"

        row = {
            "Population": psize,
            "Max-evals": max_evals,
        }

        # Find best algorithm for metrics to minimize (mean)
        for metric in MINIMIZE_MEAN_METRICS:
            best_algo, best_value = get_best_algorithm(df, metric, minimize=True)
            row[metric] = best_algo
            row[f"{metric}_value"] = best_value
        # Find best algorithm for metrics to maximize (mean)
        for metric in MAXIMIZE_MEAN_METRICS:
            best_algo, best_value = get_best_algorithm(df, metric, minimize=False)
            row[metric] = best_algo
            row[f"{metric}_value"] = best_value
        # Find best algorithm for std metrics (always minimize - lower variance is better)
        for metric in STD_METRICS:
            best_algo, best_value = get_best_algorithm(df, metric, minimize=True)
            row[metric] = best_algo
            row[f"{metric}_value"] = best_value
        results.append(row)

    # Create DataFrame with results
    if results:
        result_df = pd.DataFrame(results)
        # Order columns to match net-results.csv structure
        cols = [
            "Population",
            "Max-evals",
            "GD(mean)",
            "GD(mean)_value",
            "GD(std)",
            "GD(std)_value",
            "IGD(mean)",
            "IGD(mean)_value",
            "IGD(std)",
            "IGD(std)_value",
            "SPC(mean)",
            "SPC(mean)_value",
            "SPC(std)",
            "SPC(std)_value",
            "SPR(mean)",
            "SPR(mean)_value",
            "SPR(std)",
            "SPR(std)_value",
            "HV(mean)",
            "HV(mean)_value",
            "HV(std)",
            "HV(std)_value",
        ]
        result_df = result_df[[c for c in cols if c in result_df.columns]]
        return result_df

    return pd.DataFrame()


def compare_metrics(problem_name, compare_dir_path):
    """Main function to process a specific test's comparison folder."""

    # Build the comparison folder path
    comparison_folder = Path(compare_dir_path)

    if not comparison_folder.exists():
        print(f"Error: Comparison folder not found: {comparison_folder}")
        sys.exit(1)

    if not comparison_folder.is_dir():
        print(f"Error: Path is not a directory: {comparison_folder}")
        sys.exit(1)

    print(f"Processing: {comparison_folder}")

    result_df = process_comparison_folder(comparison_folder)

    if result_df.empty:
        print(f"  No valid data found in {comparison_folder}")
        sys.exit(1)

    # Save the result CSV just outside the comparison folder
    # Name it the same as the comparison folder name
    output_name = comparison_folder.name + ".csv"
    output_path = comparison_folder / output_name

    result_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"\nResults:")
    print(result_df.to_string(index=False))
