from datetime import datetime
import inspect
from pathlib import Path
import os
from multiprocessing import Pool
from tqdm import tqdm
from loares.experiments.utils import dict_to_csv
from loares.algorithms.moo.sorting import ranking_crowding, nds_fps
import pandas as pd
import numpy as np
from loares.core.population import Population
from loares.experiments.plots import (
    multi_line_plot,
    plot_2d,
    plot_3d,
    parallel_coordinates_plot,
)
from loares.metrics.moo import raw_performance_metrics
from loares.metrics.soo import bw_fitness
from loares.algorithms.moo.base import MOPopulationHandler
from loares.algorithms.soo.base import SOPopulationHandler
from loares.core.results import ResultProcessor


class post_process:
    def __init__(
        self,
        problem,
        test_name,
        psizes,
        algo_grps,
        true_f=None,
        gen_rf=False,
        rf_size=1000,
        plot_tf=False,
        plot_hist=False,
    ):
        self.problem = problem
        self.problem_info = problem.get_info()
        self.test_name = test_name
        self.psizes = psizes
        self.algo_grps = algo_grps
        self.true_f = true_f
        self.gen_rf = gen_rf
        self.rf_size = rf_size
        self.plot_tf = plot_tf
        self.plot_hist = plot_hist
        caller_frame = inspect.stack()[1]
        caller_dir = Path(caller_frame.filename).resolve().parent
        self.test_dir = caller_dir / test_name / "raw_data"
        if self.problem_info["n_obj"] > 1:
            self.populationHandler = MOPopulationHandler()
            self.metrics_calculator = raw_performance_metrics
            self.control_metric = "HV"
            self.recording_interval = 0.05
        else:
            self.populationHandler = SOPopulationHandler()
            self.metrics_calculator = bw_fitness
            self.control_metric = "best"
            self.recording_interval = 0.005

        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.result_dir = caller_dir / test_name / f"analysis-{self.timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        print(
            f"\nComparing {self.test_name} test for {self.problem_info['name']} "
            f"with RF = {self.gen_rf} and RF size = {self.rf_size}"
        )
        self.threads = 2
        self._per_algo_accumulator = []

    def extract_population_paths(self, test_dir, psize, evals):
        algo_paths = [a for a in test_dir.iterdir() if a.is_dir()]
        result_paths = []
        for path in algo_paths:
            result_paths.append(path / f"{psize}-{evals}")
        return result_paths

    @staticmethod
    def _get_seed_files(algo_dir):
        return sorted(algo_dir.glob("seed_*.h5"))

    def sort_pop_list(self, pareto_list):
        combined_pop = self.populationHandler.merge(pareto_list)
        composite_population_raw = Population(
            *nds_fps(
                self.problem,
                combined_pop,
                self.rf_size,
                ndf=True,
                seed=1,
            )
        )
        composite_population = self.populationHandler.get_refined(
            composite_population_raw
        )
        return composite_population

    def generate_rf(self, rf_path=None):
        all_result_paths = []
        for psize in self.psizes:
            result_paths = self.extract_population_paths(
                self.test_dir, psize, self.problem_info["max_evals"]
            )
            all_result_paths.extend(result_paths)

        rf_path = Path(self.result_dir.parent / "ref_front.npy")
        if rf_path.exists():
            print(f"Using Reference Front at {rf_path}")
            self.true_f = np.load(rf_path)
        else:
            print(f"Generating Reference Front")
            reference_pop = self.populationHandler.get_empty_pop(self.problem_info["n_vars"],
                                                                 self.problem_info["n_obj"],
                                                                 self.problem_info["n_constr"])
            for path in tqdm(all_result_paths):
                for sf in self._get_seed_files(path):
                    pop = ResultProcessor.read_final_population(sf)
                    reference_pop = self.sort_pop_list([reference_pop,pop])
            self.true_f = reference_pop.objectives
            np.save(rf_path, self.true_f)

    def _metrics_worker(self, hdf5_path):
        seed = ResultProcessor.read_seed(hdf5_path)
        metrics_history = {}
        final_metrics = None
        for evals, metrics in ResultProcessor.stream_metrics(
            hdf5_path, self.metrics_calculator, TF=self.true_f
        ):
            for key, value in metrics.items():
                metrics_history.setdefault(key, []).append(value)
            metrics_history.setdefault("evals", []).append(evals)
            final_metrics = metrics

        if final_metrics is not None:
            final_metrics["seed"] = seed
        metrics_history["seed"] = [seed]
        return metrics_history, final_metrics

    def run(self, psize):
        print(f"Running psize = {psize}")
        pop_dir = Path(self.result_dir / f"{psize}")
        os.makedirs(pop_dir / "parquets", exist_ok=True)

        result_paths = self.extract_population_paths(
            self.test_dir, psize, self.problem_info["max_evals"]
        )

        all_results = {}
        for path in result_paths:
            temp = {
                "Info": pd.read_json(path / "Info.json"),
            }
            name = temp["Info"]["Algorithm"]["name"]
            seed_files = self._get_seed_files(path)
            all_results[name] = {"Info": temp["Info"], "seed_files": seed_files}

        if self.true_f is None and self.problem_info["n_obj"] > 1:
            rf_path = Path(self.result_dir / "ref_front.npy")
            if rf_path.exists():
                self.true_f = np.load(rf_path)

        net_res = {}
        with Pool(processes=self.threads) as pool:
            for algo in all_results:
                seed_files = all_results[algo]["seed_files"]
                rows = pool.map(self._metrics_worker, seed_files)
                metrics_list = [h for h, _ in rows]
                final_metrics_per_run = [f for _, f in rows]

                algo_final_df = pd.DataFrame(final_metrics_per_run)
                algo_final_df.to_csv(
                    pop_dir / f"{algo}-final-metrics.csv",
                    index=False,
                    float_format="%.5f",
                )

                pareto_dir = pop_dir / "pareto_fronts" / algo
                os.makedirs(pareto_dir, exist_ok=True)

                if self.problem_info["n_obj"] > 1 and "HV" in algo_final_df.columns:
                    best_idx = np.argmax(algo_final_df["HV"])
                    best_seed_file = seed_files[best_idx]
                    plot_data = ResultProcessor.read_final_dict(best_seed_file)
                    minmax_flat = self.problem.minmax.flatten()
                    for j in range(self.problem.n_obj):
                        key = f"f{j + 1}"
                        if key in plot_data:
                            plot_data[key] = np.array(plot_data[key]) * minmax_flat[j]
                    _, algo_info, best_seed = ResultProcessor.read_metadata(
                        best_seed_file
                    )
                    dict_to_csv(
                        plot_data, pareto_dir, f"{algo_info['name']}-pareto-front"
                    )
                    plot_data["name"] = algo_info["name"]
                    plot_data["seed"] = best_seed
                    n_obj = self.problem_info["n_obj"]
                    if n_obj == 1:
                        pass
                    elif n_obj == 2:
                        plot_2d(plot_data, pareto_dir)
                    elif n_obj == 3:
                        plot_3d(plot_data, pareto_dir)
                    else:
                        parallel_coordinates_plot(plot_data, pareto_dir)

                metrics = metrics_list[0].keys()
                mean = {"name": f"{algo} (Mean)"}
                std = {"name": f"{algo} (Std)"}
                net = {
                    "Psize": all_results[algo]["Info"]["Problem"]["psize"],
                    "Max-evals": all_results[algo]["Info"]["Problem"]["max_evals"],
                }

                recording_interval = int(
                    all_results[algo]["Info"]["Problem"]["max_evals"]
                    * self.recording_interval
                )
                eval_grid = np.arange(
                    recording_interval,
                    all_results[algo]["Info"]["Problem"]["max_evals"] + 1,
                    recording_interval,
                )
                mean["evals"] = eval_grid
                std["evals"] = eval_grid
                convergence = {"name": f"{algo} (convergence pts)"}
                for m in metrics:
                    if m not in ["seed", "evals"]:
                        interpolated_values = []
                        for r in metrics_list:
                            interp_vals = np.interp(eval_grid, r["evals"], r[m])
                            interpolated_values.append(interp_vals)

                        values = np.array(interpolated_values, dtype=float)
                        mean[m] = np.mean(values, axis=0)
                        std[m] = np.std(values, axis=0)

                        convergence[m] = [np.nan, np.nan]

                        net[f"{m}(mean)"] = [mean[m][-1]]
                        net[f"{m}(std)"] = [std[m][-1]]

                all_results[algo]["mean-history"] = pd.DataFrame(mean)
                all_results[algo]["mean-history"].to_parquet(
                    pop_dir / "parquets" / f"{algo}-mean-history.parquet",
                    engine="pyarrow",
                )
                all_results[algo]["convergence-pts"] = pd.DataFrame(convergence)
                all_results[algo]["net-result"] = pd.DataFrame(net)

                net_res[algo] = pd.DataFrame(net)

        net_res = pd.concat(net_res, names=["Algorithm"]).reset_index(level=0)
        net_res.to_csv(f"{pop_dir}/net-results.csv", index=False, float_format="%.5f")

        self._per_algo_accumulator.append(net_res)

        for grp in self.algo_grps:
            if grp != "common":
                for m in metrics:
                    if m not in ["seed", "evals"]:
                        plot_data = {
                            "ydata": [],
                            "xdata": [],
                            "xlabel": "Function Evaluations",
                            "ylabel": f"{m}",
                            "point": [],
                            "legend": [],
                        }
                        for alg in self.algo_grps[grp] + self.algo_grps["common"]:
                            plot_data["ydata"].append(
                                all_results[alg]["mean-history"][m]
                            )
                            plot_data["xdata"].append(
                                all_results[alg]["mean-history"]["evals"]
                            )
                            plot_data["point"].append(
                                all_results[alg]["convergence-pts"][m]
                            )
                            plot_data["legend"].append(
                                all_results[alg]["Info"]["Algorithm"]["name"]
                            )
                        multi_line_plot(plot_data, pop_dir, f"{m}-{grp}")

    def multi_thread(self, threads=5):
        self.threads = threads
        self._per_algo_accumulator = []
        if self.true_f is None:
            if self.gen_rf and self.problem_info["n_obj"] > 1:
                self.generate_rf()
        for psize in self.psizes:
            self.run(psize)
        self._write_per_algo_csvs()
        return self.result_dir

    def _write_per_algo_csvs(self):
        if not self._per_algo_accumulator:
            return
        combined = pd.concat(self._per_algo_accumulator, ignore_index=True)
        per_algo_dir = self.result_dir / "per-algo"
        os.makedirs(per_algo_dir, exist_ok=True)
        for algo_name, group in combined.groupby("Algorithm"):
            group.to_csv(
                per_algo_dir / f"{algo_name}-net-results.csv",
                index=False,
                float_format="%.5f",
            )
        print(f"Per-algorithm CSVs saved to: {per_algo_dir}")

    def regen_convergence_plots(
        self,
        psize,
        overwrite=False,
        legend_fontsize=10,
        label_fontsize=14,
        tick_fontsize=12,
    ):
        """
        Re-generate convergence plots from saved parquet files with configurable font sizes.

        Follows the same grouping as run(): each group (BMR, BWR, BMWR) is plotted
        together with 'others' (NSGA2, NSGA3, etc.), producing 3 plots per metric.

        Parameters
        ----------
        psize : int
            Population size directory to read parquets from.
        overwrite : bool
            If True, save plots into the comparison directory (overwriting originals).
            If False (default), save into a 'replots/' subdirectory.
        legend_fontsize : int
            Font size for legend text (default 14).
        label_fontsize : int
            Font size for axis labels (default 16).
        tick_fontsize : int
            Font size for tick labels (default 12).
        """
        comparison_dir = Path(self.result_dir / f"{psize}")
        parquets_dir = comparison_dir / "parquets"

        if not parquets_dir.exists():
            raise FileNotFoundError(
                f"Parquets directory not found: {parquets_dir}. Generate metrics history first."
            )

        parquet_files = sorted(parquets_dir.glob("*-mean-history.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {parquets_dir}")

        history_dict = {}
        for pf in parquet_files:
            df = pd.read_parquet(pf, engine="pyarrow")
            algo_name = pf.name.replace("-mean-history.parquet", "")
            history_dict[algo_name] = df

        metrics = [c for c in df.columns if c not in ("name", "evals")]

        if overwrite:
            output_dir = comparison_dir
        else:
            output_dir = comparison_dir / "replots"
            os.makedirs(output_dir, exist_ok=True)

        for grp in self.algo_grps:
            if grp != "common":
                combined = self.algo_grps[grp] + self.algo_grps["common"]
                for m in metrics:
                    plot_data = {
                        "ydata": [history_dict[a][m] for a in combined],
                        "xdata": [history_dict[a]["evals"] for a in combined],
                        "xlabel": "Function Evaluations",
                        "ylabel": m,
                        "point": [[np.nan, np.nan] for _ in combined],
                        "legend": [a for a in combined],
                    }
                    multi_line_plot(
                        plot_data,
                        output_dir,
                        f"{m}-{grp}",
                        legend_fontsize=legend_fontsize,
                        label_fontsize=label_fontsize,
                        tick_fontsize=tick_fontsize,
                    )

        print(f"Replots saved to: {output_dir}")
