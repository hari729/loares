import h5py
import json

def _json_default(o):
    import numpy as np
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")

class Result():
    def __init__(self, problem_info, algo_info, seed):
        self.problem_info = problem_info
        self.algorithm_info = algo_info
        self.seed = seed
        self.history = {'pop':[],'evals':[]}

    def record(self, population, evals):
        self.history['pop'].append(population)
        self.history['evals'].append(evals)

    def stop(self, final_dict):
        self.population = self.history['pop'][-1]
        self.final_dict = final_dict

class ResultProcessor():
    def __init__(self):
        pass

    def get_metrics_history(self, result, performance_metrics, TF=None):
        metrics_history = {}
        for i,evals in enumerate(result.history['evals']):
            metrics = performance_metrics(result.history['pop'][i].objectives,
                                          TF)

            for key, value in metrics.items():
                metrics_history.setdefault(key, []).append(value)
            metrics_history.setdefault("evals", []).append(evals)

        return metrics_history

    def get_final_metric(self, result, performance_metrics, TF=None):
        return performance_metrics(result.history['pop'][-1].objectives, TF)

    def get_final_pop(self, result):
        return result.history['pop'][-1]

    def to_hdf5(self, results, path):
        with h5py.File(path, "w") as file:
            meta_grp = file.create_group("metadata")
            meta_grp.attrs["problem_info_json"] = json.dumps(results[0].problem_info)
            meta_grp.attrs["algorithm_info_json"] = json.dumps(results[0].algorithm_info)
            runs_grp = file.create_group("runs")
            for res in results:
                seed_grp = runs_grp.create_group(f"{int(res.seed):03d}")
                seed_grp.attrs["final_dict_json"] = json.dumps(res.final_dict, default=_json_default)
                fe_group = seed_grp.create_group("function_evals")
                for i,evals in enumerate(res.history['evals']):
                    grp = fe_group.create_group(f"{evals:06d}")
                    grp.create_dataset("X", data=res.history['pop'][i].solutions)
                    grp.create_dataset("F", data=res.history['pop'][i].objectives)
                    grp.create_dataset("G", data=res.history['pop'][i].constraints)
                # grp.create_dataset("M", data=population.metadata)
                #
    def from_hdf5(self, path):
        """
        Load all runs from one history.h5 and reconstruct Result objects.
        """
        from opti.core.population import Population  # local import avoids circulars
        results = []
        with h5py.File(path, "r") as file:
            meta_grp = file["metadata"]
            problem_info = json.loads(meta_grp.attrs["problem_info_json"])
            algorithm_info = json.loads(meta_grp.attrs["algorithm_info_json"])
            runs_grp = file["runs"]
            for seed_key in sorted(runs_grp.keys()):
                seed = int(seed_key)
                seed_grp = runs_grp[seed_key]
                fe_group = seed_grp["function_evals"]
                res = Result(problem_info, algorithm_info, seed)
                # eval groups are zero-padded strings, sort numerically
                eval_keys = sorted(fe_group.keys(), key=lambda k: int(k))
                for ek in eval_keys:
                    grp = fe_group[ek]
                    X = grp["X"][:]
                    F = grp["F"][:]
                    G = grp["G"][:]
                    pop = Population(X, F, G)
                    res.record(pop, int(ek))
                # reconstruct final fields
                if res.history["pop"]:
                    res.population = res.history["pop"][-1]
                    res.final_dict = json.loads(seed_grp.attrs["final_dict_json"])
                results.append(res)
        return results
