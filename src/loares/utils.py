import numpy as np
import h5py
import json
import pathlib
from pathlib import Path
from loares.plots import save_scatter_plots


def get_spec_path(spec):
    return Path(
        Path(spec["output_dir"])
        / spec["problem_name"]
        / f"{spec['solver_kwargs']['termination'][0]}-{spec['solver_kwargs']['termination'][1]}"
        / spec["algorithm_name"]
        / f"{spec['algorithm'].pop_size}"
        / f"seed_{int(spec['solver_kwargs']['seed']):03d}"
    )


def get_problem_info(spec):
    p = spec["problem"]
    bounds = "None"
    if p.xl is not None and p.xu is not None:
        bounds = str(np.column_stack([p.xl, p.xu]).tolist())
    return {
        "name": spec["problem_name"],
        "n_obj": int(p.n_obj),
        "n_vars": int(p.n_var),
        "n_constr": int(getattr(p, "n_ieq_constr", 0) + getattr(p, "n_eq_constr", 0)),
        "bounds": bounds,
        "minmax": getattr(p, "minmax", np.ones(p.n_obj)).tolist(),
    }


def get_spec_info(spec):
    return {
        "algorithm_name": spec["algorithm_name"],
        "problem_name": spec["problem_name"],
        "pop_size": spec["algorithm_kwargs"]["pop_size"],
        "seed": spec["solver_kwargs"]["seed"],
        "termination_metric": spec["solver_kwargs"]["termination"][0],
        "termination_value": spec["solver_kwargs"]["termination"][1],
        "output_dir": str(spec["output_dir"]),
    }


def json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pathlib.PosixPath,)):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


def make_final_dict(res):
    if res.archive is not None and len(res.archive) > 0:
        source = res.archive
    else:
        source = res.opt
    X = source.get("X")
    F = source.get("F")
    G = source.get("G")
    labels = (
        [f"x{i + 1}" for i in range(X.shape[1])]
        + [f"f{j + 1}" for j in range(F.shape[1])]
        + [f"g{k + 1}" for k in range(G.shape[1])]
    )
    combined = np.hstack([X, F, G])
    return {name: combined[:, idx] for idx, name in enumerate(labels)}


def write_results(res, spec):
    spec_path = get_spec_path(spec)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(spec_path.with_suffix(".h5"), "w") as h5:
        meta = h5.create_group("metadata")
        meta.attrs["spec_info"] = json.dumps(get_spec_info(spec), default=json_default)
        meta.attrs["problem_info"] = json.dumps(
            get_problem_info(spec), default=json_default
        )
        fe = h5.create_group("function_evals")

        for state in res.history:
            grp = fe.create_group(f"{state.evaluator.n_eval:06d}")
            sources = {"optimum": state.opt}
            if state.archive is not None and len(state.archive) > 0:
                sources["archive"] = state.archive
            for key, value in sources.items():
                igrp = grp.create_group(key)
                igrp.create_dataset("X", data=value.get("X"))
                igrp.create_dataset("F", data=value.get("F"))
                igrp.create_dataset("G", data=value.get("G"))

    X = res.opt.get("X")
    F = res.opt.get("F")
    G = res.opt.get("G")
    header = ",".join(
        [f"x{i + 1}" for i in range(X.shape[1])]
        + [f"f{j + 1}" for j in range(F.shape[1])]
        + [f"g{k + 1}" for k in range(G.shape[1])]
    )
    np.savetxt(
        f"{spec_path}_opt.csv",
        np.hstack([X, F, G]),
        delimiter=",",
        header=header,
        comments="",
    )
    save_scatter_plots(F, spec, f"{spec_path}_opt.pdf")

    if res.archive is not None and len(res.archive) > 0:
        X = res.archive.get("X")
        F = res.archive.get("F")
        G = res.archive.get("G")
        header = ",".join(
            [f"x{i + 1}" for i in range(X.shape[1])]
            + [f"f{j + 1}" for j in range(F.shape[1])]
            + [f"g{k + 1}" for k in range(G.shape[1])]
        )
        np.savetxt(
            f"{spec_path}_archive.csv",
            np.hstack([X, F, G]),
            delimiter=",",
            header=header,
            comments="",
        )

        save_scatter_plots(F, spec, f"{spec_path}_archive.pdf")


def read_final_state(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        fe = f["function_evals"]
        last_key = sorted(fe.keys(), key=lambda k: int(k))[-1]
        grp = fe[last_key]
        return {
            source: {k: grp[source][k][:] for k in grp[source].keys()}
            for source in grp.keys()
        }


def stream_snapshots(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        fe = f["function_evals"]
        for ek in sorted(fe.keys(), key=lambda k: int(k)):
            yield int(ek), fe[ek]
