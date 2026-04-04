import numpy as np

from pymoo.indicators.hv import HV
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.util.normalization import normalize


def calculate_spread(pf, true_pf):
    if pf.shape[0] <= 1:
        return np.nan

    # Sort by first objective for consistency
    pf = pf[np.argsort(pf[:, 0])]

    # Distances between consecutive solutions
    d = np.linalg.norm(np.diff(pf, axis=0), axis=1)
    d_bar = np.mean(d)

    # Distance to boundary points of true PF
    df = np.linalg.norm(pf[0] - true_pf[np.argmin(true_pf[:, 0])])
    dl = np.linalg.norm(pf[-1] - true_pf[np.argmax(true_pf[:, 0])])

    delta = (df + dl + np.sum(np.abs(d - d_bar))) / (df + dl + (len(d)) * d_bar)
    return delta


def performance_metrics(problem, pareto_population):

    objective_values = pareto_population.objectives
    ref_point = np.ones(problem.n_obj) + 1e-5
    truefront = problem.get_true_front()
    metrics = {}

    if truefront is not None:
        fmax = truefront.max(axis=0)
        fmin = truefront.min(axis=0)

        obj_norm = normalize(objective_values, fmin, fmax)
        tf_norm = normalize(truefront, fmin, fmax)

        gd = GD(tf_norm)
        igd = IGD(tf_norm)
        spacing = SpacingIndicator()
        hv = HV(ref_point=ref_point)

        metrics["GD"] = gd(obj_norm)
        metrics["IGD"] = igd(obj_norm)

        if objective_values.shape[0] > 1:
            metrics["SPC"] = spacing(obj_norm)
            metrics["SPR"] = calculate_spread(obj_norm, tf_norm)
        else:
            metrics["SPC"] = np.nan
            metrics["SPR"] = np.nan

        metrics["HV"] = hv(obj_norm)

    else:
        fmax = objective_values.max(axis=0)
        fmin = objective_values.min(axis=0)

        obj_norm = normalize(objective_values, fmin, fmax)
        spacing = SpacingIndicator()
        hv = HV(ref_point=ref_point)

        if objective_values.shape[0] > 1:
            metrics["SPC"] = spacing(obj_norm)
        else:
            metrics["SPC"] = np.nan

        metrics["HV"] = hv(obj_norm)

    return metrics


# ── cdist-based GD/IGD (not yet wired in) ────────────────────────────────

from scipy.spatial.distance import cdist as _cdist


def gd_cdist(F, PF):
    """Generational Distance using scipy cdist.

    Parameters
    ----------
    F : np.ndarray, shape (N, M)
        Objective values of the approximation set.
    PF : np.ndarray, shape (P, M)
        Objective values of the reference Pareto front.

    Returns
    -------
    float
        Mean minimum Euclidean distance from each point in F to PF.
    """
    D = _cdist(F, PF, metric="euclidean")
    return float(np.mean(np.min(D, axis=1)))


def igd_cdist(F, PF):
    """Inverted Generational Distance using scipy cdist.

    Parameters
    ----------
    F : np.ndarray, shape (N, M)
        Objective values of the approximation set.
    PF : np.ndarray, shape (P, M)
        Objective values of the reference Pareto front.

    Returns
    -------
    float
        Mean minimum Euclidean distance from each point in PF to F.
    """
    D = _cdist(PF, F, metric="euclidean")
    return float(np.mean(np.min(D, axis=1)))


def raw_performance_metrics(objective_values, truefront):

    # ref_point = np.ones(objective_values.shape[1]) + 1e-5
    ref_point = np.ones(objective_values.shape[1]) + 0.1
    metrics = {}

    if truefront is not None:
        if objective_values.shape[0] == 0:
            metrics["HV"] = np.nan
            metrics["SPC"] = np.nan
            metrics["GD"] = np.nan
            metrics["IGD"] = np.nan
            return metrics

        fmax = truefront.max(axis=0)
        fmin = truefront.min(axis=0)

        obj_norm = normalize(objective_values, fmin, fmax)
        tf_norm = normalize(truefront, fmin, fmax)

        # gd = GD(tf_norm)
        # igd = IGD(tf_norm)
        spacing = SpacingIndicator()
        hv = HV(ref_point=ref_point)

        # metrics["GD"] = gd(obj_norm)
        # metrics["IGD"] = igd(obj_norm)
        metrics["GD"] = gd_cdist(obj_norm, tf_norm)
        metrics["IGD"] = igd_cdist(obj_norm, tf_norm)

        if objective_values.shape[0] > 1:
            metrics["SPC"] = spacing(obj_norm)
        else:
            metrics["SPC"] = np.nan

        metrics["HV"] = hv(obj_norm)

    else:
        if objective_values.shape[0] == 0:
            metrics["HV"] = np.nan
            metrics["SPC"] = np.nan
            return metrics

        fmax = objective_values.max(axis=0)
        fmin = objective_values.min(axis=0)

        obj_norm = normalize(objective_values, fmin, fmax)
        spacing = SpacingIndicator()
        hv = HV(ref_point=ref_point)

        if objective_values.shape[0] > 1:
            metrics["SPC"] = spacing(obj_norm)
        else:
            metrics["SPC"] = np.nan

        metrics["HV"] = hv(obj_norm)

    return metrics
