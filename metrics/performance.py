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

def pindicators(truefront,objective_values,ref_point):
    fmax = truefront.max(axis=0)
    fmin = truefront.min(axis=0)

    obj_norm = normalize(objective_values,fmin,fmax)
    tf_norm = normalize(truefront,fmin,fmax)

    gd = GD(tf_norm)
    igd = IGD(tf_norm)
    spacing = SpacingIndicator()
    hv = HV(ref_point=ref_point)

    metrics = [gd(obj_norm),igd(obj_norm)]

    if objective_values.shape[0]>1:
        metrics.append(spacing(obj_norm))
        metrics.append(calculate_spread(obj_norm,tf_norm))
    else:
        metrics.append(np.nan)
        metrics.append(np.nan)
    
    metrics.append(hv(obj_norm))

    return obj_norm,tf_norm,metrics

def pindicators_notf(objective_values,ref_point):
    fmax = objective_values.max(axis=0)
    fmin = objective_values.min(axis=0)

    obj_norm = normalize(objective_values,fmin,fmax)

    spacing = SpacingIndicator()
    hv = HV(ref_point=ref_point)

    metrics = []

    if objective_values.shape[0]>1:
        metrics.append(spacing(obj_norm))
    else:
        metrics.append(np.nan)
    
    metrics.append(hv(obj_norm))

    return obj_norm,metrics

def hv(objective_values,ref_point):
    fmax = objective_values.max(axis=0)
    fmin = objective_values.min(axis=0)

    obj_norm = normalize(objective_values,fmin,fmax)

    hv = HV(ref_point=ref_point)

    hvs = np.zeros([objective_values.shape[0]])

    hv_total = hv(obj_norm)

    for i in range(0,objective_values.shape[0]):
        mask = np.arange(0,objective_values.shape[0]) != i
        hvs[i] = hv_total - hv(obj_norm[mask])

    return hvs

def gen_pindicators(objective_values,ref_point,truefront=None):
    if truefront is not None:
        fmax = truefront.max(axis=0)
        fmin = truefront.min(axis=0)

        obj_norm = normalize(objective_values,fmin,fmax)
        tf_norm = normalize(truefront,fmin,fmax)

        gd = GD(tf_norm)
        igd = IGD(tf_norm)
        spacing = SpacingIndicator()
        hv = HV(ref_point=ref_point)

        metrics = [gd(obj_norm),igd(obj_norm)]

        if objective_values.shape[0]>1:
            metrics.append(spacing(obj_norm))
            metrics.append(calculate_spread(obj_norm,tf_norm))
        else:
            metrics.append(np.nan)
            metrics.append(np.nan)
        
        metrics.append(hv(obj_norm))
    
    else:
        fmax = objective_values.max(axis=0)
        fmin = objective_values.min(axis=0)

        obj_norm = normalize(objective_values,fmin,fmax)

        spacing = SpacingIndicator()
        hv = HV(ref_point=ref_point)

        metrics = []

        if objective_values.shape[0]>1:
            metrics.append(spacing(obj_norm))
        else:
            metrics.append(np.nan)
        
        metrics.append(hv(obj_norm))

    return obj_norm,metrics