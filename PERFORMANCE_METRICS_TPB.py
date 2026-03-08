import numpy as np
from typing import Any, Dict

# (keep compute_metrics, normalized_scores, composite_J, print_evaluation unchanged)
# I'll paste the full set here for a single replaceable block.

def compute_metrics(glucose, dt_minutes=5):
    g = np.asarray(glucose, dtype=float)
    g = g[~np.isnan(g)]
    if g.size == 0:
        raise ValueError("Empty glucose array after removing NaNs.")
    n = g.size
    total_minutes = n * dt_minutes
    mean_g = float(np.mean(g))
    sd_g   = float(np.std(g, ddof=0))
    cv = 100.0 * sd_g / mean_g if mean_g > 0 else float('inf')
    pct_TIR = 100.0 * np.sum((g >= 70.0) & (g <= 180.0)) / n
    pct_TBR = 100.0 * np.sum(g < 70.0) / n
    pct_TAR = 100.0 * np.sum(g > 180.0) / n
    below = np.maximum(0.0, 70.0 - g)
    AUC_below = float(np.sum(below) * dt_minutes)
    AUC_ref = 70.0 * total_minutes
    AUC_below_norm = float(AUC_below / max(1.0, AUC_ref))
    return {
        'mean_g': mean_g, 'sd_g': sd_g, 'cv': cv,
        'pct_TIR': pct_TIR, 'pct_TBR': pct_TBR, 'pct_TAR': pct_TAR,
        'AUC_below': AUC_below, 'AUC_below_norm': AUC_below_norm,
        'total_minutes': total_minutes, 'n_samples': n
    }

def normalized_scores(metrics,
                      G_ref=110.0, mean_deadband=5.0, mean_range=80.0,
                      CV_ref=36.0, TBR_ref=5.0, alpha_auc=0.6,
                      emphasize_hypo_pow: float = 1.0):
    mean_g = metrics['mean_g']
    cv = metrics['cv']
    mean_diff = max(0.0, abs(mean_g - G_ref) - mean_deadband)
    S_mean = float(np.clip(mean_diff / mean_range, 0.0, 1.0))
    S_TIR = float(np.clip(1.0 - metrics['pct_TIR']/100.0, 0.0, 1.0))
    s_tbr_freq = float(np.clip(metrics['pct_TBR'] / TBR_ref, 0.0, 1.0))
    s_tbr_auc = float(np.clip(metrics['AUC_below_norm'], 0.0, 1.0))
    S_TBR = alpha_auc * s_tbr_freq + (1.0 - alpha_auc) * s_tbr_auc
    S_TBR = float(np.clip(S_TBR, 0.0, 1.0))
    if emphasize_hypo_pow != 1.0:
        S_TBR = float(np.clip(S_TBR ** emphasize_hypo_pow, 0.0, 1.0))
    S_var = float(np.clip(cv / CV_ref, 0.0, 1.0))
    return {
        'S_mean': S_mean, 'S_TIR': S_TIR,
        'S_TBR': S_TBR, 'S_var': S_var
    }

def composite_J(scores, weights=None):
    if weights is None:
        weights = {'w_mean':0.20, 'w_TIR':0.25, 'w_TBR':0.35, 'w_var':0.20}
    w_mean = weights.get('w_mean', 0.20)
    w_TIR  = weights.get('w_TIR', 0.25)
    w_TBR  = weights.get('w_TBR', 0.35)
    w_var  = weights.get('w_var', 0.20)
    total_w = w_mean + w_TIR + w_TBR + w_var
    if total_w <= 0:
        raise ValueError("Sum of weights must be positive.")
    w_mean /= total_w; w_TIR /= total_w; w_TBR /= total_w; w_var /= total_w
    J = (w_mean * scores['S_mean'] +
         w_TIR  * scores['S_TIR'] +
         w_TBR  * scores['S_TBR'] +
         w_var  * scores['S_var'])
    return float(J)

def print_evaluation(metrics, scores, J, weights=None):
    print("=== CGM Evaluation Summary ===")
    print(f"Samples: {metrics['n_samples']}, Duration (min): {metrics['total_minutes']:.0f}")
    print(f"Mean glucose: {metrics['mean_g']:.2f} mg/dL   SD: {metrics['sd_g']:.2f}   %CV: {metrics['cv']:.2f}%")
    print(f"%TIR (70-180): {metrics['pct_TIR']:.2f}%   %TBR (<70): {metrics['pct_TBR']:.2f}%   %TAR (>180): {metrics['pct_TAR']:.2f}%")
    print(f"AUC_below_70: {metrics['AUC_below']:.1f} mg·min/dL   AUC_below_norm: {metrics['AUC_below_norm']:.3f}")
    print("--- Normalized sub-scores (0=good .. 1=bad) ---")
    print(f"S_mean: {scores['S_mean']:.3f}   S_TIR: {scores['S_TIR']:.3f}   S_TBR: {scores['S_TBR']:.3f}   S_var: {scores['S_var']:.3f}")
    if weights:
        print("Weights:", weights)
    else:
        print("Weights: default (w_mean=0.20, w_TIR=0.25, w_TBR=0.35, w_var=0.20)")
    print(f"Composite cost J: {J:.4f}")
    print("===============================")

def _extract_glucose_from_input(glucose_or_sim: Any):
    """
    Helper: if input is array-like -> return as np.array.
    If input is an object with glucose_history / glucose / glucose_array attr -> extract it.
    Otherwise raise ValueError with a helpful message.
    """
    # if it's array-like numeric, just return it
    if isinstance(glucose_or_sim, (list, tuple, np.ndarray)):
        return np.asarray(glucose_or_sim, dtype=float)
    # object with common attribute names
    for attr in ('glucose_history', 'glucose', 'glucose_array'):
        if hasattr(glucose_or_sim, attr):
            val = getattr(glucose_or_sim, attr)
            return np.asarray(val, dtype=float)
    # maybe the sim.run returned tuple (t, glucose, ...)
    if isinstance(glucose_or_sim, tuple) or isinstance(glucose_or_sim, list):
        # try to find a 1D numeric array inside
        for item in glucose_or_sim:
            if isinstance(item, (list, tuple, np.ndarray)):
                arr = np.asarray(item, dtype=float)
                if arr.ndim == 1:
                    return arr
    raise ValueError("evaluate_glucose_trace: input must be a 1D glucose array or an object with a "
                     "'glucose_history' / 'glucose' / 'glucose_array' attribute. "
                     "You passed type: " + repr(type(glucose_or_sim)))

def evaluate_glucose_trace(glucose_or_sim, dt_minutes=5.0,
                           weights=None, config=None,
                           emphasize_hypo_pow: float = 1.0,
                           verbose: bool = True) -> Dict[str, Any]:
    """
    High-level evaluate: accepts either:
      - a numpy array (1D) of glucose (mg/dL), or
      - a simulation object (e.g. sim or self) that has attribute glucose_history / glucose / glucose_array,
      - or a tuple returned by sim.run() that contains the glucose array.
    """
    glucose = _extract_glucose_from_input(glucose_or_sim)
    metrics = compute_metrics(glucose, dt_minutes=dt_minutes)

    cfg = {}
    if config:
        allowed = ['G_ref','mean_deadband','mean_range','CV_ref','TBR_ref','alpha_auc']
        for k in allowed:
            if k in config:
                cfg[k] = config[k]

    scores = normalized_scores(metrics,
                               G_ref=cfg.get('G_ref',110.0),
                               mean_deadband=cfg.get('mean_deadband',5.0),
                               mean_range=cfg.get('mean_range',80.0),
                               CV_ref=cfg.get('CV_ref',36.0),
                               TBR_ref=cfg.get('TBR_ref',5.0),
                               alpha_auc=cfg.get('alpha_auc',0.6),
                               emphasize_hypo_pow=emphasize_hypo_pow)

    J = composite_J(scores, weights=weights)

    if verbose:
        print_evaluation(metrics, scores, J, weights=weights)

    return {'metrics': metrics, 'scores': scores, 'J': J}