"""Phase 8 drift measurement (design Section 5 / 14.4).

Every function here is RETROSPECTIVE / reporting-only: each is computed strictly after the
window it describes has already been scored, using only that window's own already-produced
features or scores (feature/prediction drift) or already-revealed labels (performance drift,
computed downstream of scoring, exactly like every PR-AUC number in Phases 3-7A). None of these
functions is called by, or feeds a result back into, `update_loop.py`'s update decisions — the
update cadence is fixed and pre-registered (Section 6), independent of any drift value. No
function here ever reads a day from a window later than the one it is measuring, so no
future-window drift measurement can influence predictions for that same future window: a drift
value for window k is not computable until window k's own scores/labels already exist.

Pure numpy/scipy; no model, no dataset access, no I/O. Safe to call directly on small synthetic
arrays in a lightweight unit test.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI between two 1-D samples, binned on the REFERENCE sample's quantiles (standard PSI
    convention) so the bin edges don't depend on the (possibly drifted) current sample."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size == 0 or current.size == 0:
        return float("nan")
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(reference, quantiles))
    if edges.size < 2:
        return 0.0  # reference is constant; nothing to compare against
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_counts / max(reference.size, 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(current.size, 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (0 = identical distributions, 1 = maximally
    different). A thin, documented wrapper so every caller in this module uses the same test."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size == 0 or current.size == 0:
        return float("nan")
    return float(stats.ks_2samp(reference, current).statistic)


def feature_distribution_drift(reference_features: np.ndarray, window_features: np.ndarray) -> dict:
    """RETROSPECTIVE. Per-feature PSI + KS between `window_features` (window k's own transaction
    features — available at prediction time regardless of labels) and `reference_features`
    (the TRAIN reference distribution). `reference_features`/`window_features`: (n, d) arrays,
    same `d` feature columns. Returns per-feature arrays plus mean/max summaries."""
    reference_features = np.asarray(reference_features, dtype=float)
    window_features = np.asarray(window_features, dtype=float)
    if reference_features.ndim != 2 or window_features.ndim != 2:
        raise ValueError("expected 2-D (n_rows, n_features) arrays")
    if reference_features.shape[1] != window_features.shape[1]:
        raise ValueError("reference and window feature matrices must have the same number of columns")
    d = reference_features.shape[1]
    psi = np.array([population_stability_index(reference_features[:, j], window_features[:, j]) for j in range(d)])
    ks = np.array([ks_statistic(reference_features[:, j], window_features[:, j]) for j in range(d)])
    return {"timing": "retrospective", "n_features": d,
            "psi_per_feature": psi.tolist(), "ks_per_feature": ks.tolist(),
            "psi_mean": float(np.mean(psi)), "psi_max": float(np.max(psi)),
            "ks_mean": float(np.mean(ks)), "ks_max": float(np.max(ks))}


def prediction_distribution_drift(reference_scores: np.ndarray, window_scores: np.ndarray) -> dict:
    """RETROSPECTIVE. PSI + KS between a window's predicted-score distribution and a reference
    (either TRAIN-era scores, or the immediately preceding window's scores — caller's choice of
    `reference_scores`). Label-free: uses only already-produced model outputs."""
    psi = population_stability_index(reference_scores, window_scores)
    ks = ks_statistic(reference_scores, window_scores)
    return {"timing": "retrospective", "psi": psi, "ks": ks}


def performance_drift(window_metrics: list[dict], primary_key: str = "pr_auc") -> dict:
    """RETROSPECTIVE and label-dependent: `window_metrics` is a list of already-computed metrics
    dicts (one per window, in chronological order, each already containing e.g. `pr_auc` —
    i.e. each window's labels were already revealed by the time this is called, consistent with
    Section 14.2). Returns each window's absolute value and its delta from window 1."""
    if not window_metrics:
        return {"timing": "retrospective", "values": [], "deltas_from_window1": []}
    values = [float(m[primary_key]) for m in window_metrics]
    deltas = [v - values[0] for v in values]
    return {"timing": "retrospective", "metric": primary_key, "values": values,
            "deltas_from_window1": deltas, "first_to_last_change": values[-1] - values[0]}
