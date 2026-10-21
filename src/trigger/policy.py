"""Label-free prediction-drift monitor and refresh trigger.

The monitor sees only model scores of the window just scored (never a label): the population stability index (PSI) of the current window's score distribution against a reference
window's score distribution, with bin edges taken from the reference's quantiles. The reference is the first window scored after the latest refresh (initially window 1), because a
refreshed model emits a different score distribution and must be compared with itself. The trigger fires when PSI exceeds a threshold fixed before any arm is run
(experiments/phase8_triggered_refresh/PREREGISTRATION_TR.md); after a refresh the reference is cleared and the next window re-baselines it (no trigger on that window).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-4


def psi(reference, current, n_bins: int = 10) -> float:
    """Population stability index of ``current`` against ``reference`` with ``n_bins`` bins whose edges are the reference quantiles. Identical samples give 0."""
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    if ref.size == 0 or cur.size == 0:
        raise ValueError("psi needs non-empty samples")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1]))
    p = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1) / ref.size
    q = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1) / cur.size
    p, q = np.clip(p, EPS, None), np.clip(q, EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


class ScorePSITrigger:
    """``trigger(k, window_scores) -> bool``: True means refresh the model now (after window k). Label-free by construction: the only inputs are the window index and scores."""

    def __init__(self, threshold: float, n_bins: int = 10):
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold, self.n_bins = float(threshold), n_bins
        self.reference = None
        self.log: list = []

    def __call__(self, k: int, scores) -> bool:
        scores = np.asarray(scores, dtype=np.float64)
        if self.reference is None:
            self.reference = scores
            self.log.append({"window": k, "psi": None, "fired": False, "note": "reference window (baseline after start or refresh)"})
            return False
        value = psi(self.reference, scores, self.n_bins)
        fired = value > self.threshold
        self.log.append({"window": k, "psi": value, "fired": bool(fired)})
        if fired:
            self.reference = None
        return bool(fired)
