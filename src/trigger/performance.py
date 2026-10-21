"""Performance-monitoring refresh trigger (the brief: "monitor drift/performance"): uses the labels of the window just scored.

After window k has been scored, its labels are available (zero label delay, the assumption Phase 8 makes for every update). The monitor compares the window's PR-AUC with the PR-AUC of the
reference window, the first window scored after the start or after the latest refresh, and fires when it has fallen by more than ``delta`` in relative terms. Labels of window k are
used only here and in the refresh that follows; they are never used to score window k. The threshold is fixed before any arm runs from TRAIN days only
(experiments/phase8_triggered_refresh/PREREGISTRATION_TRP.md).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


class PerformanceTrigger:
    """``trigger(k, scores, labels) -> bool``; ``needs_labels`` tells the walk-forward function to pass the labels of the window just scored (never earlier or later labels)."""

    needs_labels = True

    def __init__(self, delta: float):
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        self.delta = float(delta)
        self.reference = None
        self.log: list = []

    def __call__(self, k: int, scores, labels) -> bool:
        scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
        if scores.shape != labels.shape:
            raise ValueError("scores and labels must have the same length")
        if labels.sum() == 0:
            raise ValueError("the window has no positive label; PR-AUC is undefined")
        ap = float(average_precision_score(labels, scores))
        if self.reference is None:
            self.reference = ap
            self.log.append({"window": k, "pr_auc": ap, "relative_drop": None, "fired": False, "note": "reference window (baseline after start or refresh)"})
            return False
        drop = (self.reference - ap) / self.reference
        fired = drop > self.delta
        self.log.append({"window": k, "pr_auc": ap, "relative_drop": float(drop), "fired": bool(fired)})
        if fired:
            self.reference = None
        return bool(fired)
