"""The locked Phase 8 H4 decision rules as a pure function (same logic as scripts/compare_phase8.py), for the Phase 8 supplement and the held-out amendment.

Input: per-arm, per-seed lists of the five window PR-AUCs. Rules: (1) Ad degrades less (first-to-last window drop) than F in at least 2 of 3 seeds; (2) mean(Ad - F) of the
window-mean PR-AUC is positive and exceeds the larger of the two arms' seed SDs (ddof = 1); (3) the same for Ad against E. No p-values; nothing is tuned.
"""
from __future__ import annotations

import numpy as np

SEEDS = (42, 123, 2024)


def window_mean(per_seed: dict) -> dict:
    return {s: float(np.mean(w)) for s, w in per_seed.items()}


def degradation(per_seed: dict) -> dict:
    return {s: float(w[0] - w[-1]) for s, w in per_seed.items()}


def _mean_diff_exceeds_larger_sd(a: dict, b: dict):
    av, bv = np.array([a[s] for s in SEEDS]), np.array([b[s] for s in SEEDS])
    d = av - bv
    sd = max(av.std(ddof=1), bv.std(ddof=1))
    return float(d.mean()), float(sd), bool(d.mean() > sd)


def h4_decision(per_arm: dict) -> dict:
    """per_arm = {"F": {seed: [w1..w5]}, "Ad": {...}, "E": {...}} -> decision record."""
    for arm in ("F", "Ad", "E"):
        if set(per_arm[arm]) != set(SEEDS) or any(len(w) != 5 for w in per_arm[arm].values()):
            raise ValueError(f"arm {arm} must hold five window values for each of the seeds {SEEDS}")
    deg = {arm: degradation(per_arm[arm]) for arm in per_arm}
    less = [deg["Ad"][s] < deg["F"][s] for s in SEEDS]
    rule1 = sum(less) >= 2
    m = {arm: window_mean(per_arm[arm]) for arm in per_arm}
    d2, sd2, rule2 = _mean_diff_exceeds_larger_sd(m["Ad"], m["F"])
    d3, sd3, rule3 = _mean_diff_exceeds_larger_sd(m["Ad"], m["E"])
    rule2 = bool(rule2 and d2 > 0)
    rule3 = bool(rule3 and d3 > 0)
    return {"rule1_ad_degrades_less_in_2_of_3_seeds": {"result": bool(rule1), "per_seed_less_degraded": [bool(x) for x in less]},
            "rule2_mean_ad_minus_f": {"mean_diff": d2, "larger_sd": sd2, "result": rule2}, "rule3_mean_ad_minus_e": {"mean_diff": d3, "larger_sd": sd3, "result": rule3},
            "degradation_first_to_last_window": deg, "window_mean_pr_auc": m, "h4": "SUPPORTED" if (rule1 and rule2 and rule3) else "NOT ESTABLISHED (at least one locked rule failed)"}
