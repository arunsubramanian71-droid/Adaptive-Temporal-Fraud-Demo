"""Mechanical decision rules for comparison B, exactly as registered in
experiments/comparison_B_graph_features/PREREGISTRATION_B.md section 6. Inputs are recorded PR-AUC values only; nothing here
reads data, labels or models. The sample SD uses ddof = 1 over the three seeds."""
from __future__ import annotations

import numpy as np


def sample_sd(values) -> float:
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        raise ValueError("a sample SD needs at least two values")
    return float(v.std(ddof=1))


def decide_h1(b_full: float, a_refit: float, b_sub, a_sub) -> dict:
    """B versus A. ``b_sub`` and ``a_sub`` are per-seed PR-AUCs in the same seed order."""
    b_sub, a_sub = np.asarray(b_sub, float), np.asarray(a_sub, float)
    if b_sub.shape != a_sub.shape or len(b_sub) != 3:
        raise ValueError("b_sub and a_sub must each hold the three seed values in the same order")
    diffs = b_sub - a_sub
    mean_diff = float(diffs.mean())
    sd = max(sample_sd(b_sub), sample_sd(a_sub))
    wins = int((diffs > 0).sum())
    losses = int((diffs < 0).sum())
    cond = {"i_primary_B_above_A": bool(b_full > a_refit), "ii_mean_diff_exceeds_larger_sd": bool(mean_diff > sd), "iii_B_wins_at_least_2_of_3": bool(wins >= 2)}
    if all(cond.values()):
        verdict = "SUPPORTED"
    elif mean_diff < -sd and losses >= 2:
        verdict = "CONTRADICTED"
    else:
        verdict = "NOT ESTABLISHED"
    return {"verdict": verdict, "conditions": cond, "primary_diff_B_minus_A": float(b_full - a_refit), "seed_diffs": diffs.tolist(),
            "mean_seed_diff": mean_diff, "larger_sd": sd, "b_wins": wins, "b_losses": losses}


def decide_h2(learned, handcrafted) -> dict:
    """Learned graph model (three recorded seeds) versus handcrafted-feature XGBoost (three ``B_sub`` seeds), complete separation rule."""
    learned, handcrafted = np.asarray(learned, float), np.asarray(handcrafted, float)
    if len(learned) != 3 or len(handcrafted) != 3:
        raise ValueError("both groups must hold three seed values")
    lo_l, hi_l, lo_h, hi_h = float(learned.min()), float(learned.max()), float(handcrafted.min()), float(handcrafted.max())
    if lo_l > hi_h:
        verdict = "LEARNED SUPPORTED"
    elif hi_l < lo_h:
        verdict = "HANDCRAFTED AT LEAST AS GOOD (H2 contradicted)"
    else:
        verdict = "INCONCLUSIVE"
    return {"verdict": verdict, "learned_min": lo_l, "learned_max": hi_l, "handcrafted_min": lo_h, "handcrafted_max": hi_h,
            "learned_mean": float(learned.mean()), "handcrafted_mean": float(handcrafted.mean())}


def ablation_table(b_full: float, ablations: dict, b_sub) -> dict:
    """Descriptive: change of each ablation against B_full; flagged as larger than seed noise only if |change| > SD(B_sub)."""
    noise = sample_sd(b_sub)
    rows = {}
    for name, v in ablations.items():
        d = float(v - b_full)
        rows[name] = {"pr_auc": float(v), "change_vs_B_full": d, "larger_than_seed_noise": bool(abs(d) > noise)}
    return {"seed_noise_sd_B_sub": noise, "rows": rows}
