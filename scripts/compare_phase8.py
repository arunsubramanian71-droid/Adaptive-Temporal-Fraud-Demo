"""Phase 8 comparison + H4 decision rule (design Section 8 / 14.6). Read-only over saved
artifacts; trains nothing.

  python -u scripts/compare_phase8.py

Requires train_phase8_{frozen,adaptive,replay_control}.py to have already been run for all three
seeds (they have not been, in this implementation pass) -- reads their metrics_val_W{1..5}.json
files. Written now as implementation; NOT executed by this implementation pass.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import stats

from src.config import load_config

SEEDS = [42, 123, 2024]
ARMS = {"F": "frozen", "Ad": "adaptive", "E": "replay_control"}
N_WINDOWS = 5


def load_arm(exp_base: pathlib.Path, rel: str) -> dict:
    per_seed = {}
    for seed in SEEDS:
        windows = []
        for k in range(1, N_WINDOWS + 1):
            m = json.load(open(exp_base / rel / f"seed_{seed}" / f"metrics_val_W{k}.json"))
            windows.append(m["pooled"]["pr_auc"])
        per_seed[seed] = windows
    return per_seed


def degradation(per_seed_windows: dict) -> dict:
    """W1 -> W5 drop per seed (positive = degraded)."""
    return {seed: w[0] - w[-1] for seed, w in per_seed_windows.items()}


def mean_diff_exceeds_larger_sd(a: dict, b: dict) -> tuple[float, bool]:
    av = np.array(list(a.values()))
    bv = np.array(list(b.values()))
    d = av - bv
    sd = max(av.std(ddof=1), bv.std(ddof=1))
    return float(d.mean()), bool(d.mean() > sd)


def main() -> None:
    cfg = load_config()
    exp_base = cfg.experiments_dir / "phase8_adaptive_drift"

    per_seed = {arm: load_arm(exp_base, rel) for arm, rel in ARMS.items()}
    deg = {arm: degradation(per_seed[arm]) for arm in ARMS}

    # H4 rule 1: Ad degrades less than F in >= 2/3 seeds
    less_degraded = [deg["Ad"][s] < deg["F"][s] for s in SEEDS]
    rule1 = sum(less_degraded) >= 2

    # H4 rule 2: mean(Ad - F) across all windows/seeds positive and exceeds larger seed SD
    ad_all = {s: np.mean(per_seed["Ad"][s]) for s in SEEDS}
    f_all = {s: np.mean(per_seed["F"][s]) for s in SEEDS}
    mean_diff_2, rule2 = mean_diff_exceeds_larger_sd(ad_all, f_all)

    # H4 rule 3: Ad beats E (replay-control) by the same test
    e_all = {s: np.mean(per_seed["E"][s]) for s in SEEDS}
    mean_diff_3, rule3 = mean_diff_exceeds_larger_sd(ad_all, e_all)

    h4_supported = rule1 and rule2 and rule3
    out = {
        "per_seed_window_pr_auc": per_seed,
        "degradation_w1_to_w5": deg,
        "rule1_ad_degrades_less_in_2_of_3_seeds": {"result": rule1, "per_seed_less_degraded": less_degraded},
        "rule2_mean_ad_minus_f": {"mean_diff": mean_diff_2, "exceeds_larger_sd": rule2},
        "rule3_mean_ad_minus_e": {"mean_diff": mean_diff_3, "exceeds_larger_sd": rule3},
        "h4_decision": "supported" if h4_supported else "partial/not established",
        "note": "p-values below are descriptive only, n=3, uncorrected (Section 8)",
        "descriptive_paired_p_ad_vs_f": float(stats.ttest_rel(list(ad_all.values()), list(f_all.values())).pvalue),
        "descriptive_paired_p_ad_vs_e": float(stats.ttest_rel(list(ad_all.values()), list(e_all.values())).pvalue),
    }
    with open(exp_base / "comparison.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"h4_decision": out["h4_decision"], "rule1": rule1, "rule2": rule2, "rule3": rule3}, indent=1))


if __name__ == "__main__":
    main()
