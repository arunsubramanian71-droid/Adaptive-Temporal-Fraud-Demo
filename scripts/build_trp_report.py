"""Evaluate the pre-registered criteria TRP1-TRP5 (experiments/phase8_triggered_refresh/PREREGISTRATION_TRP.md) from the recorded outputs and write TRP_RESULTS.md + decision.json
under outputs_performance/. Reads recorded files only (TrP outputs, F and Ad from the state-continuity supplement, Tr from outputs/).
Usage:  .venv/Scripts/python.exe -B scripts/build_trp_report.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EXP = ROOT / "experiments" / "phase8_triggered_refresh"
OUT = EXP / "outputs_performance"
SUP = ROOT / "experiments" / "phase8_state_continuity_supplement" / "outputs" / "burn_in"
SEEDS = (42, 123, 2024)


def main() -> int:
    trp = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    sup = json.loads((SUP / "summary.json").read_text(encoding="utf-8"))["window_pr_auc"]
    tr = json.loads((EXP / "outputs" / "summary.json").read_text(encoding="utf-8"))
    wm = {"F": [float(np.mean(sup["F"][str(s)])) for s in SEEDS], "Ad": [float(np.mean(sup["Ad"][str(s)])) for s in SEEDS],
          "Tr": [tr["per_seed"][str(s)]["window_mean_pr_auc"] for s in SEEDS], "TrP": [trp["per_seed"][str(s)]["window_mean_pr_auc"] for s in SEEDS]}
    sd = {a: float(np.std(v, ddof=1)) for a, v in wm.items()}
    mean = {a: float(np.mean(v)) for a, v in wm.items()}
    refreshes = [trp["per_seed"][str(s)]["n_refreshes"] for s in SEEDS]
    w1 = []
    for s in SEEDS:
        f1 = pd.read_parquet(SUP / "frozen" / f"seed_{s}" / "predictions_W1.parquet", columns=["TransactionID", "y_score"]).set_index("TransactionID")["y_score"]
        t1 = pd.read_parquet(OUT / f"seed_{s}" / "predictions_W1.parquet", columns=["TransactionID", "y_score"]).set_index("TransactionID")["y_score"]
        w1.append(float(np.abs(f1.to_numpy() - t1.loc[f1.index].to_numpy()).max()))
    ok1 = max(w1) <= 1e-6 and len(trp["per_seed"]) == 3
    ok2 = all(abs(trp["per_seed"][str(s)]["delta"] - min(trp["delta_cap"], max(trp["delta_floor"], 2 * trp["per_seed"][str(s)]["relative_noise_r"]))) < 1e-12 for s in SEEDS)
    d_pf = mean["TrP"] - mean["F"]
    ok3 = d_pf > 0 and d_pf > max(sd["TrP"], sd["F"])
    ok4 = float(np.mean(refreshes)) < 4
    d_ap = mean["Ad"] - mean["TrP"]
    ok5 = d_ap <= max(sd["Ad"], sd["TrP"])
    outcome = "SUPPORTED" if all((ok1, ok2, ok3, ok4, ok5)) else "NOT SUPPORTED"
    dec = {"criteria": {"TRP1_execution_and_window1_equals_F": bool(ok1), "TRP2_threshold_rule_from_train_days": bool(ok2), "TRP3_benefit_over_static": bool(ok3),
                        "TRP4_fewer_refreshes_than_periodic": bool(ok4), "TRP5_parity_with_periodic": bool(ok5)},
           "window_mean_pr_auc_per_seed": wm, "mean": mean, "seed_sd": sd, "mean_TrP_minus_F": d_pf, "mean_Ad_minus_TrP": d_ap, "refreshes_per_seed": refreshes,
           "delta_per_seed": {str(s): trp["per_seed"][str(s)]["delta"] for s in SEEDS}, "relative_noise_per_seed": {str(s): trp["per_seed"][str(s)]["relative_noise_r"] for s in SEEDS},
           "window1_max_abs_score_diff_vs_F": w1, "outcome": outcome, "h4_unchanged": True, "test_data_read": False}
    (OUT / "decision.json").write_text(json.dumps(dec, indent=1) + "\n", encoding="utf-8")
    f = lambda x: f"{x:.4f}"  # noqa: E731
    L = ["# Performance-triggered refresh (arm TrP): results\n", "**Validation period (days 122-151), entity memory built over days 1-121 first. An additional analysis beside Phase 8; the locked H4 rules, the recorded H4 verdict and the score-PSI trigger result are unchanged. No held-out data was read.** Pre-registration: `PREREGISTRATION_TRP.md` (hashed before any arm ran).\n",
         "## Monitor and threshold (TRAIN days only)\n", "| Seed | Relative PR-AUC noise r (TRAIN days 112-116 vs 117-121) | delta = min(0.5, max(0.10, 2 x r)) | Refreshes (of 4) | Monitor after windows 1-4: window PR-AUC and relative drop from the reference |", "|---|---|---|---|---|"]
    for s in SEEDS:
        lg = json.loads((OUT / f"seed_{s}" / "trigger_log.json").read_text(encoding="utf-8"))["monitor_log"]
        L.append(f"| {s} | {dec['relative_noise_per_seed'][str(s)]:.3f} | {dec['delta_per_seed'][str(s)]:.3f} | {refreshes[SEEDS.index(s)]} | "
                 + "; ".join(f"W{r['window']} {r['pr_auc']:.3f}" + ("" if r["relative_drop"] is None else f" (drop {r['relative_drop']:+.3f}{', fired' if r['fired'] else ''})") for r in lg) + " |")
    L += ["", "## Window-mean PR-AUC per seed (seeds 42, 123, 2024)\n", "| Arm | Per seed | Mean | Seed SD |", "|---|---|---|---|"]
    for a, lab in (("F", "Frozen (F)"), ("Ad", "Adaptive, refresh every window (Ad)"), ("Tr", "Drift-triggered, score PSI (Tr; never fired)"), ("TrP", "Performance-triggered refresh (TrP)")):
        L.append(f"| {lab} | {', '.join(f(x) for x in wm[a])} | {f(mean[a])} | {f(sd[a])} |")
    L += ["", "## Pre-registered criteria\n", "| Criterion | Result |", "|---|---|",
          f"| TRP1 execution; window 1 of TrP equals F within 1e-6 (max difference {max(w1):.1e}) | {'MET' if ok1 else 'NOT MET'} |",
          f"| TRP2 delta follows the stated rule from TRAIN days only | {'MET' if ok2 else 'NOT MET'} |",
          f"| TRP3 mean(TrP - F) = {d_pf:+.4f} is positive and exceeds the larger seed SD ({max(sd['TrP'], sd['F']):.4f}) | {'MET' if ok3 else 'NOT MET'} |",
          f"| TRP4 mean refreshes per run {np.mean(refreshes):.2f} is below 4 | {'MET' if ok4 else 'NOT MET'} |",
          f"| TRP5 mean(Ad - TrP) = {d_ap:+.4f} does not exceed the larger seed SD ({max(sd['Ad'], sd['TrP']):.4f}) | {'MET' if ok5 else 'NOT MET'} |",
          f"| **Performance-triggered-refresh outcome** | **{outcome}** |", "",
          "## Limits\n", "Three seeds sharing the same data are not independent replications; one training run per seed and arm; GPU training is not bit-reproducible (identical-seed window means differed by about 0.004 in the Module F diagnosis); the monitor is evaluated at 4 points per run; one monitor type and one threshold rule; zero label delay; validation period only. Nothing here is a held-out result and nothing changes H4.", ""]
    data = ("\n".join(L)).encode("utf-8")
    (OUT / "TRP_RESULTS.md").write_bytes(data)
    (OUT / "TRP_RESULTS.sha256").write_bytes(hashlib.sha256(data).hexdigest().encode("ascii"))
    print(outcome, dec["criteria"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
