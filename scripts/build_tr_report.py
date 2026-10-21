"""Evaluate the pre-registered criteria TR1-TR5 (experiments/phase8_triggered_refresh/PREREGISTRATION_TR.md) from the recorded outputs and write TR_RESULTS.md + decision.json.

Reads recorded files only: the Tr outputs and the F / Ad window PR-AUCs of the state-continuity supplement (same condition). Usage:  .venv/Scripts/python.exe -B scripts/build_tr_report.py
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
OUT = EXP / "outputs"
SUP = ROOT / "experiments" / "phase8_state_continuity_supplement" / "outputs" / "burn_in"
SEEDS = (42, 123, 2024)


def main() -> int:
    tr = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    sup = json.loads((SUP / "summary.json").read_text(encoding="utf-8"))["window_pr_auc"]
    wm = {"F": [float(np.mean(sup["F"][str(s)])) for s in SEEDS], "Ad": [float(np.mean(sup["Ad"][str(s)])) for s in SEEDS],
          "Tr": [tr["per_seed"][str(s)]["window_mean_pr_auc"] for s in SEEDS]}
    sd = {a: float(np.std(v, ddof=1)) for a, v in wm.items()}
    mean = {a: float(np.mean(v)) for a, v in wm.items()}
    refreshes = [tr["per_seed"][str(s)]["n_refreshes"] for s in SEEDS]
    w1 = []
    for s in SEEDS:
        f1 = pd.read_parquet(SUP / "frozen" / f"seed_{s}" / "predictions_W1.parquet", columns=["TransactionID", "y_score"]).set_index("TransactionID")["y_score"]
        t1 = pd.read_parquet(OUT / f"seed_{s}" / "predictions_W1.parquet", columns=["TransactionID", "y_score"]).set_index("TransactionID")["y_score"]
        w1.append(float(np.abs(f1.to_numpy() - t1.loc[f1.index].to_numpy()).max()))
    tr1 = max(w1) <= 1e-6 and len(tr["per_seed"]) == 3
    tr2 = all(tr["per_seed"][str(s)]["tau"] >= tr["psi_floor"] for s in SEEDS)           # tau recorded per seed from TRAIN days only (the label-free property is a unit test)
    d_tf = mean["Tr"] - mean["F"]
    tr3 = d_tf > 0 and d_tf > max(sd["Tr"], sd["F"])
    tr4 = float(np.mean(refreshes)) < 4
    d_at = mean["Ad"] - mean["Tr"]
    tr5 = d_at <= max(sd["Ad"], sd["Tr"])
    outcome = "SUPPORTED" if all((tr1, tr2, tr3, tr4, tr5)) else "NOT SUPPORTED"
    dec = {"criteria": {"TR1_execution_and_window1_equals_F": bool(tr1), "TR2_train_only_threshold_recorded": bool(tr2), "TR3_benefit_over_static": bool(tr3),
                        "TR4_fewer_refreshes_than_periodic": bool(tr4), "TR5_parity_with_periodic": bool(tr5)},
           "window_mean_pr_auc_per_seed": wm, "mean": mean, "seed_sd": sd, "mean_Tr_minus_F": d_tf, "mean_Ad_minus_Tr": d_at, "refreshes_per_seed": refreshes,
           "tau_per_seed": {str(s): tr["per_seed"][str(s)]["tau"] for s in SEEDS}, "noise_psi_per_seed": {str(s): tr["per_seed"][str(s)]["psi_noise_train_days_112_116_vs_117_121"] for s in SEEDS},
           "window1_max_abs_score_diff_vs_F": w1, "outcome": outcome, "h4_unchanged": True, "test_data_read": False}
    (EXP / "outputs" / "decision.json").write_text(json.dumps(dec, indent=1) + "\n", encoding="utf-8")
    f = lambda x: f"{x:.4f}"  # noqa: E731
    L = ["# Drift-triggered refresh (arm Tr): results\n", "**Validation period (days 122-151), entity memory built over days 1-121 first. An additional analysis beside Phase 8; the locked H4 rules and the recorded H4 verdict are unchanged. No held-out data was read.** Pre-registration: `PREREGISTRATION_TR.md` (hashed before any arm ran).\n",
         "## Monitor and threshold (TRAIN days only)\n", "| Seed | PSI noise (TRAIN days 112-116 vs 117-121) | tau = max(0.10, 2 x noise) | Refreshes (of 4) | Monitor PSI after windows 2-4 (reference re-baselines after a refresh) |", "|---|---|---|---|---|"]
    for s in SEEDS:
        lg = json.loads((OUT / f"seed_{s}" / "trigger_log.json").read_text(encoding="utf-8"))["monitor_log"]
        L.append(f"| {s} | {dec['noise_psi_per_seed'][str(s)]:.4f} | {dec['tau_per_seed'][str(s)]:.4f} | {refreshes[SEEDS.index(s)]} | " + ", ".join("ref" if r["psi"] is None else f"{r['psi']:.3f}{' (fired)' if r['fired'] else ''}" for r in lg) + " |")
    L += ["", "## Window-mean PR-AUC per seed (seeds 42, 123, 2024)\n", "| Arm | Per seed | Mean | Seed SD |", "|---|---|---|---|"]
    for a, lab in (("F", "Frozen (F)"), ("Ad", "Adaptive, refresh every window (Ad)"), ("Tr", "Adaptive, drift-triggered refresh (Tr)")):
        L.append(f"| {lab} | {', '.join(f(x) for x in wm[a])} | {f(mean[a])} | {f(sd[a])} |")
    L += ["", "## Pre-registered criteria\n", "| Criterion | Result |", "|---|---|",
          f"| TR1 execution; window 1 of Tr equals F within 1e-6 (max difference {max(w1):.1e}) | {'MET' if tr1 else 'NOT MET'} |",
          f"| TR2 label-free monitor (unit test) and TRAIN-only threshold recorded per seed | {'MET' if tr2 else 'NOT MET'} |",
          f"| TR3 mean(Tr - F) = {d_tf:+.4f} is positive and exceeds the larger seed SD ({max(sd['Tr'], sd['F']):.4f}) | {'MET' if tr3 else 'NOT MET'} |",
          f"| TR4 mean refreshes per run {np.mean(refreshes):.2f} is below 4 | {'MET' if tr4 else 'NOT MET'} |",
          f"| TR5 mean(Ad - Tr) = {d_at:+.4f} does not exceed the larger seed SD ({max(sd['Ad'], sd['Tr']):.4f}) | {'MET' if tr5 else 'NOT MET'} |",
          f"| **Triggered-refresh outcome** | **{outcome}** |", "",
          "## Reading (post hoc, labelled as such)\n", "The label-free monitor never fired: the score-distribution PSI stayed at 0.025 or below in every seed and window, far under the threshold of 0.10, while the window PR-AUC of the frozen arm fell from about 0.6 to about 0.33. Prediction-distribution drift therefore did not signal the performance drop on these windows (consistent with the Phase 8 finding of mild covariate drift against a large performance drift), so Tr behaved exactly like the frozen arm and the triggered policy offered no benefit under the registered rule. A performance-based trigger (using the labels of the window just scored) would be a different, not pre-registered policy; it was not run and no threshold was changed.", "",
          "## Limits\n", "Three seeds sharing the same data are not independent replications; one training run per seed and arm; GPU training is not bit-reproducible (identical-seed window means differed by about 0.004 in the Module F diagnosis); the monitor is evaluated at 4 points per run; one monitor type and one threshold rule; zero label delay; validation period only. Nothing here is a held-out result and nothing changes H4.", ""]
    data = ("\n".join(L)).encode("utf-8")
    (EXP / "outputs" / "TR_RESULTS.md").write_bytes(data)
    (EXP / "outputs" / "TR_RESULTS.sha256").write_bytes(hashlib.sha256(data).hexdigest().encode("ascii"))
    print(outcome, dec["criteria"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
