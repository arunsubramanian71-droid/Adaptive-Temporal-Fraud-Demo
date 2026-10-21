"""Render experiments/phase9b_recovery_benchmark/outputs/RESULTS_9B.md from the recorded results.json (nothing is recomputed).

Usage:  .venv/Scripts/python.exe -B scripts/build_phase9b_report.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "phase9b_recovery_benchmark" / "outputs"
SEEDS = ("42", "123", "2024")
NAMES = {"A2": "A2: structure + time (label-free, risk-free)", "B1": "B1: naive shared-infrastructure groups", "B2": "B2: structure only",
         "B3a": "B3a: risk-first grouping", "B3b": "B3b: risk on the same candidates", "B0": "B0: random card sets", "A5": "A5: structure + time + risk (descriptive)"}


def main() -> int:
    r = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    f = lambda x: f"{x:.3f}"  # noqa: E731
    L = ["# Phase 9b: synthetic recovery benchmark with a corrected interference gate. Results\n",
         "**Separate from Phase 9.** The original H5a remains **NOT ESTABLISHED** under the original protocol and the original G4 failure stands; this benchmark's result is reported as **H5a-9b** and does not change either. Validation days 122-151 with real background from days up to 151; no held-out day was read. Pre-registration: `PREREGISTRATION_9B.md` (hashed before any 9b benchmark existed).\n",
         "## Validity gates (all evaluated before any detector output)\n",
         "| Seed | Planted fraction | G1 (alpha 0 top-1 %) | G2 (alpha 1 top-1 %) | G3 | G5 | G4-9b (real rows ranked among real rows) | Original G4 formula (descriptive) |", "|---|---|---|---|---|---|---|---|"]
    for s in SEEDS:
        g = r["gates"][s]
        L.append(f"| {s} | {g['planted_fraction']:.4f} | {g['G1_alpha0_top1_fraction']:.4f} ({'pass' if g['G1_pass'] else 'FAIL'}) | {g['G2_alpha1_top1_fraction']:.4f} ({'pass' if g['G2_pass'] else 'FAIL'}) | "
                 f"{'pass' if g['G3_pass'] else 'FAIL'} | {'pass' if g['G5_pass'] else 'FAIL'} | {g['G4_9b_mean_abs_delta_pct_points']:.4f} points ({'pass' if g['G4_9b_pass'] else 'FAIL'}) | "
                 f"{g['G4_original_formula_mean_abs_delta_pct_points_descriptive']:.4f} points ({'would pass' if g['G4_original_formula_would_pass_descriptive'] else 'would fail'}) |")
    L += ["", "As disclosed in the pre-registration, the original formula fails again (planted rows enter the ranking population), while the quantity it was meant to check, the effect of planting on the model's treatment of real transactions, is about a tenth of the 0.5-point limit.", "",
          "## Track-level AUPRC at Jaccard >= 0.5 (seeds 42, 123, 2024)\n", "| Method | alpha = 0 (primary) | alpha = 0.5 | alpha = 1 | all coordinated groups | groups recovered at alpha = 0 (of 12) |", "|---|---|---|---|---|---|"]
    for n, label in NAMES.items():
        a = r["track_auprc"][n]
        L.append(f"| {label} | " + " | ".join(", ".join(f(a[rn][s]["ap"]) for s in SEEDS) for rn in ("alpha0", "alpha0.5", "alpha1", "all"))
                 + " | " + ", ".join(str(a["alpha0"][s]["n_recovered"]) for s in SEEDS) + " |")
    L += ["", "## Pre-registered decision rule (Phase 9 design section 15, unchanged)\n", "| A2 versus | A2 higher in | Mean difference | Larger seed SD | Met |", "|---|---|---|---|---|"]
    for b, c in r["comparisons_alpha0"].items():
        L.append(f"| {b} | {c['A2_wins_of_3']} of 3 seeds | {c['mean_diff']:+.4f} | {c['larger_sd']:.4f} | {'yes' if c['met'] else 'no'} |")
    L += ["", "| G-cal guard: false alerts per day on the clean real validation days at the TRAIN-locked tau (limit 3.0) |", "|---|"]
    L.append("| " + "; ".join(f"seed {s}: {r['g_cal'][s]['false_alerts_per_day']:.2f} ({'pass' if r['g_cal'][s]['pass'] else 'FAIL'})" for s in SEEDS) + " |")
    s8 = r["S8_non_diagnostic"]
    L += ["", f"S8 non-diagnostic check: B1 too easy everywhere: {s8['B1_too_easy_everywhere']}; every method within noise of B0 everywhere: {s8['every_method_within_noise_of_B0_everywhere']}.", "",
          f"## Outcome: **{r['outcome']}**\n",
          "## Reading (required by the design and added for clarity)\n",
          "* At alpha = 0 the risk-based baselines B3a and B3b are blind by construction, so beating them only shows that the benchmark behaves as intended; the discriminating comparisons are B1 and B2, and both are met.",
          "* B1, B2 and B3b recover 7 to 9 of the 12 camouflaged groups somewhere in their output, about as many as A2 (8 to 9). A2's advantage is ranking: it places the groups it recovers far higher among the candidate tracks, which is what track-level AUPRC measures.",
          "* Absolute recovery is modest: A2's alpha = 0 AUPRC is 0.14 to 0.27, and A2 recovers 8 to 9 of the 12 camouflaged groups per seed.",
          "* Adding transaction risk (A5) helps where planted groups carry risk signal (alpha = 0.5 and 1) and hurts at alpha = 0; H5b was not evaluated here.",
          "* A supported H5a-9b says the tested planted patterns are recoverable by structure plus time on this background; it says nothing about whether the real data contains coordinated groups. Candidate coordinated community; not a confirmed fraud ring. This does not establish that IEEE-CIS contains fraud rings.", "",
          "## Limits\n", "The real background is identical across seeds (the seeds capture generator, detector and risk-initialisation variance only); one primary window (W = 7) with the locked gamma; planted patterns test what the generator plants; validation days only; nothing here is a held-out result.", ""]
    data = "\n".join(L).encode("utf-8")
    (OUT / "RESULTS_9B.md").write_bytes(data)
    (OUT / "RESULTS_9B.sha256").write_bytes(hashlib.sha256(data).hexdigest().encode("ascii"))
    print("wrote RESULTS_9B.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
