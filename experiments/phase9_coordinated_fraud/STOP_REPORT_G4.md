# Phase 9 — STOPPED at frozen gate G4 (hard stop S6)

Date: 2026-10-07. Execution authorization: "If ANY frozen gate or stop criterion fails: STOP and report the exact failure. Do not
silently modify the methodology." Nothing below was altered, regenerated, or waived.

## The failure
Frozen gate G4 (design §11.7 / §15.1): mean |Δ risk percentile| of REAL transactions between the clean and the augmented stream
must be ≤ 0.5 percentile points. Measured on the three eval benchmarks (days 122–151, 83,655 real rows each):

| seed | mean abs Δ (points) | G4 |
|---|---|---|
| 42 | 0.795 | **FAIL** |
| 123 | 0.462 | pass |
| 2024 | 0.799 | **FAIL** |

G1 and G2 passed for all seeds (α=0 top-1% fraction 0.001 vs the 3% limit; α=1 fraction 0.21–0.23, ratio 153–324× vs ≥ 5×);
G3 (planted fraction 4.999% ≤ 5%) and G5 (manifest consistency) passed; G0 and the timing pilot passed.

## Read-only diagnosis (diagnostics/diagnose_g4.py; nothing regenerated, no gate changed)
| seed | as-defined (planted rows in denominator) | real rows re-ranked among themselves | rows with bit-identical logit | mean abs logit diff |
|---|---|---|---|---|
| 42 | 0.795 | 0.035 | 98.0% | 0.0014 |
| 123 | 0.462 | 0.065 | 97.3% | 0.0026 |
| 2024 | 0.799 | 0.067 | 97.7% | 0.0031 |

Interpretation: ≈ 97–98% of real transactions receive exactly the same model output with and without planting. The as-defined
G4 number is dominated by the mechanical effect of 4.8% planted rows entering each day's rank denominator (planted rows sit at
mean within-day percentile 0.56–0.65), not by state interference with the background. A small number of real rows do change
strongly (max re-ranked change 63–90 points: rows sharing a hub / hijacked card with planted traffic). Whether the frozen
wording of G4 intended the as-defined quantity or the state-interference quantity is a design question; this run applies it
literally as written.

## State at the stop
* Completed and valid: unit tests (full regression suite passes, including 14 + 3 new Phase 9 tests); G0 PASSED (exact match on all
  501,214 transactions, TEST never opened); timing pilot PASSED (median 0.072 s/window; projected 0.19 h CPU); null calibration
  (97 s); dev benchmark generation and the pre-registered dev selection (γ = 0.5 for every W; locked, hashed); eval benchmark generated
  once (arrays + sealed manifest); risk streams (CPU, 9.4 s each, all seeds).
* NOT run (blocked by S6): all discovery/scoring on the eval benchmark, all baselines/ablations, real-data analysis (Tier 0/E0/E1),
  case studies, leakage-invariance suite on discovery outputs, reproducibility determinism run, final report.
* No detector output exists for the eval benchmark; no ground-truth information has been used by any detector.

## Decision needed from the user (no option is pre-authorized)
1. Amend G4 by a dated addendum so that it measures state interference (real rows re-ranked among themselves, or logit-level
   change), keeping the 0.5-point threshold — the diagnosed quantity is 0.03–0.07 points. Then continue with the existing
   benchmarks (no regeneration).
2. Keep G4 exactly as written: the benchmark is invalid under the frozen rule and Phase 9 cannot produce a valid H5a result;
   regeneration would require a new frozen design decision (e.g. lower planting fraction), not a silent re-draw.
3. Any other disposition.
