# Phase 9b: synthetic recovery benchmark with a corrected interference gate. Results

**Separate from Phase 9.** The original H5a remains **NOT ESTABLISHED** under the original protocol and the original G4 failure stands; this benchmark's result is reported as **H5a-9b** and does not change either. Validation days 122-151 with real background from days up to 151; no held-out day was read. Pre-registration: `PREREGISTRATION_9B.md` (hashed before any 9b benchmark existed).

## Validity gates (all evaluated before any detector output)

| Seed | Planted fraction | G1 (alpha 0 top-1 %) | G2 (alpha 1 top-1 %) | G3 | G5 | G4-9b (real rows ranked among real rows) | Original G4 formula (descriptive) |
|---|---|---|---|---|---|---|---|
| 42 | 0.0500 | 0.0016 (pass) | 0.1893 (pass) | pass | pass | 0.0499 points (pass) | 0.9379 points (would fail) |
| 123 | 0.0500 | 0.0000 (pass) | 0.2222 (pass) | pass | pass | 0.0653 points (pass) | 0.6986 points (would fail) |
| 2024 | 0.0500 | 0.0020 (pass) | 0.1888 (pass) | pass | pass | 0.0421 points (pass) | 0.9035 points (would fail) |

As disclosed in the pre-registration, the original formula fails again (planted rows enter the ranking population), while the quantity it was meant to check, the effect of planting on the model's treatment of real transactions, is about a tenth of the 0.5-point limit.

## Track-level AUPRC at Jaccard >= 0.5 (seeds 42, 123, 2024)

| Method | alpha = 0 (primary) | alpha = 0.5 | alpha = 1 | all coordinated groups | groups recovered at alpha = 0 (of 12) |
|---|---|---|---|---|---|
| A2: structure + time (label-free, risk-free) | 0.145, 0.267, 0.141 | 0.356, 0.218, 0.394 | 0.185, 0.170, 0.104 | 0.345, 0.359, 0.344 | 8, 9, 9 |
| B1: naive shared-infrastructure groups | 0.008, 0.014, 0.005 | 0.035, 0.023, 0.029 | 0.016, 0.025, 0.014 | 0.042, 0.052, 0.033 | 7, 9, 9 |
| B2: structure only | 0.034, 0.048, 0.045 | 0.045, 0.082, 0.103 | 0.048, 0.044, 0.029 | 0.101, 0.134, 0.124 | 8, 9, 9 |
| B3a: risk-first grouping | 0.000, 0.000, 0.000 | 0.076, 0.012, 0.062 | 0.060, 0.118, 0.014 | 0.063, 0.052, 0.034 | 0, 0, 0 |
| B3b: risk on the same candidates | 0.016, 0.045, 0.027 | 0.211, 0.174, 0.155 | 0.164, 0.179, 0.148 | 0.216, 0.240, 0.201 | 8, 9, 9 |
| B0: random card sets | 0.000, 0.000, 0.000 | 0.000, 0.000, 0.000 | 0.000, 0.000, 0.000 | 0.000, 0.000, 0.000 | 0, 0, 0 |
| A5: structure + time + risk (descriptive) | 0.033, 0.044, 0.050 | 0.376, 0.318, 0.461 | 0.285, 0.364, 0.260 | 0.322, 0.363, 0.366 | 8, 9, 9 |

## Pre-registered decision rule (Phase 9 design section 15, unchanged)

| A2 versus | A2 higher in | Mean difference | Larger seed SD | Met |
|---|---|---|---|---|
| B1 | 3 of 3 seeds | +0.1755 | 0.0716 | yes |
| B2 | 3 of 3 seeds | +0.1420 | 0.0716 | yes |
| B3a | 3 of 3 seeds | +0.1844 | 0.0716 | yes |
| B3b | 3 of 3 seeds | +0.1549 | 0.0716 | yes |

| G-cal guard: false alerts per day on the clean real validation days at the TRAIN-locked tau (limit 3.0) |
|---|
| seed 42: 1.73 (pass); seed 123: 1.37 (pass); seed 2024: 1.93 (pass) |

S8 non-diagnostic check: B1 too easy everywhere: False; every method within noise of B0 everywhere: False.

## Outcome: **H5a-9b SUPPORTED**

## Reading (required by the design and added for clarity)

* At alpha = 0 the risk-based baselines B3a and B3b are blind by construction, so beating them only shows that the benchmark behaves as intended; the discriminating comparisons are B1 and B2, and both are met.
* B1, B2 and B3b recover 7 to 9 of the 12 camouflaged groups somewhere in their output, about as many as A2 (8 to 9). A2's advantage is ranking: it places the groups it recovers far higher among the candidate tracks, which is what track-level AUPRC measures.
* Absolute recovery is modest: A2's alpha = 0 AUPRC is 0.14 to 0.27, and A2 recovers 8 to 9 of the 12 camouflaged groups per seed.
* Adding transaction risk (A5) helps where planted groups carry risk signal (alpha = 0.5 and 1) and hurts at alpha = 0; H5b was not evaluated here.
* A supported H5a-9b says the tested planted patterns are recoverable by structure plus time on this background; it says nothing about whether the real data contains coordinated groups. Candidate coordinated community; not a confirmed fraud ring. This does not establish that IEEE-CIS contains fraud rings.

## Limits

The real background is identical across seeds (the seeds capture generator, detector and risk-initialisation variance only); one primary window (W = 7) with the locked gamma; planted patterns test what the generator plants; validation days only; nothing here is a held-out result.
