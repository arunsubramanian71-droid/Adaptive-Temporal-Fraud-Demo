# Drift-triggered refresh (arm Tr): results

**Validation period (days 122-151), entity memory built over days 1-121 first. An additional analysis beside Phase 8; the locked H4 rules and the recorded H4 verdict are unchanged. No held-out data was read.** Pre-registration: `PREREGISTRATION_TR.md` (hashed before any arm ran).

## Monitor and threshold (TRAIN days only)

| Seed | PSI noise (TRAIN days 112-116 vs 117-121) | tau = max(0.10, 2 x noise) | Refreshes (of 4) | Monitor PSI after windows 2-4 (reference re-baselines after a refresh) |
|---|---|---|---|---|
| 42 | 0.0016 | 0.1000 | 0 | ref, 0.013, 0.006, 0.002 |
| 123 | 0.0031 | 0.1000 | 0 | ref, 0.013, 0.008, 0.002 |
| 2024 | 0.0015 | 0.1000 | 0 | ref, 0.025, 0.008, 0.004 |

## Window-mean PR-AUC per seed (seeds 42, 123, 2024)

| Arm | Per seed | Mean | Seed SD |
|---|---|---|---|
| Frozen (F) | 0.4324, 0.4196, 0.4451 | 0.4324 | 0.0128 |
| Adaptive, refresh every window (Ad) | 0.4724, 0.4694, 0.4783 | 0.4733 | 0.0045 |
| Adaptive, drift-triggered refresh (Tr) | 0.4324, 0.4196, 0.4451 | 0.4324 | 0.0128 |

## Pre-registered criteria

| Criterion | Result |
|---|---|
| TR1 execution; window 1 of Tr equals F within 1e-6 (max difference 1.8e-07) | MET |
| TR2 label-free monitor (unit test) and TRAIN-only threshold recorded per seed | MET |
| TR3 mean(Tr - F) = +0.0000 is positive and exceeds the larger seed SD (0.0128) | NOT MET |
| TR4 mean refreshes per run 0.00 is below 4 | MET |
| TR5 mean(Ad - Tr) = +0.0410 does not exceed the larger seed SD (0.0128) | NOT MET |
| **Triggered-refresh outcome** | **NOT SUPPORTED** |

## Reading (post hoc, labelled as such)

The label-free monitor never fired: the score-distribution PSI stayed at 0.025 or below in every seed and window, far under the threshold of 0.10, while the window PR-AUC of the frozen arm fell from about 0.6 to about 0.33. Prediction-distribution drift therefore did not signal the performance drop on these windows (consistent with the Phase 8 finding of mild covariate drift against a large performance drift), so Tr behaved exactly like the frozen arm and the triggered policy offered no benefit under the registered rule. A performance-based trigger (using the labels of the window just scored) would be a different, not pre-registered policy; it was not run and no threshold was changed.

## Limits

Three seeds sharing the same data are not independent replications; one training run per seed and arm; GPU training is not bit-reproducible (identical-seed window means differed by about 0.004 in the Module F diagnosis); the monitor is evaluated at 4 points per run; one monitor type and one threshold rule; zero label delay; validation period only. Nothing here is a held-out result and nothing changes H4.
