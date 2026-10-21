# Performance-triggered refresh (arm TrP): results

**Validation period (days 122-151), entity memory built over days 1-121 first. An additional analysis beside Phase 8; the locked H4 rules, the recorded H4 verdict and the score-PSI trigger result are unchanged. No held-out data was read.** Pre-registration: `PREREGISTRATION_TRP.md` (hashed before any arm ran).

## Monitor and threshold (TRAIN days only)

| Seed | Relative PR-AUC noise r (TRAIN days 112-116 vs 117-121) | delta = min(0.5, max(0.10, 2 x r)) | Refreshes (of 4) | Monitor after windows 1-4: window PR-AUC and relative drop from the reference |
|---|---|---|---|---|
| 42 | 0.071 | 0.142 | 1 | W1 0.610; W2 0.405 (drop +0.336, fired); W3 0.450; W4 0.408 (drop +0.094) |
| 123 | 0.112 | 0.224 | 1 | W1 0.595; W2 0.402 (drop +0.324, fired); W3 0.462; W4 0.411 (drop +0.111) |
| 2024 | 0.105 | 0.211 | 1 | W1 0.615; W2 0.431 (drop +0.300, fired); W3 0.454; W4 0.420 (drop +0.074) |

## Window-mean PR-AUC per seed (seeds 42, 123, 2024)

| Arm | Per seed | Mean | Seed SD |
|---|---|---|---|
| Frozen (F) | 0.4324, 0.4196, 0.4451 | 0.4324 | 0.0128 |
| Adaptive, refresh every window (Ad) | 0.4724, 0.4694, 0.4783 | 0.4733 | 0.0045 |
| Drift-triggered, score PSI (Tr; never fired) | 0.4324, 0.4196, 0.4451 | 0.4324 | 0.0128 |
| Performance-triggered refresh (TrP) | 0.4500, 0.4394, 0.4585 | 0.4493 | 0.0096 |

## Pre-registered criteria

| Criterion | Result |
|---|---|
| TRP1 execution; window 1 of TrP equals F within 1e-6 (max difference 1.2e-07) | MET |
| TRP2 delta follows the stated rule from TRAIN days only | MET |
| TRP3 mean(TrP - F) = +0.0169 is positive and exceeds the larger seed SD (0.0128) | MET |
| TRP4 mean refreshes per run 1.00 is below 4 | MET |
| TRP5 mean(Ad - TrP) = +0.0240 does not exceed the larger seed SD (0.0096) | NOT MET |
| **Performance-triggered-refresh outcome** | **NOT SUPPORTED** |

## Limits

Three seeds sharing the same data are not independent replications; one training run per seed and arm; GPU training is not bit-reproducible (identical-seed window means differed by about 0.004 in the Module F diagnosis); the monitor is evaluated at 4 points per run; one monitor type and one threshold rule; zero label delay; validation period only. Nothing here is a held-out result and nothing changes H4.
