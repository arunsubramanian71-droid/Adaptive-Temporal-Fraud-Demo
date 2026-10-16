# Diagnosis of the first GPU leakage-validation run (6 of 19 checks failed)

Preserved failing record: `temporal_leakage_validation_RUN1_gpu_single_pair_tolerance_6_FAILED.json`
(future_invariance x3 seeds and train_val_boundary_state_continuity x3 seeds; all other checks passed).

Model: seed-42 trained checkpoint, days <= 151, comparing predictions on past days.
Largest baseline |logit| = 568.1 (float32 ulp there = 6.1e-05). All values are max|d logit| on past days.

| Exp | What changed | Result | Reading |
|---|---|---|---|
| repeats | 7 repeats of IDENTICAL input, same dataset object | 3.81e-06 | GPU run-to-run noise floor |
| A | dataset rebuilt from identical arrays, no perturbation | 3.8e-06 | rebuilding is not the cause |
| B | corrupt only day-151 rows (features+edges), compare days <= 150 | 1.83e-04 (prob-space 5.6e-06) | deviation reproduced, yet day 151 cannot be read by steps <= 150 |
| C | CPU (deterministic), corrupt everything after c in {122,130,140,150} | 0 exactly at every cutoff | algorithm is causal; see `numerics_diagnosis.json` |
| D | identical data, GPU memory layout shifted by dummy allocations | 3.8e-06 | allocation layout is not the cause |
| E/F | corrupt day-151 features only: N(0,5^2) vs N(0,50^2) | 1.831e-04 both | identical value for a 10x larger perturbation => not information flow |
| H/I | corrupt day 151 vs days 140-145 | same worst days (124,133,131,125), same values | deviation pattern independent of what/where the future is |
| K | X replaced by a bit-for-bit copy (ndarray.copy()), ZERO content change | 1.831e-04, 82% of predictions differ | trigger is the array object/layout, not data |
| L | change one element of a TEST-period row that is never uploaded | 1.831e-04 | same |
| flags | raw["X"] is F-contiguous; ndarray.copy() returns C order | -- | ROOT CAUSE: harness copy changed memory layout -> different GPU GEMM summation order |
| K'' | zero-change copy with order="K" (layout preserved) | 3.8e-06 | layout preserved => noise floor |
| O | corrupt day-151 features, layout preserved | 3.8e-06 | = noise floor: no leak |

Conclusion: the six failures were an artifact of the validation harness (layout-changing `X.copy()` plus a
tolerance derived from a single pair of runs), not information flowing from the future. The model code was not
changed. The harness was fixed (layout-preserving perturbations with an explicit layout assertion; noise floor
measured over 4 runs; exact equality required on deterministic CPU). Final result: 19/19 on CPU (bitwise identical
past predictions at every cutoff, all seeds) and 19/19 on GPU (past-day change == noise floor).

The 3-ulp figure (1.831e-04 = 3 x 6.1e-05) is the largest-outlier logit re-rounded under a different GEMM order; in
probability space (what PR-AUC uses) the maximum change was 5.6e-06.
