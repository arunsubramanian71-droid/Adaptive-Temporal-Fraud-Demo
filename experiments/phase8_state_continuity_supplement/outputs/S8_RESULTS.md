# Phase 8 state-continuity supplement: results

**Validation period (days 122-151); an additional reading beside the recorded Phase 8 results, which are unchanged. No held-out data was read.**

Protocol: `PREREGISTRATION_S8.md` (hashed before any arm). The only change from Phase 8: the entity memory is built over days 1-121 before the first window instead of starting empty.

## Window-mean PR-AUC per seed (seeds 42, 123, 2024)

| Arm | Recorded Phase 8 (empty memory at day 122) | Supplement (memory built over days 1-121) |
|---|---|---|
| F | 0.4196, 0.4144, 0.4388 | 0.4324, 0.4196, 0.4451 |
| Ad | 0.4486, 0.4631, 0.4624 | 0.4724, 0.4694, 0.4783 |
| E | 0.4112, 0.4186, 0.4238 | 0.4336, 0.4367, 0.4325 |

## Locked H4 rules (identical logic)

| Rule | Recorded Phase 8 | Supplement |
|---|---|---|
| 1. Ad degrades less than F in at least 2 of 3 seeds | True | True |
| 2. mean(Ad - F) exceeds the larger seed SD | True (+0.0338 vs 0.0128) | True (+0.0410 vs 0.0128) |
| 3. mean(Ad - E) exceeds the larger seed SD | True (+0.0402 vs 0.0082) | True (+0.0391 vs 0.0045) |
| **H4** | **SUPPORTED** | **SUPPORTED** |

The H4 verdict does not depend on the memory initialisation; the absolute values of all three arms are higher when the memory is built first, so the recorded Phase 8 absolute values understate what the same checkpoints achieve in the deployment-faithful condition.

## Registered acceptance outcome

| Criterion | Result |
|---|---|
| S1 reproduction of Phase 8 as written, seed 42 (F and window 1 of every arm within 1e-6; Ad/E later windows within 0.01 PR-AUC) | **FAIL**: F and window 1 pass; Ad/E later windows fail |
| S2 execution (all arms and seeds; window 1 identical across arms within a seed, max difference 1.8e-07) | PASS |
| S3 causality (stand-in model test) | PASS (`tests/test_walkforward_ext.py`) |
| S4 hygiene | PASS (hashes, lint, dataset truncated at day 151) |

**S1 failed as registered for the trained arms.** 6 of 8 later windows of Ad and E differ from the recorded values by more than 0.01 PR-AUC (largest 0.0303); the individual scores differ widely (largest 1.00) because the update epochs use dropout and GPU arithmetic that are not reproducible between runs. The criterion was not changed. Post hoc diagnostic (labelled as such): the scoring path is exact (F and window 1 differ by about 1e-7, and the unit test shows the new function equals Phase 8's loop bit for bit when training is seeded identically); the window-mean PR-AUC of the re-run is Ad 0.4460 (recorded 0.4486), E 0.4082 (recorded 0.4112). Single windows are therefore noisy at the 0.01-0.03 level between runs, and recorded single-run window values should not be read finer than that.

## Limits

Three seeds sharing the same data are not independent replications; one training run per seed and arm; validation period only; zero label delay as in Phase 8.
