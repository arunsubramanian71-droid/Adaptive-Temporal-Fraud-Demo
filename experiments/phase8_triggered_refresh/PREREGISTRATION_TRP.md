# Performance-triggered refresh (arm TrP), validation period: pre-registration

Dated 2026-10-21; written and hashed (`PREREGISTRATION_TRP.sha256`) before any TrP arm exists. **Validation period (days 122-151) only; no held-out data is read; the locked H4 rules, the recorded H4 verdict and the result of the score-PSI trigger (`PREREGISTRATION_TR.md`, outcome NOT SUPPORTED) are not changed.**

## 1. Why

The brief's adaptation requirement says the adaptive model should "monitor drift/performance" and refresh "when predefined conditions are met", with the trigger "determined using validation/development data". The first triggered analysis (arm Tr) monitored only the drift of the score distribution (label-free) and never fired, so the brief's **performance** half was not tested. This analysis adds a trigger that monitors the model's PR-AUC on each window once that window's labels are available.

## 2. Design (fixed now)

* **Arm TrP** = arm Ad with the refresh decided by a performance monitor; everything else as in arm Tr and Ad (same shared checkpoints, K, learning rate, optimizer, update procedure, windows 122-127 ... 146-151, entity memory built over days 1-121, seeds 42, 123, 2024, zero label delay).
* **Monitor:** after window k (k = 1..4) is scored, its labels are available (the assumption Phase 8 makes for every update). PR-AUC of window k is compared with the PR-AUC of the reference window, the first window scored after the start or after the latest refresh; the monitor fires when the relative drop (reference - current) / reference exceeds delta. The window that re-baselines never triggers. Labels of window k are used only by this monitor and by the refresh that follows, never to score window k (`src/trigger/performance.py`; the walk-forward function passes the labels of the window just scored and no others; unit-tested).
* **Threshold delta, determined on TRAIN days only:** delta = min(0.5, max(0.10, 2 x r)), where r = |AP_A - AP_B| / mean(AP_A, AP_B), AP_A and AP_B being the PR-AUC of the initial checkpoint on TRAIN days 112-116 and 117-121 (its internal hold-out days; the checkpoint was fitted on days 1-111; state built over days 1-111). One delta per seed. No validation or held-out data enters the threshold.
* **Comparators:** F and Ad from the recorded state-continuity run (`outputs/burn_in/summary.json` of `experiments/phase8_state_continuity_supplement`), and Tr from `outputs/`. Outputs of this analysis: `experiments/phase8_triggered_refresh/outputs_performance/`. Row-level predictions stay local (publication policy).

## 3. Criteria (descriptive, fixed now; none re-decides H4)

* **TRP1 execution:** TrP runs for all three seeds and its window 1 equals that of F within 1e-6.
* **TRP2 threshold from TRAIN days only:** delta recorded per seed from the stated rule; the label-passing behaviour is unit-tested.
* **TRP3 benefit over static:** mean over seeds of the window-mean PR-AUC of TrP minus that of F is positive and exceeds the larger of the two arms' seed SDs (ddof = 1).
* **TRP4 efficiency:** the mean number of refreshes per run (of 4 possible) is below 4.
* **TRP5 parity with periodic refresh:** mean(Ad - TrP) of the window-mean PR-AUC does not exceed the larger of the two arms' seed SDs.
* **Outcome:** **SUPPORTED** only if TRP1-TRP5 all hold; otherwise **NOT SUPPORTED**, reported with which criteria held. If the monitor never fires, TrP equals F and TRP3 fails; if it always fires, TrP equals Ad and TRP4 fails; delta would not be changed in either case.

## 4. Limits stated in advance

Three seeds sharing the same data are not independent replications; one training run per seed and arm; GPU training is not bit-reproducible (identical-seed window means differed by about 0.004 in the Module F diagnosis); the monitor is evaluated at 4 points per run on windows of a few hundred positives; one monitor type and one threshold rule; zero label delay; validation period only. Nothing here is a held-out result.
