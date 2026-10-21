# Drift-triggered refresh (arm Tr), validation period: pre-registration

Dated 2026-10-21; written and hashed (`PREREGISTRATION_TR.sha256`) before any Tr arm exists. **Validation period (days 122-151) only; no held-out data is read; nothing from Phases 1-12 is modified; the locked H4 rules and the recorded H4 verdict are not changed.** This is an additional, clearly labelled analysis.

## 1. Why

The original brief's adaptation requirement reads: baseline = a static model trained once; adaptive model = "monitor drift/performance; trigger retraining or model refresh when predefined conditions are met"; "the trigger must be determined using validation/development data rather than tuned on the final test set"; compare static versus adaptive on future periods. Phase 8 tested a **fixed-cadence** refresh (every 6-day window) and recorded a drift-triggered policy as out of scope (`DESIGN_BEFORE_EXECUTION.md`). This analysis adds the triggered policy the brief names, so that the brief's wording is covered by a pre-registered experiment instead of by an interpretation.

## 2. Design (fixed now)

* **Arm Tr** = the adaptive arm Ad with the refresh decided by a monitor. Same shared initial checkpoints, learning rate, number of update epochs K, optimizer (one persistent Adam, weight decay 1e-5), update procedure (all labeled days up to the end of the window just scored, TBPTT update code) and the same windows 122-127 ... 146-151 as Phase 8; the entity memory is built over days 1-121 first (the state-continuity supplement, the deployment-faithful condition). The only difference from Ad: after window k (k = 1..4) the model is refreshed **only if the monitor fires**.
* **Monitor (label-free):** population stability index (PSI) of the score distribution of the window just scored against a reference window's, 10 bins with edges at the reference's deciles, smoothing 1e-4 (`src/trigger/policy.py`). The reference is the first window scored after the start or after the latest refresh (a refreshed model emits a different score distribution and is compared with itself); the window that re-baselines never triggers. The monitor receives scores only, never a label (tested; the policy module imports only NumPy).
* **Threshold, determined on TRAIN data only (never on validation or held-out data):** tau = max(0.10, 2 x PSI_noise), where PSI_noise is the PSI between the initial checkpoint's scores on TRAIN days 112-116 and on TRAIN days 117-121 (state built over days 1-111, evaluation mode, no labels; the checkpoint was fitted on days 1-111, so these are its internal hold-out days). One tau per seed. The floor 0.10 is the conventional "moderate shift" PSI level and is fixed here.
* **Labels:** as in Ad, labels of window k are used only after window k is scored, and only for a refresh; zero label delay is assumed (as Phase 8).
* **Comparators:** F and Ad from the recorded state-continuity run (`experiments/phase8_state_continuity_supplement/outputs/burn_in/summary.json`), same condition. Seeds 42, 123, 2024.
* **Outputs:** `experiments/phase8_triggered_refresh/outputs/` only. Row-level prediction files stay local (repository publication policy).

## 3. Criteria (descriptive, fixed now; none re-decides H4)

* **TR1 execution:** Tr runs for all three seeds; its window 1 equals that of F within 1e-6 (no update precedes it).
* **TR2 label-free, TRAIN-only trigger:** the policy module imports no data or label source (unit test) and tau and PSI_noise are recorded per seed from TRAIN days only.
* **TR3 benefit over static:** mean over seeds of the window-mean PR-AUC of Tr minus that of F is positive and exceeds the larger of the two arms' seed SDs (ddof = 1).
* **TR4 efficiency:** the mean number of refreshes per run (of 4 possible) is below 4.
* **TR5 parity with periodic refresh:** mean(Ad - Tr) of the window-mean PR-AUC does not exceed the larger of the two arms' seed SDs.
* **Triggered-refresh outcome:** **SUPPORTED** only if TR1-TR5 all hold; otherwise **NOT SUPPORTED**, reported with which criteria held. If the monitor never fires, Tr equals F and TR3 fails; if it always fires, Tr equals Ad and TR4 fails; both would be reported as such, and tau would not be changed.

## 4. Limits stated in advance

Three seeds sharing the same data are not independent replications; one training run per seed and arm; the monitor is evaluated at 4 points per run; one monitor type (score-distribution PSI) and one threshold rule; GPU training is not bit-reproducible (window-mean differences of about 0.004 between identical runs were measured in the Module F diagnosis), which TR3 and TR5 do not remove; zero label delay; validation period only. Nothing here is a held-out result.
