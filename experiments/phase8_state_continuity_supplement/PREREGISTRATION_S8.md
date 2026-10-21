# Phase 8 state-continuity supplement (validation period): pre-registration

Dated 2026-10-21; written and hashed (`PREREGISTRATION_S8.sha256`) before any arm of this supplement exists. Nothing in Phase 8 is modified: the recorded Phase 8 results, report and H4 verdict stay exactly as they are. This supplement adds a second, clearly separated reading.

## 1. The finding that motivates it (disclosed, measured before this record)

Phase 8's walk-forward code (`src/adaptive/update_loop.py::run_walk_forward`) creates the entity memory ("state") once and scores the first validation window (days 122-127) immediately: **the memory is never built over the TRAIN days 1-121 first**, so every arm starts window 1 with an empty memory. The module text and the design intend the state to be carried unreset from the training period into validation, as in Phase 7 (whose evaluation rebuilds the state from day 1 and continues without reset). Measured on the frozen arm F, seed 42, validation data only: the loop as written reproduces the recorded predictions to 1.2e-07 (so the recorded results are exactly what that loop produces), while building the state over days 1-121 first changes scores by up to 0.71-0.85 and the window-mean PR-AUC from 0.4196 to 0.4324. Only F seed 42 was measured; nothing else was run before this record.

All three arms start from the same empty memory, so the comparison among them (the H4 rules) is internally consistent, but the arms are evaluated in a condition that is not the deployment-faithful one, and the absolute F, Ad and E values are understated.

## 2. Design (fixed now)

* **Arms and seeds:** F, Ad, E; seeds 42, 123, 2024; the shared initial checkpoints `experiments/phase8_adaptive_drift/shared_init/model_checkpoint_initial_seed{S}.pt` and `hparams_seed{S}.json` (fine-tune epochs K, learning rate, threshold, chosen on TRAIN days 112-121 only). Windows, update rule, optimizer (one persistent Adam, weight decay 1e-5), replay-pool construction and base seeds are exactly Phase 8's.
* **The only change:** before window 1 the state is built by streaming days 1-121 in order with the model in evaluation mode, no labels, no gradient, and is then carried unreset through all windows. Updates still rebuild their own internal state from day 1, as in Phase 8.
* **Implementation:** a new function `src/heldout/walkforward.py::run_walk_forward_ext` (the Phase 8 files are not touched); with `burn_in_through = None` it must reproduce Phase 8 exactly.
* **Outputs** go to `experiments/phase8_state_continuity_supplement/outputs/`; nothing outside it is written.

## 3. Decision rules (the locked Phase 8 H4 rules, applied unchanged by the same logic as `scripts/compare_phase8.py`)

1. The adaptive arm degrades less (W1 to W5 PR-AUC drop) than the frozen arm in at least 2 of 3 seeds. 2. mean(Ad - F) of the window-mean PR-AUC is positive and exceeds the larger of the two arms' seed SDs. 3. The same for Ad against the replay control. H4 is SUPPORTED under this supplement only if all three hold; otherwise it is reported as partial or not established. The result is reported beside the recorded Phase 8 verdict and replaces nothing.

## 4. Acceptance criteria

* **S1 reproduction of Phase 8 as written (readiness):** with `burn_in_through = None`, seed 42, all three arms: F scores equal the recorded predictions to within 1e-6; Ad and E per-window PR-AUC within 0.01 of the recorded values (training on the GPU is not bit-reproducible; this allowance is fixed now); window 1 identical across arms within 1e-6.
* **S2 execution:** the burn-in variant runs for all three arms and seeds; window 1 is identical across arms within a seed (no update precedes it); the state was built over days 1-121 only.
* **S3 causality:** a test with a stand-in model shows labels of a window are never requested before that window is scored, that updates occur only between windows, and that the burn-in streams exactly days 1 to the day before window 1 in order.
* **S4 hygiene:** inputs and outputs hashed; Phase 9 wording lint clean; no held-out day is loaded (the dataset is truncated at day 151).

## 5. Limits stated in advance

Three seeds sharing the same data are not independent replications; validation period only; zero label delay is assumed as in Phase 8; the supplement corrects the memory initialisation only and does not address any other Phase 8 limitation.
