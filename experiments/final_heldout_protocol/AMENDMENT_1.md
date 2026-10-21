# Held-out protocol, Amendment 1: adaptive arms (F), historical-feature reference line (H) and unweighted variants (U). NOT EXECUTED.

Dated 2026-10-21. **Status: designed and frozen on paper before any held-out access. The held-out period (days 152-182) has not been opened, read, hashed or scored; this record does not authorize doing so.** It adds to, and does not replace, `HELDOUT_EVALUATION_PROTOCOL.md` (SHA-256 `59bf1565fb16653a02a2d743cc445e114fb2a6a42f1de7b2fa56261e6f68c806`, unchanged). Every design choice below was made from TRAIN and VALIDATION information only (days 1-151); no held-out label, score or feature value was seen, and none may change anything here after the fact.

## 1. Why an amendment

`PROTOCOL_REVIEW.md` found three gaps between the base protocol and the original brief. This amendment closes the two that are research requirements of the brief and can be closed without any held-out information; the third is out of scope by design (section 2).

* The brief requires comparing static and adaptive performance **on future periods**. The base protocol leaves cell F (the adaptive system) and hypothesis H4 on validation only. Module **F** evaluates the three Phase 8 arms on the held-out period.
* The brief requires judging graph gains against the historical-feature reference line. Module **H** scores that frozen reference model on the held-out period.
* Class weighting changes XGBoost's validation PR-AUC (0.5395 weighted, 0.5948 unweighted), so the level of the A-versus-B comparison depends on that choice (tabular above neural holds in both recipes on validation); a held-out reading of the unweighted variants shows whether the conclusions depend on the weighting. Module **U** scores them.

## 2. Scope decisions (mandatory / optional / unsupported)

| Item | Decision | Why |
|---|---|---|
| 19 base models (A, B, C, D, E and Phase 7A controls) | evaluated by the base protocol | unchanged |
| Adaptive system and H4 on future data | **mandatory, added here (module F)** | brief: static vs adaptive on future periods |
| Historical-feature XGBoost | **added here (module H)** as a labelled reference line, not an A-F cell | the brief asks that graph gains be judged against it |
| Unweighted variants (A_none, B_none, C unweighted, E unweighted) | **added here (module U)** | the weighting finding must be checked on a future period |
| No-relations MLP control | **not evaluated** (optional, excluded) | a one-run control with no registered held-out hypothesis; it is a VAL diagnostic of whether the relations matter, which the held-out comparison of C against the tabular models already speaks to |
| H5 / community discovery | **unsupported on held-out data** | no label-based criterion exists: communities are candidates without ground truth; the validated evidence (Phase 10 and 11) stays as recorded |
| Neural relationship-level ablations | **conditional optional, not run** | not required by the brief or by any registered hypothesis; run only if a registered criterion needs them |
| `experiments/gat_300epoch/` and superseded runs | excluded | aborted or superseded, no usable result |

## 3. Module F: the adaptive arms on the held-out period

* **Arms and seeds:** F (frozen), Ad (adaptive), E (replay control); seeds 42, 123, 2024. Initial checkpoints, learning rate, number of update epochs K and the operating threshold are exactly the Phase 8 shared artifacts, chosen on TRAIN days only (`experiments/phase8_adaptive_drift/shared_init/`, `selected_from: TRAIN_internal_holdout`, fit days 1-111, selection days 112-121): K = 15, 10, 15 and thresholds 0.935917, 0.936041, 0.948971 for seeds 42, 123, 2024; learning rate 0.001; one persistent Adam with weight decay 1e-5 per run. Nothing is re-selected.
* **Windows:** five 6-day windows 152-157, 158-163, 164-169, 170-175, 176-181 (day 182 is not scored for these arms; the window structure is that of Phase 8).
* **Entity state:** the memory is first built by streaming days 1-151 in order, evaluation mode, no labels, no gradient (the state-continuity supplement, `experiments/phase8_state_continuity_supplement/`, showed the Phase 8 loop started with an empty memory; the burn-in is the deployment-faithful condition). The state is then carried without reset.
* **Updates (Ad):** after window k is scored, the model is updated on all labeled days up to the end of window k (days 1 to the window end, including VAL days 122-151, whose labels are known by then, and the held-out days of earlier windows), with the Phase 8 TBPTT update code. Labels of window k are requested only after window k is scored.
* **Replay control (E):** after window k is scored, the update trains on TRAIN days 1-121 plus resampled TRAIN days (36 resampled in round 1, 6 more each later round; `src/adaptive/replay.py::ReplayPoolBuilder`, replay windows 122-157, 158-163, ..., 176-181), so that its day count equals Ad's (157, 163, 169, 175, 181) while every day it uses is a TRAIN day. It therefore never sees a validation or held-out label.
* **No update after the last window; no tuning on any held-out label.** The initial checkpoint scores window 1 identically for all three arms.
* **Metrics:** per window and pooled, PR-AUC (primary), ROC-AUC, strict-segment PR-AUC, and at the TRAIN-selected threshold the confusion matrix, precision, recall, F1, alerts, false positives (`src/models/metrics.compute_metrics`).
* **Pre-registered descriptive check (the locked H4 rules, `src/heldout/h4.py`, same logic as `scripts/compare_phase8.py`):** (1) Ad's W1-to-W5 PR-AUC drop is smaller than F's in at least 2 of 3 seeds; (2) mean(Ad - F) of the window-mean PR-AUC is positive and exceeds the larger of the two arms' seed SDs (ddof = 1); (3) the same for Ad against E. The held-out reading is reported as "consistent with the recorded H4 verdict" if all three hold and "not consistent" otherwise, with the numbers. The recorded H4 verdict on validation (SUPPORTED, with and without the burn-in) is not re-decided.
* **Readiness gate on validation (before any held-out access):** `scripts/run_f_walkforward.py --split val --variant readiness` repeats the burn-in run over days 122-151 and compares it with the recorded supplement run (`experiments/phase8_state_continuity_supplement/outputs/readiness/readiness_F.json`). Gate as stated before it was run: F scores and window 1 of every arm within 1e-6 (deterministic scoring path); Ad and E window-mean PR-AUC within 0.01 per seed (trained arms, not bit-reproducible).
  **Outcome, reported as it happened:** all 21 deterministic checks passed (F and window 1 of every arm, three seeds); of the six trained-arm window-mean checks **five passed and one failed at the stated 0.01** (E, seed 123: 0.0104; the others 0.0004-0.0060). The registered 0.01 is not changed and the gate is therefore recorded as **FAIL as written**. Post-hoc reading, labelled as such: the failing difference is run-to-run noise of a trained arm (dropout and GPU arithmetic), the same size as the single-window noise seen in the supplement (up to 0.03), and does not indicate a code difference, because the update path is bit-for-bit equal to Phase 8's loop under identical seeding (`tests/test_walkforward_ext.py`). If the user authorizes module F, that authorization is given with this failed-as-written gate known; the alternative is not to authorize F or to ask for a different gate to be frozen first.
* **Outputs:** `experiments/final_heldout_protocol/heldout_results_f/` only; one-shot (section 7).

## 4. Module H: historical-feature XGBoost reference line

* **Model:** `experiments/xgboost_historical_control/model.joblib`, the 12 strictly-historical account-level features (`src/features/historical.py`), A-recipe hyperparameters, seed 42, trained on days 1-121, threshold 0.914131 recorded on validation.
* **Inputs:** the feature rows of `data/features/historical_features.parquet` for the held-out TransactionIDs. These features are label-derived (counts of earlier fraud on the same entity), so a held-out row's value uses labels of strictly earlier transactions, including earlier held-out transactions. This assumes zero label delay, exactly as the validation result did; it is why the model is a reference line, not an A-F cell, and why the base protocol described its signal as the labelling convention.
* **Readiness gate:** `scripts/run_amendment_evaluation.py --split val --modules H` reproduces the recorded validation predictions within 1e-9. **Outcome:** PASS (maximum score difference 0.0; record `readiness_val_amendment/readiness_report.json`).
* **Pre-registered descriptive check:** is the pooled PR-AUC of the best base model (XGBoost, B_full, and each neural cell mean) above or below this reference line, as on validation? Direction only; no significance statement.

## 5. Module U: unweighted variants

* **Models:** `A_none` and `B_none` (XGBoost, comparison-A recipe, `scale_pos_weight` = 1, seed 42; A = the 500 comparison-A features, B = A plus the 40 graph features); GraphSAGE unweighted (seeds 42, 123, 2024, `experiments/unweighted_graphsage/`); Temporal GraphSAGE unweighted (seeds 42, 123, 2024, `experiments/unweighted_temporal_graphsage/`).
* **XGBoost model files:** the follow-up recorded metrics and predictions but not model files, so A_none and B_none were re-fitted with the identical recipe and inputs (TRAIN and VAL data only, `scripts/freeze_unweighted_xgboost.py`) and accepted only because they reproduce the recorded validation results (B_none: every validation score within 1e-9 and the same best iteration; A_none: pooled PR-AUC within 1e-9 and the same best iteration). Record: `experiments/final_heldout_protocol/frozen_amendment/reproduction.json`. If a model had not reproduced, it would have been dropped from this module.
* **Thresholds:** the recorded validation best-F1 thresholds of each unweighted model (not re-tuned); the shift in alert counts is reported.
* **Readiness gate:** `scripts/run_amendment_evaluation.py --split val --modules U`: B_none within 1e-9 on scores; A_none pooled PR-AUC within 1e-9 of the recorded value; neural models within 1e-4 on scores and 1e-6 on PR-AUC (as in the base gate). **Outcome:** PASS for all 8 models (XGBoost differences 0.0; neural maximum score difference 3.0e-07); the XGBoost re-fit reproduced A_none and B_none exactly (PR-AUC difference 0.0, same best iterations 998 and 999).
* **Pre-registered descriptive check:** for each pair (weighted, unweighted) of A XGBoost, B, GraphSAGE and Temporal GraphSAGE, is the unweighted pooled PR-AUC higher or lower, as on validation? Direction only.

## 6. Reporting

Held-out numbers go to separate files (`heldout_results_f/`, `heldout_results_amendment/`) and are never merged into validation tables. Multi-seed cells show mean and sample SD (ddof = 1) and per-seed values. Unfavourable results are reported in full. Module F is a different design from the base models (five 6-day windows, day 182 unscored, labels consumed after each window), so its numbers are compared with the base models' only on the same days if at all, and the difference in design is stated.

## 7. One-shot rule and authorization (additional to the base protocol)

* Each output folder is executed once. A crash before any held-out score is written may be repaired and repeated, with the repair documented; after any held-out metric exists nothing may be changed in response.
* The amendment is honoured only if `experiments/final_heldout_protocol/AUTHORIZATION.txt` **also** contains this amendment's SHA-256 (`AMENDMENT_1.sha256`) and the sentence `I authorize the following Amendment 1 modules: <letters>` naming the modules the user authorizes (any of F, H, U, separated by commas); a module not named is refused. The base authorization alone never unlocks an amendment module, and an amendment hash without the base authorization unlocks nothing.
* The user should name all desired modules at once: H and U share one output folder and one-shot flag.
* Commands: `.venv/Scripts/python.exe -B scripts/run_heldout_evaluation.py --split heldout` (base), `scripts/run_f_walkforward.py --split heldout` (F), `scripts/run_amendment_evaluation.py --split heldout --modules H,U` (H, U).

## 8. Interpretation limits stated in advance

One future period of 31 days (30 scored days for F), one dataset, three seeds that are not independent replications; held-out numbers describe that period only; concept drift may move every model; zero label delay for the adaptive arms and the reference line; the thresholds are transferred from validation (or TRAIN) and may be mis-calibrated on the future period, which PR-AUC does not depend on; no hypothesis other than the descriptive checks above is tested.

## 9. Authorization wording

> I authorize the following Amendment 1 modules: F, H, U. (The user may name any subset.) I understand this irreversibly uses the held-out period for those modules and that results will be reported whatever they are.
