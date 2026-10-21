# Held-out (days 152-182) evaluation: frozen protocol. NOT EXECUTED.

Dated 2026-10-20. **Status: designed and frozen on paper; the held-out period has not been read, scored or evaluated, and this record does not authorize doing so.** No model, threshold, feature set or hypothesis verdict may be changed because of anything that is later learned from this period.

## 1. Why this exists, and why it has not been run

The original brief requires a chronological split with a "latest/future period" held out, requires every preprocessing and hyperparameter decision to be made without it, and lists future-time performance among the claims to be tested (H3, H4, and the central question's emphasis on future-time evaluation). The project froze the split at TRAIN days 1-121, VAL 122-151 and held-out days 152-182, and the phase reports record that the held-out period was never loaded for training, selection or scoring. The user has repeatedly kept it sealed. Opening it is irreversible: once any model has been scored on it, it can never again serve as an untouched final test. Because the sealed status is an explicit standing restriction, the decision to open it belongs to the user, not to the project's own plan. The exact authorization wanted is in section 9.

## 2. The frozen model registry (fixed now, from VAL artifacts only)

Every model below exists as a saved artifact trained on days 1-121 (or, for the shared F checkpoint, 1-111). Nothing is re-fit, re-selected or re-thresholded. Operating-point metrics use each model's recorded VAL threshold; thresholds are transferred, not re-tuned, and the shift in alert counts is reported.

| Cell | Model | Artifact | Threshold source |
|---|---|---|---|
| A | logistic regression, random forest, XGBoost | `experiments/{logistic_regression,random_forest,xgboost}/model.joblib` + `feature_builder/` | `metrics_val.json` best-F1-on-VAL |
| B | XGBoost + 40 graph features (`B_full`, A recipe, seed 42) | `experiments/comparison_B_graph_features/outputs/arms/B_full/` | its recorded best-F1-on-VAL |
| C | GraphSAGE, seeds 42, 123, 2024 | `experiments/graphsage_robust/seed_*/` | recorded per run |
| D | GAT, seeds 42, 123, 2024 (convergence unresolved; reported as such) | `experiments/gat/` | recorded per run |
| E | Temporal GraphSAGE, seeds 42, 123, 2024; Phase 7A controls (schedule-matched static, memory ablated) | `experiments/temporal_graphsage/`, `experiments/temporal_ablation/` | recorded per run |
| F | the adaptive system | **not evaluated**: the Phase 8 walk-forward design (five 6-day windows, updates on already-scored labels) is defined only for the VAL window; a held-out version would need a new frozen design | n/a |

Excluded: `experiments/gat_300epoch/` (aborted, no usable result), every superseded run, the account-history XGBoost (a reference line whose signal is the labelling convention) and the no-relations MLP control (a one-run control). Neither is evaluated on the held-out period.

## 3. Procedure (causal, as on VAL)

* Tabular models: apply the saved TRAIN-fit feature builder to the held-out feature rows and score.
* B: the 40 graph features are computed over the full timeline with the same function (`src/features/graph_features.py`); a held-out row sees only strictly earlier rows (TRAIN, VAL and earlier held-out rows).
* Static neural models: `rolling_predict` over days 152-182 (graph masked to `edge_time <= t` for each day t), checkpoint fixed.
* Temporal models: `rolling_temporal_predict` from day 1 with entity state carried forward without reset through VAL into the held-out period, checkpoint fixed. No label of any day enters any state.
* No update, fine-tuning, early stopping, threshold search or calibration uses a held-out label.

## 4. Metrics and reporting

Same as the Phase 12 table, computed by the same code: pooled PR-AUC (primary), ROC-AUC, strict-segment PR-AUC (warm, cold, no-edge), Precision@K and Recall@K (K = 0.1%, 0.5%, 1%, 5%), and at the recorded threshold the confusion matrix, precision, recall, F1, alerts and false positives. Multi-seed cells show mean and sample SD (ddof = 1) and the per-seed values. Held-out numbers are reported in separate files and tables from VAL and are never merged with them.

## 5. Pre-registered descriptive checks (no hypothesis is re-decided)

For each question the held-out result is recorded as "same direction as VAL" or "different direction", with the numbers; no p-value, interval or significance statement is made.
1. Is pooled PR-AUC of the best tabular model (A XGBoost or B) above that of every neural graph cell, as on VAL?
2. Is Temporal GraphSAGE (E) above its memory-ablated and schedule-matched static controls in mean, and in how many of three seeds?
3. Is B above the comparison-A XGBoost?
4. Is E above GraphSAGE (C) and GAT (D)?
H2 and H1 verdicts, H4 and H5 remain exactly as recorded on VAL; this evaluation adds a held-out reading beside them, not a replacement.

## 6. Code-readiness gate (before any held-out access)

The evaluator (`src/heldout/evaluator.py`, `scripts/run_heldout_evaluation.py`) must first be run with `--split val` through exactly the same code path and reproduce the recorded VAL predictions: tabular models and B within 1e-9 on scores, neural models within 1e-4 on scores and 1e-6 on PR-AUC (GPU kernels are not bit-reproducible), with the per-model maximum differences recorded in `experiments/final_heldout_protocol/readiness_val/readiness_report.json`. Its unit tests (`tests/test_heldout_gate.py`) must pass. Only then may the held-out mode be enabled, and only by a file `experiments/final_heldout_protocol/AUTHORIZATION.txt`, created by the user, that quotes this protocol's SHA-256 (in `HELDOUT_EVALUATION_PROTOCOL.sha256`) and the sentence below; the evaluator refuses to read any held-out file otherwise, and refuses to run twice.

## 7. One-shot rule

The held-out run is executed once. If it crashes before any held-out score is written, the defect may be fixed and the run repeated, with the fix documented. After any held-out metric has been computed, no model, threshold, feature, evaluator setting or exclusion may be changed in response, and all results, including unfavourable ones, are reported. Further looks at the same period would be adaptive reuse and are not allowed.

## 8. Interpretation limits stated in advance

One future period of 31 days, one dataset, three seeds that are not independent replications; held-out numbers describe that period only; concept drift between VAL and the held-out period may move every model; F and the adaptive claim remain VAL-only; community discovery is out of scope for this evaluation.

## 9. Authorization needed from the user

> I authorize one-time evaluation of the frozen models listed in section 2 on the held-out period (days 152-182) under `experiments/final_heldout_protocol/HELDOUT_EVALUATION_PROTOCOL.md`, once the code-readiness gate in section 6 has passed. I understand this irreversibly uses the held-out period and that results will be reported whatever they are.

Until that sentence is given, in `AUTHORIZATION.txt` together with this protocol's SHA-256, no held-out file is opened, hashed, read or scored. The command afterwards is `.venv/Scripts/python.exe -B scripts/run_heldout_evaluation.py --split heldout`.
