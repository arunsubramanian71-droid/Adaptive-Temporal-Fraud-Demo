# Class-imbalance check for the tabular XGBoost: pre-registration

Dated 2026-10-20; written and hashed (`PREREGISTRATION_IMBALANCE.sha256`) before the script or any fit exists. Authority: the master instruction of 2026-10-20 and the original brief's CLASS IMBALANCE section: "Any imbalance technique must be evaluated experimentally." Every model in this project used class weighting (`class_weight='balanced'`, `scale_pos_weight` or `pos_weight` = 27.36) as a design choice without comparing it with not using it. This study evaluates that choice for the one model family where it is cheap: the comparison-A XGBoost.

Standing statement: **VAL-period results (days 122-151), one seed, VAL used for early stopping and the best-F1 threshold as in comparison A. No TEST data is read.** The neural graph models' `pos_weight` was not re-evaluated and focal loss was not tried (no evidence of a failure that would justify it); both are stated as not evaluated.

## Design (fixed now)

Features, recipe and data are exactly those of comparison A (`scripts/train_xgboost.py`; 470 features; `max_depth=6`, `learning_rate=0.05`, `n_estimators=1000`, `early_stopping_rounds=50` on VAL `aucpr`, `random_state=42`, CPU). The only variable is `scale_pos_weight`:

| Arm | `scale_pos_weight` |
|---|---|
| `none` | 1.0 (no class weighting) |
| `sqrt` | the square root of the TRAIN negatives / positives ratio |
| `full` | TRAIN negatives / positives (the recorded comparison-A setting; reused from the B pipeline's `A_refit` arm, which re-fits it exactly) |

No resampling (SMOTE is not applied to any model). The reference population, strict segments and metric code are those of Phase 12 (`src/evidence/table.py`); operating-point metrics use the best-F1-on-VAL threshold of each arm.

## Verdict rule (fixed now; one seed, so descriptive)

Primary metric: pooled VAL PR-AUC. Let `spread` = max - min over the three arms and `tol` = 0.005 (the same tolerance used as the pipeline-control bound in `PREREGISTRATION_B.md`).
* **WEIGHTING MAKES NO MATERIAL DIFFERENCE** (on the threshold-free metric) if `spread` <= `tol`.
* **WEIGHTING HELPS** if PR-AUC(`full`) exceeds both other arms by more than `tol`.
* **WEIGHTING HURTS** if PR-AUC(`full`) is below the best other arm by more than `tol`.
* Otherwise **INCONCLUSIVE**.
Operating-point behaviour (number of alerts, precision, recall at the best-F1 threshold, ROC-AUC, top-K) is reported descriptively, with no rule.

## Acceptance criteria

* **IA1** This record precedes any fit; its hash is in the output manifest.
* **IA2** Only `data/features/train_features.parquet`, `val_features.parquet`, the Phase 12 reference predictions file, this record and the `A_refit` arm of `experiments/comparison_B_graph_features/outputs/arms/A_refit/` are read; no TEST-named path is opened.
* **IA3** Each new arm scores exactly the 83,655 reference ids with identical labels.
* **IA4** The `full` arm is the existing `A_refit` result (PR-AUC 0.5395, best iteration 798), not a new fit with a different setting.
* **IA5** The verdict rule is applied as written and reported whatever it returns; outputs have SHA-256 sidecars; Phase 9 wording lint is clean.

## Limits stated in advance

One seed per arm; VAL-tuned; PR-AUC differences below `tol` are not interpreted. A threshold-free metric can be indifferent to weighting while the operating point and calibration change; the study reports both but draws no conclusion about neural models.
