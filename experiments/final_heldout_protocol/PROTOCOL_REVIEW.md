# Held-out protocol: readiness review (no authorization requested or created)

Dated 2026-10-21. Read-only. **No held-out file was opened, hashed, read or scored; `AUTHORIZATION.txt` does not exist.**

* Protocol: `experiments/final_heldout_protocol/HELDOUT_EVALUATION_PROTOCOL.md`, SHA-256 `59bf1565fb16653a02a2d743cc445e114fb2a6a42f1de7b2fa56261e6f68c806` (sidecar matches: True).
* Frozen models: 19 (all artifacts present: True); thresholds are the recorded validation best-F1 values below and are transferred, never re-tuned.
* Code-readiness gate: validation re-scoring of all 19 models reproduces the recorded predictions (maximum score difference 7.15e-07); recorded after the last change to the evaluator: True.
* Evaluator hashes at readiness: `evaluator.py` `bc6157f57279e11d...`, `run_heldout_evaluation.py` `45b3f0351d8eac9c...`.
* **Verdict: READY for informed authorization.** Readiness means the procedure is frozen and checked on validation; it does not mean the evaluation answers every question of the brief (see the gaps below).

## Frozen models and recorded validation thresholds

| Model | Recorded validation threshold |
|---|---|
| B_full | 0.809044 |
| gat_s123 | 0.885675 |
| gat_s2024 | 0.890225 |
| gat_s42 | 0.863504 |
| graphsage_s123 | 0.913051 |
| graphsage_s2024 | 0.921920 |
| graphsage_s42 | 0.919313 |
| logistic_regression | 0.872362 |
| memory_ablated_s123 | 0.963542 |
| memory_ablated_s2024 | 0.955472 |
| memory_ablated_s42 | 0.921510 |
| random_forest | 0.585129 |
| static_schedule_matched_s123 | 0.935711 |
| static_schedule_matched_s2024 | 0.931045 |
| static_schedule_matched_s42 | 0.928433 |
| temporal_graphsage_s123 | 0.944629 |
| temporal_graphsage_s2024 | 0.962289 |
| temporal_graphsage_s42 | 0.963329 |
| xgboost | 0.774651 |

## Input restrictions and one-shot behaviour (each tested without touching held-out data)

* Without `AUTHORIZATION.txt` quoting the protocol hash and the sentence, the evaluator raises before reading anything; a wrong hash or sentence is refused; a second run is refused once `heldout_results/.completed` exists.
* The held-out feature file is named in exactly one place in the code and is reachable only after the gate; feature frames are read with the label column projected away; labels are read afterwards from the graph file only to compute metrics; the temporal dataset is built without labels.
* When authorized it reads: `data/features/test_features.parquet` (tabular models and B) and the Phase 4 graph file (neural models; all days are loaded into memory, the models score days 152-182 under the causal masks); it writes only into `experiments/final_heldout_protocol/heldout_results/`.

## Consistency with the original brief

* Consistent: chronological split with the latest period held out; decisions and hyperparameters fixed without it; the brief's metrics (PR-AUC primary, ROC-AUC, precision, recall, F1, Precision/Recall@K, confusion matrices, alerts, false positives) are all computed.
* **Gap 1 (closed only by Amendment 1 module F, if authorized):** the brief's adaptation requirement ("compare static versus adaptive performance on future periods") and cell F are not evaluated by the base protocol; H4 would stay validation-only.
* **Gap 2 (modules H and U, if authorized):** the reference historical-feature XGBoost and the unweighted variants are not in the base protocol; the no-relations MLP control is deliberately excluded.
* **Gap 3 (unsupported):** community discovery (H5) is outside this evaluation; no label-based held-out criterion exists.
* **Amendment 1:** `experiments/final_heldout_protocol/AMENDMENT_1.md`, SHA-256 `37a1c0577a8ef28530a6120a027890c11ec458ca97329d14a05108db21a27e6e` (sidecar matches: True); module readiness on validation: H and U PASS (9 models); F: deterministic checks pass, trained-arm window-mean gate failed as written for 1 of 6 cases (see the amendment). Scope decisions: `SCOPE_RECONCILIATION.md`.
* Caution: GAT is evaluated as the frozen non-converged runs; a held-out difference between GAT and the others cannot separate inferiority from under-training.

## What the evaluation would make irreversible

* The held-out period could never again serve as untouched test data for any model or decision in this project.
* It is one-shot; after any held-out metric exists, no model, threshold, feature, evaluator setting or exclusion may be changed in response, and all results (including unfavourable ones) must be reported.
* Each current validation-only caveat would be supplemented by a held-out reading; verdicts H1-H5 as recorded would not be rewritten.
* Reading the held-out features and labels cannot be undone, and any later model selection that looked at the outcome would be invalid.

## Residual risks stated for the decision

* One future period of 31 days: concept drift may move every model; seeds are not independent replications.
* GPU scores differ from the recorded validation predictions by up to the figure above; this affects tie-sensitive top-K metrics only marginally.
* If the run crashes before any held-out score is written the defect may be fixed and rerun (protocol section 7); a crash while writing leaves partial outputs that must be handled by hand before a rerun.

## Exact authorization

Create `experiments/final_heldout_protocol/AUTHORIZATION.txt` containing the sentence in protocol section 9 and the protocol hash above; to add Amendment 1 modules it must also contain the amendment hash above and the sentence `I authorize the following Amendment 1 modules: <F, H, U subset>`. Then run `scripts/run_heldout_evaluation.py --split heldout` (base), `scripts/run_f_walkforward.py --split heldout` (F), `scripts/run_amendment_evaluation.py --split heldout --modules H,U` (H, U). Nothing has been created; the decision is the user's.
