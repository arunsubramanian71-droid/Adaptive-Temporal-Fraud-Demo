# Graph-level evidence for the Temporal GraphSAGE model: pre-registration

Dated 2026-10-21; written and hashed (`PREREGISTRATION_NEURAL_EVIDENCE.sha256`) before the code that produces the evidence exists. Authority: the original brief's EXPLAINABILITY section ("for graph models, provide interpretable evidence such as important neighbouring entities, influential relationships, ... neighbourhood perturbation, suspicious connected structure"); the Phase 11 scope named "graph evidence for one already-designated temporal GraphSAGE checkpoint" and its first pass deferred it; the 2026-10-21 instruction to complete independent mandatory work. This record covers the checkpoint that is the project's best neural model (Temporal GraphSAGE, seed 42, `experiments/temporal_graphsage/seed_42/model_checkpoint.pt`). It does not cover GraphSAGE or GAT: GAT never converged and attention weights are not used (they would not be a valid explanation here).

**What this is.** Counterfactual evidence about the model's output on 300 pre-registered validation transactions: how much does the model's score for a transaction depend on its link to each of its entities, and on the entity's memory of earlier days. It describes the model; it is not evidence of fraud, causation or coordination, and the model was tuned on validation, so the evidence is descriptive, not held-out. Nothing here uses a label.

## 1. Inputs (the only files read)

* The 300 ids of `experiments/phase11_explainable_evidence/xgb_explanations_val/selection_val.csv` (hash recorded in Phase 11: `623d1ee2...be963`).
* The checkpoint above, and the Phase 4 graph through the project's existing loader with the dataset truncated at day 151 and built without labels (`with_labels=False`); rows after day 151 are not used.
* The recorded predictions `experiments/temporal_graphsage/seed_42/predictions_val.parquet` (`TransactionID`, `y_score` only) for the reproduction check.
No held-out file is opened; the held-out period is never scored.

## 2. Counterfactuals (fixed now)

The model streams days in order with its entity memory. For a selected transaction `i` on day `d` and each relation `r` for which `i` has an edge:
* **Link removal.** The state at the start of day `d` is cloned. In day `d`'s snapshot, the edge of `i` for relation `r` is deleted; if that entity then has no remaining edge that day it is removed from the day's entity list (so it is not updated). Day `d` is re-scored on the clone and the change in the transaction's logit is recorded: `removal_effect_r = logit_perturbed - logit_base`.
* **Memory reset.** The state is cloned and the entity's memory for relation `r` is set to the "never seen" condition (the model then uses its learned initial state and the new-entity flag). `reset_effect_r = logit_perturbed - logit_base`. Entities that were already never seen on day `d` are marked `entity_new = 1`; for them the reset changes nothing by construction.
All other transactions, edges and earlier days are unchanged; the real stream is never modified.

## 3. Acceptance criteria (fixed now)

* **NE1 reproduction.** The unperturbed scores for the 300 transactions equal the recorded predictions to within 1e-4 on the probability.
* **NE2 selection.** Exactly the 300 Phase 11 ids (10 per day, days 122-151); the file's hash equals the recorded Phase 11 hash.
* **NE3 label-free.** The dataset is built without labels; the script has no label argument; no label column is read (static scan); no held-out day is used.
* **NE4 counterfactual validity.** Unit tests on a small synthetic dataset and model: link removal deletes exactly one edge, drops entities left without edges, leaves every other transaction's edges identical, never mutates the dataset or the live state.
* **NE5 sanity.** (a) For rows with `entity_new = 1` the memory-reset effect is exactly zero; (b) for rows whose entity was seen before, the reset changes the score in a majority of cases (`|effect| > 1e-6`); (c) link removal changes the score (`|effect| > 1e-6`) in a majority of all performed removals. A failure is reported, not repaired.
* **NE6 reporting.** Per transaction: base score, presence of each relation, the removal and reset effects, whether the entity was new, and the relation with the largest absolute removal effect; one summary by relation (mean absolute effects, share of transactions in which the relation is the largest). Descriptive only.
* **NE7 hygiene.** Outputs have SHA-256 sidecars and a manifest with input hashes; wording lint clean; a second run reproduces the outputs to within 1e-5 on the effects (GPU kernels are not bit-reproducible); nothing outside `experiments/neural_graph_evidence/` is written.

## 4. Limits stated in advance

One checkpoint, 300 transactions, validation period, VAL-tuned model; a counterfactual on one transaction's link is not a statement about the entity or about fraud; same-day transactions that share an entity influence each other, so removal effects include interaction with them; attribution across several relations is not additive.
