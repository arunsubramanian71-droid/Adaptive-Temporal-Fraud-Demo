# Relationship-level ablation of the learned temporal graph model. Pre-registration

Dated 2026-10-22; written and hashed (`PREREGISTRATION_RELATION_ABLATION.sha256`) before any ablation arm is trained. Validation period (days 122-151) only; days 1-151 are loaded; no held-out day is read. Nothing in earlier phases is changed.

## 1. Why

The original brief, section ABLATION STUDIES: "Where computationally feasible, perform additional relationship-level ablations such as: without device relationships; without address relationships; without email relationships; without specific temporal features. The purpose is to determine WHICH information actually contributes."

What already exists:
* Remove-one-relation and time-window ablations of the handcrafted graph features: comparison B, XGBoost family.
* Temporal-memory ablations of the learned temporal model: Phase 7A, memory ablated and schedule-matched static.
* Billing address and email domain were measured in Phase 4 and left out of the graph schema, so "without address" and "without email" have no edges to remove.

**What is missing** is the relation ablation of a learned graph model. Temporal GraphSAGE trains in about 2 minutes per seed on this machine (Phase 7 `resource_measurements.json`), so the ablation is computationally feasible, and under the brief's wording it is therefore done.

## 2. Design (fixed now)

* **Model and training:** Temporal GraphSAGE exactly as Phase 7, via `scripts/train_temporal_graphsage.py`'s `run_seed`. Same hyperparameters (hidden size, dropout, learning rate, weight decay, maximum epochs, patience, TBPTT window) and the same class weighting. Train days 1-121, early stopping and threshold on validation days 122-151 exactly as Phase 7, seeds 42, 123, 2024.
* **Arms:** for each relation r in {charged_to (pseudo-account), used_card (card), via_network (network proxy), from_device (device)}, the relation's edges are removed on every day before training and scoring. The model code is unchanged; the entity type of r simply never receives a message or a memory update. That is 4 relations times 3 seeds, 12 runs.
* **Reference:** the recorded Phase 7 full model (`experiments/temporal_graphsage/seed_*/metrics_val.json`). As a run-to-run noise reference, the full model is retrained once with seed 42 by the same code; this run is reported and is not used by the rule.
* **Outputs:** `experiments/relation_ablation_temporal/drop_<r>/seed_<S>/`. Model files and row-level predictions stay local (publication policy).

## 3. Decision rule (fixed now; the same style as Phase 7A)

For each relation r, with d_r,S = (recorded full pooled validation PR-AUC) - (pooled validation PR-AUC without r), for seed S:

* **"r contributes"** if d_r,S > 0 in at least 2 of 3 seeds **and** the mean of d_r,S exceeds the larger of the two arms' across-seed SDs (ddof = 1).
* **"removing r does not hurt"** if the mean of d_r,S is at most 0. In that case the model does at least as well without r.
* **Otherwise** the result is "not established".

Cold-start, warm and no-edge segment PR-AUC are reported descriptively. No p-value is computed.

## 4. Limits stated in advance

* Three seeds sharing the same data are not independent replications.
* Validation was used for early stopping and thresholds, as in Phase 7, so these are validation-period results.
* GPU training is not bit-reproducible: Phase 8 measured window-mean differences of about 0.004 between identical runs, and the seed-42 rerun measures it for this model. Differences of that size cannot be read as effects.
* Removing a relation also removes that entity type's memory, which is the meaning of "without that relationship" for a memory model.
* One learned-graph family is ablated: the temporal model, the strongest graph model.
