# Graph evidence for the graph-feature model B: pre-registration (explainability addendum)

Dated 2026-10-20, written and hashed (`GRAPH_EVIDENCE_PREREG.sha256`) before the script that produces the evidence exists. Authority: the master instruction of 2026-10-20 and the original brief's EXPLAINABILITY section ("for graph models, provide interpretable evidence such as important neighbouring entities, influential relationships ... neighbourhood perturbation"). Phase 11 deferred graph evidence; this addendum provides the part that is feasible without scoring a neural checkpoint.

**What this is and is not.** It explains the *output of the graph-feature XGBoost model* `B_full` on the same 300 VAL transactions that Phase 11 explained for model A. It describes what the model does; it is not evidence of fraud, causation or coordination. It does **not** explain the neural graph models (GraphSAGE, GAT, Temporal GraphSAGE): producing neural evidence would require scoring their checkpoints on real data, which is not done here and remains an unexplained part of the brief's requirement. Attention weights are not used.

## 1. Inputs (the only files read)

* The 300 transaction ids in `experiments/phase11_explainable_evidence/xgb_explanations_val/selection_val.csv` (hash recorded in Phase 11's manifest: `623d1ee2...be963`).
* `experiments/comparison_B_graph_features/outputs/arms/B_full/model.joblib` and `feature_columns.json`.
* `data/features/train_features.parquet` and `val_features.parquet` with the label column projected away (the builder is refit on TRAIN only, as in comparison A) and `experiments/phase9_coordinated_fraud/cache/real_le151.npz` (entity ids, days <= 151). No label is read at any point. No TEST-named file is opened.

## 2. Method (fixed now)

* **Per-transaction contributions:** the booster's native `pred_contribs` (Tree SHAP in log-odds space) for the 510 columns plus the bias, using the same function as Phase 11 (`src/explain/xgb_contrib.contributions`).
* **Relationship influence:** the 40 graph features are assigned to relations: a feature belongs to relation `r` if its name starts with `g_{r}_` (the entity's own features). Distinct-count features `g_{s}_prior_distinct_{r}` belong to relation `s` (the entity whose neighbourhood they summarise). The influence of relation `r` on a transaction is the sum of the contributions of its features (signed) and its absolute value (ranking). The summed contribution of all tabular columns is reported alongside.
* **Neighbourhood perturbation (feature-level counterfactual):** for each relation present on the transaction (`g_{r}_has_edge = 1`), set `g_{r}_has_edge` to 0 and every feature of relation `r` to NaN, leave everything else unchanged, and re-score. The change in margin is the perturbation effect. This is a counterfactual on the model's inputs, not a surgery on the underlying graph.
* **Selection of relations to compare:** per transaction, the top-ranked relation by absolute summed contribution among its present relations, and one relation drawn uniformly at random (`numpy.random.default_rng(SeedSequence([40042, TransactionID]))`) among its present relations.

## 3. Acceptance criteria (fixed now)

* **GE1 additivity:** contributions plus bias equal the margin to within 1e-4 for all 300 rows.
* **GE2 selection:** exactly the 300 Phase 11 ids, 10 per day, days 122-151; the selection file's hash equals Phase 11's recorded hash.
* **GE3 label-free:** no label column is loaded; a static scan finds no label read in the script; the model and column list are hash-recorded in the output manifest.
* **GE4 perturbation sanity (descriptive consistency check):** over the transactions with at least two present relations, the mean absolute margin change from removing the top-ranked relation is at least the mean absolute margin change from removing the randomly drawn relation. A failure is reported, not repaired.
* **GE5 reporting:** outputs give, per transaction, the per-relation summed contributions, the perturbation effects, and the ten largest-magnitude columns; and one summary table (mean absolute relation influence by relation, share of transactions where a graph feature is among the top ten). Descriptive only: no inference about fraud, no ranking of the neural models.
* **GE6 hygiene:** tests on synthetic models, determinism (byte-identical re-run apart from the manifest timestamp), SHA-256 sidecars, Phase 9 wording lint, no file outside `experiments/comparison_B_graph_features/graph_evidence/` written.

## 4. Limits stated in advance

* The model was tuned on VAL (early stopping, threshold), so these explanations are descriptive, not held-out.
* Tree SHAP shares credit among correlated features (for example `g_pa_prior_n` and `g_card_prior_n`); relation-level sums inherit that. The perturbation sets features to missing, which XGBoost routes by a learned default direction that is not a faithful "absence" of the relation.
* 300 pre-registered transactions are not representative of VAL as a whole.
