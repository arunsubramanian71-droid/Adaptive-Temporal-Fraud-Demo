# Phase 5 — GraphSAGE Fraud-Detection Experiment

**Research question:** Does learned relational structure improve fraud detection beyond the classical tabular baselines, especially under future-time and cold-start evaluation? **Scope:** the official GraphSAGE experiment plus one clean non-relational ablation. No GAT, no temporal GNN, no adaptive updating, no ring discovery, no dashboard. Trained on TRAIN, evaluated on VALIDATION only — TEST was never loaded.

---

## A. GraphSAGE architecture

Per-type input projection (`nn.Linear`, since node types have different raw dims: transaction=470, every entity type=1) → 2 `HeteroConv`/`SAGEConv` layers (`aggr="sum"` across relations) → dropout → ReLU → a single linear head applied to the **transaction** embedding only (labels exist at the transaction level, Phase 4 requirement 6). Deliberately small, per the instruction not to reach for a large architecture first.

```
x_dict --[per-type Linear + ReLU]--> h
h --[HeteroConv(SAGEConv) x4 relations, sum]--> h --[Dropout(0.3) + ReLU]--> h
h --[HeteroConv(SAGEConv) x4 relations, sum]--> h --[ReLU]--> h
h['transaction'] --[Linear(64,1)]--> logit
```

## B. Parameter count

**162,817** parameters (measured, not estimated). The non-relational control (Section below) has 34,369 — comparable order of magnitude, not padded up to look more capable than it is.

## C. Feature representation

| | Dim | Source | Normalization |
|---|---|---|---|
| Transaction | 470 | Phase 3's `TreeFeatureBuilder`, `"rf"` variant (median-imputed, fit TRAIN-only) — **the same encoding Phase 4 baked into the graph**, not re-derived | `StandardScaler`, fit on TRAIN transaction rows only, applied to all 590,540 nodes |
| Every entity type (`pseudo_account`/`card`/`netproxy`/`device_fingerprint`) | 1 | constant placeholder (Phase 4 design — no precomputed entity statistic) | n/a |

**How heterogeneous node features are handled:** the per-type `nn.Linear` projection maps every node type into the same 64-dim hidden space before any message passing happens — this is the mechanical reason a heterogeneous graph can't be silently collapsed to a homogeneous one (Phase 1, Section 16).

**How labels are used:** `isFraud` is the sole training target (`y`), applied only to TRAIN-day nodes in the loss; it is not, and has never been, a node feature (`assert_features_exclude_blocklist` verified this on the 470-dim feature set at Phase 4 construction time).

**Critical label-leakage rule, honored:** the historical/blocklist features (`account_prior_fraud_rate`, etc.) from Phase 3's `historical.py` were **not** included anywhere in this experiment's transaction features — this is the clean graph-learning baseline, exactly as instructed. Combining them is deferred to an explicit, separate future experiment.

**Note on Phase 4's saved artifact:** `hetero_graph.pt` was loaded read-only and never overwritten — the `StandardScaler` is applied to an in-memory copy for this training run only.

## D. Training configuration

| | Value |
|---|---|
| Optimizer | Adam, lr=1e-3, weight_decay=1e-5 |
| Loss | `BCEWithLogitsLoss`, `pos_weight=27.36` (TRAIN neg:pos ratio) — **not accuracy, no SMOTE** |
| Max epochs / patience | 60 / 10 (early stopping on rolling VAL PR-AUC) |
| Random seed | 42 |
| Device | ROCm GPU (`cuda:0`) |

Training never triggered early stopping — VAL PR-AUC was still improving at epoch 60 (0.4337, best epoch=60/60). See Section O.

## E. Temporal evaluation methodology

**The requirement carried over from Phase 4, implemented exactly, not weakened:** a single per-split mask is not used anywhere in this evaluation.

- **Training:** one static graph view, masked to `edge_time <= 121` (TRAIN's last day), reused unchanged across all 60 epochs — legitimate, since fitting parameters on an already-past block doesn't need a rolling cutoff (the causal-ordering constraint applies to *evaluation*, simulating deployment, not to how a training corpus that's already entirely in the past is used internally).
- **Validation (and, in a future TEST run, the same protocol):** `src/graph/rolling_eval.py`'s `rolling_predict` scores each of the 30 VAL days **separately** — for day *t*, the graph is masked to `edge_time <= t` and only that day's own transaction predictions are kept. 30 independent forward passes, not one.
- **Exactness, not approximation:** at day granularity, `edge_time <= t` for scoring day-*t* transactions is a literal instance of `information_time <= transaction_time`, not a coarsening of it — the one accepted limitation is that two transactions on the *same* day may see each other's non-label structural contribution within that day's pass (documented in `rolling_eval.py`'s docstring; day granularity matches every other temporal decision in this project, including `TransactionDT`'s own established resolution, Phase 1 Section 9).
- **No later VAL day can influence an earlier one:** by construction — day-122 transactions are scored with a graph masked to `edge_time<=122`, which cannot contain an edge from day 130.

## F. Leakage checks

- `assert_features_exclude_blocklist` re-verified on the 470-dim feature set before training.
- Rolling protocol re-verified directly: predictions for day 122 come only from a `<=122`-masked graph; day 151's predictions come from a `<=151`-masked graph — 30 distinct masked graphs were actually constructed and used, not one.
- The historical/blocklist features were confirmed absent from the training input by construction (Section C).
- TEST was never loaded by either script in this phase (`test_features.parquet` / TEST day range do not appear).

## G. Training runtime

148.8s wall time, 60 epochs, ~2.44s/epoch (1 train step + 30 rolling VAL passes). Non-relational MLP control: 3.5s, 60 epochs, ~0.04s/epoch (no rolling needed — an MLP has no graph dependency).

## H. GPU memory

**4.45 GiB peak reserved** (measured via `torch.cuda.max_memory_reserved()`), matching Phase 4's feasibility-benchmark prediction closely. Comfortably within the 15.98 GiB budget; no memory pressure observed across 60 epochs of repeated masking + forward/backward passes.

## I. Validation metrics (pooled)

| Metric | GraphSAGE |
|---|---|
| PR-AUC | **0.4337** |
| ROC-AUC | 0.8631 |
| Precision / Recall / F1 (@ threshold 0.884) | 0.497 / 0.407 / 0.447 |
| Alerts | 2,105 (of 83,655) |

## J. Cold-start vs. warm-start metrics

**A real methodological catch made and fixed during this phase, reported in full rather than smoothed over:** the first stratification attempt used a 2-way warm/cold split and got a startling "cold-start PR-AUC beats XGBoost" result (0.5983). Investigating before trusting it revealed the bug — that bucket silently merged genuine cold-start with a *different* population entirely: transactions with **no `pseudo_account` edge at all** (missing `addr1`/`D1`, 11.1% of rows, strongly correlated with `ProductCD='C'`, which Phase 1 already showed has an 11.7% base fraud rate on its own). That no-edge group has a **12.5% fraud rate** — nearly 6× the other two segments — and was responsible for 1,044 of the flawed bucket's 1,522 "cold-start" frauds. The corrected, honest 3-way split:

| Segment | n | Fraud rate | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| `has_edge_warm` (true warm-start) | 52,276 | 2.50% | **0.1948** | 0.8311 |
| `has_edge_cold` (true cold-start) | 23,024 | 2.08% | **0.3249** | 0.8169 |
| `no_pseudo_account_edge` (missing addr1/D1, mostly ProductCD='C') | 8,355 | 12.50% | 0.7118 | 0.8934 |

**This is the number that matters for the research question, and it's a plain "not yet":** GraphSAGE's *true* cold-start PR-AUC (0.3249) is well below XGBoost's (0.5592), and warm-start (0.1948) is far below both XGBoost (0.5357) and the historical control (0.8442) on the equivalent segment.

## K. Top-K results

| K | Alerts | Fraud captured | P@K | R@K |
|---|---|---|---|---|
| 0.1% | 84 | 83 | 0.988 | 0.029 |
| 0.5% | 418 | 371 | 0.888 | 0.131 |
| 1% | 837 | 652 | 0.779 | 0.231 |
| 5% | 4,183 | 1,418 | 0.339 | 0.501 |

Nearly identical to Random Forest's Top-K profile (Phase 3) at every K — consistent with GraphSAGE landing between LR and RF on pooled PR-AUC too.

## L. Comparison with all Phase 3 baselines

| Model | Pooled PR-AUC | Pooled ROC-AUC | Cold-start PR-AUC | Warm-start PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.4126 | 0.8520 | 0.4662 | 0.3936 |
| Random Forest | 0.4543 | 0.8792 | 0.5209 | 0.4287 |
| XGBoost | 0.5395 | 0.9050 | **0.5592** | 0.5357 |
| XGBoost (historical control) | 0.6771 | 0.9432 | 0.1018 | **0.8442** |
| **GraphSAGE (this phase)** | 0.4337 | 0.8631 | 0.3249 | 0.1948 |
| GraphSAGE non-relational control (MLP, same features/capacity) | 0.3734 | 0.8428 | 0.2588 | 0.1415 |

*(Phase 3's cold-start/warm-start figures use a 2-way `account_has_prior_history` split with a `.fillna("NA")` convention that behaves differently from this phase's stricter 3-way split — flagged in Phase 4's report, Section M. GraphSAGE's `has_edge_cold`/`has_edge_warm` are the closest correct analogs but are not numerically identical populations to Phase 3's; the qualitative comparison — GraphSAGE trails XGBoost on both segments — holds regardless of that definitional difference.)*

No Phase 3 artifact was overwritten — every path above still resolves to its original Phase 3 file, re-verified before writing this report.

## M. Does GraphSAGE add signal beyond XGBoost?

**No, not in this configuration.** Pooled PR-AUC (0.4337) sits between Logistic Regression and Random Forest, well below XGBoost (0.5395) and far below the historical control (0.6771, with the caveat that most of that number is itself an artifact — Phase 3). GraphSAGE ranks last of the five *pooled* comparisons except the MLP control itself.

## N. Does GraphSAGE add signal specifically on cold-start transactions?

**Relative to XGBoost: no** (0.3249 vs. 0.5592). **Relative to a no-graph model with identical features and capacity: yes, clearly** — this is the finding worth taking seriously. Holding feature representation, architecture width/depth, dropout, loss, and training recipe fixed, adding the graph raised PR-AUC on every single segment:

| Segment | GraphSAGE | MLP control (no graph) | Δ |
|---|---|---|---|
| Pooled | 0.4337 | 0.3734 | **+0.0603** |
| `has_edge_warm` | 0.1948 | 0.1415 | +0.0533 |
| `has_edge_cold` | 0.3249 | 0.2588 | +0.0661 |
| `no_pseudo_account_edge` | 0.7118 | 0.6552 | +0.0566 |

This is a real, controlled, positive result for H1 in a narrow sense: **relational structure contributes something a plain neural network on the same inputs cannot get on its own.** It does not yet mean graph structure beats a mature, well-established tabular method (XGBoost) — those are different, both true, findings, and conflating them would misrepresent this experiment.

## O. Failure modes / limitations

1. **Training had not converged at the epoch budget.** VAL PR-AUC was still rising at epoch 60/60 for both GraphSAGE and its MLP control — neither model triggered early stopping. The comparison between them is likely still valid (same budget applied equally), but the absolute gap to XGBoost may narrow with more training. Recommended next step, not redone here to avoid moving the goalposts after seeing a result.
2. **The cold-start stratification bug (Section J) is the most important thing to carry forward.** Any future stratified reporting on this dataset must separate "no entity mapping due to missing source data" from "genuinely new entity" — they are different populations with very different base rates, and conflating them silently overturns the conclusion.
3. **The dense-MLP/GNN architecture family may simply be a worse fit for this feature set than gradient-boosted trees** — a well-known general pattern for tabular data with many sparse/skewed columns (Phase 1's `V*` block, in particular). This is a plausible, non-graph-related explanation for part of the gap to XGBoost, not something this experiment can rule out on its own.
4. **Entity nodes carry no learned identity beyond message passing** (Phase 4's deliberate choice) — this is a legitimate leakage-avoidance decision, but it does mean the model has no way to represent "this specific card is unusual" independent of its neighboring transactions' features, which may particularly limit `card`'s contribution (100% coverage, but a very high-degree, low-per-node-information node type).
5. **The non-relational control's design choice is disclosed, not hidden:** a separate equal-capacity MLP was used instead of zeroing edges inside the same `HeteroConv` architecture, because the latter leaves every entity node with zero neighbors — a degenerate case for `SAGEConv`'s aggregation term, not a clean "no graph" condition. This is the methodologically correct choice, but it means the two models are not literally the same network with one switch flipped.

## P. Artifacts created

```
experiments/graphsage/
├── model_checkpoint.pt          # best-epoch state_dict
├── feature_scaler.joblib        # StandardScaler, fit TRAIN-only (reused by the MLP control)
├── config.json                  # architecture, training config, split boundaries
├── metrics_val.json             # pooled + 3-way segment metrics, canonical comparison format
├── training_history.json        # per-epoch loss / rolling VAL PR-AUC / epoch time
├── predictions_val.parquet      # TransactionID, isFraud, y_score, day, y_pred, segment
├── resource_measurements.json   # parameter count, GPU memory, epoch timings
└── metadata.json                # versions, timestamps, runtime
experiments/graphsage_control_mlp/   # same file set, for the non-relational ablation
```
No Phase 3 or Phase 4 artifact was modified — every write above targets a new experiment directory.

## Q. Recommendation for the next experiment

1. **Train longer before drawing final conclusions about the ceiling** — neither model had converged at 60 epochs. A follow-up run with a larger epoch budget (and possibly a learning-rate schedule) is the most likely single change to narrow the gap to XGBoost, and should be done before concluding graph learning "doesn't work" on this dataset.
2. **The clean H1 test this phase actually answers:** relational structure helps relative to no structure (Section N). The open question for Phase 6 is whether a *stronger* graph architecture (GAT's attention, or combining graph output with XGBoost/historical features in an ensemble) can close the remaining gap to XGBoost, particularly on cold-start.
3. **Fix the entity-key convention inconsistency flagged in Phase 4** (Section M there) before any experiment that combines the graph's `pseudo_account` with Phase 3's `historical.py` features directly — they currently refer to very slightly different populations.
4. **Always use the corrected 3-way stratification** (`src/graph/schema.py::compute_pseudo_account_segments`) for any future graph-model evaluation on this dataset — never the 2-way split this phase started with and caught.

**Status: Phase 5 complete.** GraphSAGE trained and evaluated with a genuine rolling per-day temporal protocol (not weakened), a clean non-relational ablation run, one real stratification bug caught and fixed before reporting, raw data and all prior-phase artifacts untouched. No GAT, no temporal GNN, no adaptive updating, no ring discovery, no dashboard, no Phase 6 work. Stopping here.
