# Phase 3 — Classical ML Baseline Report

**Scope:** Logistic Regression, Random Forest, XGBoost, and the XGBoost historical/blocklist control. No graph construction, no GNN, no p5_scale.py, no graph dependencies installed. All four models trained on the frozen chronological split (TRAIN days 1–121 / VAL 122–151 / TEST 152–182) and evaluated on VALIDATION only — TEST was never loaded by any training script.

---

## A. Feature representations

Two feature builders were used, deliberately different, because linear and tree models need different things — both documented in code ([lr_features.py](../src/models/lr_features.py), [tree_features.py](../src/models/tree_features.py)).

| Column class | Logistic Regression (527 features) | Random Forest / XGBoost (470 features) |
|---|---|---|
| Continuous (`C*`,`D*`,`V*`,`dist1/2`,numeric `id_*`) | median-imputed → standard-scaled | raw; **RF**: median-imputed (can't accept NaN) — **XGBoost**: real NaN kept, native learned-split handling |
| `card1/2/3/5`, `addr1/2` | frequency-encoded (anonymized tokens; raw magnitude is meaningless to a linear model) | **kept raw** — trees split on thresholds, not distances, so raw magnitude is standard/defensible practice here |
| `TransactionAmt` | `log1p` (heavy right skew) | raw (trees are monotonic-transform invariant) |
| `hour`, `day_of_week` | sin/cos (cyclical; raw ints imply false ordinal distance for a linear model) | raw integers |
| Low-cardinality categoricals (25 cols, ≤5 levels) | one-hot, explicit "missing" level | ordinal-encoded, explicit "missing" **and** "unseen-in-TRAIN" as two distinct reserved codes (avoids ~65 sparse dummy columns trees don't need) |
| High-cardinality strings (`id_30/31/33`, `DeviceInfo`, up to 1,551 levels) | frequency-encoded | frequency-encoded (same encoder/columns as LR) |
| Boolean flags (27) | used directly as 0/1 | used directly as 0/1 |

**Baseline 4 (control)** uses neither builder — it is a standalone 12-feature set built from [historical.py](../src/features/historical.py): for both `card1` and the Phase-1 pseudo-account (`card1+addr1+(day−D1)`), `prior_txn_count`, `prior_fraud_count`, `prior_fraud_flag`, `has_prior_history`, `prior_fraud_rate`, `days_since_prev_txn` — all strictly `information_time < transaction_time`. No `TransactionAmt`, no `V/C/D` columns, nothing else.

**Not one-hot encoded anywhere:** `card1` (13,553 levels), `addr1`, `DeviceInfo` (1,551), `id_30/31/33` — per your explicit instruction, these are frequency-encoded (LR) or kept raw (trees), never expanded into dummy columns.

**Bug fixed en route:** Phase 2's `FrequencyEncoder` stringified dict keys only implicitly via JSON round-trip, which silently breaks for numeric-keyed columns (`card1`, etc.) — only ever exercised on string columns before. Fixed to stringify consistently in `fit`/`transform`; Phase 2's full regression suite re-verified unaffected.

---

## B. Training configurations

| | Logistic Regression | Random Forest | XGBoost | XGBoost (historical control) |
|---|---|---|---|---|
| Imbalance handling | `class_weight='balanced'` | `class_weight='balanced_subsample'` | `scale_pos_weight=27.36` (TRAIN neg:pos) | `scale_pos_weight=27.36` |
| Key hyperparameters | `solver=lbfgs`, `max_iter=2000` | `n_estimators=200`, `max_depth=16`, `min_samples_leaf=5` | `n_estimators=1000`, `max_depth=6`, `lr=0.05`, `tree_method=hist`, early stopping (50 rounds) on VAL | **identical** to full XGBoost — deliberately, so any gap is attributable to the feature set alone |
| SMOTE | not used | not used | not used | not used |
| Threshold selection | best-F1 on VAL | best-F1 on VAL | best-F1 on VAL | best-F1 on VAL |
| `random_state` | 42 | 42 | 42 | 42 |

Accuracy was not optimized or reported as a headline metric anywhere in this phase.

---

## C. Runtime / resource usage

Feature count, memory, and expected cost were reported before each training run, per your instruction; none required a manual pause under the current (auto-execution) policy — all were CPU-bound, dense-matrix operations well below GPU-scale.

| | Feature build | Model fit | Total | Peak matrix memory |
|---|---|---|---|---|
| Logistic Regression | 13.1s | 95.6s (641 lbfgs iters) | 109.4s | 880 MB (TRAIN) |
| Random Forest | 9.6s | 48.0s | 59.0s | 785 MB (TRAIN) |
| XGBoost | 8.5s | 206.3s (best_iter=798/1000) | 216.0s | 785 MB (TRAIN, 37% NaN, left native) |
| XGBoost (control) | negligible | 1.8s (best_iter=95/1000) | 2.6s | 20 MB (TRAIN) |

---

## D. Validation metrics

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Threshold |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.4126 | 0.8520 | 0.4746 | 0.3837 | 0.4243 | 0.8724 |
| Random Forest | 0.4543 | 0.8792 | 0.4816 | 0.4254 | 0.4517 | 0.5851 |
| XGBoost | 0.5395 | 0.9050 | 0.6173 | 0.4643 | 0.5300 | 0.7747 |
| **XGBoost (historical control)** | **0.6771** | **0.9432** | 0.7777 | 0.5492 | 0.6437 | 0.9141 |

Confusion matrices (VAL, n=83,655, 2,828 fraud), at each model's own best-F1 threshold:

| Model | TP | FP | TN | FN | Alerts |
|---|---|---|---|---|---|
| Logistic Regression | 1,085 | 1,201 | 79,626 | 1,743 | 2,286 |
| Random Forest | 1,203 | 1,295 | 79,532 | 1,625 | 2,498 |
| XGBoost | 1,313 | 814 | 80,013 | 1,515 | 2,127 |
| XGBoost (historical control) | 1,553 | 444 | 80,383 | 1,275 | 1,997 |

The historical control has the *best* pooled numbers on every metric — see sections J/K for why this is not the straightforward good news it looks like.

---

## E. Threshold analysis

All four thresholds were chosen by maximizing F1 on VALIDATION only — TEST was never opened. Note the thresholds are not comparable in absolute terms across models (each model's score distribution differs); what matters is that each was picked independently, on the same data.

---

## F. Top-K results

| K | Alerts | LR captured / R@K | RF captured / R@K | XGB captured / R@K | Control captured / R@K |
|---|---|---|---|---|---|
| 0.1% | 84 | 84 / 0.0297 | 84 / 0.0297 | 84 / 0.0297 | 84 / 0.0297 |
| 0.5% | 418 | 366 / 0.1294 | 391 / 0.1383 | 391 / 0.1383 | 417 / 0.1475 |
| 1% | 837 | 631 / 0.2231 | 686 / 0.2426 | 748 / 0.2645 | 831 / 0.2938 |
| 5% | 4,183 | 1,410 / 0.4986 | 1,452 / 0.5134 | 1,649 / 0.5831 | 1,936 / 0.6846 |

At every K, all four models capture identical fraud at the top 0.1% (84/84 — the most obvious cases are obvious to everyone). The gap opens up as K grows, and the control leads throughout — again, driven by the effect quantified in K below.

---

## G. Leakage checks

- **Blocklist enforcement**: `assert_features_exclude_blocklist` run programmatically against all four models' final feature lists (527 / 470 / 470 / 12) — passed in every case; `TransactionID`, `TransactionDT`, `isFraud` confirmed absent.
- **Fit-on-TRAIN-only**: all statistical encoders (`FrequencyEncoder`, `MedianImputer`, `StandardScaler`, ordinal category maps) fit exclusively on TRAIN, applied unmodified to VAL — inherited from and consistent with Phase 2's validated discipline.
- **TEST isolation**: none of the four training scripts reference `test_features.parquet` or the TEST split anywhere; confirmed by code inspection (no such path string appears in any of them).
- **Historical control — independently verified, not just argued**: `run_leakage_self_check` re-derives `prior_txn_count` and `prior_fraud_count` for 30 sampled rows per entity using only rows strictly earlier in a tie-broken chronological order, and compares against the vectorized computation. **Passed.** (One real subtlety handled: ~17,191 rows share an exact `TransactionDT` with another row per Phase 1; a naive strict-inequality check would misjudge ties, so the check uses the same stable-sort tie-breaking the actual feature computation uses.)
- **Explicit design note on the historical control's timeline**: its features are computed over the *full* chronological labeled dataset (all three splits concatenated in time order), not fit-restricted to TRAIN. This is intentional and different from the statistical encoders above — see the [historical.py](../src/features/historical.py) module docstring for the full reasoning: an entity's transaction history accumulates over real elapsed time regardless of which artificial split a later transaction falls into, and restricting it to "history within TRAIN only" would produce a weaker, less realistic feature and undermine the entire point of this control (reproducing, as a classical feature, the exact account-level effect a graph model would also have access to). `information_time < transaction_time` is still strictly enforced — the self-check confirms this — it is just not additionally restricted to `information_time` being before the VAL/TEST boundary.

---

## H. Artifacts created

```
experiments/
├── logistic_regression/      {model.joblib, config.json, metrics_val.json, predictions_val.parquet,
│                               coefficients.json, metadata.json, feature_builder/}
├── random_forest/            {..., feature_importances.json, feature_builder/}
├── xgboost/                  {..., feature_importances.json, feature_builder/}
└── xgboost_historical_control/  {model.joblib, config.json, metrics_val.json, predictions_val.parquet,
                                   feature_importances.json, metadata.json}
data/features/historical_features.parquet   # reusable 12-column entity-history feature set
reports/phase3_stratified_metrics.json      # cold-start vs has-history breakdown, all 4 models
```
Every `metrics_val.json` shares one schema ([metrics.py](../src/models/metrics.py)) — the comparison table above was built directly from these files, not re-derived.

---

## I. Comparison table

*(see D and F above — repeated here per your requested structure)*

| Model | PR-AUC | ROC-AUC | F1 | Alerts | TP | FP |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.4126 | 0.8520 | 0.4243 | 2,286 | 1,085 | 1,201 |
| Random Forest | 0.4543 | 0.8792 | 0.4517 | 2,498 | 1,203 | 1,295 |
| XGBoost | 0.5395 | 0.9050 | 0.5300 | 2,127 | 1,313 | 814 |
| XGBoost (historical control) | 0.6771 | 0.9432 | 0.6437 | 1,997 | 1,553 | 444 |

---

## J. Interpretation

**The ranking is not what it looks like pooled.** Feature importances make the mechanism explicit: for the historical control, `account_prior_fraud_rate` alone accounts for **78.9%** of total feature importance. This is the pseudo-account's own historical fraud rate — a near-direct echo of Vesta's account-level labeling convention (Phase 1, Section 10), not a discovery about fraud *behavior*.

Stratifying every model's VAL predictions by `account_has_prior_history` (30.3% cold-start, 69.7% has-history) tells the real story:

| Model | Pooled PR-AUC | Cold-start PR-AUC (n=25,374, 3.04% fraud) | Has-history PR-AUC (n=58,281, 3.53% fraud) |
|---|---|---|---|
| Logistic Regression | 0.4126 | 0.4662 | 0.3936 |
| Random Forest | 0.4543 | 0.5209 | 0.4287 |
| XGBoost | 0.5395 | **0.5592** | 0.5357 |
| XGBoost (historical control) | 0.6771 | **0.1018** | **0.8442** |

The historical control's commanding pooled lead **inverts** on cold-start: **0.1018 is the worst score of any model there** — barely above what the 3.04% base rate alone would suggest, since an account with no history *by definition* carries none of this control's signal. Its extraordinary 0.8442 on has-history transactions is doing all the work of its pooled 0.6771. Full-feature XGBoost, by contrast, holds up on both segments (0.5592 cold-start, 0.5357 has-history) — it has learned something beyond account identity.

**This is a direct, empirical confirmation of the concern Phase 1 raised on paper.** A model — or, prospectively, a graph architecture — that scores well pooled on this dataset may be doing nothing more than recovering which accounts Vesta already knew were compromised. The cold-start segment is the only part of this evaluation where that explanation is unavailable, and it is exactly where a genuinely new signal (behavioral, or eventually structural/graph) would have to prove itself.

**On the discrepancy with Phase 1's exact reconnaissance numbers:** Phase 1's EDA reported 59.86%/0.08% fraud rates for TEST-period transactions with/without prior fraud, using only TRAIN+VAL as the history source. This report's `run_xgboost_historical_control` implementation instead accumulates history continuously across the full chronological timeline (including earlier-in-VAL transactions), which is the more realistic, production-faithful choice and is what a deployed system — or a graph model — would actually have. It gives 45.56%/0.50% on the equivalent TEST-period check (a smaller but still ~91× gap) — the qualitative finding is unchanged; the exact figures differ because the two calculations answer subtly different questions, both legitimately.

The C-column and V-column feature-importance overlap between Random Forest and XGBoost (`C13`, `C14`, `V258`, `V91`, `V70`, `C8` appear near the top of both) is a reassuring cross-model consistency check — two different tree algorithms converge on similar signal, for whatever these anonymized columns represent (Phase 1's stated limitation: their real-world meaning is unknown).

---

## K. Does the historical/blocklist control change the apparent value of graph information?

**Yes — substantially, and in a specific, actionable way.** It does not mean graph information is worthless; it means the *bar* for demonstrating real graph value is much higher, and pooled metrics cannot clear it.

1. **A future graph model's pooled PR-AUC must beat 0.6771 to look novel at all** — and per this baseline's own decomposition, most of that number is available to *any* model that can compute "has this account defrauded before," no graph required. A graph model that lands at, say, 0.70 pooled has very possibly added little beyond what a two-line SQL query already captures.
2. **The meaningful test is the cold-start segment.** The historical control scores 0.1018 there — it has no other option, by construction. Full-tabular XGBoost already reaches 0.5592. For H1 (graph structure beyond transaction-level features) and H5 (coordinated-community discovery) to be genuinely supported, a graph model needs to beat **0.5592 on cold-start specifically**, not merely beat 0.6771 pooled. Beating the pooled number while trailing XGBoost on cold-start would mean the graph model is a more expensive way to re-derive the same account-history signal.
3. **This reframes exactly what a graph model could add on cold-start**: a first-seen account has no history of its own, but it may share a device, network, or billing address with an *other* account that does have history. That shared-entity path is the only mechanism by which graph structure could plausibly help precisely where this control cannot — which is a sharper, more falsifiable version of H1 than "graph beats tabular" pooled.
4. **Recommendation for every future model report, starting immediately**: report PR-AUC/ROC-AUC pooled **and** split by cold-start/has-history, always alongside this control's numbers as the reference line. A single pooled number is no longer sufficient evidence of anything on this dataset.

---

## L. Recommendation for Phase 4

Do not begin graph or GNN work yet, per your instruction — this stops here. When Phase 4 is approved, the concrete groundwork this phase provides:

- **Reference baselines are locked**: XGBoost full-feature (pooled 0.5395, cold-start 0.5592) is the tabular bar; the historical control (pooled 0.6771, cold-start 0.1018) is the "don't be fooled by this" reference.
- **The graph schema Phase 1 proposed should be built and evaluated with the cold-start/has-history split as a first-class output**, not an afterthought — this report shows pooled numbers alone would be actively misleading for judging H1.
- **The historical features built here** (`data/features/historical_features.parquet`) are a legitimate, reusable input — a future graph model may reasonably include them as node features (they are leakage-safe by the same construction), as long as ablations report performance with and without them, so cold-start gains can be attributed correctly.

**Status: Phase 3 complete. Raw data untouched (byte-identical throughout). No graph work, no GNN, no p5_scale.py, no graph dependencies installed. Awaiting your instruction before Phase 4.**
