# Phase 12 — Consolidated validation-evidence table: scope and acceptance record (first step)

Dated 2026-10-18. Written BEFORE any Phase 12 code, table or output exists. Its SHA-256 is in `SCOPE_AND_ACCEPTANCE.sha256`.

Standing statement for every Phase 12 output: **this is a descriptive table of recorded validation-period (VAL, days 122-151) predictions.** Every model in it was tuned on VAL (early stopping, epoch selection or thresholds) except where stated for F. No held-out (TEST) comparison exists, TEST is sealed, and nothing here supports a claim of generalization, statistical significance, causal attribution, fraud or coordination.

## 1. Authority and boundaries

* Authority: the user's instruction of 2026-10-18 authorizing only this scope record and its sidecar. The evidence table is **not built** until this record is complete and reviewed.
* Frozen and preserved: Phases 1-11, all experiment artifacts, all Phase 11 outputs. No existing file is modified.
* This step and the later build do **not**: train or score any model, create new predictions, open any checkpoint (`.pt`, `.joblib`), access TEST contents, run Phase 10 empirical tiers, start any ablation, use Git, or install anything.

## 2. Requirements (documented or instructed; not invented here)

| ID | Requirement | Source |
|---|---|---|
| R1 | The final comparison set is A tabular only, B tabular + graph features, C GraphSAGE, D GAT, E graph + temporal information, F full adaptive system; relationship-level ablations (device, address, email, specific temporal features) "where computationally feasible" | Original project brief, ABLATION STUDIES (first message of the earliest saved session; not a repo file) |
| R2 | Metrics to report: PR-AUC (primary), ROC-AUC, precision, recall, F1, Precision@K, Recall@K, confusion matrix at selected thresholds, false-positive count, number of alerts, fraud captured in top-K | Original brief, EVALUATION |
| R3 | Chronological evaluation; TRAIN 1-121, VAL 122-151, TEST 152-182 sealed; hyperparameters and preprocessing chosen without TEST | Original brief, EVALUATION; `src/config.py` |
| R4 | Report pooled results and the cold-start / warm-start split, always next to the historical-control reference line | `reports/phase3_baseline_report.md` section K item 4; `reports/phase4_graph_report.md` section N item 2 |
| R5 | The historical-feature models are a reference, not a graph-feature model; graph-model gains must be read against the cold-start segment | `reports/phase3_baseline_report.md` sections J-K |
| R6 | No Phase 6C number (`experiments/gat_300epoch/`) may be used in any comparison table or report | `reports/phase6c_gat_convergence_status.md` section 3 |
| R7 | The temporal-versus-static gain is confounded by training regime and is decomposed only by the Phase 7A chain; Phase 8 uses a different protocol (checkpoint trained on days 1-111, TRAIN-only threshold, fixed-cadence updates, metric = mean of five window PR-AUCs) | `reports/phase7_temporal_gnn_report.md` sections Headline, W; `reports/phase7a_temporal_ablation_report.md` section N; `reports/phase8_concept_drift_report.md` sections B, C, E, H |
| R8 | Standing rules: no label is used to generate, select, rank or tune anything; strict temporal causality; never invent results; no new dependency without reporting the need | project rules |
| U1 | Use existing stored validation predictions only, with one consistent metric implementation | user, 2026-10-18 |
| U2 | Include A, C, D and E; include F only with its distinct evaluation basis stated explicitly | user |
| U3 | B is missing: do not fabricate it and do not substitute the historical-feature baseline | user |
| U4 | Mark each comparison comparable or not comparable and explain why | user |
| U5 | Record source paths, model/config identity, data window, metric basis, seeds where applicable and source hashes | user |
| U6 | Preserve frozen artifacts and Phase 11 outputs; no training, scoring, new predictions, TEST access, Phase 10 empirical tier, ablation or Git | user |

## 3. Declared source set (fixed now)

All paths are under `experiments/`. Each model run directory contributes `predictions_val.parquet`, `metrics_val.json` and `config.json` (the four tabular directories also `metadata.json`). Only these files may be read.

| Cell | Model and run identity | Runs and seeds | Fit window and selection | Files |
|---|---|---|---|---|
| **A** | `logistic_regression` (527 features), `random_forest` and `xgboost` (470 features; XGBoost keeps NaN native, best iteration 798) | one run each; seed in each `config.json` | fit on TRAIN 1-121; XGBoost early stopping on VAL; thresholds best-F1 on VAL | 12 |
| A reference | `xgboost_historical_control` (12 account-history features) | one run | as above | 4 |
| **C** | HeteroSAGE, hidden 64, 2 layers, 162,817 parameters, `graphsage_robust/seed_{42,123,2024}` plus `graphsage_robust/summary.json` | 3 seeds, max 150 epochs, patience 20 | fit on TRAIN 1-121; epoch and threshold selection on VAL (as recorded) | 10 |
| C control | `graphsage_robust/mlp_control_extended` (same features, no relations, 34,369 parameters) | one run, seed 42; reached its 150-epoch cap without early stopping | as above | 3 |
| **D** | GAT, 4 heads, 364,545 parameters, `gat/seed_{42,123,2024}` plus `gat/summary.json` | 3 seeds, max 150 epochs; 0 of 3 early-stopped | as above | 10 |
| **E** | Temporal GraphSAGE (recurrent entity memory), 197,889 parameters, `temporal_graphsage/seed_{42,123,2024}` plus `summary.json` | 3 seeds; 18 windowed steps per epoch | as above | 10 |
| E controls (Phase 7A) | `temporal_ablation/static_schedule_matched` and `temporal_ablation/memory_ablated_temporal`, 3 seeds each, plus each `summary.json` | 3 seeds | as above | 20 |
| **F** | the E architecture; `phase8_adaptive_drift/{frozen,adaptive,replay_control}/seed_{42,123,2024}/` files `predictions_val_W{1..5}.parquet` and `metrics_val_W{1..5}.json`; `shared_init/hparams_seed{S}.json` | 3 seeds x 3 arms x 5 windows | shared checkpoint trained on days 1-111; update settings and threshold chosen on TRAIN days 112-121; adaptive arm updates on already-scored VAL labels | 93 |
| Cross-check only | `temporal_graphsage/comparison.json`, `temporal_ablation/comparison.json`, `phase8_adaptive_drift/comparison.json` (recorded values to compare against; never used as inputs to the table) | | | 3 |

The Phase 8 arm called "E" (replay control) is named **F-replay** here to avoid a clash with the brief's cell E; the other arms are **F-frozen** and **F-adaptive**.

**Source-set fingerprint:** 165 files. Build the lines `<posix path>\t<sha256 of file bytes>\n` for every file above, sort by path, join as UTF-8 with LF, and hash the text with SHA-256. Value at the time of writing:
`30f729b276a5ed4f5693704c5cf0d24b59c7d3817e33d37382b1a2f9bec43224`.
The build must recompute it at its start and end, write the per-file hash manifest, and stop if it differs.

**Excluded and never read:** `experiments/gat_300epoch/` (R6); `experiments/graphsage/` (Phase 5 single run, 60-epoch cap reached, superseded by Phase 6A); `experiments/graphsage_control_mlp/` (stale control, replaced by the Phase 6A extended control); `experiments/temporal_graphsage/pilot/`; every checkpoint and `model.joblib`; every feature file; any path containing "test"; `temporal_ablation` partial logs.

## 4. Proposed design (my choices; not requirements)

* **P1 Metric implementation.** One implementation: the existing, unmodified `src/models/metrics.py` (`average_precision_score` and `roc_auc_score` as called there, and `top_k_stats` with its default fractions 0.1%, 0.5%, 1%, 5%), imported read-only. `select_best_f1_threshold` is never called. Operating-point metrics (precision, recall, F1, alerts, confusion matrix) are computed only at a **recorded** threshold (A and the GNN runs: the best-F1-on-VAL value in `metrics_val.json`; F: the TRAIN-only value in `hparams_seed{S}.json`) and carry an "operating-point origin" label.
* **P2 Population, day and segment provenance.** The `A` prediction files have no `day` or `segment` columns. Day and the strict 3-way segment (`has_edge_warm`, `has_edge_cold`, `no_pseudo_account_edge`) are taken by `TransactionID` from the reference file `temporal_graphsage/seed_42/predictions_val.parquet`; every other file that carries these columns must agree with it exactly (AC4).
* **P3 Aggregation.** Per run, then per cell: per-seed values, mean and sample standard deviation (ddof = 1, which reproduces the reported Phase 6A convention), minimum and maximum. A cells with one run show the value only. No p-values, intervals or significance language.
* **P4 F basis, shown explicitly.** For each F arm and seed: (i) the Phase 8 basis, the mean of the five window PR-AUCs; (ii) the same metric code applied to the concatenation of the five stored windows (83,655 rows). A banner states the distinct basis. For the adaptive and replay arms, later windows are scored by updated weights, so the concatenation mixes weight states.
* **P5 Windows.** Window-level PR-AUC uses Phase 8's windows (122-127, 128-133, 134-139, 140-145, 146-151), assigned from `day`.
* **P6 Layout.** One row per run and per cell with the columns of requirement U5 plus the metrics, an "includes VAL-tuned selection" flag and the comparability label; a separate comparability matrix; a row for **B** reading "MISSING: no artifact exists; not substituted". Formats: CSV, JSON, Markdown, manifest. Location: `experiments/phase12_validation_evidence_table/`.

## 5. Comparability pre-registration (proposed labels, fixed before the table exists)

"Evaluation-comparable" means the same 83,655 VAL rows and labels scored by the same metric code. It never implies that a difference is attributable to the factor named.

| Comparison | Label | Reason |
|---|---|---|
| A (LR, RF, XGBoost) vs C, D, E | **Comparable on evaluation basis; attribution not supported** | same rows, labels and metrics; but different input encodings (raw/NaN-native versus imputed and standardized plus graph), different training regimes, A is one run versus three seeds, and all were tuned on VAL |
| C vs its MLP control | **Comparable, with limits** | same features, width and recipe; the control isolates the relations (Phase 6A); but the control is one run (seed 42) that reached its epoch cap, so it has no seed variance and may be under-trained |
| C vs D (architecture) | **Not comparable** | D reached the epoch cap in 3 of 3 seeds, so "worse" versus "under-trained" is unresolved; Phase 6C was aborted and is excluded (R6) |
| C vs E (temporal information) | **Not comparable as a single-factor comparison** | training regime differs (1 versus 18 optimizer steps per epoch); use the Phase 7A chain |
| Phase 7A chain: C reference, static schedule-matched, memory-ablated, E | **Comparable** | pre-registered ablation, same graph, features, seeds and protocol; n = 3 |
| F-frozen vs F-adaptive vs F-replay | **Comparable** | same shared checkpoint per seed, same windows and protocol; the adaptive arm updates on already-scored VAL labels |
| F (any arm) vs A, C, D or E | **Not comparable** | checkpoint trained on days 1-111 versus 1-121; TRAIN-only versus VAL-tuned threshold and no VAL epoch selection; different metric basis; adaptive updates use VAL labels; this holds even for F-frozen versus E despite the same architecture |
| A reference (historical control) vs anything | **Reference line only** | not an A-F cell; same rows, but its signal is the account-level labeling artifact documented in Phase 3 |
| Any statement about B or H2 | **Not available** | no graph-feature XGBoost exists; the historical-feature baseline is a different construct and is not substituted |
| Window-level PR-AUC across cells | Same labels as the rows above | window membership is identical, so only the training and protocol differences above apply |

## 6. Proposed acceptance criteria (for the later build)

| ID | Criterion |
|---|---|
| AC1 | Only the declared source set is read: the build logs every opened path and the set equals the declared files plus code; no `.pt`, `.joblib`, feature file, TEST-named file or excluded folder is opened. The fingerprint recomputed at start and end equals the value in section 3 |
| AC2 | Metrics come from the single implementation in P1; `select_best_f1_threshold` is neither imported nor called (static check); no threshold is selected or tuned |
| AC3 | Recomputed pooled PR-AUC and ROC-AUC, strict-segment PR-AUC and top-K statistics agree with the recorded `metrics_val.json` values and the cross-check files within an absolute 1e-9; the maximum difference is reported; any mismatch is reported and the recorded value is never substituted |
| AC4 | Population identity: all non-F files hold the same 83,655 `TransactionID`s with identical `isFraud`; `day` and `segment` agree exactly across every file that has them; the five F window files partition the same 83,655 ids with `window` consistent with P5 |
| AC5 | Labels are used only to compute metrics from the stored files, never to select rows, rank, set thresholds or choose settings |
| AC6 | Every row carries source paths, model and config identity, data window, metric basis, seed and source hashes (U5); the F banner and the B "MISSING" row are present; the excluded folders are absent |
| AC7 | Every comparison in section 5 carries its label and reason; no other comparison is reported; no row implies attribution, significance or generalization |
| AC8 | The cross-check files (`comparison.json` x3) are compared, not used as inputs; discrepancies are listed |
| AC9 | Phase 9 wording lint passes on all Phase 12 text and source; the VAL-tuned banner is on every table |
| AC10 | Outputs are deterministic (byte-identical on re-run apart from manifest timestamps) and have SHA-256 sidecars; manifests list source and output hashes |
| AC11 | Unit tests on synthetic prediction files cover the metric aggregation, population-identity checks, segment-join, comparability labelling and the exclusion guard; a mutation-sensitivity check shows the tests fail on injected defects |
| AC12 | A before/after SHA-256 comparison of every existing project file shows no change; the only new files are Phase 12 files |

## 7. Blockers and open items

* **B is missing** (no artifact). It stays a "MISSING" row. Producing B needs new leakage-safe graph-feature construction and a model fit, which is a separate authorization.
* **Not blockers for this build, but stated limits:** A has one run per model, so no seed variance; the GAT convergence question is unresolved; F is on a distinct basis; every number is VAL-tuned.
* **Unknown until built:** whether recomputation reproduces every recorded value (AC3) and whether `day` and `segment` agree across files (AC4). A failure of either stops the affected subtask and is reported.

## 8. Out of scope

B, the relationship-level and temporal-feature ablations, GAT convergence work, any trigger-based or common-basis F, any TEST evaluation, bootstrap or paired tests, graph evidence, Phases 13 onward.

## 9. Stop conditions

Stop and report if: the fingerprint changes; an unlisted file would be read; a TEST-named or checkpoint file would be opened; a recomputed value disagrees with its recorded value; the population or segment check fails; a threshold would have to be selected; or a new dependency would be needed.
