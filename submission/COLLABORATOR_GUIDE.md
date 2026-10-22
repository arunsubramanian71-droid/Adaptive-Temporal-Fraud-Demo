# Collaborator guide: a tour of the code

For a developer who has cloned the repository and wants to run it, read it and extend it. Setup and commands are in the root `README.md`; this guide explains where things are and how they fit together. Dated 2026-10-22.

## 1. What the project does, in three sentences

It studies whether graph structure and temporal behaviour improve fraud detection on the IEEE-CIS data compared with strong tabular models, with a strict chronological split and leakage-safe features. It also looks for *candidate* coordinated communities of cards and shared infrastructure (never confirmed: the data has no labels for coordination). Around the research sits working software: a causal scorer for new transactions, a local API and dashboard, a synthetic end-to-end demonstration and 381 tests.

## 2. What is empirical and what is synthetic

| You see | It is |
|---|---|
| Numbers in `experiments/`, `reports/`, `submission/MANUSCRIPT.md` and the dashboard's *Evidence* tab | **recorded validation-period results** on IEEE-CIS (days 122-151); models were tuned on that period |
| `scripts/demo_end_to_end.py` output, `demo_output/`, the dashboard's *Score transactions (demo)* tab | **synthetic**: a 40-day fake stream from `src/demo/synthetic.py`; it exercises the real modules and reproduces no research result |
| The dashboard's *Research model* tab | the **real recorded XGBoost** scoring rows you supply (needs local artifacts, see `ARTIFACT_INVENTORY.md`) |
| The *Candidate communities* tab | aggregate-only cards of exploratory real-data candidates (no member identities) |
| Anything about days 152-182 | **nothing exists yet**: the held-out period is sealed (section 7) |

## 3. Data flow through the code

```
data/raw/train_*.csv
  -> src/data/load.py, merge.py              memory-conscious CSV loading; transaction + identity merge on TransactionID
  -> src/data/splits.py                      day = TransactionDT // 86400; train 1-121, val 122-151, test 152-182 (configs/data.yaml)
  -> src/data/validate.py                    row and fraud counts, ordering, no target leakage
  -> src/features/preprocessing.py           time features, D-column clipping, missingness flags, TRAIN-fit encoders/imputers
  -> data/features/*_features.parquet        (scripts/build_processed_data.py)

  -> src/models/tree_features.py, lr_features.py   model-specific encoding (TRAIN-fit; saved next to each model)
  -> scripts/train_{logistic_regression,random_forest,xgboost}.py
  -> src/models/metrics.py                   every model's validation metrics; best-F1 threshold chosen on VAL only

  -> src/graph/schema.py, build.py           entity keys (card, pseudo-account = card1+addr1+(day-D1), network proxy, device) and the timed graph
  -> src/graph/rolling_eval.py               static GNNs: score day t on the graph masked to edges with time <= t
  -> src/graph/temporal.py                   temporal GNN: day slices are the model's only data accessor; labels unreachable from it
  -> src/models/{graphsage,gat,temporal_graphsage,temporal_meanmem}.py
```

## 4. Where each capability lives

| Capability | Code | Entry point | Tests |
|---|---|---|---|
| Leakage-safe graph features (40, label-free, strictly earlier rows) | `src/features/graph_features.py` | `scripts/run_comparison_b.py` | `tests/test_graph_features.py` (brute-force equivalence, truncation) |
| Label-derived history features (reference line) | `src/features/historical.py` | `scripts/build_historical_features.py` | a leakage self-check runs at build time; the checker itself is tested in `tests/test_preprocessing.py` |
| Temporal modelling (GRU entity memory, day by day) | `src/models/temporal_graphsage.py`, `src/graph/temporal.py`, `src/graph/temporal_checks.py` | `scripts/train_temporal_graphsage.py` | `tests/test_phase8_adaptive.py`, `tests/test_walkforward_ext.py` (need PyTorch) |
| Drift measurement (PSI, KS, temporal PR-AUC) | `src/adaptive/drift.py` | `scripts/measure_phase8_drift.py` | no dedicated unit test; the recorded measurements are in `experiments/phase8_adaptive_drift/drift_measurements/` and the Phase 8 report |
| Adaptive updating (frozen / periodic refresh / replay control), triggers | `src/adaptive/update_loop.py`, `replay.py`, `windows.py`, `src/heldout/walkforward.py`, `src/trigger/` | `scripts/train_phase8_*.py`, `scripts/run_triggered_refresh*.py` | `tests/test_phase8_adaptive.py` (windows, replay pools, score-then-update order), `tests/test_walkforward_ext.py`, `tests/test_trigger_*.py`, `tests/test_h4_rules.py` |
| Fraud scoring of new transactions (history + new, causal) | `src/inference/scorer.py` | `scripts/score_transactions.py` | `tests/test_inference_demo.py` |
| Research-model scoring (recorded XGBoost, exactly as evaluated) | `src/inference/research_model.py` | `scripts/run_dashboard.py --research-model` | `tests/test_research_serving.py` |
| Candidate-community discovery | `src/coordination/graphs.py` (windowed card-infrastructure graphs, Louvain), `scoring.py` (structure, temporal synchrony, model risk), `pipeline.py`, `tracking.py` | `scripts/run_phase9_*.py`, `scripts/run_phase9b_benchmark.py` | `tests/test_phase9_core.py`, `tests/test_phase9b_records.py` |
| Synthetic planted-group benchmark | `src/coordination/benchmark.py`, `metrics.py`, `nullmodel.py` | `scripts/build_phase9_benchmark.py` | `tests/test_phase9_core.py` |
| Community evolution over time | `src/evolution/tracker.py` | `scripts/run_phase10_tests.py` | `tests/test_phase10_*.py` |
| Explanations | `src/explain/xgb_contrib.py` (TreeSHAP contributions), `graph_evidence.py`, `neural_graph_evidence.py`, `cards.py` | `scripts/run_phase11_*.py`, `scripts/run_*_evidence.py` | `tests/test_phase11_*.py`, `tests/test_graph_evidence.py` |
| Consolidated evidence table | `src/evidence/` | `scripts/run_phase12_table.py` | `tests/test_phase12_*.py` |
| Held-out evaluation (gated, one-shot) | `src/heldout/` | `scripts/run_heldout_evaluation.py`, `run_amendment_evaluation.py`, `run_f_walkforward.py` | `tests/test_heldout_gate.py`, `test_amendment_gate.py`, `test_readiness_guards.py`, `test_f_runner_gate.py` |
| API and dashboard | `src/serving/app.py`, `src/serving/static/dashboard.html`, `src/serving/demo.py` | `scripts/run_dashboard.py` | `tests/test_serving.py`, `scripts/verify_dashboard_e2e.py`, `scripts/verify_dashboard_browser.py` |
| Synthetic demonstration | `src/demo/` | `scripts/demo_end_to_end.py` | `tests/test_inference_demo.py` |

## 5. How the API and the dashboard communicate

One process (`scripts/run_dashboard.py`) builds a `Services` object (`src/serving/app.py`) holding the optional scoring bundle and research model, and serves both the static page and JSON endpoints on 127.0.0.1. The page (`dashboard.html`, plain JavaScript, no external resources) calls `GET /api/evidence`, `/api/communities`, `/api/status` and `/api/research/model` when it loads, and `POST /api/score` or `/api/research/score` when you click a scoring button. The server reads only a fixed list of recorded files (`READABLE` in `app.py`, logged in `Services.opened`), never a TEST-named path, a checkpoint or a transaction-level file. Scoring calls `src/inference/scorer.py` (bundle mode) or `src/inference/research_model.py` (research mode); both validate input and return 400 with an explanation for bad rows.

## 6. Which artifacts you need

* **Demonstration, dashboard (demo mode), core tests:** nothing beyond the clone and `requirements.txt`.
* **Neural tests:** add `requirements-gnn.txt` (PyTorch, PyTorch Geometric).
* **Research-model mode and the 21 artifact-dependent tests:** the IEEE-CIS data and the trained files, which are not distributed. `ARTIFACT_INVENTORY.md` lists each file and the command that regenerates it.

## 7. Rules you must keep if you change anything

* **The held-out period (days 152-182) is sealed.** Do not open `data/features/test_features.parquet` or any TEST-named file, and do not create `experiments/final_heldout_protocol/AUTHORIZATION.txt`. The held-out runners refuse anyway, and the evaluation is one-shot.
* **Frozen records.** Pre-registrations, protocols, amendments and their `.sha256` sidecars must stay byte-identical (`tests/test_integrity_sidecars.py` checks every sidecar). New results go into new files.
* **Readiness fingerprints.** The held-out readiness records fingerprint the code they ran. If you edit a file under `src/` or `scripts/` that the evaluators load, the guard tests will tell you to re-run `scripts/run_heldout_evaluation.py --split val` and `scripts/run_amendment_evaluation.py --split val --modules H,U` (validation only, about a minute each, need the artifacts and PyTorch).
* **Wording.** Real-data groups are *candidate coordinated communities*. `src/coordination/lint.py` enforces this in the files it is applied to.
* **No row-level data in Git.** Prediction files, per-transaction explanations and anything with TransactionIDs plus values or labels stay local (`.gitignore` and `.git/info/exclude`).

## 8. Where to start reading

1. `README.md` (root) and `submission/MANUSCRIPT.md` sections 1-3, for the question and the design.
2. `configs/data.yaml` and `src/data/splits.py`: the split everything depends on.
3. `src/features/graph_features.py`: a short, self-contained example of the causality rule, with its brute-force test in `tests/test_graph_features.py`.
4. `src/inference/scorer.py` and `src/serving/app.py`: the software path you can run without data.
5. `src/models/temporal_graphsage.py` and `src/adaptive/update_loop.py`: the temporal and adaptive research core.
6. `src/coordination/pipeline.py`: candidate-community discovery.
7. `submission/TRACEABILITY_MATRIX.md`: every requirement of the original brief and where it is met.

## 9. Open research decisions

The owner must still decide: the held-out evaluation scope and authorization, the Module F readiness gate (registered gate failed as written; corrected proposal A2-a-r1 not active), acceptance of alternative evidence for Phase 12 AC12, confirmation of the reconstructed Phase 13 scope, and remediation of the public Git history. All are in `submission/HELDOUT_DECISION_PACKAGE.md`. None of them blocks running or reading the software.
