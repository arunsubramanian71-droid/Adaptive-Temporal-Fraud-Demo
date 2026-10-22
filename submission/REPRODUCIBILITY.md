# Reproducibility guide

Two levels of reproduction exist and they must not be confused:

* **Level 1, software:** the demonstration and the test suite run on any machine with the Python environment (`README.md` sections 2-4). They use synthetic data and prove that the code runs and behaves as specified. They do **not** reproduce any research number.
* **Level 2, research results:** need the IEEE-CIS files, the recorded intermediate files, and (for neural models) a GPU. In most cases you do not need to re-run anything: the recorded outputs, their configurations and their hash records are the evidence.

## Environment and determinism

**Recommended interpreter: Python 3.12** (all recorded manifests were produced with 3.12.0; in clean virtual environments built only from `requirements.txt`, Python 3.12.0 passes 234 of 234 executed tests, while Python 3.14.3 passes 233 of 234 plus one environment-metadata difference, the interpreter version in a rebuilt manifest; evidence in `submission/clean_env_results/`). Python 3.12.0, library versions in `requirements.txt` (core) and `requirements-gnn.txt` (neural). The neural experiments used an AMD ROCm build of PyTorch on Windows; GPU kernels are not bit-reproducible across machines, which is why the Phase 7 and 8 leakage checks use exact equality on CPU and a stated tolerance on GPU. Seeds are 42, 123 and 2024 everywhere; XGBoost is deterministic for a fixed seed and thread count except that CPU histogram builds with many threads may reorder floating-point additions (the B pipeline control re-fit reproduced the recorded comparison-A PR-AUC exactly: 0.5395, best iteration 798). Random seeds, split boundaries (`configs/data.yaml`: train 1-121, validation 122-151, held-out 152-182), thresholds and hyperparameters are recorded in each experiment's `config.json` / `metadata.json`.

## Experiment map

| Experiment | Command (from the project root, `.venv/Scripts/python.exe -B`) | Needs | Outputs | Do not |
|---|---|---|---|---|
| Processed data and splits | `scripts/build_processed_data.py`, `scripts/run_validation.py` | `data/raw/train_*.csv` | `data/processed/`, `data/features/` | edit raw files |
| Tabular baselines (A) | `scripts/train_logistic_regression.py`, `train_random_forest.py`, `train_xgboost.py`, `train_xgboost_historical_control.py` (after `build_historical_features.py`) | processed features | `experiments/{logistic_regression,random_forest,xgboost,xgboost_historical_control}/` | overwrite without copying aside |
| Graph | `scripts/build_graph.py`, `validate_graph.py` | processed data | `data/processed/graph/` | install compiled PyG extensions |
| GraphSAGE, GAT (C, D) | `scripts/train_graphsage_robustness.py`, `train_graphsage_control_mlp_extended.py`, `train_gat_robustness.py` | GPU, graph | `experiments/graphsage_robust/`, `experiments/gat/` | run `train_gat_convergence_300epoch.py` (aborted Phase 6C) |
| Temporal GraphSAGE and controls (E) | `scripts/train_temporal_graphsage.py`, `train_static_schedule_matched.py`, `train_temporal_mean_memory.py`, then `validate_temporal_gnn.py`, `validate_temporal_ablation.py` | GPU | `experiments/temporal_graphsage/`, `experiments/temporal_ablation/` | skip the leakage validation |
| Adaptive updating (F) | `scripts/train_phase8_init.py`, `train_phase8_frozen.py`, `train_phase8_adaptive.py`, `train_phase8_replay_control.py`, `measure_phase8_drift.py`, `validate_phase8_leakage.py`, `compare_phase8.py` | GPU | `experiments/phase8_adaptive_drift/` | change the locked rule |
| Coordinated communities (H5) | `scripts/run_phase9_*.py`, `validate_phase9_*.py` in the order of the Phase 9 closure report | graph cache | `experiments/phase9_coordinated_fraud/` | change gates, thresholds or the frozen design |
| Explanations and cards | `scripts/run_phase11_explanations.py`, `run_phase11_cards.py` | XGBoost model, Phase 9 tier outputs | `experiments/phase11_explainable_evidence/` | run with TEST-named inputs |
| Consolidated table | `scripts/run_phase12_table.py` | recorded predictions | `experiments/phase12_validation_evidence_table/outputs/` | re-run finalization expecting AC12 to change |
| **Comparison B** | `scripts/run_comparison_b.py`, then `scripts/build_b_outputs.py` | train and validation feature files, the Phase 9 entity cache (CPU; about 13 XGBoost fits, roughly one hour) | `experiments/comparison_B_graph_features/outputs/` | run before `PREREGISTRATION_B.sha256` verifies |
| Graph evidence for B | `scripts/run_b_graph_evidence.py` | B_full model | `experiments/comparison_B_graph_features/graph_evidence/` | substitute a different selection |
| Imbalance check | `scripts/run_imbalance_check.py` | comparison B finished | `experiments/imbalance_check_xgboost/outputs/` | add arms after seeing results |
| Neural graph evidence (Temporal GraphSAGE seed 42) | `scripts/run_neural_graph_evidence.py`, then `scripts/build_neural_evidence_report.py`; tests `tests/test_neural_graph_evidence.py` | GPU (about 10 minutes), the checkpoint, the Phase 4 graph | `experiments/neural_graph_evidence/outputs/` | change the registered criteria after seeing results |
| Imbalance follow-up (unweighted) | `scripts/run_unweighted_xgboost.py`, `scripts/run_unweighted_neural.py --family static` and `--family temporal`, then `scripts/build_unweighted_results.py` | comparison B finished; GPU for the neural runs (about 25 + 10 minutes) | `experiments/imbalance_check_xgboost/outputs/arms_unweighted/`, `experiments/unweighted_graphsage/`, `experiments/unweighted_temporal_graphsage/` | add arms after seeing results |
| Dashboard and API (Phase 13) | `scripts/run_dashboard.py --demo` (or `--bundle DIR`); tests `tests/test_serving.py`; mutation check `scripts/check_serving_test_sensitivity.py` | requirements.txt only | none (serves recorded aggregates; reads only the files listed in `src/serving/app.py` `READABLE`) | bind to a non-local address without understanding that there is no authentication |
| Dashboard end-to-end check | `scripts/verify_dashboard_e2e.py` (starts and stops its own server process) | requirements.txt only | `experiments/phase13_serving_dashboard/test_results/e2e_verification.json` | none |
| Held-out protocol review | `scripts/verify_heldout_protocol_ready.py` (read-only; opens no held-out file) | none | `experiments/final_heldout_protocol/PROTOCOL_REVIEW.md`, `PROTOCOL_REVIEW_FACTS.json` | none |
| Dashboard in a real browser (optional) | `scripts/verify_dashboard_browser.py --research-rows PATH` with Playwright from `requirements-browser-check.txt` (separate environment; installed Chrome) | the research model for the research checks | `experiments/phase13_serving_dashboard/test_results/browser/` | do not point `--out-dir` at a folder that is published while using real rows for screenshots (the script itself photographs only synthetic placeholder rows) |
| Held-out readiness gate | `scripts/run_heldout_evaluation.py --split val` (re-scores the 19 frozen models on validation and compares with the recorded predictions) | GPU | `experiments/final_heldout_protocol/readiness_val/` | none |
| Held-out evaluation | `scripts/run_heldout_evaluation.py --split heldout` is **refused** until the user creates `experiments/final_heldout_protocol/AUTHORIZATION.txt` (protocol section 9); one-shot | explicit user authorization | `experiments/final_heldout_protocol/heldout_results/` | open any held-out file by any other route |

The comparison-B script verifies the SHA-256 of its pre-registration, refuses any path outside its input list (any TEST-named path, any checkpoint), checks the entity-id cache against the feature files, runs the real-data truncation check, and resumes where it stopped (completed arms with an identical configuration are skipped).

## Outputs and integrity

Every result folder contains SHA-256 sidecars; integrity records are listed in `README.md` section 7 and `submission/INTEGRITY_SUBMISSION.json`. To check a file: `python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" <file>` and compare with its sidecar. The Phase 12 limitation (AC12 BLOCKED) applies to the history of 322 files that have no authentic pre-build hash; it is not resolved by this package.

## What was not reproduced during packaging

No neural model was re-trained or re-scored while preparing this package; their numbers come from the recorded predictions and reports. No model was scored on the held-out period. (Until 2026-10-22 one split-count test opened its feature table to re-check the Phase 1 row and fraud counts; see `HELDOUT_DECISION_PACKAGE.md` section 1a.)
