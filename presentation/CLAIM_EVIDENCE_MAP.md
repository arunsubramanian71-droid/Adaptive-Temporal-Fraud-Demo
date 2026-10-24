# Claim and Evidence Map: Adaptive Temporal Graph Intelligence for Fraud Detection

**Project:** Adaptive Temporal Graph Intelligence for Fraud Detection & Coordinated Fraud Discovery  
**Evaluation Scope:** Validation-period only (Days 122–151 of IEEE-CIS). Held-out period (Days 152–182) remains sealed by design.  
**Verification Date:** 2026-10-23  

This document provides a line-item verification map connecting every factual, empirical, and architectural claim in the presentation deck to its exact source file, code implementation, and quantitative result artifact in the repository.

---

## 1. Data Pipeline, Problem Formulation & Leakage Safeguards

| # | Slide Claim | Source File / Implementation | Result Artifact / Metric | Verification Status & Caveats |
|---|---|---|---|---|
| 1.1 | **Dataset Scale:** IEEE-CIS Fraud Detection dataset contains 590,540 labelled transactions with 3.499% fraud prevalence (imbalance 27.58:1). | `configs/data.yaml`<br>`src/data/build_processed_data.py` | `data/processed/build_log.json`<br>`reports/phase1_reconnaissance.md` | **Verified.** Exact transaction count: 590,540. Fraud count: 20,663 (3.499%). |
| 1.2 | **Chronological Split:** Train: Days 1–121 (417,559 txns, 14,721 fraud); Validation: Days 122–151 (83,655 txns, 2,828 fraud); Held-out: Days 152–182 (89,326 txns, sealed). | `configs/data.yaml`<br>`src/data/splits.py` | `submission/MANUSCRIPT.md` §3<br>`reports/phase2_eda_stats.json` | **Verified.** Timestamp derived as `TransactionDT // 86400`. Encoders and scalers fitted on Train only. |
| 1.3 | **Labelling Artifact:** A single backward-looking feature (prior fraud on the same pseudo-account) separates 59.86% fraud from 0.08%, presenting massive risk of target leakage in unconstrained GNNs. | `src/features/historical.py`<br>`reports/phase1_reconnaissance.md` | `submission/MANUSCRIPT.md` §1<br>`reports/phase3_baseline_report.md` | **Verified.** Accounts for why the historical-feature model scores 0.6771 PR-AUC overall but collapses to 0.0369 on cold-start transactions. Kept as a reference line, never an evaluation cell. |
| 1.4 | **Temporal Causality & Leakage Invariants:** All features and graph structures use strictly $t' < t$ (features) or $t' \le t$ (same-day non-label graph connections). No future labels used. | `src/features/graph_features.py`<br>`src/graph/temporal.py` | `tests/test_preprocessing.py`<br>`tests/test_phase8_adaptive.py` | **Verified.** Interventional tests confirm corrupting future days yields bit-identical historical scores. |
| 1.5 | **Heterogeneous Graph Schema:** 5 node types (transaction: 590k, pseudo-account: 200k, card: 13.5k, netproxy: 2.4k, device: 4.9k) and 4 relation types (`charged_to`, `used_card`, `via_network`, `from_device`). | `src/graph/build_graph.py`<br>`src/graph/schema.py` | `reports/phase4_graph_report.md`<br>`submission/MANUSCRIPT.md` §3 | **Verified.** Candidate relations with high hub degree and zero signal (billing address, email domain, device model string) were measured and excluded. |

---

## 2. Model Performance & Core Baselines (Validation Period)

| # | Slide Claim | Source File / Implementation | Result Artifact / Metric | Verification Status & Caveats |
|---|---|---|---|---|
| 2.1 | **Tabular Baseline Superiority:** Tuned tabular XGBoost achieves 0.5395 PR-AUC (weighted) and 0.5948 (unweighted), outperforming every static and temporal neural graph model. | `src/models/xgboost_model.py`<br>`scripts/train_xgboost.py` | `experiments/phase12_validation_evidence_table/outputs/table_report.md`<br>`submission/figures/figure_data.json` | **Verified.** Tuned on validation (best iter 798, early stopping). Subsampled XGBoost (3 seeds) achieves 0.5616 ± 0.0023. |
| 2.2 | **Simpler Baselines:** Logistic Regression achieves 0.4126 PR-AUC (ROC-AUC 0.8520); Random Forest achieves 0.4543 PR-AUC (ROC-AUC 0.8792). | `scripts/train_logistic_regression.py`<br>`scripts/train_random_forest.py` | `experiments/logistic_regression/metrics_val.json`<br>`experiments/random_forest/metrics_val.json` | **Verified.** Recomputed across 8,184 metric values in Phase 12 with 0.0 discrepancy. |
| 2.3 | **Neural Graph Performance (Cells C & D):** HeteroSAGE achieves 0.4637 ± 0.0167 PR-AUC (3 seeds), beating the no-relations MLP control (0.4167). GAT achieves 0.3320 ± 0.0052. | `src/models/graphsage.py`<br>`src/models/gat.py` | `reports/phase6a_graphsage_robustness_report.md`<br>`reports/phase6b_gat_report.md` | **Verified.** GAT failed to converge within the 150-epoch cap; "worse" and "under-trained" cannot be disentangled. Phase 6C 300-epoch run was aborted and excluded. |
| 2.4 | **Temporal GraphSAGE (Cell E):** Temporal GraphSAGE with GRU entity memory achieves 0.4927 ± 0.0035 PR-AUC across 3 seeds. | `src/models/temporal_graphsage.py`<br>`src/graph/temporal.py` | `reports/phase7_temporal_gnn_report.md`<br>`submission/MANUSCRIPT.md` §5.1 | **Verified.** Highest performing graph architecture, but remains 0.0468 PR-AUC below the standard tabular XGBoost baseline. |
| 2.5 | **Comparison B (XGBoost + 40 Handcrafted Graph Features):** Adding 40 label-free graph features to XGBoost yields 0.5440 PR-AUC (+0.0046 single-run) and 0.5611 ± 0.0034 (3 seeds subsampled vs 0.5616 for tabular alone). | `src/features/graph_features.py`<br>`scripts/run_comparison_b.py` | `experiments/comparison_B_graph_features/outputs/B_RESULTS.md` | **Verified.** Pre-registered decision rule: H1-direct is **NOT ESTABLISHED** (mean seed delta -0.0005 vs SD 0.0034). H2 is **CONTRADICTED** (handcrafted features beat learned representations). |

---

## 3. Temporal Entity Memory & Relation Ablations

| # | Slide Claim | Source File / Implementation | Result Artifact / Metric | Verification Status & Caveats |
|---|---|---|---|---|
| 3.1 | **Recurrent Memory Benefit (H3):** Temporal GraphSAGE (0.4927) beats its memory-ablated control (0.4790) in 3 of 3 seeds (+0.0137 mean difference) and static schedule-matched (0.4792) in 2 of 3 seeds (+0.0135). | `src/models/temporal_graphsage.py`<br>`scripts/compare_phase7a.py` | `reports/phase7a_temporal_ablation_report.md`<br>`submission/MANUSCRIPT.md` §5.3 | **Verified.** Effect is modest and highly concentrated in cold-start transactions (+0.027 PR-AUC, 3 of 3 seeds). H3 is **partially supported**. |
| 3.2 | **Temporal Relation Ablations:** Removing `charged_to` (pseudo-account) lowers PR-AUC by -0.0309 across all 3 seeds (0.4618 vs 0.4927). Removing `from_device` drops PR-AUC by -0.0073. | `experiments/relation_ablation_temporal/`<br>`scripts/train_temporal_graphsage.py` | `experiments/relation_ablation_temporal/RESULTS_RELATION_ABLATION.md` | **Verified.** Pre-registered 3-seed ablation. Removing `used_card` (-0.0038) or `via_network` (-0.0053) falls within run-to-run noise (0.0058 full-model rerun). |

---

## 4. Concept Drift & Continual Adaptive Updating

| # | Slide Claim | Source File / Implementation | Result Artifact / Metric | Verification Status & Caveats |
|---|---|---|---|---|
| 4.1 | **Severe Performance Drift vs Mild Covariate Drift:** Covariate shift is very mild (mean PSI ~ 0.01 across 470 features), but model performance drops steeply (PR-AUC drops from ~0.60 to ~0.33 over 30 days). | `src/adaptive/drift_detector.py`<br>`scripts/measure_phase8_drift.py` | `reports/phase8_concept_drift_report.md`<br>`submission/MANUSCRIPT.md` §5.4 | **Verified.** Demonstrates that distribution shift in inputs does not necessarily correlate with label-conditional degradation. |
| 4.2 | **Periodic Adaptation Mitigates Drift (H4):** Across five 6-day validation windows, Adaptive Fine-tuning achieves window-mean PR-AUC 0.4580 ± 0.0082 vs Frozen 0.4243 ± 0.0128 and Replay Control 0.4179 ± 0.0063. | `src/adaptive/update_loop.py`<br>`src/heldout/walkforward.py` | `experiments/phase8_adaptive_drift/comparison.json`<br>`submission/figures/figure_data.json` | **Verified.** All 3 locked pre-registered rules pass: Adaptive > Frozen in 3 of 3 seeds (mean +0.0338 > SD 0.0128), and Adaptive > Replay (+0.0402 > bound 0.0082). Replay confirms compute alone does not explain gain. |
| 4.3 | **Triggered Refresh Failure:** A label-free score-distribution PSI monitor never fired (PSI <= 0.025 vs threshold 0.10) while frozen PR-AUC plunged. Performance trigger fired once per seed (0.4493 mean), beating frozen but lagging periodic (0.4733). | `src/trigger/drift_trigger.py`<br>`src/trigger/performance_trigger.py` | `experiments/phase8_triggered_refresh/`<br>`submission/MANUSCRIPT.md` §5.4 | **Verified.** Both triggered refresh strategies are **not supported** by their pre-registered rules. |

---

## 5. Candidate-Community Discovery, Explainability & Class Imbalance

| # | Slide Claim | Source File / Implementation | Result Artifact / Metric | Verification Status & Caveats |
|---|---|---|---|---|
| 5.1 | **Candidate Communities (H5):** Bipartite Louvain clustering discovers suspicious infrastructure-sharing groups. Real-data clusters are strictly *candidate coordinated communities*, not confirmed fraud rings. | `src/coordination/louvain.py`<br>`src/coordination/scoring.py` | `reports/phase9_closure_report.md`<br>`submission/MANUSCRIPT.md` §5.5 | **Verified.** Original synthetic benchmark failed frozen interference gate G4. Pre-registered Phase 9b benchmark passed (recovery AUPRC 0.14–0.27 vs <0.05 baselines). Zero ground-truth fraud ring labels exist in IEEE-CIS. |
| 5.2 | **Transparent Explanations:** XGBoost uses exact TreeSHAP contributions summing to margin (error < 8.1e-06). Temporal GNN uses counterfactual link removal (pseudo-account logit change 3.16). | `src/explain/tree_shap.py`<br>`src/explain/counterfactual.py` | `experiments/phase11_explainable_evidence/PHASE11_CLOSURE_REPORT.md` | **Verified.** Evaluated on 300 label-free validation transactions. TreeSHAP describes model margin, not external real-world causality. |
| 5.3 | **Class Imbalance Penalty:** Full class weighting (scale_pos_weight 27.36) degrades XGBoost PR-AUC from 0.5948 (unweighted) to 0.5395 (weighted). | `experiments/imbalance_check_xgboost/`<br>`src/models/xgboost_model.py` | `experiments/imbalance_check_xgboost/outputs/arms_table.csv`<br>`submission/figures/figure_data.json` | **Verified.** A 0.0553 PR-AUC penalty—larger than any graph feature or neural architectural gain in the study. Subsampled: 0.6065 unweighted vs 0.5616 weighted. |

---

## 6. Software Architecture, Verification & Open Boundaries

| # | Slide Claim | Source File / Implementation | Result Artifact / Metric | Verification Status & Caveats |
|---|---|---|---|---|
| 6.1 | **Zero-Dependency Serving:** Local HTTP API and interactive dashboard built strictly on Python's standard library (no Flask/FastAPI/Node). | `src/serving/app.py`<br>`src/serving/static/dashboard.html` | `scripts/verify_dashboard_e2e.py`<br>`experiments/phase13_serving_dashboard/` | **Verified.** 31/31 e2e route checks pass. Serves recorded validation tables, aggregate community cards, synthetic demo scoring, and research XGBoost. |
| 6.2 | **Comprehensive Test Suite:** 381 automated tests with zero-leakage invariant checks; 306/327 pass on a clean clone without data files. | `scripts/run_completion_tests.py`<br>`tests/test_*.py` (40 modules) | `submission/test_results.json`<br>`submission/COMPLETION_STATUS_2026-10-22.md` | **Verified.** Verified on development machine and clean clones. |
| 6.3 | **Sealed Held-Out Period:** Days 152–182 remain sealed under cryptographic hash guards; zero held-out numbers are reported or known. | `experiments/final_heldout_protocol/`<br>`src/heldout/oneshot.py` | `submission/HELDOUT_DECISION_PACKAGE.md`<br>`submission/HELDOUT_AUTHORIZATION_REQUEST.md` | **Verified.** Ready for execution but deliberately unrun awaiting formal owner authorization. |
