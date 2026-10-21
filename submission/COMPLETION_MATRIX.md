# Completion ledger

Dated 2026-10-20/21. Sources of the plan: the original brief (the first message of the earliest saved session; not a repository file), the user's stage-by-stage instructions recorded in the saved sessions, and the phase scope records. **The brief has no phase numbering and no roadmap file exists. Phases 2-12 were defined stage by stage by the user. A Phase 13 is referenced in the user's own instructions ("Do not start Phases 11-13") but its specification was never recovered; the best-supported reading (the last items of the user's own ordered list after Phase 7: dashboard and final report, plus the brief's production-style inference) is built and recorded as a reconstructed scope in `experiments/phase13_serving_dashboard/SCOPE_AND_ACCEPTANCE.md`. `FINAL_COMPLETION_RECORD.md` supersedes the Phase 13 statements of this ledger and of `reports/PROJECT_SYNTHESIS_REPORT.md` (which said Phase 13 was undefined).**

Status words: **PASS** = the required work was executed and its acceptance criteria were met; **FAIL** = a registered criterion failed; **BLOCKED** = cannot be completed without a decision, authorization or evidence that is not available. A failed *hypothesis* is not a failed phase when the pre-registered experiment was correctly completed; it is listed under "scientific outcome".

## 1. Brief requirement checklist

| Brief requirement | Evidence | Status | Scientific outcome / limitation |
|---|---|---|---|
| Strong tabular baselines (Levels 1-3) and the prior-fraud reference (Level 3.5) | Phase 3 report; `experiments/{logistic_regression,random_forest,xgboost,xgboost_historical_control}/` | PASS | XGBoost is the strongest; class weighting later found to hurt it (section 3) |
| Level 4: XGBoost + leakage-safe handcrafted graph features (comparison B) | `experiments/comparison_B_graph_features/` (pre-registration, 13 arms, tests) | PASS (executed under its criteria BA1-BA9) | H1-direct not established; H2 contradicted by the rule |
| Levels 5-7: GraphSAGE, GAT, one temporal architecture | Phases 5, 6A, 6B, 7 | PASS (executed); GAT convergence unresolved | no neural model beats XGBoost on validation |
| Heterogeneous graph with time-aware handling and leakage control | Phase 4; Phase 7/8 interventional checks | PASS | |
| Class imbalance evaluated experimentally | `experiments/imbalance_check_xgboost/` (two pre-registrations) | PASS (executed); GAT not covered | weighting hurts XGBoost; neural models insensitive or slightly better unweighted |
| Chronological evaluation; metrics incl. PR-AUC, ROC-AUC, P@K, R@K, confusion matrices, alerts | Phase 12 table; `src/models/metrics.py` | PASS on VAL | |
| **Held-out (TEST) evaluation** | `experiments/final_heldout_protocol/` (frozen protocol + Amendment 1; evaluator; readiness gate 19/19 models PASS on validation; amendment H, U PASS, F deterministic checks PASS and trained-arm gate FAIL as written 1 of 6) | **BLOCKED** | needs the user's one-time authorization; nothing was read or scored |
| Coordinated-community discovery + synthetic benchmark (H5) | Phase 9; Phase 9b (`experiments/phase9b_recovery_benchmark/`) | **FAIL** under the original protocol (frozen gate G4 failed on 2 of 3 seeds; protocol halted; exploratory continuation under a recorded deviation); Phase 9b, a new separately pre-registered benchmark with a corrected interference gate: all gates passed | H5a-c not established under the original protocol; **H5a-9b supported** (A2 beats every baseline on camouflaged groups in 3 of 3 seeds; G-cal holds); does not change the original verdict |
| Concept drift analysis | Phase 8 report section J | PASS | mild covariate drift, large performance drift |
| Adaptive updating vs static (H4) | Phase 8 | PASS | H4 supported under the locked rule (n = 3); the brief's drift-triggered wording was added as a pre-registered validation analysis (`experiments/phase8_triggered_refresh/`): label-free score-PSI trigger never fired (NOT SUPPORTED); a performance trigger fired once per seed and beat the frozen arm with 1 refresh instead of 4 but not the periodic arm (parity criterion NOT MET, NOT SUPPORTED); H4 unchanged |
| Explainability: SHAP-type for tabular models | Phase 11 | PASS (first pass) | descriptive, VAL-tuned |
| Explainability: evidence for graph models | `experiments/comparison_B_graph_features/graph_evidence/` (graph-feature model); `experiments/neural_graph_evidence/` (Temporal GraphSAGE seed 42: counterfactual link removal and memory reset on 300 transactions) | PASS for the designated neural checkpoint (executed; registered criterion NE5(a) FAILED at GPU float noise, reported); GraphSAGE and GAT not examined | attention weights not used (GAT never converged); descriptive, VAL-tuned |
| Ablations A-F | Phase 12 table + comparison B | PASS (B added); F on its own basis | |
| Relationship-level ablations ("where computationally feasible") | B ablations (4 relations + time windows); `experiments/relation_ablation_temporal/` (Temporal GraphSAGE, pre-registered) | PASS (executed) | pseudo-account and device contribute to the learned temporal model; card and network proxy do not hurt when removed; address/email not in the graph |
| Reproducibility (seeds, configs, hashes, scripts) | `README.md`, `REPRODUCIBILITY.md`, integrity records | PASS, except AC12 history (below) | |
| Engineering architecture; production-style inference; dashboard / API | `src/`, `scripts/`, `tests/`; `src/inference/scorer.py` + `scripts/score_transactions.py`; **`src/serving/` + `scripts/run_dashboard.py` (Phase 13, reconstructed scope)** | PASS (executed under the reconstructed criteria P13-1 to P13-10) | the recorded research XGBoost is served on `/api/research/*` exactly as evaluated (tested against the recorded predictions); standard library only: FastAPI and Streamlit were not installed (the brief lists them as "likely"); the Phase 13 specification itself was not recovered, so whether this matches the user's original intent is unconfirmed |
| Executable demonstration + tests | `scripts/demo_end_to_end.py`; `scripts/run_tests.py`; `submission/test_results.json` | PASS | demonstration is synthetic and reproduces no research result |
| Final submission package | `submission/` | PASS (ready, with limitations disclosed) | |

## 2. Phase ledger

| Phase | Original requirement | Evidence | Status | Unresolved limitation |
|---|---|---|---|---|
| 1 Reconnaissance | dataset/schema report, leakage and entity analysis | `reports/phase1_reconnaissance.md` | PASS | |
| 2 Data pipeline, EDA | frozen chronological split, validated processed data | `reports/phase2_eda_report.md`; `tests/test_{preprocessing,splits,validate}.py` | PASS | |
| 3 Tabular baselines | LR, RF, XGBoost, historical control on VAL | `reports/phase3_baseline_report.md` | PASS | class-weighting choice not tested at the time; tested later |
| 4 Graph construction | schema, leakage checks, feasibility | `reports/phase4_graph_report.md` (8 of 8 checks) | PASS | |
| 5 GraphSAGE | one official run + non-relational control | `reports/phase5_graphsage_report.md` | PASS (superseded by 6A for the reference) | 60-epoch cap reached |
| 6A GraphSAGE robustness | 3 seeds, larger budget, control | `reports/phase6a_...` | PASS | one seed hit the cap |
| 6B GAT | 3 seeds under the same protocol | `reports/phase6b_gat_report.md` | PASS (executed) | all seeds hit the cap; convergence unresolved |
| 6C GAT 300 epochs | convergence resolution | `reports/phase6c_gat_convergence_status.md` | **FAIL / aborted**, excluded from all conclusions | |
| 7 Temporal GraphSAGE | one temporal architecture, leakage validation | `reports/phase7_temporal_gnn_report.md` (validation 19 of 19) | PASS | gain confounded by training regime |
| 7A Temporal ablation | schedule-matched and memory-ablated controls | `reports/phase7a_...` | PASS | n = 3 |
| 8 Drift and adaptation | H4 rule, leakage validation | `reports/phase8_concept_drift_report.md` (90 of 90) | PASS | n = 3; zero label delay |
| 9 Coordinated communities | pre-registered benchmark + gates | `reports/phase9_closure_report.md`; Phase 9b results | **FAIL (gate G4)**; closed with exploratory observations; Phase 9b (separate) supported H5a-9b | original H5 not established |
| 10 Community evolution | design + core module; empirical tiers | `experiments/phase10_community_evolution/` (63 of 63 tests, 18 of 18 defects detected) | core PASS; empirical tiers **BLOCKED** | thresholds unchosen; not authorized; not part of the brief |
| 11 Explainable evidence | explanations + cards (first pass) | `PHASE11_CLOSURE_REPORT.md` (30 of 30; 14 of 14) | PASS (first pass); graph evidence for neural models deferred | |
| 12 Validation-evidence table | AC1-AC12 | `PHASE12_CLOSURE_REPORT.md`; `AC12_GIT_SNAPSHOT_VERIFICATION.json` | AC1-AC11 PASS; **AC12 BLOCKED**; **Phase 12 not formally complete** | per-file recovery: 550 of 651 files with pre-build evidence (434 via the authenticated first Git commit), 0 contradictions, 101 with none; not the registered method |
| 13 | specification **not recovered**; reconstructed as dashboard + API + final report | `experiments/phase13_serving_dashboard/`; `src/serving/`; 14 tests, 12 of 14 injected defects detected and 2 equivalent | PASS against the reconstructed criteria; **scope unconfirmed by the user** | the original Phase 13 may have meant something else |

## 3. Work completed in this execution (all new files; no historical artifact changed)

| Item | What was done | Verification | Outcome |
|---|---|---|---|
| Comparison B | pre-registration; `src/features/graph_features.py`; 13 fits | 13 feature tests incl. brute-force equivalence and real-data truncation check; 14 of 14 injected defects detected; 12 output tests | H1-direct **NOT ESTABLISHED**; H2 **contradicted** |
| Graph evidence for B | pre-registration; `src/explain/graph_evidence.py`; 300 transactions | 7 unit tests; GE1-GE4 PASS; byte-identical re-run | descriptive |
| Class-imbalance study | two pre-registrations; unweighted XGBoost, GraphSAGE and Temporal GraphSAGE | output tests UA/IA | weighting hurts XGBoost (0.5948 vs 0.5395); tabular > neural in both recipes |
| Held-out protocol + evaluator | frozen protocol; `src/heldout/evaluator.py`; gate tests | 8 gate tests; readiness: 19 of 19 frozen models reproduce recorded validation predictions (max score difference 6.56e-07) | held-out run **BLOCKED** on authorization |
| Inference, demonstration | `src/inference/`, `src/demo/`, two scripts | 12 tests; demonstration deterministic | PASS |
| Submission package | README, guides, manuscript, ledger, claim checker, integrity record | `scripts/verify_submission_claims.py`; `submission/INTEGRITY_SUBMISSION.json` | see `submission/test_results.json` |

## 4. Disclosures

* Reading `data/processed/split_metadata.json` to obtain the train and validation row counts also displayed the held-out period's aggregate row and fraud counts; they were not used anywhere.
* The readiness gate and the neural training scripts load the Phase 4 graph file, which contains held-out-period nodes; rows after day 151 are dropped immediately (readiness gate) or hidden by time masks (training scripts), as in every earlier phase. No held-out label or score was produced.
* A dry run once wrote files under a stray `C:\c\Users\...` path; they were deleted; nine empty folders remain there for the user to remove.

## 5. Remaining blockers and decisions (precise)

1. **Held-out evaluation:** create `experiments/final_heldout_protocol/AUTHORIZATION.txt` quoting the protocol's SHA-256 and the sentence in its section 9, then run `scripts/run_heldout_evaluation.py --split heldout`. The evaluator is ready and checked on validation. The base protocol alone leaves the adaptive system (H4 on future data) and the historical reference line unevaluated; `experiments/final_heldout_protocol/AMENDMENT_1.md` (modules F, H, U; additional hashed authorization needed) closes them, and `SCOPE_RECONCILIATION.md` gives the model-by-model decision. Module F's trained-arm window-mean readiness gate failed as written in 1 of 6 cases (0.0104 against 0.01; not changed).
2. **Phase 12 AC12:** cannot be satisfied retrospectively as registered; per-file recovery covers 550 of 651 files (0 contradictions), 101 with no evidence.
3. **Phase 9 / H5:** the original G4 failure stands; the new, separately pre-registered Phase 9b benchmark was run and supports H5a-9b (separate result).
4. **Phase 13 scope:** confirm or correct the reconstruction.
5. **Optional:** explanations of GraphSAGE; GAT unweighted; GraphSAGE/GAT relation ablations (the temporal model was ablated); Phase 10 empirical tiers.
6. **Public Git history:** six transaction-level evidence files remain in earlier commits; make the repository private or approve a history rewrite.

## 6. Overall acceptance

**The entire project acceptance checklist has NOT fully passed**: the held-out evaluation (a brief requirement) is BLOCKED on authorization, Phase 9's frozen gate FAILED (the separate Phase 9b benchmark supports H5a-9b), Phase 12's AC12 is BLOCKED as registered, graph-level explanation exists only for Temporal GraphSAGE seed 42 (with one registered sanity criterion failed at float noise level), and the Phase 13 scope is a reconstruction awaiting confirmation. Everything else in the checklist above is complete and verified.
