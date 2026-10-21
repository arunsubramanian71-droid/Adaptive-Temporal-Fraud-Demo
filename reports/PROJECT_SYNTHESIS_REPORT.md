# Project synthesis report

*Adaptive Temporal Graph Intelligence for Fraud Detection & Coordinated Fraud Discovery.* (The title's final words are abbreviated here on purpose, following the Phase 9 closure report: the terminology rule forbids ring wording for real-data outputs.)

Dated 2026-10-19. Documentation only. Sources: the original research brief (the first message of the earliest saved session, `29b7adc9-d8fc-4476-bf2a-463265759ff5.jsonl`; it is not a repository file), the phase reports in `reports/`, the closure reports and scope records of Phases 9-12, and the Phase 12 evidence table. No experiment was run, no metric was recomputed, no model was trained or scored, no TEST content was opened, and no existing file was modified to produce this report. Every number below is copied from an existing report or from the Phase 12 evidence table; where this report adds a judgement, the text says so.

**All results here are validation-period (VAL, days 122-151) results. No TEST evaluation has been performed. Most models were tuned on VAL (early stopping, epoch selection or thresholds). Nothing here supports a claim about held-out or future-time generalization, statistical significance, causal attribution, fraud or coordination.**

**Phase 12 is not formally complete under its registered criteria: AC12 is BLOCKED.** Phase 13 is not defined in any project document and is not a documented requirement; this report is not a Phase 13 specification.

## 1. Research question and the A-F comparisons

**Question (brief):** does heterogeneous relational structure plus temporal behavioural information improve fraud detection and coordinated-group discovery over strong conventional tabular ML, particularly under future-time evaluation and concept drift? The brief says negative results are to be reported honestly. The split is chronological: TRAIN days 1-121, VAL 122-151, TEST 152-182 (sealed).

| Cell | Comparison (brief) | Existing artifacts | Status |
|---|---|---|---|
| A | tabular only | Phase 3: logistic regression, random forest, XGBoost (one run each); historical-feature XGBoost kept as a reference line | Evidence exists |
| **B** | **tabular + graph features** | **none** | **NOT TESTED / MISSING.** No graph-feature XGBoost was ever built. The historical-feature baseline is a different construct and is not substituted. |
| C | GraphSAGE | Phases 5 and 6A (6A is the official reference): 3 seeds plus a no-relations MLP control (one run) | Evidence exists |
| D | GAT | Phase 6B (official): 3 seeds, all reached the 150-epoch cap without early stopping. Phase 6C was aborted, produced no usable result and is excluded. | Evidence exists; convergence unresolved |
| E | graph model + temporal information | Phase 7 (Temporal GraphSAGE, GRU entity memory) and the Phase 7A controls (schedule-matched static, memory-ablated): 3 seeds each | Evidence exists |
| F | full adaptive system | Phase 8: frozen, adaptive and replay-control arms, 3 seeds, five 6-day windows | Evidence exists, on a distinct basis |

Relationship-level ablations (device, address, email, specific temporal features) were requested in the brief "where computationally feasible". I found no report that records one; none were run in the phases covered here.

The graph (Phase 4, built from the data; Phase 1 and Phase 4 justify the schema) has 590,540 transaction nodes and four entity types (pseudo-account, card, netproxy, device fingerprint) with four relations; candidates with no label signal above chance (billing address, email domain, product, device-info string) were excluded on the record. Phase 1 and Phase 3 documented that a single prior-fraud flag separates classes very strongly (the account-level labelling artifact), so the historical-control XGBoost (pooled PR-AUC 0.6771 but cold-start far lower) is a reference line, never an A-F cell.

### 1.1 Consolidated validation-evidence table (Phase 12; all values recomputed from stored VAL predictions with one metric implementation)

Cells with several seeds show mean ± sample SD over seeds. Cold-start PR-AUC uses the strict 3-way segmentation used for every graph model.

| Cell | Model / arm | PR-AUC | ROC-AUC | PR-AUC cold-start |
|---|---|---|---|---|
| A | logistic regression | 0.4126 | 0.8520 | 0.2972 |
| A | random forest | 0.4543 | 0.8792 | 0.3807 |
| A | XGBoost | 0.5395 | 0.9050 | 0.4064 |
| A reference | XGBoost, historical features (reference line only) | 0.6771 | 0.9432 | 0.0369 |
| C | GraphSAGE (Phase 6A), 3 seeds | 0.4637 ± 0.0167 | 0.8616 ± 0.0038 | 0.3417 ± 0.0113 |
| C control | MLP, no relations (one run, reached its epoch cap) | 0.4167 | 0.8547 | 0.3007 |
| D | GAT (Phase 6B), 3 seeds | 0.3320 ± 0.0052 | 0.8653 ± 0.0027 | 0.3161 ± 0.0055 |
| E | Temporal GraphSAGE (Phase 7), 3 seeds | 0.4927 ± 0.0035 | 0.8667 ± 0.0009 | 0.3562 ± 0.0071 |
| E control | static GraphSAGE, schedule-matched (Phase 7A) | 0.4792 ± 0.0114 | 0.8554 ± 0.0045 | 0.3288 ± 0.0050 |
| E control | temporal, memory ablated (Phase 7A) | 0.4790 ± 0.0076 | 0.8571 ± 0.0054 | 0.3305 ± 0.0065 |
| **B** | **tabular + graph features** | **MISSING** | | |

F rows are on a **distinct basis** (mean of five window PR-AUCs; shared checkpoint trained on days 1-111; threshold from TRAIN days 112-121; the adaptive arm updates on already-scored VAL labels) and are **not comparable with A, C, D or E**:

| F arm (3 seeds) | PR-AUC (mean of 5 windows) | ROC-AUC | PR-AUC cold-start |
|---|---|---|---|
| frozen | 0.4243 ± 0.0128 | 0.8525 ± 0.0134 | 0.3544 ± 0.0103 |
| adaptive | 0.4580 ± 0.0082 | 0.8570 ± 0.0072 | 0.3723 ± 0.0023 |
| replay control | 0.4179 ± 0.0063 | 0.8347 ± 0.0149 | 0.3584 ± 0.0053 |

Pre-registered comparability labels (Phase 12 scope record, section 5): A versus C/D/E is comparable on the evaluation basis only (attribution not supported); C versus its MLP control is comparable with limits; C versus D is not comparable (D never converged); C versus E is not a single-factor comparison (training regime differs); the Phase 7A chain and the three F arms are comparable within themselves; F versus anything else is not comparable. "Comparable" never implies that a difference is attributable to the factor named. The source is `experiments/phase12_validation_evidence_table/outputs/table_report.md`.

## 2. H1-H5 verdicts

The verdict wording is this report's synthesis. The "documented basis" column restates what the cited reports say; nothing is upgraded beyond them. All rows are VAL only.

| Hypothesis | Verdict | Documented basis | Limitations carried from the reports |
|---|---|---|---|
| **H1** graph structure adds information beyond transaction-level features | **NOT ESTABLISHED.** A narrow positive exists against a no-relations control only. | GraphSAGE 0.4637 ± 0.0167 versus the MLP control 0.4167 (Phases 5, 6A, 12). No graph model reaches XGBoost (0.5395); the best, Temporal GraphSAGE, has 0.4927 pooled and 0.3562 cold-start against XGBoost's 0.4064. Phases 5 and 6A: relational structure helps relative to no structure, which is different from beating a mature tabular method. | The direct test, B versus A, does not exist. The control is one run at its epoch cap. XGBoost is a single run tuned on VAL. |
| **H2** learned graph representations beat handcrafted graph features | **NOT TESTED** | B is missing (section 1; Phase 12 table). | No statement about H2 is available. |
| **H3** temporal information improves generalization to future transactions over static graph modelling | **PARTIALLY SUPPORTED, modest.** Pooled temporal-versus-static is not established. Future-time generalization is untested. | Phase 7: +0.029 pooled over static GraphSAGE in 3 of 3 seeds, but confounded by training regime (about 17 optimizer steps per epoch versus 1) and the report does not claim causation. Phase 7A: temporal beats its memory-ablated control in 3 of 3 seeds (about +0.0137 pooled, exceeding both arms' seed SDs) and beats the schedule-matched static model in 2 of 3 seeds (mean about +0.0135, not established by the pre-registered rule). The cold-start gain over the static model was 3 of 3 seeds (about +0.027). Modest, concentrated in cold-start. | n = 3 seeds. Descriptive p-values only. The memory-ablated control has about 9% fewer parameters. One alternative memory was tested. TEST was not used, so no future-time claim is supported. |
| **H4** adaptive updating reduces drift damage | **SUPPORTED under the pre-registered rule** (Phase 8, VAL window protocol) | All three locked rules pass: adaptive degrades less than frozen in 3 of 3 seeds; mean difference +0.0338 exceeds the larger seed SD 0.0128; adaptive beats the replay control by +0.0402 (above 0.0082). Window means 0.4580 versus 0.4243 (frozen) and 0.4179 (replay). | n = 3; margins not large; drift is observed only inside the 30-day VAL span; zero label delay assumed; the adaptive arm uses VAL labels after scoring; the F basis is not comparable with A-E. |
| **H5** graph analysis identifies suspicious coordinated communities | **NOT ESTABLISHED** (H5a, H5b, H5c) | Phase 9 closure report: the synthetic benchmark failed frozen validity gate G4 (seeds 42 and 2024; passed on 123), so the quantitative test was never run. Real-data Tier 0 and E0 are exploratory and descriptive: no verdict. Not evidence that the hypotheses are false. | E0 specification written after Tier 0 results existed; null tail proportions are descriptive references, not exact p-values; seeds are not independent replications; 82.9% of VAL transactions carry no netproxy or device edge; no ground-truth coordination labels exist. |

D (GAT) note: all three seeds reached the epoch cap, so "GAT performs worse" and "GAT is under-trained" are not separated. Phase 6C did not resolve this and no claim of either kind is supported.

## 3. Comparison B

**B (tabular + graph features) is NOT TESTED / MISSING.** No artifact exists. It is recorded as MISSING in the Phase 12 table and in its scope record and is not replaced by the historical-feature baseline. Consequence: H2 cannot be assessed, and H1 lacks its most direct test. Producing B would need new graph-feature engineering and training, which has not been authorized (section 8).

## 4. Phase 11 outputs, Phase 10 status

**candidate coordinated community; not a confirmed fraud ring.** This does not establish that IEEE-CIS contains fraud rings. Nothing in these outputs is evidence of fraud, causality or coordination.

**Phase 11 first pass (closure report `experiments/phase11_explainable_evidence/PHASE11_CLOSURE_REPORT.md`):**
* XGBoost per-transaction explanations for a pre-registered, label-free VAL selection of 300 transactions (30 days × 5 highest-scoring + 5 seeded uniform draws): native contributions for all 470 features, additivity to the margin (maximum absolute error 8.1e-06), top-10 features per transaction.
* Aggregate-only investigation cards for Phase 9 Tier 0 candidate communities: 24 cards for seed 42 (window W = 7), plus selection tables for seeds 123 and 2024.
* **VAL caveat (from the report):** VAL labels shaped XGBoost's iteration count (798) and threshold, so these explanations are descriptive, not held-out. The 300-transaction selection is small and pre-registered; nothing generalizes to the VAL population or to TEST. Tree SHAP contributions are additive in log-odds, shared across correlated features, and do not identify causes; features are anonymised.
* Cards are aggregate-only. Member identities, descriptive attributes, label-based columns, risk percentile and a per-community null match are unavailable and not fabricated. E0 appears only as a day-level descriptive reference.
* Deferred by decision: graph evidence (the temporal checkpoint was not run on real data). No dashboard or API exists.
* The report records its own verification: A1-A10 pass, 30 of 30 tests, 14 of 14 injected defects detected.
* Transaction-level outputs are derived data; whether to publish them is the user's decision (the existing ignore rules do not cover them).

**Phase 10** (`experiments/phase10_community_evolution/`): design documents with approved decisions and two errata, plus the core module `src/evolution/tracker.py` and its tests (63 of 63 passing per `core_implementation/core_test_results.json`). No empirical tier, Stage 0 or real-data run was authorized or performed; numeric thresholds were not chosen.

## 5. Phase 12 status

* The consolidated evidence table is built (`experiments/phase12_validation_evidence_table/outputs/`): 165 pre-registered source files, one metric implementation (`src/models/metrics.py`, unmodified), recorded thresholds only, B kept MISSING.
* Recorded verification (closure report, `test_results/`): recomputed values agree with the recorded metrics for 65 run/window files, 8,184 values, maximum absolute difference 0.0; 173 cross-check items, 0 discrepancies; population identity across 83,655 rows; byte-identical rebuild; 27 of 27 tests; 17 of 17 injected defects detected.
* AC1-AC11 are recorded as PASS only to the extent supported by the evidence named in each row of the closure report (limits in its section 3.2).
* **AC12 (before/after SHA-256 of every pre-existing file): BLOCKED — Partially verified. Authentic pre-build evidence covers 300 of 651 pre-existing files, with zero contradictions. The remaining 322 files cannot be retrospectively verified because no trustworthy pre-build hashes survive. AC12 is not passed.**
* **Phase 12 is not formally complete under its registered criteria.** The registered acceptance criteria were not changed.

## 6. TEST

**No TEST evaluation has been performed.** Per the phase reports and the Phase 9-12 records, TEST (days 152-182) was never loaded by any training, selection or scoring script, and no label from it was used. TEST-named data files exist on disk; their contents were not opened for this report. The Kaggle competition test files carry no labels (Phase 1). Therefore **the results support validation-period claims only.**

## 7. Reproduction guide

This is a map of existing files. Script-to-phase assignments follow the file names and the reports; the scripts were **not re-executed** for this report. Training scripts write into existing experiment folders and need the GPU environment recorded in the reports; running one over a frozen folder would overwrite it (this is a caution, not a plan). `scripts/train_gat_convergence_300epoch.py` is the aborted Phase 6C script and must not be run (its report says so). All commands use `.venv/Scripts/python.exe -B`. `scripts/run_tests.py` runs every `tests/test_*.py` module without pytest.

| Phase | Report / record | Scripts | Tests and validation | Integrity evidence |
|---|---|---|---|---|
| 1-2 | `reports/phase1_reconnaissance.md`, `phase2_eda_report.md` | `run_eda.py`, `build_processed_data.py` | `tests/test_preprocessing.py`, `test_splits.py`; `run_validation.py` (dataset-level checks) | none of the Phase 9-12 type found |
| 3 | `reports/phase3_baseline_report.md` | `train_logistic_regression.py`, `train_random_forest.py`, `train_xgboost.py`, `train_xgboost_historical_control.py`, `build_historical_features.py` | no phase-specific test script identified | none of the Phase 9-12 type found |
| 4 | `reports/phase4_graph_report.md` | `estimate_graph_schema.py`, `build_graph.py`, `validate_graph.py`, `graph_feasibility_benchmark.py` | `tests/test_validate.py`; graph validation (8 of 8 checks per the report) | none of the Phase 9-12 type found |
| 5-6 | `phase5_graphsage_report.md`, `phase6a_graphsage_robustness_report.md`, `phase6b_gat_report.md`, `phase6c_gat_convergence_status.md` | `train_graphsage.py`, `train_graphsage_control_mlp*.py`, `train_graphsage_robustness.py`, `train_gat_robustness.py`, `graphsage_stratified_corrected.py`, `stratified_comparison.py` | no phase-specific test script identified | per-run `config.json`, `metrics_val.json`, `summary.json` under `experiments/` |
| 7, 7A | `phase7_temporal_gnn_report.md`, `phase7a_temporal_ablation_report.md` | `train_temporal_graphsage.py`, `train_static_schedule_matched.py`, `train_temporal_mean_memory.py`, `compare_static_temporal.py`, `compare_phase7a.py`, `phase7a_common.py` | `validate_temporal_gnn.py` (19 of 19 per Phase 7 report), `validate_temporal_ablation.py`; leakage outputs in `experiments/temporal_graphsage/leakage_validation/` | per-run files; `comparison.json` |
| 8 | `phase8_concept_drift_report.md`, `experiments/phase8_adaptive_drift/DESIGN_BEFORE_EXECUTION.md` | `train_phase8_init.py`, `train_phase8_frozen.py`, `train_phase8_adaptive.py`, `train_phase8_replay_control.py`, `measure_phase8_drift.py`, `compare_phase8.py` | `tests/test_phase8_adaptive.py`, `validate_phase8_leakage.py` (90 of 90 per the report) | `comparison.json`; leakage validation folder |
| 9 | `reports/phase9_closure_report.md`; `experiments/phase9_coordinated_fraud/` | `build_phase9_benchmark.py`, `run_phase9_g0.py`, `..._timing_pilot.py`, `..._calibration.py`, `..._dev_selection.py`, `..._benchmark_gates.py`, `..._risk.py`, `..._tier0.py`, `..._e0_tau.py`, `..._e0.py`, `make_phase9_exploratory_report.py` | `tests/test_phase9_core.py`, `test_phase9_segments.py`, `validate_phase9_e0.py`, `validate_phase9_leakage.py` | `phase9_closure_integrity.json` and `.sha256` (34 artifacts); `exploratory_realdata/MANIFEST*.json` |
| 10 | `experiments/phase10_community_evolution/` (design, addenda, errata) | `run_phase10_tests.py`, `check_phase10_test_sensitivity.py` | `tests/test_phase10_*.py` (63 of 63; 18 of 18 injected defects per the recorded results) | `INTEGRITY*.json` with `.sha256` sidecars (six records) |
| 11 | `experiments/phase11_explainable_evidence/PHASE11_CLOSURE_REPORT.md`, `SCOPE_AND_ACCEPTANCE.md` | `run_phase11_explanations.py`, `run_phase11_cards.py`, `run_phase11_tests.py`, `check_phase11_test_sensitivity.py` | `tests/test_phase11_explain.py`, `test_phase11_outputs.py` (30 of 30; 14 of 14 injected defects) | `INTEGRITY_PHASE11.json` and `.sha256`; `manifest_val.json`, `manifest_cards.json` |
| 12 | `experiments/phase12_validation_evidence_table/PHASE12_CLOSURE_REPORT.md`, `SCOPE_AND_ACCEPTANCE.md` | `run_phase12_table.py`, `run_phase12_tests.py`, `check_phase12_test_sensitivity.py`, `finalize_phase12_integrity.py` | `tests/test_phase12_evidence.py`, `test_phase12_outputs.py` (27 of 27; 17 of 17 injected defects) | `INTEGRITY_PHASE12.json` and `.sha256` (30 files); `outputs/manifest_phase12.json`; `PRECHANGE_MANIFEST_closure_update.tsv`; **AC12 BLOCKED** |

Phase 12 reproduction (from the closure report):

```
.venv/Scripts/python.exe -B scripts/run_phase12_table.py
.venv/Scripts/python.exe -B scripts/run_phase12_tests.py
.venv/Scripts/python.exe -B scripts/check_phase12_test_sensitivity.py
```

The build is recorded as deterministic (byte-identical on one rebuild into a temporary folder). The test scripts overwrite the recorded test results, which must then be re-hashed by `scripts/finalize_phase12_integrity.py`.

**What the existing evidence supports about integrity.** The Phase 9, 10, 11 and 12 integrity records list per-file SHA-256 hashes and were checked read-only against the current files on 2026-10-19 with 0 mismatches (Phase 9: 34 artifacts; Phase 12: 30 files, with the closure report at `23b8e04f8cd1bbb043bb8dea6a69ec91056a0ca0ead2c95c4907d017a988d389` and the scope record at `d854aad556dacec44b7a7f783ef21966239e8ae5b2d95caca1e84a99ed956d4c`). Those checks were run in a working session; they are not recorded in a project file other than the Phase 12 closure report and records. For Phases 1-8 I found no phase-level hash integrity record, so no hash-verification claim is made for them, and no claim is made that they are unchanged since they ran. The Phase 12 AC12 limitation (322 of 651 pre-existing files without authentic pre-build hashes) applies to that history.

## 8. Limitations and future work

### 8.1 Limitations (documented)
* VAL only; every graph, tabular and temporal model except the F arms was tuned on VAL; F's adaptive arm uses VAL labels after scoring. No held-out comparison, significance test or causal statement exists.
* A has one run per model; C control has one run at its cap; D never converged; n = 3 seeds elsewhere with descriptive p-values only; seeds do not replicate the real data.
* F is not comparable with A, C, D or E.
* Zero label delay is assumed in Phase 8. Drift is observed only within a 30-day span.
* The dataset has no ground truth for coordinated groups; communities are candidates only. Infrastructure fields are project-defined proxies, and 82.9% of VAL transactions have no infrastructure edge.
* AC12 is BLOCKED: 322 of 651 pre-existing files cannot be retrospectively verified.
* Process note: a dry-run build was once written to a stray `C:\c\Users\...` folder outside the project; its files were deleted and 9 empty folders remain there for the user to remove.

### 8.2 Essential unresolved claims (documented as open; each would need its own explicit authorization)
* **H2:** NOT TESTED until B exists. B also bears on H1 through the B-versus-A comparison.
* **Generalization:** no TEST evaluation exists, so no future-time or held-out claim (the brief's primary evaluation) is supported. A TEST evaluation, if separately authorized, would have to be one-shot and pre-registered; nothing in this report authorizes or specifies one.
* **H5a-H5c:** not established because the benchmark failed gate G4; any new benchmark design would need new frozen decisions (Phase 9 closure report).
* **D versus C:** "worse" versus "under-trained" is unresolved.
* **Pooled temporal-versus-static (H3):** not established at n = 3 (Phase 7A report).

### 8.3 Optional extensions (not needed to state the findings above)
Relationship-level ablations; Phase 10 empirical tiers; graph evidence for explanations (deferred in Phase 11); E1 label-based supporting evidence (Phase 9 future work); logistic-regression coefficient contributions; a dashboard or API (the brief says not to add these unless approved).

### 8.4 Recommendations (this report's judgement, not documented findings)
* Cite this report with its banner: VAL-period, descriptive, no TEST.
* Lead any temporal claim with the cold-start result, as the Phase 7A report recommends.
* Keep B listed as an uncompleted comparison and H2 as not tested unless and until B is separately authorized and built.
