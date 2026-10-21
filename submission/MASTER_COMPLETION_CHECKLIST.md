# Master completion checklist

Dated 2026-10-21. Status words: **PASS** verified by evidence; **FAIL** a verified requirement is not satisfied; **BLOCKED** cannot be completed without a missing decision, authorization or evidence; **NOT RUN** not executed; **OPTIONAL** not required for the declared scope. Sources of requirements: the original brief (the first message of the earliest saved session; not a repository file), the user's staged instructions, and the phase scope records. Where a requirement cannot be verified from the available source material it says so. `COMPLETION_MATRIX.md` and `FINAL_COMPLETION_RECORD.md` carry the phase-by-phase ledgers; this file is the consolidated checklist.

## 1. Hypotheses (scientific outcome, not software status)

| Item | Status | Evidence / outcome | Next action | Safe now? |
|---|---|---|---|---|
| H1 graph structure beyond transaction features | PASS (executed); **not established** | comparison B (`experiments/comparison_B_graph_features/`): pre-registered rule not met | none | n/a |
| H2 learned graphs beyond handcrafted graph features | PASS (executed); **contradicted by the rule** | same | none | n/a |
| H3 temporal information helps future transactions | PASS (executed); partially supported, modest | Phase 7 / 7A; future-time wording untested | held-out reading is BLOCKED (below) | no |
| H4 adaptive updating reduces drift degradation | PASS on validation (supported, n = 3); also supported with the entity memory built first (`phase8_state_continuity_supplement`) | `comparison.json`, `decision_compare.json` | held-out reading: Amendment 1 module F, BLOCKED on authorization | no |
| H5 coordinated communities | **FAIL** (frozen gate G4); not established | G4 mean absolute change 0.7954, 0.4622, 0.7994 against 0.5 (seeds 42, 123, 2024); recorded in `phase9_closure_report.md`, hash of the locked configuration matches its sidecar | needs a new benchmark design and new frozen decisions | no (user decision) |

## 2. Models, controls, ablations

| Item | Status | Evidence | Next action |
|---|---|---|---|
| Logistic regression, random forest, XGBoost; historical-feature reference | PASS | Phase 3 | none |
| XGBoost + graph features (B), 13 arms | PASS | comparison B tests and outputs | none |
| GraphSAGE, GAT, Temporal GraphSAGE, Phase 7A controls | PASS (executed); **GAT convergence unresolved** | Phases 5-7A | none; GAT result is reported as non-converged |
| GAT 300 epochs | **FAIL / aborted**, excluded | `phase6c_gat_convergence_status.md` | none (not a conclusion) |
| Class imbalance experiment (XGBoost, GraphSAGE, Temporal) | PASS | two pre-registrations; weighting hurts XGBoost (0.5395 vs 0.5948) | GAT unweighted NOT RUN (OPTIONAL) |
| Relationship-level ablations (XGBoost family) | PASS | comparison B | none |
| Relationship-level ablations (neural models) | NOT RUN; OPTIONAL ("where computationally feasible") | no registered hypothesis depends on them | needs its own pre-registration and your decision |
| Adaptive system F (cell F) | PASS on validation | Phase 8 | held-out: BLOCKED |
| Drift-**triggered** refresh (the brief: "monitor drift/performance; trigger retraining or model refresh when predefined conditions are met", threshold from development data) | **PASS (both executed, pre-registered, validation only); outcomes NOT SUPPORTED.** Phase 8 used a fixed cadence. Arm Tr (label-free score-PSI monitor, threshold from TRAIN days only): never fired, equals the frozen arm. Arm TrP (performance monitor on each window's PR-AUC, threshold from TRAIN days only): fired once per seed, beat the frozen arm (+0.0169, above the larger seed SD 0.0128) with 1 refresh instead of 4, but stayed below periodic refresh (mean Ad - TrP +0.0240 against a seed SD of 0.0096), so the parity criterion TRP5 is NOT MET | `experiments/phase8_triggered_refresh/` (two pre-registrations hashed before their runs, `TR_RESULTS.md`, `TRP_RESULTS.md`, tests) | none; the brief's requirement is to implement and evaluate, which is done; H4 as locked is unchanged |

## 3. Temporal evaluation, leakage, held-out safeguards

| Item | Status | Evidence |
|---|---|---|
| Chronological split, preprocessing and thresholds without the final period | PASS | Phase 2, split records |
| Leakage prevention (feature information time <= prediction time; causal graph masks; label projection in the evaluator) | PASS | Phase 4/7/8 checks, `test_graph_features`, `test_heldout_gate`, `test_inference_demo` |
| Held-out (days 152-182) evaluation | **BLOCKED** | protocol hash `59bf1565...c806` (unchanged); Amendment 1 (modules F, H, U) frozen and hashed; `AUTHORIZATION.txt` and all `heldout_results*` folders do not exist |
| Amendment 1 readiness on validation | H, U **PASS** (9 models); base 19 models **PASS** (max difference 9.54e-07); F: 21 deterministic checks PASS, trained-arm window-mean gate **FAIL as written** (1 of 6: E seed 123, 0.0104 against the registered 0.01; not changed) | `readiness_val*`, `readiness_F.json` |
| Cause of the F gate miss | PASS (diagnosed): **GPU nondeterminism**, not a code defect | three identical-seed GPU runs differ (window means 0.4330, 0.4273, 0.4281; first window bitwise equal; later windows differ in single scores by up to about 0.97); two identical-seed CPU runs are bitwise equal; five-run SD of the window mean 0.0044; `diagnostics/F_readiness_diagnosis_E_seed123*.json` |
| Next action for F | decision for you: authorize F knowing the gate failed as written, or freeze a different (distribution-based) gate first (requires an amendment) | | |

## 4. Reproducibility and integrity

| Item | Status | Evidence / limit |
|---|---|---|
| Seeds, configs, split boundaries, metrics saved | PASS | per-experiment `config.json` / `metadata.json` |
| Phase 12 AC12 (before/after hash of every pre-existing file) | **BLOCKED** | 651 pre-existing files: 300 verified with zero contradictions, 29 self-consistent sidecars, 322 with no authentic pre-build hash; nothing can be reconstructed and nothing is fabricated |
| Forward integrity record for the completion work | PASS | `INTEGRITY_SUBMISSION.json` (refreshed at the end of this pass) |
| NE5(a) of the neural graph evidence | **FAIL as registered** (13 of 102 never-seen resets nonzero, largest 9.5e-07); diagnosed as GPU floating-point noise (CPU exact zero); rule and result unchanged | `neural_graph_evidence/diagnostics/NE5_diagnosis.json` |
| Phase 8 memory initialisation | disclosed; supplement leaves H4 SUPPORTED; its own reproduction criterion S1 FAILED for the trained arms (not bit-reproducible) and was not changed | `phase8_state_continuity_supplement/` |
| AC12 forward integrity (added 2026-10-21; **does not change AC12**) | PASS as a forward fact only | none of the 434 files of the first Git commit has changed in content since (270 differ only in line endings after the byte-exact re-add; none deleted; `git diff --ignore-cr-at-eol e70238f HEAD`); the first commit was made after the work, so this is not pre-build evidence and AC12 remains BLOCKED |

## 5. Automated tests and environments

| Check | Result |
|---|---|
| Project environment, `scripts/run_completion_tests.py` | **325 of 325** pass in 33 modules (`submission/test_results.json`) |
| `scripts/run_tests.py` | exit 0, "ALL TEST MODULES PASSED" (before the fix it reported failures because two new modules returned an exit code instead of a boolean; `tests/test_runner_contract.py` now guards this and injects a failure to prove detection) |
| Clean Python 3.12.0 (requirements.txt only) | **275 of 275** executed tests pass with the current suite (earlier suite: 257 of 257); `pip check` clean |
| Clean Python 3.14.3 (requirements.txt only) | **274 of 275** pass + **1 environment-metadata difference** (earlier suite: 256 of 257) (Phase 12 AC10 manifest records the interpreter version; the environment-independent companion test passes); `pip check` clean; **supported interpreter: 3.12** |
| Fresh GitHub clone, Python 3.12.0, requirements.txt only | before the workflow: 246 of 259 pass, 13 fail only for withheld artifacts; **with the core-versus-artifact workflow (final measurement): 262 of 275 pass, 13 NOT RUN (artifact missing, listed in `tests/artifact_requirements.json`, never counted as passed), 0 fail**; `--require-artifacts` exits 1 on a clone; protocol hash verified in the clone |
| Skipped (needs PyTorch; **a skip is not a pass**) | `test_f_runner_gate`, `test_neural_graph_evidence`, `test_phase8_adaptive`, `test_trigger_walkforward`, `test_walkforward_ext` in a core-only environment; they pass in the project environment |

## 6. Phase 13 (reconstructed scope; unconfirmed by you)

| Item | Status |
|---|---|
| Dashboard + API behaviour, error handling, shutdown | PASS: `scripts/verify_dashboard_e2e.py` 31 of 31 checks, exit 0; `tests/test_serving.py` |
| Scope | **BLOCKED on your confirmation**: the original Phase 13 specification was never recovered |
| Mismatches / assumptions against the original brief | (1) the brief suggests `api/`, `dashboard/`, `notebooks/`, `models/` folders; the implementation lives in `src/serving/` and `scripts/`, no notebooks exist; (2) FastAPI and Streamlit are "likely technologies" in the brief; the standard library is used (minimum-stack rule); (3) "production-style inference" is the scorer plus the endpoint, but the scoring bundle is a synthetic demo bundle; the research XGBoost is not served; (4) drift monitoring (PSI/KS) exists offline in Phase 8, not as a live dashboard feature; (5) whether Phase 13 meant these items at all is unverifiable |

## 7. Explanations

| Item | Status |
|---|---|
| Tabular explanations: native XGBoost tree contributions (TreeSHAP values via the library; the `shap` package is not installed or used) | PASS (first pass) |
| Evidence for graph-feature model B | PASS |
| Neural graph evidence | PASS for Temporal GraphSAGE seed 42 (NE5(a) failed, see section 4); GraphSAGE and GAT explanations NOT RUN (GAT never converged, attention not valid); OPTIONAL |

## 8. Setup, demo, documentation, presentation

| Item | Status | Evidence |
|---|---|---|
| Setup check, demo (byte-identical on re-run), dashboard | PASS | exit 0 for `check_setup.py` and `demo_end_to_end.py` (twice, outputs identical); dashboard E2E 31 of 31 |
| README, demo guide, completion records agree | PASS after this pass | test counts corrected (325 tests, 33 modules), PyTorch-skip statement corrected, clean-environment results recorded; `scripts/verify_submission_claims.py` traces every number |
| Manuscript | PASS (validation-period study; held-out not done) | three figures added (`submission/figures/`, drawn from recorded results, hashed, tested) |
| Hygiene | PASS with notes | one empty folder `experiments/gat_300epoch/seed_123` kept (aborted-run evidence); `__pycache__` is git-ignored; nine empty `C:\c\...` folders outside the repository left in place |

## 8b. Decision package

The exact frozen protocol, the amendment, the options with their consequences, the Module F gate options and the other open decisions are in `HELDOUT_DECISION_PACKAGE.md`. Nothing was authorized.

## 9. What remains before the research can be called complete

1. Held-out evaluation: your authorization (base protocol, or base + Amendment 1 modules F, H, U). **BLOCKED.**
2. F readiness gate failed as written (diagnosed as GPU noise): your acceptance or a new frozen gate. **BLOCKED on a decision.**
3. Phase 13 scope confirmation. **BLOCKED.**
4. H5 / G4: new design needed. **BLOCKED.**
5. AC12: cannot be completed retrospectively. **BLOCKED.**
6. Drift-triggered refresh: **both the score-PSI trigger and the performance trigger were executed (pre-registered, validation only); both outcomes NOT SUPPORTED** (see section 2). Neural relationship ablations, GAT unweighted, GraphSAGE/GAT explanations: **NOT RUN / OPTIONAL.**

**The project is not 100% complete.** Software, tests, demo, documentation and the validation-period research are complete and verified; the items above are genuinely unresolved.
