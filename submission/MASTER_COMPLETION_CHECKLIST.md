# Master completion checklist

Dated 2026-10-21; updated 2026-10-22 (decision-preparation pass). Status words: **PASS** verified by evidence; **FAIL** a verified requirement is not satisfied; **BLOCKED** cannot be completed without a missing decision, authorization or evidence; **NOT RUN** not executed; **OPTIONAL** not required for the declared scope. Sources of requirements: the original brief (the first message of the earliest saved session; not a repository file), the user's staged instructions, and the phase scope records. Where a requirement cannot be verified from the available source material it says so. `COMPLETION_MATRIX.md` and `FINAL_COMPLETION_RECORD.md` carry the phase-by-phase ledgers; this file is the consolidated checklist.

## 1. Hypotheses (scientific outcome, not software status)

| Item | Status | Evidence / outcome | Next action | Safe now? |
|---|---|---|---|---|
| H1 graph structure beyond transaction features | PASS (executed); **not established** | comparison B (`experiments/comparison_B_graph_features/`): pre-registered rule not met | none | n/a |
| H2 learned graphs beyond handcrafted graph features | PASS (executed); **contradicted by the rule** | same | none | n/a |
| H3 temporal information helps future transactions | PASS (executed); partially supported, modest | Phase 7 / 7A; future-time wording untested | held-out reading is BLOCKED (below) | no |
| H4 adaptive updating reduces drift degradation | PASS on validation (supported, n = 3); also supported with the entity memory built first (`phase8_state_continuity_supplement`) | `comparison.json`, `decision_compare.json` | held-out reading: Amendment 1 module F, BLOCKED on authorization | no |
| H5 coordinated communities | **FAIL** under the original protocol (frozen gate G4), kept; **H5a-9b SUPPORTED** on a new, separately pre-registered benchmark | original: G4 0.7954, 0.4622, 0.7994 against 0.5; diagnosis: the formula counts planted rows in the ranking population (model interference itself 0.03-0.07). Phase 9b: fresh seeds, corrected gate (0.050, 0.065, 0.042), A2 alpha = 0 AUPRC 0.145, 0.267, 0.141, beats each baseline 3 of 3 seeds, G-cal 1.73, 1.37, 1.93 per day; `experiments/phase9b_recovery_benchmark/` | none (H5b/H5c not evaluated; optional) | n/a |

## 2. Models, controls, ablations

| Item | Status | Evidence | Next action |
|---|---|---|---|
| Logistic regression, random forest, XGBoost; historical-feature reference | PASS | Phase 3 | none |
| XGBoost + graph features (B), 13 arms | PASS | comparison B tests and outputs | none |
| GraphSAGE, GAT, Temporal GraphSAGE, Phase 7A controls | PASS (executed); **GAT convergence unresolved** | Phases 5-7A; code review 2026-10-22 found no GAT defect (same width, depth, dropout and optimizer as GraphSAGE); the seed-42 curve gained 0.0087 PR-AUC over its last 30 epochs (about 0.33, against GraphSAGE's 0.46) | none; the 300-epoch rerun stays aborted on the owner's earlier instruction |
| GAT 300 epochs | **FAIL / aborted**, excluded | `phase6c_gat_convergence_status.md` | none (not a conclusion) |
| Class imbalance experiment (XGBoost, GraphSAGE, Temporal) | PASS | two pre-registrations; weighting hurts XGBoost (0.5395 vs 0.5948) | GAT unweighted NOT RUN (OPTIONAL) |
| Relationship-level ablations (XGBoost family) | PASS | comparison B | none |
| Relationship-level ablations (neural models) | **PASS (executed, pre-registered)**: the brief's "where computationally feasible" applies (about 2 minutes per Temporal GraphSAGE run) | `experiments/relation_ablation_temporal/` (12 arms + a seed-42 noise rerun): pseudo-account contributes (-0.0309, 3 of 3 seeds), device contributes (-0.0073, 3 of 3; close to the 0.0058 run-to-run noise), card and network proxy: removing them does not hurt | none; address and email are not in the graph (Phase 4) |
| Adaptive system F (cell F) | PASS on validation | Phase 8 | held-out: BLOCKED |
| Drift-**triggered** refresh (the brief: "monitor drift/performance; trigger retraining or model refresh when predefined conditions are met", threshold from development data) | **PASS (both executed, pre-registered, validation only); outcomes NOT SUPPORTED.** Phase 8 used a fixed cadence. Arm Tr (label-free score-PSI monitor, threshold from TRAIN days only): never fired, equals the frozen arm. Arm TrP (performance monitor on each window's PR-AUC, threshold from TRAIN days only): fired once per seed, beat the frozen arm (+0.0169, above the larger seed SD 0.0128) with 1 refresh instead of 4, but stayed below periodic refresh (mean Ad - TrP +0.0240 against a seed SD of 0.0096), so the parity criterion TRP5 is NOT MET | `experiments/phase8_triggered_refresh/` (two pre-registrations hashed before their runs, `TR_RESULTS.md`, `TRP_RESULTS.md`, tests) | none; the brief's requirement is to implement and evaluate, which is done; H4 as locked is unchanged |

## 3. Temporal evaluation, leakage, held-out safeguards

| Item | Status | Evidence |
|---|---|---|
| Chronological split, preprocessing and thresholds without the final period | PASS | Phase 2, split records |
| Leakage prevention (feature information time <= prediction time; causal graph masks; label projection in the evaluator) | PASS | Phase 4/7/8 checks, `test_graph_features`, `test_heldout_gate`, `test_inference_demo` |
| Held-out (days 152-182) evaluation | **BLOCKED** | protocol hash `59bf1565...c806` (unchanged); Amendment 1 (modules F, H, U) frozen and hashed; `AUTHORIZATION.txt` and all `heldout_results*` folders do not exist |
| Readiness on validation (re-run 2026-10-22 after the guard changes) | base 19 models **PASS** (max difference 7.15e-07); H, U **PASS** (9 models; neural at most 1.8e-07); F: 21 deterministic checks PASS, trained-arm window-mean gate **FAIL as written** (1 of 6: E seed 123, 0.0104 against the registered 0.01; not changed) | `readiness_val*`, `readiness_F.json`; earlier records archived under `previous_before_*` |
| Readiness guards in the held-out runners (2026-10-22) | **PASS**: every runner refuses before any read unless its registered gate passed and the code is unchanged since (SHA-256 fingerprint of the 21 and 23 code files each gate ran; content-based, so a fresh clone is judged the same way); Module F is refused today; validation runs and diagnosis records can no longer be overwritten | `src/heldout/codehash.py`, `tests/test_readiness_guards.py` (10 tests), `tests/test_f_runner_gate.py` (4) |
| Cause of the F gate miss | PASS (diagnosed): **GPU nondeterminism**, not a code defect | three identical-seed GPU runs differ (window means 0.4330, 0.4273, 0.4281; first window bitwise equal; later windows differ in single scores by up to about 0.97); two identical-seed CPU runs are bitwise equal; five-run SD of the window mean 0.0044; `diagnostics/F_readiness_diagnosis_E_seed123*.json` |
| Proposed F gate A2-a (2026-10-21) | **calibration error found 2026-10-22**: its SD comes from 5 runs, so the false-failure rate is 4.0 % per cell and 21.7 % over six (t, 4 df), not the stated 0.3 % / 1.6 %; the file is kept unchanged | `AMENDMENT_2_PROPOSAL_F_GATE.md` |
| Next action for F | **decision for you**: approve the corrected, inactive revision **A2-a-r1** (10 runs per cell, t with 9 df, 1.6 % over six cells; SHA-256 `37ca86ee...56eb`), or exclude F. Authorizing F with the gate failed as written is withdrawn: it conflicts with your rule that a run may proceed only after its gate passed or was validly amended | `AMENDMENT_2_PROPOSAL_F_GATE_REV1.md`, `HELDOUT_DECISION_PACKAGE.md` section 3 |

## 4. Reproducibility and integrity

| Item | Status | Evidence / limit |
|---|---|---|
| Seeds, configs, split boundaries, metrics saved | PASS | per-experiment `config.json` / `metadata.json` |
| Phase 12 AC12 (before/after hash of every pre-existing file) | **BLOCKED** (registered method impossible) | per-file recovery 2026-10-22 (`AC12_GIT_SNAPSHOT_VERIFICATION.json`, `scripts/verify_ac12_git_snapshot.py`): of 651 files, 434 content-identical to the first Git commit (pushed 2026-10-16 per GitHub's server log, before the 2026-10-19 build), 65 covered by the scope fingerprint, 13 by hashes in pre-build session logs, 38 only by project integrity records; **0 contradictions; 101 with no pre-build evidence** (32 model files, 19 data files incl. 3 TEST-named never opened, 18 Phase 9 arrays, 3 row-level predictions, 29 others). Derived consistency (not provenance, not counted; `AC12_DERIVED_EVIDENCE.json`): 20 sidecars match companions with pre-build evidence, 18 model files reproduce pre-build predictions through the readiness gates; 63 have no evidence of any kind |
| Forward integrity record for the completion work | PASS | `INTEGRITY_SUBMISSION.json` (refreshed at the end of this pass) |
| NE5(a) of the neural graph evidence | **FAIL as registered** (13 of 102 never-seen resets nonzero, largest 9.5e-07); diagnosed as GPU floating-point noise (CPU exact zero); rule and result unchanged | `neural_graph_evidence/diagnostics/NE5_diagnosis.json` |
| Phase 8 memory initialisation | disclosed; supplement leaves H4 SUPPORTED; its own reproduction criterion S1 FAILED for the trained arms (not bit-reproducible) and was not changed | `phase8_state_continuity_supplement/` |
| Publication of transaction-level files (found 2026-10-22: the repository is public) | FIXED at the branch tip; history remains | six files with per-transaction contributions, counterfactual effects or feature values of 300 validation transactions had been committed (`b28a03b`, `edb6167`); they are now untracked (kept locally; tests that read them registered as artifact-dependent); **they remain in the public Git history** (21 of 25 commits; a raw URL at `de16aa0` still returns HTTP 200). Competition rule 7B prohibits making Competition Data available to non-participants and requires notifying Kaggle. A verified backup bundle of the full history exists outside the repository. Row-level prediction files are now excluded locally (`.git/info/exclude`; `.gitignore` unchanged) and a test fails if one is ever tracked. Making the repository private, notifying Kaggle, or rewriting history: **your decision** (`HELDOUT_DECISION_PACKAGE.md` section 7) |

## 5. Automated tests and environments

| Check | Result |
|---|---|
| Project environment, `scripts/run_completion_tests.py` | **358 of 358** pass in 38 modules (`submission/test_results.json`, 2026-10-22); `run_core_tests.py --require-artifacts` 358 of 358 |
| `scripts/run_tests.py` | exit 0, "ALL TEST MODULES PASSED" (before the fix it reported failures because two new modules returned an exit code instead of a boolean; `tests/test_runner_contract.py` now guards this and injects a failure to prove detection) |
| Clean Python 3.12.0 (requirements.txt only) | **305 of 305** executed tests pass with the current suite (2026-10-22; earlier suites: 293 of 293, 257 of 257); `pip check` clean |
| Clean Python 3.14.3 (requirements.txt only) | **304 of 305** pass + **1 environment-metadata difference** (2026-10-22; earlier suites: 292 of 293, 256 of 257) (Phase 12 AC10 manifest records the interpreter version; the environment-independent companion test passes); `pip check` clean; **supported interpreter: 3.12** |
| Fresh GitHub clone, Python 3.12.0, requirements.txt only | before the workflow: 246 of 259 pass, 13 fail only for withheld artifacts; with the core-versus-artifact workflow (2026-10-21): 274 of 293 pass, 19 NOT RUN; **re-measured 2026-10-22 on a fresh clone of `6120da2`: 285 of 305 pass, 20 NOT RUN (artifact missing, listed in `tests/artifact_requirements.json`, never counted as passed), 0 fail**; `--require-artifacts` exits 1 on a clone; readiness-guard tests 10 of 10 in the clone; protocol hash verified in the clone |
| Skipped (needs PyTorch; **a skip is not a pass**) | `test_f_runner_gate`, `test_neural_graph_evidence`, `test_phase8_adaptive`, `test_trigger_walkforward`, `test_walkforward_ext` in a core-only environment; they pass in the project environment |

## 6. Phase 13 (reconstructed scope; unconfirmed by you)

| Item | Status |
|---|---|
| Research model served by the API (`/api/research/*`, the recorded comparison-A XGBoost) | PASS: HTTP scores equal direct inference and the recorded validation predictions; labels ignored; rows independent; live check 12 of 12; browser rendering checked |
| Dashboard + API behaviour, error handling, shutdown | PASS: `scripts/verify_dashboard_e2e.py` 31 of 31 checks, exit 0; `tests/test_serving.py` |
| Scope | **BLOCKED on your confirmation**: the original Phase 13 specification was never recovered; recovered requirements and reconstructed assumptions are separated in `PHASE13_SCOPE_CONFIRMATION.md` |
| Mismatches / assumptions against the original brief | (1) the brief suggests `api/`, `dashboard/`, `notebooks/`, `models/` folders; the implementation lives in `src/serving/` and `scripts/`, no notebooks exist; (2) FastAPI and Streamlit are "likely technologies" in the brief; the standard library is used (minimum-stack rule); (3) "production-style inference": the research XGBoost is served exactly as evaluated (row above); the synthetic demo bundle is a separate, labelled mode; the neural models are not served; (4) drift monitoring (PSI/KS) exists offline in Phase 8, not as a live dashboard feature; (5) whether Phase 13 meant these items at all is unverifiable |

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
| README, demo guide, completion records agree | PASS after this pass | test counts corrected (358 tests, 38 modules, 2026-10-22), PyTorch-skip statement corrected, clean-environment results recorded; `scripts/verify_submission_claims.py` traces every number |
| Manuscript | PASS (validation-period study; held-out not done) | three figures added (`submission/figures/`, drawn from recorded results, hashed, tested) |
| Hygiene | PASS with notes | one empty folder `experiments/gat_300epoch/seed_123` kept (aborted-run evidence); `__pycache__` is git-ignored; nine empty `C:\c\...` folders outside the repository left in place |

## 8a. Traceability and defense

`TRACEABILITY_MATRIX.md` maps each requirement of the original brief to code, a verification command, the observed result and the gap; `PROFESSOR_DEFENSE_GUIDE.md` is the presenter's technical guide.

## 8b. Decision package

The exact frozen protocol, the amendment, the options with their consequences, the Module F gate options and the other open decisions are in `HELDOUT_DECISION_PACKAGE.md`. Nothing was authorized.

## 9. What remains before the research can be called complete

1. Held-out evaluation: your authorization (recommended: base + Amendment 1 modules F, H, U, in two steps). **BLOCKED.**
2. F readiness gate failed as written (diagnosed as GPU noise): approve A2-a-r1 (recommended) or exclude F; the decision and the validation measurement must come before any held-out run. **BLOCKED on a decision.**
3. Phase 13 scope confirmation. **BLOCKED.**
4. AC12: the registered method cannot be completed retrospectively. **BLOCKED.** Per-file recovery covers 550 of 651 files (0 contradictions); 101 have no pre-build evidence (38 of them with derived consistency). Accepting the alternative evidence is your decision.
5. Public Git history still contains six transaction-level evidence files: make the repository private and notify Kaggle (recommended), and optionally rewrite history. **Your decision.**
6. H5 / G4: the original failure stays; the new pre-registered Phase 9b benchmark was run and **H5a-9b is supported** (separate result). H5b/H5c on 9b: OPTIONAL, not run.
7. Drift-triggered refresh: **both the score-PSI trigger and the performance trigger were executed (pre-registered, validation only); both outcomes NOT SUPPORTED** (see section 2). Neural relationship ablation of the temporal model: **executed** (pre-registered; pseudo-account and device contribute). GraphSAGE/GAT relation ablations, GAT unweighted, GraphSAGE/GAT explanations: **NOT RUN / OPTIONAL** (one learned family answers the ablation question; a non-converged GAT makes weighting and attention results uninterpretable).

**The project is not 100% complete.** Software, tests, demo, documentation and the validation-period research are complete and verified; the items above are genuinely unresolved.
