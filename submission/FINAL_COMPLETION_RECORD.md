# Final completion record (through Phase 13)

Dated 2026-10-21 (updated after the pre-submission verification). It supersedes the Phase 13 statements of `COMPLETION_MATRIX.md` and of `reports/PROJECT_SYNTHESIS_REPORT.md` and rewrites no historical record. Four things are kept apart: **software tests** (does the code do what it specifies), **demo and dashboard verification** (does the executable work as documented), **experiment execution** (was the pre-registered protocol carried out), **hypothesis support** (what the results say), and **genuine blockers**.

## 1. Phase 13: what was recovered, what was built, what is uncertain

**The original Phase 13 specification was not recovered.** Searching the saved sessions and the repository found no definition. Evidence that a Phase 13 was planned: the user's Phase 10 instruction ("Do not start Phases 11-13") and "Phase 13 not started" in every Phase 10-12 record; and after Phase 7 the user listed the remaining items in order: adaptive updating / drift, community discovery, explainability, final ablation suite, **dashboard, final report** (Phases 8, 9, 11, 12 correspond to the first four). The brief adds production-style inference, an `api/` and `dashboard/` layout, FastAPI/Streamlit as "likely" technologies and "investigation output".

**Built under a reconstructed scope** (`experiments/phase13_serving_dashboard/SCOPE_AND_ACCEPTANCE.md`, hashed before any code): a standard-library local API and dashboard. **Phase 13 passes its reconstructed criteria P13-1 to P13-10; the original Phase 13 scope remains uncertain** and this is not claimed to be the original specification. The criterion-by-criterion trace to code and executed evidence is `experiments/phase13_serving_dashboard/CRITERIA_TRACE.md`.

**Verification performed in this pass:**
* 14 of 14 serving tests; 12 of 14 injected defects detected, 2 equivalent (defence in depth).
* End-to-end run (`scripts/verify_dashboard_e2e.py`): the documented command started in a separate process; 31 of 31 checks (launcher refusals, page, every route, scoring of the synthetic sample, all error classes, shutdown with the port confirmed closed).
* **One deviation found and fixed:** the row limit returned 413; the registered criterion maps it to 400. Code and test now return 400 (body over 1 MB stays 413).
* **Rendering:** real screenshots in the in-app browser (dark colour scheme in a narrow pane, light scheme at 1000x700) of the evidence tables, a scored synthetic sample, the candidate-community cards and the status tab (`test_results/screenshots/`, hashed). A usability fix came out of it (the Basis column was too wide; now shortened with the full text in tooltips). Not checked: phone width, other browsers, keyboard-only use, print.
* Earlier statement "DOM inspection only" is superseded: screenshots now exist.

## 2. The clean-environment test failure: resolved as a test-design issue

* **Cause:** `tests/test_phase12_outputs.py::test_AC10_sidecars_and_deterministic_rebuild` compares the whole rebuilt manifest with the recorded one apart from the timestamp. The manifest records the build interpreter's version, so under Python 3.14.3 it differs from the recorded `3.12.0` in `versions.python` only. All 11 data outputs are byte-identical in both interpreters.
* **Verdict:** not a reproducibility defect; the registered AC10 wording ("byte-identical on re-run apart from manifest timestamps") implicitly assumes the same environment.
* **Correction without touching historical records:** the registered test and the Phase 12 integrity record are unchanged. A new companion test (`tests/test_phase12_environment_independence.py`) checks the environment-independent property (data outputs byte-identical; manifests equal except timestamp and recorded versions; the rebuilt manifest records the running interpreter). `scripts/run_core_tests.py` reports a failure of the registered test as "environment-metadata difference", **never counts it as passed**, and does so only if the companion passes.
* **Actual results** (`submission/clean_env_results/`; new virtual environments installed only from `requirements.txt`, no PyTorch): Python 3.12.0: **234 of 234** executed tests pass (registered AC10 test included); Python 3.14.3: **233 of 234** pass plus **1 environment-metadata difference** (the registered test), the same interpreter-version difference. Two modules that need PyTorch (37 tests) are skipped and named.

## 3. Software tests, demo, dependencies (actual results)

| Check | Result |
|---|---|
| Full suite in the project environment (`scripts/run_completion_tests.py`) | **303 of 303** tests pass in 29 modules (`submission/test_results.json`) |
| Clean-environment counts | The 234/233 figures in section 2 are from the earlier suite. Re-run on 2026-10-21 with the current suite in two fresh environments built only from `requirements.txt` (`submission/clean_env_results/*_2026-10-21.json`): Python 3.12.0 **257 of 257** pass; Python 3.14.3 **256 of 257** pass plus **1 environment-metadata difference** (the registered Phase 12 AC10 interpreter-version test, never counted as passed). `pip check` is clean in both. Four PyTorch-dependent modules are skipped and named (`test_f_runner_gate`, `test_neural_graph_evidence`, `test_phase8_adaptive`, `test_walkforward_ext`); a skip is not a pass |
| Dependency installation | `pip install -r requirements.txt` succeeded in two fresh environments (Python 3.12.0 and 3.14.3); `pip check` clean in the project environment |
| Documented commands on Windows | `scripts/check_setup.py` OK in PowerShell and in cmd.exe with backslash paths; with forward slashes cmd.exe fails (documented in the README); demo, dashboard and test commands run |
| Demonstration | `scripts/demo_end_to_end.py` from the project root: five stages, invariants true; deterministic; identical numbers in all three environments; synthetic data only, labelled as such (the 3.14 clean environment exposed that it imported PyTorch through a frozen module: fixed by `src/demo/_compat.py`, frozen files untouched) |
| Numeric claims | `scripts/verify_submission_claims.py`: every number traced (with a self-test that rejects invented numbers) |
| Sensitive data | no secrets, e-mail addresses or tokens in the new files; one run log contained the Windows home path and was redacted (noted in the file); transaction-level validation files exist under `experiments/` as in earlier phases and are not in `submission/` |
| Not possible / not done | no run on Linux or macOS; no accessibility or load testing of the dashboard |

## 4. Audit of mandatory versus optional work

| Item | Classification and evidence | Status |
|---|---|---|
| **Neural graph-level explanations** | Mandatory in the brief ("for graph models, provide interpretable evidence such as ..."); deferred in Phase 11 first pass. Independent and authorized (validation only, no label). | **Done for the designated model** (Temporal GraphSAGE seed 42): pre-registered counterfactuals (link removal, memory reset) on 300 transactions, `experiments/neural_graph_evidence/`. **Registered criterion NE5(a) FAILED** (never-seen-entity resets not exactly zero in 13 of 102 cases; largest 9.54e-07, GPU float noise); reported, not repaired; the rest passed. GraphSAGE and GAT not examined (attention invalid for a non-converged GAT). |
| Relationship-level ablations | Brief: "where computationally feasible ... additional". Conditional/optional. Done for XGBoost + graph features (4 relations + time windows). Address and email relations are not in the graph (excluded in Phase 4). Retraining neural models without a relation is feasible (minutes per fit) but not run. | Optional; **not done**; I can run it under a pre-registration if wanted |
| Phase 12 AC12 | Registered criterion; no authentic pre-build hashes survive for 322 of 651 files. | **Genuinely BLOCKED**; nothing fabricated; not reopened |
| Phase 9 G4 | Frozen gate failed on 2 of 3 seeds. | FAIL; criterion not altered |
| Held-out evaluation | Mandatory in the brief; blocked on authorization. | **BLOCKED** on the user |
| Dashboard / API | Part of the reconstructed Phase 13 | built; scope unconfirmed |

## 5. Ledger by phase (software correctness | execution | scientific outcome | status)

| Phase | Status |
|---|---|
| 1-2 | PASS |
| 3 Tabular | PASS; class weighting later found to lower XGBoost (0.5948 unweighted vs 0.5395) |
| 4 Graph | PASS |
| 5, 6A GraphSAGE | PASS; below XGBoost, above its no-relations control |
| 6B GAT | PASS (executed); never converged; no inferiority claim |
| 6C | FAIL / aborted, excluded |
| 7, 7A Temporal | PASS; modest cold-start-concentrated gain; H3 partial |
| 8 Adaptation | PASS; H4 supported under the locked rule (n = 3) |
| 9 Communities | FAIL (frozen gate G4); H5 not established |
| 10 | core PASS; empirical tiers BLOCKED / not authorized |
| 11 | PASS (first pass); graph evidence for the neural model now exists (above) |
| 12 | AC1-AC11 PASS; **AC12 BLOCKED**; **not formally complete** |
| **13** | **PASS against the reconstructed criteria; original scope uncertain** |
| Completion work (B, imbalance, graph evidence, protocol, evaluator, package) | PASS (executed); H1-direct not established; H2 contradicted by the rule; tabular above neural in both recipes |

## 6. Held-out protocol review (decision report; nothing requested or created)

`experiments/final_heldout_protocol/PROTOCOL_REVIEW.md` (generated by the read-only `scripts/verify_heldout_protocol_ready.py`). Findings: protocol SHA-256 `59bf1565fb16653a02a2d743cc445e114fb2a6a42f1de7b2fa56261e6f68c806`, sidecar matches; 19 frozen models, all artifacts present, recorded validation thresholds listed; the validation readiness gate was **re-run after a hardening change** (every feature frame is now read with the label column projected away) and passes for all 19 models (maximum score difference 9.54e-07); gate, one-shot and label-projection behaviours are unit-tested without touching held-out data; `AUTHORIZATION.txt` and `heldout_results/` do not exist.
**Ready for informed authorization of the base protocol**, with three stated gaps against the brief that were examined in the scope reconciliation of 2026-10-21 (section 6A): the adaptive system / H4 on future data (closed by the separate, hashed Amendment 1 module F); the historical-feature reference and the unweighted variants (Amendment 1 modules H and U; the MLP control deliberately excluded); community discovery (unsupported on held-out data). **Irreversible consequences:** the held-out period can never again be untouched test data; one-shot; nothing may change in response; all results must be reported. **Exact authorization:** `AUTHORIZATION.txt` containing the section-9 sentence of the protocol and the hash above, then `scripts/run_heldout_evaluation.py --split heldout`. I am waiting for your decision (see section 6A for the options and their consequences).

## 6A. Scope reconciliation before any authorization (2026-10-21; nothing requested or created)

* **Decisions and evidence:** `experiments/final_heldout_protocol/SCOPE_RECONCILIATION.md` (model-by-model: mandatory, sufficient, missing, optional, unsupported). The existing protocol does **not** cover every mandatory requirement: it lacks the adaptive system (H4 on future data) and the historical reference line.
* **Amendment 1** (`AMENDMENT_1.md`, SHA-256 in `AMENDMENT_1.sha256`; the base protocol file and hash are unchanged): module F (adaptive arms, five 6-day windows 152-181, memory built over days 1-151, TRAIN-only hyperparameters), module H (historical-feature XGBoost reference), module U (A_none, B_none, unweighted GraphSAGE and Temporal GraphSAGE). Authorization needs the base sentence and protocol hash **and** the amendment hash and `I authorize the following Amendment 1 modules: <letters>`.
* **Readiness on validation:** H and U PASS (9 models; XGBoost differences 0.0, neural at most 3.0e-07; A_none and B_none re-fits reproduce the recorded validation results exactly). Base 19-model gate re-run after the evaluator was parametrized: PASS, maximum difference 9.54e-07. **Module F: all 21 deterministic checks pass; the trained-arm window-mean gate failed as written for 1 of 6 cases (E, seed 123: 0.0104 against 0.01); recorded as FAIL as written, not changed.**
* **Phase 8 memory initialisation:** the Phase 8 loop starts with an empty entity memory; the pre-registered supplement (`experiments/phase8_state_continuity_supplement/`) leaves H4 SUPPORTED. Its own reproduction criterion S1 failed for the trained arms (training not bit-reproducible) and was not changed.
* **NE5:** the failed registered criterion is diagnosed as GPU floating-point nondeterminism (exact zero on CPU); the registered result stands.
* **Neural relationship-level ablations:** conditional optional (the brief's "where computationally feasible"); not run; no registered hypothesis depends on them.
* **Python:** 3.12 recommended (section 2 evidence).

## 7. Files and preservation

Historical Phases 1-12 artifacts are unchanged except the three Phase 12 files documented as revised on 2026-10-19; `submission/INTEGRITY_SUBMISSION.json` compares the project with the last persistent baseline and lists every new file with its hash. Git was not used; `.gitignore` was not modified; the nine empty `C:\c\...` folders were left; a previous readiness record was kept under `experiments/final_heldout_protocol/readiness_val/previous/`.

## 8. Overall acceptance

**Not 100% complete.** The held-out evaluation (a brief requirement) is blocked on your authorization; Phase 9's frozen gate failed; Phase 12's AC12 is blocked; the Phase 13 scope is a reconstruction; one registered sanity criterion of the neural graph evidence failed at float-noise level; GraphSAGE/GAT explanations and neural relationship ablations are not done (optional). Everything that can be completed without those decisions is complete and verified.
