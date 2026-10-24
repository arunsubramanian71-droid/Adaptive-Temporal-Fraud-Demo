# Project completion report: macOS completion pass, 2026-10-23

Scope: the public demonstration repository at HEAD `975b0db` on a collaborator's Mac (Apple M4, macOS `27.0.1`), worked on as the brief "complete the project" asked: inspect, run, find defects, fix them with tests, verify, document. Companion files in this folder: `INITIAL_STATE_REPORT.md` (state before any edit), `FINAL_VERIFICATION_REPORT.md` (every command and result), `COMPLETION_MANIFEST.json` (machine-readable status), `CHANGED_FILES.json` (hashes before and after), `evidence/` (raw outputs).

## A. Executive summary

**Engineering:** the supported demonstration now installs and runs on macOS from a fresh clone with documented commands. It passes 19 of 19 acceptance steps there, 321 of 342 core tests (21 NOT RUN because the data and model files are withheld; 0 failed), and 374 of 396 with PyTorch installed (22 NOT RUN; 0 failed). The real-browser check passes 123 of 124 (1 NOT RUN, research scoring, model withheld). Eight defects were found and fixed, each with a regression test or a documented reason why it could not be exercised; the two most important were a setup check that reported success on a Mac where XGBoost cannot load, and verification commands that overwrote the project's dated evidence records. The dashboard gained an Overview tab built only from recorded results, a visible synthetic-demo indicator, readable status, data and model availability, and basic hardening against requests from other sites.

**Research:** nothing in the research changed, and no result was reproduced here: the IEEE-CIS data and the trained models are not on this machine (they are withheld from the public repository by design and need a Kaggle account and the competition rules' acceptance). The held-out evaluation remains **BLOCKED: EXPLICIT AUTHORIZATION REQUIRED**; nothing in this session opened a held-out file. The research is therefore not complete, and this report does not claim otherwise.

## B. Repository changes (all uncommitted; nothing was committed or pushed)

| File | Change | Why |
|---|---|---|
| `scripts/check_setup.py` | imports every required package; on a failed import prints the loader's own reason and the fix for this platform | it read version metadata only and printed `SETUP OK` on a Mac where `import xgboost` fails (D1) |
| `scripts/verification_output.py` (new), `scripts/verify_dashboard_e2e.py`, `scripts/verify_research_serving_e2e.py` | records go to `demo_output/verification/` or `--out`; only `--record` writes the tracked record; `--out` into `experiments/` is refused | every run overwrote the dated records (D2) |
| `scripts/verify_dashboard_browser.py` | platform-aware interpreter default; untracked output unless `--record`; private temporary folder; checks for the Overview tab, the status rows and two error states | Windows-only default (D3), record overwrite (D2), file-deletion risk in a chosen folder (D7) |
| `scripts/quickstart_unix.sh` (new) | macOS/Linux equivalent of the Windows quick-start (same five steps, same guarantees) | no supported path for macOS or Linux (D4) |
| `src/serving/app.py` | `/api/evidence` carries the recorded H4 decision; `/api/status` reports research data and model file presence (existence only); Host-header check, JSON-only POST, `frame-ancestors`/`base-uri`/`form-action`, `X-Frame-Options`, `Referrer-Policy` | Overview needs (D5); security review (D6) |
| `src/serving/static/dashboard.html` | Overview tab (opens first), header mode chips, readable status, chart of recorded validation PR-AUC, recorded verdicts, robust error states | professor-facing presentation without hard-coded numbers (D5) |
| `tests/test_setup_check.py`, `tests/test_verification_outputs.py` (new); `tests/test_serving.py`, `tests/test_handoff_docs.py` (tests added, none removed or weakened) | 15 new tests | regression tests for D1-D6 |
| `.gitignore` | `demo_output/` ignored | generated, reproducible output appeared as untracked files (D8) |
| `README.md`, `submission/DEMO_GUIDE.md` | macOS/Linux instructions, OpenMP requirement, verifier behaviour, troubleshooting rows, current test counts, curl examples (each executed) | documentation matched Windows only |
| `submission/completion_2026-10-23_macos/` (new) | this report, the initial state, the verification report, the manifest, the evidence | required records |
| `.claude/launch.json` (new, local tool configuration) | the desktop app's preview configuration used to view the dashboard | not part of the project; can be deleted |

Not changed: any experiment, result, model, protocol, pre-registration, split, threshold or integrity record. `presentation/` and the pre-existing modification of `experiments/phase13_serving_dashboard/test_results/e2e_verification.json` were left exactly as found.

## C. Defects and root causes

| # | Defect (confirmed by execution) | Root cause | Fix and regression test |
|---|---|---|---|
| D1 | On a Mac without Homebrew's OpenMP runtime, `check_setup.py` printed `xgboost 3.4.1 ok` and `SETUP OK` (exit 0) although `import xgboost` fails with `Library not loaded: @rpath/libomp.dylib`. Reproduced by copying the XGBoost package and pointing its only library search path at a missing folder | the check compared installed versions; the macOS wheel looks for `libomp.dylib` only in `/opt/homebrew/opt/libomp/lib` | each package is imported; the failure line shows the loader's message; the fix is chosen by platform (review found the first version picked the Homebrew fix on every platform, because XGBoost's message names all three runtimes). `tests/test_setup_check.py` (4 tests, XGBoost's real message text) |
| D2 | Running `verify_dashboard_e2e.py` (as was done on this Mac at 12:39) rewrote the tracked 2026-10-22 record, whose SHA-256 is pinned in `submission/INTEGRITY_SUBMISSION.json`; the browser verifier would have replaced the 25 recorded screenshots, making `tests/test_serving.py` fail on any machine without the research model | output paths were hard-coded to the tracked records | default output is untracked; `--record` is explicit; `--out` into `experiments/` is refused. `tests/test_verification_outputs.py` (5 tests; they restore the tracked bytes even when failing) |
| D3 | `verify_dashboard_browser.py` could not start the server on macOS or Linux without `--python` | default interpreter `.venv/Scripts/python.exe` | platform default; test in `tests/test_verification_outputs.py` |
| D4 | No setup path for macOS/Linux; the review then found that `quickstart_unix.sh --python` (no value) looped forever and that a folder passed as `--python` was accepted | Windows-only tooling; argument loop without a value check | `scripts/quickstart_unix.sh` validates options before any step; `tests/test_handoff_docs.py` runs the three option errors and the syntax check |
| D5 | The dashboard opened on a dense table, had no persistent demo indicator, showed raw status keys and could not say whether research data or a model file were present | scope of the original page | Overview tab and status changes; 3 tests in `tests/test_serving.py` (no decimal metric and no typed run count in the page source; H4 from its record; file presence by existence only) and 7 new real-browser checks |
| D6 | The API answered any Host name (a page on another site could re-point its own name at 127.0.0.1 and read the API), accepted non-JSON POST bodies and could be framed | missing request and header hardening (found by the ECC security review) | Host check (skipped only with `--allow-remote`), 415 for non-JSON POST, frame and form directives; 2 tests in `tests/test_serving.py` |
| D7 | The browser verifier emptied `<out-dir>/_tmp` after use, including files that were there before | fixed scratch folder inside a user-chosen directory | `tempfile.mkdtemp()`; this branch runs only with the research model, so it was not executed here |
| D8 | `demo_output/` showed as untracked after every demonstration | not in `.gitignore` | ignored |

One further issue was caught by review before it was ever served: the first Overview text said the graph models "stay below the tabular models", which is false for logistic regression and random forest. The page now names the registered comparison arm (`B_sub`) and takes every run count from the records.

## D. Verification evidence

See `FINAL_VERIFICATION_REPORT.md` for each command, its exit code and the evidence file. Summary of what was executed in this session:

| Check | Where | Result |
|---|---|---|
| Baseline core tests (unchanged code) | project `.venv` | 306 of 327, 21 NOT RUN, 0 failed, 5 modules skipped |
| Baseline with PyTorch (unchanged code) | isolated env, CPU PyTorch `2.9.1` | 359 of 381, 22 NOT RUN, 0 failed (first run of the five PyTorch modules on macOS) |
| Final core tests | project `.venv`, and the clean clone's fresh `.venv` | 321 of 342, 21 NOT RUN, 0 failed (identical in both) |
| Final with PyTorch | isolated env | 374 of 396, 22 NOT RUN, 0 failed, 42 modules |
| `scripts/quickstart_unix.sh` | clean clone, new `.venv` from PyPI | QUICKSTART OK (setup check, core tests, demonstration), 479 s including downloads |
| Clean-clone acceptance | clean clone | 19 of 19 (attempt 1: 16 of 17 because the test harness itself started the server with SIGINT ignored; fixed and re-run; both logs kept) |
| Dashboard end to end | clean clone | 31 of 31 |
| Real browser (Chrome `154`, desktop and phone, light and dark) | clean clone | 123 of 124, 1 NOT RUN (research scoring) |
| Claim checker | working tree | PASS |
| Security scan (patterns, file names, withheld artifacts, history) and ECC security review | working tree | no secret, no withheld artifact, the six per-transaction files absent from every commit; findings D6-D7 fixed |

## E. Model and research results

No model was trained, loaded or evaluated in this session; none can be, because the data and the trained models are absent. Every result the dashboard and documents show is a **recorded validation-period result** (days 122-151) read from the project's result files, unchanged. The dashboard's chart reproduces, at four decimals, the values in `experiments/phase12_validation_evidence_table/outputs/evidence_cells.csv` and `experiments/comparison_B_graph_features/outputs/table_with_B.csv` (checked by the browser harness against the API). The reported values the brief lists (XGBoost 0.5395, XGBoost + 40 graph features 0.5440, GraphSAGE 0.4637, GAT 0.3320, Temporal GraphSAGE 0.4927) are present in those records; their reproducibility status is "recorded; not re-run on this machine". The README records that the comparison-A XGBoost was regenerated byte-identically from the feature tables on 2026-10-23 on the development side; that was not repeated here.

What was checked without data: the code defines exactly the 40 pre-registered graph features (names and the 12-feature time-window ablation match `PREREGISTRATION_B.md` section 4, whose hash matches its sidecar); the split boundaries in `configs/data.yaml` are read, not hard-coded, and inclusive at both ends; the recorded counts are internally consistent (417,559 + 83,655 + 89,326 = 590,540 rows; 3.499% fraud overall); the leakage tests (backward-only features by brute-force equivalence, future-row invariance, history-after-new refusal, label-free scoring) all ran and passed.

## F. Acceptance checklist

Status words: VERIFIED (executed here with evidence), FAILED, BLOCKED, NOT APPLICABLE.

| # | Criterion | Status | Evidence or reason |
|---|---|---|---|
| 1 | A new user can install and launch the supported demonstration from the documented instructions | VERIFIED (macOS); Windows: historical 2026-10-22 record, not re-run; Linux: BLOCKED (no Linux machine) | `evidence/clean_clone/` |
| 2 | Correct behaviour when research data is missing | VERIFIED | status tiles and `/api/status`; 21 tests NOT RUN, never counted |
| 3 | Correct behaviour when a model artifact is missing or corrupted | VERIFIED | `--research-model` exits 2 with the reason; damaged-model and tampered-bundle tests pass |
| 4 | Every displayed research result traces to evidence | VERIFIED | page source holds no metric; API reads the recorded files; browser check compares chart to records; claim checker PASS |
| 5 | Historical and newly reproduced results are distinguished | VERIFIED | nothing was reproduced; every surface says "recorded, validation period" |
| 6 | Future observations cannot influence earlier predictions | VERIFIED for the code (tests on fixtures); the real-data leakage checks are historical records (data absent) | `tests/test_graph_features.py`, `tests/test_inference_demo.py`, `tests/test_phase9_core.py` |
| 7 | The held-out boundary is respected | VERIFIED | no TEST-named file opened; no `AUTHORIZATION.txt`; runner guards tested |
| 8 | Tests exercise the actual functionality | VERIFIED, with the NOT RUN tests listed; review strengthened two weak assertions | `evidence/final_local/` |
| 9 | The frontend represents the backend state accurately | VERIFIED | browser checks: tiles, chips and status rows against `/api/status`; error states |
| 10 | The synthetic workflow runs end to end | VERIFIED | demonstration (byte-identical twice), scoring through API and page |
| 11 | Public-release files are safe to publish | VERIFIED for the scanned scope; LOW note: six tracked pickle files (scalers and Phase 9 null tables) carry code-level trust; one hash checked | section J |
| 12 | Platform claims are backed by real checks | VERIFIED (macOS Apple Silicon); Linux documented as untested | README section 4 |
| 13 | Known limitations are documented | VERIFIED | README section 11; Overview "Limitations" |
| 14 | Remaining blockers are identified | VERIFIED | section G |
| 15 | Engineering and research completion are reported separately | VERIFIED | section K |
| 16 | Held-out evaluation executed under the frozen protocol | BLOCKED | explicit authorization required (section G) |
| 17 | Original experiments reproduced on this machine | BLOCKED | data and models absent |
| 18 | Clean-clone steps: clone, environment, install, configure (none needed), data check, model check, start, open, score, API, invalid input, communities, research-model availability, unavailable data, tests, stop, restart, health, no secrets needed | VERIFIED (macOS) | `evidence/clean_clone/acceptance_console.txt`; the page was opened in Chrome by the browser check |

## G. Remaining blockers

| Blocker | Why it cannot be resolved here | Who resolves it | Exact next action |
|---|---|---|---|
| Held-out evaluation (days 152-182) | requires the owner's one-time, irreversible authorization; the data lives on the development machine | the project owner | follow `submission/HELDOUT_AUTHORIZATION_REQUEST.md`: decide the Module F gate (approve A2-a-r1 or exclude F), then create `experiments/final_heldout_protocol/AUTHORIZATION.txt` with the text in `HELDOUT_DECISION_PACKAGE.md` section 4, then run `scripts/run_heldout_evaluation.py --split heldout` (and `scripts/run_amendment_evaluation.py --split heldout` for H and U) on the machine with the data |
| Research data and models absent on this Mac | redistribution is forbidden; download needs a Kaggle account and acceptance of the rules (not something to do on the owner's behalf) | the collaborator or owner | README section 7: download `train_transaction.csv` and `train_identity.csv` into `data/raw/`, run `scripts/build_processed_data.py` and `scripts/train_xgboost.py`; then `scripts/run_core_tests.py --require-artifacts` and `scripts/verify_dashboard_browser.py --research-rows ...` |
| Revised dashboard not yet verified with the research model | needs the withheld model | owner, on the development machine | `scripts/make_research_sample.py --n 10 --out <outside the repo>` then `scripts/verify_dashboard_browser.py --research-rows <file> --record` (replaces the 2026-10-22 browser record deliberately) |
| Windows re-run of the 2026-10-23 changes | no Windows machine here | owner | `powershell -ExecutionPolicy Bypass -File scripts/quickstart_windows.ps1`, then `scripts/run_core_tests.py` (expected 321 of 342 on a clone; not yet observed) |
| Linux support | no Linux machine here | anyone with Linux | `bash scripts/quickstart_unix.sh`; report the result |
| Phase 12 AC12, Phase 13 scope confirmation, Phase 9 G4 | historical; unchanged by this session | owner | `submission/COMPLETION_MATRIX.md` section 5 |

## H. Clean installation and demonstration (macOS, as executed)

```
brew install libomp
git clone https://github.com/arunsubramanian71-droid/Adaptive-Temporal-Fraud-Demo.git
cd Adaptive-Temporal-Fraud-Demo
bash scripts/quickstart_unix.sh --python /path/to/python3.12
.venv/bin/python -B scripts/run_dashboard.py --demo
```

Then open `http://127.0.0.1:8765/`; stop with Ctrl+C. `--python` is needed only when `python3.12` is not on PATH (it was not on this Mac; uv's or python.org's 3.12 both work). The executed run cloned the local checkout (HEAD plus this session's changes copied in), not GitHub, because the changes are not pushed.

## I. Experiment reproduction

Unchanged and not executed here: `submission/REPRODUCIBILITY.md` lists each experiment's command, inputs and outputs, and README section 8 the order. The held-out procedure is separate and one-shot (section G); never run it without the authorization file.

## J. Remaining risks

* **Integrity records are now older than the code.** `submission/INTEGRITY_SUBMISSION.json` lists ten of the files changed in this session and `experiments/phase13_serving_dashboard/INTEGRITY_PHASE13.json` five; their hashes describe the 2026-10-22/23 release. They were not rewritten (they are dated records). `CHANGED_FILES.json` lists every changed file with its hash at HEAD and now. Re-finalizing them is the owner's decision.
* **The working-tree copy of `e2e_verification.json`** differs from the recorded one (an earlier run on this Mac). `git restore experiments/phase13_serving_dashboard/test_results/e2e_verification.json` returns the recorded version; it was left for the owner.
* **The tracked browser record and screenshots show the page before this revision.** They remain valid for that page; the revised page's record is `evidence/browser/`.
* **Pickles.** Six tracked `.joblib`/`.pkl` files are loaded only by research scripts; loading a pickle runs code, so they are trusted like source code. The Phase 9 null tables match their recorded SHA-256.
* **Validation-only research,** GAT non-convergence, three dependent seeds: unchanged (README section 11).
* **Performance** was measured on synthetic data only (`evidence/performance_synthetic.json`: demo bundle training 0.38 s; scoring 1 row 4.4 ms, 30 rows 5.0 ms, 2,000 rows with 3,000 history rows 44 ms; peak memory about 214 MB); research-model latency is unmeasured.
* **Dependencies:** pinned; installed cleanly from PyPI for Python `3.12.13` on Apple Silicon; no vulnerability database was queried (no scanner is installed and none was added).

## K. Completion status

* **Engineering: complete for the supported demonstration on macOS (Apple Silicon), verified from a clean clone.** Not verified here: Windows after these changes, Linux, and the research-model paths that need withheld artifacts.
* **Research: not complete.** The mandatory held-out evaluation is BLOCKED on explicit authorization; no experiment was re-run on this machine; AC12 and the Phase 13 scope remain as recorded. No claim of 100 % completion or correctness is made.

## L. Recommended next action per blocker

1. Held-out: the owner makes Decision 1 (Module F gate) and, if willing, Decision 2 (authorization) in `HELDOUT_AUTHORIZATION_REQUEST.md`.
2. Integrity: review this diff, `git restore` the e2e record, commit, then re-finalize the integrity records with `scripts/finalize_submission_integrity.py` if the owner wants them to cover the new state.
3. Dashboard evidence with the research model: run the browser check with `--research-rows ... --record` on the development machine.
4. Windows and Linux: run the respective quick-start script once and record the result.
