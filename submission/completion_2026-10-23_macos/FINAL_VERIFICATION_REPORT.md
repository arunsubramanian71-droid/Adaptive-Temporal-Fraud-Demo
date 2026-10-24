# Final verification report, 2026-10-23 (macOS, Apple Silicon)

Every row in section 1 was **executed** in this session; its output is in `evidence/`. Section 2 lists commands that are documented but were **not executed**, and why. "Clean clone" means a `git clone` of the local checkout at HEAD `975b0db` with this session's changed and new files copied in (nothing else: not the owner's working-tree edit, `presentation/` or `.claude/`), with a `.venv` created and installed from PyPI by `scripts/quickstart_unix.sh` (pip cache disabled).

## 1. Executed

| # | Command | Environment | Exit | Result | Evidence |
|---|---|---|---|---|---|
| 1 | `.venv/bin/python -B scripts/check_setup.py` (before edits) | project `.venv` | 0 | SETUP OK | `evidence/baseline/check_setup_before_edits.txt` |
| 2 | `.venv/bin/python -B scripts/run_core_tests.py` (before edits) | project `.venv` | 0 | 306/327, 21 NOT RUN, 0 failed, 5 modules skipped | `evidence/baseline/run_core_tests_before_edits.txt` |
| 3 | `python -B scripts/run_core_tests.py --json ...` (before edits) | isolated env: requirements + `requirements-gnn.txt` (CPU PyTorch `2.9.1`, PyG `2.8.0.post1`) | 0 | 359/381, 22 NOT RUN, 0 failed, 40 modules | `evidence/baseline/run_core_tests_with_pytorch_before_edits.json` |
| 4 | `check_setup.py` with XGBoost's library search path pointed at a missing folder (copy of the package on `PYTHONPATH`) | project `.venv` | 0 before the fix, 1 after | before: `xgboost ok`, `SETUP OK` (defect D1); after: `CANNOT BE IMPORTED (XGBoostError: Library not loaded: @rpath/libomp.dylib)`, the Homebrew fix, `SETUP INCOMPLETE` | console output in this session; reproduced by `tests/test_setup_check.py` |
| 5 | `.venv/bin/python -B scripts/run_core_tests.py --json ...` (final) | project `.venv` | 0 | 321/342, 21 NOT RUN, 0 failed, 5 modules skipped | `evidence/final_local/run_core_tests.json` |
| 6 | same (final) | isolated env with PyTorch | 0 | 374/396, 22 NOT RUN, 0 failed, 42 modules | `evidence/final_local/run_core_tests_with_pytorch.json` |
| 7 | `scripts/run_core_tests.py --require-artifacts` | project `.venv` | 1 | fails by design: withheld artifacts are missing | console output |
| 8 | `bash scripts/quickstart_unix.sh --skip-tests` with `python3.12` absent from PATH | clean clone | 1 | stops at step 1 with the install instructions; no `.venv` created | `evidence/clean_clone/quickstart_console.txt` (first part) |
| 9 | `bash scripts/quickstart_unix.sh --python <uv CPython 3.12.13>` | clean clone | 0 | QUICKSTART OK; 479 s (downloads included) | `evidence/clean_clone/quickstart_unix_fresh_venv.txt`, `quickstart_console.txt` |
| 10 | clean-clone acceptance script (19 steps; listed in its console log) | clean clone | - | 19 passed, 0 failed (attempt 1: 16 of 17; its stop/restart steps were invalid because the harness, not the project, started the server with SIGINT ignored) | `evidence/clean_clone/acceptance_console.txt`, `acceptance_console_attempt1_harness_bug.txt`, `clean_clone_acceptance.sh` |
| 11 | `scripts/verify_dashboard_e2e.py --out ...` | clean clone | 0 | 31/31 | `evidence/clean_clone/e2e_verification.json` |
| 12 | `scripts/verify_research_serving_e2e.py --out ...` | clean clone | 3 | NOT RUN: model and validation features withheld | `evidence/clean_clone/research_e2e.txt` |
| 13 | `scripts/run_dashboard.py --research-model` without the model | clean clone | 2 | refuses with "model.joblib does not exist ..." | `evidence/clean_clone/research_mode_start.txt` |
| 14 | `scripts/demo_end_to_end.py` twice | clean clone | 0, 0 | both output folders byte-identical (every file, including the pickled bundles) | `evidence/clean_clone/synthetic_outputs_sha256.txt` (the synthetic rows themselves were moved out of the repository) |
| 15 | `scripts/verify_dashboard_browser.py --out-dir ...` (Playwright `1.64.0`, Chrome `154`) | clean clone | 0 | 123/124 passed, 0 failed, 1 NOT RUN (research scoring: model withheld); 29 screenshots | `evidence/browser/browser_verification.json` |
| 16 | the README's curl examples (health, status, sample, score, invalid body) | demo server | 0 | as documented (30 rows scored, 2 alerts, 400 with explanation) | console output |
| 17 | `.venv/bin/python -B scripts/verify_submission_claims.py` | working tree | 0 | PASS (every number in the checked documents traced) | console output |
| 18 | performance measurement on synthetic data | project `.venv` | 0 | see the file | `evidence/performance_synthetic.json`, `evidence/measure_performance.py` |
| 19 | secret and artifact scan: secret patterns in tracked files, sensitive file names, withheld-artifact patterns, the six per-transaction files across all history | working tree, `git log --all` | 0 | nothing found | console output |
| 20 | ECC `security-reviewer` and `python-reviewer` agents on the diff | read-only | - | security: 0 critical, 0 high, 1 medium (the owner's modified record), 4 low (fixed); code: 1 high, 3 medium, 3 low, 2 weak tests (fixed, except where stated in the completion report) | summarized in `PROJECT_COMPLETION_REPORT.md` sections C and J |

Ports used: 8766 (preview), 8791 (acceptance); the owner's dashboard on 8765 was never touched. Every server started here was stopped; port 8791 was confirmed free afterwards.

## 2. Documented but not executed

| Command | Why not |
|---|---|
| `scripts/run_completion_tests.py`, `scripts/run_tests.py` (full suite with artifacts) | need the withheld data and models |
| `scripts/verify_research_serving_e2e.py` (full run), `scripts/make_research_sample.py`, `scripts/verify_dashboard_browser.py --research-rows ...` | need the research model and validation features |
| every training, build and experiment script (`scripts/build_*.py`, `scripts/train_*.py`, `scripts/run_*` experiments) | need the IEEE-CIS data; neural ones a GPU |
| `scripts/run_heldout_evaluation.py --split heldout`, `scripts/run_amendment_evaluation.py --split heldout`, `scripts/run_f_walkforward.py --split heldout` | **sealed**: need the owner's authorization; never to be run without it |
| `scripts/quickstart_windows.ps1` and the Windows commands | no Windows machine (historical Windows results are recorded in the README) |
| any Linux run | no Linux machine |

## 3. Integrity of this record

`CHANGED_FILES.json` lists, for every file this session changed or added outside this folder, its SHA-256 at HEAD (or "new") and now, plus the SHA-256 of each file in this folder except itself. The owner's pre-existing working-tree edit and `presentation/` are excluded and were not modified.
