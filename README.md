# Adaptive Temporal Graph Intelligence for Fraud Detection & Coordinated Fraud Discovery

A research project and working software system on the IEEE-CIS Fraud Detection data. It asks whether **heterogeneous relational structure and temporal behaviour improve fraud detection and the discovery of coordinated groups compared with strong tabular models, particularly on future data and under drift.** It contains the full experimental pipeline (tabular baselines, leakage-safe graph features, GraphSAGE, GAT, a temporal graph network with entity memory, drift and adaptive updating, candidate-community discovery, explanations), a frozen but not yet executed held-out protocol, a local API and dashboard, a synthetic demonstration that runs on any machine, and 381 automated tests.

> **Status in one paragraph.** Every research result is a **validation-period** result (days 122-151). The held-out period (days 152-182) is sealed: no model has been scored on it, and running that one-shot evaluation awaits the owner's authorization. On validation, a tuned XGBoost on transaction features is at least as good as every graph model tried; recurrent entity memory gives a small, cold-start-concentrated gain; periodic refresh reduces within-validation drift degradation; candidate-community discovery failed its original frozen benchmark gate and is supported only on a separately pre-registered second benchmark. Groups found on real data are *candidate* communities: the dataset has no labels for coordinated fraud, and nothing here claims that any group is fraudulent or coordinated.

*(The title's last words are abbreviated on purpose. The project's terminology rule reserves ring wording for confirmed groups, and none exist here.)*

---

## Contents

1. [What is implemented](#1-what-is-implemented)
2. [Architecture](#2-architecture)
3. [Repository layout](#3-repository-layout)
4. [Requirements](#4-requirements)
5. [Quick start: install, test, demonstrate (no dataset needed)](#5-quick-start-install-test-demonstrate-no-dataset-needed)
6. [The dashboard and API](#6-the-dashboard-and-api)
7. [Data and model artifacts: what is in the repository and how to get the rest](#7-data-and-model-artifacts)
8. [Reproducing the research experiments](#8-reproducing-the-research-experiments)
9. [Research findings (validation period)](#9-research-findings-validation-period)
10. [Tests and verification](#10-tests-and-verification)
11. [Known issues, limitations and open decisions](#11-known-issues-limitations-and-open-decisions)
12. [Troubleshooting](#12-troubleshooting)
13. [Reproducing the demonstration on another machine](#13-reproducing-the-demonstration-on-another-machine)
14. [Further documentation](#14-further-documentation)

---

## 1. What is implemented

| Capability | Where | Status |
|---|---|---|
| Chronological split (train days 1-121, validation 122-151, held-out 152-182), TRAIN-fit preprocessing | `configs/data.yaml`, `src/data/`, `src/features/preprocessing.py` | implemented, tested |
| Leakage-safe historical and graph features (a row only sees strictly earlier rows) | `src/features/historical.py`, `src/features/graph_features.py` | implemented, brute-force-equivalence tests |
| Heterogeneous transaction graph (card, pseudo-account, network proxy, device) with edge times | `src/graph/` | implemented; Phase 4 validation |
| Tabular baselines: logistic regression, random forest, XGBoost | `scripts/train_*.py`, `src/models/` | trained and recorded |
| XGBoost + 40 graph features (comparison B), class-imbalance study | `scripts/run_comparison_b.py`, `experiments/comparison_B_graph_features/`, `experiments/imbalance_check_xgboost/` | pre-registered, executed |
| GraphSAGE, GAT, Temporal GraphSAGE (GRU entity memory) and their controls | `src/models/`, `src/graph/temporal.py` | trained (GPU), recorded; GAT did not converge |
| Concept drift (PSI/KS, temporal PR-AUC) and adaptive updating (frozen / periodic refresh / replay control; triggered refresh) | `src/adaptive/`, `src/trigger/`, `experiments/phase8_*` | executed on validation |
| Candidate-community discovery (Louvain on the suspicious subgraph, scoring, tracking, aggregate cards) | `src/coordination/`, `src/evolution/`, `src/explain/cards.py` | executed; synthetic benchmarks |
| Explanations (XGBoost TreeSHAP contributions; counterfactual graph evidence for the temporal model) | `src/explain/` | executed |
| Held-out evaluation (frozen protocol, gated evaluators, one-shot guards) | `experiments/final_heldout_protocol/`, `src/heldout/`, `scripts/run_*_evaluation.py` | **ready, not run** (needs authorization) |
| Production-style scoring of new transactions from stored history | `src/inference/scorer.py`, `scripts/score_transactions.py` | implemented, tested |
| Local API + dashboard (standard library): recorded evidence, demo scoring, research-model scoring, candidate communities, status | `src/serving/`, `scripts/run_dashboard.py` | implemented, tested, verified in a real browser |
| Synthetic end-to-end demonstration | `src/demo/`, `scripts/demo_end_to_end.py` | runs anywhere, byte-identical output |

The original project brief and the phase-by-phase requirement trace are in `submission/TRACEABILITY_MATRIX.md` (every requirement with its source, mandatory/optional status, evidence and status).

## 2. Architecture

```
 IEEE-CIS train_transaction.csv + train_identity.csv   (not redistributed; see section 7)
        |
        |  scripts/build_processed_data.py   chronological split, TRAIN-fit encoders, leakage checks
        v
 data/features/{train,val,test}_features.parquet           data/processed/train_merged.parquet
        |                                                          |
        |  scripts/build_graph.py                                  |  scripts/build_historical_features.py
        v                                                          v
 heterogeneous graph: transaction - card / pseudo-account /   label-derived history features (reference line)
 network proxy / device, every edge timed
        |
        +--> tabular models (LR, RF, XGBoost)  --+
        +--> XGBoost + 40 backward-looking graph features (B)
        +--> GraphSAGE / GAT (static, graph masked to edges with time <= t for each day t)
        +--> Temporal GraphSAGE (entity memory carried day by day)
                                                  |
                     one metric module (src/models/metrics.py): PR-AUC, ROC-AUC, P@K, R@K, confusion, alerts
                                                  |
        Phase 8: drift measurement and adaptive arms        Phases 9-11: candidate communities, tracking,
        (frozen / periodic refresh / replay control)        explanations, aggregate cards
                                                  |
        Phase 12: consolidated validation table     Held-out protocol (frozen, gated, NOT RUN)
                                                  |
 Serving (src/serving/app.py, one local HTTP server)
   GET  /                      dashboard page (src/serving/static/dashboard.html)
   GET  /api/evidence          recorded validation tables (read from experiments/ at request time)
   GET  /api/communities       aggregate-only candidate-community cards
   GET  /api/status            what has and has not been done
   POST /api/score             scoring with a bundle (synthetic demo bundle, or one trained with score_transactions.py)
   GET  /api/research/model    the recorded comparison-A XGBoost (only with --research-model)
   POST /api/research/score    scores rows exactly as the research evaluation did
```

The dashboard is a single static page; its JavaScript calls the JSON endpoints above on the same server. The neural models are batch research models: they score whole days of graph state and are not served row by row (section 11).

## 3. Repository layout

```
configs/data.yaml              split boundaries, expected counts, feature blocklist
src/
  data/ features/ graph/       loading, splits, preprocessing, leakage-safe features, graph construction
  models/                      LR/RF/XGBoost feature builders, GraphSAGE, GAT, Temporal GraphSAGE, metrics
  adaptive/ trigger/           drift, walk-forward update loop, replay control, refresh triggers
  coordination/ evolution/     candidate-community discovery, synthetic benchmarks, tracking over time
  explain/ evidence/           contributions, graph evidence, community cards; Phase 12 table builder
  heldout/                     gated held-out evaluator (authorization, readiness, one-shot guards)
  inference/                   production-style scorer and the research-model wrapper
  serving/                     local HTTP API and dashboard page
  demo/                        synthetic fixture and the end-to-end demonstration
scripts/                       one script per experiment, check or tool (entry points; see sections 5-10)
tests/                         40 test modules (run without pytest)
experiments/                   recorded results, designs, pre-registrations and integrity records, one folder per study
reports/                       phase reports and reports/PROJECT_SYNTHESIS_REPORT.md
submission/                    manuscript, completion records, guides, decision package, recorded test results
requirements.txt               core dependencies (demo, dashboard, scorer, tabular models, most tests)
requirements-gnn.txt           extra dependencies for the neural-graph experiments (PyTorch, PyTorch Geometric)
requirements-browser-check.txt optional: Playwright for the real-browser dashboard check
```

## 4. Requirements

* **Operating system:** Windows 11 (tested). Linux and macOS are expected to work for the core path but have not been tested; use `.venv/bin/python` instead of `.venv/Scripts/python.exe` there.
* **Python:** 3.12 (tested with 3.12.0; recommended). Python 3.14.3 also runs the core suite; one registered test then reports the different interpreter version stored in a recorded manifest (an environment-metadata difference, never counted as a pass; section 10).
* **For the demonstration, dashboard and core tests:** about 1 GB of disk for the environment, no GPU, no dataset, no account, no secrets. There are no environment variables to set, so no `.env` file is needed.
* **For the research experiments:** the IEEE-CIS files (about 1.3 GB raw, a further 1.3 GB processed), about 2 GB for experiment outputs, 16 GB of RAM or more, and a GPU for the neural models (the recorded runs used an AMD Radeon RX 7900 GRE with ROCm 7.2 and PyTorch 2.9.1).

## 5. Quick start: install, test, demonstrate (no dataset needed)

All commands run from the repository root in **PowerShell**. Forward slashes in `.venv/Scripts/python.exe` work in PowerShell; in `cmd.exe` use backslashes.

**Option A: one script** (creates `.venv` if it is missing, installs `requirements.txt`, checks the setup, runs the core tests and the demonstration; it never deletes anything, never downloads data and never touches the held-out period):

```
powershell -ExecutionPolicy Bypass -File scripts/quickstart_windows.ps1
```

**Option B: step by step**

```
py -3.12 -m venv .venv                      # or: python -m venv .venv  (with Python 3.12 on PATH)
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -B scripts/check_setup.py
.venv/Scripts/python.exe -B scripts/run_core_tests.py
.venv/Scripts/python.exe -B scripts/demo_end_to_end.py --out-dir demo_output
.venv/Scripts/python.exe -B scripts/run_dashboard.py --demo
```

Then open http://127.0.0.1:8765/ and stop the server with Ctrl+C.

**What you should see**

* `check_setup.py` ends with `SETUP OK for the demonstration`. It lists the optional neural-network stack and the research data files as missing on a fresh machine; that is expected.
* `run_core_tests.py` on a fresh clone ends with `306/327 tests passed`. The other 21 tests need data or model files that are not published; they are reported as **NOT RUN (withheld artifact missing)**, and five modules that need PyTorch are **SKIPPED**. Neither is counted as a pass. The exit code is 0.
* `demo_end_to_end.py` prints five stages (synthetic stream, two models with and without graph features, history-based scoring with explanations, candidate-community discovery, leakage invariants) and writes `demo_output/`. It reproduces no research result, and every number it prints is on synthetic data.
* The dashboard opens on the *Evidence (validation)* tab (section 6).

## 6. The dashboard and API

`scripts/run_dashboard.py` starts one local server for both the API and the dashboard. It binds to 127.0.0.1 only and has no authentication by design (`--allow-remote` overrides the binding; do not use it on an untrusted network).

| Command | What it serves |
|---|---|
| `.venv/Scripts/python.exe -B scripts/run_dashboard.py` | recorded evidence, communities and status; scoring answers 404 |
| `... run_dashboard.py --demo` | the above plus scoring with a bundle trained in memory on the **synthetic** fixture (every response says so) |
| `... run_dashboard.py --bundle DIR` | scoring with a bundle you trained with `scripts/score_transactions.py train` |
| `... run_dashboard.py --demo --research-model` | also the recorded research XGBoost on `/api/research/*` (needs local artifacts, section 7) |
| `... --port 8800` | another port if 8765 is taken |

**Tabs:** *Evidence (validation)*: the recorded Phase 12 and comparison-B tables and the verdicts. *Score transactions (demo)*: *Load synthetic sample*, then *Score*; the result is labelled SYNTHETIC DEMONSTRATION BUNDLE, and alerts list the model's largest feature contributions. *Research model*: choose a JSON file of rows written by `scripts/make_research_sample.py`, then *Score with the research model*. *Candidate communities*: 24 aggregate-only cards (no member identities). *Status and limits*: what has and has not been done. The page works with the keyboard (Tab, arrow keys between tabs), in light and dark colour schemes and at phone width.

**API examples (PowerShell):**

```
Invoke-RestMethod http://127.0.0.1:8765/api/status
$s = Invoke-RestMethod http://127.0.0.1:8765/api/sample                      # demo server only
$body = @{ history = $s.history; new = $s.new } | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8765/api/score -Method Post -ContentType 'application/json' -Body $body
try { Invoke-RestMethod http://127.0.0.1:8765/api/score -Method Post -ContentType 'application/json' -Body '{"new": []}' }
catch { $_.ErrorDetails.Message }       # the 400 response body, with the explanation
```

(Windows PowerShell 5.1 prints only `(400) Bad Request` for an uncaught error; the server's explanation is in `ErrorDetails`.)

**Errors:** every invalid request gets a JSON error with a status code: 400 (invalid rows or JSON; the message says what to fix), 404 (unknown path, or no bundle/research model loaded), 405 (wrong method), 413 (body over 1 MB), 500 (internal error, never with a stack trace). `TransactionID` must be an integer; `TransactionAmt` must be a finite number; at most 5,000 rows per request.

## 7. Data and model artifacts

**What a fresh clone contains:** all source code, configuration, tests, scripts, reports, the recorded aggregate results and integrity records, the dashboard's evidence files and the dashboard screenshots. That is everything the quick start, the dashboard (demo mode) and the core tests need.

**What it does not contain, and why:** the IEEE-CIS data (competition rules forbid redistribution), anything derived row by row from it (feature tables, prediction files, per-transaction explanations), trained model files and checkpoints, and raw-vocabulary encoder files. These are excluded by `.gitignore`. The full classification is in `submission/ARTIFACT_INVENTORY.md`.

**How to obtain them (research use only):**

1. Accept the competition rules and download the data from Kaggle: https://www.kaggle.com/c/ieee-fraud-detection/data (a Kaggle account is required; the rules allow non-commercial research use and forbid redistribution). Place `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv` and `sample_submission.csv` in `data/raw/`. Only the two `train_*` files are used; the Kaggle test files have no labels and are never used for evaluation.
2. Build the processed data: `.venv/Scripts/python.exe -B scripts/build_processed_data.py` (it validates the expected row and fraud counts in `configs/data.yaml`).
3. Train the model you need, for example the research XGBoost served by `--research-model`: `.venv/Scripts/python.exe -B scripts/train_xgboost.py` (writes `experiments/xgboost/model.joblib` and its encoder files).

> **Caution:** training scripts write into the existing `experiments/` folders, which hold the recorded results. In a copy where those records matter, copy the folder aside first. The recorded predictions let the tests check that a regenerated model reproduces the recorded validation scores (`tests/test_research_serving.py`). On 2026-10-23, step 3 was re-executed in an isolated clone from copies of the TRAIN and VAL feature tables (about 3.5 minutes on a CPU). The regenerated `model.joblib`, its encoder files, `metrics_val.json` and every validation prediction were byte-identical to the recorded ones, and the research-model serving check passed 12 of 12 there. Step 2 was not re-executed, because it reads the raw file, which includes the sealed held-out days; its recorded counts are in `data/processed/build_log.json`.

Without these artifacts, the research-model tab says that the model is not loaded and its button is disabled, and the 21 artifact-dependent tests report NOT RUN.

## 8. Reproducing the research experiments

With the data in place: `scripts/build_processed_data.py`, `scripts/run_validation.py`, `scripts/build_historical_features.py`, `scripts/build_graph.py`, then each phase's training scripts (`scripts/train_*.py`; the neural ones need `requirements-gnn.txt` and a GPU), and for comparison B `scripts/run_comparison_b.py` followed by `scripts/build_b_outputs.py`. `reports/PROJECT_SYNTHESIS_REPORT.md` section 7 maps every phase to its scripts, tests and integrity records; `submission/REPRODUCIBILITY.md` lists each experiment's command, inputs and outputs. Every experiment folder records its seeds, configuration and split boundaries.

**Never run** anything that opens `data/features/test_features.parquet` or a TEST-named file, and never run `scripts/run_heldout_evaluation.py --split heldout`, `scripts/run_amendment_evaluation.py --split heldout` or `scripts/run_f_walkforward.py --split heldout`. They refuse without the owner's `AUTHORIZATION.txt`, and the evaluation is one-shot (`submission/HELDOUT_DECISION_PACKAGE.md`). The `--split val` readiness checks are safe.

## 9. Research findings (validation period)

| Question | Evidence | Outcome |
|---|---|---|
| H1: graph structure beyond transaction features | comparison B, pre-registered | **not established**: XGBoost + 40 graph features is not reliably different from XGBoost (PR-AUC 0.5440 against 0.5395, single seed; 3-seed variants 0.5611 against 0.5616) |
| H2: learned graph representations beyond handcrafted graph features | comparison B, Phase 12 table | **contradicted**: GraphSAGE 0.4637, GAT 0.3320 (did not converge), Temporal GraphSAGE 0.4927, all below the tabular models |
| H3: temporal information helps future transactions | Phase 7, 7A controls | **partially supported, modest**: the gain over controls is small and concentrated on cold-start transactions; no future-period reading yet |
| H4: adaptive updating reduces drift degradation | Phase 8, three seeds, locked rules | **supported on validation**: window-mean PR-AUC 0.4580 periodic refresh, 0.4243 frozen, 0.4179 replay control; drift-triggered and performance-triggered refresh were not supported by their registered rules |
| H5: graph analysis identifies coordinated communities | Phase 9 benchmark; Phase 9b | **not established** under the original protocol (frozen gate G4 failed); a separately pre-registered benchmark (Phase 9b) supports H5a-9b without changing that verdict |
| Class imbalance | two pre-registrations | weighting **hurts** XGBoost (0.5948 unweighted against 0.5395 weighted) |
| Which relations matter (temporal model) | pre-registered relation ablation | the pseudo-account link carries most of the relational signal; device contributes a little; card and network proxy do not |

Read `submission/MANUSCRIPT.md` for the full write-up and `reports/PROJECT_SYNTHESIS_REPORT.md` for the phase-by-phase record. Every number above comes from a recorded result file, and `scripts/verify_submission_claims.py` checks that every number quoted in the submission documents appears in those records.

## 10. Tests and verification

| Command | What it checks | Result on 2026-10-22 |
|---|---|---|
| `.venv/Scripts/python.exe -B scripts/run_core_tests.py` | all tests, reporting withheld-artifact tests as NOT RUN and PyTorch modules as SKIPPED | fresh clone: 306 of 327 executed pass, 21 NOT RUN, 0 fail |
| `... scripts/run_completion_tests.py` | full suite with every artifact present; writes `submission/test_results.json` | 381 of 381 in 40 modules (development machine) |
| `... scripts/run_core_tests.py --require-artifacts` | fails if any withheld artifact is missing | exit 0 on the development machine, exit 1 on a clone (by design) |
| `... scripts/verify_dashboard_e2e.py` | starts the documented server, exercises every route and error, stops it | 31 of 31 |
| `... scripts/verify_research_serving_e2e.py` | research-model mode end to end (needs local artifacts) | 12 of 12 |
| `scripts/verify_dashboard_browser.py` | the real page in headless Chrome, desktop and phone, light and dark (optional; see below) | 123 of 123 |
| `... scripts/verify_submission_claims.py` | every number in the submission documents traces to a recorded result | PASS |

The test modules run without pytest; each prints `[PASS]`/`[FAIL]` lines. A skipped module or a NOT RUN test is never counted as passed.

**Optional real-browser check** (uses an installed Chrome or Edge; nothing is downloaded; keep this environment outside the repository):

```
python -m venv "$env:TEMP\fraud-browser-check"
& "$env:TEMP\fraud-browser-check\Scripts\python.exe" -m pip install -r requirements-browser-check.txt
& "$env:TEMP\fraud-browser-check\Scripts\python.exe" -B scripts/verify_dashboard_browser.py
```

Without `--research-rows`, and on a machine without the research model, the research-scoring checks are reported NOT RUN. With the local artifacts, write a rows file first: `.venv/Scripts/python.exe -B scripts/make_research_sample.py --n 10 --out "$env:TEMP\fraud_research_rows.json"` and pass `--research-rows "$env:TEMP\fraud_research_rows.json"`.

## 11. Known issues, limitations and open decisions

* **No held-out evaluation yet.** The protocol is frozen, hashed and readiness-checked on validation; the one-shot run awaits the owner's authorization. Module F (adaptive arms) failed its registered readiness gate as written, because of GPU nondeterminism; a corrected gate (A2-a-r1) is proposed and not active. Options and consequences: `submission/HELDOUT_DECISION_PACKAGE.md`.
* **Validation-only results.** Models were tuned on the validation period; optimism is expected and unmeasured.
* **GAT did not converge** within its budget; its results cannot separate inferiority from under-training.
* **Phase 12 acceptance criterion AC12 is BLOCKED as registered.** Its before/after file hashes cannot be taken after the fact; alternative evidence covers 550 of 651 files with no contradiction.
* **Phase 13 (serving and dashboard) was built to a reconstructed scope.** The original specification was never recovered (`submission/PHASE13_SCOPE_CONFIRMATION.md`).
* **The neural models are not served.** They score whole days of graph state with entity memory, not single posted rows.
* **History of this repository.** This public repository was published on 2026-10-23 with a filtered history: six per-transaction evidence files, removed from the branch tip on 2026-10-22, were also removed from every earlier commit, so commit IDs after the first commit differ from those quoted in the dated records (`submission/COMMIT_MAP.txt` maps old to new). The original development repository is kept privately.
* **Three seeds** share data and are not independent replications.

## 12. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `'py' is not recognized` or the wrong Python version | install Python 3.12 from python.org, then `py -3.12 -m venv .venv` (or call that Python's full path) |
| `ModuleNotFoundError: yaml` / `pandas` / `xgboost` | the environment is not installed or you used the system `python`; use `.venv/Scripts/python.exe` after `pip install -r requirements.txt` |
| `cannot be loaded because running scripts is disabled` (PowerShell) | run the quick-start script with `powershell -ExecutionPolicy Bypass -File scripts/quickstart_windows.ps1`; nothing system-wide is changed |
| `error: [WinError 10048] Only one usage of each socket address ...` when starting the dashboard | another program (or another dashboard) already listens on port 8765: stop it, or add `--port 8800` |
| The *Research model* tab says the model is not loaded | expected without the research artifacts (section 7); start with `--research-model` after generating them |
| `run_dashboard.py --research-model` exits with `error: ... model.joblib does not exist` or `could not be loaded` | the model file is missing or damaged; regenerate it with `scripts/train_xgboost.py` (section 7) |
| `[SKIPPED] ... needs torch` | PyTorch is optional; install `requirements-gnn.txt` only to run the neural tests and experiments |
| `NOT RUN (withheld artifact missing)` | expected on a fresh clone; the named file is not published (section 7) |
| Clone fails with `Filename too long` | clone into a short folder or use `git -c core.longpaths=true clone ...` (the longest path is 122 characters) |
| Hash checks fail after checkout | `core.autocrlf` rewrote line endings; `.gitattributes` (`* -text`) prevents this in a normal clone. Re-clone without overriding it |
| `error: refusing to read '...test...'` | intended: TEST-named paths are refused because the held-out period is sealed |
| In `cmd.exe`, `.venv/Scripts/python.exe` is not recognized | write `.venv\Scripts\python.exe` in `cmd.exe` |

## 13. Reproducing the demonstration on another machine

1. `git -c core.longpaths=true clone https://github.com/arunsubramanian71-droid/Adaptive-Temporal-Fraud-Demo.git` and `cd Adaptive-Temporal-Fraud-Demo`. A GitHub "Download ZIP" copy also works (same test results); extract it to a short folder.
2. Install Python 3.12, then follow section 5 (Option A or B).
3. Expect: `SETUP OK`, 306 of 327 core tests passing with 21 NOT RUN and 0 failing, the five-stage demonstration, and the dashboard on http://127.0.0.1:8765/ in demo mode.
4. Without the IEEE-CIS data the research-model tab cannot score; everything else in the dashboard works, including the recorded evidence tables and the candidate-community cards.
5. For the research model, obtain the data under the competition rules and follow section 7.

A collaborator's tour of the code (where each part lives and where to start reading) is in `submission/COLLABORATOR_GUIDE.md`.

## 14. Further documentation

| File | Content |
|---|---|
| `submission/README.md` | the submission package index and detailed usage notes |
| `submission/MANUSCRIPT.md` | the research write-up (validation-period study) |
| `submission/PROFESSOR_DEFENSE_GUIDE.md`, `submission/DEMO_GUIDE.md` | presenting and defending the project; the demonstration script |
| `submission/COLLABORATOR_GUIDE.md` | code tour for a new developer |
| `submission/ARTIFACT_INVENTORY.md` | what is in the repository, what is not, and how to obtain it |
| `submission/TRACEABILITY_MATRIX.md` | every requirement of the original brief, with source, status and evidence |
| `submission/HELDOUT_DECISION_PACKAGE.md` | the frozen held-out protocol, options and open decisions |
| `submission/FINAL_COMPLETION_RECORD.md`, `submission/MASTER_COMPLETION_CHECKLIST.md` | what is done, verified and blocked |
| `submission/REPRODUCIBILITY.md` | per-experiment commands, inputs and outputs |
| `reports/PROJECT_SYNTHESIS_REPORT.md` | phase-by-phase research record |

**Licence and data.** The code is the author's research project. The IEEE-CIS data belongs to its providers and is governed by the Kaggle competition rules. Do not commit the data or anything derived from it row by row.
