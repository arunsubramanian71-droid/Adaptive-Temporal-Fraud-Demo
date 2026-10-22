# Adaptive Temporal Graph Intelligence for Fraud Detection & Coordinated Fraud Discovery: submission package

A validation-period study on the IEEE-CIS Fraud Detection data of tabular, graph and temporal-graph models, adaptive updating under drift, and exploratory discovery of *candidate* coordinated communities. **All research results are validation-period results (days 122-151). No held-out (TEST) evaluation has been performed.** The package contains the code, tests, recorded results, a runnable demonstration and the manuscript. Read `MANUSCRIPT.md` for the findings and `COMPLETION_MATRIX.md` for what is and is not finished. `MASTER_COMPLETION_CHECKLIST.md` is the consolidated requirement-by-requirement status, `TRACEABILITY_MATRIX.md` maps every requirement of the original brief to code, a verification command and the observed result, `PROFESSOR_DEFENSE_GUIDE.md` is the presenter's technical guide, and `HELDOUT_DECISION_PACKAGE.md` holds the frozen held-out protocol, the amendment options and the open decisions.

*(The title's final words are abbreviated on purpose: the project's terminology rule forbids ring wording for real-data outputs, and no group found here is confirmed to be fraudulent or coordinated.)*

## 1. Two kinds of "run it"

| What | Needs | Time | What it proves |
|---|---|---|---|
| **Software demonstration** (section 3) | Python environment only; builds its own synthetic data | about a minute | the real modules run end to end: graph features, tabular models, history-based scoring, explanations, candidate-community discovery. **It reproduces no research result.** |
| **Research experiments** (section 6) | the IEEE-CIS files, the recorded artifacts, a GPU for the neural models | hours | the recorded numbers; you normally read the recorded outputs instead of re-running |

## 2. Setup (tested on Windows 11, Python 3.12.0)

Python 3.12 is the recommended interpreter: it matches the recorded manifests and passes every executed test in a clean environment; Python 3.14.3 differs only in one interpreter-version metadata test (`REPRODUCIBILITY.md`).

```
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

The commands are written for PowerShell and Git Bash, where `.venv/Scripts/python.exe` works as shown. **In cmd.exe write the paths with backslashes** (`.venv\Scripts\python.exe -B scripts\check_setup.py`); a forward-slash path is not accepted there (checked). On Linux or macOS use `.venv/bin/python` in place of `.venv/Scripts/python.exe` (not tested there). `requirements.txt` covers the demonstration, the scorer, comparison B and the Phase 10-12 tests. The neural-graph code additionally needs PyTorch and PyTorch Geometric (`requirements-gnn.txt`; the recorded runs used an AMD ROCm build of PyTorch; do not install the compiled PyG extensions). No secrets or accounts are needed anywhere.

## 3. Setup check, demonstration and dashboard

```
.venv/Scripts/python.exe -B scripts/check_setup.py
```

Checks the Python version, every pinned requirement, the optional neural-network stack and the presence of the research data files (existence only; nothing is opened). It prints an actionable message for anything missing.

### 3.1 Demonstration

```
.venv/Scripts/python.exe -B scripts/demo_end_to_end.py --out-dir demo_output
```

It prints five stages (synthetic stream, tabular-only versus tabular-plus-graph-features models, history-based scoring with explanations, candidate-community discovery with the Phase 9 modules, invariants) and writes `demo_output/` (scores, community table, `demo_summary.json`; byte-identical on re-run). It refuses to write inside `experiments/`. `DEMO_GUIDE.md` is the presenter's script.

### 3.2 Dashboard and API (Phase 13)

```
.venv/Scripts/python.exe -B scripts/run_dashboard.py --demo
```

Open http://127.0.0.1:8765/ . Standard library only; binds to 127.0.0.1. Tabs: the recorded validation-period evidence tables (Phase 12 and comparison B, with the imbalance study), scoring of posted rows (the demo bundle is trained on synthetic data and every response says so), the 24 aggregate-only candidate-community cards of the exploratory Phase 9 output, and the status and limits. JSON endpoints: `/health`, `/api/status`, `/api/model`, `/api/evidence`, `/api/communities`, `/api/sample` (demo only) and `POST /api/score`. The page shows no transaction-level or held-out data. The scope of this phase is a **reconstruction** (the original Phase 13 specification was not recovered): see `experiments/phase13_serving_dashboard/SCOPE_AND_ACCEPTANCE.md` section 1.

**Research-model mode.** `scripts/run_dashboard.py --demo --research-model` also serves the recorded comparison-A XGBoost (`experiments/xgboost`) on `GET /api/research/model` and `POST /api/research/score`, through the same TRAIN-fit encoder and the recorded validation threshold, with native per-feature contributions for alerted rows. Rows must follow the processed-feature schema (464 input columns, listed by `/api/research/model`); labels in posted rows are ignored; about 150 rows fit in one request (1 MB body limit). `scripts/make_research_sample.py` writes validation rows (label removed) to your temporary folder for the dashboard's *Research model* tab; it refuses to write inside the repository. The synthetic demo bundle and the research model are separate modes, and every response says which one produced it. The trained model file is not distributed with the repository (regenerate with `scripts/train_xgboost.py`). Verified by `tests/test_research_serving.py` (HTTP scores equal direct inference and the recorded validation predictions) and the live check `scripts/verify_research_serving_e2e.py`.

**Input rules (both modes).** `TransactionID` must be an integer (it is reported back unchanged; in research rows it may be omitted or null); `TransactionAmt` in the demo/bundle mode must be a finite number (a missing amount would corrupt the backward-looking amount features of later rows of the same entity). Every invalid request gets a 400 with a message that says what to fix; a damaged bundle or model file is refused at start-up with exit code 2.

**Real-browser check (optional).** `scripts/verify_dashboard_browser.py` starts the documented server command, drives the page in headless Chrome (desktop 1366x900 and phone 390x844, light and dark colour schemes) and records 123 checks: navigation and keyboard use, scoring and research scoring, file selection, empty, error and unavailable states, a double click, a server stop and restart, contrast, horizontal overflow, console errors and failed requests. It needs Playwright in a separate environment (`requirements-browser-check.txt`; the installed Chrome or Edge is used, nothing is downloaded). In PowerShell:

```
python -m venv "$env:TEMP\fraud-browser-check"
& "$env:TEMP\fraud-browser-check\Scripts\python.exe" -m pip install -r requirements-browser-check.txt
.venv/Scripts/python.exe -B scripts/make_research_sample.py --n 10 --out "$env:TEMP\fraud_research_rows.json"
& "$env:TEMP\fraud-browser-check\Scripts\python.exe" -B scripts/verify_dashboard_browser.py --research-rows "$env:TEMP\fraud_research_rows.json"
```

The record and 25 screenshots are in `experiments/phase13_serving_dashboard/test_results/browser/` (123 of 123 passed on 2026-10-22). The published screenshots never show an IEEE-CIS row: the research tab is photographed with synthetic placeholder rows, and the real validation rows are used only for the functional checks.

### 3.3 Scoring files from the command line (the production-style path)


```
.venv/Scripts/python.exe -B scripts/score_transactions.py train --train train.csv --valid valid.csv --features f0,f1,f2 --out-dir bundle/
.venv/Scripts/python.exe -B scripts/score_transactions.py score --bundle bundle/ --history history.csv --new new.csv --out scores.csv --explain
```

A scored row only ever sees earlier rows; labels are ignored when scoring; TEST-named files are refused (exit code 2).

## 4. Tests

```
.venv/Scripts/python.exe -B scripts/run_tests.py          # full environment (needs requirements-gnn.txt for the neural-network tests)
.venv/Scripts/python.exe -B scripts/run_core_tests.py      # any environment: skips, and names, the modules that need PyTorch
```

The full run covers all 39 test modules (the recorded run is `submission/test_results.json`, written by `scripts/run_completion_tests.py`). Two clean environments installed only from `requirements.txt` (no PyTorch) were checked: on Python 3.12.0 the core runner passes every executed test; on Python 3.14.3 it passes every executed test except that one registered Phase 12 test reports the different Python version string stored in its manifest (reported, never counted as passed; an environment-independent companion test passes). Counts are in `FINAL_COMPLETION_RECORD.md`; the modules that need PyTorch (five in a core-only environment: test_f_runner_gate, test_neural_graph_evidence, test_phase8_adaptive, test_trigger_walkforward, test_walkforward_ext) are skipped and named, never counted as passed. Mutation-sensitivity checks for the newest code: `scripts/check_comparison_b_test_sensitivity.py`. None of the tests trains a research model or reads TEST data. (Until 2026-10-22 one split-count test opened the held-out feature table and checked its row count, fraud count and day range against the Phase 1 figures; no model was scored on it. It now reads only TRAIN and VAL; see `HELDOUT_DECISION_PACKAGE.md` section 1a.)

**Not published:** raw data, model files and checkpoints, feature tables, row-level prediction files, and (since 2026-10-22) six per-transaction evidence files of 300 validation transactions each; earlier commits of this public repository still contain those six files. `demo_output/` is generated by the demo and is not committed.

**Core tests versus artifact-dependent tests (fresh GitHub clone).** The repository does not contain the raw data, model files and checkpoints, feature tables or row-level prediction files (`.gitignore` publication policy). Twenty-two tests need such artifacts (thirteen research-record tests, three research-model serving tests, five tests that read the six unpublished per-transaction evidence files, and the split-count test, which until 2026-10-22 passed silently without its feature tables); they are listed, with the file patterns they need, in `tests/artifact_requirements.json`, and their assertions are unchanged. `scripts/run_core_tests.py` reports a test whose artifacts are missing as **NOT RUN (withheld artifact missing)**, never as passed, and names the missing pattern; `scripts/run_core_tests.py --require-artifacts` turns every such test into a failure (use it in the full project directory, where all 376 tests pass, see `submission/test_results.json`). Measured on a fresh clone of commit `6120da2` with Python 3.12.0 and `requirements.txt` only (`submission/clean_env_results/fresh_clone_python3.12.0_core_tests_2026-10-22.txt`): 285 of 305 tests pass, 20 are NOT RUN for missing artifacts (the registered twenty-one minus one inside a skipped PyTorch module), none fails, five PyTorch-dependent modules are skipped and named; exit code 0, and exit code 1 with `--require-artifacts`. (One of those 285 was the split-count test passing silently; it is now NOT RUN on a clone. On a fresh clone of the integrity-pass release `8a9e983` (`fresh_clone_python3.12.0_core_tests_2026-10-22_8a9e983.txt`): 290 of 311 pass, 21 NOT RUN, none fails, exit 0; exit 1 with `--require-artifacts`.) To run the thirteen, rebuild the artifacts with the project's build and training scripts (`scripts/build_processed_data.py`, `scripts/train_*.py`) on the IEEE-CIS files. On Windows, clone into a short folder or use `git -c core.longpaths=true clone ...`: the longest repository path has 122 characters, so a clone folder deeper than about 135 characters exceeds the 260-character path limit. A `.gitattributes` (`* -text`) keeps checked-out bytes equal to committed bytes: the integrity hashes are byte-exact and a Windows `core.autocrlf` rewrite would break them.

## 5. Repository map

| Path | Content |
|---|---|
| `src/data`, `src/features` | loading, splits, preprocessing, leakage-safe historical features, **`graph_features.py` (comparison B)** |
| `src/graph`, `src/models`, `src/adaptive` | heterogeneous graph, GraphSAGE / GAT / Temporal GraphSAGE, metrics, adaptive updating |
| `src/coordination`, `src/evolution` | candidate-community discovery, synthetic benchmark, tracking (Phases 9-10) |
| `src/explain`, `src/evidence`, `src/inference`, `src/demo`, `src/serving`, `src/heldout` | explanations and cards, the Phase 12 evidence table builder, the scorer, the demonstration, the local API and dashboard (Phase 13), the gated held-out evaluator |
| `scripts/` | one script per experiment or check (see `reports/PROJECT_SYNTHESIS_REPORT.md` section 7 for the phase map) |
| `experiments/` | recorded results, designs and integrity records per phase; `comparison_B_graph_features/`, `imbalance_check_xgboost/`, `unweighted_graphsage/`, `unweighted_temporal_graphsage/`, `final_heldout_protocol/` and `phase13_serving_dashboard/` are new |
| `reports/` | phase reports and `PROJECT_SYNTHESIS_REPORT.md` |
| `tests/` | unit and property tests |

## 6. Reproducing the research experiments

Place the competition files in `data/raw/` (`train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv`; they are not redistributed here). Then, in order: `scripts/build_processed_data.py`, `scripts/run_validation.py`, `scripts/build_historical_features.py`, `scripts/build_graph.py`, the training scripts of each phase, and for comparison B `scripts/run_comparison_b.py` followed by `scripts/build_b_outputs.py`. The phase-by-phase map of scripts, tests and integrity records is `reports/PROJECT_SYNTHESIS_REPORT.md` section 7. Training scripts write into existing `experiments/` folders, so **copy a folder aside before re-running a phase**.

**Do not repeat:** `scripts/train_gat_convergence_300epoch.py` (aborted Phase 6C; its outputs are excluded); anything that opens `data/features/test_features.parquet`, `data/raw/test_*.csv` or any TEST-named file (sealed; see `experiments/final_heldout_protocol/HELDOUT_EVALUATION_PROTOCOL.md`); re-running Phase 12 finalization against the current folder without the pre-change manifest in mind.

Environment records: library versions are in `requirements.txt`; each experiment folder holds its `config.json` / `metadata.json` with seeds, thresholds and split boundaries.

## 7. Provenance and integrity

Per-file SHA-256 records exist for Phases 9-12 and the new work: `experiments/phase9_coordinated_fraud/phase9_closure_integrity.json`, `experiments/phase10_community_evolution/INTEGRITY*.json`, `experiments/phase11_explainable_evidence/INTEGRITY_PHASE11.json`, `experiments/phase12_validation_evidence_table/INTEGRITY_PHASE12.json`, and `submission/INTEGRITY_SUBMISSION.json`. **Phase 12 is not formally complete:** its acceptance criterion AC12 (before/after hashes of every existing file) is BLOCKED, partially verified, 300 of 651 pre-existing files covered with zero contradictions and 322 not retrospectively verifiable. That limitation is not resolved by anything in this package.


**Publication note.** Transaction-level prediction and contribution files (validation period; ids, labels, scores) exist under `experiments/` as in earlier phases (for example the arms of comparison B and the Phase 11 selections). None is inside `submission/`. Whether to publish them, and the ignore rules that cover them, are the project owner's decision; `.gitignore` was not modified.

## 8. What the package claims, in one paragraph

On the validation period, a tuned XGBoost on transaction features is at least as good as every neural graph model tried, with or without class weighting; removing class weighting raises XGBoost's PR-AUC by about 0.05 (a pre-registered check); adding forty label-free backward-looking graph features to XGBoost does not reliably change it (pre-registered rule: not established, `experiments/comparison_B_graph_features/outputs/B_RESULTS.md`); recurrent entity memory gives a small cold-start-concentrated gain over its controls; periodic weight updates on newly revealed labels reduce within-validation degradation under the pre-registered rule (H4, three seeds); and coordinated-community discovery remains not established because the synthetic benchmark failed a frozen validity gate. **The held-out evaluation is not done** (a frozen protocol, a separately hashed Amendment 1 adding the adaptive arms, the historical reference line and the unweighted variants, and validation-checked evaluators exist; it awaits the user's authorization). A new, separately pre-registered recovery benchmark (Phase 9b, corrected interference gate) supports H5a-9b without changing the original verdict, and a pre-registered relationship-level ablation shows that the pseudo-account link carries most of the learned temporal model's relational signal. A local dashboard and API (Phase 13, reconstructed scope) expose the recorded aggregates and a scoring endpoint. Nothing is claimed about held-out performance, significance or causation. See `FINAL_COMPLETION_RECORD.md` and `COMPLETION_MATRIX.md` for the full status.
