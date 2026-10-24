# Demonstration guide (about 10 minutes)

Purpose: show that the project is a working, tested software system and show, without over-claiming, what it found. Two parts: a live software demonstration on synthetic data (3 minutes), then a walk through the recorded research results (7 minutes). **Say at the start: "the live demo uses synthetic data and reproduces no research result; the research results are validation-period results and no held-out evaluation has been done."**

**macOS and Linux:** every command below is written for Windows; use `.venv/bin/python` instead of `.venv/Scripts/python.exe` (on macOS, `brew install libomp` once; README section 4). The macOS path was rehearsed end to end on 2026-10-23 (Apple Silicon): setup, tests, demonstration and dashboard.

## Before the session

```
.venv/Scripts/python.exe -B scripts/run_tests.py
```
Expect `ALL TEST MODULES PASSED` (about two to three minutes) on the machine that holds every withheld artifact: 381 tests in 40 modules on 2026-10-22, and 396 in 42 modules since the 15 tests added on 2026-10-23 (those 15 were run on macOS only). On a fresh clone, or any machine without the data and model files, run `scripts/run_core_tests.py` instead: it reports the artifact-dependent tests as NOT RUN. Have `reports/PROJECT_SYNTHESIS_REPORT.md`, `experiments/comparison_B_graph_features/outputs/B_RESULTS.md` and `submission/MANUSCRIPT.md` open.

## Part 1: live software demonstration (3 minutes)

```
.venv/Scripts/python.exe -B scripts/demo_end_to_end.py --out-dir demo_output
```

| Stage printed | What to say | What it exercises |
|---|---|---|
| `[1]` synthetic stream | "forty days of fake transactions with a planted shared-device burst; labels depend on a numeric feature and on a set of compromised devices" | `src/demo/synthetic.py` (fixture only) |
| `[2]` two models | "same recipe, with and without 40 backward-looking graph features (comparison A versus B); numbers are on a synthetic holdout and mean nothing for the research" | `src/features/graph_features.py`, the XGBoost protocol, `src/models/metrics.py` |
| `[3]` scoring | "a new transaction is scored from stored history only; the explanation lists the model's own tree contributions" | `src/inference/scorer.py`, `src/explain/xgb_contrib.py` |
| `[4]` candidate communities | "the project's label-free Louvain-based discovery finds the planted burst on the first day the window covers it; on real data such groups are only *candidates*" | `src/coordination/*` |
| `[5]` invariants | "labels are ignored, and removing later rows never changes an earlier score: the leakage rule, checked live" | scorer tests |

Optional (1 minute): `... scripts/score_transactions.py score ... --new test_new.csv ...` with any TEST-named file shows the guard refusing with exit code 2.

Likely questions: *Is the demo the research?* No; the recorded experiments need the dataset and a GPU. *Why synthetic?* The research data is not redistributed and the neural models need a GPU; the demo runs the same modules on a fixture so the mechanics can be inspected.

## Part 1b: the dashboard (2 minutes)

```
.venv/Scripts/python.exe -B scripts/run_dashboard.py --demo
```

Open http://127.0.0.1:8765/ in a browser (leave the terminal running; Ctrl+C stops it). Walk the six tabs. **Overview** opens first: read the research question aloud, point at the status tiles (research data and model not on this machine, held-out period sealed, synthetic demo bundle, API responding) and the SYNTHETIC DEMO MODE chip, then the chart. Say: "every bar is a recorded validation result read from the result files when the page loads; the dashed bar is a reference line that uses earlier labels, not a comparable model; tabular XGBoost is at least as good as every graph model". The verdict list beside it carries the recorded verdicts (H1 not established, H2 contradicted, H4 supported, H5 not established under the original protocol). **Evidence (validation)** shows the recorded tables and the standing banner; **Score transactions**: click *Load synthetic sample*, wait for the message, then *Score* (the answer is labelled synthetic); **Candidate communities** shows 24 aggregate-only cards (say: candidates, not confirmed to be fraudulent or coordinated; no member identities exist in this output); **Status and limits** shows that the held-out evaluation has not been performed. The **Research model** tab is part 1c. To show error handling, replace the request text with `{"new": []}` and click *Score*: the page shows the server's 400 message.

## Part 1c: the research model in the dashboard (2 minutes)

```
.venv/Scripts/python.exe -B scripts/make_research_sample.py --n 25
.venv/Scripts/python.exe -B scripts/run_dashboard.py --demo --research-model
```

Open the *Research model* tab, choose the JSON file the first command wrote (in your temporary folder), and click *Score with the research model*. Say: "this is the recorded comparison-A XGBoost, served exactly as evaluated; the scores equal the recorded validation predictions (tested); it is not the synthetic demo bundle; a score is not evidence of fraud." Point at the model-file hash, the recorded threshold and the contributions for alerted rows. Needs the local model file and validation features (not in the repository).

## Part 2: the recorded research (7 minutes)

1. **Question and protocol (1 min).** Does relational and temporal structure improve fraud detection over strong tabular models, especially on future data and under drift? Chronological split: train days 1-121, validation 122-151, held-out 152-182 sealed. Everything is validation-period.
2. **The consolidated table (2 min).** `reports/PROJECT_SYNTHESIS_REPORT.md` section 1.1 and `B_RESULTS.md`. Figures 1-3 are in `submission/figures/`. Walk the rows: tabular models, the new graph-feature model B, GraphSAGE, GAT, Temporal GraphSAGE, controls. Point at the F rows and say they are on a different basis and not comparable.
3. **Verdicts (2 min).** H1-H5 in `MANUSCRIPT.md` section 6: what is supported, what is not, and why (GAT never converged; the original synthetic benchmark failed a frozen gate, and a separately pre-registered second benchmark, Phase 9b, supports H5a-9b without changing that verdict; no held-out evaluation).
4. **Honest gaps (2 min).** `COMPLETION_MATRIX.md`: the held-out evaluation is blocked pending an authorization, graph-level explanations exist only for Temporal GraphSAGE seed 42 (one registered sanity criterion failed at GPU float noise), and Phase 12's before/after hash criterion is BLOCKED (partially verified).

## Part 1d: temporal and adaptive mechanics (optional, 2 minutes)

With PyTorch installed (`requirements-gnn.txt`): `.venv/Scripts/python.exe -B tests/test_walkforward_ext.py`. It runs the walk-forward update loop on a toy temporal graph and prints each check (labels used only after a window is scored, replay uses TRAIN days only). Say: "these are the mechanics; the results on the real data are recorded, because they need the dataset and hours of GPU time". The full sequence, with what is live and what is recorded, is `PROFESSOR_DEFENSE_GUIDE.md` section 6b.

## If something breaks live

* Demo fails to import xgboost or pandas: the environment is not installed; run `powershell -ExecutionPolicy Bypass -File scripts/quickstart_windows.ps1` (Windows) or `bash scripts/quickstart_unix.sh` (macOS / Linux), or the setup in the root `README.md` section 5.
* macOS: `Library not loaded: @rpath/libomp.dylib` when XGBoost loads: `brew install libomp`, then start again.
* The dashboard does not start because the port is in use: another dashboard is still running; stop it (Ctrl+C in its terminal) or add `--port 8800` and open `http://127.0.0.1:8800/`.
* `error: [WinError 10048] ...` when starting the dashboard: a dashboard is already running on that port; stop it or add `--port 8800`.
* Demo output differs from this guide's numbers: only the structure is promised; the files in `demo_output/` are deterministic for a fixed environment, and `tests/test_inference_demo.py` checks that two runs are byte-identical.
* Never open or score a TEST-named file during the demonstration.
