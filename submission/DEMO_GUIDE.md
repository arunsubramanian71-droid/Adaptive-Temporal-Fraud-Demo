# Demonstration guide (about 10 minutes)

Purpose: show that the project is a working, tested software system and show, without over-claiming, what it found. Two parts: a live software demonstration on synthetic data (3 minutes), then a walk through the recorded research results (7 minutes). **Say at the start: "the live demo uses synthetic data and reproduces no research result; the research results are validation-period results and no held-out evaluation has been done."**

## Before the session

```
.venv/Scripts/python.exe -B scripts/run_tests.py
```
Expect `ALL TEST MODULES PASSED` (about two to three minutes; 325 tests in 33 modules). Have `reports/PROJECT_SYNTHESIS_REPORT.md`, `experiments/comparison_B_graph_features/outputs/B_RESULTS.md` and `submission/MANUSCRIPT.md` open.

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

Open http://127.0.0.1:8765/ in a browser (leave the terminal running; Ctrl+C stops it). Walk the four tabs: **Evidence (validation)** shows the recorded tables and the standing banner; **Score transactions**: click *Load synthetic sample*, wait for the message, then *Score* (the answer is labelled synthetic); **Candidate communities** shows 24 aggregate-only cards (say: candidates, not confirmed to be fraudulent or coordinated; no member identities exist in this output); **Status and limits** shows that the held-out evaluation has not been performed.

## Part 2: the recorded research (7 minutes)

1. **Question and protocol (1 min).** Does relational and temporal structure improve fraud detection over strong tabular models, especially on future data and under drift? Chronological split: train days 1-121, validation 122-151, held-out 152-182 sealed. Everything is validation-period.
2. **The consolidated table (2 min).** `reports/PROJECT_SYNTHESIS_REPORT.md` section 1.1 and `B_RESULTS.md`. Figures 1-3 are in `submission/figures/`. Walk the rows: tabular models, the new graph-feature model B, GraphSAGE, GAT, Temporal GraphSAGE, controls. Point at the F rows and say they are on a different basis and not comparable.
3. **Verdicts (2 min).** H1-H5 in `MANUSCRIPT.md` section 6: what is supported, what is not, and why (GAT never converged; the synthetic benchmark failed a frozen gate; no held-out evaluation).
4. **Honest gaps (2 min).** `COMPLETION_MATRIX.md`: the held-out evaluation is blocked pending an authorization, graph-level explanations exist only for Temporal GraphSAGE seed 42 (one registered sanity criterion failed at GPU float noise), and Phase 12's before/after hash criterion is BLOCKED (partially verified).

## If something breaks live

* Demo fails to import xgboost or pandas: the environment is not installed; run the setup in `README.md` section 2.
* Demo output differs from this guide's numbers: only the structure is promised; the files in `demo_output/` are deterministic for a fixed environment, and `tests/test_inference_demo.py` checks that two runs are byte-identical.
* Never open or score a TEST-named file during the demonstration.
