# Initial state, 2026-10-23 (before any edit in this session)

Machine-readable copy: `initial_state.json` in this folder. Raw outputs: `evidence/baseline/`. Nothing below was edited after the fact.

## Repository

| Item | State |
|---|---|
| Checkout | `~/Sandbox/Adaptive-Temporal-Fraud-Demo`, branch `main`, HEAD `975b0db` ("Handoff release ..."), 35 commits, remote `arunsubramanian71-droid/Adaptive-Temporal-Fraud-Demo` |
| What it is | the sanitized **public demonstration** repository (filtered history). It is not the complete research repository: the IEEE-CIS data, feature tables, trained models and per-transaction evidence files are withheld by design (`submission/ARTIFACT_INVENTORY.md`) |
| Uncommitted change found | `experiments/phase13_serving_dashboard/test_results/e2e_verification.json`, rewritten by a run of `scripts/verify_dashboard_e2e.py` on this Mac at 12:39 (31 of 31 checks passed). Its hash no longer matches `submission/INTEGRITY_SUBMISSION.json`. Left untouched; copy in `evidence/baseline/` |
| Untracked | `demo_output/` (synthetic demonstration output, 12:39) and `presentation/` (the owner's deck and notes; not touched) |
| Running | a dashboard on 127.0.0.1:8765 started 12:41; left running |

## Machine and environment

* macOS `27.0.1` on Apple M4 (arm64), 16 GB memory, 34 GiB free disk; Chrome, Firefox and Safari installed; Homebrew `libomp` installed.
* `.venv`: created by uv from CPython `3.12.13`, all ten `requirements.txt` packages at their pinned versions, **no pip** inside, no PyTorch. `python3.12` is **not** on PATH (Homebrew provides 3.13 and 3.14).
* Research data and models: absent from the checkout and from the rest of this Mac (searched). The held-out period was never opened, then or later.

## Baseline checks (unchanged code)

| Command | Result |
|---|---|
| `scripts/check_setup.py` | exit 0, `SETUP OK for the demonstration` |
| `scripts/run_core_tests.py` (project `.venv`) | 306 of 327 pass, 21 NOT RUN (withheld artifacts), 0 fail, 5 modules SKIPPED (need PyTorch); 16 s |
| `scripts/run_core_tests.py` (isolated environment with `requirements-gnn.txt`, CPU PyTorch `2.9.1`) | 359 of 381 pass, 22 NOT RUN, 0 fail, all 40 modules run; 32 s |

The README's fresh-clone figure (306 of 327, 21 NOT RUN, 0 fail) and the brief's figure (306 pass, 0 fail, 21 not run) both match on this Mac. Every NOT RUN test needs a withheld file named in `tests/artifact_requirements.json`; none needs a GPU, a credential or a service.
