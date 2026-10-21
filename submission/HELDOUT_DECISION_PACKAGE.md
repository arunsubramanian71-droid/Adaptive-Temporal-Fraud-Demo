# Held-out evaluation: decision package (nothing requested, created or run)

Dated 2026-10-21. **The held-out period (days 152-182) has not been opened, read, hashed or scored. `AUTHORIZATION.txt` does not exist and no held-out result exists. This package is the information needed to decide; it authorizes nothing.** The evaluation is one-shot and irreversible: once any held-out metric exists, the period can never again be untouched test data.

## 1. The frozen protocol (base)

* File: `experiments/final_heldout_protocol/HELDOUT_EVALUATION_PROTOCOL.md`, SHA-256 `59bf1565fb16653a02a2d743cc445e114fb2a6a42f1de7b2fa56261e6f68c806` (verified in a fresh clone of GitHub; sidecar matches).
* Models: 19 frozen artifacts (logistic regression, random forest, XGBoost; B_full; GraphSAGE, GAT and the schedule-matched static control, three seeds each; Temporal GraphSAGE and the memory-ablated control, three seeds each); recorded validation thresholds transferred, nothing re-fitted or re-tuned; metrics as in the Phase 12 table; pre-registered descriptive checks (protocol section 5).
* Code-readiness gate (validation): 19 of 19 reproduce the recorded predictions (maximum score difference 9.54e-07).
* Authorization sentence (protocol section 9), to be placed by you in `experiments/final_heldout_protocol/AUTHORIZATION.txt` together with the hash above: *I authorize one-time evaluation of the frozen models listed in section 2 on the held-out period (days 152-182) under `experiments/final_heldout_protocol/HELDOUT_EVALUATION_PROTOCOL.md`, once the code-readiness gate in section 6 has passed. I understand this irreversibly uses the held-out period and that results will be reported whatever they are.*
* Command afterwards: `.venv/Scripts/python.exe -B scripts/run_heldout_evaluation.py --split heldout` (refuses without the file, refuses to run twice).

## 2. Amendment 1 (separately hashed; optional modules)

* File: `experiments/final_heldout_protocol/AMENDMENT_1.md`, SHA-256 `37a1c0577a8ef28530a6120a027890c11ec458ca97329d14a05108db21a27e6e`. Honoured only if `AUTHORIZATION.txt` also quotes this hash and a sentence `I authorize the following Amendment 1 modules: <letters>`.
* **F**: adaptive arms (frozen, adaptive, replay control) on five 6-day windows 152-181, entity memory built over days 1-151, TRAIN-only hyperparameters, locked H4 rules. **H**: historical-feature XGBoost reference line. **U**: eight unweighted models.
* Readiness on validation: H and U PASS; F deterministic checks PASS (21 of 21), trained-arm window-mean gate FAIL as written (1 of 6: E seed 123, 0.0104 against 0.01).

## 3. Options and exact consequences

| Option | What it evaluates | What it leaves unmet | Consequence |
|---|---|---|---|
| **A. Base protocol only** | 19 base models | the brief's "static vs adaptive on future periods" (cell F, H4), the historical reference line, the weighting reading | irreversibly spends the period; F, H, U cannot be added later without a second look at the same period |
| **B. Base + F, H, U (recommended)** | everything above plus the adaptive arms, the reference line and the unweighted variants | nothing within the frozen scope | one-shot use for 19 + 8 models, 3 arms x 3 seeds, 1 reference; F is authorized knowing its trained-arm readiness gate failed as written (see section 4) |
| **C. Base + F, H** | as B without the weighting reading | the weighting question | U cannot be added later |
| **D. Do not authorize** | nothing | every held-out requirement | requirement stays BLOCKED; everything else in the submission is unaffected |

## 4. The Module F readiness gate: options

The failure is **numerical, not a code defect** (`experiments/phase8_state_continuity_supplement/diagnostics/`): identical-seed GPU runs differ (window means 0.4330, 0.4273, 0.4281; the five-run SD of the window mean is 0.0044; individual scores in later windows differ by up to about 0.97) while identical-seed CPU runs are bitwise equal and the deterministic parts (F, window 1) are exact to 1e-7. The registered 0.01 tolerance compared one run with one earlier run, which is tighter than the measured noise (a difference of two runs has SD about 0.006).

1. **Accept the gate as failed-as-written** and authorize F knowingly (my reading: the deterministic code path is proved; the trained arms are noisy).
2. **Approve the proposed Amendment 2** (`experiments/final_heldout_protocol/AMENDMENT_2_PROPOSAL_F_GATE.md`, hashed in its sidecar; not active). Option A2-a (recommended, prospective): measure the run-to-run SD of every trained-arm cell with five identical-seed validation runs, then require a fresh readiness repeat within 3 x sqrt(2) x that SD; the 21 deterministic checks keep 1e-6. Option A2-b (retrospective, post hoc): apply 0.0187 to the existing record, which would pass all six cells. An authorization would have to quote the new hash.
3. **Do not authorize F.**

## 5. Risks and limits stated in advance

One 31-day period (30 scored days for F), one dataset; three seeds are not independent replications; concept drift may move every model; GPU nondeterminism adds about 0.004 of noise to a trained arm's window mean, so small held-out F differences cannot be read finer than that; zero label delay for F and H (H's features are label-derived); thresholds transferred from validation (alert counts will shift); the adaptive arms use validation labels; GAT is reported non-converged.

## 6. What cannot be completed without this evaluation

The brief's held-out evaluation itself; static versus adaptive **on a future period** (the ADAPTATION section); the future-time wording of H3 and H4; the descriptive held-out checks of protocol section 5 (including whether the best tabular model stays above the neural cells). Validation results are not substituted for them anywhere in this project.

## 7. Other open decisions (unrelated to held-out access)

* **Phase 13 scope:** the original specification was never recovered; the dashboard and API are a reconstruction (`experiments/phase13_serving_dashboard/SCOPE_AND_ACCEPTANCE.md` section 1). Confirm or correct.
* **Phase 9 gate G4:** failed on seeds 42 and 2024 and stays failed; the original H5a stays not established. A new, separately pre-registered benchmark (Phase 9b, corrected interference gate, fresh seeds) has since been run: all gates passed and **H5a-9b is supported** (`experiments/phase9b_recovery_benchmark/outputs/RESULTS_9B.md`). It is a separate result and does not change the original verdict.
* **Phase 12 AC12:** BLOCKED as registered. A per-file recovery found pre-build evidence for 550 of 651 files (434 content-identical to the first Git commit, pushed 2026-10-16 per GitHub's server log) with 0 contradictions and none for 101; accepting it is the evaluator's decision.
* **Public Git history:** six transaction-level evidence files remain in the history of the public repository (commits `b28a03b`, `edb6167`) although they are no longer tracked: make the repository private, or approve a history rewrite.
