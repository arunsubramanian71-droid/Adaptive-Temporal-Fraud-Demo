# Held-out evaluation and open decisions: decision package (nothing authorized, created or run)

Dated 2026-10-22 (updated the same day after an integrity pass, section 1a). **No model has been scored on the held-out period (days 152-182) and no held-out score, metric or result exists. `experiments/final_heldout_protocol/AUTHORIZATION.txt` does not exist.** One correction to earlier versions of this package, which said the period had not been "opened, read or hashed": until 2026-10-22 a split-count regression test opened the held-out feature table on every full test run and checked its row count, fraud count and day range against the Phase 1 figures already in `configs/data.yaml` (section 1a). This package is the information needed to decide. It authorizes nothing.

The evaluation is one-shot and irreversible: once any held-out metric exists, the period can never again be untouched test data. Earlier versions of this package are in the Git history.

## 0. Recommendations in one place

| Decision | Recommendation | Why |
|---|---|---|
| Held-out scope | **Option B**: base protocol + Amendment 1 modules F, H, U, run in two steps (section 4) | the only option that leaves no held-out requirement of the brief unmet; F runs only if its readiness gate passes |
| Module F gate | **A2-a-r1** (`AMENDMENT_2_PROPOSAL_F_GATE_REV1.md`), measured on validation **before** any held-out access | the registered gate fails correct code about half the time (section 3); A2-a as written fails it about one time in five; A2-b is post hoc |
| AC12 | accept the documented alternative evidence as an "accepted alternative", leaving the registered status BLOCKED | the registered before/after method cannot be done after the fact; nothing contradicts the recovered evidence |
| Phase 13 | confirm the reconstruction or name changes (`PHASE13_SCOPE_CONFIRMATION.md`) | the original specification was never recovered |
| Public history | **make the repository private now, and notify Kaggle** (rule 7B); a history rewrite is optional afterwards | reversible, takes effect immediately, keeps all history; a rewrite cannot recall copies already made |

## 1. Verified state (2026-10-22)

| Item | Observed |
|---|---|
| Branch and commits | `main`; local `HEAD` = remote `origin/main` = `6c5b295` at the start of the decision-preparation pass (25 commits) and `2247ff4` at the start of the integrity pass (29 commits; `git ls-remote`) |
| Base protocol | `HELDOUT_EVALUATION_PROTOCOL.md`, SHA-256 `59bf1565fb16653a02a2d743cc445e114fb2a6a42f1de7b2fa56261e6f68c806`, unchanged |
| Amendment 1 | `AMENDMENT_1.md`, SHA-256 `37a1c0577a8ef28530a6120a027890c11ec458ca97329d14a05108db21a27e6e`, unchanged |
| Base readiness (19 models) | **PASS**, re-run 2026-10-22 13:15 after the one-shot guard (section 1a); largest score difference 8.94e-07 (registered tolerance 1e-4; tabular and B exact); the record fingerprints the 22 code files it ran. The previous record is kept under `readiness_val/previous_before_one_shot_guard/` |
| Amendment readiness H, U (9 models) | **PASS**, re-run 2026-10-22 13:16; XGBoost differences 0.0, neural at most 2.98e-07; 24 code files fingerprinted; previous record under `readiness_val_amendment/previous_before_one_shot_guard/` |
| Amendment readiness F | **FAIL as written** (unchanged record): 21 of 21 deterministic checks pass; 5 of 6 trained-arm checks pass; E seed 123 0.0104 against 0.01 |
| New guards (this pass) | Every held-out runner now refuses before reading anything unless its registered readiness gate passed with unchanged code. Module F is therefore refused today. The F runner and the diagnosis script refuse to overwrite completed validation runs (A2-a step 2 as written would have overwritten the registered FAIL record). |
| One-shot and scope guards (integrity pass, section 1a) | A runner refuses once its output folder holds **any** file (before, only the `.completed` flag written at the very end counted, so a crash after the first score file would have allowed a second look, against protocol section 7). Held-out scope is fixed: the base runner evaluates all 19 models together; the H/U runner must run exactly the H/U modules `AUTHORIZATION.txt` authorizes (they share one folder and flag); the F runner refuses any subset of arms or seeds. `src/heldout/oneshot.py`; 7 new tests |
| The six transaction-level files | untracked since `4356385`, kept locally, excluded in `.git/info/exclude`; present in the trees of **21 commits** (`b28a03b` to `de16aa0`; 21 of the 29 commits up to `2247ff4`, re-counted 2026-10-22; no later commit contains them) of the public repository |
| Backup before any history decision | `C:\Users\ArunHBK\Adaptive-Temporal-Fraud_backup_2026-10-22_6c5b295.bundle`: 8,140,366 bytes, SHA-256 `1d6cd399016b768138715f38cdd00af3c78ca352fd67c10045bcf44f012fbf04` (sidecar beside it); `git bundle verify`: complete history |

## 1a. Integrity pass (2026-10-22, afternoon): findings that bear on the held-out decision

* **A test opened the held-out feature table.** `tests/test_splits.py::test_built_split_matches_expected_regression_counts` looped over the three splits and read `data/features/test_features.parquet` in full, then compared its row count (89,326), fraud count (3,114) and day range (152-182) with the Phase 1 figures in `configs/data.yaml`. It ran in every full-suite run in the project directory, including the runs that produced the recorded totals and the first run of this pass. Only pass or fail was printed.
  * What was exposed: the period's row and fraud counts, which were already in the frozen configuration since Phase 1 (before the seal). No model was scored on the period. No per-row value, score or metric was produced or shown.
  * Effect on the held-out evaluation: none on its validity, in my assessment. The frozen models, thresholds and hypotheses were fixed on TRAIN and validation data, and nothing in this read could inform a choice. The statement "not opened or read" in earlier documents was nevertheless inaccurate, and the frozen protocol's own wording (hashed, unchanged) says the period "has not been read".
  * Fixed: the test now reads only the TRAIN and VAL tables. Two new checks fail if any test loops over all three split tables or returns early when its data are missing.
* **The same test passed without its data.** In a fresh clone it printed "SKIP" and returned, which counted as a pass, so the fresh-clone totals overstated the passes by one. It is now registered in `tests/artifact_requirements.json` and reports NOT RUN.
* **One-shot rule.** The runners checked only the `.completed` flag, written last, while prediction files (scores with labels) were written model by model. A crash in between would have allowed a repeat after held-out scores existed, which protocol section 7 forbids. Any file in the output folder now counts as executed. Separately, a held-out run narrower than the registered or authorized scope would have spent the one-shot period for the rest; that is now refused (section 1).
* **Dashboard note.** The dashboard's "No held-out evaluation has been performed" was a fixed sentence and the status flag checked only the base folder. Both now follow the existence of any held-out output in the three folders, so the page cannot keep that sentence after an authorized run.
* **Re-done because the guarded code changed:** both validation readiness gates (PASS, section 1) and the AC12 derived-consistency record, which pins the readiness records' hashes (counts unchanged: 20, 18, 38 and 63).
* **Working tree.** At the start of this pass, 14 tracked files had been reverted to `6c5b295` in the working tree only, after a rewind request in the previous session. That left a sidecar without its document and a failing test. Nothing had been committed or pushed. The committed state `2247ff4` was restored after a backup; the reverted versions are identical to `6c5b295` in history.

## 2. What each part evaluates (exact)

### 2.1 Base protocol (part of options A, B, C)
* **Models (19)**, all trained on days 1-121 and evaluated without re-fitting or re-thresholding:
  * cell A: `experiments/logistic_regression`, `experiments/random_forest`, `experiments/xgboost`;
  * cell B: `experiments/comparison_B_graph_features/outputs/arms/B_full`;
  * cell C: GraphSAGE `experiments/graphsage_robust/seed_{42,123,2024}`;
  * cell D: GAT `experiments/gat/seed_{42,123,2024}`, convergence unresolved and reported as such;
  * cell E: Temporal GraphSAGE `experiments/temporal_graphsage/seed_{42,123,2024}`;
  * Phase 7A controls: `experiments/temporal_ablation/static_schedule_matched/seed_*` and `experiments/temporal_ablation/memory_ablated_temporal/seed_*`.
* **Metrics:**
  * pooled PR-AUC (primary), ROC-AUC, and PR-AUC in three strict segments;
  * P@1 % and R@1 %;
  * at each model's recorded validation threshold: precision, recall, F1, alerts and false positives.
* **Pre-registered descriptive checks** (protocol section 5; direction only, no significance claim):
  1. best tabular model above every neural cell;
  2. E above its two Phase 7A controls;
  3. B above A-XGBoost;
  4. E above C and D.

  These give held-out readings beside H1/H2 (checks 1, 3) and H3 (checks 2, 4). No verdict is re-decided.
* **Outputs:** `heldout_results/`. These include row-level `predictions_heldout_*.parquet` files, which are never published (excluded locally; a test fails if one is ever tracked).
* **Command:** `.venv/Scripts/python.exe -B scripts/run_heldout_evaluation.py --split heldout`.

### 2.2 Amendment 1 modules
* **F, adaptive arms.**
  * Arms and seeds: F (frozen), Ad (adaptive), E (replay control), seeds 42, 123 and 2024.
  * Windows: 152-157, 158-163, 164-169, 170-175, 176-181 (day 182 is not scored).
  * State and updates: entity memory built over days 1-151; TRAIN-chosen K and thresholds; labels of a window are used only after it is scored.
  * Locked H4 rules: Ad's W1-to-W5 drop is smaller than F's in at least 2 of 3 seeds; mean(Ad - F) exceeds the larger seed SD; the same holds for Ad against E.
  * This gives the held-out reading of H4 and the brief's "static vs adaptive on future periods".
  * Outputs: `heldout_results_f/`. Command: `scripts/run_f_walkforward.py --split heldout`.
* **H, reference line.**
  * Historical-feature XGBoost, `experiments/xgboost_historical_control/model.joblib`, threshold 0.914131.
  * Check: is the best base model above or below it?
  * Its features are label-derived and assume zero label delay.
* **U, weighting.**
  * `A_none`, `B_none`, GraphSAGE unweighted ×3, Temporal GraphSAGE unweighted ×3.
  * Check: is the unweighted variant higher or lower than the weighted one, per pair?
* **Outputs and command for H and U:** one shared folder `heldout_results_amendment/` with one one-shot flag. Command: `scripts/run_amendment_evaluation.py --split heldout --modules H,U`.

### 2.3 The options

| Option | Evaluates | Readiness today | Held-out requirements left unmet | Irreversible effect |
|---|---|---|---|---|
| **A. Base only** | 19 base models | PASS | static vs adaptive on future periods (F, H4), the reference line (H), the weighting check (U) | period spent; F, H, U can never be read on an untouched period |
| **B. Base + F, H, U (recommended)** | 19 + 8 models, 3 arms x 3 seeds, 1 reference | base, H, U PASS; **F refused until a valid gate passes** | **B1** (F's gate passes): none. **B2** (it fails): F (H4 future reading) stays unevaluated permanently | period spent for everything run |
| **C. Base + F, H** | as B without U | as B | the weighting check; in C2 also F | as B |
| **D. Do not authorize** | nothing | n/a | every held-out requirement | none; the project stays incomplete under your completion rule |

## 3. Module F readiness gate: full comparison

Evidence (validation only):
* identical-seed GPU runs of E seed 123 have window means 0.4330, 0.4273 and 0.4281, plus 0.4367 and 0.4263 for the two recorded runs;
* the SD over the five runs is 0.0044, and 0.0031 over the three runs within one process;
* two identical CPU runs are bitwise equal;
* window 1 and arm F are exact to 1e-7.

The variation therefore comes from nondeterministic GPU kernels (AMD Radeon RX 7900 GRE, ROCm 7.2, torch 2.9.1), not from the code.

"False failure" below means a correct pipeline failing the gate by noise alone. It is estimated assuming every cell has the measured SD and normal noise, with six independent cells.

| Gate | Trained-arm rule | Deterministic part | False failure, one cell / six cells | Designed before... | Status | Cost | Verdict |
|---|---|---|---|---|---|---|---|
| **Registered** (Amendment 1 section 3) | one repeat within 0.01 of the recorded run | 21 checks at 1e-6 | about **10.8 % / 50 %** (0.01 is 1.6 SD of a two-run difference) | the held-out period and the failure | **FAIL as written** (0.0104) | done | stays in the record; cannot be met reliably on this GPU |
| **A2-a** (original proposal) | fresh repeat within 3 x sqrt(2) x s_cell, s_cell from 5 runs | unchanged | stated 0.3 % / 1.6 %; **actual 4.0 % / 21.7 %** (t with 4 df) | the held-out period; **after** the failure | proposal, not active | about 1 h GPU | flawed calibration (error found 2026-10-22, before any A2 run); its step 2 would have overwritten the FAIL record |
| **A2-b** | 0.0187 fixed, applied to the existing record | unchanged | not meaningful: chosen after seeing the result, from one cell | the held-out period only | would pass 6 of 6 | none | **not recommended**: post hoc |
| **A2-a-r1** (recommended) | mean of 10 runs per cell within 4.094 x s_cell x sqrt(1.1) of the recorded run (t, 9 df) | unchanged at 1e-6, checked in all 60 runs plus a fresh 21-check run | **0.27 % / 1.6 %** | the held-out period; after the failure (disclosed); before any A2 measurement | proposal, not active; outcome unknown | about 1.5-2.5 h GPU, validation only | recommended. Tolerance about 0.019 at s = 0.0044. If it fails, F is not ready. |
| **A2-c** CPU-deterministic (not drafted) | new CPU reference and repeat, every score within 1e-6 | all checks exact | about 0 % | would be drafted now | not written | about 3 h CPU for the gate + about 2 h CPU for the held-out F run | strongest defect detection. The reference run changes device (the validation H4 verdict was on GPU) and it needs its own hashed amendment. |
| **Exclude F** | none | none | n/a | n/a | n/a | none | leaves the H4 future-period reading unmet |

**Withdrawn option:** "authorize F knowing the gate failed as written" (offered in the previous version) conflicts with your rule that a run must not proceed unless its readiness gate "has passed or been validly amended". The runner now enforces that rule.

**Order matters.** The F gate must be decided and measured before any held-out run. Otherwise a later gate decision could be suspected of being shaped by held-out results.

## 4. Exact authorization text (Option B)

**Step 1, in chat** (approves the validation measurement; reads no held-out data):
> I approve Amendment 2 revision 1 (A2-a-r1) as the Module F trained-arm readiness gate.

Its SHA-256 is in `experiments/final_heldout_protocol/AMENDMENT_2_PROPOSAL_F_GATE_REV1.sha256`; the value is repeated in the report you receive with this package. After approval:
* the A2 support (decision script, F authorization check, tests) is implemented and the readiness gates are re-run;
* the 60 runs and the fresh deterministic run are made;
* the pass or fail result is recorded and reported to you before anything else.

**Step 2, the file `experiments/final_heldout_protocol/AUTHORIZATION.txt`.** Protocol section 6 says it is "created by the user", so create it yourself. If you ask me to write it, I will write exactly your text and record that it was written on your instruction. Content:

```
Protocol SHA-256: 59bf1565fb16653a02a2d743cc445e114fb2a6a42f1de7b2fa56261e6f68c806
I authorize one-time evaluation of the frozen models listed in section 2 on the held-out period (days 152-182) under `experiments/final_heldout_protocol/HELDOUT_EVALUATION_PROTOCOL.md`, once the code-readiness gate in section 6 has passed. I understand this irreversibly uses the held-out period and that results will be reported whatever they are.
Amendment 1 SHA-256: 37a1c0577a8ef28530a6120a027890c11ec458ca97329d14a05108db21a27e6e
I authorize the following Amendment 1 modules: F, H, U
I understand this irreversibly uses the held-out period for those modules and that results will be reported whatever they are.
Amendment 2 revision 1 SHA-256: <the value in AMENDMENT_2_PROPOSAL_F_GATE_REV1.sha256>
I approve Amendment 2 revision 1 (A2-a-r1) as the Module F trained-arm readiness gate.
```

For other scopes: Option A uses the first two lines only. Option C uses `modules: F, H` and drops U. B without F uses `modules: H, U` and drops the last two lines.

Commands, in this order (each one-shot, each refuses without its authorization and readiness):
1. `scripts/run_heldout_evaluation.py --split heldout`
2. `scripts/run_amendment_evaluation.py --split heldout --modules H,U` (`--modules` must name exactly the H/U modules in `AUTHORIZATION.txt`: `H` for Option C, `H,U` for B; otherwise refused before any read)
3. `scripts/run_f_walkforward.py --split heldout` (refused unless A2-a-r1 passed; arms and seeds are fixed)

Results are reported whatever they are; no model, threshold or setting is changed afterwards.

## 5. AC12: alternative-evidence consequences

* **Registered method:** a SHA-256 before/after comparison of every pre-existing file. It cannot be done after the fact, because no "before" hashes were taken. **Status stays BLOCKED as registered**; the Phase 12 records are not rewritten.
* **Recovered evidence** (`AC12_GIT_SNAPSHOT_VERIFICATION.json`) for 651 files:
  * 434 are content-identical to the first Git commit `e70238f`, pushed 2026-10-16T13:44:48Z (GitHub activity API), before the build of 2026-10-19 12:59 IST;
  * 65 are covered by the pre-build scope fingerprint, 13 by hashes in pre-build session logs, and 38 by project integrity records;
  * **0 contradictions; 101 files have no pre-build evidence.**
* **The 101 files:**
  * 32 model files and checkpoints;
  * 19 data files: raw ×5 including `test_identity.csv` and `test_transaction.csv`, which were never opened; processed ×10; features ×4 including `test_features.parquet`;
  * 18 Phase 9 benchmark arrays and caches;
  * 3 row-level prediction files;
  * 29 others: 22 hash sidecars, two integrity JSON records, and five LR/RF encoder files.
* **Further sources searched, none usable:**
  * the `data/` records hold no hashes;
  * the Phase 9 manifests hash only the seed-42 risk;
  * Windows shadow copies cannot be queried without administrator rights;
  * there is no OneDrive project copy and File History is not configured;
  * re-downloading from Kaggle would mean hashing TEST-named files, which is not allowed.
* **New this pass: derived evidence, labelled as not provenance and not counted** (`AC12_DERIVED_EVIDENCE.json`, `scripts/check_ac12_derived_evidence.py`):
  * 20 of 22 sidecars equal the hash of a companion that has pre-build evidence;
  * 18 model files (the 17 base models plus the historical XGBoost) reproduce, through the readiness gates, recorded validation predictions that have pre-build evidence (largest difference 7.2e-07).
  * That leaves 63 files with no evidence of any kind.
* **If you accept the alternative:** AC12 is recorded as an "accepted alternative". The basis is 550 of 651 files with pre-build evidence, 0 contradictions, 38 more with derived consistency, and 63 unverifiable (listed). The registered status stays BLOCKED beside it.
* **If you do not:** AC12 remains BLOCKED, and under your completion rule the project cannot be declared complete.

## 6. Phase 13 scope

The original specification was never recovered. `submission/PHASE13_SCOPE_CONFIRMATION.md` separates the 7 recovered requirements (production-style inference; suggested `api/` and `dashboard/` folders; FastAPI and Streamlit as "likely" under a minimum-stack rule; investigation output with candidate wording; explanations; the dashboard and final report in your ordered list; evaluation honesty) from the 7 assumptions made (local-only; standard library; aggregate-only data; a demo mode and the research XGBoost; neural models not served; acceptance criteria P13-1 to P13-10).

Verified: dashboard 31 of 31 checks; research serving 12 of 12; HTTP scores equal the recorded validation predictions. Confirm, or name changes.

## 7. Public-history remediation: choices and risks

* **Facts:**
  * The repository is public with 0 forks and 0 stars. The six files were first published in the push of 2026-10-21T10:05:51Z.
  * Their contents are TransactionIDs of 300 validation transactions with per-transaction contributions, counterfactual effects or feature values.
  * A raw URL at commit `de16aa0` returned HTTP 200 on 2026-10-22 (re-checked with a header-only request); the same path on `main` returns 404. The GitHub API reports `visibility: public`, 0 forks, 0 stars, 0 watchers.
* **Licence:** the IEEE-CIS competition rules (sponsor: IEEE Computational Intelligence Society):
  * 7A allows non-commercial use, including academic research;
  * **7B** says competitors may not "publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition", and must "notify Kaggle immediately" of possible unauthorized access and work with Kaggle to rectify it.

  Whether per-transaction values derived from the data count as Competition Data is a legal reading I cannot settle. Raw feature values of identified transactions most likely do.

| Choice | Effect | Risks | Reversible? |
|---|---|---|---|
| **1. Make the repository private (recommended, now)** | stops public raw and web access immediately; keeps all history | links in the manuscript stop working for others; GitHub caches and anyone who already cloned keep copies | yes (owner setting) |
| **2. Notify Kaggle (rule 7B)** | meets the notification duty | none technical; it is your communication | n/a |
| **3. History rewrite** (`git filter-repo` on the six paths, then a force-push) | removes the files from every commit | Destructive: the hash of `b28a03b` and of every later commit changes (24 of 25 today, plus any new ones). The AC12 snapshot `e70238f` is the only unaffected commit. Commit references in documents and tests (for example `4356385`, `de16aa0`) break and would need updating. GitHub keeps unreachable commits reachable by hash until support purges them. Needs your explicit approval; the verified backup bundle above exists. | only from the backup |
| **4. Leave as is** | nothing | continued public availability; conflicts with 7B | n/a |

I will not change visibility, contact anyone, or rewrite history without your explicit instruction.

## 8. Risks and limits stated in advance (held-out)

* One 31-day period (30 scored days for F) and one dataset; three seeds are not independent replications.
* Concept drift may move every model.
* GPU nondeterminism adds about 0.004 of noise to a trained arm's window mean.
* Zero label delay for F and H; the adaptive arms use validation labels.
* Thresholds are transferred from validation, so alert counts will shift.
* GAT is reported as non-converged.
* Validation results are never substituted for held-out results anywhere in this project.
