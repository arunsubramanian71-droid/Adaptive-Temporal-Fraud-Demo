# Phase 12 - consolidated validation-evidence table: closure report

Dated 2026-10-19. Scope record: `SCOPE_AND_ACCEPTANCE.md` (sidecar `SCOPE_AND_ACCEPTANCE.sha256`), written and hashed before any Phase 12 code or output existed.

**Descriptive table of recorded validation-period (VAL, days 122-151) predictions. Every model was tuned on VAL (early stopping, epoch selection or thresholds) except where stated for F. No held-out (TEST) comparison exists, TEST is sealed, and nothing here supports a claim of generalization, statistical significance, causal attribution, fraud or coordination.**

## 1. Status

The evidence table is built, recomputed and checked. Acceptance criteria AC1-AC11 are recorded as PASS only to the extent supported by the verification evidence named in each row of section 3 (limits in section 3.2). **AC12 is BLOCKED — Partially verified** (section 3.1); AC12 is not passed. **Phase 12 is not formally complete.**

Not done, by instruction: B (kept MISSING), training, scoring, new predictions, ablations, TEST contents, Phase 10 empirical tiers, graph evidence, Git.

## 2. What was produced (`experiments/phase12_validation_evidence_table/`)

| Path | Content |
|---|---|
| `outputs/table_report.md` | rendered table with banners, B row, comparability matrix |
| `outputs/evidence_runs.csv` | 38 rows: 20 non-F runs and 18 F rows (9 arm-seeds x 2 bases), with source paths, hashes, model identity, data window, metric basis, seed |
| `outputs/evidence_cells.csv` | per-cell mean, sample SD (ddof = 1), min, max; the B row reading MISSING |
| `outputs/evidence_windows.csv` | window-level PR-AUC (Phase 8 windows) for every run |
| `outputs/comparability_matrix.csv`, `evidence_table.json` | the 10 pre-registered comparison labels, read from the scope record |
| `outputs/recomputation_check.json`, `crosscheck_recorded.json`, `population_check.json`, `io_audit.json`, `source_manifest.tsv`, `manifest_phase12.json` | AC3, AC8, AC4, AC1 evidence and reproducibility metadata |
| `test_results/test_results.json`, `test_log.txt`, `test_sensitivity.json` | recorded test run and mutation-sensitivity check |
| `frozen_files_after.tsv`, `INTEGRITY_PHASE12.json` (+ sidecars) | hashes of every pre-existing file after the work (a post-build record); integrity record |
| `PRECHANGE_MANIFEST_closure_update.tsv` (+ sidecar) | hashes of 696 files taken before this documentation revision (TEST-named files by size and modification time only) |

Code: `src/evidence/{sources,table,build,outputs}.py`; scripts `scripts/run_phase12_{table,tests}.py`, `scripts/check_phase12_test_sensitivity.py`, `scripts/finalize_phase12_integrity.py`; tests `tests/test_phase12_evidence.py` (16) and `tests/test_phase12_outputs.py` (11).

## 3. Acceptance criteria

The criteria are those registered in section 6 of the scope record and are unchanged. Each PASS below rests only on the evidence named in its row.

| ID | Status | Evidence |
|---|---|---|
| AC1 declared source set only; stable fingerprint | **PASS** | scope sidecar matched; fingerprint recomputed at start and end equals the recorded `30f729b2...3224`; 165 declared files opened, nothing else; no TEST-named, checkpoint, feature or excluded-folder path opened |
| AC2 single metric implementation; no threshold tuning | **PASS** | `src/models/metrics.py` unmodified (hash in the manifest); `select_best_f1_threshold` never imported (static test); every row's threshold equals the recorded one (A-E: `metrics_val.json`; F: `hparams_seed{S}.json`) |
| AC3 recomputation agrees with recorded values within 1e-9 | **PASS** | 65 run/window files, 8,184 recorded values compared, maximum absolute difference 0.0; the only recorded key not recomputed is `threshold_selection` (how a threshold was chosen; recomputing it would need the forbidden threshold search) |
| AC4 population identity | **PASS** | all 20 non-F files hold the same 83,655 ids, identical labels and, where present, identical `day` and `segment`; each of the 9 F arm-seeds partitions the same 83,655 ids across five windows consistent with the registered definition |
| AC5 labels only to compute metrics | **PASS** | stored labels used only inside the metric function; no threshold, row selection or setting derived from them (static and equality tests) |
| AC6 required fields, F banner, B MISSING, no excluded folder | **PASS** | all fields present on every row; both F bases for all 9 arm-seeds; the B row reads MISSING and the historical control appears only as the reference line; the excluded folders appear only in the "never read" disclosure line |
| AC7 comparability labels | **PASS** | the matrix equals section 5 of the scope record (10 rows), read from it; no other comparison reported |
| AC8 cross-check files compared, not inputs | **PASS** | 173 items from three recorded `comparison.json` files compared; 0 discrepancies, maximum difference 0.0 |
| AC9 wording lint | **PASS** | Phase 9 lint clean on all Phase 12 text outputs and sources; VAL-tuned banner on the table |
| AC10 determinism; sidecars; manifests | **PASS** | a rebuild into a temporary folder is byte-identical for every data output (manifests identical apart from the timestamp); every output has a matching sidecar |
| AC11 tests and mutation sensitivity | **PASS** | 27 of 27 tests pass; 17 of 17 injected defects detected, unmutated modules pass. History: the first sensitivity run detected 16 of 17 (the early fingerprint gate was masked by a second, later check); the gate test was strengthened to require the early gate's own message and an early stop, and the run was repeated |
| AC12 before/after SHA-256 of every existing file | **BLOCKED** | BLOCKED — Partially verified. Authentic pre-build evidence covers 300 of 651 pre-existing files, with zero contradictions. The remaining 322 files cannot be retrospectively verified because no trustworthy pre-build hashes survive. AC12 is not passed. |

### 3.1 AC12 detail

The pre-build snapshot (653 files hashed at the start of the work) was lost before the comparison because the harness cleared the scratchpad. A read-only recovery assessment (2026-10-19) harvested earlier per-file hashes from the project's sidecars, JSON integrity records and manifests, text records and the saved session logs (entries dated before the build start) and compared each with the current file:

* **Covered, equal to the current file:** 139 files have an authentic earlier per-file hash (103 from integrity records and manifests, 33 from hash lists printed in the session logs, 3 from sidecars). A further 161 of the 165 declared source files are covered collectively: the fingerprint written in the scope record on 2026-10-18, before the build, equals the fingerprint of the current files (4 of the 165 were already counted individually). Total 300 of 651.
* **Contradictions:** none. An initial pass reported two differences; both were an error in the assessment's own line-pairing pattern, which was corrected and the analysis re-run.
* **Not verifiable:** 351 files have no independent earlier hash. 29 of them are sidecars that match their target files (self-consistent, not independent). **322 have no authentic pre-build hash of any kind:** Phase 9 files 47; Phase 3-6 experiment outputs 62 (GraphSAGE robustness 13, GAT 11, GraphSAGE 8, logistic regression 8, random forest 7, GAT 300-epoch 6, MLP control 6, historical control 2, XGBoost 1); temporal ablation outputs 38; Phase 8 outputs 27; temporal GraphSAGE outputs 26; source files 38, scripts 41 and tests 6 of Phases 1-9; reports 8; data files 28 (including the 3 TEST-named files, which were never opened or hashed); `.gitignore` 1.
* **Not counted as verification:** a modification-time scan (no pre-existing file is newer than the date rollover) and the reconciliation of file counts (651 pre-existing plus the 2 scope-record files that already existed when the snapshot was taken equals the snapshot's reported 653). Both were reported earlier as weaker substitute evidence; neither is part of the 300.
* `frozen_files_after.tsv` records the current hashes of all pre-existing files. It is a post-build record: it supports future comparisons and does not verify the past.

### 3.2 Limits of the PASS statuses (AC1-AC11)

* AC1 and AC5 rest on the build's logged file opens, the end-to-end fingerprint and static and equality tests, not on an operating-system level read trace.
* AC2 rests on a static scan (no threshold-selection function is imported) and equality of every threshold with the recorded one.
* AC3 and AC8 cover the 65 run/window files and 173 cross-check items; the recorded `threshold_selection` key is not recomputed.
* AC4 covers the 20 non-F files and the 9 F arm-seeds; AC6 and AC7 are structure checks on the outputs; AC9 uses the Phase 9 lint patterns only.
* AC10 shows determinism by one rebuild into a temporary folder.
* AC11 rests on 27 tests and 17 injected defects; it shows these tests fail on those defects, not that no other defect exists.

## 4. Descriptive facts (not findings)

* The Phase 8 mean-of-windows PR-AUC is reproduced for all three F arms to the four decimals reported (0.4243, 0.4580, 0.4179), and every per-window value matches Phase 8's `comparison.json` with difference 0.0. The concatenated-window values differ slightly (the adaptive and replay arms are scored by updated weights in later windows).
* Recorded and recomputed values agree exactly, which means the recorded Phases 3-8 metrics are reproducible from the stored predictions on this single implementation.
* A cell comparison is a statement about recorded VAL numbers under the labels of section 5, not about causes.

## 5. Choices, deviations and corrections during the work

* Fixed in advance (scope record): operating-point metrics only at recorded thresholds; ddof = 1 for SD; Phase 8's windows; two bases for F.
* A dry-run build was written by mistake to a Windows path derived from an MSYS-style path string, creating a folder tree under `C:\c\` outside the project. Its 24 files were deleted; removal of `C:\c` itself was blocked as a protected path, so **9 empty folders remain under `C:\c\Users\...`** (no files). They can be deleted by the user.
* Test-side corrections (no acceptance criterion or tolerance changed): a guard test wrongly expected an undeclared-but-unrefused path to be refused; a static scan matched the guard's own refusal list; a CSV parser setting (exact float round trip) and an exclusion-disclosure line needed handling in two output tests; a gate test was strengthened (AC11).
* The scratchpad (outside the project) was cleared by the harness during the work; no project file was affected, but it removed the pre-build snapshot (AC12; see section 3.1 for the recovery assessment).

## 6. Limitations

* A has one run per model, so no seed variance; D never converged (so C versus D is not an architecture comparison); F is on a distinct basis and is not comparable with A, C, D or E; every number is VAL-tuned.
* Row order matters for top-K ties, so each run is scored in its stored order; this is what reproduces the recorded values.
* B is missing; H2 cannot be assessed from this table. Producing B needs new graph-feature engineering and training, which is a separate authorization.
* AC12 is BLOCKED — Partially verified: 322 of the 651 pre-existing files cannot be retrospectively verified (section 3.1).

## 7. Reproduction

```
.venv/Scripts/python.exe -B scripts/run_phase12_table.py
.venv/Scripts/python.exe -B scripts/run_phase12_tests.py
.venv/Scripts/python.exe -B scripts/check_phase12_test_sensitivity.py
.venv/Scripts/python.exe -B scripts/finalize_phase12_integrity.py --since 2026-10-19T00:00:00 --prebuild-count 653 --prebuild-phase12-files 2
```

The build is deterministic; the test scripts overwrite the recorded test results, which must then be re-hashed by the finalization script.

## 8. Revision record

Revised 2026-10-19 in a documentation-only step. **Changed:** this report (AC12 wording; sections 1, 2, 3, 3.1, 3.2, 5, 6, 8) and `INTEGRITY_PHASE12.json` with its sidecar. **Created:** `PRECHANGE_MANIFEST_closure_update.tsv` with its sidecar (taken before the change). **Not changed:** the evidence table and every file under `outputs/` and `test_results/`, `frozen_files_after.tsv`, the scope record and the acceptance criteria, all code and tests, and Phases 1-11. Previous versions: closure report `9616bac0e90a3a67e124ff404474152d91697cbc93eb9c36983840ac65ad0e60`; integrity record `f96e0480eba22b4f86258b87ff97a6b1354eb467034631d32704591ca6003b61`; pre-change manifest `2b952b2fe3d47fcd86f6476f492f576d777ec0d7d220fda5e0f3f73ac7f33275`.
