# Phase 11 — Explainable evidence: first-pass closure report

Dated 2026-10-18. Scope record: `SCOPE_AND_ACCEPTANCE.md` (sidecar `SCOPE_AND_ACCEPTANCE.sha256`), written and hashed before any Phase 11 code or output existed.

**candidate coordinated community; not a confirmed fraud ring.** This does not establish that IEEE-CIS contains fraud rings. Nothing in this phase is evidence of fraud, causality or coordination. Model contributions describe a model's output; community cards describe aggregate fields of candidate communities.

## 1. Status

The two approved first-pass deliverables are complete and verified:
1. XGBoost per-transaction explanations for the pre-registered, label-free VAL selection (300 transactions).
2. Aggregate-only investigation cards for Phase 9 candidate communities (24 cards for seed 42; selection tables for seeds 123 and 2024).

**Deferred (user decision):** graph evidence. The Temporal GraphSAGE checkpoint was not run on real data and no risk-scoring stream was created. **Not started:** Phases 12 and 13. Phase 10 empirical tiers, Stage 0 and TEST were not touched. No dependency was installed; no dashboard or API exists.

## 2. What was produced (`experiments/phase11_explainable_evidence/`)

| Path | Content |
|---|---|
| `xgb_explanations_val/selection_val.csv` | 300 selected VAL transactions: 30 days x (5 top by model score + 5 seeded uniform draws, seed 40042) |
| `xgb_explanations_val/contributions_val.csv` | all 470 native XGBoost contributions, the bias and the margin for each selected transaction |
| `xgb_explanations_val/top_features_val.json` | the 10 largest-magnitude features per transaction with feature values |
| `xgb_explanations_val/summary_val.json`, `manifest_val.json` | acceptance-check values, model and selection parameters, input and output hashes |
| `community_cards/cards_W7_seed42.json`, `cards_W7_seed42_report.md` | 24 cards (W = 7, seed 42), with the E0 day-level reference section |
| `community_cards/appendix_selection_all_seeds_W7.csv` | selected tracks and statistics for seeds 42, 123, 2024 |
| `community_cards/cards_summary.json`, `manifest_cards.json` | per-seed tau, group lists, hash links |
| `test_results/test_results.json`, `test_log.txt`, `test_sensitivity.json` | recorded test run and mutation-sensitivity check |
| `INTEGRITY_PHASE11.json` (+ `.sha256`) | hashes of all code, tests, outputs and the frozen-artifact comparison |

Code: `src/explain/{xgb_contrib,cards,outputs}.py`; scripts `scripts/run_phase11_{explanations,cards,tests}.py` and `scripts/check_phase11_test_sensitivity.py`; tests `tests/test_phase11_explain.py` (22) and `tests/test_phase11_outputs.py` (8).

## 3. Acceptance results (criteria fixed in the scope record)

| ID | Result |
|---|---|
| A1 additivity to the margin | **Pass.** Real rows: max absolute error 8.1e-06 (bound 1e-4). Toy models also pass. |
| A2 probability consistency | **Pass.** `sigmoid(margin)` versus the stored `y_score`: max difference 8.5e-08 (bound 1e-6). The model's own `predict_proba` on the rebuilt features equals the stored `y_score` exactly (max difference 0.0). |
| A3 batch invariance | **Pass.** Max difference 0.0 (bound 1e-5) on 20 rows evaluated singly. |
| A4 reproducible selection | **Pass.** Recomputed from the label-free id, day and score columns it reproduces the stored selection exactly: 300 rows, 10 per day, no duplicates, groups disjoint. |
| A5 label-free | **Pass.** The selection and explanation functions have no label argument; the loaders project the label column away (tested on files that contain a label column); a static scan finds the label name only as the exclusion constant. |
| A6 temporal window | **Pass.** All selected days are 122-151; the TEST-named file guard and the day guard are tested (including TEST-period rows in an otherwise full VAL table). |
| A7 hash linkage | **Pass.** Every output has a matching sidecar; the manifests' input and output hashes (model, feature-builder files, VAL features, stored predictions, Tier 0 CSVs, tau locks, E0 results, scope record) match the files on disk. |
| A8 wording lint | **Pass.** Phase 9 lint reports no violation on all Phase 11 text outputs and sources. |
| A9 cards use only existing fields | **Pass.** Cards are rebuilt from the Tier 0 CSV columns and tau locks and equal the stored cards; the unavailable-field list is on every card; discovery was not re-run. |
| A10 frozen artifacts | **Pass.** 612 existing project files hashed before the work: 0 changed, 0 missing. At the time of the comparison there were 35 new files, all Phase 11 (this report and the integrity record were added afterwards). The 34 Phase 9 closure artifacts and the Phase 10 records are unchanged. |

Test run: 30 of 30 pass. Sensitivity check: 14 of 14 injected defects detected, unmutated modules pass everything.

Descriptive facts (not findings): for seed 42 the Tier 0 file has 237 tracks, 45 of them alerted at tau = 0.8581; this equals the independent E0 count (`obs_alerted_tracks` = 45). The 24 seed-42 cards contain one track in two groups (track 92: top peak score and hub-dominated high-S), 4 left-censored tracks, no right-censored track, and 16 tracks with at least one alerted day. Seeds 123 and 2024 have 240 and 252 tracks, with 34 and 47 alerted.

## 4. Choices recorded in advance and deviations during the work

* **Interpretations fixed in the scope record:** "5 seeded uniform draws" was read as 5 per day, which gives 300 transactions; the alternative reading (5 in total, 155 transactions) was not used. Phase 9 section 16(d) ("hub-dominated high-S") is not defined in the design, so a parameter-free rank-sum rule was fixed (peak `u_s` rank plus peak hub-share rank).
* **Scope record corrected before any output existed:** the wording lint flagged a phrase in row D3; the text was corrected and the sidecar rewritten (hash `33156de5...` replaced by the final `271901a9...`). No output existed at that time.
* **Output naming:** two card outputs differed only by letter case, so on Windows their sidecars collided; the outputs were renamed and regenerated (own files only, before verification).
* **Test-side corrections (no expected value in an acceptance criterion was changed):** a CSV float-parsing setting in one output test; an arbitrary file-count guard replaced by a named-file check; two tests added after the sensitivity check showed that score ties at the top boundary and TEST-period rows with full VAL days were not exercised.
* Score ties: checked on the label-free score columns; no day has a tie inside the top 5 or between the 5th and 6th score, so the tie rule did not affect the real selection.

## 5. Limitations

* VAL labels shaped XGBoost's iteration count (798) and threshold, so VAL explanations are descriptive, not held-out. The selection is a small, pre-registered sample (300 transactions); nothing here generalizes to the VAL population or to TEST.
* Tree SHAP contributions are additive in log-odds and shared across correlated features; features are anonymised, so columns are named and mechanisms are not. The contributions do not identify causes.
* Cards are aggregate-only. Not available and not fabricated: member identities, new-member counts, pseudo-account counts, descriptive attributes, risk percentile or learned-state signature, any label-based column, and a per-community null match. E0 appears only as a day-level descriptive reference (not exact, not a significance test). Tier 0 identity is unstable for most tracks and censored at both ends of a 30-day window. Only W = 7 was carded.
* Transaction-level outputs (`selection_val.csv`, `contributions_val.csv`, `top_features_val.json`) are derived data; `.gitignore` was not modified, and its existing rules do not cover this folder. Whether to publish them is a decision for the user.
* No empirical claim, threshold or parameter selection is made in this phase.

## 6. Reproduction

```
.venv/Scripts/python.exe -B scripts/run_phase11_explanations.py
.venv/Scripts/python.exe -B scripts/run_phase11_cards.py
.venv/Scripts/python.exe -B scripts/run_phase11_tests.py
.venv/Scripts/python.exe -B scripts/check_phase11_test_sensitivity.py
```

Verified by regenerating into scratch space: every data output (selection, contributions, top features, summaries, cards, appendix) is byte-identical on re-run; the manifests differ only in their `created_local` timestamp. Re-running the test scripts overwrites the recorded test results, which must then be re-hashed in `INTEGRITY_PHASE11.json`.
