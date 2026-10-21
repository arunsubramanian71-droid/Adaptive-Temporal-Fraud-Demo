# Phase 11 — Explainable evidence: scope and acceptance record (first pass)

Dated 2026-10-18. Written BEFORE any Phase 11 code or output exists. Its SHA-256 is in `SCOPE_AND_ACCEPTANCE.sha256`.

Standing statement for every Phase 11 output: **candidate coordinated community; not a confirmed fraud ring.** This does not establish that IEEE-CIS contains fraud rings. Model contributions explain a model's output; they are not evidence of fraud, causality or coordination.

## 1. Authority and status

* No project document defines or authorizes Phase 11 (read-only inspection of 2026-10-18). Authority for this pass is the user's explicit instruction of 2026-10-18, summarized in section 3.
* Frozen: Phases 1-10 and every frozen artifact. TEST (days 152-182) is never touched. Phases 12-13 are not started.
* Graph evidence is **deferred**: the Temporal GraphSAGE checkpoint is not run on real data and no risk-scoring stream is created.

## 2. Documented requirements (sources; not invented here)

| # | Requirement | Source |
|---|---|---|
| D1 | Explainability: SHAP "where appropriate" for tabular models; for graph models, interpretable evidence (important neighbouring entities, influential relationships, attention information where valid, neighbourhood perturbation, suspicious connected structure); "Do not claim that attention weights automatically constitute perfect causal explanations." | Original project brief, EXPLAINABILITY section (first message of the earliest saved session, not a repo file; quoted here so it is hashed in the repo) |
| D2 | Minimum technology stack; no unnecessary technologies; no dashboard until requested | Original brief, TECHNOLOGY POLICY and CURRENT TASK |
| D3 | Communities are candidates; no community is described as confirmed fraud or confirmed coordination unless ground truth exists (candidate / suspicious-coordinated-community wording only) | Original brief; Phase 9 design section 17 (lint with two whitelisted disclaimers); `src/coordination/lint.py` |
| D4 | Case-study template and pre-registered, seeded selection: (a) top-10 distinct tracks by peak S+T score, (b) 5 largest tracks, (c) 5 tracks sampled uniformly from the alerted set, (d) 5 hub-dominated high-S tracks; presentation seed 42, other seeds in appendix tables; sampling seed 40000+S; clean real run on days 122-151 | `experiments/phase9_coordinated_fraud/DESIGN_BEFORE_EXECUTION.md` sections 16 and line 481 |
| D5 | Splits: TRAIN 1-121, VAL 122-151, TEST sealed; day = `TransactionDT // 86400` | `src/config.py`, `src/data/splits.py` `compute_day`, `src/coordination/frozen.py` |
| D6 | The XGBoost model used VAL labels for early stopping (best iteration 798) and for threshold selection | `scripts/train_xgboost.py` lines 81-100; `experiments/xgboost/config.json` |
| D7 | Tier 0 community-day CSVs are aggregate-only (day, track, sizes, scores, hub share, density, conductance, member-transaction composition); they contain no member card or entity identifiers | `experiments/phase9_coordinated_fraud/exploratory_realdata/tier0_community_days_W{3,7,14}_seed{42,123,2024}.csv` |
| D8 | E0 is a descriptive reference, not exact and not a significance test; tau locks are TRAIN-calibrated, exploratory | `E0_ADDENDUM_post_Tier0_exploratory.md` sections 3-4; `e0_tau_seed{S}.json` |
| D9 | E1 (label-based supporting evidence) and any use of labels, risk calibration, R_p, R_h are not authorized; Phase 9 case studies were not completed and each future-work item needs its own decision | `reports/phase9_closure_report.md` (not-completed and future-work lists); `PRE_RUN_RECORD.md` section 1 |
| D10 | Standing rules: no labels to generate, select, rank or tune; strict temporal causality; never invent results; no new dependencies without reporting the need | project rules |

## 3. Decisions given by the user for this pass (2026-10-18)

1. Defer graph evidence (no checkpoint run on real data, no new risk stream).
2. First-pass deliverables: (i) XGBoost per-transaction explanations, (ii) aggregate-only Phase 9 candidate-community investigation cards.
3. VAL days 122-151 are used descriptively, acknowledging D6.
4. Label-free transaction selection: top 5 per day by model score plus 5 seeded uniform draws, seed 40042; overlap resolved deterministically; final count documented.
5. Isolated `src/explain/` module with native XGBoost contributions; toy-model tests before real VAL features.
6. Cards from existing aggregate-only CSVs; do not re-run discovery and do not fabricate missing fields.
7. No dashboard or API, no dependency installation, no TEST, no Phase 10 empirical tier, no labels to select, rank or tune anything.

## 4. Proposed implementation choices (mine; each is recorded so it is fixed before any output exists)

* **P1 Model and feature path.** `experiments/xgboost/model.joblib`; features rebuilt exactly as in training with `TreeFeatureBuilder.load(cfg, experiments/xgboost/feature_builder).to_matrix(val_df, for_model="xgb")` (470 features). Contributions: the model's booster `predict(DMatrix, pred_contribs=True)` (native Tree SHAP in log-odds margin space); the last column is the bias. `shap` is not used and nothing is installed.
* **P2 Label-free data access.** `data/features/val_features.parquet` is read with an explicit column projection that excludes `isFraud`; `predictions_val.parquet` is read for `TransactionID` and `y_score` only. The label column is never loaded by the selection or explanation code. The file `test_features.parquet` is never opened; any day outside 122-151 is rejected.
* **P3 Selection (interpretation recorded).** "5 seeded uniform draws" is read as **5 per day**. For each VAL day d in 122..151: (i) the 5 transactions with the highest `y_score` (descending; ties by `TransactionID` ascending); (ii) 5 draws without replacement, uniformly, from the day's remaining transactions (sorted by `TransactionID`, the top 5 excluded) with `numpy.random.default_rng(numpy.random.SeedSequence([40042, d]))`. Overlap is resolved by this exclusion, so the two groups are disjoint by construction. **Final count: 30 days x 10 = 300 transactions** (150 `top_score`, 150 `uniform_draw`). The alternative reading (5 draws in total, 155 transactions) is not used.
* **P4 Explanation output per selected transaction.** `TransactionID`, `day`, `group`, `y_score`, margin, bias, all 470 feature contributions, the top 10 features by absolute contribution with the feature value, and the contributions' sum. Features are anonymised (V*, C*, id_*): explanations name columns, never mechanisms.
* **P5 Cards.** Window W = 7 (the Phase 9 primary window) only; W = 3 and 14 are sensitivity and are not carded. For each seed S in {42, 123, 2024}: groups (a) top-10 distinct tracks by peak `score_st` (the A2 score; verified equal to the mean of `u_s` and `u_t`), (b) 5 largest tracks by peak `n_cards`, (c) 5 tracks sampled uniformly without replacement from the alerted tracks (a track is alerted if at least one of its community-days has `score_st >= tau_S`, tau from `e0_tau_seed{S}.json`, W = 7) with `numpy.random.default_rng(40000 + S)` over track ids sorted ascending, (d) 5 hub-dominated high-S tracks. Ties in (a)(b): lower track id. A track may belong to several groups; memberships are listed. Full cards for seed 42; selection tables for seeds 123 and 2024 as an appendix.
* **P5d Operationalization of group (d)** (Phase 9 section 16 does not define it): rank all tracks by peak `u_s` (descending) and by peak `hub_share` (descending); the key is the sum of the two ranks, ascending, ties by track id; the first 5 are taken. No threshold is introduced.
* **P6 Card content (aggregate-only).** Per card: size (cards, infrastructure entities), lifecycle timeline from the daily rows (n_cards, n_infra, density, hub share, conductance, `u_s`, `u_t`, `score_st`, member transactions, alert flag), first and last observed day with left-censoring (first day 122) and right-censoring (last day 151) flags, segment x infra_observed composition summed over days, group memberships. **Not available and not fabricated**, stated on every card: member identities, new-member counts, pseudo-account counts, descriptive attributes (ProductCD, card4, card6, email domain), risk percentile or learned-state signature, any label-based column, and a per-community null-matched comparison (E0 provides day-level reference counts only; they are described as a descriptive reference).
* **P7 Outputs.** Under `experiments/phase11_explainable_evidence/`: `xgb_explanations_val/` (selection CSV, contributions CSV, summary JSON, manifest) and `community_cards/` (cards JSON, Markdown rendering, appendix tables). Plain UTF-8/LF files with a `.sha256` sidecar each.
* **P8 Wording.** Phase 9 lint (`src/coordination/lint.py`) is applied to every Phase 11 text output and source file; only the two whitelisted disclaimers may mention the prohibited phrases.

## 5. Acceptance tests (fixed before implementation)

| ID | Test | Pass criterion |
|---|---|---|
| A1 | Contribution additivity | for every explained row, `abs(sum(contributions) + bias - margin) <= 1e-4` (float32 arithmetic); exceeding it is reported as a failure, never relaxed |
| A2 | Probability consistency | the sigmoid of the recomputed margin equals the stored `y_score` within `1e-6` for all selected rows (max difference reported) |
| A3 | Batch invariance | a row's contributions computed alone equal those computed in a batch (tolerance 1e-5) |
| A4 | Reproducible selection | two runs give identical ids; the selection has exactly 300 rows, 10 per day, no duplicates, top group disjoint from the uniform group |
| A5 | Label-free selection | selection is identical when the label column is absent or replaced by a poisoned object; static check that selection and explanation code never reference the label column |
| A6 | Temporal data window | every selected row has day in 122..151; no TEST file is opened; only the row's own features are used (no future rows) |
| A7 | Artifact hash linkage | the manifest records SHA-256 of the model, feature builder files, `val_features.parquet`, `predictions_val.parquet`, Tier 0 CSVs, tau locks and all outputs; all match files on disk |
| A8 | Wording lint | Phase 9 `lint_files` reports no violation on all Phase 11 text outputs and source |
| A9 | Cards use only existing fields | every card field comes from the Tier 0 CSV columns or the tau locks; the unavailable-field list is present on every card; discovery is not re-run |
| A10 | Frozen artifacts | a before/after SHA-256 comparison of every existing project file shows no change |

## 6. Deferred, excluded, not started

* **Deferred:** graph evidence (Temporal GraphSAGE checkpoint, perturbation or attention-based evidence, any real-data model scoring stream).
* **Excluded in this pass:** random-forest or library-SHAP explanations (no installation), any label-based column (E1), per-community null match, member-level community data, TEST, Phase 10 empirical tiers and Stage 0, dashboard and API.
* **Not started:** Phases 12 and 13.

## 7. Limitations recorded in advance

* VAL labels shaped the XGBoost iteration count and threshold (D6), so VAL explanations describe a model that has seen VAL for tuning; they are descriptive and not a held-out result.
* Tree SHAP contributions are additive in log-odds and are shared across correlated features; they do not identify causes.
* Community cards describe Louvain-based candidate communities on aggregate fields; Tier 0 identity is unstable for most tracks and censored at both ends of a 30-day window.
* Transaction-level outputs (ids, scores, contributions) are derived data; this pass does not modify `.gitignore`.

## 8. Stop conditions

Stop the affected subtask and report if: any acceptance test fails or needs a relaxed tolerance; a required field is absent from the existing aggregates; the recomputed scores do not match `predictions_val.parquet`; a TEST path or a day outside 122-151 would be read; a label would be needed to select, rank or tune; or a new dependency would be required.
