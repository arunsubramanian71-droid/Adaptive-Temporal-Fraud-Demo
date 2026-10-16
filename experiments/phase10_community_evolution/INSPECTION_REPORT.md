# Phase 10 — Read-only inspection report (temporal community evolution)

Dated 2026-10-14. Companion to `DESIGN_BEFORE_EXECUTION.md` in this folder. **Nothing was implemented or executed.** Inspection consisted only of reading files, reading existing JSON/CSV outputs, and introspecting the installed NetworkX function signatures. No experiment, benchmark, model run, discovery run or threshold computation was performed, and no existing file was modified.

Terminology: the objects studied are **candidate communities** (eligible Louvain components of the windowed card-infrastructure graph). Nothing here is, or is claimed to be, confirmed fraud or proof of coordination.

## 1. What was inspected
* **Phase 9 design and records:** `DESIGN_BEFORE_EXECUTION.md` (sections 1-12, 15, 18, 20-24), `STOP_REPORT_G4.md`, `eval_benchmark_G1_G2_G4.json`, `diagnostics/g4_diagnosis.json`, `timing_pilot.json`, `locked_config.json`, `null_calibration_summary.json`, `dev_selection.json`, `reports/phase9_closure_report.md`.
* **Phase 9 exploratory records:** `TIER0_REPORT.md`, `tier0_summary.json`, all nine `tier0_community_days_*.csv`, `E0_ADDENDUM_post_Tier0_exploratory.md`, `E0_REPORT.md`, `e0_results.json`, `e0_tau_seed*.json`, `validation_S2_S3_S4.json`, `validation_E0.json`, `MANIFEST_E0_execution.json`, `log_tier0.txt`.
* **Code:** `src/coordination/{tracking,pipeline,graphs,scoring,nullmodel,data,frozen,realrun,realdesc,e0,segments_array,lint,runner}.py`, `scripts/{validate_phase9_leakage,validate_phase9_e0,run_phase9_tier0,run_phase9_e0,run_phase9_dev_selection,run_phase9_timing_pilot,run_tests}.py`, `tests/test_phase9_core.py`, `src/graph/temporal.py` (symbol list), `configs/{graph,data}.yaml`.
* **Phases 1-8 reports (read for context only):** `phase1_reconnaissance.md` (turnover and summary findings), `phase4_graph_report.md` (entity types, hubs), `phase7_temporal_gnn_report.md`, `phase7a_temporal_ablation_report.md`, `phase8_concept_drift_report.md`.
* **Environment introspection:** `inspect.signature` of `networkx.community.louvain_communities` and the source head of `leiden_communities` (NetworkX 3.6.1).

## 2. Reusable components
| Component (location) | Verdict | Conditions and limits |
|---|---|---|
| Real arrays and label-free view (`data.get_real`, `GraphView`) | **Reuse** | `Arrays` also carries labels; Phase 10 code must receive only `.view()` (poison-label test). Rows are capped at day 151. |
| Window graph and Louvain candidates (`graphs.window_rows/aggregate/louvain_candidates`) | **Reuse unchanged** | NetworkX Louvain has no warm-start/initial-partition argument (signature: `G, weight, resolution, threshold, max_level, seed`), so day-to-day partitions are computed independently. |
| Candidate scoring (`pipeline.discover`, `scoring.ArrivalTable/t_raw_for`, locked `NullTables`, `record_score`) | **Reuse, descriptive only** | The A2 score is a ranking heuristic. H5a is not established, so it has no validated meaning. |
| Thresholds `e0_tau_seed{42,123,2024}.json` (hash-locked) | **Reuse for the "alerted" subset only** | Valid for W = 7, gamma = 0.5 only. TRAIN-only calibration, 30 serially dependent windows. |
| Tracker (`tracking.Tracker`, `lifecycle_table`) | **Baseline only, not the measuring instrument** | Structural limits in section 3. A new module is required; the Phase 9 module must not be edited. |
| Validated run wrapper (`realrun.run_discovery`, `hash_output`) | **Reuse** | Passed S2/S3/S4 on the real-data path. Returns `out["comm"]` records with card and infrastructure sets, so an external lineage module can consume them without touching the pipeline. |
| Descriptors (`realdesc.describe_day`, `segment_codes`, `infra_observed`) | **Reuse** | Hub share, density, conductance, segment x infra_observed composition. Segmentation is G0-validated. |
| Null generator (`e0.rewired_arrays`) | **Reuse with caveat** | Property-tested (P0-P6) for days 116-151, W = 7. Other day ranges and W need re-validation. The key scheme is fixed (50000 + S, null id 1), so Phase 10 would regenerate the same histories as E0 for identical (S, k, day). A fresh key space needs a new wrapper (unresolved D5). |
| Leakage-validation helpers (`Poison`, `corrupt_after`, `truncate_after`, `corrupt_inside` in `validate_phase9_leakage.py`) | **Reuse the pattern** | They live in a script; importable via `sys.path`, as `validate_phase9_e0.py` does. |
| `lint.py`, `runner.require_gates/load_locked` | **Reuse** | Terminology lint applies to Phase 10 text and identifiers. |
| Phase 9 planted benchmark (`benchmark.py`), risk stream (`risk.py`), metrics (`metrics.py`), model checkpoints | **Do not reuse** | The benchmark failed a frozen validity gate and is tied to risk and fraud-style features. Phase 10 needs no model, no risk, no labels. |
| Leiden | **Not available** | `leiden_communities` is stubbed to require a backend (`implemented_by_nx=False`, "backend required"); none is installed (Phase 9 fact F13). No installation is proposed. |

## 3. Does the Phase 9 tracker measure splits, merges, persistence and membership evolution? No.
**Implementation facts** (`tracking.py`): matching uses only day t-1 -> t and only pairs with Jaccard >= 0.5; assignment is greedy one-to-one; a "split" flag is written only when a new community has a candidate pair >= 0.5 with a parent already taken; a "merged" flag only for such pairs on terminated parents; there is no dormancy or re-linking; unmatched communities start new tracks with no parent unless such a pair exists.

**Why zero splits/merges is structural, not a finding.** Same-day communities are disjoint card sets. If two disjoint children A, B of a parent P both have Jaccard >= 0.5 with P, then |A and P| >= |P|/2 and |B and P| >= |P|/2, and these intersections are disjoint subsets of P, so both equal |P|/2 exactly and each child is a subset of P. Only an exact bisection can raise a split flag (symmetrically for merges). A 70/30 split continues the larger child and starts the smaller one as an unflagged new track with no parent pointer; a 70/30 merge ends the smaller parent unflagged.

**Other limits that follow from Jaccard >= 0.5:**
* J <= min/max of the two sizes, so a continuing community can change size by at most a factor of 2 per day; faster growth or contraction breaks identity. Cross-check on the existing CSVs: the minimum size ratio over all continuing steps is exactly 0.500 at W = 7 (3 seeds) and W = 14, and 0.579-0.600 at W = 3; all continuing steps are on consecutive days.
* No gap tolerance. The design's disappearance convention (declared g = 3 days later, section 8) has no matching counterpart, so a one-day absence ends a track. The Tier 0 report counts 204 of 237 W = 7 tracks (seed 42) as "observable disappearances" (86%); this cannot distinguish genuine disappearance from identity breaks.
* The test `test_tracker_continue_split_merge` passes a 5-card parent splitting into [1,2,3] and [4,5,90] (second child Jaccard 0.33) and asserts only that the second child has a different id. It never asserts a split flag or parent link, so it does not detect the limitation.

**Empirical context from existing outputs (descriptive; no new computation):** at W = 7, 30-32% of records lie on tracks of age >= 2 days and median track age is 1; of each day's candidates, about 3-14 of 6-19 (seed 42) are tracks not present the previous day. Communities whose first recorded size is >= 17 cards continue far less often (5-6% of such tracks age >= 2) than smaller ones (24-28%), although the two longest-lived W = 7 tracks per seed (age >= 10) have median first size 378-398 cards. Across Louvain seeds on identical data, record counts differ (339/354/366) and track counts differ (237/240/252). The mean consecutive-day Jaccard of continuing tracks is 0.82, but window overlap makes consecutive windows share 6/7 of their rows at W = 7, so high overlap is partly mechanical and cannot be interpreted without a null.

**Not determinable read-only.** The Tier 0 CSVs hold community metrics, not card sets, so how many true splits, merges, gap-relinks and fast-growth cases the tracker missed cannot be quantified without re-running candidate generation. That diagnostic is proposed as Stage 0 and needs authorization.

## 4. Discrepancies and cautions found (reported, not repaired)
1. Phase 9 design section 6 step 6 describes "others start new tracks flagged split" and section 8 describes g = 3 disappearance declaration; the implementation flags only pairs with Jaccard >= 0.5 and has no dormancy. This is a design-text versus implementation-effect gap, not an error in the recorded numbers.
2. Phase 9 design section 3 defines a candidate coordinated community by structural excess and synchrony; the implemented "candidate" is an eligible Louvain component (>= 3 cards, >= 1 infrastructure entity). A "candidate suspicious coordinated community" additionally requires the risk component, which Tier 0/E0 did not use. Phase 10 must therefore use its own terms (candidate community; alerted candidate community = A2 score at or above a locked threshold) and must not equate them with the Phase 9 defined terms.
3. The Phase 9 timing pilot planned E0 as "4 x K x 3" runs; the executed E0 used 30 days (already recorded in the Phase 9 addendum and closure report). No contradiction in recorded results was found.
4. The Phase 9 closure report's statement that zero split/merge flags are structural is consistent with this inspection.
5. The Phase 9 design section 23 explicitly designs nothing for Phase 10; all Phase 10 content is new.
6. No inconsistency was found between the Phase 9 closure report numbers and the artifacts cited here (the closure report's own integrity record verified 49 values).

## 5. Entity types, relationships and time boundaries available (from `configs/*.yaml`, `src/graph/temporal.py`, Phase 4)
* Node types: transaction, **card** (`card1` token, 100% coverage, not a card number), **pseudo_account** (card1 + addr1 + D1 anchor, 88.7%), **netproxy** (id_19, id_20; 23.6%), **device_fingerprint** (DeviceInfo, DeviceType, id_30-33; 12.0%). Relations (transaction -> entity, reversed for message passing): `charged_to`, `used_card`, `via_network`, `from_device`.
* Excluded entity types (hubs, no signal): billing address, email domain, product, DeviceInfo alone.
* Time: day granularity (`TransactionDT // 86400`, undisclosed reference instant, no calendar dates); TRAIN 1-121, VAL 122-151, TEST 152-182 (sealed); same-day mutual visibility; zero label delay (an assumption, irrelevant when no labels are used).
* Only 17.1% of VAL transactions carry a netproxy or device edge, so communities describe only the infrastructure-observed minority; 82.9% are unobservable to this graph.

## 6. Runtime and memory, from recorded timings only (no new profiling)
* Candidate generation plus S/T scoring per day: 0.053 s (W = 3), 0.093 s (W = 7), 0.157 s (W = 14) (`dev_selection.json`: 3.1/5.6/9.3 s for 60 days). E0 executed about 9,180 day-runs in 910 s, about 0.099 s per day-run including rewiring and counting. Median community-detection time per window 0.0717 s (`timing_pilot.json`).
* Estimates by arithmetic only: 112 days x 9 (W, gamma) x 3 seeds is about 3,000 day-runs, a few minutes; the coherent null with K = 100, 3 seeds and 112 days at W = 7 is about 33,600 day-runs, about an hour; lineage and overlap computations are negligible (about 10-25 communities per day). The cost of generating a new synthetic evolution benchmark was not recorded in Phase 9 and cannot be estimated from existing logs.
* Memory: the cached real arrays file is 26 MB; the 942 MB feature matrix is not needed (label-free, model-free). Several array copies fit comfortably.

## 7. Proposed validation tests (detail in the design)
Toy-event suite with exact assertions (including cases the Phase 9 tracker is expected to miss); lineage invariants (at most one primary predecessor, conservation of members, ordering invariance, determinism); online/retrospective separation; future-invariance, label-invariance (poison) and determinism on the real-data path; Phase 9 compatibility check (candidate generation reproduces recorded Tier 0 hashes); null-generator property tests for any new day range; re-typing from the stored overlap graph equals online typing; censoring-aware survival estimator on synthetic data of known hazard; static checks (no label/risk/benchmark imports); terminology lint.

## 8. Unresolved decisions identified (grounded reasons in the design)
Evaluation period and reuse of VAL; event-threshold constants (identity, relation, coverage, growth, dormancy); numeric success floors and reproducibility bands; use of the alerted subset; hub-cap sensitivity (a Phase 9 rescoping lever); scope and gates of a synthetic evolution benchmark; Phase 10 null key space; whether any label-based retrospective stage is ever wanted; usefulness to investigators (not testable without labels or human review); the Stage 0 diagnostic re-run.
