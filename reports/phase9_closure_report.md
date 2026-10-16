# Phase 9 — Closure report (coordinated-group discovery, hypothesis H5)

Project: *Adaptive Temporal Graph Intelligence for Fraud Detection & Coordinated Fraud Discovery*. (The project title's final words are abbreviated here on purpose: the Phase 9 terminology rule forbids ring wording for real-data outputs.)
Dated 2026-10-14. Sources: only existing Phase 9 artifacts (design, G4 stop report, diagnostics, Tier 0 report, E0 addendum, validation records, threshold-calibration records, E0 report, execution manifest) and the existing Phase 1-8 reports. No experiment was run and no existing file was modified to produce this report.

> Exploratory real-data analysis conducted under an explicitly authorized scope deviation after the original Phase 9 protocol halted at S6. Original G4 failed on seeds 42 and 2024 and passed on seed 123. H5a is NOT ESTABLISHED under the original protocol.

## 1. Bottom line
| Item | Status |
|---|---|
| **H5a** (primary: structure + time beats transaction-level methods on camouflaged planted groups) | **NOT ESTABLISHED** under the original protocol. The synthetic benchmark failed a frozen validity gate (G4), so the quantitative test was never run. |
| **H5b** (secondary: learned risk signal adds to structure + time) | **NOT ESTABLISHED** (never evaluated; cannot substitute for H5a). |
| **H5c** (secondary: learned temporal entity state adds beyond risk ranks) | **NOT ESTABLISHED** (never evaluated; cannot substitute for H5a). |
| **H5-real** (structure beyond nulls on real data; later fraud beyond calibrated risk) | Part (i): **exploratory, descriptive observations only** (Tier 0, E0), no verdict. Part (ii) (E1): **not run**. |
| Phase 9 | **Closed** at this report. E1 and Phase 10 were not started. |

"Not established" means the evidence required by the pre-registered rule was not produced. It is not evidence that the hypotheses are false: H5a, H5b and H5c were never tested on a valid benchmark.

## 2. What was completed and what was not
**Completed (all recorded in existing artifacts):**
* Design finalised before execution (R-A..R-F, U1..U4 approved) and frozen: `DESIGN_BEFORE_EXECUTION.md`.
* **G0** (array-level segmentation equivalence, TEST never opened): passed on every recorded criterion (1-7), exact match on all 501,214 transactions (`segmentation_equivalence_gate.json`).
* **Timing pilot**: passed (median 0.0717 s per window; projected 0.188 h CPU) (`timing_pilot.json`).
* **Null calibration** (10 rewirings x 8 TRAIN windows per configuration) and **one-shot dev selection**: locked gamma = 0.5 for W = 3, 7 and 14 (primary W = 7). gamma = 0.5 is the edge of the frozen grid, so the optimum may lie outside it (`locked_config.json`).
* **Evaluation benchmark** generated once per seed (42, 123, 2024) and **causal risk streams** computed.
* Benchmark validity gates on the evaluation benchmark: G3 and G5 passed, G1 and G2 passed, **G4 failed** (section 3).
* Read-only G4 forensics and two amendment proposals (a "G4s" gate), **neither approved**.
* **Exploratory real-data continuation** under an explicit scope deviation: S2/S3/S4 validations, **Tier 0**, E0 addendum (dated, hashed), E0 validation (42 checks), TRAIN-only threshold calibration, **E0** execution (degree-preserving null only).

**Not completed:**
* Any discovery or scoring on the evaluation benchmark; the H5a, H5b and H5c decision rules; baselines B0-B3b and ablations A1-A10; the G-cal guard; the dose-response and stratified synthetic analyses.
* Synthetic-benchmark leakage suite on discovery outputs and the full reproducibility run (the S2/S3/S4 validations cover only the exploratory real-data path).
* Case studies (design section 16); **E1** (not authorised); the time-shuffled null of E0 (omitted by decision); the fraud base rate of the structurally unobservable population (requires labels).
* The final report originally planned for Phase 9 (items A-S of the execution authorisation) in its original form. This document is the closure record instead.

## 3. The original G4 failure and why it stands
**Frozen definition** (design 11.7 and 15.1): mean absolute change in within-day risk percentile of real transactions between the clean stream and the augmented stream (planted rows included in the day's ranking population) must be at most 0.5 percentile points. The stop was hard stop S6.

**Measured on days 122-151 (83,655 real rows per seed):**
| seed | mean abs change (points) | G4 |
|---|---|---|
| 42 | 0.7954 | **FAIL** |
| 123 | 0.4622 | pass |
| 2024 | 0.7994 | **FAIL** |

Other gates on the same benchmark: G1 passed (alpha = 0 top-1% fraction 0.0007-0.0014 against a limit of 0.03); G2 passed (alpha = 1 versus alpha = 0 ratio 153.5-324.0 against a required 5); G3 passed (planted fraction 4.999%, limit 5%); G5 passed.

**Forensics (read-only, `g4_diagnosis.json`; nothing regenerated).** Ranking the real rows among themselves (planted rows removed from the denominator) gives mean changes of 0.0345, 0.0646 and 0.0671 points (seeds 42, 123, 2024), so the as-defined number is dominated by planted rows entering each day's ranking population (planted share 4.76% of day rows; planted rows' mean within-day percentile 0.648, 0.564, 0.647). The model itself changed for few real rows: 98.0%, 97.3% and 97.7% had bit-identical logits; mean absolute logit change 0.0014, 0.0026, 0.0031. A small number of real rows moved a lot (largest single re-ranked change 74.0, 63.0, 90.2 points). The as-defined maximum per seed is 73.4, 62.3, 88.1 points (a different quantity).
**No justification for the 0.5 threshold exists in the design, and none is invented here.** Whether the frozen wording intended denominator effects or model interference is not stated anywhere in the design.

**Why it remains unchanged.** The formula is explicit and was applied literally. Changing it after seeing the failure would be a post-hoc change. A separate gate "G4s" was proposed twice (initial and revised proposals) and the decision taken was to **retain the original G4 failure and not approve G4s or any amendment**. No threshold, gate or design decision was changed, and the benchmark was not regenerated. All G4 artifacts (`eval_benchmark_G1_G2_G4.json`, `STOP_REPORT_G4.md`, `diagnostics/`) are preserved unchanged.

**Unresolved ambiguity (recorded, not resolved, no further dev gates computed).** Design 11.7 says "G1-G5 before any detector output is examined" without naming a benchmark. G1, G2 and G4 were computed only on the evaluation benchmark; for the development benchmark only G3 and G5 were computed and no development risk stream exists. The G4 stop report listed the development selection as "completed and valid" without stating this. The design does not say whether the gates apply to the development benchmark.

## 4. Tier 0 (exploratory, descriptive, no statistical test)
Clean real data, days 122-151, frozen locked configuration, label-free and risk-free, three Louvain seeds (`TIER0_REPORT.md`, hash-verified inputs). Selected facts:
* 82.9% of the 83,655 validation transactions carry no netproxy or device edge, so infrastructure-based discovery cannot see them.
* About 11 eligible candidate communities per day at W = 7 (about 11-12 at W = 3, about 17 at W = 14). At W = 7, 42-47% of community-day records have 17 or more card tokens (up to 531). Their meaning is unknown.
* Most tracks last one day (median age 1 at every W); a few persist for all 30 days. Zero split/merge flags are structural under the frozen Jaccard rule, not a finding.
* Mean null-calibrated percentiles of observed candidates at W = 7: u_S 0.713-0.717, u_T 0.583-0.596 (all W: 0.616-0.735 and 0.582-0.617). This is a descriptive hint, not a null test.
* Member transactions over-represent the no-edge segment (about 17% versus 10% of the validation population) because those transactions almost always carry an infrastructure edge. The three segments were never merged.
* No alerts, labels, risk scores, R_p or R_h were used.

## 5. E0 (exploratory, descriptive reference only)
**Specification status.** The frozen design gives E0 one line. The addendum (`E0_ADDENDUM_post_Tier0_exploratory.md`, SHA-256 `80a63e637bbedec2934a0c7384f747aba4a46d40107b0662b7cb1268090705c5`) was written **after Tier 0 results existed** and is **not a preregistration**. The time-shuffled null was omitted entirely. Only the degree-preserving null was run: one coherent randomized history per replicate (existing `rewire`, per-day keyed streams, K = 100 per seed) scored with the **observed arrival table**.
**Thresholds** (TRAIN-only, per seed, 30 days 89-121 excluding 96, 103, 118; locked and hashed; no degenerate case):
| seed | tau | TRAIN alerts/day | tied at tau |
|---|---|---|---|
| 42 | 0.8581 | 2.000 | 1 |
| 123 | 0.8599 | 1.967 | 1 |
| 2024 | 0.8543 | 2.000 | 1 |

**Results** (alerts = community-day records with A2 score at or above tau; 30 days 122-151):
| seed | A observed | null A mean (SD) | null min / q05 / q50 / q95 / max | null histories with A >= observed | descriptive reference tail proportion |
|---|---|---|---|---|---|
| 42 | 52 | 32.7 (5.9) | 21 / 23 / 33 / 40 / 56 | 1 of 100 | 0.0198 |
| 123 | 41 | 31.8 (5.4) | 17 / 23 / 32 / 40 / 42 | 3 of 100 | 0.0396 |
| 2024 | 58 | 32.5 (6.5) | 18 / 22 / 32 / 44 / 49 | 0 of 100 | 0.0099 |

Descriptive observations: the observed totals are about 1.3-1.8 times the null means and above the null 95th percentile in each seed. Observed alerts are spread over 34-47 tracks (top track 5-7% of alerted community-days). The excess is concentrated in days 122-131 (28 of 52, 25 of 41 and 33 of 58 observed alerts; null mean about 11 for those days); over the remaining 20 days observed counts (24, 16, 25) are near the null mean (about 21). This split was chosen after seeing the numbers and is unplanned and untested. The null world produces about 1.1 alerts per day against about 2 per day set on real TRAIN windows; the observed validation rate is 1.37-1.93 per day.

**Required labelling.** The null distribution and the tail proportions are **descriptive references only. They are not exact permutation p-values and not statistically calibrated significance.** Reason (addendum section 3): the arrival table is held at its observed value in every null history, so the statistic is not a function of the history alone and exchangeability is not established. The size and direction of the departure are unknown. The smallest attainable value with K = 100 is 0.0099. No alpha, significance threshold or verdict was applied. The three seeds share identical real data and differ only in Louvain seed and permutation streams: they are **not independent replications** and were not pooled.
**Interpretation.** A difference from the degree-preserving null is **not evidence of fraud and not evidence of confirmed coordination.** The null re-randomizes each day's card-infrastructure pairing, which removes a card's reuse of the same infrastructure across days; ordinary benign persistence (households, shared egress, repeated device use) is enough to produce counts above it. Candidate coordinated community; not a confirmed fraud ring. This does not establish that IEEE-CIS contains fraud rings.

## 6. Established findings, exploratory observations, unestablished hypotheses, future work
**Established (recorded facts of what was executed and measured):**
* G0 passed; timing pilot passed; configuration locked before the evaluation benchmark existed.
* Original G4 failed on seeds 42 and 2024 and passed on seed 123; G1, G2, G3, G5 passed; the failure is dominated by the ranking-population effect (the diagnostic values in section 3).
* The exploratory real-data discovery path passed S2 (9 configurations x 4 label variants, including a poisoned-label object), S3 (45 configurations, cutoffs 60, 100, 121, 136, 150, with positive controls) and S4 (including fresh-interpreter runs) (`validation_S2_S3_S4.json`), and 42 further E0 checks (unit, toy driver, report rendering, null-generator properties, S2/S3/S4 extensions, Tier 0 hash reproduction) (`validation_E0.json`).
* E0 reproduced each seed's Tier 0 output hash, executed exactly as specified with no recorded errors or deviations, and all outputs are locked and hashed (`MANIFEST_E0_execution.json`).
**Exploratory observations (not findings):** the Tier 0 descriptives (section 4) and the E0 descriptive comparisons (section 5), including the unplanned early-period concentration.
**Not established / failed:** H5a, H5b, H5c (section 1); benchmark validity under the original protocol (G4).
**Future work (none authorised; each needs its own decision):**
* Any new benchmark validity design or gate. This would require new frozen decisions; G4s was proposed and not approved.
* A ruling on whether G1-G5 apply to the development benchmark.
* E1 (label-based, supporting-only evidence) with its calibration step.
* A recomputed-arrival-table null (the unadopted exactness option), a validated coherent time-shuffled null, or size-bucket stratification of E0.
* Examination of the early-period concentration with a prespecified plan.
* Phase 10 (not started).

## 7. Limitations
* H5a, H5b and H5c were never evaluated on a valid benchmark; nothing here supports or refutes them.
* The E0 specification is post-Tier-0 and exploratory; the likely direction was known when it was written. The time-shuffled null is missing; the tail proportions are descriptive; K = 100 limits resolution.
* The E0 threshold uses 30 serially dependent TRAIN windows whose rows overlap the null-calibration windows; it is in-sample by construction.
* The unit is the community-day, so persistent communities count repeatedly; the degree-preserving null randomizes pairing only and destroys persistence.
* One dataset, one validation period; seeds do not provide independent replications of the real data. Louvain communities are algorithm-dependent; lifecycle statistics are censored at both ends of a 30-day window.
* Infrastructure fields are project-defined proxies with unverified semantics, 82.9% of validation transactions have no infrastructure edge, and the data has no coordination labels. Even a passing synthetic benchmark could only show recoverability of tested planted patterns.
* gamma = 0.5 is at the frozen grid edge; G1/G2/G4 were not computed on the development benchmark.
* No claim is made that any candidate community is fraudulent or coordinated.

## 8. Deviations, incidents and corrections (all recorded, nothing hidden)
* Tier 0, E0 and its thresholds were authorised as an explicit scope deviation after hard stop S6; they do not waive, amend or reinterpret any gate.
* `PRE_RUN_RECORD.md` (preserved unchanged) says the candidate E0 resolutions were "written before any Tier 0 output was seen". They were not recorded in a hashed file before Tier 0, so that statement cannot be verified and is disregarded; the addendum states the post-Tier-0 status correctly.
* Process incidents without effect on results: a first validation run used the wrong interpreter; a harness bug (array compared to a string) crashed one validation before any result; neither was a validation failure.
* The first threshold-calibration lock hashed LF-normalised text while Windows stored CRLF files. Content equality was proven and only the `.sha256` files were rewritten; thresholds were not recomputed (`e0_tau_lock_note.md`). The calibration script source still contains that defect (it cannot overwrite locks).
* The first S3 validation processed validation-period days by hash only, before thresholds existed; its record also exposes record counts. No content was inspected.
* Correction: the 2026-10-12 chat summary of the E0 validation said 44 checks; the validation record contains **42** (15 groups). The record is authoritative.
* The threshold-calibration report remarked that Tier 0 validation-period scores sit above the TRAIN-fixed thresholds, which suggested a higher alert rate; the observed E0 rate (1.37-1.93 per day) is at or below the TRAIN target of 2 per day.

## 9. Provenance
The integrity record (`experiments/phase9_coordinated_fraud/phase9_closure_integrity.json`) holds SHA-256 hashes of this report and of every artifact it relies on, with the verification results. Frozen Phase 9 artifacts, all Phase 1-8 files and all G4/STOP artifacts are unchanged. H5a remains **NOT ESTABLISHED** under the original protocol.
