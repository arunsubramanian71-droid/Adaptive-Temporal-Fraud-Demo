# E0 exploratory addendum (post-Tier-0 specification)

Dated 2026-10-12 (local time 20:09 +05:30). Authorization: user message "AUTHORIZE E0 ADDENDUM AND VALIDATION IMPLEMENTATION ONLY" (2026-10-12).

> Exploratory real-data analysis conducted under an explicitly authorized scope deviation after the original Phase 9 protocol halted at S6. Original G4 failed on seeds 42 and 2024 and passed on seed 123. H5a is NOT ESTABLISHED under the original protocol.

## 0. Status of this document (mandatory disclosure)
* This is a **post-Tier-0 exploratory specification. It is not a preregistration made before Tier 0.** The frozen Phase 9 design (sections 10, 12, 15.1) gives only a one-line description of E0 ("observed counts of high-S/T communities vs counts under degree-preserving and time-shuffled nulls on the same real windows (permutation p-values, K_null >= 100 on VAL, seeded)"). It does not specify the alert threshold, the counting unit, the statistic, the null mechanics, or the p-value convention.
* Every choice below was written after the Tier 0 report existed (TIER0_REPORT.md showed observed u_S and u_T means above 0.5), so the likely direction of E0 was known when these choices were made.
* This addendum does not amend, waive or reinterpret any original benchmark validity gate. It does not change any frozen threshold. Original G4 FAILED on seeds 42 (0.795) and 2024 (0.799) and PASSED on seed 123 (0.462). H5a remains NOT ESTABLISHED under the original protocol. E0 results will not be used to justify a G4 amendment, a G4s gate, or any threshold change.
* Scope: E0 (degree-preserving null only). E1, labels, risk, R_p, R_h, risk-based baselines, Phase 10 and all benchmark work are excluded.
* The time-shuffled null is **omitted entirely**. The frozen design names it; omitting it is a scope reduction relative to the frozen E0 description. No replacement is developed here. A coherent time-shuffled null would need its own design, assumptions and validation, presented for separate approval.

## 1. Specification (candidates approved for this addendum)
* Configuration: W = 7, gamma = 0.5 (locked), Louvain seed S in {42, 123, 2024}, null tables `null_tables.pkl` (hash-verified against `locked_config.json`), m_min = 3, edge-weight cap 5. W = 3 and 14 are not run in E0.
* Score: the A2 score exactly as `pipeline.record_score(rec, ("s","t"))`, the float64 mean of u_S and u_T.
* **TRAIN-only alert threshold tau_S (per seed).** Days: real clean TRAIN days 89..121 excluding the null-calibration days {96, 103, 118} (the intersection of 89..121 with `nullmodel.calibration_days()`), i.e. 30 days. Pipeline: `run_discovery` on the real arrays with Louvain seed S. rate(s) = (number of community-day records on those days with score >= s) / 30. **tau_S = the smallest distinct observed score s with rate(s) <= 2.0.** Alert iff score >= tau_S (ties at tau are alerts). If no distinct observed score satisfies rate(s) <= 2.0, or there are no records, E0 is reported **not computable** and execution stops; no replacement rule is invented. tau_S is written to a locked, hashed file before any VAL alert count is computed.
* Counting unit: the community-day record (eligible Louvain components, >= 3 card tokens, >= 1 infrastructure entity), per design section 12. Persistent communities count on every day they are present.
* Statistic: A = sum over t = 122..151 of the number of alerted community-days (equivalent to the mean per day). Per-day counts are reported descriptively. No per-day tests.
* K = 100 null histories per seed.

## 2. Primary null: coherent degree-preserving history
For seed S and replicate k (k = 0..99; validation tests use k >= 1,000,000 which the experiment never uses):
1. For each day d in [116, 151], take rows with card >= 0 and, separately for netproxy and device, infrastructure >= 0.
2. Build per-day `WindowEdges` whose `card` field carries the ROW INDEX (netproxy rows, kind 0, then device rows, kind 1, infrastructure id offset by n_net, both in ascending row order).
3. Apply the existing `graphs.rewire` once per day with `numpy.random.default_rng(SeedSequence([50000 + S, 1, k, d]))`.
4. Write the permuted infrastructure values back into copies of the net / dev arrays by row index. Days outside [116, 151] are untouched. One randomized history therefore generates every overlapping window (each row has one assignment used by all windows containing it).
5. Run the unchanged `realrun.run_discovery` on the rewired arrays with W = 7, gamma = 0.5, Louvain seed S (same as the observed run), days 122..151, and the **observed (real) `ArrivalTable`** for the T component.
Existing functions reused unchanged: `graphs.rewire`, `graphs.window_rows`, `graphs.aggregate`, `graphs.louvain_candidates` (via `pipeline.discover`), `scoring.ArrivalTable`/`t_raw_for`, the locked `NullTables`, `pipeline.discover`, `pipeline.record_score`, `realrun.run_discovery`. New code: the row-index wrapper around `rewire`, the tau calibration driver, the E0 driver and summary, the tail-proportion function, and the validation scripts.

**Preserved:** each day's per-kind multiset of card ids; each day's per-infrastructure transaction counts (row-level degree sequence); each row's day, kind and card; the set of rows that have an infrastructure edge; window overlap; the real T arrival table.
**Destroyed:** card-infrastructure pairing; a card's reuse of the same infrastructure across days (persistence); the link between a transaction's netproxy and its device; exactness of the capped, aggregated weighted degrees and unique-edge counts used by Louvain.
**Interpretation (mandatory in every output):** a difference between observed and null counts is a difference from this specific degree-preserving null. Ordinary benign persistence (households, shared egress, repeated device use) is sufficient to produce one. It is not evidence of fraud and not evidence of confirmed coordination; candidate communities remain "candidate coordinated community; not a confirmed fraud ring", and this does not establish that IEEE-CIS contains fraud rings.

## 3. Review of the claim that the permutation procedure is exact (required before implementation)
**What would make the tail proportion exact.** Let G be the group of within-(day, kind) permutations of infrastructure assignments on days 116..151, and H the history. Suppose (i) under the null hypothesis H0 the law of the observed history is invariant under G (equivalently: conditional on its orbit, the observed history is uniformly distributed over the orbit); (ii) the K null histories are independent uniform draws from that orbit (the rewiring does this: independent uniform permutations per group); (iii) the statistic is a **fixed deterministic function of the history alone**, with every other input (tau_S, the null tables, the configuration, the Louvain seed, the canonical graph construction) fixed in advance and independent of the history; (iv) ties are counted as extreme (>=). Then the K+1 histories are exchangeable and (1 + #{A_k >= A_obs}) / (1 + K) is super-uniform under H0.
**Assumptions that are met:** (i)-(ii) are properties of H0 and of the generator (checked by the property tests); tau_S and the tables are TRAIN-only and fixed before VAL; Louvain with a fixed seed on a canonically ordered graph is deterministic (S4); ties use >=.
**Assumption that is NOT met by the approved design.** The T component reads the arrival table (`ArrivalTable.M`, `birth`) computed from the OBSERVED history and holds it fixed for every null history. The statistic applied to the observed history is F(H_obs; M(H_obs)); applied to a null history it is F(H_k; M(H_obs)), not F(H_k; M(H_k)). The statistic is therefore not a function of the history alone, and the conditional law of the history given M(H_obs) is not invariant under G because rewiring changes M. Exchangeability is not established.
**Consequences (binding on all outputs).**
* The procedure is **not called exact, not called statistically calibrated, and not called a significance test**. It is a **descriptive reference distribution** and its tail proportion is a **descriptive reference tail proportion**, with the label "descriptive; not an exact or calibrated p-value".
* The direction and size of the departure from exactness are unknown.
* Even where exactness would hold, H0 is a narrow hypothesis (uniform within-group arrangement independent across days); it is not a model of "no coordination".
**Possible remedy NOT adopted here (requires explicit approval, decision O1).** Recompute the arrival table from each null history (call `run_discovery` with `table=None`), making the statistic a function of the history alone. That would restore exchangeability under (i)-(iv) but changes how T is evaluated under the null compared with the approved design, so it is a new methodological decision. It is not implemented.

## 4. Reporting plan (descriptive only)
No alpha, no declaration of significance, no binary or three-way verdict, no "consistent/inconsistent" wording. Per seed report: tau_S, the realized TRAIN rate at tau_S and the number of records tied at tau_S, A_obs, the null summary of A_k (mean, SD, min, q05, q50, q95, max, count with A_k >= A_obs) and the descriptive reference tail proportion, per-day observed counts beside per-day null means, and the concentration of observed alerts by track. Seeds share identical real data and differ only in the Louvain seed and permutation streams: they are **not independent replications**, are not pooled, and no combined p-value is formed. With K = 100 the smallest attainable tail proportion is 1/101 (about 0.0099). No multiplicity adjustment is made because nothing is tested.

## 5. Required validation before any tau calibration or E0 computation
Script `scripts/validate_phase9_e0.py`; record `exploratory_realdata/validation_E0.json`. Hash-only comparison; no alert counts, A, tau or tail proportion is computed. Validation null histories use k >= 1,000,000.
* Unit tests: tau selection (distinct scores, ties, degenerate, empty, float exactness), tau day set, tail proportion, report rendering (banner present; no alpha / "significant" / verdict wording; lint clean).
* Null-generator property tests P0-P6: assembly equals `rewire` on `window_rows(v, d, 1)`; overlapping windows share rows; per-(day, kind) multisets of card and infrastructure ids preserved, only infrastructure changes; nothing moves across days or kinds; causality (days <= c unchanged when data after c is corrupted); determinism across processes, order and batching, different k differs; small-case uniformity.
* S2 (poisoned / random / absent labels) over the observed path, null path and tau path; static module check including the new module.
* S3 at cutoffs 125, 136, 150 for the observed and null paths (random-future, burst, truncated, with positive control) and at cutoff 121 for the tau path.
* S4 in-process and fresh-interpreter (random PYTHONHASHSEED).
* Integrity: the observed E0 path reproduces the Tier 0 output hash recorded in `tier0_summary.json` for W7, seeds 42, 123, 2024.
Any failure stops the work and is reported.

## 6. Sequence and gates (each later step needs its own explicit authorization)
(1) this addendum hashed (done before implementation); (2) implement; (3) run the validations above; (4) separate authorization: calibrate and lock tau_S; (5) separate authorization: run E0; (6) report. The E0 driver refuses to run unless this addendum's hash, a passing validation record whose recorded source hashes match the current files, and locked tau files all verify. Stop conditions: any validation failure; no admissible tau; any gate or hash mismatch; the S7 timing gate.

## 7. Open decisions (not decided here)
* O1: arrival-table treatment under the null (section 3).
* O2: whether to report additional descriptive stratifications (size bucket).
* O3: whether any threshold or formal verdict is ever wanted (none is set).
* O4: any coherent time-shuffled null (separate proposal required).

## 8. Environment
python 3.12.0, numpy 2.4.4, scipy 1.18.1, networkx 3.6.1, pandas 3.0.5 (project `.venv`).
