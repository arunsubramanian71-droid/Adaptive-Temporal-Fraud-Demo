# Phase 10 — Temporal evolution of candidate communities: design (written BEFORE any implementation or execution)

**Status: DESIGN DRAFT FOR REVIEW. NOT AUTHORIZED FOR IMPLEMENTATION OR EXECUTION.** Dated 2026-10-14. Nothing in this document has been implemented or run. Companion: `INSPECTION_REPORT.md` (read-only inspection that grounds the choices below). Phases 1-9 are frozen and unchanged by this document. Phase 9 outcome carried forward: H5a, H5b and H5c are NOT ESTABLISHED under the original protocol; original G4 failed on seeds 42 and 2024 and passed on seed 123; Tier 0 and E0 were exploratory and descriptive only.

Wording rule: the objects are **candidate communities**. Nothing in this phase is, or is claimed to be, confirmed fraud or proof of coordination. Candidate coordinated community; not a confirmed fraud ring. This does not establish that IEEE-CIS contains fraud rings.

Where a decision cannot be grounded in existing evidence it is marked **[UNRESOLVED]** with the reason (collected in section 15); constants marked **[JUDGMENT]** are pre-declared values that no existing data justifies and need explicit approval.

## 1. Research questions and explicit hypotheses
Research questions (questions, not assumed capabilities):
* **RQ1** How do candidate communities persist, disappear, or reappear over time?
* **RQ2** How does membership change (who joins, who leaves, and why)?
* **RQ3** Can growth, contraction and changes in activity be measured reproducibly?
* **RQ4** Can these measurements be summarised as investigative context without any claim of confirmed fraud or coordination?

Why a new instrument is required: the Phase 9 tracker structurally cannot register splits, merges, gap re-appearances or size changes beyond a factor of two (INSPECTION_REPORT section 3). Any RQ1-RQ3 answer therefore needs a new, separately validated tracking module (new files only; Phase 9 code is not edited).

Hypotheses (each may fail; failure outcomes are pre-declared in section 9):
* **H10-1 (method validity; gating).** A lineage tracker that stores the full overlap graph and types events from it reproduces scripted lifecycle events exactly on a hand-built toy suite and recovers them on a synthetic evolution benchmark better than the Phase 9 tracker (for split, merge, dormancy re-link, and growth/contraction) and above pre-declared floors. If H10-1 fails, no real-data evolution result is produced.
* **H10-2 (reproducibility).** On identical real data, evolution summaries are reproducible across Louvain seeds 42, 123, 2024 within pre-declared bands. If not, identity-level conclusions are reported as not reproducible at that (W, gamma).
* **H10-3 (persistence beyond a coherent null).** Membership retention and lineage persistence on the observed history exceed those on coherent degree-preserving histories. This may hold trivially because benign persistence suffices (stated as a possibility, not a result); the useful output is the effect size by stratum, not the rejection.
* **H10-4 (robustness).** The direction of H10-2 and H10-3 summaries is unchanged across the sensitivity grid (section 8). Reversals are reported as instability.
* **H10-5 (descriptive context, no utility claim).** Each alerted community-day can be assigned exactly one evolution state from a closed list, deterministically, with seed-to-seed agreement above a pre-declared band, and the number of lineages is smaller than the number of community-days. **Usefulness to investigators is NOT tested** and cannot be tested without labels or human review ([UNRESOLVED] D8).

## 2. Entities, relationships and time (what exists in this dataset)
* Node types: transaction, **card** (`card1` token; 100% coverage; not a card number), **pseudo_account** (card1 + addr1 + D1 anchor; 88.7%), **netproxy** (id_19, id_20; 23.6% of transactions), **device_fingerprint** (DeviceInfo, DeviceType, id_30-33; 12.0%). Relations: `charged_to`, `used_card`, `via_network`, `from_device` (edge time = source transaction day).
* The community graph is the Phase 9 one: bipartite **card token x infrastructure entity (netproxy or device)** inside a trailing window (t-W, t]. **Card token is the unit of membership** (project rule); pseudo-account is descriptive only and never defines membership or linkage. Excluded entity types (billing address, email domain, product, DeviceInfo alone) stay excluded.
* Time: day granularity; TRAIN 1-121, VAL 122-151, **TEST 152-182 sealed and never opened**; same-day mutual visibility; no calendar dates. Only infrastructure-observed activity (17.1% of VAL transactions) is visible; the unobservable majority is reported and never interpreted.

## 3. Operational definitions
Notation: for day t, window length W and Louvain seed S, the **eligible communities** C(t) are connected components of the Louvain partition (resolution gamma) with at least 3 distinct card tokens and at least one infrastructure entity (identical to Phase 9). Same-day communities are disjoint card sets. For a community C: card set M(C), infrastructure set I(C), size n = |M|, internal link mass. For an earlier community C and a later community C' with x = |M(C) intersect M(C')|:
* Jaccard J = x / (|M| + |M'| - x); forward retention k_f = x / |M|; backward inclusion k_b = x / |M'|.
The **overlap graph** stores, for every ordered pair of communities on consecutive observed days (and between a community and the last community of each dormant lineage), (x, J, k_f, k_b) **without any threshold**. All event typing is derived from it, so threshold sensitivity needs no new Louvain run.

Constants (all fixed before execution; selection rule in section 9): theta_id (identity), theta_ev (relation), theta_cov (coverage), delta (size change), g (dormancy / disappearance), w (newcomer window).

| Concept | Definition |
|---|---|
| **Lineage (community identity)** | A chain of communities linked by continuation or re-link edges. An analytic construct built on a heuristic partition; it is not a real-world group. |
| **Continuation (persistence step)** | One-to-one greedy match of a community on day t to a lineage's last community with J >= theta_id; ties broken by larger x, then lower lineage id, then lower community index. With theta_id = 0.5, g = 0 and no event typing this equals the Phase 9 rule (baseline B10-1). **Extended continuation (needed because J is at most min/max of the sizes):** if J < theta_id but k_f >= theta_ev (absorption growth) or k_b >= theta_ev (contraction to a core), and the relation is the unique such successor of C and C is its unique such predecessor, the link continues the lineage and is typed growth or contraction. Whether to adopt this extended rule is [UNRESOLVED] D12; toy cases T3 and T4b test exactly it. |
| **Dormancy and re-link** | A lineage with no continuation stays dormant for up to g further days; a community with J >= theta_id to its last card set on a day within that period re-links the lineage ("reappearance"). Look-back is backward only. |
| **Appearance (birth)** | A community that is not a continuation, re-link, split child or merge product. Subtype by origin: newcomer-dominated (more than half of members first seen in the dataset within (t-w, t]) or regrouped (otherwise). |
| **Disappearance** | A lineage with no continuation or re-link on any of days last+1 .. last+g; declared at the end of day last+g (known then, never earlier). Last-active day is retrospective. |
| **Membership change (pair C -> C')** | retained = x; joined = M' minus M, subtyped as newcomer (dataset first-seen inside (t-w, t]), transfer (member of another community on the earlier day), re-entry (active in the graph but in no eligible community earlier), other; left = M minus M', subtyped as expired (no infrastructure-linked transaction in the new window), transfer, or isolated (active but in no eligible community). Churn = 1 - J. |
| **Growth / contraction** | For a continuation: r = n(C') / n(C); growth if r >= 1 + delta, contraction if r <= 1 / (1 + delta), else stable. Unlike the Phase 9 tracker, r is not limited by the J threshold because relations are typed from the overlap graph (a community that more than doubles is typed as absorption growth when k_f >= theta_ev, not as a new birth). |
| **Split** | A parent C with at least two children C' having k_b(C -> C') >= theta_ev (each child mostly derived from C) and sum of x over those children >= theta_cov * |M(C)|. The best continuation child (if any) carries the lineage; the others start lineages with a parent pointer and a split flag. |
| **Merge** | A child C' with at least two parents C having k_f(C -> C') >= theta_ev and sum of x >= theta_cov * |M(C')|. The best continuation parent carries the lineage; the others end with a merge pointer. |
| **Activity** | Per community-day: infrastructure-linked member transactions on exactly day t (not window-pooled), distinct active members that day, window link mass, and the Phase 9 arrival-burst statistic T (descriptive). Activity span (days with activity) is reported separately from presence (days the community exists in the window partition) because a W-day window keeps a single burst visible for up to W days. |
| **Alerted candidate community** | A candidate community whose Phase 9 A2 score is at or above the locked threshold for that seed. Not the Phase 9 CSCC (which also needs a risk component) and not a validated notion. |
| **Online vs retrospective** | Online fields use only days <= t (age so far, overlap with previous, size change, activity). Retrospective fields (total lifetime, declared disappearance, survival) carry an `available_at` day and never feed any online field or any selection. |

## 4. Strict temporal causality
* Every quantity for day t uses rows with day <= t only: window (t-W, t], dormancy look-back <= g days, first-seen days computed from data <= t, arrival tables read only at columns <= t.
* No component accepts a future-dependent argument. Tests: future-invariance (corrupting or deleting all data after day c leaves every online output for days <= c bitwise identical, including the lineage module, with positive controls), and a separate test that retrospective fields are never read by online code.
* Louvain communities on day t depend only on the day-t window graph; lineage state at t depends only on earlier outputs.
* TEST (days 152-182) is never loaded; any row with day > 151 raises, as in Phase 9. The cached arrays truncated at day 151 are the only data source.

## 5. Labels and "suspicious" selection
* Fraud labels are **not used anywhere in Phase 10**: not for generating communities, not for tracking, not for thresholds, not for strata, not for selecting which groups to report.
* The "alerted" subset is defined solely by the locked label-free Phase 9 threshold. No group is selected retrospectively by its fraud rate.
* Enforced by: label-poison test, static import checks (no label arrays, no risk or benchmark modules), and by loading arrays through `GraphView` only.
* A label-based retrospective context stage (an E1-like step) is **not part of this design**; if ever wanted it needs a separate design and authorization ([UNRESOLVED] D9).
* The risk model, R_p and R_h are not used (H5b/H5c are not established, and Phase 10 needs no model).

## 6. Null models and synthetic validation
**Null models (real data)**
* **N1 coherent degree-preserving history** (reuse `e0.rewired_arrays` logic; one randomized history generates every overlapping window; per-day keyed streams). Preserves: each day's card multiset, per-infrastructure daily transaction counts, window overlap. Destroys: card-infrastructure pairing, cross-day reuse of the same infrastructure by a card. It is a valid reference only for **membership-only statistics** (functions of the history alone); statistics that use the observed arrival table (alerted subset, T) are descriptive references, not exact, because the table is held fixed. The null hypothesis is narrow and false for real data (benign persistence suffices), so effect sizes, not rejections, are the product. Reuse of the E0 key space versus a new key space is [UNRESOLVED] D5.
* **Seed-noise floor.** The same data run with seeds 42, 123, 2024: measures how much community identity varies from the algorithm alone. It is a measurement reference, not a hypothesis null. If seed agreement is low, day-to-day persistence cannot be interpreted.

**Synthetic validation of the tracker (no real data, no labels)**
* **Toy suite (card-set sequences, no graph)**, exact assertions: (T1) stable community; (T2) growth by +20%, (T3) by x2.5 in one step (the Phase 9 tracker breaks identity); (T4) contraction by 20%, (T4b) contraction to a core by more than half in one step; (T5) even bisection split; (T6) 70/30 split (Phase 9 tracker: unflagged); (T7) three-way split; (T8) 70/30 merge; (T9) three-way merge; (T10) one-day gap then return, (T11) gap of g and g+1 days; (T12) birth with all-newcomer members; (T13) birth by regrouping existing cards; (T14) death; (T15) crossing (two lineages exchanging members); (T16) member swap with constant size; (T17) empty days; (T18) order-permutation invariance of the community list; (T19) identical inputs give identical lineage ids; (T20) split followed by re-merge.
* **Synthetic evolution benchmark** (graph level, entity arrays only: day, card, netproxy, device; no features): scripted lifecycles (growth, contraction, split, merge, dormancy, death, newcomer-heavy vs regrouped births) planted into a real background, with a sealed script of ground-truth events per group-day. It is **independent of the Phase 9 benchmark and its gates**. Dev and eval parts with separate generator seeds, constants chosen on dev only, eval generated once, manifest sealed before detector output. Validity gates are defined on the quantity that matches their purpose (the lesson of the Phase 9 G4 failure, where the gate's quantity and purpose diverged): V1 planted transaction share <= cap [JUDGMENT: 2%]; V2 script consistency (every scripted event realized in the arrays); V3 background distortion: change in the clean background's community structure caused by planting must not exceed the cross-seed noise floor measured on the same clean days (rule fixed in advance; numeric value comes from the measured floor); V4 recoverability precondition: planted groups must be recoverable by the Louvain partition (J >= 0.5 to the observable truth) on a pre-declared fraction of active group-days [JUDGMENT], otherwise tracker errors cannot be separated from partition errors; V5 locked constants before eval generation. Any validity failure is a hard stop with no silent regeneration.
* **Oracle comparator.** The tracker is also applied to the true planted card sets (no Louvain) to separate tracker logic errors from partition noise.

## 7. Metrics and baselines
**Metrics (all descriptive; intervals approximate and labelled)**
* **Persistence:** age so far; lineage lifetime with censoring (survival curve; left-censored lineages reported separately, estimates conditional on birth inside the analysis period; right-censored at day 151); hazard by age; dormancy duration; re-link rate; presence span vs activity span.
* **Membership:** retention k_f, Jaccard, churn; joins and leaves by subtype; newcomer share; per-day member flux.
* **Growth/contraction:** distribution of r; fractions by class (grow/stable/contract); absorption-growth and fast-contraction counts.
* **Activity:** transactions and active members per community-day; change vs the lineage's own past mean (past days only); link mass; T.
* **Events:** counts and rates per 100 community-days for birth, death, split, merge, re-link, by stratum.
* **Compression:** lineages versus community-days.
* **Reproducibility:** per-day community match rate between seeds (J >= 0.5); correlation of per-day event counts; lifetime-distribution distance; agreement of evolution-state labels.
* **Strata:** size bucket {3-4, 5-8, 9-16, 17+}; hub-share class; segment (cold / warm / no-edge, never merged) x infra_observed composition; alerted vs all.
* **Tracking quality (synthetic):** event precision/recall per type with +/- 1 day tolerance; identity continuity (ID switches per planted group); lineage purity.

**Baselines**
* **B10-1** Phase 9 tracker (J >= 0.5, one step, no dormancy), run on the same communities.
* **B10-2** infrastructure-anchored persistence (identity = the infrastructure entity, using the existing star candidates): a Louvain-free reference showing how much instability comes from the partition rather than the data.
* **B10-3** N1 histories (observed versus null).
* **B10-4** the seed-noise floor.
* **B10-5** (synthetic only) the oracle comparator.

## 8. Sensitivity analyses and special structures
* Grid: W in {3, 7, 14}; gamma in {0.5, 1.0, 2.0} (primary W = 7, gamma = 0.5 inherits the Phase 9 lock; gamma = 0.5 sits at the edge of the grid); seeds 42, 123, 2024; theta_id in {0.3, 0.5, 0.7}; theta_ev in {0.3, 0.5, 0.7}; g in {0, 1, 3}; delta in {0.25, 0.5, 1.0} [JUDGMENT grid]. Threshold-dependent results are reported for the whole grid; a single primary setting is fixed on synthetic dev data only (section 9).
* **Giant components and shared infrastructure.** Phase 9 found about 40-47% of W = 7 community-days have >= 17 card tokens (up to about 530) and a giant component covering most entities. Communities are stratified by size and hub share; all headline summaries are shown overall and without the largest stratum. A pre-declared sensitivity applies the Phase 9 hub cap (window degree > 300 cards) uniformly [UNRESOLVED] D6 because that lever was defined as a rescoping decision in Phase 9.
* **Missing identities.** Card tokens are always present; missing pseudo-account keys (about 10% of VAL transactions, the no-edge segment) and missing infrastructure keys (82.9%) make activity invisible, not zero. Visibility is reported next to every evolution quantity.
* **Cold-start entities.** New tokens enter as joiners/newcomers and are separated from returning tokens; segment composition (cold / warm / no-edge) is reported separately and never merged.
* **Unstable boundaries.** Louvain has no warm start and no installed Leiden, so partitions are recomputed daily; boundary instability is quantified by the seed-noise floor and by the unthresholded overlap graph.
* **Window artifacts.** Persistence is reported with presence and activity separated and with N1 histories that share the same window overlap.

## 9. Success and failure criteria (fixed before execution)
| Gate / hypothesis | Pass condition | Failure consequence |
|---|---|---|
| Toy suite (H10-1) | 100% of toy assertions pass | Stop; no benchmark or real-data run |
| Lineage invariants | All property tests pass | Stop |
| Causality, label, determinism tests | All pass with positive controls | Stop; no result is produced |
| Phase 9 compatibility | Candidate generation reproduces the recorded Tier 0 output hashes | Stop |
| Synthetic validity gates V1-V5 | All pass | Stop; no silent regeneration; report as benchmark failure |
| Tracking quality (H10-1) | v2 exceeds B10-1 on split, merge, re-link and growth/contraction event F1; precision and recall >= 0.9 for continuation, appearance, disappearance and >= 0.7 for split, merge, re-link, growth/contraction [JUDGMENT] | Stop; report tracker not validated |
| Oracle comparator | Reported; shows the tracker/partition error split | Report |
| Reproducibility (H10-2) | Bands on seed agreement: high >= 0.8, moderate 0.5-0.8, low < 0.5 [JUDGMENT] | If low: identity-level results reported as not reproducible at that setting (a negative result, not a failure of Phase 10) |
| Beyond-null (H10-3) | Reported as effect sizes by stratum with a randomization reference, labelled descriptive for alerted subsets | None; negative results reported |
| Robustness (H10-4) | Direction unchanged across the grid | Reversals reported as instability |
| Context descriptors (H10-5) | One state per alerted community-day, deterministic; seed agreement in a band; compression factor reported | Report |

Selection of the single primary (theta_id, theta_ev, theta_cov, delta, g): the setting with the best synthetic-dev event F1 under a rule fixed in advance (ties to the smaller value of each constant), evaluated once on synthetic eval; **never chosen using real-data outcomes**. Because the constants are not grounded in existing evidence, this rule and the floors above are [JUDGMENT] and need approval.

## 10. Staging
* **Stage 0** diagnostics and tests: build tests; re-run Phase 9 candidate generation storing card sets (needed to quantify what the Phase 9 tracker missed, which the inspection could not determine); verify the Tier 0 hashes. Needs authorization.
* **Stage 1** toy suite and lineage invariants.
* **Stage 2** synthetic evolution benchmark dev, lock, eval, tracking-quality gate.
* **Stage 3** real-data development period (TRAIN days 40-121, label-free) for reproducibility and sensitivity diagnostics; no constants selected from real-data evolution outcomes.
* **Stage 4** descriptive run on VAL days 122-151 with constants frozen ([UNRESOLVED] D1 on the period).
* **Stage 5** report with the descriptive-versus-validated distinction.
Each stage is separately authorized, mirrors the Phase 9 stop rule (any failed gate stops and is reported), and a deviation is recorded as a dated, hashed addendum, never as an edit of this document.

## 11. Runtime and memory (arithmetic from recorded timings only; nothing profiled)
Per day: 0.053 s (W = 3), 0.093 s (W = 7), 0.157 s (W = 14); E0 averaged 0.099 s per day-run including rewiring. Real-data observed runs on 112 days over the whole grid: about 3,000 day-runs, a few minutes. The N1 null (K = 100, 3 seeds, 112 days, W = 7): about 33,600 day-runs, about an hour. Lineage and overlap computation per day is small (about 10-25 communities). Arrays 26 MB; no features are loaded. The synthetic benchmark generation cost is unknown ([UNRESOLVED] D7). A timing gate like Phase 9's S7 (stop and ask if a median window exceeds 60 s or projected CPU exceeds 24 h) applies.

## 12. Tests, provenance and reproducibility
* New files only (e.g. `src/evolution/`, `tests/test_phase10_*.py`, `scripts/*phase10*.py`); the existing regression runner auto-discovers `tests/test_*.py`, so Phase 10 unit tests must be fast and heavy validation must live in scripts. Phase 1-9 files stay untouched and are re-hashed before and after.
* Tests: section 6 toy suite; lineage invariants; online/retrospective separation; future-invariance, label-poison and determinism on the real-data path (including fresh interpreter with random hash seed); Phase 9 compatibility hash check; null-generator property tests for any new day range; re-typing from the stored overlap graph equals online typing; survival estimator on synthetic data of known hazard; static checks; terminology lint over text and identifiers (ring wording prohibited except the two whitelisted disclaimers).
* Provenance: seeds 42, 123, 2024 (Louvain seed = S); locked constants file with hash written before the evaluation stage; environment versions; every output hashed; a manifest per stage; a dated, hashed addendum for any deviation; design file hashed before implementation.

## 13. Limitations
* IEEE-CIS has no ground-truth labels for coordinated groups of any kind. Real-data community evolution **cannot establish that actual coordinated fraud exists**, and no Phase 10 output may be worded that way.
* Communities are heuristic Louvain partitions (no Leiden available) of a graph built from project-defined proxies (netproxy, device fingerprint) with unverified semantics; only 17.1% of transactions are visible to it.
* The Phase 9 scores are unvalidated (H5a, H5b and H5c not established); the alerted subset inherits that and its thresholds come from 30 serially dependent TRAIN windows.
* Window overlap creates mechanical persistence; the null controls for it only for the stated narrow null.
* One dataset, one period, three seeds that share identical data; 30 validation days give heavy censoring; per-lineage sample sizes are small.
* The synthetic benchmark can only show tracker recoverability on scripted lifecycles in a particular generator; it says nothing about real behavior.
* All thresholds and floors marked [JUDGMENT] are not derived from data.

## 14. Descriptive community evolution versus validated evidence of suspicious coordination
| Phase 10 produces | Phase 10 does not produce |
|---|---|
| Reproducible descriptions of how candidate communities of a heuristic partition persist, change membership, grow, contract, split, merge, pause and vanish | Evidence that any community is fraudulent or coordinated |
| Effect sizes against a narrow randomization reference and a seed-noise floor | A calibrated significance test of coordination |
| Event-type labels and lineage summaries usable as context fields | A validated ranking or alert policy; evidence of usefulness to investigators |
| A validated (or explicitly unvalidated) tracking instrument | Recall or precision on real data |
Validated evidence of suspicious coordination would require ground truth the dataset does not contain; if ever pursued it is a separate programme with its own design.

## 15. Unresolved decisions (need your approval; none is guessed)
* **D1 Period and VAL reuse.** TRAIN 40-121 for development and VAL 122-151 for the descriptive run is proposed; VAL was already used by Phase 9 E0, and a longer horizon (days 40-151) would reduce censoring but blurs the development/evaluation split.
* **D2 Event constants** (theta_id, theta_ev, theta_cov, delta, g, w). Only theta_id = 0.5, g = 3 and w = 3 have precedent (Phase 9 frozen constants: Jaccard 0.5, disappearance 3 days, 3-day arrival window); theta_ev, theta_cov and delta are ungrounded.
* **D3 Numeric floors and reproducibility bands** (section 9) [JUDGMENT].
* **D4 Population.** All candidate communities (primary) versus the alerted subset (secondary, descriptive); reuse of the locked thresholds for W = 7 only.
* **D5 Null key space.** Reuse E0 histories (k = 0..99) or create new keys, which needs a new wrapper and its own property tests.
* **D6 Hub-cap sensitivity** (300 cards) as a Phase 10 sensitivity.
* **D7 Scope, cap and cost of the synthetic evolution benchmark** (including the 2% share cap and the recoverability fraction).
* **D8 Usefulness to investigators** cannot be tested without labels or human review; the design leaves it unclaimed.
* **D9 Any label-based retrospective stage.** None is proposed.
* **D10 New module versus changes to Phase 9 code.** Recommended: new module, Phase 9 untouched.
* **D11 Stage 0 diagnostic re-run** of Phase 9 candidate generation with stored card sets.
* **D12 Extended continuation rule** for size changes beyond a factor of two (section 3). Without it, absorption growth and contraction to a core appear as a death plus a birth; with it, the rule's own thresholds are another ungrounded choice. No other event-typing alternative is proposed without evidence.
