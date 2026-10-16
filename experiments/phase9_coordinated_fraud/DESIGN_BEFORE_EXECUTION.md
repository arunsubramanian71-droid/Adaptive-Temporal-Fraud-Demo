# Phase 9 — Coordinated Fraud Discovery: Design (written BEFORE any execution)

**Status: DESIGN FINAL — all review decisions resolved (R-A to R-F and U1–U4, 2026-10-07; zero unresolved design
decisions; see §24). EXECUTION NOT AUTHORIZED. Nothing in this document has been executed.** No Phase 9 model has been
trained (Phase 9 trains no model at all), no Phase 9 pipeline code exists, no benchmark has been generated, no
community has been detected. The only computation performed during this design stage is one label-free,
TRAIN-period-only structural profile of the existing Phase 4 graph (Appendix B), used solely to ground the
feasibility estimates in Section 18. Phases 1–8 (code, methodology, results, checkpoints, reports) were read, not
modified.

The 2026-10-07 review corrections (§24) were applied in place because the design had not yet been authorized for
execution. From the moment execution is authorized, any deviation must be recorded as a dated addendum, never as an
edit of an earlier section. Execution requires a separate, explicit authorization.

---

## 1. Research question and hypotheses

**H5 (as posed):** *Can the learned temporal graph and transaction relationships identify groups of entities
exhibiting coordinated suspicious behavior that transaction-level fraud classification alone cannot characterize?*

H5 contains three separable claims, because "learned temporal graph" and "transaction relationships" are different
ingredients and Phase 7A already showed the learned temporal memory is a modest, segment-specific contributor
(robust only for cold-start; pooled gain over a schedule-matched static model *not established*). Treating H5 as one
undifferentiated claim would let a structure-only success be misreported as a success of the learned model. The
design therefore tests:

| Id | Claim | Evaluated on |
|---|---|---|
| **H5a (PRIMARY — the only primary success criterion)** | Graph structure + temporal behaviour (label-free, risk-free) recovers *camouflaged* coordinated groups better than methods that rely on transaction-level classification, and better than naive shared-entity grouping | Synthetic planted benchmark (controlled, quantitative) |
| **H5b (secondary)** | Adding the learned transaction-level fraud-risk signal to structure+time improves recovery/ranking where risk is partially informative, and improves discrimination against benign structural look-alikes | Synthetic planted benchmark |
| **H5c (secondary)** | Adding the learned temporal entity state: within the risk signal, the learned *temporal entity state* of the Temporal GraphSAGE adds beyond transaction-level risk ranks alone | Synthetic planted benchmark |
| **H5-real** | On real IEEE-CIS data, (i) structure exceeding degree-preserving/time-shuffled nulls exists, and (ii) flagged communities show later fraud in excess of what transaction-level calibrated risk predicts | Real data — *weak, indirect evidence only; cannot validate coordination* |

**H5a is the ONLY primary success criterion (approved R-B).** H5b and H5c are secondary and **may never substitute
for, rescue, or stand in for H5a**: if H5a is not established, the overall conclusion is "not established" regardless
of any H5b/H5c result (rule restated in §15).

**Null/negative results are explicitly acceptable and pre-declared as possible.** Phase 1 recorded that on this dataset
the "ring" is mostly one compromised account reused, that shared-device sharing carries no label homophily at the
available granularity (DeviceInfo 0.89×; device composite 1.47×), and that the obvious burst signature is absent
(39,459 repeated-purchase groups, mean fraud rate 2.75% < 3.5% base). Phase 9 is not designed to establish that fraud rings exist; it is
designed to measure, honestly, what graph-plus-time analysis can and cannot say, with the synthetic benchmark
supplying the only quantitative evidence and real data supplying qualitative and weakly-inferential evidence.

## 2. Constraints inherited from Phases 1–8 (facts the design must respect)

| # | Fact | Source | Design consequence |
|---|---|---|---|
| F1 | IEEE-CIS has **no fraud-ring / coordination / person / merchant labels** | Phase 1 §17 | No recovery precision/recall on real data. Quantitative H5 evidence must come from a planted benchmark. |
| F2 | Pseudo-account label homophily 86.47% (35×) is largely an **account-level labelling artifact** | Phase 1 §10, R2 | Real-data label analyses (E1) use the **card token** as the clustering/evaluation unit and card-token-level permutation nulls (approved U1); no label feature enters discovery. |
| F3 | Cross-entity lifts are only 1.47–1.80× (card 1.80, netproxy 1.70, device 1.47) | Phase 1 §10 | Shared infrastructure is weakly associated with fraud; coordination ≠ fraud; structure must stand on its own. |
| F4 | `card` = `card1` anonymised token, **not** a card number or cardholder; `netproxy` = (id_19,id_20) and `device_fingerprint` = composite of DeviceInfo/DeviceType/id_30–33 are **project-defined proxies with unverified semantics** | `graph.yaml`, Phase 1 §17 | "Distinct cards sharing infrastructure" ≠ "distinct people sharing a device". All language must say "card tokens" / "infrastructure entities". |
| F5 | Infrastructure coverage is small: netproxy 23.6%, device 12.0% of transactions | Phase 4 | Infrastructure-based discovery can only observe a minority of transactions. Unobservable majority must be reported, not silently dropped. |
| F6 | **Sharing is the base rate.** TRAIN-period profile (Appendix B): 61.1% of netproxy entities and 37.6% of device entities touch ≥3 distinct card tokens; in a 7-day window ≈283 netproxies and ≈168 devices have ≥3 distinct cards | Appendix B | A "≥3 cards share an entity" rule flags hundreds of structures per week, nearly all presumably benign. A null model is mandatory, not optional. |
| F7 | The entity co-occurrence graph is **one giant component**: 89.3% of 170,733 active entity nodes (77.5% even after removing the 471 entities with >200 transactions) | Appendix B | Connected components are useless as communities. Candidate generation must be windowed, degree-aware, and null-calibrated. |
| F8 | Temporal GraphSAGE entity state is inductive: entity states are per-entity memory (64-d) with no node-id embeddings; day-level causal API (`TemporalDataset.day_slice`), persistent `TemporalState` | Phase 7 | Planted synthetic entities can be scored by the unmodified model; risk scores and entity states can be extracted causally. |
| F9 | Risk-model performance **degrades over VAL** (Phase 8: Frozen pooled PR-AUC ≈0.59 at W1 → ≈0.34 at W5, mean over seeds; mean feature PSI only ≈0.01) | Phase 8 §G, §J | Raw risk probabilities are non-stationary; any risk-based component must be rank-normalised within day, or "escalation" will partly measure model drift. |
| F10 | Phase 8 Ad/E arms consumed VAL labels; the **frozen-parameter shared-init checkpoints** (fit on days 1–111 only) never saw VAL labels or VAL-based selection | Phase 8 §E | Risk source for Phase 9 is the Phase 8 shared-init checkpoint only. Phase 7 checkpoints (epoch-selected on VAL) and Phase 8 Ad/E weights are **forbidden** as risk sources. |
| F11 | Day granularity, same-day mutual visibility, zero label delay are the project's established conventions | Phases 4–8 | Prediction time for a community-day record = end of day *t*; zero-label-delay caveat carries over but is largely moot because discovery uses **no labels**. |
| F12 | TEST (152–182) sealed; VAL = 122–151; TRAIN = 1–121; strict 3-way segmentation (cold / warm / no-edge) | Phases 2–8 | All Phase 9 data **use** is days ≤151 (see the disclosed convention in §5); G0 reads no TEST rows at all (§9); segmentation preserved unchanged. |
| F13 | Environment: AMD ROCm/Windows, no `torch_sparse`/`pyg_lib`; networkx 3.6.1 (Louvain, label propagation), scipy 1.18.1, sklearn, pandas present; igraph/leidenalg/python-louvain **absent** and may not be installed | Env probe (this design stage) | Community detection = networkx Louvain on CPU; no new packages. |

## 3. What "coordinated suspicious behavior" means here (operational definition)

Because no ground truth exists, "coordinated" and "suspicious" are defined **separately and structurally**, so that
nothing is called suspicious merely because a classifier said so.

A **candidate coordinated community (CCC)** is a set of ≥ *m_min* distinct **card tokens** (default *m_min* = 3),
observed in a trailing window of *W* days, such that:

* **(C1) Structural coordination** — the card tokens are linked through *shared infrastructure entities*
  (netproxy and/or device_fingerprint) more densely than a degree-preserving null predicts (hub-corrected; §7.1).
  Links through the *same card token* (i.e. multiple pseudo-accounts of one card) are **excluded** from coordination
  evidence, because Phase 1 showed within-card/account reuse is the dominant, label-artifact-laden pattern and would
  be rediscovered trivially.
* **(C2) Temporal synchrony** — member cards *arrive on* (first appear on) the shared infrastructure, or become
  co-active on it, in a burst unexpected relative to that infrastructure's own past arrival rate (§7.2).

A **candidate suspicious coordinated community (CSCC)** is a CCC whose combined score additionally includes
**(C3) risk concentration/escalation** from the learned model (§7.3). C3 is a *scoring* component only; it never
defines candidacy or membership.

Consequences stated up front: (i) a CCC is a structural/temporal hypothesis, not a fraud finding; (ii) structure+time
cannot distinguish "coordinated-benign" (e.g. shared corporate network egress) from "coordinated-fraudulent" — that
is precisely why C3 exists as a separate, ablatable component; (iii) coordination that does not pass through
netproxy/device entities (e.g. via address, email, merchant) is **unobservable** in this schema (excluded entity types
had lift ≈ chance and giant hubs, `graph.yaml`) and is out of scope.

## 4. Entities and relationships that may be used

| Category | Items | Rule |
|---|---|---|
| **Allowed — existing node types** | `transaction`, `card`, `pseudo_account`, `netproxy`, `device_fingerprint` | Exactly the Phase 4 schema. |
| **Allowed — existing relations** | `charged_to`, `used_card`, `via_network`, `from_device` (+ reverse) with `edge_time` = source-transaction day | Unchanged; time-masked by `edge_time ≤ t` and additionally by `edge_time > t−W` (a new *mask*, not a new schema). |
| **Allowed — derived (not new entities)** | card–infrastructure incidence per window; infrastructure arrival counts; card/pseudo-account first-seen day (from past data); within-day risk rank; entity memory states from the frozen model | Derived only from data with day ≤ *t*. |
| **Forbidden as linking entities** | `billing_addr`, `email_domain`, `product`, `device_info_alone` (excluded in `graph.yaml`: lift ≈ 0.9–1.16, hubs 46k–440k) | May appear **post-hoc as descriptive attributes** in case studies only, never in discovery or scoring. |
| **Forbidden — would be invented** | person/cardholder identity, merchant, IP address, calendar dates, label-reporting date, any "ring" label | Not in the dataset (Phase 1 §17). |
| **Forbidden as inputs to discovery/scoring** | any `isFraud` label at any time | Labels are touched only by retrospective evaluation code (§10, §12). |

**Unit of coordination (approved R-A): the card token** (`card` node, 100% coverage). Coordination evidence is
computed **only between distinct card tokens**. **Within-card multi-pseudo-account linkage is excluded from coordination
evidence** — several pseudo-accounts of one card token never count as multiple coordinated actors and never contribute to
S, T, community membership size (*m_min* counts distinct card tokens), or matching.

**Project-wide rule (approved R-A and U1):**
* **card token = the coordination unit** (candidate generation, membership, size, matching, scoring);
* **card token = the E1 clustering/evaluation unit** (§10) — the dependence-aware permutation null, the observed/expected
  counts, and any interval are all defined over card tokens;
* **pseudo_account = descriptive/contextual only**: it may be reported to characterise a community (e.g. account churn
  per card token) and to define the project's cold/warm/no-edge reporting segments, but it **must not define
  coordination, membership, or E1 clustering**, and no score, candidate rule, or hypothesis test of H5a–c uses
  pseudo-account linkage. Card-token clustering is also the conservative choice for the account-level labelling artifact
  (F2): several pseudo-accounts of one card token collapse into one evaluation unit.

## 5. Causal protocol and leakage controls

**Prediction time.** A community-day record *(c, t)* is produced at the end of day *t* using only data with
`day ≤ t` (day granularity, same-day mutual visibility — identical to Phases 7–8). Records are **append-only**: a
later merge/split never rewrites an earlier record.

**Windowing.** Structure at day *t* is computed on the bipartite card–infrastructure graph restricted to edges with
`t−W < edge_time ≤ t` (*W* ∈ {3, 7, 14}; primary 7). Warm-up: for the first evaluation day, the trailing window
reaches into days before the evaluation period — past data only.

**Leakage-control matrix.**

| Leakage channel | Control | Verified by |
|---|---|---|
| Future transactions | `edge_time ≤ t` mask; `TemporalDataset.day_slice(d)` only accessor for model; builder refuses `max_day > 151` | Future-invariance test: corrupt all data after day *t* (features, edges, planted rows) → all outputs for days ≤ *t* bitwise identical (CPU) |
| Future labels | **Discovery and scoring code never receives labels** (labels are not arguments of any discovery function) | Label-invariance test: replace all labels with random bits → discovery outputs bitwise identical; static import check that discovery modules do not reference `y`/`isFraud` |
| Future aggregates | All degrees, arrival-rate baselines, null parameters, rank normalisations use **trailing/past-only** data; **no full-graph IDF/degree statistic** (a full-graph hub degree is a future aggregate) | Test that every statistic's input day set ⊆ (−∞, *t*]; future-invariance test above |
| Future graph structure | Community detection at *t* sees only the windowed graph; track matching uses community *t−1* → *t* only | Same |
| Post-event information | Lifecycle labels that need hindsight (disappearance, total lifetime, final size) are **retrospective descriptors with delayed availability**, never inputs to any score. Only **age-so-far** may enter a score | Static check on score inputs; unit test that lifecycle fields absent from scoring signatures |
| Retrospective influence on current prediction | Calibration constants (null percentile tables, alert thresholds) are fit on **TRAIN-period background only** and **locked** before VAL is touched | Config hash recorded; locked file must pre-date eval generation (ordering enforced in code) |
| Risk model contamination | Risk = Phase 8 **frozen-parameter shared-init** checkpoint (days 1–111 only), per seed. Phase 7 checkpoints and Phase 8 Ad/E weights forbidden (F10) | Checkpoint path whitelist + hash |
| Benchmark ground-truth leakage | Ground-truth manifest is written to a sealed file; detector code path cannot import it; scoring script reads it only after detector outputs are on disk | Static import check; file-ordering/hash check |
| TEST | **Never used, inspected, or reported**; any day > 151 raises; **G0 (§9) opens only the TRAIN and VAL feature files and never loads the shared graph file** | Assertion in every loader; G0 input-file assertions; test |
| Same-day label-free ordering | Within a day, no ordering information is used (day-level snapshot) | By construction |

**TEST-seal statement and disclosed convention (U2).** The G0 comparison is TEST-free at the file level (§9). Outside G0,
the pipeline obtains real-data arrays through the existing Phase 7/8 path (`load_raw_arrays` → `build_temporal_dataset`),
which reads the Phase 4 graph file as in Phases 7–8; rows with day > 151 are removed before any dataset, graph, model,
metric, or statistic is built (`build_temporal_dataset` raises for `max_day` > 151), and no TEST-period value is inspected,
used, or reported. This is the unchanged project convention; the U2 clarification changes only G0.

**Zero-label-delay note.** Phase 8's zero-label-delay assumption does not affect discovery (no labels used). It
matters only for the retrospective real-data statistic E1 (§10), where labels of days (*t*, *t*+7] are read *after* the
fact and never fed back.

## 6. Candidate community generation

Executed independently for each evaluation day *t* and each configuration (*W*, resolution γ):

1. **Windowed bipartite graph.** Nodes: card tokens and infrastructure entities (netproxy, device_fingerprint) with at
   least one edge in `(t−W, t]`. Edge weight *A_ci* = `min(#transactions of card c on infrastructure i in window, 5)`
   (cap limits one-day volume bursts dominating).
2. **Observability filter.** Transactions without any infrastructure edge cannot create links (F5). Counted and
   reported; not discarded from reporting.
3. **Hub handling.** No infrastructure entity is deleted a priori. Hub effects are handled by (a) the modularity null
   model in step 4 and (b) a *reported* diagnostic: fraction of a community's link mass carried by entities whose
   window degree exceeds the 90th percentile. A pre-registered **hub cap** (window degree > *H_cap* = 300 cards) is
   a *rescoping lever*: it may be applied **only** if the timing-pilot gate (§18) trips **and** you approve it in the
   recorded rescoping decision; it is then applied uniformly to all methods. It is never applied automatically.
4. **Community detection.** networkx `louvain_communities` (weighted, `seed` fixed, resolution γ ∈ {0.5, 1.0, 2.0}) on
   the bipartite graph. Louvain can emit internally disconnected communities, so each community is split into its
   connected components post-hoc (Leiden is unavailable and cannot be installed — limitation, §22).
5. **Eligibility.** Keep communities with ≥ *m_min* = 3 card tokens and ≥ 1 infrastructure entity. Everything else is
   dropped from the candidate set (count reported).
6. **Identity tracking across days** (§8): match to day *t−1* communities by Jaccard on card sets (≥ 0.5 continues; ties
   → larger overlap, then lower track id; splits: largest-overlap child continues, others start new tracks flagged
   `split`; merges: largest-overlap track continues, others terminate flagged `merged`). The **same tracker is applied
   to every method** (so tracker quality is not a confound).

Candidate generation uses **no labels and no model output.** This is the circularity firewall (§7.4).

## 7. Community suspiciousness scoring

Each component is converted to a **null-calibrated percentile** *u* ∈ [0,1] (size-bucketed: cards 3–4, 5–8, 9–16,
17+), so components with different scales combine without fitted weights.

### 7.1 Structural coordination score S
Bipartite modularity contribution of the community, *Q_C* = Σ_{c∈C, i∈I_C} (A_ci − k_c k_i / 2m) (window degrees *k*,
total weight *m*) — the excess of internal card–infrastructure link mass over the degree-preserving configuration
model. *u_S* = percentile of *Q_C* among communities found by the identical pipeline on **degree-preserving null
graphs** (same-day endpoint swaps between edges, preserving per-day degree sequences), *K_null* = 10 rewirings per
sampled TRAIN window, pooled over 8 sampled TRAIN windows per configuration, then **locked**.

### 7.2 Temporal synchrony score T
For each infrastructure entity *i* in the community: *n_new(i)* = distinct cards first seen on *i* during the last
*w* = 3 days (first-seen determined from all past data ≤ *t*). Baseline rate λ_i estimated from the preceding 28 days
(strictly before the *w* days), shrunk toward the pooled rate of that infrastructure type (empirical Bayes) for entities
with sparse history; one-sided Poisson tail probability *p_i* = P(N ≥ *n_new* | 3λ_i). *T_raw* = max_i(−log10 *p_i*).
*u_T* = percentile against a **time-shuffled null** (day labels permuted among edges of the same infrastructure
entity), pooled/locked on TRAIN as above. Arrival synchrony relative to the entity's *own* baseline is what separates a
burst from a permanently busy shared egress.

### 7.3 Risk components (scoring only; learned model)
Source: Phase 8 shared-init frozen-parameter checkpoint for the seed, run as a causal stream (persistent state) over
the clean or augmented data.
* **R_p — transaction risk concentration.** Within-day risk **rank percentiles** (F9: raw probabilities drift) of the
  community's member transactions over the last 3 days; community mean compared with a **size-matched random-card-set
  null** drawn from the same day's active cards → *u_Rp*.
* **R_h — learned temporal-state signature.** From the frozen model's `card` entity states **S_card(t)** (64-d):
  (i) *cohesion* = mean pairwise cosine similarity of member-card states; (ii) *shift* = mean cosine distance between
  each member's S_card(t) and S_card(t−3). Each compared with the same size-matched random-card-set null; *u_Rh* = mean
  of the two percentiles. Phase 7A makes it plausible that R_h adds little; "no contribution" is an acceptable outcome.
  For fresh (cold-start) members the state is the learned prior h0, so R_h is reported separately by member
  composition (§9).

### 7.4 Combination and the circularity firewall
Final score = **unweighted mean** of the selected components' *u* (pre-registered; no fitted weights; weight
sensitivity explored on the dev benchmark only). Circularity is prevented structurally:

1. Candidate **membership** is decided by S/T machinery only (no risk, no labels).
2. R enters only as a **ranking** component on already-formed candidates.
3. The *risk-free* score (S+T) is the primary object for every real-data claim; any statement involving R is labelled
   "includes model risk" and **cannot** be used as evidence of coordination (a community with high mean risk will
   trivially contain more fraud because the risk model is itself label-trained — F2).
4. Real-data label evaluation (E1) is performed on **S+T-ranked** communities and conditions on transaction-level
   calibrated risk (§10).
5. **Alert threshold.** The alert threshold τ on the final score is fixed on TRAIN background-only windows so that the
   expected false-alert rate is **≤ 2 alerts/day** (pre-registered target), then locked. Ranking metrics (AUPRC) do not
   use τ; alert-stream metrics use the separate fixed budget *B* (§12).
6. No label-derived feature (including "prior fraud on this account/card") is used anywhere in discovery or scoring —
   explicitly **excluded** because it would re-introduce the Phase 1 labelling artifact as a blocklist.

## 8. Temporal community behaviour

Measured per track, causally; every definition states what is knowable when.

| Behaviour | Definition | Knowable at | Used as score input? |
|---|---|---|---|
| **Formation** | First day a track exists with ≥ *m_min* cards and no predecessor | Day of formation | No (the flag, not a feature) |
| **Growth** | New member cards over the last *w* days ÷ size at *t−w* | Day *t* | Only through T (arrival burst), not as a separate input |
| **Persistence** | Age-so-far = days since formation; survival curve computed retrospectively | Age-so-far at *t*; survival after the fact | **Age-so-far only** (allowed); total lifetime never |
| **Disappearance** | Track has no successor for *g* = 3 consecutive days; declared at *t*+*g*, **timestamped as last-active day but known only at *t*+*g*** | *t*+*g* | **Never** |
| **Behavioural escalation/change** | (i) R_p escalation: mean R_p over the last 3 days minus mean over the track's first 3 days, flagged when it exceeds the 95th percentile of the same difference for size-matched random card sets (TRAIN null); (ii) velocity: transactions/member/day change vs the track's own first-3-day baseline | Day *t* (requires age ≥ 6) | Reported as event flags; not inputs to S/T |

Lifecycle labels are **retrospective descriptors** (they use hindsight to be declared) and are used for evaluation and
case-study narration only. Because R_p escalation can be confounded with risk-model drift (F9), rank-normalised risk
and a random-card-set null are mandatory.

## 9. Populations: cold-start / warm / no-edge

The project's strict 3-way segmentation is preserved **unchanged** and never merged:

* `has_edge_cold` — pseudo-account edge exists, no prior history (genuine cold-start);
* `has_edge_warm` — pseudo-account edge exists with prior history;
* `no_pseudo_account_edge` — no pseudo-account key (missing entity). **Not** folded into cold-start.

Coordination discovery additionally depends on a **separate, orthogonal observability flag** `infra_observed` (the
transaction has ≥1 netproxy/device edge). Reporting is therefore a **3 × 2 cross-tabulation** per community and
overall: segment × infra_observed. Specific handling:

* A transaction without infrastructure edges (76.4% of transactions have no netproxy edge and 88.0% have no device edge, F5;
  the share with *neither* over the full graph is a label-free quantity to be measured and reported in the first
  execution step) cannot contribute to C1/C2. They are the **structurally unobservable population**; their count, fraud base
  rate and segment composition are reported at the top of every results table, and no claim is made about them.
* Community **membership composition** is reported as the fraction of member transactions in each of the three
  segments (computed at each transaction's own time, hence causal).
* Segment-specific reporting of detection quality in the synthetic benchmark: planted members are generated as **fresh
  (cold-start)**, **hijacked-warm** (existing real cards with ≥14 days prior history), or **mixed**; recovery and R_h
  behaviour are reported by composition. This honours the Phase 7A finding that temporal benefits concentrate in
  cold-start, and tests whether that concentration carries to community signals — without assuming it.
* **HARD IMPLEMENTATION GATE G0 (approved R-F; made TEST-free by construction, approved U2).** Segment labels for
  planted transactions require an **array-level** equivalent of `compute_pseudo_account_segments` (the existing function
  takes a `HeteroData`). **No benchmark generation (dev or eval), no planted-data segmentation, and no segment-stratified
  Phase 9 result may be produced until the array-level implementation passes the equivalence test on real data.** G0
  operates **only on explicitly day-truncated TRAIN+VAL data (days ≤ 151)**:
  1. *TEST-free input.* The G0 comparison input is derived from `train_features.parquet` and `val_features.parquet`
     **only**, concatenated in that order (which reproduces the first 501,214 rows of the Phase 4 frame exactly, including
     its row order). The TEST feature file is **never opened**, and the shared Phase 4 graph file (`hetero_graph.pt`, which
     contains TEST rows) is **not loaded** for G0. Assertions: every row has day ≤ 151; row count = 501,214
     (417,559 TRAIN + 83,655 VAL, per the configured expected counts).
  2. *Reference.* The existing, unmodified `add_derived_columns`, `build_hetero_graph` (with a one-column zero placeholder
     for transaction features — segmentation uses only day and pseudo-account edges) and
     `compute_pseudo_account_segments` are applied to that day-truncated frame, producing a **day-truncated `HeteroData`**
     and the reference labels. Comparison is keyed by position **and** TransactionID (entity node indices differ from the
     Phase 4 artifact because ids are built from the truncated key set; segmentation depends only on key identity).
  3. *Exact-match criterion.* The array-level function must return **identical labels (zero mismatches, zero tolerance)
     for every one of the 501,214 real transactions through day 151**; the three labels (`has_edge_cold`,
     `has_edge_warm`, `no_pseudo_account_edge`) are each reconciled by count and by row.
  4. *Tie to the Phase 4 artifact without opening it.* For all 83,655 VAL transactions the reference labels must equal the
     `segment` labels already saved by Phase 7 (`experiments/temporal_graphsage/seed_*/predictions_val.parquet`, produced
     by the same function on the full graph), zero mismatches, joined by TransactionID. This reads only VAL-period saved
     outputs.
  5. *Future independence.* The array-level result on inputs truncated at day *c* must equal the result on the full
     ≤ 151 input restricted to days ≤ *c*, for *c* ∈ {60, 121, 136} (labels use strictly prior history only).
  6. *Same-day semantics.* Labels follow row order — an earlier same-day row with the same pseudo-account key counts as
     prior history, exactly as in the existing function; the gate asserts this on real same-day duplicates, and planted
     rows must carry a deterministic documented within-day order.
  7. *Toy cases.* Hand-built arrays covering first occurrence, repeat occurrence, same-day duplicates, missing
     pseudo-account key, and a planted-style synthetic entity.
  8. *Enforcement.* The test writes a gate-record file (SHA-256 of the two input files, counts, per-criterion pass/fail);
     every benchmark-generation script **refuses to run** unless a passing gate record exists. Failure is a hard stop (S9,
     §20), diagnosed and fixed within Phase 9 — never waived.

## 10. What can and cannot be evaluated quantitatively on real data

**Answer (approved R-C, unchanged): coordination cannot be validated on real IEEE-CIS data — there is no ground-truth
coordination label — so real IEEE-CIS cannot validate H5 quantitatively; H5 is quantitatively evaluable only on a
controlled synthetic benchmark.** Real data supports three tiers of *indirect* evidence, each with its own claim limit:

| Tier | Analysis | Allowed claim | Prohibited claim |
|---|---|---|---|
| **0 — Descriptive discovery** | Run the locked pipeline on clean real days 122–151; report counts, sizes, lifecycle statistics, segment × observability composition, hub diagnostics | "The pipeline flags *N* candidate communities per day with these properties" | Any statement of validity |
| **1 — Null-controlled structure (E0)** | Compare observed counts of high-S/T communities with counts under degree-preserving and time-shuffled nulls on the same real windows (permutation p-values, K_null ≥ 100 on VAL for this test only, seeded) | "Real data does/does not contain coordination-like structure exceeding these nulls" | That the structure is fraudulent |
| **2 — Forward excess fraud (E1, weak, label-based, retrospective)** | For community-days *(c,t)* with *t* ≤ 144, compare the observed number of member **card tokens** with ≥ 1 fraud-labelled transaction on days (*t*, *t*+7] with the expected number under **calibrated** per-transaction risk (per card token 1 − Π(1 − p̂_i) over its transactions in the window; calibration map fit on TRAIN-internal holdout days 112–121 scores — descriptive expectation only); inference by a permutation null that shuffles community membership among same-day, same-risk-stratum **card tokens** (**card token is the clustering/evaluation unit; pseudo-account is not used**, approved U1) | "Flagged S+T communities do/do not show later fraud beyond transaction-level risk" | Recall/precision of coordinated groups on real data; that coordination exists |

Tier 2 is **supporting-only**: it can neither establish nor refute H5a/H5b/H5c. It is subject to the labelling artifact
(account-level labels mean forward fraud is correlated within accounts regardless of coordination) and to the
unmodelled label-delay (Phase 1 R3); cluster-level units and the calibrated-expectation baseline mitigate but do not
remove this.

**Considered and excluded:** training a model on community features to predict next-window fraud (adds training,
adds a second labelling-artifact route, answers a classification-utility question rather than H5, violates the
"no unnecessary complexity" requirement).

## 11. Synthetic planted-pattern benchmark

**Justification.** Phase 1 declared this mandatory: it is the only route to a quantitative H5 number. **Claim limit
(binding, approved R-D):** synthetic success means **only** that the *tested planted patterns* (P1–P3 against decoys
D1–D3, at the tested parameter ranges) are *recoverable under the stated assumptions* — realistic background noise,
the stated camouflage, partial observability, and hub confounding. It is **never** to be interpreted as evidence that
IEEE-CIS contains real fraud rings or any real coordinated behaviour. Synthetic results are reported in separate tables,
figures, and a separate report section from real-data results; no sentence may combine them into one claim.

**Preservation (approved 2026-10-07).** The benchmark is preserved as designed, because quantitative ground truth is
otherwise unavailable. The following elements are **non-removable and non-weakenable**: camouflage regimes
(α ∈ {0, 0.5, 1}); hard-negative decoys D1–D3; popular-hub infrastructure cases; partial observability (*q_obs*); dev/eval
separation with the locked-config ordering; the validity gates G0–G5. Changing any of them requires a dated addendum
with explicit approval.

**Naming rule (approved 2026-10-07).** Synthetic constructs are called *planted coordinated groups* (and *decoy groups*);
the word "ring" is not used for them in code identifiers (e.g. `group_id`, not `ring_id`), manifest fields, file names, plots, or
reports.

### 11.1 Construction principle
Plant synthetic entities and transactions into an **in-memory copy** of the real arrays (same dict interface as
`load_raw_arrays`, so unmodified `build_temporal_dataset` and `TemporalGraphSAGE` apply). The Phase 4 graph file and
all Phase 1–8 artifacts are never written. Real background provides genuine hub structure, benign sharing, and
temporal texture ("realistic noise"). Synthetic transaction ids are in a disjoint range; rows are merged in stable
day order. **Planted transactions ≤ 5% of real transactions in the planted period** (generator scales counts to
enforce this).

### 11.2 Planted patterns
| Pattern | Description | Ground-truth label |
|---|---|---|
| **P1 shared-infrastructure burst group** | *n* card tokens (and their pseudo-accounts) first appear on *k* shared infrastructure entities within a formation span Δ | planted coordinated group |
| **P2 account-cycling group** | Persistent infrastructure; new card/pseudo-account members appear, transact briefly, vanish (formation/disappearance churn) | planted coordinated group |
| **P3 dormant-then-escalate group** | First third of its life benign-looking, then riskier feature profile (behavioural escalation with a recorded escalation day) | planted coordinated group |
| **D1 steady shared-egress decoy** | Large benign group on a popular infrastructure entity, continuous arrival, no synchrony (corporate-NAT-like) | hard negative |
| **D2 synchronous-on-hub decoy** | Burst of new cards on a *high-baseline-arrival* infrastructure entity (burst not surprising relative to the entity's own rate) | hard negative |
| **D3 small household decoy** | 2 cards persistently sharing one entity (below *m_min*) | hard negative |

### 11.3 Parameters (pre-registered ranges; each planted group draws independently, seeded)
* group size *n* ∈ {3, 4, 6, 8, 12, 16, 24, 32}; infrastructure count *k* ∈ {1, 2, 3}, netproxy/device mix random;
* infrastructure novelty: *fresh* (new synthetic entity) / *rare real* (real entity with trailing degree ≤ p25) /
  *popular real hub* (≥ p90) — popular-hub groups are the hardest by design;
* formation span Δ ∈ {1, 3, 7} days (synchrony); duration 4–20 days (eval); right-censoring by the end of VAL is
  recorded;
* member composition: fresh / hijacked-warm / mixed; per-member 1–4 transactions/day while active;
* **partial observability** *q_obs* ∈ {1.0, 0.5, 0.25}: probability a planted transaction carries the infrastructure
  edge (mimics 12–24% real coverage);
* **camouflage α** ∈ {0 (camouflaged), 0.5 (partial), 1 (overt)}: fraction of planted transactions whose 470-d
  feature vector is resampled whole from the real **fraud** pool, remainder from the real **non-fraud** pool. Pools use
  only real transactions from days **strictly before** the planted day and ≥ 112 for the eval benchmark (out of sample
  for the days-1–111 risk model — avoids in-sample risk inflation); dev pools use days < planted day. Labels are used
  **only** inside the generator to build pools and set planted transactions' `isFraud` (planted coordinated groups 1, decoys 0); the
  detector never sees labels;
* P3 escalation: α rises from 0 to 1 at the recorded escalation day.

### 11.4 Ground truth definition
Manifest entries per planted group: member card tokens (+pseudo-accounts), infrastructure entities, formation day, per-
member first-seen days, active interval, escalation day (P3), regime α, composition, *q_obs*, pattern id, decoy flag.
**Observable ground truth at day *t*:** *G_t* = planted cards with ≥ 1 transaction on or before *t* — a card that has
not yet transacted cannot be in any detector input, so early partial overlap is not penalised as error.
**Background label noise:** real background entities not planted are treated as negatives although real unknown
coordination may exist among them; reported precision is therefore a **lower bound**.

### 11.5 Temporal generation and noise
Arrival times follow the pattern specs above on top of the real daily volume; amounts/features come from resampled real
vectors, so marginals match reality; infrastructure for "rare/popular real" groups reuses real entity ids, so planted
groups share hubs with real traffic. Known artefact (documented): resampled whole feature vectors carry real-entity-specific
fields (e.g. D-columns, card attributes) inconsistent with the synthetic entity.

### 11.6 Dev / eval separation (train/evaluation separation)
| | **Dev benchmark** | **Eval benchmark** |
|---|---|---|
| Period | planted in days 62–121 | planted in days 122–151 (formation ≤ 146) |
| Generator seed | 10000 + S | 20000 + S |
| Purpose | choose γ **for each *W*** from the frozen grid using the **label-free S+T** score (primary analyses use *W* = 7; *W* = 3 and 14 are pre-registered sensitivity analyses, each with its own dev-selected γ); calibrate null tables | single, final, run-once evaluation |
| Risk component | **not tuned** (fixed equal-weight rule) — dev planted days ≤111 would be in-sample for the risk model | evaluated |
| Parameter ranges | as §11.3 | as §11.3 with ranges extended at the small end (*n* ∈ {3, 6, 12, 24}, Δ = 7, duration 4–20) to test generalisation |

Order is enforced in code: the locked-configuration file (hash recorded) must exist **before** the eval-benchmark
generation script will run; the manifest is sealed; detector outputs are written before the scoring script opens the
manifest. Planted counts per seed: dev 60 coordinated groups + 30 decoy groups; eval 36 coordinated groups (12 per regime α) +
18 decoy groups, scaled to the ≤ 5% cap.

### 11.7 Benchmark validity gates (G0 before any generation; G1–G5 before any detector output is examined; failure → stop and review, never silent regeneration)
* **G0 segmentation equivalence (hard implementation gate, §9, approved R-F and U2):** computed on explicitly
  day-truncated TRAIN+VAL data (days ≤ 151, TEST file never opened, shared graph file never loaded), the array-level
  segmentation matches the existing `compute_pseudo_account_segments` exactly for all 501,214 real transactions, ties to
  the saved Phase 7 VAL segment labels, and passes the future-independence and toy-case checks. **Benchmark generation
  must not start until G0 passes.**
* **G1 camouflage:** for α = 0 planted transactions, fraction in the top 1% of within-day risk ≤ 3%.
* **G2 sensitivity:** α = 1 planted transactions' top-1% fraction ≥ 5× the α = 0 fraction.
* **G3 planting fraction** ≤ 5%.
* **G4 interference:** mean |Δ risk percentile| of real transactions between clean and augmented streams ≤ 0.5
  percentile points (planting must not distort the background it is meant to be embedded in).
* **G5 manifest consistency:** every manifest member exists in the augmented arrays with the specified timing.

## 12. Metrics

**Why not accuracy-style metrics:** the negative class (all background cards/entities) outnumbers planted groups by
orders of magnitude and is label-noisy, so accuracy, entity-level ROC-AUC, and whole-partition ARI/NMI are dominated by
background singletons and are **excluded**. Metrics are community-level and ranking-based.

**Matching.** A discovered community-day record *c* at day *t* matches planted group *g* (active at *t*) if
J_t = |cards(c) ∩ G_t| / |cards(c) ∪ G_t| ≥ 0.5 (primary), ≥ 0.3 (sensitivity). Unmatched records, and records matching
decoys or background, are false alerts.

**Synthetic — primary (retrospective ranking quality).** *Track-level AUPRC at J ≥ 0.5*: tracks ranked by their score
(max over life), positive if the track ever matches a planted coordinated group. Track-level (not record-level) so a
persistent planted group is one positive, not many.

**Synthetic — secondary.**
* Causal alert-stream metrics at a fixed budget *B* alerts/day, *B* ∈ {5, 10, 20} (primary 10; chosen as an
  illustrative analyst capacity, with the other values as sensitivity): planted-group recall (detected if any matched
  alert during its active span), false alerts per day, mean **detection latency** (days from formation to first matched
  alert, using only information available at that day).
* Membership quality on matched pairs: card-level precision/recall/F1.
* Lifecycle: formation-day error; track-identity fragmentation (ID switches per planted group); disappearance-declaration delay;
  escalation-detection latency (P3) and false-escalation rate on decoys/background.
* Stratified reports by regime α, composition, infrastructure novelty, Δ, size, *q_obs*; dose–response of
  method-minus-baseline against α.
* 95% bootstrap intervals over planted groups within a seed; mean/SD/min/max across the 3 seeds.

**Real data.** Counts of alerted communities/day vs null expectation (E0); persistence/stability distributions
(consecutive-day Jaccard, survival); hub-share, density, conductance; E1 excess-fraud statistic with cluster-permutation
p-value and interval. No recovery metric is computed on real data.

## 13. Baselines and controls (what "improvement" means)

Improvement means beating the best of the following on the same data, same tracker, same alert budget:

| Id | Method | Question it answers |
|---|---|---|
| **B0** | Random control: size-matched random card sets from the same day's active cards | Chance level |
| **B1** | Naive shared-infrastructure grouping: every infrastructure entity with ≥ 3 distinct cards in window → one group, ranked by size | What an analyst with a SQL `GROUP BY` gets |
| **B2** | Structure-only hub-corrected communities (S only; no time, no risk) | Does the community/null machinery alone help? |
| **B3a** | **Risk-first grouping** ("transaction-level classification alone"): take transactions in the daily top-1% risk percentile, group their cards through shared infrastructure *among high-risk transactions only*, rank groups by mean risk | The direct comparator for H5's "transaction-level classification alone" |
| **B3b** | Same candidates as the proposed method, ranked by R_p only | Separates candidate generation from scoring |
| **B4** (real data only) | Degree-preserving + time-shuffled null communities | Is real structure above null? |

Not included: a label-aware (prior-fraud/blocklist) baseline — discovery is label-free by design and a label-aware
baseline would mostly re-measure the labelling artifact (Phase 1 R2).

## 14. Ablations (graph structure vs risk signal vs temporal information vs community structure)

| Id | Variant | Isolates |
|---|---|---|
| A1 | S | structure alone |
| A2 | S + T | **+ temporal information** |
| A3 | S + R_p | structure + risk, no time |
| A4 | R_p on structure candidates (= B3b) | risk alone |
| A5 | S + T + R_p | full (transaction risk) |
| A6 | S + T + R_h | learned temporal state instead of transaction risk |
| A7 | S + T + R_p + R_h | both |
| A8 | Star neighbourhoods (per-infrastructure groups) scored with S+T+R_p | **community detection** vs simple groups |
| A9 | Random partition of the same candidate pool preserving size distribution | negative control for community structure |
| A10 | Cumulative graph (no trailing window) vs trailing window | whether windowing itself, not the scores, drives results |

Primary comparisons are pre-declared (§15); the remainder are descriptive with intervals.

## 15. Statistical design and decision rules

Seeds S ∈ {42, 123, 2024}. Derived seeds: dev generator 10000+S, eval generator 20000+S, Louvain `seed`=S, null
rewiring 30000+S, case-study sampling 40000+S, permutation tests 50000+S. Risk checkpoint = Phase 8 shared-init for S.
**Seed caveat:** the real background is identical across seeds, so seeds capture generator, detector, and risk-model
initialisation variance, **not** sampling variance of the real data; intervals are therefore optimistic for real-world
transfer (reported).

**Decision rules (fixed in advance; same style as Phases 7A/8):**
* **H5a supported** iff, on the eval benchmark's camouflaged (α = 0) groups, A2 (S+T) beats **each** of B1, B2, B3a, B3b on
  track-level AUPRC@J≥0.5 in ≥ 2 of 3 seeds, **and** the mean difference exceeds the larger of the two compared
  methods' across-seed SDs, **and** guard G-cal holds: on real background-only VAL days, false alerts/day at the
  TRAIN-calibrated threshold ≤ 1.5× target. If G-cal fails, H5a is reported **not established** regardless of the other
  conditions (precision claims would be unreliable).
* *Reading note for H5a:* at α = 0 the risk-based baselines B3a/B3b are blind **by construction**, so beating them is
  expected and only shows the benchmark behaves as intended; the discriminating comparisons are **B1 and B2**, and the
  dose–response curve over α shows where risk becomes informative. Reports must say this.
* **H5b (secondary) supported** iff A5 beats A2 under the same rule on partial/overt regimes (α ∈ {0.5, 1}) pooled, and on the
  decoy-discrimination comparison (false alerts on D1/D2 decoys at matched recall).
* **H5c (secondary) supported** iff A6 (or A7) beats the matching R_p variant (A5) under the same rule.
* **Non-substitution (approved R-B):** H5a is the **only** primary success criterion. No H5b or H5c result — however
  strong — may substitute for, compensate for, or rescue a failed or unestablished H5a, and no combination of secondary
  results can be counted toward an overall "supported". Secondary results are reported as secondary.
* **Overall wording:** "supported" is permitted **only if H5a holds**, and then only as: *the tested planted patterns are
  recoverable by structure and temporal behaviour under the stated assumptions* (approved R-D) — **never** as evidence that
  IEEE-CIS contains real fraud rings or real coordinated behaviour. If H5a holds and H5b/H5c do not, the report must
  state that **graph structure and temporal behaviour — not the learned model — drive the discovery**. If H5a fails:
  "not established," irrespective of H5b/H5c. H5-real (E0/E1) can only be worded as consistent or inconsistent; it never
  upgrades or downgrades H5a–c.
* Paired/Welch p-values are descriptive only (n = 3, uncorrected). Only H5a is primary; H5b, H5c, ablations, and
  stratified analyses are secondary and multiplicity is stated.
* No rule may be altered after results are seen.

### 15.1 Frozen parameter registry (approved U3, 2026-10-07)

Every value below is **frozen exactly as written**. None may be tuned or altered after any evaluation result is seen;
changing one requires a dated addendum with explicit approval.

| Parameter | Frozen value |
|---|---|
| Minimum community size *m_min* | 3 distinct card tokens |
| Trailing window *W* | {3, 7, 14} days; **primary *W* = 7**; *W* = 3 and 14 are pre-registered sensitivity analyses |
| Louvain resolution γ | {0.5, 1.0, 2.0} (γ is selected per *W* on the dev benchmark only, §11.6) |
| Null calibration | 10 rewirings × 8 sampled TRAIN windows per (*W*, γ) configuration, then locked |
| TRAIN-calibrated alert target | ≤ 2 false alerts/day on TRAIN background-only windows |
| G-cal false-alert guard | ≤ 1.5× the target on real background-only VAL days |
| Alert budgets *B* (alerts/day) | {5, 10, 20}; **primary *B* = 10** |
| Jaccard matching | 0.5 primary; 0.3 sensitivity |
| Planted counts per seed | dev: 60 coordinated groups + 30 decoy groups; eval: 36 coordinated groups (12 per camouflage regime α) + 18 decoy groups |
| Planting cap | planted transactions ≤ 5% of real transactions in the planted period |
| Validity gates | G1: α = 0 top-1% risk fraction ≤ 3%; G2: α = 1 fraction ≥ 5× the α = 0 fraction; G3: planting fraction ≤ 5%; G4: mean \|Δ risk percentile\| of real transactions ≤ 0.5 points; G5: manifest consistency; G0 as in §9 |
| Other frozen constants | edge-weight cap 5; *w* = 3 days for T/R_p/R_h windows; 28-day baseline for T; *g* = 3 days for disappearance; Jaccard 0.5 for track continuation; size buckets {3–4, 5–8, 9–16, 17+}; *K_null* ≥ 100 for E0 on VAL |

## 16. Case studies of real discovered communities

Selection (pre-registered, seeded, to prevent cherry-picking), per seed, from the **clean real** run on days 122–151:
(a) top-10 distinct tracks by peak S+T score; (b) 5 largest tracks; (c) 5 tracks sampled uniformly from the alerted
set; (d) 5 **hub-dominated high-S** tracks (false-positive-prone). Presentation uses seed 42; the other seeds appear in
appendix tables.

Characterisation template per case: size (cards, pseudo-accounts, infrastructure entities); lifecycle timeline
(daily active members, new members, transactions, S/T/R series); segment × infra_observed composition; hub-share and
density; risk percentile distribution **and** R_h signature (labelled "includes model risk"); comparison against its
null-matched community; descriptive attributes (ProductCD, card4/card6, P_emaildomain — **post-hoc descriptors only**).
A separate column, clearly headed *"retrospective — labels revealed after scoring, not validation"*, shows the fraud
card-token rate with a card-token-clustered interval. Every case carries the standing statement: *"candidate coordinated community; not
a confirmed fraud ring."* Failure-mode cases (benign-looking, hub-dominated) are included deliberately.

## 17. Terminology (enforced)

Allowed: "candidate coordinated community" (CCC), "candidate suspicious coordinated community" (CSCC), "shared-
infrastructure community", "planted coordinated group" / "decoy group" (synthetic only).

**Prohibited in every real Phase 9 output (report, tables, plots, case studies, logs, file names, identifiers):** any
statement that a real community is a "detected", "discovered", or "confirmed" fraud ring; "fraud ring members";
"criminal network"; "gang"; "syndicate"; "mule ring/network"; "organised fraud"; and any claim of precision/recall on real
data. The synthetic benchmark never uses "ring" for planted constructs (naming rule, §11).

**Lint (enforced by test over the final report, case-study files, plot labels, manifests, and Phase 9 code
identifiers):** the prohibited terms above fail the test, with exactly two whitelisted uses — the standing disclaimers
*"candidate coordinated community; not a confirmed fraud ring"* and *"this does not establish that IEEE-CIS contains fraud
rings"*. This design document is exempt only where it quotes Phase 1 or defines the prohibition itself.

## 18. Computational feasibility (measured, not assumed)

Measured (Appendix B, label-free, TRAIN days ≤ 121, 417,559 transactions): entity types active: pseudo_account 152,592,
card 12,275, netproxy 2,168, device_fingerprint 3,736; distinct entity pairs: 41,641 card–netproxy, 34,980
card–device, 30,939 netproxy–device, 152,592 pseudo_account–card, 51,018 pseudo_account–netproxy, 46,574
pseudo_account–device (≈ 0.36M across all six pair types); max transaction degree: card 10,407, netproxy 4,900, device
2,556; max distinct cards on one netproxy 427, on one device 835. Per 7-day window: ≈ 685 netproxy and ≈ 582 device
entities active, ≈ 283 / ≈ 168 with ≥ 3 distinct cards.

Estimates (order of magnitude, to be confirmed by the timing pilot):
* **Bipartite window graph:** ≈ 10³–10⁴ card nodes, ≈ 10⁴–10⁵ weighted edges per window — trivially in memory.
* **Candidate communities per window:** ≈ 10²–10³ before eligibility filtering; ≈ 10⁴ community-day records per method
  per seed over 30 days.
* **Louvain (networkx, pure Python):** seconds to ≈ 30 s per window → eval run ≈ 30 days × ≈ 12 method variants × 3 seeds ≈
  10³ runs ≈ hours of CPU; null calibration: 10 rewirings × 8 TRAIN windows × 9 configs ≈ 720 runs ≈ 1–6 h (the dominant
  cost, one-off). **This estimate is uncertain by ~5×.**
* **Risk scoring:** one causal stream of the frozen model over days 1–151 (≈ 501k rows): GPU peak ≈ 1.9 GiB (Phase
  7A/8 measurement), seconds–minutes per pass; 3 seeds × (dev, eval, clean) = 9 passes. CPU also feasible.
* **Planted-array memory:** features 501,214 × 470 × 4 B ≈ 0.94 GB per copy; built and freed sequentially (31 GB RAM
  system). Planted rows ≈ ≤ 5% of the planted period.
* **ROCm/Windows:** only the risk-scoring stream uses the GPU through already-validated code; community detection is
  CPU (networkx/scipy). No `torch_sparse`/`pyg_lib`/new package is needed. **Bottleneck:** networkx Louvain speed and
  null calibration, not memory.
* **Timing-pilot gate (approved R-E):** before full execution, time community detection on **one TRAIN window** (no
  outcome information involved). **If the median community-detection time exceeds 60 seconds/window, OR the projected
  total CPU runtime exceeds 24 hours, execution STOPS and a rescoping decision is requested from you.** **No fallback is
  pre-authorized** (approved U4). No rescoping — fewer windows, reduced null calibration, a reduced configuration grid,
  the hub cap, or anything else — is applied unless and until you approve it in a recorded decision. **Louvain is never silently replaced by another algorithm**; replacing it
  would itself require your explicit approval as a design change.

## 19. Reproducibility

Seeds as §15; all stochastic components take explicit seeds; node and community ordering sorted by id with a fixed
tie-break; CPU for community detection (deterministic); GPU risk stream compared at the Phase 7A noise-floor convention,
CPU exact where practical. Every run writes a config JSON whose hash is recorded in a manifest with SHA-256 of every
artifact, environment versions, split boundaries (TRAIN 1–121, VAL 122–151, TEST sealed), checkpoint hashes. A
determinism test re-runs one window twice and compares output hashes. No hard-coded absolute paths; all scripts accept
`--seed`.

## 20. Failure criteria and stop conditions

**Hard stops (halt and report; no workaround):**
* **S1** any day > 151 used, referenced, inspected, or reported; or a TEST file opened by G0; or the shared graph
  file loaded by G0;
* **S2** label-invariance test fails (any discovery output changes when labels are randomised) or discovery code
  references labels;
* **S3** future-invariance test fails (outputs for days ≤ *t* change when data after *t* is corrupted);
* **S4** determinism test fails;
* **S5** detector code path can access the ground-truth manifest, or manifest-before-detector ordering violated;
* **S6** any benchmark validity gate G1–G5 fails;
* **S7** timing-pilot gate tripped (median > 60 s/window or projected CPU > 24 h) or memory limit exceeded → **STOP and
  request a rescoping decision (§18)**; **no fallback is pre-authorized**; Louvain is not replaced; the grid, null
  calibration, and hub cap are not changed without your explicit approval;
* **S8** benchmark non-diagnostic: B1 achieves AUPRC ≥ 0.95 in *every* regime (too easy) or every method (including the
  full one) is within noise of B0 in *every* regime (too hard) — report as "benchmark non-diagnostic", draw no
  conclusion;
* **S9** gate G0 (array-level segmentation equivalence, §9) not passed: any mismatch with the existing
  `compute_pseudo_account_segments` on the day-truncated TRAIN+VAL input (days ≤ 151), any mismatch with the saved Phase 7
  VAL labels, a failed future-independence or toy-case check, or no passing gate record — **benchmark generation must not proceed**; diagnose and
  fix within Phase 9, never waive.

**Soft failures (reported, claims downgraded, not concealed):** G-cal false-alert guard violated; any decision-rule
condition fails; null-calibrated percentiles unstable across TRAIN windows (reported with intervals); R_h shows no
contribution; real-data E0 shows no excess structure.

## 21. Artifacts Phase 9 will produce

* **Code** (`src/coordination/`): `benchmark.py` (planting + manifest), `windows_graph.py` (windowed bipartite graph),
  `communities.py` (Louvain + connectivity split), `nullmodel.py` (rewiring, time-shuffle, calibration tables),
  `scoring.py` (S, T, R_p, R_h, combination), `tracking.py`, `lifecycle.py`, `baselines.py` (B0–B3b), `metrics.py`,
  `segments_array.py` (array-level segmentation; **gated by G0**), `leakage.py`. Identifiers use "group", never "ring".
* **Scripts:** `scripts/build_phase9_benchmark.py`, `run_phase9_discovery.py`, `evaluate_phase9_benchmark.py`,
  `run_phase9_realdata.py`, `validate_phase9_leakage.py`, `compare_phase9.py`, `make_phase9_case_studies.py`.
* **Tests** (`tests/test_phase9_*.py`, run by `scripts/run_tests.py`): window/mask arithmetic; planted-array
  invariants (ids disjoint, ≤ 5%, chronological, manifest consistency); **G0 segmentation equivalence with the existing
  function on real days ≤ 151 (hard gate; writes a gate record the generators require)**; label-invariance; future-invariance; determinism; matching/Jaccard/AUPRC unit tests on hand-built toy
  cases; tracker unit tests (continue/split/merge); null-model invariants (degree preservation); terminology lint.
* **Experiment outputs** (`experiments/phase9_coordinated_fraud/`): `segmentation_equivalence_gate.json` (G0 record);
  timing-pilot record; locked config + hash; `benchmark_dev/`,
  `benchmark_eval/` (augmented-array manifests, sealed ground truth); `discovery/` per seed × method × configuration
  (community-day records, tracks); `ablations/`; `real_discovery/`; `leakage_validation/` (logs + JSON);
  `case_studies/`; `comparison.json`; `manifest.json` (hashes, environment).
* **Tables:** per-seed, per-regime metric tables; baseline/ablation comparison with seed-matched differences;
  segment × observability composition; E0/E1 tables; resource table.
* **Plots:** dose–response (method − baseline vs α); AUPRC by regime; detection-latency distributions; alerts-per-day
  vs recall curves; lifecycle timelines; null-vs-observed S/T distributions; community-size distributions; case-study
  timelines.
* **Report:** `reports/phase9_coordinated_fraud_report.md` with separate sections for (A) objective & definitions,
  (B) protocol & leakage controls, (C) synthetic benchmark (controlled evidence), (D) real-data discovery (descriptive
  and weak evidence), (E) case studies, (F) ablations, (G) decision-rule outcomes, (H) limitations, (I)
  reproducibility.

## 22. Design audit (performed after drafting) and open issues

Audit performed over: leakage, circular reasoning, unsupported claims, invented labels, invalid metrics,
confounding, unnecessary complexity, computational feasibility. **Result: no blocking flaw found in the design; the
issues below were identified and are handled in the sections cited; the review items were decided on 2026-10-07 (see
"Review decisions" below and §24).**

| # | Area | Finding | Handling / status |
|---|---|---|---|
| A1 | Circularity | Risk scores are label-trained; real-data enrichment of high-risk communities would be tautological | Risk never defines candidacy; E1 uses S+T-ranked communities and conditions on calibrated per-transaction risk; R claims labelled "includes model risk" (§7.4, §10) — **resolved in design** |
| A2 | Labelling artifact | Account-level labelling (F2) inflates any label-based community statistic | Card-token-level unit and permutation null (U1); E1 supporting-only (§10) — **mitigated, residual limitation** |
| A3 | Tautology of planted benchmark | Planting "shared infra + burst" and scoring "shared infra + burst" rewards the method for recovering its own template | Dose–response over camouflage; decoys; patterns violating method assumptions (popular-hub groups, low *q_obs*, no-synchrony D1); dev/eval parameter shift; binding claim limit (§11) — **mitigated; must be acknowledged in the report** |
| A4 | Base rate of sharing | 61% of netproxies share ≥3 cards; giant component 89% | Null-calibrated percentiles, windowing, G-cal guard (§5–7, §15) — **resolved in design; failure possible and pre-declared** |
| A5 | Drift confound | Risk model degrades over VAL (F9); escalation could be drift | Within-day rank normalisation + random-card-set nulls (§7.3, §8) — **resolved in design** |
| A6 | In-sample risk on dev data | Risk model trained on days ≤111; dev planted groups in 62–121 would have in-sample scores | R not tuned on dev; eval pools from days ≥112 (§11.3, §11.6) — **resolved in design** |
| A7 | Invented labels | Planted ground truth is synthetic; "suspicious" scores are not labels | Terminology rules, synthetic/real separation, lint test (§11, §17) — **no invented real-data labels** |
| A8 | Metrics validity | Entity-level accuracy/ROC/ARI would be dominated by background | Excluded with justification (§12) — **resolved** |
| A9 | Confounding by tracker | Different trackers per method would confound | One shared tracker (§6) — **resolved** |
| A10 | Multiplicity | Many ablations/metrics | Only H5a primary; rest secondary/descriptive (§15) — **resolved** |
| A11 | Unnecessary complexity | New neural models, learned combination, forward-utility classifier | None used; fixed-weight combination; forward-utility model excluded (§10) — **resolved** |
| A12 | Seed variance | Real background identical across seeds | Stated; intervals optimistic (§15) — **disclosed limitation** |
| A13 | Algorithm limitation | Louvain, not Leiden (cannot install); disconnected communities possible | Connectivity post-split; resolution grid (§6) — **disclosed limitation** |
| A14 | Semantics | `netproxy`/`device_fingerprint` are project proxies with unverified semantics; `card` is a token | Language rules (F4, §3) — **disclosed limitation** |
| A15 | Coverage | Infrastructure observable for a minority of transactions; coordination via other channels invisible | Observability flag, reporting of the unobservable population (§9) — **disclosed scope limit** |
| A16 | Feasibility | Null calibration and networkx Louvain runtime uncertain ≈5× | Timing-pilot gate, S7 stop-and-request (§18, §20) — **gated; approved R-E** |
| A17 | Infrastructure for segments | `compute_pseudo_account_segments` takes `HeteroData`; planted data needs an array-level version | Equivalence test is a **hard gate G0** on day-truncated TRAIN+VAL data (TEST file never opened) before any benchmark generation (§9, §11.7, S9) — **approved R-F and U2; hard implementation gate** |

**Review decisions (all APPROVED on 2026-10-07; execution still not authorized):**
* **R-A — APPROVED:** unit of coordination = card token; within-card multi-pseudo-account linkage excluded from
  coordination evidence; pseudo-account descriptive/contextual only (§4).
* **R-B — APPROVED:** H5a is the only primary criterion; H5b (transaction-level risk) and H5c (learned temporal entity
  state) are secondary and cannot substitute for H5a (§1, §15).
* **R-C — APPROVED, unchanged:** real IEEE-CIS cannot validate H5 quantitatively (no ground-truth coordination label)
  (§10).
* **R-D — APPROVED:** synthetic success = recoverability of the tested planted patterns under the stated assumptions; never
  proof that IEEE-CIS contains real fraud rings (§11, §15).
* **R-E — APPROVED:** timing-pilot gate; median > 60 s/window or projected CPU > 24 h → STOP and request a rescoping
  decision; Louvain not silently replaced (§6, §18, §20 S7).
* **R-F — APPROVED, upgraded to a HARD implementation gate G0:** array-level segmentation must match the existing
  `compute_pseudo_account_segments` exactly on day-truncated real data (days ≤ 151; see U2) before any benchmark
  generation (§9, §11.7, §20 S9).

**Final decisions U1–U4 (all resolved on 2026-10-07; execution still not authorized):**
* **U1 — APPROVED WITH CHANGE:** E1 uses the **card token** as its clustering/evaluation unit; pseudo-account is not used for
  E1 clustering. Project-wide rule: card token = coordination unit = E1 evaluation unit; pseudo-account =
  descriptive/contextual only (§2 F2, §4, §10, §16, A2).
* **U2 — CHANGED:** G0 operates on explicitly day-truncated TRAIN+VAL data (days ≤ 151) read from the TRAIN and VAL
  feature files only; the TEST file is never opened and the shared graph file is never loaded for G0; all
  exact-match, future-independence, toy-case, and gate-record requirements retained, plus a tie to the saved Phase 7 VAL
  segment labels (§5, §9, §11.7, S1, S9).
* **U3 — APPROVED, FROZEN EXACTLY:** all numerical defaults frozen in §15.1; not to be tuned or altered after seeing
  evaluation results.
* **U4 — APPROVED:** no fallback is pre-authorized; if the timing gate trips → STOP and request a rescoping decision; no
  silent algorithm change, grid reduction, null-calibration reduction, or hub cap (§6, §18, S7).

**Unresolved design decisions: none.**

## 23. Scope boundary

Phase 9 only. It trains no model, modifies no Phase 1–8 artifact, builds no dashboard, adds no explainability
suite, no production-inference path, no adaptive-update or drift component, and designs nothing for Phase 10+. The
Phase 8 checkpoints are read-only inputs. Execution requires a separate explicit authorization (not yet given). The
design review of 2026-10-07 authorised **no** implementation, benchmark generation, Louvain run, timing experiment, or
installation.

## 24. Review log

**2026-10-07 — design-level clarifications/corrections (design text only; no code, data, model, or artifact touched).**

| Item | Decision | Sections changed |
|---|---|---|
| R-A | Card token = unit of coordination; within-card multi-pseudo-account linkage excluded from evidence; pseudo-account descriptive/contextual only | §4 (unit paragraph rewritten); §22 |
| R-B | H5a sole primary; H5b/H5c secondary, non-substitutable | Header; §1 (table labels + new paragraph); §15 (labels + non-substitution bullet + overall wording); §22 |
| R-C | Real-data limitation kept exactly | §10 (answer sentence restated); §22 |
| R-D | Synthetic success = recoverability of tested planted patterns under stated assumptions; never proof of real fraud rings | §11 (claim limit rewritten); §15 (overall wording); §22 |
| R-E | Timing gate: > 60 s/window or > 24 h projected → STOP and request rescoping; Louvain not silently replaced; hub cap no longer automatic | §6 step 3; §18 (gate bullet); §20 (S7); §22 |
| R-F | Array-level segmentation equivalence = hard gate G0 with exact-match criterion, future-independence check, toy cases, gate record, refusal in generators | §9 (bullet rewritten); §11.7 (G0 added); §20 (S9 added); §21 (tests, artifacts); §22 (A17) |
| Terminology | Synthetic "ring" → "coordinated group"/"decoy group" (names, parameters, counts, plots, identifiers); prohibited-term list extended; lint covers code identifiers and manifests with two whitelisted disclaimers | §1 (one phrase); §10 (Tier-2 row); §11 (naming rule; P1–P3 names; §11.3/11.5/11.6); §12; §17; §21; §22 |
| Benchmark | Preserved; camouflage, decoys, popular-hub cases, partial observability, dev/eval separation declared non-removable/non-weakenable | §11 (preservation paragraph) |

Unchanged by this review: every methodological element not listed above (candidate generation, scoring components, null
models, metrics, baselines, ablations, leakage controls, populations, reproducibility, artifacts other than the two
additions noted).

**2026-10-07 (second review) — final decisions U1–U4 (design text only; no code, data, model, or artifact touched).**

| Item | Decision | Sections changed |
|---|---|---|
| U1 | **Approved with change:** E1 clustering/evaluation unit = card token (not pseudo-account); project-wide rule card token = coordination unit = E1 unit; pseudo-account descriptive/contextual only | Header; §2 (F2); §4 (project-wide rule paragraph); §10 (Tier-2 row: card-token statistic and permutation null); §16 (case-study column); §22 (A2, final decisions) |
| U2 | **Changed:** G0 on explicitly day-truncated TRAIN+VAL data (≤ 151) from the TRAIN and VAL feature files only; TEST file never opened; shared graph file never loaded for G0; reference rebuilt with the existing unmodified functions; tie to saved Phase 7 VAL labels; exact-match, future-independence, same-day-semantics, toy-case, gate-record requirements retained; wider TEST convention stated explicitly | §2 (F12); §5 (TEST row + disclosed-convention paragraph); §9 (G0 rewritten, 8 criteria); §11.7 (G0); §20 (S1, S9); §22 (A17) |
| U3 | **Approved, frozen exactly:** all numerical defaults in a frozen registry; clarified that γ is selected per *W* on dev and primary *W* = 7 | §11.6 (Purpose cell); §15.1 (new registry); §22 |
| U4 | **Approved:** no fallback pre-authorized; STOP and request rescoping; no silent algorithm change, grid or null-calibration reduction, or hub cap | §18 (gate bullet); §20 (S7); §22 |

After this review: **zero unresolved design decisions.** Execution is still not authorized.

---

## Appendix A — Files and artifacts inspected for this design (read-only)

`configs/graph.yaml`, `configs/data.yaml`; `src/graph/{schema,build,temporal,rolling_eval,temporal_checks}.py`;
`src/models/{temporal_graphsage,temporal_meanmem,metrics}.py`; `src/adaptive/{windows,replay,hparams,update_loop,
drift,leakage}.py`; `scripts/{train_temporal_graphsage,train_phase8_*,validate_phase8_leakage,validate_temporal_ablation,
run_tests}.py`; `tests/test_phase8_adaptive.py`; `reports/phase{1,4,7,7a,8}_*.md` (Phase 1 §10, §17, risks R2/R3; Phase 4
§B–§J; Phase 7A conclusions; Phase 8 §G, §J); `experiments/phase8_adaptive_drift/{shared_init/*,comparison.json}`; Phase
1–8 experiment directory structure; environment import check for networkx/scipy/igraph/leidenalg (nothing installed).

## Appendix B — Design-stage structural profile (label-free, TRAIN days ≤ 121 only)

Executed once inline; script saved for provenance as `design_recon_trainonly_labelfree.py`, output
`design_recon_trainonly_labelfree.json`. Reads graph structure only; **no labels, no model, no VAL/TEST rows.**
Key results: active entities (pseudo_account 152,592 / card 12,275 / netproxy 2,168 / device 3,736); share of netproxy
entities with ≥3 distinct cards 61.1% (≥2: 74.3%), of device entities 37.6% (≥2: 52.0%); entity co-occurrence graph:
170,733 active nodes, 4,694 components, giant component 152,454 nodes (89.3%); after removing entities with >200
transactions (471 removed): giant 77.5%; transactions by number of non-card entities carried: 0 → 5,686; 1 → 346,342;
2 → 9,752; 3 → 55,779. Used only for §18 feasibility and the F6/F7 facts; no threshold in this design was tuned to
these numbers beyond the order-of-magnitude estimates stated.
