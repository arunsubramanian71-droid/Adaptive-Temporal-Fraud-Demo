# Phase 10 — Design addendum: approved D12 decisions and consolidated event-typing specification

**Status: DESIGN DOCUMENTATION ONLY. PHASE 10 IMPLEMENTATION REMAINS UNAUTHORIZED.** Dated 2026-10-18. Nothing in this addendum has been implemented, executed, tested or run. No Stage 0 re-run was performed, the sealed TEST period (days 152-182) was not accessed, and no Phase 1-9 file or artifact was modified. Phases 11-13 are not started.

Wording rule (unchanged from `DESIGN_BEFORE_EXECUTION.md`): the objects are **candidate communities**. Nothing here is, or is claimed to be, confirmed fraud or proof of coordination. Candidate coordinated community; not a confirmed fraud ring. No confirmed fraud-ring labels exist in this dataset.

Phase 9 outcome carried forward and unchanged: H5a, H5b and H5c are NOT ESTABLISHED under the original protocol; original G4 failed on seeds 42 and 2024 and passed on seed 123; Tier 0 and E0 were exploratory and descriptive only. No Phase 9 threshold, result, report, addendum, manifest or integrity record is changed by this addendum.

## 0. Relationship to the existing Phase 10 documents (historical evidence is preserved)

| File | Role | Status |
|---|---|---|
| `DESIGN_BEFORE_EXECUTION.md` (+ `.sha256`) | Design draft of 2026-10-14, sections 1-15, unresolved decisions D1-D12 | **Unchanged.** Kept as historical review evidence. Its D12 wording (ordered passes, extended continuation rule, `theta_cov`, a single gap parameter `g`) is **superseded** by section 3 of this addendum where they differ. |
| `INSPECTION_REPORT.md` (+ `.sha256`) | Read-only inspection of Phase 1-9 sources | **Unchanged.** |
| `INTEGRITY.json` (+ `INTEGRITY.sha256`) | Creation-time hashes of the two documents and of 53 inspected sources | **Unchanged.** Re-verified by `INTEGRITY_ADDENDUM.json`. |
| `DESIGN_ADDENDUM_D12_approved_decisions.md` (this file, + `.sha256`) | Final approved D12 decisions and specification | **New.** |
| `INTEGRITY_ADDENDUM.json` (+ `INTEGRITY_ADDENDUM.sha256`) | Hashes of this addendum, re-verification of the original records and of the Phase 9 closure record | **New.** |

Where this addendum and the 2026-10-14 draft disagree on D12, **this addendum governs**. Where they agree or the draft is silent, the draft stands. The draft's decisions D1, D3-D11 are **not settled here** (section 10).

Supersession of draft D12 / D2 wording (nothing in a frozen artifact is affected; the draft is a Phase 10 document):

| Draft wording | Approved replacement |
|---|---|
| D12 "extended continuation rule" beyond a factor of two | No extended continuation. Size-changing relations are recorded as **relation edges** inside event components; identity is assigned only on identity edges. Extended policy: `break_with_link` (3.6). |
| D2 constants `theta_cov` | Dropped as a threshold. Coverage is stored as descriptors (3.4). |
| D2 single gap `g` (precedent: Phase 9 disappearance 3 days) | Two parameters `g_r` (re-link gap) and `g_d` (declaration delay), with primary design `g_d = g_r + 1` (3.8). Phase 9 corresponds to the compatibility setting `(g_r, g_d) = (0, 3)` (7). |
| D2 `w` (arrival window, precedent 3) | `w` is a labelling parameter only (3.11). |

## 1. Decision register (approved)

| # | Decision | Where specified |
|---|---|---|
| 1 | Extended policy: `break_with_link` | 3.6 |
| 2 | Primary tie key: Jaccard, then overlap, then deterministic IDs; compatibility mode uses the legacy Phase 9 tie key | 3.5, 7 |
| 3 | Re-link rule **R-A**: a dormant lineage may re-link only to a community with no relation edge to an active lineage | 3.7 |
| 4 | Separate gap parameters `g_r` and `g_d`; primary design `g_d = g_r + 1`; **no numerical `g_r` is chosen** | 3.8, 10 |
| 5 | Satellite reporting **S1**: all qualifying partition relations reported with size, share and coverage descriptors; no arbitrary importance cutoff | 3.10 |
| 6 | Newcomer majority is strict: `2 * n_new > community_size` | 3.11 |
| 7 | Compatibility mode documents and reproduces the legacy Phase 9 flags and lifecycle descriptors without modifying Phase 9 | 7 |
| 8 | Final accounting definitions and equations: `V_pre = A + D`, `N = s + r + b`, `A = s + e + z`, `V_post = V_pre + b - e - d`; every term defined; disappearances declared today distinguished from those declared earlier | 4 |
| 9 | Canonical event-record schema, invariants I1-I9, agreed toy specifications and corrections including T30b, T33', T23' | 5, 6, 8 |
| 10 | Corrected lifecycle distinction between compatibility mode and event-typing mode; explicit statement that Phase 9's three-day delay is a lifecycle reporting rule and its code does not literally test three consecutive days of absence | 7.3 |
| 11 | Conservative conclusions: communities are candidates, not confirmed fraud rings; frozen Phase 9 results and thresholds unchanged | 11 |

These choices are structural design rules. **None of them is an empirical finding**, and none fixes a numerical threshold (section 10).

## 2. Evidence base used (read-only; nothing was re-run)

* Phase 9 tracker `src/coordination/tracking.py`: a community pairs with a previous-day track only if Jaccard J >= 0.5 (inclusive); candidate pairs are sorted by `(-J, parent track id, community list index, previous-list index)`; assignment is greedy one-to-one; an unselected track never matches again (no re-link; the previous-day list is rebuilt from current-day tracks only); a `("merged", t)` flag is appended per unselected pair; for an unassigned community with any pair, the parent pointer is the parent of its first (highest-J) pair and a `("split", t)` flag is appended to that parent; `lifecycle_table` sets `ended = last_day < last_observed_day` and `disappearance_known_at = last_day + G_DISAPPEAR (3)` regardless of cause.
* Phase 9 Tier 0 script (`scripts/run_phase9_tier0.py`): "observable disappearance" = ended and `last_day <= T - 3`.
* Because pairs require J >= 0.5 and communities within a day are disjoint, the only split or merge a pair set can contain is an exact bisection-type configuration; observed zero splits/merges in Phase 9 is therefore structural, not a finding about the data.
* **Documented discrepancies in the frozen Phase 9 material (recorded, not repaired):** (a) the Phase 9 design text says ties go to the larger overlap, then the lower track id; the code breaks J-ties by lower parent track id, then list index, with no overlap term. (b) The Phase 9 design (section 8) describes disappearance as "no successor for 3 consecutive days"; the code does not test three consecutive days of absence (see 7.3). Neither discrepancy changes any recorded Phase 9 result, and neither requires a change to a frozen artifact. Compatibility mode reproduces the **code**.

No decision in this addendum conflicts with existing evidence, and none requires changing a frozen Phase 1-9 artifact.

## 3. Consolidated D12 specification

### 3.1 Input contract
Per-day list of **non-empty, finite** sets of card ids; sets within a day pairwise disjoint; duplicates collapsed (frozenset); days consecutive; an empty list for a day is allowed; optional `first_seen` map with no day greater than `t`. Any violation **raises an error**; there is no silent repair (T29, T35).

### 3.2 Arithmetic
All quantities are rationals `p/q` (`Fraction`), compared by integer cross-multiplication; **no floating point**. For a pair with overlap `x`, previous-size `a`, current-size `b`: `J = x/(a+b-x)`, `k_f = x/a`, `k_b = x/b`, `r = b/a`. Denominators are never zero for non-empty sets. All comparisons are inclusive: `J >= theta_id` (as in Phase 9), `k >= theta_ev`; growth iff `b*q >= a*(q+p)` for `delta = p/q`; contraction iff `b*(q+p) <= a*q`. The **single strict exception** is the newcomer majority `2*n_new > size` (3.11).

### 3.3 Pair flags and edge sets
Three flags per pair: **ID** (`J >= theta_id`), **F** (`k_f >= theta_ev`), **D** (`k_b >= theta_ev`).
* **Q** = qualifying relation edges: any flag set; active lineage x day-`t` community. In event-typing mode ID-flagged edges are a subset of Q.
* **S** = selected identity edges: ID-flagged edges chosen by 3.5 (under `break_with_link`, only ID-flagged edges can be selected).
* **Rl** = re-link edges: dormant lineage x community with `J >= theta_id`; never in Q.
* Event kind belongs to the **component**, not to the pair; flags and selection are separate record fields.

### 3.4 Components
Components are the connected components of the bipartite graph of Q between active lineages (seen on day `t-1`) and day-`t` communities. Shapes: `one_to_one` (continue or succession), `one_to_many` (split), `many_to_one` (merge), `many_to_many` (reorganize); isolated lineage `(1,0)`; isolated community `(0,1)`. `theta_cov` is not a threshold; coverage is stored as descriptors `sum(x)/a` and `sum(x)/b`. Every relation edge lies in exactly one component with exactly one event kind (T36).

### 3.5 Identity assignment and its guarantees
Global greedy over ID-flagged edges with key `kappa = (-J exact, -x, lineage id, canonical rank)`, i.e. **primary tie key: Jaccard, then overlap, then deterministic IDs**. Canonical rank = order by smallest card id; node id = `(day, smallest card id)`.

*Guaranteed (proof sketches recorded in the review; to be re-stated as tests):* deterministic and independent of list order; the selection is a maximal matching in the identity edges; it is blocking-free under the shared total-order preferences given by `kappa`, and it is the unique such matching (existence by a scan argument; uniqueness because the globally best remaining edge must belong to any blocking-free matching, else it blocks, then induction on the reduced instance); it decomposes over components; it concerns stage 1 only.

*Not guaranteed:* maximum weight or maximum cardinality (see T22); stability across stages; correctness of identity as truth; robustness to Louvain variability. Non-uniqueness arises for non-strict orders and for side-specific cyclic preferences, neither of which `kappa` produces.

The legacy Phase 9 key `(-J, parent/lineage id, community rank/list index)` applies **only in compatibility mode**. The two keys differ only when two edges tie on J with unequal overlap (T30b).

### 3.6 `break_with_link` (extended policy)
A `one_to_one` component whose only edge is relation-only (no ID flag) **ends the lineage as `succession`** and the community is **born** (`born_succession`) with a relation parent; the lineage does not continue and nothing is selected as an extended identity. Compatibility mode builds no relation edges (Q := empty); identity edges are still evaluated and S is a subset of the identity edges.

### 3.7 Re-link rule R-A
A dormant lineage may re-link only to a community that has **no relation edge to an active lineage** (singleton / no-Q-edge communities), by one-to-one greedy over `J >= theta_id` with the gap in the key; active lineages have priority. R-B (allowing a dormant lineage to re-link to a community that has relation edges) was examined and **not adopted**; its only visible effect is in T33'. Consequently, under R-A the dual-role fields of the schema (5) are forced empty.

### 3.8 Gap parameters and daily order
* `g_r` = maximum number of consecutive absent days across which a re-link is allowed. A lineage with last day `L` is eligible on day `t` iff `2 <= t - L <= g_r + 1`.
* `g_d >= g_r + 1` = declaration delay: a lineage whose last day is `L` is declared `disappeared` at the **end of day `L + g_d`**, after that day's re-linking. Primary design: `g_d = g_r + 1`. `g_d = 1` means no dormancy.
* Compatibility setting `(g_r, g_d) = (0, 3)`.
* **No numerical `g_r` is chosen here.** The values in the fixtures below (`g_r = 2`, `g_d = 3`) are test inputs, not selected constants, and are not justified by data.

Daily processing order: (1) validate input; (2) stage 1: identity assignment and components over active lineages; (3) stage 2: re-link of dormant lineages (R-A); (4) births numbered after re-links, in canonical node order; (5) end-of-day declarations (linking precedes declaration on day `L + g_d`); (6) status updates (active / dormant / closed).

Right-censoring at the end of day `T`: **active** if `L = T`; **undetermined** if `T - g_d < L < T`; **disappeared** if `L <= T - g_d`.

### 3.9 End reasons, origins, identity basis
* End reasons, by component kind and `s` (number of selected identity edges in the component): `succession` (`one_to_one`, `s = 0`); `dissolved` (`one_to_many`, `s = 0`); `merged` (`many_to_one`, unselected parents; target = the child's lineage); `reorganized` (`many_to_many`, unselected); `disappeared` (declared at `L + g_d`).
* Born origins: `pure`, `born_split_child`, `born_merge_product`, `born_reorganization_child`, `born_succession`.
* Identity basis values: `selected_identity`, `selected_extended` (defined only under a `continue` policy; **unused** here), `relinked`, `born`.

### 3.10 Satellite reporting S1
Report **all** qualifying partition relations with size, share and coverage descriptors; **no arbitrary importance cutoff**. Limits of S1 (must accompany any use): it describes partition-type relations determined by `theta_id`, `theta_ev` and the tie key, not substantive importance; satellites are reported next to 50/50 splits and only the descriptors distinguish them (T34); only active lineages are covered, so relation-only edges of dormant lineages are unreported (re-links are a separate record type); the unthresholded overlap store keeps all `x > 0` pairs; Louvain partitions are heuristic (no Leiden backend is installed), seed-dependent, and window overlap is mechanical; no claim of fraud or coordination is made.

### 3.11 Newcomer rule and `w`
A community is newcomer-dominated iff `2 * n_new > size`, where `n_new` counts cards first seen in the window `(t - w, t]`; a card first seen exactly at day `t - w` is not a newcomer (T12/T13). `w` only labels the birth subtype and does not enter identity assignment, components or accounting.

## 4. Lineage accounting (final)

At the start of day `t` each live lineage has `u = t - L` (days since last seen). Classes: **A** (`u = 1`, active); **D_r** (`2 <= u <= g_r + 1`, dormant and re-linkable); **D_c** (`g_r + 2 <= u <= g_d`, "closed": not re-linkable, not yet declared); **E_before** (ended by an event earlier); **X_before** (declared disappeared earlier). `D = D_r + D_c`; **`V_pre = A + D`** (live lineages at the start of day `t`).

During day `t`:
* `s` = active lineages with a selected identity edge;
* `e` = unselected active lineages with at least one relation edge; they end through an explicit event record;
* `z` = isolated active lineages with no relation edge; they are unseen today;
* `r` = dormant lineages re-linked today (`r <= |D_r|`);
* `b` = communities born today;
* `N` = number of communities today;
* `d` = lineages **declared disappeared at the end of day `t`**: alive, still unseen after re-linking, with `u = g_d`. They come from `z` only if `g_d = 1`, from `D_r` if `g_d = g_r + 1`, and from `D_c` if `g_d > g_r + 1`. Event-ended lineages are never declared. Lineages declared on earlier days are in `X_before` and are **not** counted in `d`.

Equations:
* `A = s + e + z`
* `N = s + r + b`
* `V_post = N + z + D - r - d = s + b + z + D - d = V_pre + b - e - d`
* total ever created `Lambda = V + E + X` increases by exactly `b` per day
* next-day classes: `A' = s + r + b`, `D' = z + D - r - d`

Per non-singleton component, births `= n - s` under R-A (`n - s - r_c` under R-B, not adopted); for singleton communities `N_0 = r + b_0`.

**Compatibility mode is the same equations with Q := empty:** `e = 0`, `z = A - s`, `r = 0`, `N = s + b`, `V_post = V_pre + b - d`, with `(g_r, g_d) = (0, 3)`.

*Verification status (paper only):* the equations were checked by hand on T22 day 2, T26 (`theta_ev = 2/3`, `break_with_link`) day 2, and T33' days 2-3 (under both R-A and R-B), plus a paper exercise of compatibility `(0, 3)` that exercises `D_c`. **Not verified:** later-day declarations and `D_c` in the reported event-typing cases; these are test obligations (section 8), not established facts.

## 5. Canonical event-record schema

Records are emitted in the order `(day, record type, primary id)`.

* **NodeRecord**: `node_id = (day, smallest card id)`; `size`; `identity {lineage_id, basis, edge_id, gap}`; `origin` (born nodes only); `component_ref`; `component_role` in {`selected_member`, `unselected_member`, `none`}; `relation_parents [{parent_node, parent_lineage_before, edge_id}]`; `identity_predecessor`; `size_class`; `headline_eligible`.
* **EdgeRecord**: `edge_id = (kind, parent_node, child_node)`; `kind` in {`relation`, `relink`}; `x, a, b`; `J, k_f, k_b` as exact fractions; `flags`; `selected`; `day` = the child node's day.
* **ComponentRecord**: `component_id = (day, smallest node id)`; `shape`; `lineages`; `nodes`; edge ids; selected edge ids; descriptors (`sum(x)/a`, `sum(x)/b`); counts `s`, `born_count`, `relinked_count`, ends; with `unselected_count = born_count + relinked_count` and `relinked_count = 0` under R-A.
* **LineageEnd**: `lineage_id`; `day` (event day; for `disappeared` the declaration day `L + g_d`); `reason`; `target_nodes`; `target_lineages`; `via_edges`; `available_at`.

The dual-role case (relation to an active lineage **and** a selected re-link identity) is possible only under R-B (not adopted); under R-A the dual-role fields are forced empty.

Temporal ordering and reporting use the **community-day graph** `G_cd` (acyclic by construction). The lineage-id graph `LG` is a quotient that may contain cycles (for example T20) and is never used for ordering.

## 6. Invariants (I1-I9)

* **I1** one identity per community-day; each lineage has at most one node per day.
* **I2** relation edges never imply continuation: under `break_with_link` no relation-only edge is selected.
* **I3** every re-link refers to an eligible dormant lineage (not ended, not declared, not active at `t-1`, `2 <= t - L <= g_r + 1`) and a permitted community (R-A: `component_ref` null), with `J >= theta_id` (exact) and one-to-one.
* **I4** every ending has an explicit reason and valid targets; ended lineage ids never reappear.
* **I5** output is deterministic and temporally ordered: sorted by `(day, record type, primary id)`; references point to the same or an earlier day; `available_at >= event day`.
* **I6** the balance equations of section 4 hold every day.
* **I7** every relation edge is in exactly one ComponentRecord.
* **I8** selected edges form a matching.
* **I9** every selected edge has the ID flag (the `selected_extended` exception applies only under a `continue` policy, which is not used).

## 7. Compatibility mode (reproduces frozen Phase 9 behaviour; Phase 9 is not modified)

### 7.1 Contract
For **canonical pipeline inputs only** (lists sorted by smallest card id, pairwise disjoint non-empty sets, consecutive days) the new module in compatibility mode must reproduce the Phase 9 `Tracker`: continuation assignments; new track ids sequential by list index; parent pointers (parent of the highest-J pair, ties to the lower track id); the legacy flags (`("merged", t)` appended per unselected pair; `("split", t)` appended to the parent of an unassigned community's first pair); and the `(day, size)` lists. The tie key is the legacy `(-J, parent/lineage id, community rank/list index)`.

### 7.2 Intentional differences and scope limits
The new module rejects skipped days, empty sets and overlapping sets; it has no list-order dependence. The equivalence is **not** claimed for non-canonical lists or for primary mode. It is establishable by seeded random canonical sequences against the Phase 9 `Tracker` imported read-only (T31). A real-data comparison would need Stage 0 card sets and is **not authorized**.

### 7.3 Corrected lifecycle distinction
> In compatibility mode (event typing off, Q := empty) every unselected active lineage, with or without an identity edge, is an isolated lineage (`z`) and is declared disappeared at the end of day `L + g_d`, reproducing Phase 9's `lifecycle_table` (`ended = last_day < T`; known at `last_day + 3`). In event-typing mode an unselected active lineage with at least one relation edge instead ends by event on day `t` with an explicit reason and target references and is never declared disappeared; only lineages without a relation edge, and dormant or closed lineages, are declared at `L + g_d`.

Explicit statement: **Phase 9's three-day delay is a lifecycle reporting rule.** `tracking.py` sets `disappearance_known_at = last_day + 3` for every ended track regardless of cause, and does not literally test three consecutive days of absence. This is consistent in effect because Phase 9 tracks never re-link. Compatibility mode reproduces this code behaviour and does not "correct" it.

## 8. Toy suite specification (hand-derived; test inputs, not chosen constants)

Fixture F: `theta_id = 1/2`, `theta_ev = 2/3`, `delta = 1/2`, `g_r = 2`, `g_d = 3`, `w = 3` unless a case overrides. `[a-b]` is a card range; `L0, L1, ...` are lineage ids in canonical birth order. Expected outcomes are stated under the final rules of section 3. Detailed, individually derived rows are marked **(derived)**; rows marked **(re-expressed)** keep the fixture inputs of the earlier agreed table and restate its expected outcomes in the final vocabulary, and must be re-derived when the tests are written (they are not new findings).

| ID | Input | Expected | Detects |
|---|---|---|---|
| T1 (re-expressed) | d1 [1-10]; d2 [1-10] | L0 selected, stable | identity churn |
| T2a / T2b (derived) | [1-10] -> [1-12] / [1-15] | selected, `r = 1.2` stable / `r = 1.5` exactly grow (`15*2 >= 10*3`) | growth boundary |
| T4a (derived) | [1-15] -> [1-10] | selected, `r = 2/3` exactly contract (`10*3 <= 15*2`) | contraction boundary |
| T3 / T4b (re-expressed) | [1-10] -> [1-25]; [1-20] -> [1-8] | `J = 2/5`, no ID flag; F (or D) flag set: `one_to_one` relation-only. `break_with_link`: L0 ends `succession`, successor born `born_succession` with relation parent L0. **Compat-only row:** no relation edge, L0 unselected isolated, dormant then declared at `L + g_d`, new birth | extended policy |
| T5 (re-expressed) | [1-10] -> [1-5], [6-10] | `J = 1/2` both (inclusive), tie on J and overlap, lower canonical rank: [1-5] continues L0; [6-10] has the ID flag, is unselected, `born_split_child` with relation parent L0 | tie determinism |
| T6, T7 (re-expressed) | [1-10] -> [1-7], [8-10]; [1-15] -> [1-8], [9-12], [13-15] | selected edge to the large part; remaining parts `born_split_child` (relation parent L0) | n-way split |
| T8, T9 (re-expressed) | d1 [1-7] = L0, [8-10] = L1, d2 [1-10]; d1 [1-8], [9-12], [13-15], d2 [1-15] | L0 selected; other lineages unselected members of a `many_to_one` component, `merged`, target = L0 | n-way merge |
| T10, T11, T14, T32 (re-expressed, explicit per parameters) | T10: present d1, d3. T11: present d1, d4 or d1, d5. T14: present d1-d2, absent d3-d5. T32: censoring at `T` | T10: `(g_r, g_d) = (2, 3)` or `(1, 2)`: re-link at d3; `g_d = 1`: disappeared at end d2, d3 birth. T11 `(2, 3)`: d4 re-link (`u = 3 <= g_r + 1`); with d5, L0 declared at end d4 (`L + g_d`), d5 birth; `(1, 2)`: declared end d3, d4 birth. T14 `(2, 3)`: dormant ends d3, d4; declared end d5 (`available_at = 5`); observation ending d4 gives undetermined. T32 `g_d = 3`: `L = T-3` disappeared, `T-2` undetermined, `T` active | gap boundaries; same-day re-link vs declaration; censoring |
| T12 / T13 (derived) | day 10, community of 8 cards, `w = 3`, first-seen days | 5 first seen in (7, 10]: newcomer-dominated; 4: regrouped (strict `2*n_new > size`); first_seen = 7 is not a newcomer | strict majority; causality |
| T15 (derived) | A = [1-10] = L0, B = [11-20] = L1 -> C1 = [1-5, 11-15], C2 = [6-10, 16-20] | `theta_ev = 1/2`: K2,2 `many_to_many` reorganization; L0, L1 `reorganized`; L2, L3 `born_reorganization_child`. `theta_ev = 2/3`: no edges, L2, L3 pure births, L0, L1 dormant | crossing; threshold sensitivity |
| T16 (re-expressed) | [1-10] -> [3-12] | selected, stable; retained 8, joined {11, 12}, left {1, 2} | swap read as growth |
| T17, T18/19 (re-expressed) | empty days; permuted inputs; repeat runs | no error, ids stable; identical records and ids | empty-day failure; order dependence |
| T20 (re-expressed) | [1-10]; [1-5], [6-10]; [1-10] | d2 as T5. d3: `J = 1/2` both, tie, overlap equal, lower lineage id: L0 selected; L1 unselected, `merged`, target L0. `G_cd` acyclic; `LG` has the cycle L0 -> L1 -> L0 | acyclicity; ordering by `G_cd` |
| T21 (re-expressed) | `theta_id = 1/3`: L0 = [1-6], L1 = [7-12] -> C = [1-4, 7-10] | `J = 2/5` both, tie, L0 selected; `k_f = 2/3` exactly for both: L1 `merged`, target L0 | tie plus equality boundary |
| T22 (derived) | `theta_id = 1/5`: L0 = [1-10], L1 = [11-20] -> Ca = [1-6, 11-14], Cb = [7-10, 21-26] | `J` values 3/7, 1/4, 1/4; selected (L0, Ca); L1 `reorganized`; Cb = L2 born; a maximum-weight alternative has weight 1/2 vs 3/7 and is unstable; which is "true" is unresolved | greedy is not maximum-weight |
| T23' (derived) | A = [1-9, 21] = L0; D = [4-10, 31, 32, 33]; C = [1-10] | `J(D, C) = 7/13`, `J(A, C) = 9/11`; C continues L1 by active priority; valid for `theta_id` in (3/7, 7/13] and `theta_ev` in (3/5, 1]. **Does not distinguish R-A from R-B** | active priority |
| T24 (re-expressed) | X = [1-40], Y = [41-80] -> Z = [1-80] | `J = 1/2` both, tie, L0 selected; L1 `merged`, target L0; growth `r = 2` | equal-halves merge |
| T25 (re-expressed) | G = [1-60] -> [1-20], [21-40], [41-60] | no ID flag, `k_b = 1` each: `one_to_many`, `s = 0`: L0 `dissolved`; three `born_split_child` with relation parent L0 | hub-removal fragmentation |
| T26 (derived) | P = [1-10] -> A = [1-6, 11-40], B = [7-10] | `theta_ev = 1/2`: `one_to_many` without carrier. `theta_ev = 2/3` with `break_with_link`: only (P, B) qualifies; L0 ends `succession`, B born `born_succession` (L2), A pure birth (L1) | threshold sensitivity |
| T27 (re-expressed) | P = [1-12] -> A = [1-8, 21-40]; variant `x = 7` | `x = 8`: `k_f = 2/3` exactly, relation-only edge, `break_with_link` succession; `x = 7`: no edge, A pure birth | rational equality boundary |
| T28 (re-expressed) | `theta_id = 1/3`: `J = 2/6` matches, `2/7` not; `delta = 1/3`: 9 -> 12 grow, 9 -> 11 stable, 12 -> 9 contract, 12 -> 10 stable | float-free comparisons | arithmetic |
| T29 (derived) | invalid inputs: empty set; overlapping same-day sets; `first_seen > t`; non-consecutive day | error raised, no state change | silent repair |
| T30 (derived) | `theta_id = 1/3`: P = [1-10] -> A = [1-6, 11-16], B = [7-10] | `J(A) = 3/8`, `J(B) = 2/5`: Jaccard-first picks B | tie-key order |
| T30b (derived) | `theta_id = 1/3`, `theta_ev = 2/3`: L0 = [1-5]; C2 = {1, 2, 11}; C1 = {3, 4, 5, 12, 13, 14, 15} | `J` tie at 1/3; `(J, x, IDs)` selects C1, `(J, IDs)` selects C2; documents that the primary key `(J, overlap, IDs)` and the legacy key differ exactly here | overlap term |
| T31 (derived) | seeded random canonical sequences | compatibility mode equals the Phase 9 `Tracker` (ids, parents, flags, `(day, size)` lists) | compat claim |
| T33' (derived) | X = [6-10]; P = [1-5, 8, 9, 10]; day 3: A = [1-5], B = [8, 9, 10] | R-A: B = L2 `born_split_child`, L0 stays dormant. R-B: L0 re-links B. Valid for `theta_id` in (3/8, 3/5], `theta_ev` in (3/5, 1], `g_r >= 1`. The original T33 with `k_b` exactly 2/3 was fragile and is replaced | R-A vs R-B |
| T34 (re-expressed) | P (100 cards) -> a large carrier plus a 3-member spin-off whose cards all come from P | the spin-off is reported as a partition relation with size, share and coverage descriptors under S1; no importance cutoff, and no claim about its substantive importance | satellite label inflation |
| T35 (re-expressed) | skipped day, empty set, overlap | rejected by the new module; Phase 9 behaviour documented | compat scope |
| T36 (re-expressed) | property: every pair is in exactly one component with one event kind | holds | component partition |

Property tests: no blocking pair in the identity matching; conservation (`sum_j x_ij <= a_i`, `sum_i x_ij <= b_j`); components partition all lineages and communities; each lineage at most once per day and ids monotone with birth; a topological order exists on `G_cd`; all record references resolve; the balance equations hold every day.

## 9. What is and is not established

* Established (on paper): the logical consistency of the rule set above, as checked in the review rounds; the greedy guarantees stated in 3.5 under the stated preference system; the accounting equations on the cases named in section 4.
* **Not** established: any empirical behaviour on IEEE-CIS; that any threshold value is appropriate; that identity assigned by the greedy rule corresponds to a real continuing entity (T22 shows the truth is unresolved in constructed cases); the compatibility equivalence on real data; anything about fraud or coordination.

## 10. Still open / judgment items (not decided by this addendum)

* **Numerical values** of `theta_id`, `theta_ev`, `delta`, `w`, `g_r` (and hence `g_d`). Only `theta_id = 1/2` and the Phase 9 setting `(0, 3)` have precedent. Any chosen value must carry a documented justification and be labelled a judgment, not an empirical result.
* Draft decisions D1 (period and VAL reuse), D3 (numeric floors and reproducibility bands), D4 (population), D5 (null key space), D6 (hub-cap sensitivity), D7 (scope, cap and cost of the synthetic benchmark, including the 2% planted share and recoverability fraction), D8 (investigator usefulness, unclaimed), D9 (label-based stage, none proposed), D10 (new module; recommended), D11 (Stage 0 re-run, not authorized) remain **as stated in `DESIGN_BEFORE_EXECUTION.md` section 15**.
* Family-persistence reporting.
* Which of two symmetric or competing identities is "true" in constructed cases (T5, T15, T22): a convention, not a discovery.
* The null model (N1, coherent degree-preserving history) is validated only for the default day range 116-151, mostly seed 42, per the inspection report.

## 11. Conservative conclusions

* Communities in this phase are **candidates**. They are not confirmed fraud rings, and lineage continuity, splits, merges and disappearances are descriptions of window-to-window overlap between heuristic Louvain partitions, not evidence of coordination.
* Fraud labels, risk scores and model predictions must never generate or select Phase 10 communities. Strict temporal causality applies; TEST (days 152-182) stays sealed.
* H5a, H5b and H5c remain NOT ESTABLISHED under the original protocol; Tier 0 and E0 remain exploratory only. Frozen Phase 9 results and thresholds are unchanged.
* This addendum records design decisions. It authorizes no implementation, code, test run, experiment, Stage 0 re-run or TEST access.

## 12. Integrity

See `INTEGRITY_ADDENDUM.json`. The sidecar of this file is `DESIGN_ADDENDUM_D12_approved_decisions.sha256` (the SHA-256 of the file's bytes, written as 64 hexadecimal characters). A document cannot contain its own hash; the hash is held only in the sidecar and in `INTEGRITY_ADDENDUM.json`.
