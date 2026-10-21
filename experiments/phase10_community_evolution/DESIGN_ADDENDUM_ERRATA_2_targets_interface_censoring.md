# Phase 10 — Documentation errata 2: ending targets, module interface, right-censoring

**Status: DOCUMENTATION ONLY. PHASE 10 IMPLEMENTATION REMAINS UNAUTHORIZED.** Dated 2026-10-18. Nothing here has been implemented, executed, tested or run. No Stage 0 re-run, no TEST access (days 152-182 sealed), no Phase 1-9 file or artifact modified, no Git operation performed. Phases 11-13 not started.

Wording rule (unchanged): the objects are **candidate communities**; none is, or is claimed to be, a confirmed fraud ring or proof of coordination. Frozen Phase 9 results and thresholds are unchanged; H5a, H5b and H5c remain NOT ESTABLISHED under the original protocol.

## 0. Scope and precedence

This amendment closes exactly three specification blockers (named E2-A, E2-B, E2-C) from the implementation-readiness specification. It reopens no settled decision. Precedence: **errata 2 > errata 1 > D12 addendum > 2026-10-14 draft**, only for the points stated here. All earlier Phase 10 files are preserved unchanged. Test obligations that follow from these rules stay in the implementation checklist (section 4); no toy row is re-derived here.

## E2-A. Ending targets for `LineageEnd`

**Rule.** For every lineage that ends by an event (`succession`, `dissolved`, `merged`, `reorganized`), the targets are exactly the day-`t` communities **directly adjacent to the ended lineage through its own relation edges**:
* `via_edges` = the ended lineage's own relation EdgeRecords (every Q edge whose parent is that lineage's last node), ordered by `edge_id`;
* `target_nodes` = the child node of each of those edges, in the same order;
* `target_lineages` = the lineage id of each of those nodes, in the same order (lineages born on day `t` included).
The three lists have equal length and correspond position by position. For `disappeared`, all three are empty.

**Consistency check against the existing schema and examples.**
* Field set unchanged: `target_nodes`, `target_lineages`, `via_edges` exist in addendum section 5 and errata 1 B3.
* `succession` (one child), `merged` (one child, one edge per ended lineage) and `dissolved` (a single parent adjacent to every child of its component) give the same lists as the errata 1 wording. T3, T4b, T8, T9, T20, T21, T24, T25 and T26 are unchanged. T15 (complete bipartite component) is unchanged.
* Only `reorganized` in a component that is **not** complete bipartite differs. Errata 1 B3 said "the component's child nodes"; that sentence is **replaced** by the adjacency rule. The earlier wording was inconsistent with its own `via_edges` field ("the children the ending maps to").
* T22 under this rule: `LineageEnd(L1, reorganized, day 2, target_nodes [n(2,1)], target_lineages [L0], via_edges [(relation, n(1,11), n(2,1))])`. The target lineage L0 is the lineage that continues in Ca; a target may be a continuing lineage.
No conflict with existing approved evidence was found.

## E2-B. Module interface (input contract)

The module is incremental, in the spirit of the Phase 9 `Tracker.step(t, card_sets)`, with the validation required by addendum 3.1.

**Call:** `step(day, communities, first_seen=None)`.
* `day`: an integer. The first accepted call fixes the start day (any integer). Every later call must have `day` equal to the previous accepted day plus 1. A day cannot be skipped; it must be supplied with an empty list.
* `communities`: a list or tuple; each element is a collection of integer card ids, collapsed to a frozenset (duplicates disappear). Order is irrelevant: the module sorts canonically by smallest card id, and output records are emitted in canonical order. **An empty day is represented by an empty list**; an empty community (empty set) is invalid.
* `first_seen`: `None`, or a mapping from integer card id to an integer day. It is supplied **per call** and is not cumulative. Every value must be `<= day`. Entries for cards that are in no community of this call are permitted and ignored, but their values are still validated. **Omission** is represented by `None` or by a missing key; per errata 1 B3 a community whose cards are not all present in the mapping has `birth_subtype = null`.
* Construction parameters (thresholds, `delta`, `g_r`, `g_d`, `w`, mode) are fixed when the module is constructed, have no defaults, and are not arguments of `step`.

**Validation.** All checks run **before any state change**. Any of the following raises a single documented exception, `InvalidInputError` (a subclass of `ValueError`): `day` not an integer or not equal to the previous accepted day plus 1; `communities` not a list or tuple; an empty community; a card id that is not an integer; two communities of the same call sharing a card; a malformed `first_seen` or any value greater than `day`. A rejected call leaves the module's state byte-for-byte unchanged, and the same day may be resubmitted. There is no silent repair.

**Return value.** `step` returns the records emitted for that day (NodeRecord, EdgeRecord, ComponentRecord, LineageEnd), ordered by invariant I5. Declarations that fall due at the end of day `t` are emitted by the step of day `t`.

**Consistency check.** Matches addendum 3.1 (consecutive days, empty list allowed, optional `first_seen`, no day greater than `t`, errors not repairs), T29 (empty set, overlap, `first_seen > t`, non-consecutive day, no state change), T12/T13 (`first_seen` passed with the day-10 call; days 1-9 may be empty lists) and Phase 9's `step(t, card_sets)` (any start day, explicit `t`). Phase 9 itself performs no validation (T35); that difference is intentional (addendum 7.2). No conflict.

## E2-C. Right-censoring: ended, declared, right-censored

**Horizon.** The observation horizon `T` is the day of the **most recent accepted `step`**. Horizons later than the supplied data are produced by stepping empty days; nothing is inferred about days not yet supplied.

**Definitions** (for a lineage whose last seen day is `L`, evaluated at `T`; `L` is updated by a re-link):
* **Ended by event**: a `LineageEnd` with reason `succession`, `dissolved`, `merged` or `reorganized` exists. The ending is determined on the event day (event-typing mode only).
* **Declared** (`disappeared`): not ended by event, not re-linked, and `L + g_d <= T`; the `LineageEnd(disappeared)` record has been emitted on day `L + g_d`.
* **Right-censored** (`undetermined`): not ended by event and `L < T` but `L + g_d > T`, i.e. `T - g_d < L < T`. Its fate is not yet known: it may still re-link (while eligible) or be declared later. This is **not** an event and no record is emitted.
* **Active**: `L = T`.

**Output.** A read-only query, `classification()`, evaluated at the horizon `T`. It returns one row per lineage ever created, with `lineage_id`, `last_day` (`L`), and `status` in {`active`, `undetermined`, `disappeared`, `ended_by_event`}; plus `declared_day = L + g_d` when `status = disappeared`, `known_at_if_unseen = L + g_d` when `status = undetermined`, and `end_reason` when `status = ended_by_event`. It never changes state and emits no records. Before any accepted `step` it returns an empty list. The statement of addendum 3.8 ("active if `L = T`; undetermined if `T - g_d < L < T`; disappeared if `L <= T - g_d`") applies to lineages **not ended by event**; event-ended lineages are never declared (addendum section 4) and take status `ended_by_event`.

**Invariant (for the checklist).** `status = disappeared` if and only if a `LineageEnd(disappeared)` for that lineage has been emitted.

**Legacy Phase 9 descriptors are unchanged.** In compatibility mode, `classification()` rows additionally carry the two legacy lifecycle fields with exactly the Phase 9 `lifecycle_table` semantics: `ended = (last_day < T)` with **no censoring adjustment**, and `disappearance_known_at = last_day + g_d` (that is, `+ 3`) **whenever `ended` is true, even when that value exceeds `T`**, otherwise `None`. The mapping to the three-way status is a derived view and does not alter the legacy fields: `active` iff not `ended`; `undetermined` iff `ended` and `known_at > T`; `disappeared` iff `ended` and `known_at <= T`. The Phase 9 Tier 0 "observable disappearance" (ended and `last_day <= T - 3`) coincides with `disappeared`. In event-typing mode the legacy fields are not reproduced (the three-way status is the only censoring output). The remaining legacy fields (`first_day`, `size_*`, `parent`, `flags`, `age_days`) and their container remain governed by the T31 comparison.

**Summary of the three notions.** *Ended* is Phase 9's descriptive `last_day < T`, or an explicit event ending. *Declared* means the delay `g_d` has elapsed within the observed data. *Right-censored* means ended in the legacy sense but not yet declarable at `T`. A lineage can be legacy-ended and right-censored at the same time.

**Consistency check.** Equivalent to addendum 3.8 and section 4 for non-event-ended lineages, to Phase 9 `lifecycle_table` (`ended`, `known_at = last_day + 3`) and to the Tier 0 rule. Clarification only: addendum 3.8 did not say that event-ended lineages are excluded; section 4 already did. No conflict.

## 3. Effect on earlier specifications (no rows re-derived)

* Errata 1 B3, LineageEnd paragraph: the sentence on `target_nodes` is replaced by E2-A; everything else in that paragraph stands.
* The container question for compat lifecycle descriptors (errata 1 B2.5) is settled for `ended` and `disappearance_known_at` by E2-C; the other legacy fields remain a T31 matter.
* T14, T32: `T` is the last stepped day; shorter or longer horizons are realized by truncating or appending empty days. T31: horizons beyond the data (`last day + 1 ... + 4`) are realized by stepping empty days on both sides (equivalent for Phase 9, whose `lifecycle_table` takes `T` as an argument).
* T22: the `LineageEnd(L1, ...)` record is as given in E2-A.
* T29: atomicity is asserted through `InvalidInputError` and an unchanged state; T12/T13 pass `first_seen` with the day-10 call.

## 4. Test obligations carried to the implementation checklist

1. Adjacency: for every event ending, `via_edges`, `target_nodes` and `target_lineages` have equal length, correspond position by position, and equal the ended lineage's own relation edges; T22 `reorganized` yields `[n(2,1)]`.
2. Interface validation: each rejection case raises `InvalidInputError` and leaves state unchanged; a rejected day can be resubmitted; start day is free; `None`/missing `first_seen` gives `birth_subtype = null`; empty list is accepted.
3. Classification: boundary lineages at `L = T`, `T - g_d + 1`, `T - g_d` for several `g_d`; `status = disappeared` iff a `LineageEnd(disappeared)` was emitted; event-ended lineages are `ended_by_event`; empty-query before any step.
4. Compat legacy fields: `ended` and `disappearance_known_at` equal Phase 9 `lifecycle_table` at every horizon of T31, including `known_at > T`; derived status matches the mapping above.
5. All other obligations of the implementation-readiness specification are unchanged.

## 5. Items not resolved here

None of the three blockers remains open. Unresolved items of errata 1 (`headline_eligible`, membership subtypes, `birth_subtype` for two origins, the draft's grid/selection rule/taxonomy, persistence of the overlap store) and the draft decisions D1, D3-D11 are unchanged.

## Integrity

See `INTEGRITY_ERRATA_2.json`. The sidecar of this file is `DESIGN_ADDENDUM_ERRATA_2_targets_interface_censoring.sha256` (SHA-256 of the file's bytes, 64 lowercase hexadecimal characters). A document cannot contain its own hash.
