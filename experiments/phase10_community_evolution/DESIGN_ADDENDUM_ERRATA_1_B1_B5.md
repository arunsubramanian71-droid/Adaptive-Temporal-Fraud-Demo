# Phase 10 — Documentation errata 1 to the D12 addendum (items B1-B5)

**Status: DOCUMENTATION ONLY. PHASE 10 IMPLEMENTATION REMAINS UNAUTHORIZED.** Dated 2026-10-18. Nothing here has been implemented, executed, tested or run. No Stage 0 re-run, no TEST access (days 152-182 sealed), no Phase 1-9 file or artifact modified, no Git operation performed. Phases 11-13 not started.

Wording rule (unchanged): the objects are **candidate communities**; none is, or is claimed to be, a confirmed fraud ring or proof of coordination. Frozen Phase 9 results and thresholds are unchanged; H5a, H5b and H5c remain NOT ESTABLISHED under the original protocol.

## 0. Scope and precedence

This errata resolves only the five specification gaps B1-B5 identified in the implementation-readiness review. It reopens **no** settled D12 decision (decisions 1-11 of `DESIGN_ADDENDUM_D12_approved_decisions.md` stand). It introduces no new research threshold and no new data requirement; each clarification below is the smallest reading consistent with the approved text, or removes/defers a field that the approved inputs cannot support.

Files left **unchanged and preserved as historical evidence**: `DESIGN_BEFORE_EXECUTION.md`, `INSPECTION_REPORT.md`, `INTEGRITY.json`, `DESIGN_ADDENDUM_D12_approved_decisions.md`, `INTEGRITY_ADDENDUM.json` and all their sidecars. Precedence where texts differ: **this errata > the D12 addendum > the 2026-10-14 draft**.

Approved inputs (unchanged, addendum 3.1): per-day lists of non-empty, finite, pairwise disjoint card-id sets on consecutive days; optional `first_seen` (card -> day, no day > `t`). Nothing else is available to the module (no transactions, no graph activity, no labels, no risk scores).

## B1. Re-link matching key (stage 2)

Re-link candidates are edges (dormant lineage `i`, community `j`) with: lineage eligible on day `t` (not ended, not declared, not active at `t-1`, `2 <= t - L_i <= g_r + 1`); community `j` has **no relation edge** to any active lineage (R-A) and is not otherwise assigned; and `J_ij >= theta_id` (exact, inclusive).

Define `gap_i = t - L_i - 1`, the number of consecutive absent days. Eligibility (`2 <= t - L_i <= g_r + 1`) gives `1 <= gap_i <= g_r`.

**Key:** `kappa_relink = (-J exact, -x, gap, lineage id, canonical rank)`, sorted ascending; global greedy one-to-one over all candidate edges in that order. This is the approved order Jaccard -> overlap -> deterministic IDs; the existing addendum requirement that "the gap is in the key" is met by placing the gap **after overlap and before the IDs**, i.e. as a tie-break that is consulted only when two candidate edges tie exactly on `J` and on `x` (a shorter absence wins). It never overrides Jaccard or overlap. Stage 1 (active lineages) is completed first, so active lineages keep priority (unchanged).

Guarantees: the key is a strict total order on edges (lineage ids and canonical ranks are distinct), so the stage-2 matching is deterministic, independent of list order, a maximal matching in the candidate edges, and blocking-free under the shared total order, by the same argument as addendum 3.5 restricted to stage 2. No maximum-weight or cross-stage stability claim is made.

Status: the position of `gap` in the key is a **tie-break convention**, not an empirical choice and not a discovered fact. It affects results only on exact `(J, x)` ties between dormant lineages with different gaps. Compatibility mode has no re-link (`g_r = 0`).

## B2. ComponentRecords for isolated nodes; scope of event records

1. **Isolated nodes do not receive ComponentRecords.** A ComponentRecord exists only for a connected component of the relation graph Q that contains **at least one relation edge**. The shapes `(1,0)` and `(0,1)` in addendum 3.4 are descriptive terms for isolated lineages and isolated communities; no record with those shapes is emitted. ComponentRecord `shape` takes only the four values `one_to_one`, `one_to_many`, `many_to_one`, `many_to_many`.
2. An isolated community (a pure birth, or a community re-linked under R-A) has `component_ref = null` and `component_role = none`. This keeps re-linked communities under R-A consistent with `component_ref = null` (I3).
3. An isolated active lineage (`z`) produces no node on the day it is unseen and no record until it is declared (LineageEnd, reason `disappeared`, at `L + g_d`).
4. `component_role` for a community node: `selected_member` = it has a selected identity edge inside a component; `unselected_member` = it is in a component with no selected edge to it; `none` = it is in no component.
5. **Compatibility mode (Q := empty) emits no ComponentRecords and no relation EdgeRecords.** Invariants I7 (every relation edge in exactly one ComponentRecord) and I3's `component_ref` clause concern event-typing mode. In compatibility mode the module reproduces the Phase 9 tracker outputs listed in addendum 7.1 (continuation assignments, track ids, parent pointers, legacy flags, `(day, size)` lists, and lifecycle descriptors); identity assignment appears in `NodeRecord.identity` with `edge_id = null`. Which container carries the compat lifecycle descriptors (`LineageEnd` records of reason `disappeared` or a lifecycle table) is an implementation detail, to be fixed by the T31 comparison fields; it is not a research decision.

## B3. Schema fields: definitions, removals and deferrals

Everything below uses only the approved inputs. Notation: for a pair, `x` = overlap, `a` = size of the lineage's last node, `b` = size of the day-`t` community; `J = x/(a+b-x)`, `k_f = x/a`, `k_b = x/b` (exact fractions, addendum 3.2).

**NodeRecord** (one per community-day):
* `node_id = (day, smallest card id)`; `size = b`.
* `identity.lineage_id`; `identity.basis` in {`selected_identity`, `relinked`, `born`}. The value `selected_extended` is **reserved and never emitted** (extended continuation is not adopted).
* `identity.edge_id` = id of the selecting EdgeRecord (the selected relation edge for `selected_identity`, the re-link edge for `relinked`); `null` for `born` and for all compat-mode nodes.
* `identity.gap` = `0` for `selected_identity`; `t - L - 1` (= `gap_i` of B1) for `relinked`; `null` for `born`.
* `origin` = one of `pure`, `born_split_child`, `born_merge_product`, `born_reorganization_child`, `born_succession` for `basis = born`; `null` otherwise.
* `identity_predecessor` = the lineage's last node (the day `t-1` node for `selected_identity`; the last node before the gap for `relinked`); `null` for `born`.
* `relation_parents` = one entry `{parent_node, parent_lineage_before, edge_id}` for **every** relation edge in Q whose child is this node (including the selected one), ordered by `edge_id`; empty for isolated communities.
* `component_ref`, `component_role`: as in B2.
* `size_class`: defined only for `basis` in {`selected_identity`, `relinked`}, comparing `b` with the size `a` of `identity_predecessor`: `grow` iff `b*q >= a*(q+p)`; `contract` iff `b*(q+p) <= a*q`; otherwise `stable` (`delta = p/q`, addendum 3.2; the two cases are mutually exclusive for `delta > 0`). `null` for `born`.
* `birth_subtype` (**new field name for the existing `w`-labelling rule**, addendum 3.11): in {`newcomer_dominated`, `regrouped`, `null`}. It is defined only for `origin = pure` and only if every card of the community has a `first_seen` entry. `n_new` = number of cards with `t - w < first_seen <= t`; `newcomer_dominated` iff `2*n_new > size` (strict), else `regrouped`. For any other origin, or when any card lacks a `first_seen` entry, it is `null`. The draft (section 3, "Appearance") applies the subtype to births other than split children and merge products; the origins `born_succession` and `born_reorganization_child` did not exist in the draft and are **left unsubtyped (null)**; extending the subtype to them is an unresolved later-tier question.
* **`headline_eligible` is removed from the required output schema.** It cannot be defined from the approved inputs: the draft's headline summaries depend on size strata, hub share and alert status (draft sections 7-8; D3, D4, D6), none of which is an input of the module. It is marked an **unresolved later-tier field**; no implementation or test may require it.

**EdgeRecord**: `edge_id = (kind, parent_node, child_node)`; `kind` in {`relation`, `relink`}; `x, a, b`; `J, k_f, k_b` as exact fractions; `flags` = the three booleans ID (`J >= theta_id`), F (`k_f >= theta_ev`), D (`k_b >= theta_ev`) computed for every emitted edge; `selected`; `day` = the child node's day. `relation` EdgeRecords are emitted for **every** edge in Q (selected or not). `relink` EdgeRecords are emitted **only for selected re-links** (losing candidates are not emitted; their overlaps remain recoverable from the inputs). The unthresholded overlap store of the draft (section 3) is not a field of the event-record schema; whether it is persisted as a separate artifact is unresolved (later tier).

**ComponentRecord**: `component_id = (day, smallest child node id)`; `shape` (B2); `lineages` = parent lineage ids; `nodes` = child node ids; `edge_ids`; `selected_edge_ids`; counts `s` (selected edges), `born_count` (child nodes with a `born_*` origin), `relinked_count` (`0` under R-A), `ends` (number of lineages in the component that end by event); `unselected_count = born_count + relinked_count`; and the exact descriptors below.

**Coverage and share formulas (S1).** For a component `K` with parent lineages `P_K` (sizes `a_i`), child nodes `C_K` (sizes `b_j`) and relation edges `E_K` with overlaps `x_ij` (only edges in Q are counted):
* `parent_coverage(i) = ( sum over (i,j) in E_K of x_ij ) / a_i`  (fraction of parent `i` found in its component's children through qualifying edges);
* `child_coverage(j) = ( sum over (i,j) in E_K of x_ij ) / b_j`  (fraction of child `j` drawn from its component's parents through qualifying edges);
* `component_parent_coverage = ( sum over E_K of x_ij ) / ( sum over P_K of a_i )` and `component_child_coverage = ( sum over E_K of x_ij ) / ( sum over C_K of b_j )`;
* `share(i -> j) = x_ij / a_i` (equal to `k_f`: the portion of parent `i` that child `j` holds) and `child_fraction(i -> j) = x_ij / b_j` (equal to `k_b`).
All are exact fractions with denominators `>= 1`. Because same-day sets are disjoint, `sum_j x_ij <= a_i` and `sum_i x_ij <= b_j`, so every coverage lies in `[0, 1]`. Coverage counts only qualifying edges, so it is a lower bound on the true overlap coverage. The S1 satellite report for a child node lists, for each of its relation parents: the node `size`, `share`, `child_fraction` and the component's `parent_coverage` of that parent; there is no importance cutoff and the S1 limits of addendum 3.10 apply unchanged.

**LineageEnd**: `lineage_id`; `day` (event day; for `disappeared`, the declaration day `L + g_d`); `reason` (addendum 3.9); `target_nodes` = child nodes of the ending component that the ending maps to (`succession`: the successor; `dissolved`, `reorganized`: the component's child nodes; `merged`: the child node of the merge); empty for `disappeared`; `target_lineages` = lineage ids of those nodes (newly born ids included); `via_edges` = relation edge ids; `available_at = day`.

## B4. Membership-change scope

Using only the approved inputs, the module's required output describes membership change **by counts**, all derivable from each EdgeRecord: `retained = x`, `joined = b - x`, `left = a - x`, and `churn = 1 - J`. Member sets are fixtures' inputs and are not required output fields (toy rows assert counts, not set contents).

The draft's membership-change **subtypes** (draft section 3, "Membership change") are **not** in the required output and are **deferred to a later descriptive tier** (unresolved):
* *re-entry*, *expired* and *isolated*, and the residual *other*, require graph activity or transaction data (activity "in the graph" but in no eligible community; infrastructure-linked transactions in the new window) that the input contract does not supply; they cannot be established and are **not claimed**.
* *transfer* and the joined-card *newcomer* subtype could in principle be computed from the card-set lists and `first_seen`, but they are not required here; adopting them is a later-tier decision and does not block implementation.
No subtype wording may appear in any record, test expectation or report of the core module.

## B5. Supersession map (draft `DESIGN_BEFORE_EXECUTION.md` -> this errata / addendum)

Line numbers refer to the unchanged draft file.

| Draft passage | Status | Replaced by |
|---|---|---|
| Line 35 and D2 (line 156): constants `theta_cov`, single `g` | `theta_cov` dropped; `g` split | Addendum 3.4 (coverage as descriptors), 3.8 (`g_r`, `g_d`); errata B3 formulas. `w` is a labelling parameter only (3.11) |
| Line 40 "Continuation" including **Extended continuation** (and D12, line 166) | Superseded | Identity on ID-flagged edges by global greedy (addendum 3.5; tie order J -> x -> IDs, which agrees with the draft's "larger x, then lower lineage id, then lower index" after J); `break_with_link` (3.6). Draft sentence on Phase 9 equivalence -> compat mode (section 7) |
| Line 41 "Dormancy and re-link" | Refined | `g_r` and R-A (addendum 3.7, 3.8); key per errata B1 |
| Line 42 "Appearance (birth)" subtype | Refined | Origins (addendum 3.9) and `birth_subtype` (errata B3); applies to `origin = pure` only |
| Line 43 "Disappearance" (`last+g`) | Refined | Declaration at `L + g_d` (addendum 3.8, 4, 7.3) |
| Line 44 "Membership change" subtypes | Reduced | Counts only (errata B4); subtypes deferred |
| Line 45 "Growth / contraction", incl. "absorption growth ... not a new birth" | Rule kept, application changed | Size thresholds in addendum 3.2 and errata B3 `size_class`; a relation-only size jump now yields `succession` plus `born_succession`, not absorption growth |
| Lines 46-47 "Split", "Merge" (`theta_cov`, "best continuation child/parent") | Superseded | Event components and shapes (addendum 3.4, 3.9); S1 (3.10) |
| Lines 33, 53 (overlap graph; dormancy look-back `<= g`) | Refined | Look-back for re-link `<= g_r + 1` days back; declaration delay `g_d`. Unthresholded overlap store: errata B3 (not a required field) |
| Line 95 grid `g in {0, 1, 3}`; line 117 selection rule over `(theta_id, theta_ev, theta_cov, delta, g)` | **Unresolved, later tier** | Must be re-expressed over `(theta_id, theta_ev, delta, g_r)` with `g_d = g_r + 1`; the grid values and the selection rule remain [JUDGMENT] and are not approved |
| Lines 77, 81, 110 (event taxonomy: birth/death/split/merge/re-link; F1 floors) | **Unresolved, later tier** | Event vocabulary is now the end reasons and origins of addendum 3.9; the evaluation taxonomy and floors (D3) must be restated before the synthetic tier |
| Section 15 decisions D1, D3-D11 | Not settled here | Unchanged (addendum section 10) |

## Items that cannot be closed without a new research decision (none blocks the core module)

1. `headline_eligible`: removed; any later definition needs the D3/D4/D6 stratification decisions.
2. Membership-change subtypes (transfer, newcomer, re-entry, expired, isolated, other): deferred; the last four need data outside the input contract.
3. `birth_subtype` for `born_succession` and `born_reorganization_child`: left `null`.
4. Re-expression of the draft's grid (line 95), selection rule (line 117) and evaluation taxonomy/floors (lines 77, 81, 110) for the synthetic tier.
5. Persistence of the unthresholded overlap store as an artifact.

Implementation-level details that are **not** research decisions: the input-error exception type; the compat container for lifecycle descriptors (B2.5).

## Effect on toy rows (no rows re-derived here)

T16 asserts counts (`x = 8`, joined `2`, left `2`), not member sets. T12/T13 assert `birth_subtype` for `origin = pure` with `first_seen` supplied. T34 uses the `share`, `child_fraction` and `parent_coverage` formulas above and does not assert `headline_eligible`. No other toy row is changed by this errata.

## Integrity

See `INTEGRITY_ERRATA_1.json`. The sidecar of this file is `DESIGN_ADDENDUM_ERRATA_1_B1_B5.sha256` (SHA-256 of the file's bytes, 64 lowercase hexadecimal characters). A document cannot contain its own hash.
