# Phase 10 core module: implementation notes (finalization pass, 2026-10-18)

Documentation of implementation details of `src/evolution/tracker.py`. These notes add no research decision, no threshold and no result. The objects are candidate communities; nothing here is evidence of fraud or coordination. Governing specification: `DESIGN_ADDENDUM_D12_approved_decisions.md`, `DESIGN_ADDENDUM_ERRATA_1_B1_B5.md`, `DESIGN_ADDENDUM_ERRATA_2_targets_interface_censoring.md` (all unchanged). Phase 10 empirical tiers and parameter selection remain unauthorized.

## 1. Implementation choices checked against the specification

| Choice | Where in the code | Specification check | Verdict |
|---|---|---|---|
| **Parameter validation** at construction: thresholds `theta_id`, `theta_ev` in (0, 1]; `delta` > 0; `g_r` >= 0, `g_d` >= `g_r` + 1, `w` >= 1, all integers; thresholds and `delta` must be `Fraction` or integer (floats rejected); compat mode requires `g_r = 0` | `EvolutionTracker.__init__` | Addendum 3.8 requires `g_d >= g_r + 1` and allows `g_d = 1`; addendum 3.2 forbids floating point; errata 2 E2-B fixes parameters at construction with no defaults; the compat setting is `(g_r, g_d) = (0, 3)` (3.8), consistent with Phase 9 having no re-link. The ranges for `theta`, `delta`, `w` are not stated by the specification; the module is stricter only where a value would be meaningless (`theta` outside (0, 1], non-positive `delta` or `w`). Every value used in the approved toy cases, fixtures and sensitivity settings is accepted (tests pass). | Consistent. Implementation detail. |
| **Compat nodes carry `size_class`** | `step`, `selected_identity` branch | Errata 1 B3: `size_class` is defined for `basis` in {`selected_identity`, `relinked`} by comparing `b` with the size of `identity_predecessor`, `null` for `born`; no carve-out for compat mode, and compat nodes have basis `selected_identity`. | Consistent; this is what the text prescribes, not an extra choice. |
| **Compat `LineageEnd(disappeared)` records** (no other reason is emitted in compat) | `step`, declaration block | Errata 1 B2 item 5 names the container of compat lifecycle descriptors (`LineageEnd` of reason `disappeared`, or a lifecycle table) as an implementation detail, and excludes only ComponentRecords and relation EdgeRecords from compat. Errata 2 E2-C states the invariant "`status = disappeared` iff a `LineageEnd(disappeared)` has been emitted" without a mode exception. | Consistent. Implementation detail. |

## 2. Further implementation details (not in the specification, none conflicts with it)

* Compat mode uses `identity.gap = 0` and `edge_id = null` for selected nodes (errata 1 B3 / B2 item 5).
* `legacy_lifecycle()` exists only in compat mode and raises `InvalidInputError` if called in event mode (the exception type is reused for a wrong-mode call).
* Booleans are rejected as day or card ids; numpy integers are accepted and converted to `int`.
* Compat mode accepts any `g_d >= 1` as a parameter. Equivalence with the Phase 9 `Tracker` is claimed and tested only at `g_d = 3` (Phase 9's frozen constant); other values test the parameterization only.
* Results are deterministic because every ordering is an explicit total order on integers (no hash-order dependence).

## 3. Limitations that remain

* Expected values and the oracle in the tests were written from the specification by the same author as the module: they detect implementation slips, not misreadings of the specification.
* The sensitivity check covers 18 specific injected defects only.
* Compatibility with the Phase 9 `Tracker` is shown on 14 fixed and 500 seeded random synthetic canonical sequences, not on real data; real-data comparison needs Stage 0 card sets and is unauthorized.
* No threshold, parameter value, null-model property, benchmark result or real-data result exists; none is claimed.
