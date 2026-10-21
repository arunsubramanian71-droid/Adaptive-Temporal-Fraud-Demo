# Research scope reconciliation before any held-out authorization

Dated 2026-10-21. Read-only with respect to the held-out period: **no held-out file was opened, hashed, read or scored; `AUTHORIZATION.txt` does not exist; no held-out evaluator or arm has been run in held-out mode.** The base protocol (`HELDOUT_EVALUATION_PROTOCOL.md`, SHA-256 `59bf1565fb16653a02a2d743cc445e114fb2a6a42f1de7b2fa56261e6f68c806`) is unchanged. The additions it needs are in `AMENDMENT_1.md`.

The classification used: **mandatory** (a requirement of the brief or a registered hypothesis that needs a held-out reading), **already sufficiently evaluated** (the held-out reading would add nothing the validation evidence plus another held-out cell does not already give), **optional** (useful, not required), **unsupported** (cannot be evaluated on held-out data by a valid design).

## 1. Model-by-model decision for the held-out evaluation

| Model / question | Covered by the base protocol? | Decision | Where it is handled | Reason |
|---|---|---|---|---|
| Logistic regression, random forest, XGBoost (cell A) | yes | mandatory, **sufficient** | base protocol | frozen, thresholds transferred |
| XGBoost + graph features B_full (cell B) | yes | mandatory, **sufficient** | base protocol | H1-direct and H2 readings need B vs A on the future period |
| GraphSAGE x3 seeds (C), GAT x3 (D, convergence unresolved), Temporal GraphSAGE x3 (E) | yes | mandatory, **sufficient** | base protocol | GAT reported as non-converged |
| Phase 7A controls: schedule-matched static, memory ablated, x3 seeds | yes | mandatory, **sufficient** | base protocol | H2 (memory/time matters) needs them |
| **Adaptive system F and H4 on future data** | **no** | **mandatory, MISSING** | **Amendment 1 module F** | the brief: "compare static versus adaptive performance on future periods"; a valid design exists from TRAIN/VAL information only |
| Historical-feature XGBoost (reference line) | no | **mandatory as a reference, MISSING** | **Amendment 1 module H** | the brief asks graph gains be judged against it; not an A-F cell, labelled as a label-derived reference |
| Unweighted variants of A, B, C, E | no | **optional-recommended, MISSING** | **Amendment 1 module U** | class weighting changes XGBoost's validation PR-AUC (0.5395 weighted, 0.5948 unweighted), so the level of every XGBoost-based comparison (A against B) depends on this choice, while tabular above neural holds in both recipes on validation; a held-out reading shows whether that survives on a future period. Cheap (no new training beyond re-fitting two XGBoost models, accepted only after exact reproduction). |
| No-relations MLP control | no | **optional, excluded** | not added | one-run diagnostic with no registered held-out hypothesis; GraphSAGE against the tabular models on the future period already speaks to "do relations help" |
| H5 community discovery | no | **unsupported on held-out data** | none | communities are candidates without ground truth; the frozen Phase 9 gate G4 failed (unchanged); a label-based held-out criterion would be invented |
| Neural relationship-level ablations | no | **optional** (the brief says "where computationally feasible"; the XGBoost-based ablations B are done) | not added | no registered hypothesis depends on them; training them would be new work whose design should be decided by the user (see section 3) |
| `experiments/gat_300epoch/`, superseded runs, Phase 10 empirical tiers | no | excluded / unauthorized | none | aborted or not part of the brief |

Conclusion: **the existing protocol does not cover every mandatory held-out requirement.** The adaptive system (F, H4) and the reference line (H) are missing, and U is recommended. An evaluation that ran only the base protocol would leave the brief's "static versus adaptive on future periods" requirement unmet and H4 validation-only.

## 2. Why the amendment is valid (no held-out information used)

* Every design choice (windows, K, learning rate, thresholds, replay construction, burn-in span, H4 rules, tolerances) comes from Phase 8, the shared TRAIN-only selection (`selected_from: TRAIN_internal_holdout`), the validation-period state-continuity supplement or recorded validation metrics.
* The only tolerance chosen after seeing results (0.01 on the window-mean PR-AUC of the trained arms in the F readiness gate) was set after the validation supplement and is disclosed in `AMENDMENT_1.md`.
* The base protocol file is untouched, so its hash is unchanged. The amendment is honoured only if the user's authorization also quotes the amendment hash and names the modules.

## 3. Neural relationship-level ablations: conditional optional

The brief: "Relationship-level ablations (where computationally feasible)". Evidence available: remove-one-relation and time-window ablations for the graph-feature XGBoost (comparison B, executed), the memory-ablated and schedule-matched controls for Temporal GraphSAGE (Phase 7A), the non-relational MLP control (Phase 5), and the counterfactual link removal and memory reset on the designated Temporal GraphSAGE checkpoint (neural graph evidence). What is missing is retraining GraphSAGE/Temporal GraphSAGE with each relation removed. It is feasible but costs on the order of 4 relations x 3 seeds x 2 families of GPU training; the earlier GAT never converged. Decision: **optional, not mandatory**; no registered hypothesis (H1-H4) turns on it and the partial evidence above answers the "where feasible" clause for the XGBoost family and for temporal memory. Not run. It would need its own pre-registration and a user decision.

## 4. NE5 (never-seen-entity memory reset must be exactly zero): diagnosis, rule unchanged

The registered criterion NE5(a) failed: 13 of 102 resets of never-seen entities changed a logit by up to 9.5e-07. `scripts/diagnose_ne5.py` (validation only, no label) reruns the same counterfactuals: **on the CPU every one of the 102 resets and the no-op perturbation is exactly zero; on the GPU even a perturbation that changes nothing, done twice, differs at the same 1e-6 scale** (`experiments/neural_graph_evidence/diagnostics/NE5_diagnosis.json`). Verdict: **floating-point nondeterminism of GPU kernels, not an implementation defect and not a genuine behaviour of the model.** The registered rule (exact zero) and the registered result (FAILED) are not changed; the diagnosis is a labelled post-hoc addition.

## 5. Phase 8 memory initialisation (found while checking F)

Phase 8's walk-forward loop starts the entity memory empty at day 122 instead of building it over days 1-121. The recorded Phase 8 values are exactly what that loop produces (1e-7). The state-continuity supplement (`experiments/phase8_state_continuity_supplement/`, pre-registered before any arm ran) reruns the three arms with the memory built first: **H4 stays SUPPORTED under the same locked rules** (mean Ad - F +0.0410 against a larger SD 0.0128; Ad - E +0.0391 against 0.0045; Ad degrades less in at least 2 of 3 seeds), and all three arms' absolute PR-AUC rises. The pre-registered reproduction criterion S1 for the trained arms (window PR-AUC within 0.01 of the recorded run) **failed** (6 of 8 later windows differ by more than 0.01, largest 0.0303) because training is not bit-reproducible; the criterion was not changed and a labelled diagnostic shows the window-mean differences are about 0.003. Phase 8 outputs are untouched.

## 6. Consequences of the choices (exact)

| Choice | Consequence |
|---|---|
| Authorize the base protocol only | Evaluates the 19 models; leaves H4 and cell F, the reference line and the weighting question without a future-period reading. The brief's adaptive-future requirement would remain unmet and must be reported as such. Irreversibly spends the held-out period on that evaluation: modules F, H, U cannot be added later without being a second look at the same period (adaptive reuse the base protocol's section 7 forbids, though not leakage into any frozen model). |
| Authorize base + Amendment 1 modules F, H, U (recommended) | One-shot irreversible use of the held-out period for the 19 base models, 8 unweighted models, the 3 adaptive arms and the reference line. Closes the brief's adaptive-future and reference-line requirements. Costs: the F run is about 3 x 3 arms of GPU walk-forward (several minutes each; the validation analogue took about 7 minutes in total) plus a few minutes for H and U. |
| Authorize base + F + H only | As above without the weighting reading; U then cannot be added later. |
| Do not authorize | The held-out requirement of the brief stays BLOCKED, as recorded. Everything else in the submission is unaffected. |

Whatever is authorized: no model, threshold, setting or exclusion may change in response to a result; unfavourable results are reported in full; community discovery (H5) and the Phase 9 G4 failure, and Phase 12 AC12 (BLOCKED: 300 of 651 pre-existing files covered, 322 unverifiable), are unchanged by any of this.
