# Held-out evaluation: the authorization request in one page

Dated 2026-10-22. Every fact below was re-checked on that date **without opening any held-out file**. The full options, risks and alternatives are in `HELDOUT_DECISION_PACKAGE.md`. This page asks for two decisions and authorizes nothing by itself.

## The twelve facts the decision rests on

| # | Item | Verified state |
|---|---|---|
| 1 | Frozen protocol | `experiments/final_heldout_protocol/HELDOUT_EVALUATION_PROTOCOL.md` (19 frozen models, metrics, one-shot rule, authorization wording) and `AMENDMENT_1.md` (modules F, H, U) |
| 2 | Hashes (sidecars match) | protocol `59bf1565fb16653a02a2d743cc445e114fb2a6a42f1de7b2fa56261e6f68c806`; Amendment 1 `37a1c0577a8ef28530a6120a027890c11ec458ca97329d14a05108db21a27e6e`; proposed Amendment 2 revision 1 (A2-a-r1, not active) `37ca86ee8a99c6fdf185ffa2639134c990351ce8823782ab7160dcc52a1a56eb` |
| 3 | Consistency with the brief | the brief requires a latest/future test period and "compare static vs adaptive performance on future periods" (`TRACEABILITY_MATRIX.md` R12, R17); base + F + H + U covers both |
| 4 | Models and arms | base: logistic regression, random forest, XGBoost, B_full, GraphSAGE x3, GAT x3, Temporal GraphSAGE x3, two Phase 7A controls x3. F: frozen, adaptive and replay arms x 3 seeds. H: the historical-feature XGBoost reference. U: A_none, B_none, unweighted GraphSAGE x3, unweighted Temporal GraphSAGE x3 |
| 5 | Thresholds and metrics | each model's recorded validation best-F1 threshold, transferred unchanged (F: TRAIN-selected); PR-AUC (primary), ROC-AUC, segment PR-AUC, P@K and R@K, confusion matrix, alerts, false positives |
| 6 | Data access | only after `AUTHORIZATION.txt` exists: `data/features/test_features.parquet` (tabular, B), the graph file (neural; masked day by day), the historical feature file (H); labels only to compute metrics |
| 7 | Order | (1) base, (2) H and U together, (3) F only if its gate has passed |
| 8 | Irreversible consequences | the period can never again be untouched test data; one-shot per output folder (any output file blocks a repeat); every result reported, nothing changed afterwards |
| 9 | Readiness gates (re-checked today, code fingerprints unchanged) | base **PASS** (19 of 19, largest score difference 8.94e-07); H, U **PASS** (9 of 9); F **FAIL as written** (E seed 123: 0.0104 against 0.01; GPU nondeterminism), so the F runner refuses |
| 10 | Outputs and failure behaviour | `heldout_results/`, `heldout_results_amendment/`, `heldout_results_f/` (metrics CSV, manifest with hashes, row-level predictions that are never published); a crash before any score is written may be repaired and repeated; after that, a repeat is refused |
| 11 | Independent prerequisites | complete: protocol frozen and hashed, evaluators tested, readiness passed for base, H and U, guards tested (one-shot, scope, authorization, code fingerprints), `AUTHORIZATION.txt` absent, no held-out output exists |
| 12 | No tuning on results | thresholds and models are frozen; the runners fit nothing; the protocol forbids changes after any held-out metric exists |

## Decision 1: the Module F gate (needed before the held-out run if F is included)

The registered gate cannot be met reliably on this GPU (about 50 % false failure over six cells). Choose one:

* **Approve A2-a-r1** (recommended): reply *"I approve Amendment 2 revision 1 (A2-a-r1) as the Module F trained-arm readiness gate."* I then implement its decision script and authorization check (with tests), re-run the readiness gates, make its 60 validation-only runs (about 1.5-2.5 h GPU), and report pass or fail before anything else. **Approve before the runs**, so that the outcome cannot influence the approval.
* **Exclude F:** the brief's future-period static-versus-adaptive comparison then stays unmet permanently.

## Decision 2: the held-out authorization (irreversible)

Create `experiments/final_heldout_protocol/AUTHORIZATION.txt` yourself (protocol section 6 says it is created by the user), with the text in `HELDOUT_DECISION_PACKAGE.md` section 4 for your chosen scope. Recommended: Option B (base + F, H, U). The base and H/U parts can run as soon as the file exists. F runs only after A2-a-r1 has passed. Then tell me to run the commands. I report every result as it is.

**What stays blocked until then:** R12 and R17 of the traceability matrix (all future-period readings of H1-H4). Nothing else in the software depends on this decision.
