# Phase 13: scope confirmation (recovered requirements versus reconstructed assumptions)

Dated 2026-10-22. **The original Phase 13 specification was not recovered.** Every saved session (four transcripts, all message types) was searched again on 2026-10-22. Before the reconstruction, Phase 13 appears only inside prohibitions ("Do not start Phases 11-13", "Do not begin Phases 11-13"), never with a definition. This document separates what the original sources require from what I assumed, and states what you are asked to confirm.

## 1. Requirements recovered from original sources

| # | Requirement | Source |
|---|---|---|
| R1 | "production-style inference" is one of the system's components | brief, list after "The system must combine" |
| R2 | suggested layout includes `api/` and `dashboard/` folders ("Prefer a structure such as") | brief, ENGINEERING ARCHITECTURE |
| R3 | FastAPI and Streamlit are "Likely technologies", but "do not install or use every technology automatically. Use the minimum stack necessary"; no unnecessary microservices | brief, TECHNOLOGY POLICY |
| R4 | the community pipeline ends in "investigation output"; communities must be called candidates unless ground truth exists | brief, the coordinated-community section |
| R5 | explanations for model outputs; attention weights are not causal explanations | brief, EXPLAINABILITY |
| R6 | after Phase 7 the remaining items, in order: adaptive updating and drift, community discovery, explainability, final ablation suite, **dashboard, final report** | the user's Phase 7 authorization (session `29b7adc9`) |
| R7 | evaluation honesty: chronological split, final period kept out of every decision | brief, EVALUATION |

## 2. Assumptions I made (not in any source)

| # | Assumption | Why |
|---|---|---|
| A1 | Phase 13 = dashboard + API + final report (R1, R6) | best-supported reading of the remaining items |
| A2 | local only: binds 127.0.0.1, no authentication or deployment | no deployment requirement exists; safest default |
| A3 | Python standard library instead of FastAPI or Streamlit | R3 "minimum stack"; no new dependency |
| A4 | dashboard shows recorded validation aggregates and aggregate-only community cards; no transaction-level or held-out data | R4, R7, data-licence restrictions |
| A5 | two scoring modes, labelled separately: a synthetic demo bundle, and the research XGBoost served exactly as evaluated | R1 without redistributing data |
| A6 | the neural models are not served (they score whole days of graph state, not single posted rows) | serving them needs a streaming graph store, against R3 |
| A7 | acceptance criteria P13-1 to P13-10 (`experiments/phase13_serving_dashboard/SCOPE_AND_ACCEPTANCE.md`), written before the code | needed something testable |

## 3. Verification of the current implementation against these items (verified 2026-10-22)

| Item | Evidence | Result |
|---|---|---|
| P13-1 startup, local binding, refusals | `scripts/verify_dashboard_e2e.py` (live) | 31 of 31 checks |
| P13-2 HTTP errors as JSON, no stack trace, size limits | same; `tests/test_serving.py` | pass |
| P13-3 causal scoring, labels ignored | `tests/test_serving.py`, `tests/test_research_serving.py` | pass |
| P13-4 evidence endpoint reads recorded files | `tests/test_serving.py` | pass |
| P13-5 communities endpoint aggregate-only, disclaimers | `tests/test_serving.py` | pass |
| P13-6 access control, path traversal, held-out names | E2E script, `tests/test_serving.py` | pass |
| P13-7 self-contained page, validation banner | E2E script; browser rendering check (2026-10-21); real-browser check `scripts/verify_dashboard_browser.py` (2026-10-22: 123 of 123, desktop and phone, light and dark, after fixing the 13 defects it found) | pass |
| P13-8 to P13-10 tests, documentation, boundaries | test suite; README; demo guide; mutation check 14 of 16 detected, 2 equivalent (2026-10-22) | pass |
| R1 research-model inference | `scripts/verify_research_serving_e2e.py` (live, 12 of 12); HTTP scores equal the recorded validation predictions | pass |

## 4. Gaps against the original sources (not defects of the reconstruction, but choices you may want changed)

* No folders named `api/` and `dashboard/` (the code is in `src/serving/`); R2 says "such as".
* FastAPI and Streamlit are not used (R3 allows this).
* No entity-relationship graph view and no live drift panel: the dashboard shows aggregate community cards and recorded drift results, not interactive graphs.
* The neural models are not served (A6).

## 5. What you are asked to confirm

Either **confirm** "Phase 13 = local dashboard + scoring API + final report, as implemented (A1-A7)", or **name the changes** (for example FastAPI endpoints, a Streamlit interface, an entity-graph view, a drift panel). Each named change would be implemented and tested against the same safeguards; none requires held-out data.
