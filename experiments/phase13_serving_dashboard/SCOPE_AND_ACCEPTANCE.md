# Phase 13: serving API and investigation dashboard: scope and acceptance record (reconstructed)

Dated 2026-10-21. Written and hashed (`SCOPE_AND_ACCEPTANCE.sha256`) before any Phase 13 code exists.

## 1. Provenance of this scope: what is documented and what is not

**The original Phase 13 specification (its title, deliverables and acceptance criteria) was not recovered.** No project file defines it, and the saved sessions contain no definition. What exists is documentary evidence that a Phase 13 was planned and what the remaining plan looked like when the later phases were still undefined:

| Evidence | Source |
|---|---|
| After Phase 7 the user listed the remaining items in order: adaptive updating / concept drift, suspicious community / ring discovery, explainability, **final ablation suite, dashboard, final report** | user message of the Phase 7 authorization (session `29b7adc9-...jsonl`, entry 1656) |
| Phases 8 (adaptive/drift), 9 (community discovery), 11 (explainability) and 12 (consolidated ablation/evidence table) were later carried out under those names; Phase 10 (community evolution) was inserted | phase reports and scope records |
| The user's Phase 10 design instruction says "Do not start Phases 11-13", and every Phase 10-12 record says "Phases 11-13 not started" or "Phase 13 not started" | session `64d369ab-...jsonl` entry 499; `experiments/phase10_community_evolution/INTEGRITY*.json` |
| The original brief lists "production-style inference" among the system's components, an `api/` and `dashboard/` folder in its suggested layout, FastAPI and Streamlit among "likely technologies", and ends the community pipeline with "investigation output"; it also warns against unnecessary services and against installing every technology | brief, sections "The system must combine", "ENGINEERING ARCHITECTURE", "TECHNOLOGY POLICY", "COORDINATED COMMUNITY DISCOVERY" (the ring-discovery section) |
| Phase 11 and the Phase 12 scope explicitly excluded a dashboard and an API ("No dashboard/API"), which fits them being deferred to a later phase | Phase 11 and Phase 12 records |

**Reconstruction (my inference, to be confirmed by the user):** Phase 13 is the last two items of the user's own remaining list, the **dashboard and the final report**, together with the brief's **production-style inference** layer. The final report exists (`submission/MANUSCRIPT.md`, `reports/PROJECT_SYNTHESIS_REPORT.md`). This phase therefore builds the serving API and the investigation dashboard. If the user's original Phase 13 meant something different, that difference is an unresolved scope gap and this record does not close it.

## 2. Requirements taken from the brief (documented)

* D1 A production-style inference path (exists as a command line scorer; an HTTP interface is added).
* D2 An investigation output for candidate communities, never calling them confirmed fraudulent or coordinated groups.
* D3 Minimum technology stack; no unnecessary services; no technology installed without a demonstrated need.
* D4 Evaluation honesty: validation versus held-out results kept separate; held-out period sealed.
* D5 Explanations for model outputs, with the caveats already recorded.

## 3. Design choices (mine, not requirements)

* **No new dependency.** The API and dashboard use the Python standard library (`http.server`, `json`) and one self-contained HTML page with plain JavaScript (no external scripts, fonts or CDN). FastAPI and Streamlit are not installed; the brief calls them "likely", not required. They remain a possible drop-in and are not claimed.
* **Local only.** The server binds to 127.0.0.1 and refuses another address unless a flag explicitly allows it. No authentication, multi-user or deployment features.
* **Aggregate data only.** The dashboard serves recorded validation-period aggregates (Phase 12 and comparison B tables, Phase 11 aggregate-only community cards) and scores rows that the user posts. It never serves transaction-level files, member identities or anything from the held-out period.
* **Scoring bundle.** The scoring endpoint uses a bundle chosen by the operator at startup (`--bundle`, or `--demo` which trains the synthetic-fixture bundle in memory and labels every response `synthetic_demo: true`). Clients cannot name file paths.
* **Endpoints:** `GET /` (dashboard), `GET /health`, `GET /api/status`, `GET /api/model`, `GET /api/evidence`, `GET /api/communities`, `POST /api/score`.

## 4. Acceptance criteria (fixed now)

* **P13-1 Startup and binding.** `scripts/run_dashboard.py --demo` starts from the project root with no new dependency, binds 127.0.0.1, and refuses a non-local `--host` unless `--allow-remote` is given. A held-out-named bundle path is refused.
* **P13-2 HTTP behaviour.** Unknown path: 404 JSON; wrong method: 405 JSON; invalid JSON or schema: 400 with an actionable message; body over 1 MB or more than 5,000 rows: 413 / 400; all errors are JSON; the server never returns a stack trace.
* **P13-3 Scoring is real and causal.** `POST /api/score` calls `src/inference/scorer.score`; a response score for a row equals the library result for the same inputs; scores of early rows do not change when later rows are removed; labels in the posted rows are ignored; explanations appear only for alerted rows; every response carries the bundle's `synthetic_demo` flag and a statement that scores are not evidence of fraud.
* **P13-4 Evidence endpoint.** Values are read from the recorded files at request time (not hard-coded) and equal those files (tested against `evidence_cells.csv`, `table_with_B.csv` and `unweighted_decision.json`); the response carries the validation-period banner, lists B and the F basis separately, and states that no held-out evaluation exists.
* **P13-5 Communities endpoint.** Only the aggregate fields of `cards_W7_seed42.json`; no member identities; the response carries both whitelisted disclaimers; the Phase 9 wording lint is clean on every response body.
* **P13-6 Access control.** The server serves only its single HTML page and the endpoints above; path traversal and any path containing a held-out name return 404; no endpoint reads a TEST-named file, a checkpoint or a transaction-level file; this is checked by tests and by a log of every file the server opened.
* **P13-7 Dashboard page.** Self-contained (no external URL), shows the validation-only banner and the candidate-community wording, builds all dynamic content with `textContent` (no HTML injection), and renders correctly in the in-app browser (screenshot recorded).
* **P13-8 Tests and records.** Automated tests pass (HTTP tests against a server on an ephemeral port); a mutation-style check shows the access-control and causality tests fail on injected defects; outputs and sources are hashed; wording lint clean.
* **P13-9 Documentation.** README, demonstration guide and reproducibility guide describe the commands; the final completion record states this phase's status and this reconstruction caveat.
* **P13-10 Boundaries.** No Git, no held-out access, no model training or scoring on research data (the demo bundle is trained on the synthetic fixture), no change to any Phase 1-12 artifact, no new dependency.

## 5. Out of scope

Authentication, deployment, streaming ingestion (Kafka and similar), multi-user features, a database, a map or graph visualisation library, any claim that scores or communities indicate fraud, and any held-out result.

## 6. Limits stated in advance

The page is a demonstration surface for recorded aggregates and for scoring posted rows; it is not a monitoring system. Community cards cover one window (7 days) and one seed of the exploratory Phase 9 output. The synthetic demo bundle reproduces no research result.
