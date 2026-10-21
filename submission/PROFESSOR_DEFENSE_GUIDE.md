# Professor defense guide

A concise technical guide for presenting and defending the project. Every command below was run on the development machine on 2026-10-21 (Windows 11, Python 3.12.0). All research numbers are **validation-period** results (days 122-151). **No held-out evaluation has been performed.** It is a separate, one-shot decision (`HELDOUT_DECISION_PACKAGE.md`).

## 1. The question and the honest one-line answer

*Does heterogeneous relational structure plus temporal behaviour improve fraud detection and coordinated-fraud discovery over strong tabular models, especially on future data and under drift?* On the validation period, a tuned XGBoost on transaction features is at least as good as every graph model tried, with or without class weighting. Recurrent entity memory gives a small gain, concentrated on cold-start transactions, over its controls. Periodic refresh reduces within-validation degradation (H4 supported on validation). Coordinated-community discovery is not established, because the synthetic benchmark failed a frozen validity gate.

## 2. Architecture and data flow

```
IEEE-CIS train files --(build_processed_data.py: chronological split, TRAIN-fit encoders)--> data/features/{train,val,test}_features.parquet
   |                                                                                         |
   |--(build_graph.py: transaction - card / pseudo-account / network proxy / device)--> heterogeneous graph with edge times
   |                                                                                         |
tabular models (LR, RF, XGBoost)      B: XGBoost + 40 backward-looking graph features      GraphSAGE / GAT (static, causal masks)
                                                                                           Temporal GraphSAGE (GRU entity memory, day by day)
   \__________________________ validation metrics (one metric module) ______________________/
                                   |
   Phase 8 drift + adaptation (frozen / periodic refresh / replay control; triggered refresh analyses)
   Phase 9-11 candidate communities (Louvain on the suspicious subgraph, scored, aggregate cards), explanations
   Phase 12 consolidated table; held-out protocol (frozen, gated, not run)
   Serving (src/serving): evidence + communities (recorded aggregates), demo scoring (synthetic bundle), research scoring (the recorded XGBoost)
```

## 3. Major algorithms and design decisions (why each exists)

* **Chronological split, TRAIN-only fitting.** Every encoder, threshold and hyperparameter came from TRAIN or validation. TEST is sealed, and guards refuse TEST-named paths.
* **Graph schema from the data, not imagined.** Billing address and email domain were measured in Phase 4 and left out: they showed no signal or only weak lift, with very large hubs. No customer, account or IP identifiers were invented. The pseudo-account is an explicit derivation, `card1 + addr1 + (day - D1)`.
* **Causality.** Graph features use only strictly earlier rows. Static GNNs score day t on the graph masked to edges with time <= t. The temporal model carries entity memory forward day by day and never sees a label of a day it scores.
* **One temporal architecture.** Temporal GraphSAGE with GRU entity memory, chosen for its fit to the data's entity-reuse structure. The Phase 7A controls (memory ablated, schedule-matched static) separate memory from training regime.
* **Adaptation.** Phase 8 tests a frozen model, a periodic refresh, and a replay control that matches compute using TRAIN days only, under locked rules. Two pre-registered triggered-refresh analyses follow the brief's "monitor drift/performance" wording.
* **Communities as candidates.** Louvain on the suspicious subgraph, scored on concentration, temporal compactness, density and conductance. They are always called *candidate coordinated communities*, because IEEE-CIS has no ring labels.
* **Explanations.** XGBoost's native per-feature contributions (TreeSHAP values) sum to the model margin. For the temporal GNN, pre-registered counterfactuals (link removal, memory reset). Attention is not used, because GAT never converged.

## 4. Results to show (validation period)

Use `MANUSCRIPT.md` section 5 and figures 1-3 (`submission/figures/`):
* **Tabular versus graph.** Tabular XGBoost is at 0.5395 PR-AUC with class weighting, 0.5948 without, and 0.5616 with subsampling. B (plus graph features) is not reliably different, and H1-direct is not established. GraphSAGE scores 0.4637, GAT 0.3320 (non-converged), Temporal GraphSAGE 0.4927.
* **Imbalance.** Class weighting hurts XGBoost: 0.5948 unweighted against 0.5395 fully weighted.
* **Drift and adaptation.** Window-mean PR-AUC is 0.4580 adaptive, 0.4243 frozen and 0.4179 replay control; H4 is supported under the locked rule.
* **Memory-initialisation supplement.** With the entity memory built first, H4 stays supported.
* **Triggered refresh.**
  * Score-distribution trigger: never fired.
  * Performance trigger: fired once per seed, beat the frozen arm, but stayed below periodic refresh. Both are NOT SUPPORTED by their registered rules.

## 5. Failures and limitations to state plainly

Phase 9 gate G4 failed on 2 of 3 seeds, so H5 is not established. GAT never converged. NE5(a) failed at GPU float noise (exact on CPU). Module F's readiness gate failed as written because of GPU nondeterminism; a proposed, not active, Amendment 2 is in `experiments/final_heldout_protocol/`. Phase 12 AC12 is blocked, since 322 of 651 files have no recoverable pre-build hash. Phase 13's original specification was never recovered. There is no held-out evaluation. The three seeds share data and are not independent replications.

## 6. Live demonstration (verified commands)

```
.venv/Scripts/python.exe -B scripts/check_setup.py
.venv/Scripts/python.exe -B scripts/demo_end_to_end.py --out-dir demo_output
.venv/Scripts/python.exe -B scripts/make_research_sample.py --n 25
.venv/Scripts/python.exe -B scripts/run_dashboard.py --demo --research-model
```
Open http://127.0.0.1:8765/ and walk through:
1. **Evidence (validation):** the recorded tables and the standing banner.
2. **Score transactions (demo):** click *Load synthetic sample*, then *Score*. The response is labelled SYNTHETIC DEMONSTRATION BUNDLE.
3. **Research model:** choose the file written by `make_research_sample.py` (it goes to your temporary folder, outside the repository), then click *Score with the research model*. The response is labelled RESEARCH MODEL, shows the model-file hash and the recorded validation threshold, and lists the largest contributions for alerted rows.
4. **Candidate communities:** the aggregate-only cards, with the two disclaimers.
5. **Status and limits.**

Stop with Ctrl+C. To show the evidence that it works, run:
```
.venv/Scripts/python.exe -B scripts/run_tests.py
.venv/Scripts/python.exe -B scripts/verify_dashboard_e2e.py
.venv/Scripts/python.exe -B scripts/verify_research_serving_e2e.py
```

## 7. Likely questions

* **Is the demo the research?** The synthetic tab is not. The research tab serves the recorded XGBoost exactly as evaluated: the HTTP scores equal the recorded validation predictions, and a test checks this.
* **Why not serve the GNNs?** They score whole days of graph state with entity memory, not single posted rows. Serving them would need a streaming graph store, which the brief's minimum-stack rule argues against.
* **Why no held-out numbers?** The period is one-shot. Opening it is a separate decision, and the frozen protocol is ready.
* **Did you tune on validation?** Yes, and it is stated everywhere. That is exactly why the held-out evaluation matters.
* **Are the communities fraud rings?** No. They are candidates; the dataset has no ring labels, and the validity gate failed.
* **Why did graphs not help?** A plausible reading, not a tested claim: the transaction features already encode much entity history (the C, D and V columns), so relational signal adds little beyond them on this dataset. The cold-start gain of the memory model is the one consistent relational effect measured.
