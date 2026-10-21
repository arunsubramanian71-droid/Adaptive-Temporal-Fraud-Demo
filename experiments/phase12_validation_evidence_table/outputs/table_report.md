# Phase 12 - consolidated validation-evidence table

**Descriptive table of recorded validation-period (VAL, days 122-151) predictions. Every model was tuned on VAL (early stopping, epoch selection or thresholds) except where stated for F. No held-out (TEST) comparison exists, TEST is sealed, and nothing here supports a claim of generalization, statistical significance, causal attribution, fraud or coordination.**

**F rows are on a distinct evaluation basis: the shared checkpoint was trained on days 1-111, the threshold came from TRAIN days 112-121, and the adaptive arm updates on already-scored VAL labels. F is not comparable with A, C, D or E.**

All values are recomputed from stored validation predictions with one metric implementation (`src/models/metrics.py`, unmodified). Cells with several seeds show mean ± sample SD (ddof = 1) over seeds; one-run cells show the value. Operating-point metrics (precision, recall, F1, alerts) are in the CSV with their origin and are not compared across origins.

Recomputation against the recorded `metrics_val.json` files: 65 runs/windows checked, max absolute difference 0 (tolerance 1e-09). Cross-check files: 173 items compared, 0 beyond tolerance.

| Cell | Model / arm | Basis | Runs | PR-AUC | ROC-AUC | PR-AUC warm | PR-AUC cold | PR-AUC no-edge | P@1% | R@1% |
|---|---|---|---|---|---|---|---|---|---|---|
| A | logistic_regression | pooled VAL | 1 | 0.4126 | 0.8520 | 0.1671 | 0.2972 | 0.7067 | 0.7539 | 0.2231 |
| A | random_forest | pooled VAL | 1 | 0.4543 | 0.8792 | 0.2046 | 0.3807 | 0.7366 | 0.8196 | 0.2426 |
| A | xgboost | pooled VAL | 1 | 0.5395 | 0.9050 | 0.3496 | 0.4064 | 0.7905 | 0.8937 | 0.2645 |
| A-reference | xgboost_historical_control | pooled VAL | 1 | 0.6771 | 0.9432 | 0.9299 | 0.0369 | 0.4960 | 0.9928 | 0.2938 |
| C | GraphSAGE (Phase 6A) | pooled VAL | 3 | 0.4637 ± 0.0167 | 0.8616 ± 0.0038 | 0.2611 ± 0.0440 | 0.3417 ± 0.0113 | 0.7182 ± 0.0125 | 0.8092 ± 0.0249 | 0.2395 ± 0.0074 |
| C-control | MLP control, no relations (Phase 6A extended) | pooled VAL | 1 | 0.4167 | 0.8547 | 0.1758 | 0.3007 | 0.7029 | 0.7575 | 0.2242 |
| D | GAT (Phase 6B) | pooled VAL | 3 | 0.3320 ± 0.0052 | 0.8653 ± 0.0027 | 0.2639 ± 0.0077 | 0.3161 ± 0.0055 | 0.4033 ± 0.0097 | 0.5791 ± 0.0312 | 0.1714 ± 0.0092 |
| E | Temporal GraphSAGE (GRU memory, Phase 7) | pooled VAL | 3 | 0.4927 ± 0.0035 | 0.8667 ± 0.0009 | 0.3213 ± 0.0111 | 0.3562 ± 0.0071 | 0.7271 ± 0.0049 | 0.8423 ± 0.0115 | 0.2493 ± 0.0034 |
| E-control | Static GraphSAGE, schedule-matched (Phase 7A) | pooled VAL | 3 | 0.4792 ± 0.0114 | 0.8554 ± 0.0045 | 0.3253 ± 0.0445 | 0.3288 ± 0.0050 | 0.7132 ± 0.0129 | 0.8284 ± 0.0050 | 0.2452 ± 0.0015 |
| E-control | Temporal, memory ablated (Phase 7A) | pooled VAL | 3 | 0.4790 ± 0.0076 | 0.8571 ± 0.0054 | 0.2951 ± 0.0279 | 0.3305 ± 0.0065 | 0.7295 ± 0.0130 | 0.8248 ± 0.0147 | 0.2441 ± 0.0044 |
| F | F-frozen | F: mean of 5 window PR-AUCs | 3 | 0.4243 ± 0.0128 | 0.8525 ± 0.0134 | 0.1822 ± 0.0192 | 0.3544 ± 0.0103 | 0.6966 ± 0.0055 | n/a | n/a |
| F | F-frozen | F: 5 windows concatenated | 3 | 0.4297 ± 0.0142 | 0.8555 ± 0.0131 | 0.1725 ± 0.0202 | 0.3545 ± 0.0107 | 0.7187 ± 0.0068 | 0.8152 ± 0.0197 | 0.2413 ± 0.0058 |
| F | F-adaptive | F: mean of 5 window PR-AUCs | 3 | 0.4580 ± 0.0082 | 0.8570 ± 0.0072 | 0.2348 ± 0.0135 | 0.3723 ± 0.0023 | 0.7209 ± 0.0041 | n/a | n/a |
| F | F-adaptive | F: 5 windows concatenated | 3 | 0.4542 ± 0.0048 | 0.8507 ± 0.0078 | 0.2225 ± 0.0108 | 0.3668 ± 0.0018 | 0.7259 ± 0.0058 | 0.8236 ± 0.0138 | 0.2438 ± 0.0041 |
| F | F-replay_control | F: mean of 5 window PR-AUCs | 3 | 0.4179 ± 0.0063 | 0.8347 ± 0.0149 | 0.1934 ± 0.0181 | 0.3584 ± 0.0053 | 0.7041 ± 0.0050 | n/a | n/a |
| F | F-replay_control | F: 5 windows concatenated | 3 | 0.4148 ± 0.0121 | 0.8294 ± 0.0157 | 0.1751 ± 0.0212 | 0.3490 ± 0.0064 | 0.7087 ± 0.0115 | 0.7686 ± 0.0252 | 0.2275 ± 0.0075 |
| B | tabular + graph features | - | 0 | **MISSING** | | | | | | |

**B is MISSING.** no artifact exists (no graph-feature XGBoost); the historical-feature baseline is a different construct and is not substituted; H2 cannot be assessed from this table.

## Comparability (labels fixed in the scope record, section 5)

| Comparison | Label | Reason |
|---|---|---|
| A (LR, RF, XGBoost) vs C, D, E | Comparable on evaluation basis; attribution not supported | same rows, labels and metrics; but different input encodings (raw/NaN-native versus imputed and standardized plus graph), different training regimes, A is one run versus three seeds, and all were tuned on VAL |
| C vs its MLP control | Comparable, with limits | same features, width and recipe; the control isolates the relations (Phase 6A); but the control is one run (seed 42) that reached its epoch cap, so it has no seed variance and may be under-trained |
| C vs D (architecture) | Not comparable | D reached the epoch cap in 3 of 3 seeds, so "worse" versus "under-trained" is unresolved; Phase 6C was aborted and is excluded (R6) |
| C vs E (temporal information) | Not comparable as a single-factor comparison | training regime differs (1 versus 18 optimizer steps per epoch); use the Phase 7A chain |
| Phase 7A chain: C reference, static schedule-matched, memory-ablated, E | Comparable | pre-registered ablation, same graph, features, seeds and protocol; n = 3 |
| F-frozen vs F-adaptive vs F-replay | Comparable | same shared checkpoint per seed, same windows and protocol; the adaptive arm updates on already-scored VAL labels |
| F (any arm) vs A, C, D or E | Not comparable | checkpoint trained on days 1-111 versus 1-121; TRAIN-only versus VAL-tuned threshold and no VAL epoch selection; different metric basis; adaptive updates use VAL labels; this holds even for F-frozen versus E despite the same architecture |
| A reference (historical control) vs anything | Reference line only | not an A-F cell; same rows, but its signal is the account-level labeling artifact documented in Phase 3 |
| Any statement about B or H2 | Not available | no graph-feature XGBoost exists; the historical-feature baseline is a different construct and is not substituted |
| Window-level PR-AUC across cells | Same labels as the rows above | window membership is identical, so only the training and protocol differences above apply |

## Notes

* "Evaluation-comparable" means the same 83,655 VAL rows and labels scored by the same metric code. It never implies that a difference is attributable to the factor named.
* Excluded and never read: `experiments/gat_300epoch/`, `experiments/graphsage/`, `experiments/graphsage_control_mlp/`, the temporal pilot, every checkpoint and feature file.
* The Phase 8 replay-control arm is named F-replay to avoid a clash with cell E.
* No held-out comparison, significance test, interval or causal statement is made here.
