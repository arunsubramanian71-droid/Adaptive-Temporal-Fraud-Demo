# Comparison B (tabular + handcrafted graph features): results

**Validation-period (VAL, days 122-151) results. Comparison A and B use VAL for early stopping and the operating threshold, like every other model in the table except the F arms. No TEST data was read. Nothing here supports a held-out, significance, causal, fraud or coordination claim.**

Protocol: `PREREGISTRATION_B.md` (written and hashed before any B code or output). Decision rules are applied mechanically by `scripts/run_comparison_b.py` from recorded values. Verdicts below are those the rules return; they are not adjusted.

## Pre-registered verdicts

| Question | Verdict | Basis |
|---|---|---|
| H1-direct: do the 40 graph features add to the transaction-level XGBoost? | **NOT ESTABLISHED** | primary B − A = +0.0046; mean seed difference -0.0005 against the larger SD 0.0034; B won 2 of 3 seeds |
| H2: do learned graph representations (E, Temporal GraphSAGE) beat B? | **HANDCRAFTED AT LEAST AS GOOD (H2 contradicted)** | E seeds 0.4888–0.4956 (mean 0.4927) versus B_sub 0.5572–0.5636 (mean 0.5611) |
| H2, secondary reading with C (GraphSAGE) | HANDCRAFTED AT LEAST AS GOOD (H2 contradicted) | C seeds 0.4515–0.4827 versus B_sub 0.5572–0.5636 |
| BA6 pipeline control (A re-fit equals the recorded A) | PASS | re-fit 0.5395 versus recorded 0.5395 (|diff| 0.0000, bound 0.005) |

Per-seed pooled PR-AUC: A_sub 0.5590, 0.5633, 0.5625; B_sub 0.5624, 0.5636, 0.5572; E (recorded) 0.4938, 0.4888, 0.4956; C (recorded) 0.4515, 0.4569, 0.4827 (seeds 42, 123, 2024).

## How to read the verdicts

* The H2 rule compares `B_sub` with the recorded neural runs. The arm without graph features, `A_sub` (seeds 0.5590, 0.5633, 0.5625), is also above every E seed (lowest A_sub 0.5590 versus highest E 0.4956). So the H2 outcome mostly reflects the strength of the tuned tabular XGBoost, not a specific benefit of the 40 graph features; the graph features' own contribution is the H1-direct row.
* The subsampling recipe (`*_sub`) scores higher than the comparison-A recipe for both feature sets. Arms that stopped within ten trees of the 1,000-tree cap (A_sub_s123, A_sub_s42, B_sub_s123, B_sub_s42) were still improving, so the cap may have limited them; the cap belongs to the unchanged comparison-A recipe and was not raised.

## Table (pooled VAL; cells with several runs show the mean, SD in the CSV)

| Cell | Model | Runs | PR-AUC | ROC-AUC | PR-AUC cold-start | PR-AUC warm | PR-AUC no-edge | P@1% | R@1% |
|---|---|---|---|---|---|---|---|---|---|
| A | logistic_regression | 1 | 0.4126 | 0.8520 | 0.2972 | 0.1671 | 0.7067 | 0.7539 | 0.2231 |
| A | random_forest | 1 | 0.4543 | 0.8792 | 0.3807 | 0.2046 | 0.7366 | 0.8196 | 0.2426 |
| A | xgboost | 1 | 0.5395 | 0.9050 | 0.4064 | 0.3496 | 0.7905 | 0.8937 | 0.2645 |
| A | XGBoost, comparison-A recipe, re-fit in the B pipeline (pipeline control) | 1 | 0.5395 | 0.9050 | 0.4064 | 0.3496 | 0.7905 | 0.8937 | 0.2645 |
| A | XGBoost, A features, subsample 0.8 / colsample 0.8 (3 seeds) | 3 | 0.5616 ± 0.0023 | 0.9141 | 0.4067 | 0.4073 | 0.7984 | 0.9060 | 0.2682 |
| B | XGBoost, A features + 40 graph features (B, A recipe) | 1 | 0.5440 | 0.9062 | 0.4163 | 0.3554 | 0.7947 | 0.8913 | 0.2638 |
| B | XGBoost, B features, subsample 0.8 / colsample 0.8 (3 seeds) | 3 | 0.5611 ± 0.0034 | 0.9166 | 0.4131 | 0.3987 | 0.8006 | 0.8965 | 0.2653 |
| C | GraphSAGE (Phase 6A) | 3 | 0.4637 ± 0.0167 | 0.8616 | 0.3417 | 0.2611 | 0.7182 | 0.8092 | 0.2395 |
| C-control | MLP control, no relations (Phase 6A extended) | 1 | 0.4167 | 0.8547 | 0.3007 | 0.1758 | 0.7029 | 0.7575 | 0.2242 |
| D | GAT (Phase 6B) | 3 | 0.3320 ± 0.0052 | 0.8653 | 0.3161 | 0.2639 | 0.4033 | 0.5791 | 0.1714 |
| E | Temporal GraphSAGE (GRU memory, Phase 7) | 3 | 0.4927 ± 0.0035 | 0.8667 | 0.3562 | 0.3213 | 0.7271 | 0.8423 | 0.2493 |
| E-control | Static GraphSAGE, schedule-matched (Phase 7A) | 3 | 0.4792 ± 0.0114 | 0.8554 | 0.3288 | 0.3253 | 0.7132 | 0.8284 | 0.2452 |
| E-control | Temporal, memory ablated (Phase 7A) | 3 | 0.4790 ± 0.0076 | 0.8571 | 0.3305 | 0.2951 | 0.7295 | 0.8248 | 0.2441 |

F rows (distinct evaluation basis) and the historical-feature reference line are in the Phase 12 table and are not repeated here.

## Relationship-level and temporal-feature ablations of B (descriptive; A recipe, seed 42)

Seed noise (sample SD of `B_sub` PR-AUC): 0.0034. B_full PR-AUC: 0.5440.

| Ablation | PR-AUC | Change vs B_full | Larger than seed noise |
|---|---|---|---|
| B_drop_pa | 0.5402 | -0.0038 | yes |
| B_drop_card | 0.5443 | +0.0003 | no |
| B_drop_net | 0.5375 | -0.0066 | yes |
| B_drop_dev | 0.5453 | +0.0012 | no |
| B_drop_time_windows | 0.5456 | +0.0016 | no |

Ablation changes are single fits at one seed: a change inside the noise band cannot be told from zero. These ablations apply to the handcrafted-feature model only; relationship-level ablations of the neural graph models were not run (each neural fit takes far longer and the GAT never converged).

## Comparability (labels from the pre-registration, section 8)

* B versus A: comparable (same rows, labels, metrics and recipe; only the features differ).
* B versus C, D, E: comparable on the evaluation basis only; attribution is not supported (different encodings and training regimes; all tuned on VAL).
* B versus F: not comparable.

## Limits

* One handcrafted feature set (40 features) fixed before results; a null or negative result does not speak for other handcrafted sets.
* The B features use same-entity history with strictly earlier timestamps; the neural graph models see same-day transactions at day granularity.
* Window features include the boundary second (a prior transaction exactly 86,400 s or 604,800 s earlier counts), an implementation choice made explicit after the pre-registration text said "preceding".
* Everything is tuned on VAL; no held-out claim.
* XGBoost is deterministic without subsampling, so seed variation exists only in the `*_sub` arms; those arms use a recipe that differs from comparison A.

