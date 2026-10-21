# Unweighted follow-up: results

**Validation-period (VAL, days 122-151) results; VAL used for early stopping and the best-F1 threshold; no TEST data was read. Weighted and unweighted results are separate readings under different recipes; the recorded weighted results are unchanged.**

Protocol: `PREREGISTRATION_UNWEIGHTED_FOLLOWUP.md`. Rules are applied mechanically from recorded values.

## Rule 1: effect of class weighting (pooled PR-AUC, three seeds each)

| Family | Unweighted seeds | Weighted seeds (recorded) | Verdict |
|---|---|---|---|
| XGBoost A features (sub) | 0.6080, 0.6081, 0.6033 (mean 0.6065) | 0.5590, 0.5633, 0.5625 (mean 0.5616) | **UNWEIGHTED BETTER** |
| XGBoost B features (sub) | 0.6062, 0.6106, 0.6019 (mean 0.6062) | 0.5624, 0.5636, 0.5572 (mean 0.5611) | **UNWEIGHTED BETTER** |
| GraphSAGE | 0.4587, 0.4598, 0.4706 (mean 0.4630) | 0.4515, 0.4569, 0.4827 (mean 0.4637) | **NO CLEAR DIFFERENCE** |
| Temporal GraphSAGE | 0.4996, 0.4978, 0.5020 (mean 0.4998) | 0.4938, 0.4888, 0.4956 (mean 0.4927) | **UNWEIGHTED BETTER** |

## Rules 2-4: the registered comparisons under no class weighting

| Question | Verdict under no weighting | Basis |
|---|---|---|
| H1-direct (B versus A) | **NOT ESTABLISHED** | primary B − A = +0.0043; mean seed difference -0.0003 against the larger SD 0.0044; B won 1 of 3 seeds |
| H2 (learned E versus handcrafted B_none_sub) | **HANDCRAFTED AT LEAST AS GOOD (H2 contradicted)** | E_none 0.4978–0.5020; B_none_sub 0.6019–0.6106 |
| H2 secondary (C_none) | HANDCRAFTED AT LEAST AS GOOD (H2 contradicted) | C_none 0.4587–0.4706 |
| Tabular above every neural seed? | **yes** | lowest tabular unweighted seed 0.6019; highest neural unweighted seed 0.5020 |

Single-fit values: A_none (seed 42, comparison-A recipe) 0.5948; B_none 0.5991.

Scope: GAT was not re-run (never converged; about 12 GiB of GPU memory). Three seeds sharing the data are not independent replications. These readings sit beside, and do not replace, the recorded weighted results.
