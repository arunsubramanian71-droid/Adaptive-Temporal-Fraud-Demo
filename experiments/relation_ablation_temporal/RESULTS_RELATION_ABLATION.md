# Relationship-level ablation of Temporal GraphSAGE: results

**Validation period (days 122-151); recorded Phase 7 full model as reference; pre-registration `PREREGISTRATION_RELATION_ABLATION.md` (hashed before any arm). No held-out data was read.**

| Relation removed | Pooled PR-AUC without it (seeds 42, 123, 2024) | Full model (recorded) | Mean difference (full - without) | Seeds where full is higher | Larger seed SD | Verdict |
|---|---|---|---|---|---|---|
| charged_to | 0.4556, 0.4600, 0.4700 | 0.4938, 0.4888, 0.4956 | +0.0309 | 3 of 3 | 0.0074 | **charged_to contributes** |
| used_card | 0.4976, 0.4989, 0.4930 | 0.4938, 0.4888, 0.4956 | -0.0038 | 1 of 3 | 0.0035 | **removing used_card does not hurt** |
| via_network | 0.4952, 0.4994, 0.4995 | 0.4938, 0.4888, 0.4956 | -0.0053 | 0 of 3 | 0.0035 | **removing via_network does not hurt** |
| from_device | 0.4864, 0.4852, 0.4846 | 0.4938, 0.4888, 0.4956 | +0.0073 | 3 of 3 | 0.0035 | **from_device contributes** |

Run-to-run noise reference: retraining the full model with seed 42 by the same code gives pooled PR-AUC 0.4995 against the recorded 0.4938 (absolute difference 0.0058); differences of that size are not effects.

Segment PR-AUC (means over seeds) are in `results.json`. Three seeds sharing the same data are not independent replications; validation was used for early stopping and thresholds, as in Phase 7.
