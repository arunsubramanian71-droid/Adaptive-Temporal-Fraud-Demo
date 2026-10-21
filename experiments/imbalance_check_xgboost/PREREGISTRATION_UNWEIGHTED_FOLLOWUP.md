# Unweighted follow-up: pre-registration

Dated 2026-10-20; written and hashed (`PREREGISTRATION_UNWEIGHTED_FOLLOWUP.sha256`) before any arm of this follow-up exists. It extends `PREREGISTRATION_IMBALANCE.md`.

**Why.** The registered class-imbalance check (XGBoost, A features, one seed, `scale_pos_weight` none / square root / full) returned WEIGHTING HURTS: with no class weighting pooled VAL PR-AUC was 0.5948 against 0.5395 for the recorded comparison-A setting (this result existed when this follow-up was written; the study's weighted-recipe conclusions were already recorded). Every model in the project, including the neural graph models and comparison B, used full class weighting. The brief says every imbalance technique must be evaluated experimentally, and this result puts the weighted-recipe comparisons (H1-direct, H2, the XGBoost-versus-neural gap) at risk of being artefacts of a handicapped recipe. This follow-up re-applies the already registered comparisons under no class weighting. It does **not** replace any recorded weighted result: weighted and unweighted results are reported side by side and kept separate.

Standing statement: VAL-period results (days 122-151), VAL used for early stopping and the best-F1 threshold as everywhere else; no TEST data is read.

## Arms (fixed now; nothing else is added after seeing results)

| Arm | Model | Class weighting | Recipe | Seeds |
|---|---|---|---|---|
| `A_none` | XGBoost, 470 A features | none | comparison-A recipe | 42 (already recorded as arm `none` of the imbalance check) |
| `B_none` | XGBoost, 510 B features | none | comparison-A recipe | 42 |
| `A_none_sub`, `B_none_sub` | same, subsample/colsample 0.8 | none | `*_sub` recipe of comparison B | 42, 123, 2024 |
| `C_none` | GraphSAGE (Phase 6A architecture and recipe) | `pos_weight` removed (plain binary cross-entropy) | otherwise identical to `train_graphsage_robustness.py` | 42, 123, 2024 |
| `E_none` | Temporal GraphSAGE (Phase 7) | `pos_weight` removed | otherwise identical to `train_temporal_graphsage.py` | 42, 123, 2024 |

GAT is not re-run: it never converged and a 150-epoch fit needs about 12 GiB of GPU memory; this is stated as a limit, not a result. Neural training scripts are reused unmodified; only the loss weighting is overridden by a wrapper, and the neural outputs go to new folders (`experiments/unweighted_graphsage/`, `experiments/unweighted_temporal_graphsage/`).

## Decision rules (applied mechanically, as in `PREREGISTRATION_B.md`; sample SD with ddof = 1)

1. **Weighting effect per family**, pooled VAL PR-AUC, three seeds each:
   * for XGBoost with subsampling (`A_none_sub` versus the recorded `A_sub`, `B_none_sub` versus `B_sub`), GraphSAGE (`C_none` versus recorded Phase 6A) and Temporal GraphSAGE (`E_none` versus recorded Phase 7): **UNWEIGHTED BETTER** if the lowest unweighted seed exceeds the highest weighted seed; **WEIGHTED BETTER** if the highest unweighted seed is below the lowest weighted seed; otherwise **NO CLEAR DIFFERENCE**.
2. **H1-direct under no weighting:** the rule of `PREREGISTRATION_B.md` section 6 with `B_none` for `B_full`, `A_none` for `A_refit`, `B_none_sub` for `B_sub`, `A_none_sub` for `A_sub`.
3. **H2 under no weighting:** the complete-separation rule with `E_none` (and secondarily `C_none`) as the learned arm and `B_none_sub` as the handcrafted arm.
4. **Does the tabular-versus-neural ordering change?** Reported as yes/no: whether the lowest of `A_none_sub` and `B_none_sub` seeds exceeds the highest of `C_none` and `E_none` seeds.
The weighted-recipe verdicts already recorded stay as they are; this follow-up adds a second reading under a different recipe.

## Acceptance criteria

* **UA1** This record precedes every arm; its hash is in the manifests.
* **UA2** XGBoost arms read only the files of `PREREGISTRATION_B.md` section 2; neural arms use the existing training scripts' own inputs and only days <= 151 are used for training and evaluation; no TEST-named file is opened and no held-out day is scored.
* **UA3** Each arm scores exactly the 83,655 reference ids with identical labels.
* **UA4** Rules 1-4 are applied as written and reported whatever they return; outputs have SHA-256 sidecars; wording lint clean.
* **UA5** Neural unweighted runs record in their `config.json` that the effective positive-class weight is 1.0.

## Limits stated in advance

Three seeds sharing the same data are not independent replications; neural models are run once per seed at the recorded epoch budget; early stopping on VAL means all arms are optimistic in the same way; a different weighting (for example square-root) for the neural models is not tried; GAT is not covered.
