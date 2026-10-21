# Comparison B (tabular + handcrafted graph features): pre-registration

Dated 2026-10-20. Written and hashed (`PREREGISTRATION_B.sha256`) BEFORE any B feature code, test, fit or output exists. Authority: the user's master instruction of 2026-10-20 to complete the original project, whose brief lists cell B ("Tabular + graph features") and roadmap Level 4 ("XGBoost + leakage-safe handcrafted graph features"). This record is a design choice made by me within that authority; nothing in it was tuned on a result.

Standing statement: **every number produced under this record is a validation-period (VAL, days 122-151) number. The recipe, as for comparison A, uses VAL for early stopping and the best-F1 threshold. No TEST data is read; no held-out claim follows.**

## 1. Questions resolved

* **H1-direct:** does adding handcrafted, label-free, strictly backward-looking graph features to the existing XGBoost recipe improve VAL ranking beyond the same recipe on transaction-level features only (B versus A)?
* **H2:** do learned graph representations (Temporal GraphSAGE, cell E; GraphSAGE, cell C) beat the handcrafted graph features (B)?
* **Which information contributes:** leave-one-relation-out and no-temporal-window ablations of B (the brief's relationship-level ablations "where computationally feasible"; feasible here because XGBoost fits in minutes, not for the graph neural models).

## 2. Inputs (the only files read)

* `data/features/train_features.parquet`, `data/features/val_features.parquet` (TRAIN days 1-121, VAL days 122-151).
* `experiments/phase9_coordinated_fraud/cache/real_le151.npz`: per-transaction entity ids for the four graph relations, already truncated to days <= 151 by the Phase 9 loader. Its row order is identical to train+val features concatenated (checked at run time by `TransactionID` equality).
* The existing fitted tabular encoder code `src/models/tree_features.py` (refit on TRAIN only, as comparison A did).
* Reference population and strict segments: `experiments/temporal_graphsage/seed_42/predictions_val.parquet` (columns `TransactionID`, `isFraud`, `day`, `segment`), exactly as Phase 12 (P2).
* Per-seed comparison values for E and C, and the recorded A: `experiments/phase12_validation_evidence_table/outputs/evidence_runs.csv`.
* Never read: any TEST-named file, `test_features.parquet`, the raw competition files, checkpoints, `experiments/gat_300epoch/`.

## 3. Entities and timing

Entities are the four Phase 4 relations: `pa` (pseudo-account), `card`, `net` (netproxy), `dev` (device fingerprint). An entity id of -1 means no edge. For a transaction with time `t = TransactionDT` (seconds), a feature may use only transactions of the same entity with `TransactionDT` strictly less than `t`. This is stricter than the graph neural models, which see same-day transactions at day granularity. No label, score, prediction or any quantity derived from a label enters any feature. Features are computed once over the TRAIN+VAL timeline (days 1-151); for each row they depend only on earlier rows.

## 4. The 40 features (fixed now; column names are final)

For each relation `r` in (`pa`, `card`, `net`, `dev`), and `s` ranging over the other three relations:

| Feature | Definition (prior = same entity, strictly earlier time) | Value when the row has no edge for `r` |
|---|---|---|
| `g_{r}_has_edge` | 1 if the row has an edge for `r` | 0 |
| `g_{r}_prior_n` | number of prior transactions | NaN |
| `g_{r}_prior_n_1d` | prior transactions in the preceding 86,400 s | NaN |
| `g_{r}_prior_n_7d` | prior transactions in the preceding 604,800 s | NaN |
| `g_{r}_secs_since_last` | `t` minus the time of the latest prior transaction (NaN if none) | NaN |
| `g_{r}_prior_amt_logmean` | mean of `log1p(TransactionAmt)` over prior transactions (NaN if none) | NaN |
| `g_{r}_amt_dev` | `log1p(TransactionAmt)` of this row minus `g_{r}_prior_amt_logmean` | NaN |
| `g_{r}_prior_distinct_{s}` | number of distinct relation-`s` entities among prior transactions of the entity (rows with no `s` edge are not counted) | NaN |

That is 4 x (1 + 6) + 4 x 3 = 40 features. Missing values stay NaN (XGBoost handles them natively, as in comparison A). B uses the 470 comparison-A features plus these 40 (510 columns).

Ablation groups (fixed now):
* `drop_pa`, `drop_card`, `drop_net`, `drop_dev`: remove every `g_` feature whose name contains the relation tag (`pa`, `card`, `net`, `dev`) as the entity or as the `{s}` of a distinct-count feature.
* `drop_time_windows`: remove `g_*_prior_n_1d`, `g_*_prior_n_7d`, `g_*_secs_since_last` (12 features).

## 5. Model recipe and arms (no search)

The recipe is exactly comparison A's XGBoost (`scripts/train_xgboost.py`): `n_estimators=1000`, `max_depth=6`, `learning_rate=0.05`, `tree_method="hist"`, `eval_metric="aucpr"`, `scale_pos_weight` = TRAIN negatives / positives, `early_stopping_rounds=50` on VAL, CPU, `n_jobs=-1`. Threshold for operating-point metrics: best F1 on VAL (as A). Nothing is tuned.

Because XGBoost without subsampling does not depend on its seed, seed variation requires subsampling. A secondary arm family therefore adds `subsample=0.8`, `colsample_bytree=0.8` with seeds 42, 123, 2024, for both feature sets.

| Arm | Features | Recipe | Seeds | Fits |
|---|---|---|---|---|
| `A_refit` | 470 | A recipe | 42 | 1 (pipeline control) |
| `B_full` | 510 | A recipe | 42 | 1 |
| `A_sub` | 470 | A recipe + subsample/colsample 0.8 | 42, 123, 2024 | 3 |
| `B_sub` | 510 | same | 42, 123, 2024 | 3 |
| `B_drop_pa`, `B_drop_card`, `B_drop_net`, `B_drop_dev`, `B_drop_time_windows` | 510 minus group | A recipe | 42 | 5 |

## 6. Decision rules (applied mechanically by the run script from recorded values; fixed now)

Primary metric: pooled VAL PR-AUC. Sample SD uses ddof = 1 over the three seeds.

**H1-direct (B versus A):**
* SUPPORTED iff (i) PR-AUC(`B_full`) > PR-AUC(`A_refit`); (ii) mean over seeds of [PR-AUC(`B_sub`, s) - PR-AUC(`A_sub`, s)] > max(SD(`B_sub`), SD(`A_sub`)); (iii) `B_sub` > `A_sub` in at least 2 of the 3 seeds.
* CONTRADICTED iff the mean difference in (ii) is < -max SD and `B_sub` < `A_sub` in at least 2 seeds.
* Otherwise NOT ESTABLISHED.

**H2 (learned versus handcrafted graph information):**
* LEARNED SUPPORTED iff min over the three E seeds > max over the three `B_sub` seeds.
* HANDCRAFTED AT LEAST AS GOOD (H2 contradicted) iff max over the three E seeds < min over the three `B_sub` seeds.
* Otherwise INCONCLUSIVE.
* The same rule is reported for C (GraphSAGE) as a secondary reading. E and C values are the recorded per-seed values in `evidence_runs.csv`.

**Ablations:** descriptive. Each change in PR-AUC is reported against `B_full`; a change is called larger than seed noise only if its absolute value exceeds max(SD(`B_sub`)).

Cold-start (strict segment) PR-AUC, ROC-AUC and Precision/Recall@K are reported alongside and are descriptive; they carry no decision rule.

## 7. Acceptance criteria for the B work (fixed now; independent of the hypothesis outcomes)

* **BA1** This record and its sidecar exist before any B code or output; its hash is recorded in the B manifest.
* **BA2** The feature function takes no label argument (static check). Future-invariance: changing every row after a cutoff leaves earlier rows' features bit-identical (tests plus a real-data truncation check at cutoff days 60, 100 and 121).
* **BA3** On random small inputs the function equals an independent brute-force implementation exactly, including ties in `TransactionDT`, missing edges and window edges.
* **BA4** Only the files in section 2 are opened (logged; any TEST-named path or checkpoint is refused); the entity-id cache agrees with the feature files (`TransactionID` order, and the card id partition equals the `card1` partition).
* **BA5** Every arm scores exactly the 83,655 reference ids with identical labels; strict segments come from the reference.
* **BA6** Pipeline control: `A_refit` pooled PR-AUC is within 0.005 of the recorded 0.5395. If it is not, the mismatch is reported and the same-pipeline `A_refit` is used for B versus A; the recorded value is never replaced.
* **BA7** Metrics use the unmodified `src/models/metrics.py` through the Phase 12 `evaluate_frame`; no threshold function other than best-F1-on-VAL (as A) is used.
* **BA8** The decision rules in section 6 are applied exactly as written and the outcome is reported whatever it is.
* **BA9** Unit tests pass and a mutation-sensitivity check detects every injected defect; Phase 9 wording lint is clean; outputs have SHA-256 sidecars and an integrity record.

## 8. Comparability labels (for the extended table, new files only)

* B versus A: **comparable** (same rows, labels, metrics and recipe; only the features differ); attribution to graph structure is limited to the 40 named features.
* B versus C, D or E: **comparable on the evaluation basis; attribution not supported** (different encodings and training regimes; all tuned on VAL).
* B versus F: **not comparable** (the F basis differs, Phase 12 section 5).

## 9. Out of scope and caveats

No TEST evaluation, no new neural training, no historical or blocklist (label-derived) features in B, no change to any Phase 1-12 artifact (the Phase 12 table is extended only through new files). B uses the same VAL for early stopping and threshold as A, so B and A share the optimism of VAL tuning. The 40 features are one defensible handcrafted set chosen before results; a different set could give a different answer, and a null result for these 40 features is not a statement about all handcrafted graph features.
