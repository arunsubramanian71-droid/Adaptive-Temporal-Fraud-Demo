# Graph-level evidence for Temporal GraphSAGE (seed 42): results

**Validation-period (days 122-151) counterfactuals on the 300 pre-registered Phase 11 transactions for a model tuned on that period. Descriptive; not evidence of fraud, causation or coordination; no held-out data was read; no label was used.**

Protocol: `PREREGISTRATION_NEURAL_EVIDENCE.md` (hashed before the code existed). Counterfactuals: removal of the transaction's own link to one entity, and reset of that entity's memory to the never-seen condition, each on a clone of the state at the start of the day.

## Registered acceptance outcome

| Criterion | Result |
|---|---|
| NE1 reproduction (max probability difference to the recorded predictions 6.96e-08, bound 1e-4) | PASS |
| NE2 selection (300 Phase 11 ids, hash equal) | PASS |
| NE3 label-free | PASS |
| NE5 sanity (a: reset of never-seen entities exactly zero; b: reset of seen entities changes the score in a majority; c: link removal changes the score in a majority) | **FAIL** |
| NE7 repeatability (second run, max difference in effects 3.81e-06, bound 1e-5) | PASS |

**NE5 failed as registered, in part (a).** Of 102 resets of never-seen entities, 13 were not exactly zero; the largest absolute value is 9.54e-07. Part (b) passed (share changed 1.00) and part (c) passed (share changed 1.00). The registered criterion demanded bit-exact zeros but GPU kernels are not bit-reproducible, which NE7 itself measured; the registration did not allow for that. The criterion was not changed. Post hoc diagnostic (labelled as such): the nonzero values are float32 rounding size, six orders of magnitude below the median effect for seen entities (0.45) and below the smallest one (1.78e-04); the same invariant holds exactly on CPU in the unit test of the counterfactual functions.

## Descriptive results

| Relation | Share of transactions with the link | Mean absolute logit change on link removal | Mean absolute change on memory reset (entity seen before) | Share with entity new | Share where it is the largest removal effect |
|---|---|---|---|---|---|
| pseudo-account | 0.520 | 3.158 | 2.398 | 0.571 | 0.410 |
| card | 1.000 | 1.058 | 0.659 | 0.017 | 0.227 |
| network proxy | 0.597 | 1.839 | 0.283 | 0.011 | 0.350 |
| device | 0.117 | 1.548 | 0.616 | 0.171 | 0.013 |

300 transactions; 670 link removals and 670 memory resets performed.

## Limits

One checkpoint, 300 transactions, validation period; a counterfactual on one transaction's link is not a statement about the entity or about fraud; transactions that share an entity on the same day influence each other, so effects include that interaction; effects across relations are not additive. GraphSAGE and GAT were not examined (GAT never converged; attention weights are not used).
