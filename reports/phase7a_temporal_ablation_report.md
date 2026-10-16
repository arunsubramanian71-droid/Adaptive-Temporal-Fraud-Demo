# Phase 7A — Temporal Mechanism Ablation

Status: **Phase 7A COMPLETE.** Arms B and C trained (3 seeds each); leakage/validity checks complete for both
(arm C on CPU and GPU, arm B on GPU); regression checks complete; comparison against the fixed Phase 6A (A)
and Phase 7 (D) references complete. TEST was never loaded. No Phase 8 work was performed.

This report is written entirely from artifacts already on disk (`experiments/temporal_ablation/`,
`experiments/graphsage_robust/`, `experiments/temporal_graphsage/`); nothing in Sections E–N was recomputed
or altered to produce this document, and no additional experiment was run beyond what Section B specifies.

---

## A. Objective

Phase 7 found Temporal GraphSAGE (D) beat static GraphSAGE (A) by +0.029 pooled PR-AUC (0.4927 vs 0.4637).
That report itself flagged an unresolved confound: D trains with truncated BPTT over 7-day windows (18
optimizer steps per epoch, gradient-norm clipping 1.0) while A trains with 1 full-batch step per epoch, no
clipping. The +0.029 gap could reflect the recurrent memory, the different optimization schedule, or both.

Phase 7A isolates these by adding two controlled arms:
- **B** — static GraphSAGE, same architecture as A, but trained on D's schedule (18 steps/epoch, clip 1.0).
- **C** — D's full causal streaming pipeline and training regime, with the GRU memory replaced by an
  order-agnostic cumulative-mean memory.

Comparing A→B isolates the schedule effect; B→C isolates the causal-streaming/architecture effect; C→D
isolates the value of the *learned, order-sensitive* recurrence itself, holding everything else fixed.

## B. Experimental controls (as pre-registered)

Full specification: [`experiments/temporal_ablation/DESIGN_BEFORE_EXECUTION.md`](../experiments/temporal_ablation/DESIGN_BEFORE_EXECUTION.md),
written and committed to disk *before* any Phase 7A training ran. Summary:

| | Model | Params | Training regime | Source |
|---|---|---|---|---|
| A | Static GraphSAGE | 162,817 | 1 full-batch step/epoch, no clipping | Phase 6A (fixed reference, not re-run) |
| B | Static GraphSAGE, schedule-matched | 162,817 | 18 windowed steps/epoch, clip 1.0 | Phase 7A (new) |
| C | Temporal, memory ablated (cumulative-mean) | 179,201 | Phase 7 regime: 18 steps/epoch, clip 1.0 | Phase 7A (new) |
| D | Temporal GraphSAGE (GRU memory) | 197,889 | 18 windowed steps/epoch, clip 1.0 | Phase 7 (fixed reference, not re-run) |

Corrected figure from Phase 7: the report said "~17 optimizer steps per epoch"; the exact count is **18**
(seventeen 7-day windows covering days 1–119, plus one 2-day window for days 120–121). Verified directly
from the training code and confirmed in B and C's logs (`assert steps_per_epoch == 18`, both passed).

Seeds 42/123/2024, TRAIN days 1–121, VALIDATION days 122–151, TEST (152–182) sealed and never loaded in any
arm. All four arms use the same Phase 4 graph, features, TRAIN-only StandardScaler, BCE loss with TRAIN
pos_weight, Adam (lr 1e-3, weight decay 1e-5), max_epochs 150 / patience 20, rolling per-day evaluation, and
the strict 3-way segmentation (`has_edge_warm` / `has_edge_cold` / `no_pseudo_account_edge`). Thresholds were
selected on VALIDATION only, per arm/seed.

Pre-registered reading rules (fixed before training): "temporal gain persists vs X" requires D beating X in
pooled PR-AUC in 3/3 seeds **and** the mean difference exceeding the larger of the two arms' seed SDs.
Pairwise p-values (n=3, paired t-test and Welch's t-test) are reported as descriptive only, uncorrected; no
claim of significance is made from them.

## C. Architecture

**Arm B** is exactly `HeteroSAGE` from Phase 5/6A ([`src/models/graphsage.py`](../src/models/graphsage.py)) —
per-type projection → 2× HeteroConv(SAGEConv, sum) → dropout 0.3 → linear head — unchanged. No architectural
difference from A.

**Arm C** (`TemporalGraphSAGEMeanMemory`, [`src/models/temporal_meanmem.py`](../src/models/temporal_meanmem.py))
keeps D's day-by-day streaming structure and layer-2 read-out unchanged, replacing only the memory:

```
u_t      = W_r hx_t                                     per-transaction message (Linear 64->64, like SAGEConv's lin_l)
Sum_e(d) = Sum_e(d-1) + sum_{t in day d, t~e} u_t        running sum   } state = [Sum | Cnt] per entity
Cnt_e(d) = Cnt_e(d-1) + count(t in day d, t~e)           running count }
mbar_e   = Sum_e(d) / Cnt_e(d)                           mean over ALL messages from every txn <= d
s_e      = MLP_e(Dropout(ReLU(mbar_e)))                  MLP_e: Linear(64,192) -> ReLU -> Linear(192,64)
```

Layer 2 (entity→transaction read-out) and the head are bit-for-bit the same as D. Removed relative to D: the
GRU cell, the learned initial state `h0`, the elapsed-time input, and state-as-root in layer 1. The memory is
a **permutation-invariant function of the multiset of past messages** — it does not depend on the order or
timing of past events, only on which events occurred. Parameter count: 179,201 (vs D's 197,889, −9.4%; the
difference is D's SAGEConv root weights (16,384), `h0` (256), and GRU-vs-MLP parameter difference (2,048) —
disclosed as a residual capacity gap that cannot be fully excluded as a contributing factor, see Section P).

## D. Training-regime matching

Verified directly from B and C's training logs and a dataset-level check in the validation suite:

- **`training_windows_train_only_disjoint_chronological`** (PASS): 18 windows, chronologically ordered,
  disjoint, covering exactly the 417,559 TRAIN transactions and no others.
- **`training_graph_has_no_edges_after_train_end`** (PASS): the TRAIN-masked graph B trains on has max
  `edge_time` = 121.
- Both B and C use gradient-norm clipping at 1.0 (`assert` in both scripts confirms 18 steps/epoch actually
  executed). Clipping was **not** present in Phase 6A (A) — this is a deliberate, disclosed deviation made to
  match D, not a tuning choice.
- Everything else (architecture, optimizer, lr, weight decay, dropout, epoch cap, patience, evaluation
  protocol, segmentation, threshold selection) is identical across A/B/C/D as specified in Section B.

## E. Three-seed results

Pooled PR-AUC, per seed (VALIDATION, rolling per-day evaluation):

| Arm | seed 42 | seed 123 | seed 2024 | mean | std | min | max |
|---|---|---|---|---|---|---|---|
| A (static, 1 step/epoch) | 0.4515 | 0.4569 | 0.4827 | 0.4637 | 0.0167 | 0.4515 | 0.4827 |
| B (static, 18 steps/epoch, clip 1.0) | 0.4779 | 0.4912 | 0.4686 | 0.4792 | 0.0114 | 0.4686 | 0.4912 |
| C (temporal, mean memory) | 0.4711 | 0.4796 | 0.4863 | 0.4790 | 0.0076 | 0.4711 | 0.4863 |
| D (temporal, GRU memory) | 0.4938 | 0.4888 | 0.4956 | 0.4927 | 0.0035 | 0.4888 | 0.4956 |

Full per-seed metrics (ROC-AUC, precision/recall/F1, P@K/R@K, cold/warm/no-edge PR-AUC) are in
`experiments/temporal_ablation/comparison.json`; segment breakdowns are in Sections H–J below.

## F. Convergence

| Arm | mean epochs run | mean best epoch | all seeds early-stopped? |
|---|---|---|---|
| A | 130.3 | 114.0 | 2/3 (seed 2024 hit max_epochs=150, not converged) |
| B | 80.3 | 60.3 | 3/3 |
| C | 79.7 | 59.7 | 3/3 |
| D | 73.7 | 53.7 | 3/3 |

B, C and D all converge (early-stop) well before 150 epochs on the 18-steps/epoch schedule; A's higher
epoch count reflects its 1-step/epoch schedule needing more epochs to reach an equivalent number of gradient
updates. See `experiments/temporal_ablation/comparison_plot.png` and `.../static_schedule_matched|memory_ablated_temporal/convergence_curves.png` for the training-loss and validation-PR-AUC curves.

## G. Pooled (main comparison)

| Comparison | mean Δ (pooled PR-AUC) | per-seed Δ | wins/3 | paired p (desc.) | Welch p (desc.) | Δ > larger SD? |
|---|---|---|---|---|---|---|
| D − B | +0.0135 | [+0.0159, −0.0024, +0.0270] | 2/3 | 0.256 | 0.168 | Yes |
| D − C | +0.0137 | [+0.0226, +0.0092, +0.0093] | **3/3** | 0.091 | 0.070 | Yes |
| C − B | −0.0002 | [−0.0068, −0.0116, +0.0177] | 1/3 | 0.981 | 0.977 | No |
| B − A | +0.0155 | [+0.0264, +0.0343, −0.0141] | 2/3 | 0.409 | 0.262 | No |
| D − A | +0.0290 | [+0.0423, +0.0319, +0.0129] | 3/3 | 0.078 | 0.089 | Yes |
| C − A | +0.0153 | [+0.0197, +0.0227, +0.0035] | 3/3 | 0.124 | 0.251 | No |

Applying the pre-registered reading rule literally:

- **D vs B ("does the temporal gain persist against a schedule-matched static baseline?"): rule NOT met.**
  D beats B in only 2/3 seeds (seed 123 goes the other way, D 0.4888 < B 0.4912), even though the mean
  difference nominally exceeds the larger seed SD. Per the rule fixed in advance, this is reported as
  **not established** — the original Phase 7 static-vs-temporal comparison substantially overstated the
  temporal advantage: roughly half of the +0.029 pooled PR-AUC gap over A (0.0155 of 0.0290) is attributable
  to the training-schedule change alone (B − A), not to anything temporal.
- **D vs C ("does the learned recurrent memory add signal beyond having access to the same history?"): rule
  MET.** D beats C in 3/3 seeds, by a fairly consistent +0.009 to +0.023, exceeding both arms' seed SDs.
- **C vs B ("does the causal streaming pipeline/architecture matter on its own, without recurrence?"):**
  essentially a wash (mean Δ −0.0002, 1/3 wins) — the memory-ablated temporal pipeline performs the same as
  schedule-matched static once the schedule is equalized.

## H. Cold-start (`has_edge_cold`)

| Arm | mean | std |
|---|---|---|
| A | 0.3417 | 0.0113 |
| B | 0.3288 | 0.0050 |
| C | 0.3305 | 0.0065 |
| D | 0.3562 | 0.0071 |

D − C: +0.0257, 3/3 wins, exceeds SD — the recurrent memory's cold-start advantage over the mean-memory
control is the cleanest positive result in this ablation (also the smallest descriptive p-values: paired
0.081, Welch 0.010). D − B: +0.0274, 3/3 wins, exceeds SD — also robust. B and C are statistically
indistinguishable from each other on cold-start (mean Δ +0.0017). Interestingly B is *below* A here (−0.0129);
schedule-matching alone did not help cold-start static GraphSAGE.

## I. Warm-start (`has_edge_warm`)

| Arm | mean | std |
|---|---|---|
| A | 0.2611 | 0.0440 |
| B | 0.3253 | 0.0445 |
| C | 0.2951 | 0.0279 |
| D | 0.3213 | 0.0111 |

Warm-start is the noisiest segment (large seed SDs, especially A and B). D − B is essentially flat (mean
−0.0040, 1/3 wins) — on warm entities, schedule-matched static is not distinguishably worse than the full
temporal model. D − C is positive (+0.0262) but does not clear the "exceeds larger SD" bar because C's own
seed spread is large. No warm-start comparison in this table meets the pre-registered "gain persists" bar
except B/D/C all beating A directionally (though B−A and C−A also fail the SD bar due to warm-start's
volatility).

## J. No-edge (`no_pseudo_account_edge`)

| Arm | mean | std |
|---|---|---|
| A | 0.7182 | 0.0125 |
| B | 0.7132 | 0.0129 |
| C | 0.7295 | 0.0130 |
| D | 0.7271 | 0.0049 |

All four arms perform similarly here (0.71–0.73); none of the D-vs-B/C differences are large relative to the
segment's baseline level, and D − C is actually slightly negative (−0.0024, 1/3 wins). This segment has no
pseudo-account edges at all, so it is the least informative for judging entity-memory mechanisms specifically;
it is included for completeness per the strict 3-way stratification policy, not because a memory effect is
expected here.

## K. Top-K (pooled, mean over seeds)

| Arm | P@0.1% | R@0.1% | P@0.5% | R@0.5% | P@1% | R@1% | P@5% | R@5% |
|---|---|---|---|---|---|---|---|---|
| A | 0.968 | 0.029 | 0.896 | 0.132 | 0.809 | 0.240 | 0.361 | 0.534 |
| B | 1.000 | 0.030 | 0.914 | 0.135 | 0.828 | 0.245 | 0.375 | 0.554 |
| C | 1.000 | 0.030 | 0.931 | 0.138 | 0.825 | 0.244 | 0.368 | 0.545 |
| D | 0.996 | 0.030 | 0.915 | 0.135 | 0.842 | 0.249 | 0.381 | 0.564 |

At very tight alert budgets (0.1–0.5%), B, C and D are essentially tied and all edge out A slightly; D pulls
ahead of B and C mainly at the 1% and 5% budgets. This is consistent with the pooled PR-AUC finding: most of
D's apparent edge over B is concentrated in the mid-recall range, and D's edge over C is small but consistent
across budgets.

## L. Leakage validation

**Arm C** — full temporal leakage suite (`src/graph/temporal_checks.py`, unchanged from Phase 7) plus an
order-agnostic-memory test, run on the trained checkpoints for all 3 seeds:

- **CPU (deterministic, exact-equality tolerance): 21/21 passed** — future_invariance (bitwise-identical past
  predictions at cutoffs 122/130/140/150), past_sensitivity (positive control: corrupting days 116–121 changes
  later predictions by up to 1.6e2–9.6e2 in logit space, confirming the checks can detect dependence and that
  the mean-memory control genuinely uses history), train/val boundary continuity, per-step access-only-own-day,
  and labels-never-reach-state.
- **GPU (noise-floor tolerance): 21/21 passed** — same checks, tolerances set from a measured 4-run noise
  floor (1.2e-4–4.9e-4), all deviations within floor.
- **Order-agnostic memory test** (chronological vs. a shuffled day order, days 1–40, untrained-vs-checkpoint
  comparison suppressed to structural property only): arm C's state differs by relative error ~3e-7 between
  orderings (float-summation rounding only) — **order-agnostic as designed**. Arm D (GRU), run as a positive
  control through the same test, differs by relative error ~2.0 (order strongly matters) — confirming the test
  has power to detect order-dependence and that C's memory is genuinely different in kind from D's.
- Phase 4 graph validation suite: 8/8 (re-run, unchanged from Phase 4).

**Arm B** — interventional checks of the static rolling protocol, run on the trained checkpoints for all 3
seeds:

- Dataset-level (both attempts): scaler fit TRAIN-only (verified against an independent refit); the 18
  training windows are disjoint, chronologically ordered, and cover exactly the 417,559 TRAIN transactions
  (no TEST, no VALIDATION rows enter training); the TRAIN-masked training graph's edges never exceed day 121;
  every VALIDATION day's masked graph is confirmed to have no edge touching any transaction with day greater
  than that day — **including every TEST-period row and edge, which are physically present in the saved
  Phase 4 graph file but structurally unreachable through `edge_time` masking**; scored days are 122–151 only,
  TEST (152+) never scored.
- Per-seed (**GPU, noise-floor tolerance — see deviation below**), 3/3 seeds: `rolling_logits` reproduces the
  production `rolling_predict` scores to within 2.5e-7–7.3e-7; future-invariance holds at cutoffs 122/136/151
  (corrupting every transaction's features, every future edge's endpoint, and every future label beyond the
  cutoff — including all of TEST — leaves predictions for days at or before the cutoff unchanged, within the
  measured GPU noise floor); past-sensitivity positive control confirms the test can detect dependence
  (corrupting days 116–121 changes later predictions by 56–197 in logit space).
- **Total: 20/20 passed.**

**Deviation from the design document, disclosed:** the design specified arm B's future-invariance check on
CPU with exact-equality tolerance (as was done for Phase 7's original suite). In practice, this project has
no `torch_sparse`/`pyg_lib` installed (a standing environment constraint), so PyG's CPU message-passing for
the *full* 590K-node graph is very slow: a first CPU attempt was still computing seed 42's baseline forward
passes after 20+ minutes of wall time with no other check yet started, versus ~1 minute total for the
equivalent GPU run. That CPU attempt was interrupted twice — once by the user's stop-all-background-tasks
instruction (partial log preserved as
`experiments/temporal_ablation/validation_log_armB_cpu.PARTIAL_INTERRUPTED_20260920.txt`, 5/5 dataset-level
checks passed before the interruption) and once by me, after confirming it was not going to finish in
practical time (preserved as `...PARTIAL_TOO_SLOW_20260925.txt`). I switched arm B's per-seed interventional
checks to GPU with noise-floor tolerance — the same fallback protocol Phase 7 and arm C already use
successfully — rather than continuing an impractically slow CPU run. This is a disclosed change in *tolerance
regime*, not a weakening of what is being tested: the same corruptions, the same cutoffs, and the same
pass/fail logic apply; only the numerical tolerance (0 vs. a measured noise floor) differs, exactly as it does
for Phase 7's own D-model GPU checks. No arm B result in Sections E–K was recomputed differently because of
this; it affects only the leakage-check log's execution.

## M. Resource usage

| Arm | params | mean train wall (s) | mean peak GPU reserved (GiB) | mean peak GPU allocated (GiB) | steps/epoch | mean epochs run |
|---|---|---|---|---|---|---|
| A | 162,817 | 323.7 | 5.21 | 4.48 | 1 | 130.3 |
| B | 162,817 | 3302.9 (raw) / ~554.6 (corrected, see below) | 5.21 | 4.48 | 18 | 80.3 |
| C | 179,201 | 97.1 | 1.89 | 1.34 | 18 | 79.7 |
| D | 197,889 | 125.7 | 1.88 | 1.36 | 18 | 73.7 |

**Arm B wall-time correction (disclosed):** seeds 42 and 123 each contain exactly one anomalous epoch
(4332.5 s and 3912.3 s respectively, vs. a median of ~6.96 s/epoch for every other epoch across all three arm
B seeds) — consistent with a system suspend/idle gap during the run, not additional computation. Seed 2024
ran cleanly end-to-end at 331.75 s / 48 epochs = 6.91 s/epoch, matching the pre-run smoke-test estimate
(~7.9 s for 2 epochs including one-time warmup). Excluding the one anomalous epoch per affected seed: seed 42
= 664.7 s (96 epochs, 6.92 s/epoch), seed 123 = 667.3 s (97 epochs, 6.88 s/epoch), seed 2024 = 331.7 s
(6.91 s/epoch) — a consistent ~6.9 s/epoch across all three seeds. The corrected mean (~554.6 s total wall,
i.e., ~554.6 s of actual training) is what should be used for any runtime comparison; the raw
`training_wall_seconds` field in the saved JSON is left unmodified (per the instruction not to alter existing
results) and this correction is documented here rather than silently applied to the artifact.

With that correction, arm B (static, schedule-matched) trains in about the same per-epoch wall time as arm A
per epoch (~2.4s/epoch is Phase 6A's historical figure; B's ~6.9s/epoch reflects doing a full-graph forward
pass 18 times instead of once), but converges in fewer epochs (80.3 vs 130.3 mean), so total wall time is
comparable across A and B once B's suspend anomaly is excluded. C is markedly cheaper than both A and B
(97s mean, 1.89 GiB peak) because it processes one day's snapshot at a time rather than the whole graph, the
same efficiency Phase 7's D already exhibited (125.7s, 1.88 GiB) — this is an architectural property of the
temporal streaming pipeline (shared by C and D), not of the memory mechanism being ablated.

## N. Comparison A vs B vs C vs D

Ranking by mean pooled PR-AUC: **D (0.4927) > B (0.4792) ≈ C (0.4790) > A (0.4637).**

- A→B (+0.0155, schedule only): not established as robust by the pre-registered rule (2/3 wins, doesn't clear
  the SD bar), but directionally the largest single jump in the chain and larger than B→D's residual gap in
  two of three seeds. The training schedule is doing real work.
- B→C (−0.0002, causal streaming pipeline replacing single full-batch training, with the same schedule and no
  recurrence): no distinguishable difference. Streaming day-by-day with a permutation-invariant memory
  performs the same, pooled, as static full-graph training on the same schedule.
- C→D (+0.0137, adding the learned recurrent GRU memory in place of the cumulative mean): the one comparison
  in this ablation that clears the pre-registered bar on the pooled metric (3/3 wins) and most clearly on
  cold-start (3/3 wins, smallest p-values of any comparison in this report).
- A→D (+0.0290, Phase 7's original headline number): decomposes as roughly half schedule (B−A = 0.0155) and
  half genuine memory contribution beyond a schedule-matched, streaming-but-memoryless baseline (D−C = 0.0137),
  with C−B (≈0) confirming that the streaming/architecture change itself contributes ~nothing beyond schedule.

## O. Interpretation

**Does Temporal GraphSAGE still beat the schedule-matched static model?** Not reliably, on pooled PR-AUC, by
the pre-registered criterion — D beats B in only 2 of 3 seeds, and in one seed B is actually slightly ahead
(0.4912 vs 0.4888). The original Phase 7 finding that temporal beats static by +0.029 conflated two effects;
once the schedule is matched, the remaining pooled gap (D−B mean +0.0135) is smaller than the seed-to-seed
noise in this experiment lets us call established. **Cold-start is the one segment where D beats B robustly**
(3/3 wins, +0.027 mean, smallest p-values) — this is the strongest evidence in this report that something
about D specifically (rather than just training longer/differently) helps the segment that matters most for
a fraud-detection cold-start use case.

**Does it beat the memory-ablated control?** Yes, on pooled PR-AUC, by the pre-registered criterion (3/3 wins,
mean +0.0137, exceeds both arms' seed SDs), and most clearly on cold-start (3/3 wins, +0.026, smallest
descriptive p-values of the whole table: paired 0.081, Welch 0.010). Since B and C perform statistically
indistinguishably from each other (C−B ≈ 0, 1/3 wins), and C differs from D *only* in the memory mechanism
(same streaming pipeline, same schedule, similar parameter count), the C→D gap is the cleanest available
attribution in this design: **the learned, order-sensitive recurrent memory itself does appear to add real
signal beyond what a same-history, order-agnostic aggregate captures — but the effect is smaller than Phase
7's headline number, concentrated mainly in cold-start, and not visible on the no-edge segment or reliably on
warm-start.**

**No causal claim is made from significance alone.** All p-values here are descriptive, n=3, uncorrected for
multiple comparisons; several point estimates (e.g., D−B, B−A) have wide seed-to-seed variation and could
plausibly be zero. What licenses the qualitative conclusions above is the pre-registered win-count-plus-SD
rule and the pattern of internal consistency (C≈B, C<D≈A+memory-only-gain), not the p-values themselves.

## P. Limitations

- n=3 seeds per arm is small; several comparisons that fail the pre-registered rule (D−B on pooled, all
  warm-start comparisons) might look different with more seeds, in either direction.
- Arm C has 9.4% fewer parameters than D (179,201 vs 197,889); while both retain nearly the same non-memory
  computation (proj/read/head identical), a small capacity confound in C's favor-or-against cannot be fully
  excluded as a partial explanation for the C-vs-D gap.
- Arm B's wall-clock resource numbers required a disclosed post-hoc correction for what looks like a system
  suspend/idle gap during two of its three runs (Section M); the underlying cause was not independently
  confirmed (e.g., via OS event logs), only inferred from the pattern of a single multi-thousand-second epoch
  amid an otherwise consistent ~6.9s/epoch series.
- Arm B's per-seed leakage validation ran on GPU with noise-floor tolerance rather than CPU with exact
  equality, as originally specified, because the CPU path was impractically slow in this environment (no
  `torch_sparse`/`pyg_lib`); Section L documents this as a tolerance-regime change, not a scope reduction.
- Warm-start metrics are noisy (SDs of 0.03–0.05 on a base rate around 0.26–0.33) across every arm; no
  strong claims are made about warm-start beyond "the arms are hard to distinguish there."
- This design tests one specific alternative memory (order-agnostic cumulative mean); it does not rule out
  that some other non-recurrent mechanism (e.g., a recency-weighted but still order-agnostic average) could
  close more of the C-to-D gap. That would be a different, follow-on ablation, not run here.
- As in Phase 7, TEST was never loaded or scored; all numbers in this report are VALIDATION-only.

## Q. Whether temporal memory itself adds measurable signal

Yes, but modestly and unevenly: pooled PR-AUC shows a real (3/3-seed, SD-exceeding) but small (+0.0137) gain
from D over the memory-ablated control C, concentrated mainly in cold-start (+0.0257, the cleanest result in
this study) and, more weakly and with more seed variance, in warm-start (+0.0262, does not clear the SD bar).
It is not visible on the no-edge segment, which is expected since that segment has no pseudo-account-edge
history for any memory mechanism to use. The learned GRU's advantages over a fixed cumulative mean — gating,
recency-awareness via the elapsed-time input, and order-sensitivity — appear to matter specifically where an
entity's transaction history is sparse and its order/timing informative (cold-start), less so at the pooled
level where the bulk of the data is warm/no-edge.

## R. Recommendation for next phase

1. Treat the temporal-vs-fully-schedule-matched-static comparison (D vs B) as **not yet established** at
   n=3 seeds; if a future phase needs a defensible temporal-vs-static headline number, either run more seeds
   for A/B/D or report the pooled comparison with this caveat attached, and lead with the cold-start result
   (which is robust) rather than the pooled one (which is not).
2. Any future temporal-modeling work should keep C's schedule/pipeline (18-step windowed training, causal
   day-by-day streaming) as the standard baseline for judging a memory mechanism's marginal value, rather than
   Phase 6A's 1-step/epoch static baseline — B and C already show that streaming-without-memory and matched
   static are indistinguishable, so C is a stronger and cheaper control than A for that purpose going forward.
3. The clearest actionable finding is that recurrent memory's value is concentrated in cold-start. If a later
   phase pursues further temporal-architecture work, cold-start PR-AUC (not pooled) is the metric most likely
   to show a genuine, reproducible effect of memory-mechanism choices.
4. Per the standing instruction, this report stops here: no Phase 8 (adaptive updating), drift monitoring,
   ring discovery, explainability, dashboard, or further ablation was started.

---

### Appendix: artifacts used

- Design (written before execution): `experiments/temporal_ablation/DESIGN_BEFORE_EXECUTION.md`
- Arm B training: `experiments/temporal_ablation/static_schedule_matched/{seed_42,seed_123,seed_2024}/`,
  `summary.json`, `metadata.json`, `train_log_B_static_schedule_matched.txt`, `convergence_curves.png`
- Arm C training: `experiments/temporal_ablation/memory_ablated_temporal/{seed_42,seed_123,seed_2024}/`,
  `summary.json`, `metadata.json`, `train_log_C_memory_ablated.txt`, `convergence_curves.png`
- Comparison: `experiments/temporal_ablation/comparison.json`, `comparison_plot.png`
  (generated by `scripts/compare_phase7a.py`, read-only over the above)
- Leakage/validity: `experiments/temporal_ablation/leakage_validation/leakage_validation_armC_cpu.json` (21/21),
  `leakage_validation_armC_cuda.json` (21/21), `leakage_validation_armB_cuda.json` (20/20),
  `graph_validation_output_armC_{cpu,cuda}.txt` (8/8), preserved partial arm-B CPU logs
  (`validation_log_armB_cpu.PARTIAL_INTERRUPTED_20260920.txt`, `...PARTIAL_TOO_SLOW_20260925.txt`)
- Regression checks (this session, read-only over unchanged artifacts): `scripts/run_tests.py` (22/22 unit
  tests), `scripts/run_validation.py` (8/8 dataset-level checks) — both confirm raw data, processed data, and
  earlier-phase artifacts are unmodified.
- Fixed references: `experiments/graphsage_robust/` (arm A, Phase 6A), `experiments/temporal_graphsage/`
  (arm D, Phase 7) — neither was re-run or modified.
