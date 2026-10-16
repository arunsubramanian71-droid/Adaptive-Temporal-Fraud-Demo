# Phase 7A — Temporal mechanism ablation: design, fixed BEFORE any training run

Written before any Phase 7A model was trained. Each arm has ONE pre-specified configuration; no arm is
tuned, re-run for selection, or re-configured after seeing results. Seeds 42, 123, 2024. VALIDATION only
(TEST never used). Phase 7 (arm D) and Phase 6A (arm A) results are fixed references and are not re-run.

## Question
Is Phase 7's +0.029 pooled PR-AUC (Temporal 0.4927 vs Static 0.4637) attributable to the recurrent temporal
memory, or to the different optimization regime (18 windowed optimizer steps/epoch + gradient clipping vs 1
full-batch step/epoch)?

## Arms
| | Model | Training regime | Status |
|---|---|---|---|
| A | Static GraphSAGE (Phase 6A) | 1 full-batch step/epoch, no clipping | fixed reference |
| B | **Static GraphSAGE, schedule-matched** | 18 windowed steps/epoch, clip 1.0 | NEW |
| C | **Temporal pipeline, memory ablated** (order-agnostic cumulative-mean memory) | Phase 7 regime (18 steps, clip 1.0) | NEW |
| D | Temporal GraphSAGE (Phase 7, GRU memory) | 18 windowed steps/epoch, clip 1.0 | fixed reference |

Correction of a Phase 7 figure: the Phase 7 report said "~17 optimizer steps per epoch". The exact count is
**18** (days 1-121 in 7-day windows = 17 windows of 7 days + one 2-day window [120,121]). Arm B matches 18.

## Arm B — schedule-matched static GraphSAGE (exact specification)
Unchanged from Phase 6A: architecture `HeteroSAGE` (hidden 64, 2 layers, dropout 0.3, 162,817 params), Phase 4
graph, TRAIN-masked static view (edge_time <= 121) used for training, TRAIN-only StandardScaler, BCE with
pos_weight = TRAIN neg:pos (27.36), Adam lr 1e-3 weight_decay 1e-5, max_epochs 150, patience 20, rolling per-day
evaluation `rolling_predict` (edge_time <= t for each validation day t), strict 3-way segments, thresholds
chosen on validation only.

Changed (regime only):
* Each epoch = 18 optimizer steps, one per Phase-7 window: days [1-7],[8-14],...,[113-119],[120-121], processed in
  chronological order (same order as arm D).
* Each step: one full forward pass over the whole TRAIN-masked graph (same message passing as Phase 6A), loss =
  mean BCE over ONLY that window's transactions (the same transactions arm D uses at that step), backward, gradient
  norm clipped at 1.0, Adam step.
* Gradient clipping 1.0 is ADDED to match arm D (D needs it for recurrence; matching it removes it as a confound;
  it was not used in Phase 6A). This is a deliberate, documented deviation from Phase 6A's regime.
Not changed: graph architecture, features, dropout, evaluation, early stopping on validation PR-AUC.

## Arm C — memory-ablated temporal control (exact specification)
Phase 7 pipeline retained exactly: 1-day snapshots, causal streaming (`TemporalDataset.day_slice`), 7-day
truncated-BPTT windows with `detach_state`, the SAME `train_epoch` function (imported, not re-implemented), clip 1.0,
Adam 1e-3 / wd 1e-5, BCE pos_weight, 150 epochs / patience 20, same causal rolling evaluation
(`rolling_temporal_predict`), same features, same graph, no labels in state, TEST refused by the dataset builder.

Removed: the recurrent GRU memory, the learned initial state `h0`, the elapsed-time input, and the state-as-root
term of layer 1. Replaced by an ORDER-AGNOSTIC cumulative-mean memory:

  for each relation r (entity type e), day d:
    u_t   = W_r * hx_t                      (per-transaction message; W_r = Linear(64,64); hx = ReLU(Linear(470,64)(x)))
    Sum_e(d) = Sum_e(d-1) + sum_{t on day d linked to e} u_t
    Cnt_e(d) = Cnt_e(d-1) + #{t on day d linked to e}
    mbar_e   = Sum_e(d) / Cnt_e(d)          (mean of ALL messages from every transaction <= d; permutation-invariant)
    s_e      = MLP_e( Dropout_0.3( ReLU(mbar_e) ) )      MLP_e = Linear(64,192) - ReLU - Linear(192,64)
  layer 2 (entity -> txn): identical to arm D: z = ReLU( sum_r SAGE_r(s_e ; root = hx) ); logit = Linear(64,1)(z)

Design intent. C gives entities the same information as D (all history <= d, same-day included, no labels) and the
same causal streaming/training regime, but the memory is a fixed cumulative mean: no recurrence, no gating, no
recency weighting, no elapsed-time input, no dependence on the ORDER or timing of past events. (This is also
exactly what static GraphSAGE's mean aggregation over `edge_time <= t` computes, because SAGEConv's lin_l commutes
with the mean.) The per-entity MLP replaces the GRU so capacity is approximately preserved (25k vs 25k parameters
per entity type). Parameter counts: C = 179,201 vs D = 197,889 (-9.4%; the difference is lin_r roots 16,384 + h0 256 +
GRU-vs-MLP 2,048). A capacity difference of this size cannot be fully excluded as a factor; it is disclosed.
The order-agnostic property is TESTED, not assumed (state after processing days in shuffled order must equal the
in-order state for C; the same test on D is a positive control that must show large differences).
A stateless (reset-every-day) variant is deliberately NOT run: that would remove history rather than
ordering, answering a different question and adding an unrequested arm.

## Pre-specified comparisons and reading rules
Primary: D vs B and D vs C on pooled PR-AUC, seed-matched (same seed number) differences with wins/3 and mean
diff; secondary: cold, warm, no-edge; also A vs B (does schedule alone matter) and C vs B (causal streaming
pipeline vs static). n = 3 per arm: paired p-values and Welch p-values are DESCRIPTIVE only and uncorrected.
Reading rules fixed in advance:
 * "Temporal gain persists" requires D > B in pooled PR-AUC in 3/3 seed pairs AND mean(D) - mean(B) larger than the
   larger of the two arms' seed SDs; otherwise it is reported as not established / disappeared.
 * "Memory contributes" requires D > C by the same criterion. If C ~ D, recurrence per se adds no measurable signal
   beyond having history; if C ~ B ~ A, neither streaming nor memory matters.
 * No causal claim is made from significance tests alone.

## Leakage / validity plan
Full temporal leakage suite (`temporal_checks.py`) on arm C for all seeds, CPU (exact) and GPU; plus order-agnostic
test and D-positive-control; Phase 4 graph suite (8/8); for arm B, future-invariance of the static rolling protocol
(corrupt everything after day c including all TEST rows and edges; predictions for days <= c must be unchanged) on
CPU, plus the graph suite. TEST is never used for any decision.
