# Phase 8 — Concept Drift and Adaptive Updating: Design (written BEFORE any execution)

Status: **design/review only.** Nothing in this document has been executed. No model has been trained, no
background task has been launched, no artifact other than this file exists for Phase 8. Per the standing
instruction, this is written and must be reviewed/approved before any Phase 8 code runs.

Grounding: this design treats the Phase 7A report (`reports/phase7a_temporal_ablation_report.md`) as
authoritative. In particular: **the pooled Temporal-vs-schedule-matched-static comparison (D vs B) did NOT
meet the pre-registered gain-persistence rule (2/3 seed wins)**; the one robust, 3/3-seed finding was that
Temporal GraphSAGE's recurrent memory beats the memory-ablated control specifically on **cold-start**
(`has_edge_cold`, +0.0257 mean, exceeds both arms' seed SDs, smallest descriptive p-values in the whole
study). Phase 8 does not re-litigate that finding; it carries it forward as a reporting requirement (Section
7) and as the reason a specific backbone is chosen (Section C).

---

## 0. What Phase 8 is and is not

Phase 7/7A studied **architecture** (does a recurrent memory beat a schedule-matched static model, or a
memory-ablated causal-streaming control?) with weights fixed after one training run. Phase 8 studies a
**different, orthogonal axis**: given a fixed architecture, does periodically updating its *weights* as new
labels become available reduce performance degradation under distribution shift, versus freezing weights
after the initial TRAIN fit? The temporal architecture's recurrent *state* already evolves causally during
inference in Phase 7 (`TemporalState` carried forward, never reset — this is a property of the architecture,
already validated in Phase 7's `train_val_boundary_state_continuity` check). That state evolution happens in
**both** arms of Phase 8 and is *not* what is being tested here; only the schedule of **weight** updates
differs between arms. Conflating these two kinds of adaptivity would make the experiment uninterpretable, so
Phase 8 is designed to keep them separate (Section B).

---

## A. Research question

Does periodically updating model weights on newly available, chronologically later labels reduce performance
degradation under future-time distribution shift, compared with a model whose weights are frozen after the
initial TRAIN fit — while remaining strictly causal (no future information, no label used before its
information time, TEST never loaded) — and is any such benefit attributable to genuinely new chronological
information rather than simply to additional training volume/gradient steps?

**H4 (the hypothesis under test):** Adaptive updating reduces performance degradation under concept drift
relative to a frozen model, while respecting `information_time <= prediction_time` for every feature, state,
and update.

---

## B. Backbone architecture — a pragmatic, disclosed choice

**Primary backbone: Temporal GraphSAGE, exactly Phase 7/7A arm D's architecture and hyperparameters**
(`src/models/temporal_graphsage.py`, hidden=64, dropout=0.3, GRU memory, 197,889 params), reused unmodified.

This is a **computational and infrastructural choice, not a re-endorsement of D's unproven pooled advantage**:

- **Cost.** Phase 8 needs many repeated fine-tuning rounds (windows × seeds × arms). Arm D trains at
  ~1.1–1.7 s/epoch; schedule-matched static (arm B) trains at ~6.9 s/epoch (Phase 7A, Section M, corrected
  for a logged suspend anomaly) for the *same* 18-step schedule — a ~5x cost multiplier that compounds across
  every update round. On this AMD/ROCm/Windows environment (no `torch_sparse`/`pyg_lib`), that multiplier
  matters (Section I).
- **Infrastructure reuse.** `scripts/train_temporal_graphsage.py`'s `train_epoch(model, ds, opt, train_last,
  pos_weight, window)` already accepts an arbitrary `train_last` day — Phase 8's periodic updates are literally
  repeated calls to this *existing, leakage-validated* function with an advancing `train_last`, not new
  training code. The causal dataset accessor (`TemporalDataset.day_slice`) and the full Phase 7
  leakage-check suite (`src/graph/temporal_checks.py`) are reused as-is.
- **Explicitly not a claim that D beats B pooled under drift** — that question (architecture vs. schedule) was
  Phase 7A's, and its answer there was "not established, except at cold-start." Phase 8 asks a different
  question using D as a fixed, convenient vehicle. Section I flags that a static-backbone replication is an
  open option, not performed here.

No GAT, transformer, graph-transformer, RL, LLM, or new infrastructure (Kafka/Kubernetes/multi-agent) is
introduced, per the explicit constraint.

---

## C. Experimental arms

All arms share: identical architecture (Section B), identical initial fit on TRAIN (days 1–121), identical
seeds (42, 123, 2024), identical features/graph/split boundaries, identical causal streaming mechanics.

| Arm | What's frozen | What updates, and when | Information available at update `k` (boundary day `b_k`) |
|---|---|---|---|
| **F — Frozen-parameter baseline** | Model **parameters/weights only**, after the initial TRAIN-only checkpoint | No weight update ever occurs. Causal temporal/entity state is explicitly **allowed to evolve during inference**, exactly as the Phase 7 architecture already specifies (`TemporalState` carried forward, never reset, between and within windows) — this is not a weight update and is not disabled for arm F. | n/a — no updates occur |
| **Ad — Adaptive** | Nothing | Weights fine-tuned at every window boundary `b_k` (5 updates, see Section C below) | `TRAIN(1..121) ∪ VAL(122..b_k)` — i.e. every VAL day **strictly before or at** the boundary, whose window has already been scored (Section D) |
| **E — Adaptive replay-control (ablation)** | Nothing | Weights fine-tuned at every window boundary with the **same update cadence, same number of new examples, same optimizer settings** as Ad | `TRAIN(1..121)` only, re-sampled with replacement each round (no new chronological information ever enters this arm after the initial fit) |

**Initial fit (shared by F, Ad, E, per seed):** train on TRAIN days 1–121 using the existing Phase 7 training
loop; epoch count fixed by a **TRAIN-only internal holdout** — fit on days 1–111, early-stop on days 112–121
PR-AUC (10 held-out TRAIN days), exactly as in Phase 5–7 but using TRAIN's own tail instead of VAL. This
yields a single initial checkpoint per seed, identical across F/Ad/E (verified bit-for-bit, Section K's new
check `window1_frozen_equals_adaptive_before_first_update`).

*Why not reuse Phase 7's existing arm-D checkpoints:* those checkpoints were epoch-selected via early
stopping on **all 30 VAL days' PR-AUC** — appropriate for Phase 7's question, but it would hand the Frozen
arm here an "oracle" advantage (implicitly tuned against the very data it is later scored on window-by-window)
that the Adaptive arm cannot match under a strict walk-forward protocol. Re-fitting with a TRAIN-only internal
holdout removes that asymmetry so F and Ad start from a fair, equally-blind initialization. This is a
deliberate, disclosed departure from the Phase 5–7A convention, made specifically so Phase 8's F-vs-Ad
comparison is not biased in advance.

**Update mechanics for Ad (and E's matched compute):** warm-started fine-tuning, not retraining from scratch
— continue from the current weights, run `K` additional epochs of the existing TBPTT loop (`window=7` days,
grad-clip 1.0, same Adam lr) over the expanding window `TRAIN ∪ already-scored VAL`. `K` and the fine-tune
learning rate are **pre-registered once**, chosen via the same TRAIN-only internal holdout (days 112–121)
used for the initial fit's epoch count — never tuned against any VAL window. The same `K`/lr apply to every
update round, every window, every seed. This keeps the update policy simple and reproducible (no RL, no
meta-learning, no per-round hyperparameter search) and keeps `K`/lr itself leak-safe.

---

## D. Chronological evaluation protocol

VAL (days 122–151, 30 days) is split into **5 chronological windows of 6 days each**, dividing evenly with no
remainder: W1=[122–127], W2=[128–133], W3=[134–139], W4=[140–145], W5=[146–151]. TEST (152–182) is never
loaded, consistent with every prior phase.

Walk-forward loop, in strict chronological order, for F/Ad/E independently (per seed):

1. Score window `W_k` using the weights currently in effect (for Ad/E, the weights produced by the update at
   the end of `W_{k-1}`'s scoring, or the shared initial checkpoint for `k=1`). Predictions use only
   `day_slice(d)` for `d` in `W_k`, exactly Phase 7's causal accessor — no feature/edge from `d' > d` is ever
   read, for any `d` in `W_k`.
2. Only **after** `W_k` is fully scored does `W_k`'s true labels become "available." Update boundary
   `b_k = max(W_k)`.
3. For Ad: fine-tune on `TRAIN ∪ W_1 ∪ ... ∪ W_k` (expanding window) for the pre-registered `K` epochs, then
   proceed to score `W_{k+1}` with the updated weights.
4. For E: fine-tune for the same `K` epochs using `TRAIN`-only data **resampled with replacement** to the same
   example count as step 3 would have added, so E spends the same compute/gradient-steps/data-volume as Ad at
   every round, but never sees a new chronological day.
5. For F: do nothing; proceed directly to score `W_{k+1}` with the original weights.

No window's own labels are used in its own update (by construction: `W_k` is scored *before* its labels enter
the update at `b_k`, and that update only ever informs `W_{k+1}` onward). No random reshuffling of evaluation
order occurs; E's resampling affects only the *source* of E's replay training data, never the chronological
order in which any arm is scored.

Supplementary (visualization only, not a hypothesis-testing unit): predictions retain a `day` column, so a
day-level (30-point) degradation curve can also be plotted post hoc. The pre-registered statistical comparison
(Section F) uses only the 5 windows.

---

## 5. Drift measurement (diagnostic/reporting, does not drive the update policy)

Computed **after** predictions are made, never fed back into the update decision (the update policy is fixed
cadence, Section C/D — no drift-triggered branching, to avoid a feedback loop that would itself need its own
leakage argument, and to keep the policy "simple and reproducible" per the explicit constraint):

- **Feature-distribution drift** (label-free, causal): for each window `W_k`, compute the Population
  Stability Index (PSI) and a two-sample KS statistic between `W_k`'s transaction feature distribution (the
  470-dim scaled features already used by the model) and the TRAIN reference distribution, per feature,
  summarized by the mean/max PSI across features. Uses only each window's own features — available at
  prediction time regardless of labels, so no leakage risk.
- **Prediction-distribution drift** (label-free): PSI/KS between `W_k`'s predicted-score distribution and (a)
  TRAIN-era scores (from the initial checkpoint scored causally back over TRAIN) and (b) the immediately
  preceding window's scores, for each arm separately. Also label-free.
- **Performance drift** (requires labels, computed post hoc for reporting only): PR-AUC/ROC-AUC per window per
  arm, and each window's delta from `W_1`'s value. This is the primary degradation signal reported in Section
  F/G of the eventual report, but — critically — it is never computed and consulted *during* the walk-forward
  loop to decide whether/how to update; it is strictly an after-the-fact analysis artifact, exactly like every
  PR-AUC number in Phases 3–7A.

---

## 6. Update policy (simple, reproducible, no new infrastructure)

Fixed-cadence warm-started fine-tuning, as specified in Section C/D: every 6 days, `K` epochs of the existing
TBPTT training loop over the expanding TRAIN∪already-scored-VAL window, pre-registered `K` and learning rate,
identical across windows/seeds. No reinforcement learning, no online per-sample SGD, no drift-triggered
branching, no LLM/agent component, no new serving infrastructure. This is deliberately the simplest policy
that lets H4 be tested; a drift-triggered cadence (update only when measured drift crosses a threshold) is
noted as possible future work (Section I) but is out of scope here to avoid conflating "when to update" with
"does updating help at all."

---

## 7. Cold-start / warm / no-edge reporting

Every metric in Sections F/G/K is reported **pooled and split by the existing strict 3-way segmentation**
(`has_edge_warm`, `has_edge_cold`, `no_pseudo_account_edge`, via `compute_pseudo_account_segments`), per window,
per arm, per seed — unchanged machinery from Phases 5–7A. This is a hard requirement, not an option, precisely
because Phase 7A's one robust finding was cold-start-specific: if adaptive updating's benefit (if any) is also
concentrated in cold-start, that would be consistent with Phase 7A and worth highlighting; if it appears
pooled but not cold-start, or vice versa, that distinction must be reported explicitly rather than collapsed
into a single pooled number. No claim in the eventual Phase 8 report may rely on the pooled metric alone if
the segment-level pattern disagrees with it — the report must show both, as Phase 7A did.

---

## 8. Statistical / reproducibility design

- Seeds: 42, 123, 2024 (unchanged from every prior phase).
- Windows: W1–W5 as defined in Section D, pre-registered, not changed after seeing results.
- Primary metric: pooled PR-AUC per window. Secondary: ROC-AUC, Precision/Recall/F1 (threshold policy below),
  P@{0.1%,0.5%,1%,5%}, R@{0.1%,0.5%,1%,5%}, alert count and false-positive count at the registered threshold.
- **Threshold policy (a deliberate departure from Phases 5–7A, required by the walk-forward design):** earlier
  phases selected the classification threshold from the *entire* 30-day VAL period's scores after the fact —
  acceptable there because VAL was scored as one batch post-training. Phase 8 scores VAL window-by-window as
  if live, so a threshold fit on VAL would let a window "see" labels from windows not yet scored. Instead, the
  threshold is selected **once**, from the same TRAIN-only internal holdout (days 112–121) used for `K`/lr,
  and applied unchanged to every window of every arm. This is disclosed explicitly; Precision/Recall/F1
  numbers in Phase 8 are not directly comparable to Phases 5–7A's VAL-threshold numbers for this reason
  (PR-AUC/ROC-AUC/P@K/R@K remain threshold-free and are comparable).
- Decision rule for H4 (all three must hold for H4 to be reported as "supported"; partial satisfaction is
  reported honestly as partial/inconclusive, not rounded up):
  1. **Degradation slope:** per seed, `(PR-AUC_F,W1 − PR-AUC_F,W5) > (PR-AUC_Ad,W1 − PR-AUC_Ad,W5)` — Frozen
     degrades more from first to last window than Adaptive does — in at least 2/3 seeds (a 3/3 bar, as used
     for pooled claims in Phase 7A, is also reported but not required given only 5 windows per seed to show a
     trend).
  2. **Pooled magnitude:** mean-over-windows-and-seeds `(PR-AUC_Ad − PR-AUC_F) > 0`, and this mean difference
     exceeds the larger of Ad's and F's across-seed SD (same style of rule as Phase 7A's pre-registered bar).
  3. **Attribution:** Ad beats E (replay-control) by the same magnitude test in rule 2, confirming the gain is
     not explained by "more gradient steps/data volume alone."
  Paired and Welch p-values (n=3 seeds) are reported as descriptive only, uncorrected for the multiple
  windows/metrics involved — exactly Phase 7A's convention, carried forward.

---

## 9. Computational feasibility

- Backbone (Section B) chosen partly for cost: ~1.1–1.7 s/epoch at TRAIN scale; each update round is `K`
  epochs (pre-registered, expected small, e.g. 5–15) over an expanding window no larger than TRAIN+30 days —
  still cheap relative to the original 150-epoch/patience-20 budget.
- Total new training events: 3 arms (F has none; Ad and E each do 5 updates) × 3 seeds = up to 30 fine-tune
  episodes of `K` epochs each, plus 3 shared initial fits — an order of magnitude cheaper than Phase 7A's full
  3-seed, 150-epoch runs for B and C.
- GPU memory: arm D's peak reserved was 1.89 GiB in Phase 7A — no change expected; comfortably inside the
  17.2 GB AMD GPU budget with headroom for feature-drift computations run alongside.
- No `NeighborLoader`/`ClusterData`/sampling extension is used anywhere (full day-snapshot streaming, as
  Phase 7 already established as the only viable path on this ROCm/Windows/no-`pyg_lib` environment).
- Leakage-check determinism: Phase 7A found CPU exact-equality checks for the *static* architecture
  impractically slow on this environment and switched to GPU noise-floor tolerance. Phase 8's backbone is the
  cheap temporal architecture, for which Phase 7A's CPU exact-equality checks completed in reasonable time
  (arm C: full suite in well under a minute on CPU) — so CPU-exact checks remain the default for Phase 8's
  leakage suite, with GPU noise-floor as the documented fallback if runtime proves otherwise, rather than
  discovering this mid-run as happened in 7A.

---

## 10. Ablation: adaptation itself vs. simply seeing more data

This is arm **E** (Section C), defined here before any execution, not improvised afterward:

- Same architecture, same initial checkpoint, same update cadence (5 rounds), same `K` epochs/round, same
  optimizer/lr as Ad.
- The only difference: at each round, E's additional training examples are drawn **with replacement from
  TRAIN (days 1–121) only**, re-sampled to the same example count Ad would have added from the newly-scored
  VAL window, using a fixed, reproducible RNG (`seed = 1000*base_seed + window_index`). E never sees a VAL
  transaction's features, edges, or label at training time — only more repetitions of already-seen TRAIN data.
- If Ad outperforms E by the magnitude test in Section 8, that is evidence the gain comes from genuinely new
  chronological information (drift-adaptation), not merely from extra gradient steps or data volume. If Ad
  and E are indistinguishable, the honest conclusion is that H4's apparent support (if any) is confounded with
  "more training," and must be reported as such — not quietly upgraded to a drift-adaptation claim.
- This is a controlled resampling of **training input to one ablation arm**, not a reshuffling of any arm's
  evaluation order — every arm, including E, is scored on W1→W5 strictly chronologically.

---

## 11. Leakage controls

- **TEST (152–182) is never loaded**, exactly as in every prior phase; the dataset builder's existing refusal
  (`allow_test=False` default) continues to guard this.
- **Label availability rule:** a VAL day's label is available for training/updating only after that day's
  window has been fully scored. Formally, update `k` (which produces the weights used to score `W_{k+1}`) may
  use labels only from days `<= b_k = max(W_k)`; `information_time(label_d) = d`, and `prediction_time` for
  any row scored using the weights from update `k` is always `> b_k`, so `information_time <= prediction_time`
  holds for every label used in every update, by construction of the walk-forward loop (Section D).
- **State causality is unchanged from Phase 7**: `TemporalDataset.day_slice(d)` is the only data accessor;
  `TemporalState` is written only for entities touched on day `d`, from day `d`'s own slice; labels never enter
  `TemporalState` (Phase 7's `check_labels_not_used`, re-run unmodified, Section K).
- **New checks specific to Phase 8's weight-update mechanism**, to be added to a `temporal_checks`-style
  Phase-8 suite before any run is treated as valid:
  - `window1_frozen_equals_adaptive_before_first_update` — F's and Ad's (and E's) predictions for W1 must be
    bitwise identical (CPU) / within noise floor (GPU), since no update has occurred yet that could
    distinguish them from the shared initial checkpoint.
  - `update_does_not_retroactively_change_past_predictions` — re-scoring `W_1..W_{k}` with the weights produced
    by update `k` must reproduce exactly the predictions that were already recorded when those windows were
    scored with the *pre-update* weights at the time (i.e., we are not silently re-predicting the past with
    future-informed weights; each window's recorded prediction used only the weights in effect at the time it
    was scored — this is a property of the walk-forward *protocol*, and this check verifies the saved
    predictions actually reflect it).
  - `future_label_exclusion` — corrupting (relabeling) every row in windows strictly **after** boundary `b_k`
    must not change the weights/loss produced by update `k` (interventional positive-direction test, same
    style as Phase 7's `perturb_future`/`perturb_past`).
  - `replay_control_never_touches_val` — for arm E, instrument the data loader (as Phase 7's
    `check_access_pattern` already does for days) to confirm every training example in every update round has
    `day <= 121` for every seed/round.
- **Threshold/epoch-count/lr pre-registration** (Section 8) is itself a leakage control: all of `K`, the
  fine-tune learning rate, and the classification threshold are fixed from a TRAIN-only internal holdout
  *before* any VAL window is scored, and never revisited afterward.

---

## 12. Final outputs

- `experiments/phase8_adaptive_drift/DESIGN_BEFORE_EXECUTION.md` — this document.
- Scripts (to be written before execution, not yet written): `scripts/train_phase8_frozen.py`,
  `scripts/train_phase8_adaptive.py`, `scripts/train_phase8_replay_control.py`,
  `scripts/measure_phase8_drift.py`, `scripts/validate_phase8_leakage.py`, `scripts/compare_phase8.py`.
- Per-arm, per-seed artifacts under `experiments/phase8_adaptive_drift/{frozen,adaptive,replay_control}/seed_{42,123,2024}/`:
  `model_checkpoint_initial.pt`, `model_checkpoint_W{1..5}.pt` (post-update weights, one per round, for audit),
  `metrics_val_W{1..5}.json` (pooled + 3-way segment metrics, each window), `predictions_val.parquet` (with
  `day` and `window` columns), `training_history.json` (initial fit + every update round), `resource_measurements.json`.
- `experiments/phase8_adaptive_drift/drift_measurements/`: `feature_drift.json`, `prediction_drift.json`,
  `performance_drift.json` (Section 5), per window per arm.
- `experiments/phase8_adaptive_drift/leakage_validation/`: `phase8_leakage_validation_{cpu,cuda}.json`
  (Phase 7's full suite + the four new Section-11 checks), re-run graph suite (8/8) for completeness.
- `experiments/phase8_adaptive_drift/comparison.json` + `comparison_plot.png` (per-window, per-arm, per-seed
  degradation curves — pooled and 3-way segmented — plus the day-level supplementary curve).
- `reports/phase8_concept_drift_report.md` — final report, mirroring Phase 7A's structure: Objective, Arms,
  Backbone rationale, Chronological protocol, Drift measurements, Update policy, Per-window results (pooled +
  segmented), Replay-control ablation, Leakage validation, Resource usage, Statistical decision on H4,
  Interpretation, Limitations, Recommendation for next phase.

---

## 13. Methodological concerns to resolve before execution

1. **TEST stays sealed**, so "future-time distribution shift" is studied only within VAL's 30-day span. Real
   drift over a longer horizon cannot be observed here; this is a disclosed scope limitation, not something
   Phase 8 can fix without relaxing the sealed-TEST rule (not proposed).
2. **Backbone choice (temporal D) is a cost/infrastructure decision, not a re-endorsement of Phase 7's
   pooled claim.** A static-backbone replication of F/Ad/E is a clean follow-up if the user wants the drift
   question answered for the simpler architecture too; not included here to keep scope and compute bounded.
3. **The Frozen arm here will not numerically match Phase 7's reported arm-D metrics**, because its epoch
   count is selected via a TRAIN-only internal holdout rather than full-VAL early stopping (Section C). This
   is intentional (fairness to Ad) but means Phase 8's "Frozen" is a new, re-fit reference, not a reuse of
   Phase 7's checkpoint — flagging this explicitly so it isn't mistaken for an inconsistency.
4. **Window granularity (5 × 6 days)** trades a visible degradation trend against each update's data
   increment being fairly small (~6 days of VAL, on the order of a few thousand transactions). This is a
   judgment call, pre-registered here; it should not be changed after seeing results.
5. **Label-availability lag is assumed to be 0** (a day's label is usable the moment that day elapses), same
   assumption implicit in every earlier phase's TRAIN/VAL boundary. Real chargeback/confirmation delays are
   not modeled. A sensitivity run with a positive lag is possible future work, not required for H4.
6. **Pre-registration discipline is the single biggest risk to this design's validity:** `K`, the fine-tune
   learning rate, and the classification threshold must be fixed from the TRAIN-only holdout *before* touching
   any VAL window, and never adjusted afterward. If any of these end up chosen or revised using VAL performance
   at any point, every comparison in this design is invalidated, silently.
7. **Multiple comparisons** (3 arms × 5 windows × 3 seeds × pooled/3-way-segmented × several metrics) are not
   corrected for multiplicity; all p-values remain descriptive, as in Phase 7A, and no single favorable
   window/segment should be reported as if it were the headline result without also showing the full table.
8. **The replay-control ablation (E) requires new, non-trivial sampling code** (reproducible resampling from
   TRAIN matched to each round's example count) that must itself be reviewed/tested before being trusted as
   evidence for attribution (Section 10) — it is not simply "reuse of existing infrastructure" the way the
   fine-tuning loop itself is.

Do not execute any part of this design until it is explicitly approved.

---

## 14. Clarifications added on review (binding — these supersede any looser phrasing above)

### 14.1 Frozen-arm terminology

Arm **F is a "frozen-parameter baseline"**, precisely:

- Model parameters/weights are **never updated** after the initial TRAIN-only checkpoint (Section C). There
  is no fine-tuning, no gradient step, no optimizer call for arm F after its single initial fit.
- The causal temporal/entity **state is explicitly allowed to evolve during inference**, exactly as specified
  by the Phase 7 architecture (`TemporalState`, carried forward across windows, never reset — the same
  mechanism validated by Phase 7's `train_val_boundary_state_continuity` check). This state evolution happens
  in F exactly as it does in Ad and E; it is a property of the architecture, not of the arm.
- Arm F must **not** be described as having a "completely frozen internal state" anywhere in this design or
  in the eventual Phase 8 report — only its parameters are frozen. (Section 0 and Section B already made this
  distinction conceptually; this entry makes the terminology itself explicit and binding.)

### 14.2 Label availability (explicit, binding)

- A transaction (and the window containing it) is **first scored** using the weights currently in effect.
- Its true label becomes **available only after** that scoring is complete.
- **Only then** may that labeled example enter the Adaptive or Replay-control update that follows.
- **No label from a window may influence any prediction for that same window** — by construction, the update
  that consumes `W_k`'s labels produces weights used only for `W_{k+1}` onward (Section D).
- **The current design assumes zero label delay**: a day's label is treated as available the instant that
  day's window has been scored. This is stated here explicitly as **an experimental simplifying assumption,
  not a claim about real-world operational/chargeback latency**. (Already flagged as methodological concern
  #5 in Section 13; restated here as a binding definitional statement, not merely a caveat.)

### 14.3 Pre-registration of hyperparameters (explicit, binding)

`K` (fine-tuning epochs per update), the fine-tuning learning rate, and the classification threshold are
each selected **exactly once**, before Phase 8 execution begins, using **only** a TRAIN-internal holdout
(fit on TRAIN days 1–111, select on TRAIN days 112–121) — **never** using any VAL window (W1–W5). Once
selected:

- The same `K` and the same learning rate apply to **every** update round, for **every** window, for
  **every** seed, for **both** Ad and E.
- The same threshold applies to **every** window, for **every** arm, for **every** seed.
- **None of `K`, the learning rate, or the threshold may be selected, revisited, or adjusted using W1–W5
  performance**, at any point, for any reason, including after partial results are observed. Any deviation
  from this invalidates the H4 comparison (already flagged as the top methodological risk in Section 13 #6;
  restated here as a hard, binding rule rather than a risk to "watch for").

### 14.4 Drift measurement — retrospective vs. prospective (explicit, binding)

| Drift metric | Definition | Timing |
|---|---|---|
| **Feature-distribution drift** | PSI + KS statistic between window `W_k`'s own transaction feature distribution and the TRAIN reference distribution | **(a) Retrospective.** Computed after `W_k` is scored, using only `W_k`'s own (label-free) features. Reported for analysis only. |
| **Prediction-distribution drift** | PSI + KS statistic between `W_k`'s predicted-score distribution and (i) TRAIN-era scores and (ii) the immediately preceding window's scores, per arm | **(a) Retrospective.** Computed after `W_k` is scored, using only already-produced scores. Reported for analysis only. |
| **Performance drift** | PR-AUC/ROC-AUC for `W_k` per arm, and each window's delta from `W_1` | **(a) Retrospective.** Requires `W_k`'s labels, which are only available after `W_k` is scored (Section 14.2); computed strictly after the fact. |

**All three drift metrics are (a) retrospective only.** None is (b) used prospectively to make an update
decision anywhere in this design — the update policy (Section 6) is fixed-cadence, decided independently of
any measured drift value. Explicitly and specifically: **no future-window drift measurement may influence
predictions for that same future window** — this cannot happen under the current design because every drift
metric for `W_k` is computed strictly after `W_k` has already been scored and those predictions are already
recorded; there is no code path by which a drift measurement feeds back into the model before or during the
scoring it is measuring. A drift-triggered update policy remains explicitly out of scope (Section 6, Section
13 concern re: possible future work).

### 14.5 Walk-forward order (exact causal sequence, binding)

```
TRAIN (days 1-121)
  -> initialize model (TRAIN-only internal holdout selects epoch count / K / LR / threshold, Section 14.3)
  -> score W1 (days 122-127), using the initial checkpoint, for F, Ad, and E identically
  -> W1 labels become available (only now)
  -> update Ad on TRAIN ∪ W1;  update E on TRAIN (resampled, matched volume);  F: no update
  -> score W2 (days 128-133), using each arm's current weights
  -> W2 labels become available (only now)
  -> update Ad on TRAIN ∪ W1 ∪ W2;  update E on TRAIN (resampled, matched volume);  F: no update
  -> score W3 (days 134-139)
  -> W3 labels become available
  -> update Ad on TRAIN ∪ W1..W3;  update E (resampled);  F: no update
  -> score W4 (days 140-145)
  -> W4 labels become available
  -> update Ad on TRAIN ∪ W1..W4;  update E (resampled);  F: no update
  -> score W5 (days 146-151)
  [end of VAL; TEST (152-182) is never loaded at any point in this sequence]
```

Note `W1` is scored by F, Ad, and E using the **identical** initial checkpoint (no update has occurred yet),
which is why `window1_frozen_equals_adaptive_before_first_update` (Section 11) must hold exactly.

### 14.6 H4 decision rule (preserved exactly, binding)

Unchanged from Section 8, restated verbatim as the binding rule — **not to be loosened after seeing results**:

1. Adaptive must show **less degradation than Frozen in at least 2 of 3 seeds** (per-seed `W1→W5` drop).
2. The **mean Adaptive − Frozen improvement must be positive and must exceed the larger of the two arms'
   across-seed SD**.
3. **Adaptive must also beat the Replay-control (E) under the same comparison rule** (mean difference
   positive and exceeding the larger seed SD).

If all three hold, H4 is supported. If any one fails, **H4 is reported as partial / not established** —
never rounded up to "supported," and never with the rule itself relaxed, narrowed, or re-worded post hoc to
manufacture a pass. (The supplementary 3/3-seed note in Section 8 is additional descriptive context only; it
is not part of the pass/fail rule and does not gate the H4 determination.)

### 14.7 Cold-start / warm / no-edge populations (preserved, binding)

The existing strict three-way segmentation is preserved exactly as used throughout Phases 5–7A:

- **Genuine cold-start** (`has_edge_cold`),
- **Warm / has-history** (`has_edge_warm`),
- **No-edge / missing-entity** (`no_pseudo_account_edge`).

These three populations are **never merged**, in particular no-edge/missing-entity is never folded into
cold-start (the exact error Phase 5 originally made and corrected). Every metric in Section 7/Section 8 is
reported for all three populations separately, per window, per arm, per seed, in addition to pooled.

### 14.8 TEST seal (explicit, binding)

**TEST (days 152–182) remains completely sealed.** It is not loaded, and it is not used for: drift detection
(Section 14.4 — all drift metrics operate on TRAIN/VAL only), threshold tuning (Section 14.3 — threshold comes
from a TRAIN-only holdout), update decisions (Section 6/14.5 — updates consume TRAIN ∪ already-scored VAL
only), or adaptive training of any arm (Ad and E both draw exclusively from TRAIN and/or already-scored VAL).
The dataset builder's existing refusal of TEST days by default continues to be the structural guard.

### 14.9 Computational boundary (confirmed, binding)

Phase 8 execution, when authorized, will use the **already-validated Temporal GraphSAGE implementation**
(`src/models/temporal_graphsage.py`, `src/graph/temporal.py`, `scripts/train_temporal_graphsage.py`'s
`train_epoch`) unmodified, on the **available AMD/ROCm environment**, with **no `torch_sparse`/`pyg_lib`-
dependent sampling** (no `NeighborLoader`/`ClusterData`; full day-snapshot streaming only, as already
established in Phase 7), and **no unnecessary architecture changes** — the model class itself is not altered
for Phase 8; only the external training/update loop and evaluation harness around it are new.

**Nothing in Section 14 has been executed.** This remains a design document only.
