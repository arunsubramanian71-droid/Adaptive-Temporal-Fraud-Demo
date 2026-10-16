# Phase 6C — GAT Convergence-Resolution Experiment: STATUS

**Status: ABORTED / INCOMPLETE.** Recorded 2026-09-20 on the user's instruction. The 300-epoch GAT sweep was intentionally stopped before completion. **Phase 6C produced no usable result and none of it may be used in the research conclusions.**

---

## 1. What was done

| Step | State |
|---|---|
| Sustained GPU-memory feasibility check | **Completed.** This was a read of Phase 6B's already-recorded `experiments/gat/seed_*/resource_measurements.json`, not a new GPU run: peak reserved memory 12.10 / 12.32 / 12.32 GiB across the three 150-epoch seeds (single-pass feasibility estimate was 12.26 GiB), with no growth over epochs. Judged sufficient to attempt 300 epochs. |
| 300-epoch script created | **Done.** `scripts/train_gat_convergence_300epoch.py` (max_epochs=300, patience=30, otherwise identical to Phase 6B; includes a 14.5 GiB memory-abort guard). |
| 300-epoch sweep started | **Done**, in the background, seeds intended to run sequentially (42, 123, 2024). |
| 300-epoch sweep finished | **No.** It was stopped before completion. No process is running now (checked). |
| Option B (learning-rate schedule) | **Never started.** It was conditional on Option A failing to converge, which was never determined. |

## 2. What exists on disk (stated so it is not a surprise)

Read-only inspection of `experiments/gat_300epoch/` found:

- `seed_42/` — a full per-seed artifact set (`config.json`, `metrics_val.json`, `model_checkpoint.pt`, `predictions_val.parquet`, `resource_measurements.json`, `training_history.json`). Seed 42's run therefore reached the point where it saved its outputs before the sweep was stopped.
- `seed_123/` — an empty directory (started, nothing saved).
- No `seed_2024/`, no `summary.json`, no `convergence_curves.png`, no `metadata.json`. **No cross-seed aggregate, mean, or standard deviation exists.**
- The background task's console log (`b0v4ic6jg.output`) is no longer present, so the sweep's progress at the moment it stopped cannot be reconstructed from it.

**These files were not opened, summarized, or aggregated while writing this report, and their contents are deliberately not recorded here.** They are a single seed from an incomplete sweep, with no aggregate and no protocol-level completion check. They are left in place, unmodified; deleting them was not requested.

## 3. Consequences for the research record

1. **No Phase 6C performance or convergence result exists** in the sense of the pre-registered design (three seeds, aggregated, convergence assessed). The seed-42 files above do not constitute one.
2. **No Phase 6C number may be used in any research conclusion, comparison table, or later report.** That includes anything that could be read from the `seed_42/` files.
3. **The GAT convergence ambiguity from Phase 6B remains unresolved.** Phase 6B could not distinguish "GAT genuinely performs worse" from "GAT is under-trained" (all three seeds hit the 150-epoch cap), and Phase 6C did not change that. No claim that GAT is intrinsically inferior to GraphSAGE is supported, and no claim that it is merely under-trained is supported either.

## 4. Official references (unchanged)

| Experiment | Status | Pooled PR-AUC |
|---|---|---|
| **GAT — Phase 6B** | **Remains the official GAT experiment** | **0.3320 ± 0.0052** (all 3 seeds hit max_epochs=150, no early stopping — reported as such) |
| **GraphSAGE — Phase 6A** | **Remains the official GraphSAGE reference** | **0.4637 ± 0.0167** |

Nothing in Phase 6B or any earlier phase was modified by Phase 6C. The Phase 6B artifacts (`experiments/gat/`, `reports/phase6b_gat_report.md`) and Phase 6A artifacts (`experiments/graphsage_robust/`) show their original timestamps.

## 5. Not to be done without a new instruction

- Do not restart or resume the 300-epoch sweep. `scripts/train_gat_convergence_300epoch.py` is left in place as a record of what was attempted and should not be executed.
- No further GAT training or GAT hyperparameter tuning.
- No temporal GNN work.
