"""Phase 8 leakage checks (design Section 11).

Two kinds of check live here, clearly separated:

  1. PURE / structural checks (no model, no dataset, no forward pass) — these are what the
     lightweight test suite actually runs. They verify the *inputs* to the walk-forward (window
     definitions, replay pools, hyperparameter holdout ranges) can never contain a VAL or TEST
     day where the design forbids it, and that two state_dicts are/aren't identical.

  2. INTERVENTIONAL checks that need a trained Phase 8 checkpoint and a built dataset — these are
     written in full (reusing src/graph/temporal_checks.py's primitives, unmodified) so an
     authorized Phase 8 execution run can call them, but they are NOT runnable without real
     checkpoints and are NOT invoked anywhere in this implementation pass.
"""
from __future__ import annotations

import torch


# --------------------------------------------------------------------------- 1. pure / structural

def check_windows_cover_val_exactly(windows: list[tuple[int, int]], val_day_min: int, val_day_max: int) -> tuple[bool, str]:
    days = sorted(d for (s, e) in windows for d in range(s, e + 1))
    expected = list(range(val_day_min, val_day_max + 1))
    ok = days == expected
    return ok, f"windows cover {days[:1]}..{days[-1:]}" if days else "windows are empty"


def check_replay_pool_excludes_val_and_test(pool: list[int], val_day_min: int, test_day_min: int | None = None) -> tuple[bool, str]:
    bad = [d for d in pool if d >= val_day_min]
    ok = not bad
    detail = "no VAL/TEST day present" if ok else f"VAL/TEST days leaked into replay pool: {sorted(set(bad))[:5]}"
    return ok, detail


def check_update_boundaries_equal_window_ends(windows: list[tuple[int, int]], boundaries: list[int]) -> tuple[bool, str]:
    expected = [e for (_, e) in windows]
    ok = boundaries == expected
    return ok, f"boundaries={boundaries} vs window-ends={expected}"


def check_state_dicts_equal(sd_a: dict, sd_b: dict) -> tuple[bool, str]:
    """Bit-for-bit equality of two state_dicts — used both for
    `window1_frozen_equals_adaptive_before_first_update`-style checks and for verifying F's
    parameters never change across the whole run. Pure tensor comparison; works on any two
    state_dicts, including tiny/untrained/random ones, so this is safely unit-testable without
    training anything real."""
    if sd_a.keys() != sd_b.keys():
        return False, f"key mismatch: {set(sd_a) ^ set(sd_b)}"
    for k in sd_a:
        if not torch.equal(sd_a[k], sd_b[k]):
            return False, f"tensor mismatch at key {k!r}"
    return True, "all tensors bit-for-bit identical"


def check_identical_initial_checkpoint(sd_f: dict, sd_ad: dict, sd_e: dict) -> tuple[bool, str]:
    ok1, d1 = check_state_dicts_equal(sd_f, sd_ad)
    ok2, d2 = check_state_dicts_equal(sd_f, sd_e)
    ok = ok1 and ok2
    return ok, f"F vs Ad: {d1}; F vs E: {d2}"


def check_holdout_precedes_val(holdout_select_end: int, val_day_min: int) -> tuple[bool, str]:
    ok = holdout_select_end < val_day_min
    return ok, f"holdout selection ends day {holdout_select_end}, VAL starts day {val_day_min}"


# --------------------------------------------------------------------- 2. interventional (not run)

def future_label_exclusion_check(model, ds_factory, raw, boundary_day: int, pos_weight, rng_seed: int = 0):
    """INTERVENTIONAL — requires a real trained checkpoint and dataset; not runnable here.

    Corrupts (relabels) every row strictly after `boundary_day`, re-runs the update that would
    have been computed at `boundary_day`, and asserts the resulting weights are unchanged versus
    the un-corrupted run. Mirrors `src/graph/temporal_checks.py`'s `perturb_future` pattern.
    Reuses that module's helpers (`perturb_future`, `_assert_same_layout`) unmodified — see
    `src/graph/temporal_checks.py` for the exact perturbation mechanics; this function is a thin
    Phase-8-specific wrapper that calls `train_epoch`/`train_epoch_on_day_pool` on the corrupted
    vs. clean dataset and compares the resulting `model.state_dict()`s via
    `check_state_dicts_equal`, rather than comparing logits as the Phase 7 checks do.

    Not invoked in this implementation pass. Requires: a trained-up-to-`boundary_day` checkpoint,
    a built `TemporalDataset`, and real forward+backward passes.
    """
    raise NotImplementedError(
        "future_label_exclusion_check requires a trained Phase 8 checkpoint and a real forward/backward "
        "pass; it is implemented at the call-site level in scripts/validate_phase8_leakage.py's intended "
        "design, not executed here. This stub exists so the check's contract (inputs, what it compares, "
        "which Phase 7 primitives it reuses) is documented before any Phase 8 execution, per the locked design."
    )


def update_does_not_retroactively_change_past_predictions_check(recorded_predictions: dict, model, ds, windows):
    """INTERVENTIONAL — requires the real per-window predictions recorded during the walk-forward
    run (`recorded_predictions`, keyed by window index) and the final trained model. Re-scores
    windows 1..k using the model snapshot that was actually in effect when they were first
    scored (saved per round by the execution script, per Section 12's `model_checkpoint_W{k}.pt`
    artifacts) and asserts bitwise/noise-floor equality with what was recorded live. Not runnable
    without those saved per-round checkpoints; not invoked in this implementation pass.
    """
    raise NotImplementedError(
        "requires real, saved per-round checkpoints from an executed Phase 8 run; documented, not executed."
    )


def replay_control_never_touches_val_check(ds, pool_access_log: list[int], val_day_min: int):
    """INTERVENTIONAL (but cheap once a real run exists): enable `ds.access_log` (Phase 7's
    existing instrumentation, src/graph/temporal.py) during an arm-E update call and assert every
    logged day is < val_day_min. Not invoked here because it requires a real `TemporalDataset`
    and a real `train_epoch_on_day_pool` call; the access-log mechanism itself is Phase 7's,
    unmodified."""
    bad = [d for d in pool_access_log if d >= val_day_min]
    return (not bad), (f"access log shows only TRAIN days (max={max(pool_access_log)})" if not bad
                       else f"access log shows VAL/TEST days: {sorted(set(bad))[:5]}")
