"""Walk-forward arms F, Ad and E with an optional TRAIN-period state burn-in (Phase 8 supplement and held-out amendment 1).

A new function, not a change to Phase 8: with ``burn_in_through=None`` it performs exactly what ``src/adaptive/update_loop.py::run_walk_forward`` performs
(score window k, then an update for Ad and E, no update after the last window). With ``burn_in_through = d`` the entity memory is first built by streaming days 1..d in
order with the model in evaluation mode, no labels and no gradient, and carried unreset into the first window; ``d`` must be the day immediately before the first window.
Labels are requested only by the update routines, after the window just scored, and only for days up to that window's last day.
"""
from __future__ import annotations

import torch

from src.adaptive.replay import ReplayPoolBuilder
from src.adaptive.update_loop import TBPTT_DAYS, score_window, train_epoch, train_epoch_on_day_pool
from src.adaptive.windows import boundary_days


@torch.no_grad()
def burn_in(model, ds, state, through_day: int) -> int:
    """Stream days 1..through_day (those present in ``ds``) into ``state``; returns the number of days streamed. Labels are never touched."""
    model.eval()
    n = 0
    for d in ds.days:
        if d > through_day:
            break
        model.step(state, ds, d, predict=False)
        n += 1
    return n


def run_walk_forward_ext(arm: str, model, ds, windows: list, opt=None, pos_weight=None, fine_tune_epochs: int = 0, base_seed: int = 0,
                         train_day_min: int = 1, train_day_max: int = 121, burn_in_through: int | None = None, replay_windows: list | None = None,
                         trigger=None):
    """Arm "Tr" is Ad with the refresh decided by ``trigger(k, scores_of_window_k) -> bool`` (src/trigger/policy.py): the update after window k is the Ad update
    (all labeled days up to the window's end) but is performed only when the trigger fires; the trigger sees scores only, never a label."""
    if arm not in ("F", "Ad", "E", "Tr"):
        raise ValueError(f"arm must be one of 'F', 'Ad', 'E', 'Tr', got {arm!r}")
    if arm in ("Ad", "E", "Tr") and (opt is None or pos_weight is None or fine_tune_epochs <= 0):
        raise ValueError(f"arm {arm!r} requires opt, pos_weight and fine_tune_epochs > 0")
    if arm == "Tr" and trigger is None:
        raise ValueError("arm 'Tr' requires a trigger callable")
    if arm != "Tr" and trigger is not None:
        raise ValueError("a trigger is only meaningful for arm 'Tr'")
    if burn_in_through is not None and burn_in_through != windows[0][0] - 1:
        raise ValueError(f"burn_in_through must be the day before the first window ({windows[0][0] - 1}), got {burn_in_through}")
    state = model.init_state(ds)
    if burn_in_through is not None:
        burn_in(model, ds, state, burn_in_through)
    builder = ReplayPoolBuilder(base_seed, train_day_min, train_day_max, replay_windows or windows) if arm == "E" else None
    b = boundary_days(windows)
    n = len(windows)
    results = []
    for k, window in enumerate(windows, start=1):
        results.append(score_window(model, state, ds, window))
        if k == n:
            break
        if arm == "Ad" or (arm == "Tr" and trigger(k, results[-1][1])):
            for _ in range(fine_tune_epochs):
                train_epoch(model, ds, opt, train_last=b[k - 1], pos_weight=pos_weight, window=TBPTT_DAYS)
        elif arm == "E":
            pool = builder.pool_for_round(k)
            for _ in range(fine_tune_epochs):
                train_epoch_on_day_pool(model, ds, opt, pool, pos_weight, window=TBPTT_DAYS)
    return results
