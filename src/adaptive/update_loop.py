"""Phase 8 walk-forward orchestration: Arms F, Ad, E (design Sections C/D/14.5).

THIS MODULE PERFORMS REAL TRAINING AND INFERENCE when its functions are called (forward and
backward passes, GPU/CPU). It is written so an authorized Phase 8 execution run can call it, but
nothing here is invoked by this implementation pass. The lightweight test suite exercises only
the pure day-pool/window logic this module depends on (`src/adaptive/windows.py`,
`src/adaptive/replay.py`) directly, never this module's training functions.

CAUSAL STRUCTURE (why each arm receives the data it does — restated per the locked design):

  - A single `TemporalState` object is created ONCE per arm, per seed, before W1 is scored
    (`model.init_state(ds)`), and is NEVER reset. It is carried forward, day by day, through the
    entire walk-forward (122..151), exactly as Phase 7's state-carrying rule requires
    (`train_val_boundary_state_continuity`). This state is what "causal temporal/entity state is
    allowed to evolve during inference" (Section 14.1) refers to, for ALL THREE arms, including F.

  - Scoring a window calls `model.step(state, ds, d, predict=True)` day by day, in real
    chronological order, for `d` in that window — this mutates `state` (architecture-native) but
    never touches the model's weights. This is identical for F, Ad, and E.

  - An UPDATE (Ad, E only) calls a *separate* training routine (`train_epoch` for Ad, re-exported
    from the unmodified Phase 7 script; `train_epoch_on_day_pool` for E, defined below) that
    creates its OWN, fresh, internal `TemporalState` via `model.init_state(ds)` every single call
    (this is how Phase 7's `train_epoch` already behaves — see scripts/train_temporal_graphsage.py,
    not modified here). Because that internal state is fresh and local to the call, an update can
    NEVER corrupt the persistent, causally-evolving `state` object used for real scoring. An
    update only ever changes the model's WEIGHTS (via `opt.step()`), which is exactly the one
    thing that is allowed to differ between F, Ad, and E going forward in time. This is why the
    walk-forward loop below does not need to "rebuild" or "merge" state after an update: scoring
    state and training state are, by construction, two separate objects.

  - `F`: no update call ever happens. Its weights are identical, by construction, to the shared
    initial checkpoint for the entire run.

  - `Ad`: after window k is fully scored, `train_epoch(model, ds, opt, train_last=b_k, pos_weight,
    window=TBPTT_DAYS)` is called `K` times (K pre-registered epochs), where `b_k` is window k's
    last day. Each call re-walks days 1..b_k from a fresh internal state — this is "fine-tune on
    TRAIN ∪ already-scored VAL", using exactly the real chronological data, and nothing from
    window k+1 onward (which does not exist in `ds` scope at update k in the sense that `b_k` is
    the training cutoff passed to `train_epoch`, which never reads a day beyond it).

  - `E`: after window k is fully scored, `train_epoch_on_day_pool(model, ds, opt, pool, pos_weight)`
    is called `K` times, where `pool` = `ReplayPoolBuilder.pool_for_round(k)` — every entry in
    `pool` is a TRAIN day (1..121), some repeated, matching Ad's day-count at round k, but
    containing zero VAL days.
"""
from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "scripts"))
from train_temporal_graphsage import GRAD_CLIP, TBPTT_DAYS, train_epoch  # Phase 7 artifact, reused unmodified

from src.adaptive.replay import ReplayPoolBuilder
from src.adaptive.windows import boundary_days, days_in_window


def train_epoch_on_day_pool(model, ds, opt, day_pool: list[int], pos_weight, window: int = TBPTT_DAYS) -> float:
    """Arm E's update step. Deliberately mirrors Phase 7's `train_epoch`
    (scripts/train_temporal_graphsage.py) — same TBPTT windowing, same grad-clip, same loss —
    generalized to an arbitrary (possibly-repeating, pre-sorted) list of days instead of
    `[d for d in ds.days if d <= train_last]`. This duplication is necessary rather than modifying
    Phase 7's `train_epoch` in place, which is a completed Phase 7 artifact this project's
    instructions prohibit altering; the two functions are intentionally parallel in structure so
    a reader can compare them directly.

    `day_pool` must already be sorted ascending (as `ReplayPoolBuilder.pool_for_round` and
    `sample_replay_days` both return) so elapsed-time bookkeeping inside `model.step` never sees
    a negative gap. A fresh internal `TemporalState` is created here, every call — see the module
    docstring for why this is what keeps training state and scoring state from ever colliding.
    """
    model.train()
    state = model.init_state(ds)
    tot, n_tot = 0.0, 0
    for w0 in range(0, len(day_pool), window):
        lgs, ys = [], []
        for d in day_pool[w0:w0 + window]:
            lgs.append(model.step(state, ds, d, predict=True))
            ys.append(ds.labels(d))
        lg, y = torch.cat(lgs), torch.cat(ys)
        loss = F.binary_cross_entropy_with_logits(lg, y, pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        model.detach_state(state)
        tot += float(loss.item()) * y.numel()
        n_tot += y.numel()
    return tot / max(n_tot, 1)


def score_window(model, state, ds, window: tuple[int, int]):
    """Scores one window using the PERSISTENT, causally-evolving `state` (never created fresh
    here — it must be the same object passed in across the whole walk-forward). Returns
    (ids, scores, days) for every transaction in the window, in day order. Mutates `state`."""
    import numpy as np
    model.eval()
    ids_l, sc_l, day_l = [], [], []
    with torch.no_grad():
        for d in days_in_window(window):
            logits = model.step(state, ds, d, predict=True)
            ids_l.append(ds.ids_for_day(d))
            sc_l.append(torch.sigmoid(logits).cpu().numpy())
            day_l.append(np.full(logits.numel(), d))
    return np.concatenate(ids_l), np.concatenate(sc_l), np.concatenate(day_l)


def run_walk_forward(arm: str, model, ds, windows: list[tuple[int, int]], opt=None,
                      pos_weight=None, fine_tune_epochs: int = 0, base_seed: int = 0,
                      train_day_min: int = 1, train_day_max: int = 121):
    """Runs the full 5-window walk-forward for one arm ("F", "Ad", or "E") and returns, per
    window, the (ids, scores, days) tuple produced by `score_window` — exactly the causal
    sequence specified in Section 14.5:

        score W1 -> update -> score W2 -> update -> score W3 -> update -> score W4 -> update
        -> score W5 -> STOP

    An update happens strictly BETWEEN two scored windows (Ad/E only) — NEVER after the last
    window, since there is no W_{N+1} for a post-W_N update to ever be used by. This is why the
    loop below gates the update call on `k < len(windows)` rather than updating unconditionally
    after every window.

    `model` must already be the shared initial TRAIN-only checkpoint (identical object/weights
    for F, Ad, E at the start of a run — Section 14.1/14.5) and `opt` the optimizer bound to
    `model`'s parameters (required, and only used, for Ad/E). THIS FUNCTION PERFORMS REAL
    TRAINING when arm is "Ad" or "E" — it is not invoked anywhere in this implementation pass.
    """
    if arm not in ("F", "Ad", "E"):
        raise ValueError(f"arm must be one of 'F', 'Ad', 'E', got {arm!r}")
    if arm in ("Ad", "E") and (opt is None or pos_weight is None or fine_tune_epochs <= 0):
        raise ValueError(f"arm {arm!r} requires opt, pos_weight, and fine_tune_epochs > 0")

    state = model.init_state(ds)  # created ONCE; persists, unreset, across the whole walk-forward
    results = []
    replay_builder = ReplayPoolBuilder(base_seed, train_day_min, train_day_max, windows) if arm == "E" else None
    b = boundary_days(windows)
    n = len(windows)

    for k, window in enumerate(windows, start=1):
        results.append(score_window(model, state, ds, window))
        # window k is now fully scored; its labels are available starting now, and not before.
        # No update occurs after the FINAL window (k == n): there is no subsequent window for
        # that update to ever inform, so performing one would be pure dead weight relative to
        # the locked design's score-W1..W5-with-an-update-only-between-them sequence.
        if k == n:
            break
        if arm == "Ad":
            for _ in range(fine_tune_epochs):
                train_epoch(model, ds, opt, train_last=b[k - 1], pos_weight=pos_weight, window=TBPTT_DAYS)
        elif arm == "E":
            pool = replay_builder.pool_for_round(k)
            for _ in range(fine_tune_epochs):
                train_epoch_on_day_pool(model, ds, opt, pool, pos_weight, window=TBPTT_DAYS)
        # arm == "F": no update, by definition of the frozen-parameter baseline (Section 14.1)
    return results
