"""Arm E — replay-control day sampler (Phase 8 design Section 10 / 14.1 table / 14.3).

Pure day-index logic: sampling TRAIN days with replacement, reproducibly. Does not touch the
dataset, the model, or any tensor — this is deliberate, so the sampler's correctness (TRAIN-only,
with replacement, right count, reproducible) can be verified by a lightweight unit test without
ever loading the graph or running a forward pass.

Design requirements enforced here (restated from DESIGN_BEFORE_EXECUTION.md Section 10/14.3):
  - every sampled day lies in [train_day_min, train_day_max] — never a VAL day (122-151), never a
    TEST day (152-182);
  - sampling is WITH replacement (duplicates allowed, by design — this matches "more repetitions
    of already-seen TRAIN data", not a specific day appearing at most once);
  - the number of days sampled at round k equals the number of days in VAL window k, matching
    arm Ad's update volume for that round (Section 10: "resampled to the same example count Ad
    would have added");
  - sampling is controlled by a reproducible RNG seeded as `1000 * base_seed + window_index`
    (Section 10), so re-running Phase 8 with the same base seed reproduces identical replay pools.
"""
from __future__ import annotations

import numpy as np

REPLAY_SEED_MULTIPLIER = 1000  # seed = 1000*base_seed + window_index, per the locked design


def replay_rng(base_seed: int, window_index: int) -> np.random.RandomState:
    """The reproducible RNG for round `window_index` (1-indexed), per base seed."""
    if window_index < 1:
        raise ValueError(f"window_index must be >= 1 (1-indexed), got {window_index}")
    return np.random.RandomState(REPLAY_SEED_MULTIPLIER * base_seed + window_index)


def sample_replay_days(base_seed: int, window_index: int, n_days: int,
                        train_day_min: int, train_day_max: int) -> list[int]:
    """`n_days` TRAIN days, sampled WITH replacement, from [train_day_min, train_day_max],
    using the reproducible per-round RNG. Returned sorted ascending — sorting is required so
    that, when these days are later fed through the model's day-by-day TBPTT loop (which tracks
    `last_day`/elapsed-time per entity), the elapsed-time arithmetic (`d - last_day`) never goes
    negative. A repeated day simply yields elapsed_time=0 at that point in the replay sequence,
    which is a well-defined value for the architecture, not an error. See
    `src/adaptive/update_loop.py`'s `train_epoch_on_day_pool` docstring for the full argument."""
    if train_day_min > train_day_max:
        raise ValueError("train_day_min must be <= train_day_max")
    if n_days < 0:
        raise ValueError("n_days must be >= 0")
    rng = replay_rng(base_seed, window_index)
    days = rng.randint(train_day_min, train_day_max + 1, size=n_days).tolist()
    return sorted(int(d) for d in days)


class ReplayPoolBuilder:
    """Accumulates, round by round, the day-pool arm E trains on at each update.

    At round k, E's pool = TRAIN's real days (train_day_min..train_day_max, each once) UNION the
    replay samples accumulated from rounds 1..k (each round contributing len(window_k) resampled
    TRAIN days). This makes E's total day-count at round k exactly equal to Ad's expanding-window
    day-count at round k (121 + 6k for the locked 6-day/5-window design), satisfying "same update
    cadence/compute/example-count as Ad" by construction, while every entry — including every
    duplicate — is drawn from TRAIN only, so E never sees a VAL or TEST day under any call.
    """

    def __init__(self, base_seed: int, train_day_min: int, train_day_max: int,
                 windows: list[tuple[int, int]]):
        self.base_seed = base_seed
        self.train_day_min = train_day_min
        self.train_day_max = train_day_max
        self.windows = windows
        self._train_days = list(range(train_day_min, train_day_max + 1))
        self._accumulated_extra: list[int] = []
        self._next_round = 1

    def pool_for_round(self, k: int) -> list[int]:
        """The full day-pool (TRAIN + accumulated replay) to train on for update round `k`
        (1-indexed). Rounds must be requested in order 1, 2, 3, ... — this mirrors the walk-forward
        loop's own sequential structure (Section 14.5) and lets the RNG state advance round by
        round deterministically rather than needing all rounds pre-drawn up front."""
        if k != self._next_round:
            raise ValueError(f"rounds must be requested in order; expected round {self._next_round}, got {k}")
        if k > len(self.windows):
            raise ValueError(f"no window defined for round {k} (only {len(self.windows)} windows)")
        window = self.windows[k - 1]
        n_days = window[1] - window[0] + 1
        new_days = sample_replay_days(self.base_seed, k, n_days, self.train_day_min, self.train_day_max)
        self._accumulated_extra.extend(new_days)
        self._next_round += 1
        return sorted(self._train_days + self._accumulated_extra)
