"""Phase 8 chronological window definitions (Section D of the locked design).

Pure, side-effect-free day-index arithmetic. Does not touch the dataset, the model, or any
I/O. Deliberately kept this way so it is fully unit-testable without training anything.
"""
from __future__ import annotations

WINDOW_LEN_DAYS = 6
N_WINDOWS = 5


def build_windows(val_day_min: int, val_day_max: int, window_len: int = WINDOW_LEN_DAYS) -> list[tuple[int, int]]:
    """Returns [(w_start, w_end), ...] covering [val_day_min, val_day_max] in consecutive,
    non-overlapping, chronologically ordered blocks of `window_len` days, with no remainder.
    Raises if the VAL span does not divide evenly — this is a pre-registered design constant
    (Section D: 30 VAL days / 6 = 5 windows), not something to silently adjust at run time."""
    span = val_day_max - val_day_min + 1
    if span % window_len != 0:
        raise ValueError(f"VAL span of {span} days does not divide evenly into {window_len}-day windows; "
                          f"the Phase 8 design is pre-registered for exactly 5 windows of 6 days each")
    windows = []
    start = val_day_min
    while start <= val_day_max:
        end = start + window_len - 1
        windows.append((start, end))
        start = end + 1
    return windows


def assert_disjoint_chronological(windows: list[tuple[int, int]]) -> None:
    """Every window is non-empty, windows are strictly increasing, and no day is covered twice."""
    for (s, e) in windows:
        if s > e:
            raise ValueError(f"empty or inverted window: ({s}, {e})")
    for (_, e1), (s2, _) in zip(windows, windows[1:]):
        if s2 != e1 + 1:
            raise ValueError(f"windows are not contiguous/chronological: ...{e1}] then [{s2}...")


def boundary_days(windows: list[tuple[int, int]]) -> list[int]:
    """b_k = the last day of window k, for k = 1..len(windows) — the day after which that
    window's labels become available and an update (Ad/E) may use them (Section 11)."""
    return [e for (_, e) in windows]


def window_index_for_day(day: int, windows: list[tuple[int, int]]) -> int | None:
    """1-indexed window number containing `day`, or None if `day` falls outside every window
    (e.g. a TRAIN day, or — should it ever be passed by mistake — a TEST day)."""
    for i, (s, e) in enumerate(windows, start=1):
        if s <= day <= e:
            return i
    return None


def days_in_window(window: tuple[int, int]) -> list[int]:
    s, e = window
    return list(range(s, e + 1))
