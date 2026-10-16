"""Shared community-identity tracker and lifecycle descriptors (design sections 6 step 6 and 8).

One tracker implementation is applied to every method (so tracker quality is not a confound). Matching uses only day t-1 -> t:
a community continues the track whose day t-1 card set has Jaccard >= 0.5 (largest overlap first, ties -> lower track id); splits:
the largest-overlap child continues and the others start new tracks; merges: the largest-overlap track continues and the others
end. A track with no successor on day t+1 has ended; disappearance is *declared* g = 3 days later (known_at = last_day + 3).
Lifecycle fields that need hindsight (disappearance, total lifetime) are retrospective descriptors and never enter a score.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.coordination.frozen import G_DISAPPEAR, TRACK_JACCARD


@dataclass
class Track:
    tid: int
    first_day: int
    last_day: int
    parent: int | None = None
    flags: list = field(default_factory=list)
    days: list = field(default_factory=list)       # (day, size)


class Tracker:
    def __init__(self, jaccard: float = TRACK_JACCARD):
        self.jaccard = jaccard
        self.tracks: dict[int, Track] = {}
        self._next = 0
        self._prev: list[tuple[int, frozenset]] = []
        self._prev_day: int | None = None

    def step(self, t: int, card_sets: list) -> list[int]:
        sets = [frozenset(int(x) for x in c) for c in card_sets]
        prev = self._prev if self._prev_day == t - 1 else []
        pairs = []
        if prev and sets:
            inv: dict[int, list[int]] = {}
            for j, (_, ps) in enumerate(prev):
                for c in ps:
                    inv.setdefault(c, []).append(j)
            for i, s in enumerate(sets):
                cnt: dict[int, int] = {}
                for c in s:
                    for j in inv.get(c, ()):
                        cnt[j] = cnt.get(j, 0) + 1
                for j, inter in cnt.items():
                    jac = inter / (len(s) + len(prev[j][1]) - inter)
                    if jac >= self.jaccard:
                        pairs.append((-jac, prev[j][0], i, j))
        pairs.sort()
        assigned: dict[int, int] = {}
        used_prev: set[int] = set()
        n_cont_from: dict[int, int] = {}
        for negj, tid_prev, i, j in pairs:
            n_cont_from[j] = n_cont_from.get(j, 0) + 1
            if i in assigned or j in used_prev:
                continue
            assigned[i] = tid_prev
            used_prev.add(j)
        out = []
        for i, s in enumerate(sets):
            if i in assigned:
                tid = assigned[i]
                tr = self.tracks[tid]
                tr.last_day = t
            else:
                tid = self._next
                self._next += 1
                parent = None
                cand = [p for (negj, pt, ii, jj) in pairs if ii == i for p in [pt]]
                if cand:
                    parent = cand[0]
                    self.tracks[parent].flags.append(("split", t))
                tr = Track(tid, t, t, parent)
                self.tracks[tid] = tr
            self.tracks[tid].days.append((t, len(s)))
            out.append(tid)
        # merges: previous tracks that matched some community but did not continue
        for (negj, pt, i, j) in pairs:
            if j not in used_prev:
                self.tracks[pt].flags.append(("merged", t))
        self._prev = [(tid, s) for tid, s in zip(out, sets)]
        self._prev_day = t
        return out


def lifecycle_table(tracker: Tracker, last_observed_day: int) -> list[dict]:
    rows = []
    for tid, tr in sorted(tracker.tracks.items()):
        ended = tr.last_day < last_observed_day
        rows.append({"track": tid, "first_day": tr.first_day, "last_day": tr.last_day, "age_days": tr.last_day - tr.first_day + 1,
                     "size_first": tr.days[0][1], "size_last": tr.days[-1][1], "size_max": max(s for _, s in tr.days),
                     "ended": bool(ended), "disappearance_known_at": (tr.last_day + G_DISAPPEAR) if ended else None,
                     "parent": tr.parent, "flags": tr.flags})
    return rows
